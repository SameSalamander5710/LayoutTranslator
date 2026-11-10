//! Text detection: the OCR models (PaddleOCR through oar-ocr), the engine cache, and `detect`, which combines OCR with the selectable text of a PDF.

use super::*;

/// The three files every OCR model folder needs.
pub const MODEL_FILES: [&str; 3] = ["det.onnx", "rec.onnx", "dict.txt"];

/// True for a language name that is safe to use as a folder name under `models/`. The name comes from the UI (a
/// drop-down), but it is joined into a path, so anything that could leave that folder is refused: `/`, `\`, `..`, a
/// colon (a drive letter on Windows) and an empty name.
pub fn valid_lang_name(name: &str) -> bool {
    !name.is_empty() && !name.contains(['/', '\\', ':', '\0']) && !name.contains("..")
}

/// Names the first problem in an OCR model folder, or None when it is complete. An empty file counts as a problem: an
/// interrupted download leaves one, and the OCR engine then fails with an error nobody can read.
pub fn model_problem(dir: &Path) -> Option<String> {
    for f in MODEL_FILES {
        match std::fs::metadata(dir.join(f)) {
            Ok(m) if m.is_file() && m.len() > 0 => {}
            Ok(m) if m.is_file() => return Some(format!("{f} is empty")),
            _ => return Some(format!("{f} is missing")),
        }
    }
    None
}

/// Caches built OCR engines per model folder so only the first run pays the load cost.
/// Cloning is cheap (it clones the inner `Arc`) and all clones share the same engines, so the app-wide cache can be
/// moved into a blocking task.
#[derive(Default, Clone)]
pub struct EngineCache(Arc<Mutex<HashMap<String, Arc<Engine>>>>);

/// Finds the text lines of an image (raw lines, not yet grouped into blocks). `OcrEngine` is the real thing. The tests use a
/// fake that returns stored lines, so the steps after OCR (detect, group, tighten, merge, read a region) run without model
/// files. This trait is also the one door to the OCR crates (`oar-ocr`, which brings ONNX Runtime): to replace them, only
/// `ocr_lines` and the `impl` below change.
pub(super) trait TextDetector {
    fn lines(&self, im: &RgbImage) -> Result<Vec<Line>, String>;
}

/// The OCR engine for one model folder, as a `TextDetector`.
pub(super) struct OcrEngine<'a> { pub(super) dir: &'a Path, pub(super) cache: &'a EngineCache }

impl TextDetector for OcrEngine<'_> {
    fn lines(&self, im: &RgbImage) -> Result<Vec<Line>, String> { ocr_lines(im, self.dir, self.cache) }
}

pub fn ocr(img: &Path, dir: &Path, cache: &EngineCache) -> Result<Vec<Block>, String> { ocr_with(&OcrEngine { dir, cache }, img) }

fn ocr_with(det: &dyn TextDetector, img: &Path) -> Result<Vec<Block>, String> {
    let im = load_oriented(img)?.to_rgb8();
    Ok(refine(&im, group_tagged(det.lines(&im)?)))
}

// Runs the OCR engine on an in-memory image and returns the raw text lines (not yet grouped).
fn ocr_lines(im: &RgbImage, dir: &Path, cache: &EngineCache) -> Result<Vec<Line>, String> {
    let p = |n: &str| dir.join(n).to_string_lossy().to_string();
    let key = dir.to_string_lossy().to_string();
    let engine = {
        let mut m = lock(&cache.0);
        match m.get(&key) {
            Some(e) => e.clone(),
            None => {
                let det = oar_ocr::domain::TextDetectionConfig {
                    score_threshold: 0.25, // default 0.3: lower catches faint/small lines
                    box_threshold: 0.5,    // default 0.6: lower keeps smaller boxes
                    ..Default::default()
                };
                let e = Arc::new(
                    OAROCRBuilder::new(p("det.onnx"), p("rec.onnx"), p("dict.txt"))
                        .text_detection_config(det)
                        .build()
                        .map_err(|e| e.to_string())?,
                );
                m.insert(key.clone(), e.clone());
                e
            }
        }
    };
    let res = engine.predict(vec![im.clone()]).map_err(|e| e.to_string())?;
    let mut lines = vec![];
    let result = res.first().ok_or("The OCR engine returned no result")?;
    for r in &result.text_regions {
        if let Some((t, c)) = r.text_with_confidence() {
            let bbox = &r.bounding_box;
            let x0 = bbox.x_min();
            let y0 = bbox.y_min();
            let x1 = bbox.x_max();
            let y1 = bbox.y_max();
            let (fg, bg) = box_colors(im, x0, y0, x1 - x0, y1 - y0);
            lines.push(Line {
                b: [x0, y0, x1 - x0, y1 - y0],
                t: t.to_string(),
                c,
                fg,
                bg,
                layer: false,
            });
        }
    }
    Ok(lines)
}

/// Returns (blocks, number of lines taken from the text layer, whether OCR ran).
/// 1. Selectable text lines become blocks directly (no OCR).
/// 2. If the page has images (or no selectable text at all), OCR runs on a copy of the page with the
///    selectable text painted over, so only the remaining text is read. OCR lines that still fall
///    on selectable text are dropped.
/// 3. All lines are grouped into blocks together.
///
/// With `allow_ocr == false` only step 1 runs (selectable text only; images and scanned text are ignored).
pub fn ocr_hybrid(img: &Path, dir: &Path, cache: &EngineCache, layer: &TextLayer, allow_ocr: bool) -> Result<(Vec<Block>, usize, bool), String> {
    ocr_hybrid_with(&OcrEngine { dir, cache }, img, layer, allow_ocr)
}

fn ocr_hybrid_with(det: &dyn TextDetector, img: &Path, layer: &TextLayer, allow_ocr: bool) -> Result<(Vec<Block>, usize, bool), String> {
    let im = load_oriented(img)?.to_rgb8();
    let mut lines: Vec<Line> = merge_layer_lines(&layer.lines).iter().filter(|l| !l.t.trim().is_empty() && l.b[2] >= 1.0 && l.b[3] >= 1.0).map(|l| {
        let (fg, bg) = box_colors(&im, l.b[0], l.b[1], l.b[2], l.b[3]);
        Line { b: l.b, t: l.t.trim().to_string(), c: 1.0, fg, bg, layer: true }
    }).collect();
    let n_layer = lines.len();
    let run_ocr = allow_ocr && (n_layer == 0 || layer.has_images);
    if run_ocr {
        let mut masked = im.clone();
        for l in &lines {
            let r = Rect::at(l.b[0] as i32 - 2, l.b[1] as i32 - 2).of_size(l.b[2] as u32 + 4, l.b[3] as u32 + 4);
            draw_filled_rect_mut(&mut masked, r, Rgb(l.bg));
        }
        for o in det.lines(&masked)? {
            if !lines[..n_layer].iter().any(|l| overlap_frac(&o.b, &l.b) > 0.5) { lines.push(o); }
        }
    }
    Ok((refine(&im, group_tagged(lines)), n_layer, run_ocr))
}

/// Text identification method chosen in the UI:
/// - "hybrid": selectable PDF text first, OCR for the rest (plain images are just OCR'd)
/// - "ocr":    OCR only, the PDF text layer is ignored
/// - "text":   selectable PDF text only, nothing is OCR'd
///
/// Returns (blocks, lines taken from the text layer, whether OCR ran).
pub fn detect(img: &Path, dir: &Path, cache: &EngineCache, mode: &str) -> Result<(Vec<Block>, usize, bool), String> {
    detect_with(&OcrEngine { dir, cache }, img, mode)
}

fn detect_with(det: &dyn TextDetector, img: &Path, mode: &str) -> Result<(Vec<Block>, usize, bool), String> {
    let layer = if mode == "ocr" { None } else { load_layer(img) };
    match layer {
        Some(l) => ocr_hybrid_with(det, img, &l, mode != "text"),
        None if mode == "text" => Err("Selectable-text mode only works on PDFs. Choose another detection method for images.".into()),
        None => ocr_with(det, img).map(|b| (b, 0, true)),
    }
}

/// What the UI gets for a page that was just detected.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Page { pub width: u32, pub height: u32, pub original: String, pub blocks: Vec<Block>, pub layer_lines: usize, pub ocr_ran: bool }

/// `detect`, plus everything a freshly detected page needs before the UI shows it: numbered boxes (1, 2, ...) with their text
/// and page colours, the size of the page and the page image as a PNG data URL.
pub fn detect_page(img: &Path, dir: &Path, cache: &EngineCache, mode: &str) -> Result<Page, String> {
    detect_page_with(&OcrEngine { dir, cache }, img, mode)
}

fn detect_page_with(det: &dyn TextDetector, img: &Path, mode: &str) -> Result<Page, String> {
    let (blocks, layer_lines, ocr_ran) = detect_with(det, img, mode)?;
    finish_page(&load_oriented(img)?, blocks, layer_lines, ocr_ran)
}

/// The part of `detect_page` that needs no models.
fn finish_page(im: &DynamicImage, mut blocks: Vec<Block>, layer_lines: usize, ocr_ran: bool) -> Result<Page, String> {
    let rgb = im.to_rgb8();
    for (i, b) in blocks.iter_mut().enumerate() {
        b.id = i as u32 + 1;
        colors(&rgb, b);
    }
    Ok(Page { width: im.width(), height: im.height(), original: png_data_url(im)?, blocks, layer_lines, ocr_ran })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::testutil::*;

    #[test]
    fn text_mode_needs_a_text_layer() {
        let r = detect(Path::new("/nonexistent/page.png"), Path::new("/nonexistent"), &EngineCache::default(), "text");
        assert!(r.is_err());
    }

    #[test]
    fn model_problem_names_the_bad_file() {
        let d = temp_dir("model");
        assert_eq!(model_problem(&d).as_deref(), Some("det.onnx is missing"));
        for f in MODEL_FILES { std::fs::write(d.join(f), b"x").unwrap(); }
        assert_eq!(model_problem(&d), None);
        std::fs::write(d.join("det.onnx"), b"").unwrap(); // what an interrupted download leaves
        assert_eq!(model_problem(&d).as_deref(), Some("det.onnx is empty"));
        std::fs::write(d.join("det.onnx"), b"x").unwrap();
        std::fs::write(d.join("dict.txt"), b"").unwrap();
        assert_eq!(model_problem(&d).as_deref(), Some("dict.txt is empty"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn language_names_cannot_leave_the_models_folder() {
        for ok in ["en", "ja", "zh-CN", "chinese_traditional", "My Language 2"] { assert!(valid_lang_name(ok), "{ok}"); }
        for bad in ["../x", "..", "a/b", "a\\b", "..\\x", "C:", "C:\\x", "/etc", "a..b", "", "a\0b"] { assert!(!valid_lang_name(bad), "{bad:?}"); }
    }

    #[test]
    fn a_detected_page_gets_numbered_boxes_with_colours_and_its_size() {
        let loose = [35.0f32, 35.0, 330.0, 130.0];
        let blocks = vec![blk(7, loose, [0; 3]), blk(9, [5.0, 5.0, 20.0, 10.0], [0; 3])]; // the ids are replaced
        let out = finish_page(&DynamicImage::ImageRgb8(striped_page([255, 255, 255])), blocks, 3, true).unwrap();
        assert_eq!((out.width, out.height), (400, 200));
        assert_eq!(out.blocks.iter().map(|b| b.id).collect::<Vec<_>>(), vec![1, 2]);
        assert!(out.blocks[0].fg.iter().all(|c| *c >= 235), "white letters, found by colors(): {:?}", out.blocks[0].fg);
        assert!(out.original.starts_with("data:image/png;base64,"));
        assert_eq!((out.layer_lines, out.ocr_ran), (3, true));
    }

    #[test]
    fn the_page_is_sent_to_the_ui_with_the_names_it_reads() {
        let out = finish_page(&DynamicImage::ImageRgb8(RgbImage::new(10, 10)), vec![], 0, false).unwrap();
        let json = serde_json::to_value(&out).unwrap();
        for key in ["width", "height", "original", "blocks", "layerLines", "ocrRan"] { assert!(json.get(key).is_some(), "missing {key}"); }
    }

    #[test]
    fn detection_runs_without_model_files_through_a_detector() {
        let (dir, file) = page_file("det-lines", &RgbImage::from_pixel(300, 200, Rgb([255, 255, 255])));
        let det = FakeDetector::with(vec![
            line(20.0, 20.0, 200.0, 20.0, "first line", [0; 3]),
            line(20.0, 44.0, 200.0, 20.0, "second line", [0; 3]),
            line(20.0, 150.0, 100.0, 20.0, "far away", [0; 3]),
        ]);
        let page = detect_page_with(&det, &file, "ocr").unwrap();
        let blocks: Vec<(u32, &str)> = page.blocks.iter().map(|b| (b.id, b.text.as_str())).collect();
        assert_eq!(blocks, vec![(1, "first line second line"), (2, "far away")]); // the two close lines are one block
        assert_eq!((page.width, page.height, page.layer_lines, page.ocr_ran), (300, 200, 0, true));
        assert_eq!(det.calls.get(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_loose_box_from_the_detector_is_tightened_during_detection() {
        let (dir, file) = page_file("det-tight", &striped_page([255, 255, 255]));
        let det = FakeDetector::with(vec![line(35.0, 35.0, 330.0, 130.0, "white letters", [255; 3])]);
        let page = detect_page_with(&det, &file, "ocr").unwrap();
        assert_eq!(page.blocks.len(), 1);
        assert!(page.blocks[0].bbox[3] < 110.0, "the box is still {} px tall", page.blocks[0].bbox[3]);
        assert_eq!((page.blocks[0].bbox[0], page.blocks[0].bbox[2]), (35.0, 330.0), "the width is not touched");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_box_inside_another_box_from_the_detector_is_merged() {
        let (dir, file) = page_file("det-nested", &RgbImage::from_pixel(300, 200, Rgb([255, 255, 255])));
        let det = FakeDetector::with(vec![
            line(20.0, 20.0, 200.0, 60.0, "the whole sentence", [0; 3]),
            line(30.0, 30.0, 80.0, 20.0, "inside", [200, 0, 0]), // another colour, so grouping keeps it apart
        ]);
        let page = detect_page_with(&det, &file, "ocr").unwrap();
        assert_eq!(page.blocks.len(), 1);
        assert_eq!(page.blocks[0].text, "the whole sentence");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_mode_on_an_image_is_refused_without_asking_the_detector() {
        let (dir, file) = page_file("det-text-mode", &RgbImage::from_pixel(50, 50, Rgb([255, 255, 255])));
        let det = FakeDetector::failing("must not be called");
        let err = detect_with(&det, &file, "text").err().unwrap();
        assert!(err.starts_with("Selectable-text mode only works on PDFs"), "{err}");
        assert_eq!(det.calls.get(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_error_of_the_detector_reaches_the_caller_unchanged() {
        let (dir, file) = page_file("det-error", &RgbImage::from_pixel(50, 50, Rgb([255, 255, 255])));
        let err = detect_page_with(&FakeDetector::failing("the OCR engine broke"), &file, "ocr").err().unwrap();
        assert_eq!(err, "the OCR engine broke");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ocr_runs_for_a_pdf_page_only_when_it_has_images_and_never_in_text_mode() {
        let (dir, file) = page_file("det-hybrid", &RgbImage::from_pixel(300, 200, Rgb([255, 255, 255])));
        let layer = |has_images| TextLayer { lines: vec![LayerLine { b: [20.0, 20.0, 100.0, 20.0], t: "Hello".into() }], has_images, ..Default::default() };
        let ocr_lines = || vec![line(22.0, 21.0, 96.0, 18.0, "Hello again", [0; 3]), line(20.0, 150.0, 100.0, 20.0, "image text", [0; 3])];
        // selectable text only: the OCR is not needed
        let det = FakeDetector::with(ocr_lines());
        let (blocks, from_layer, ran) = ocr_hybrid_with(&det, &file, &layer(false), true).unwrap();
        assert_eq!((blocks.len(), from_layer, ran, det.calls.get()), (1, 1, false, 0));
        assert_eq!(blocks[0].text, "Hello");
        // the page has images: OCR runs for the rest; the OCR line on top of selectable text is dropped
        let det = FakeDetector::with(ocr_lines());
        let (blocks, from_layer, ran) = ocr_hybrid_with(&det, &file, &layer(true), true).unwrap();
        assert_eq!((from_layer, ran, det.calls.get()), (1, true, 1));
        assert_eq!(blocks.iter().map(|b| b.text.as_str()).collect::<Vec<_>>(), vec!["Hello", "image text"]);
        // text mode: never OCR, even with images
        let det = FakeDetector::with(ocr_lines());
        let (blocks, _, ran) = ocr_hybrid_with(&det, &file, &layer(true), false).unwrap();
        assert_eq!((blocks.len(), ran, det.calls.get()), (1, false, 0));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
