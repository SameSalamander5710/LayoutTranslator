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

pub fn ocr(img: &Path, dir: &Path, cache: &EngineCache) -> Result<Vec<Block>, String> {
    let im = load_oriented(img)?.to_rgb8();
    Ok(refine(&im, group_tagged(ocr_lines(&im, dir, cache)?)))
}

// Runs the OCR engine on an in-memory image and returns the raw text lines (not yet grouped).
pub(super) fn ocr_lines(im: &RgbImage, dir: &Path, cache: &EngineCache) -> Result<Vec<Line>, String> {
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
        for o in ocr_lines(&masked, dir, cache)? {
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
    let layer = if mode == "ocr" { None } else { load_layer(img) };
    match layer {
        Some(l) => ocr_hybrid(img, dir, cache, &l, mode != "text"),
        None if mode == "text" => Err("Selectable-text mode only works on PDFs. Choose another detection method for images.".into()),
        None => ocr(img, dir, cache).map(|b| (b, 0, true)),
    }
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
}
