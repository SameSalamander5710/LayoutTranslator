//! Reading PDFs (pdfium): rasterising pages and the selectable text layer of each page.

use super::*;

// When a PDF is opened, each page's selectable text is saved next to its page image as
// page-NNN.json (pixel coordinates at the resolution stored in it, see `TextLayer::dpi`). The OCR step reads that file if it exists.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct LayerLine { pub b: [f32; 4], pub t: String }

#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct TextLayer {
    pub lines: Vec<LayerLine>, pub has_images: bool,
    /// Resolution the page image was drawn at. Large pages are drawn at a lower resolution (see `page_scale`), so the PDF
    /// export needs the real value to give the page its real size. 0 (older files) means `PDF_DPI`.
    #[serde(default)] pub dpi: f32,
}

/// Largest gap between two pieces of text on one line, as a multiple of the line height (a normal
/// word space is about 0.25; table columns and page columns are further apart than this).
pub(super) const MAX_WORD_GAP: f32 = 0.9;

/// Joins text pieces that sit on the same line and close together (words, font changes, etc.) into whole
/// lines. Pieces from different columns or table cells stay apart because their gap is too large.
pub(super) fn merge_layer_lines(src: &[LayerLine]) -> Vec<LayerLine> {
    let mut segs: Vec<&LayerLine> = src.iter().filter(|l| !l.t.trim().is_empty() && l.b[2] > 0.0 && l.b[3] > 0.0).collect();
    segs.sort_by(|a, b| a.b[0].total_cmp(&b.b[0]));
    let mut rows: Vec<LayerLine> = vec![];
    for s in segs {
        let [x, y, w, h] = s.b;
        let k = rows.iter().rposition(|r| {
            let [rx, ry, rw, rh] = r.b;
            let overlap = (y + h).min(ry + rh) - y.max(ry);
            let gap = x - (rx + rw);
            let hh = h.max(rh);
            overlap > 0.5 * h.min(rh) && h / rh > 0.55 && h / rh < 1.8 && gap > -0.8 * hh && gap < MAX_WORD_GAP * hh
        });
        match k {
            Some(i) => {
                let r = &mut rows[i];
                let gap = x - (r.b[0] + r.b[2]);
                let (nx, ny) = (r.b[0].min(x), r.b[1].min(y));
                let (nr, nb) = ((r.b[0] + r.b[2]).max(x + w), (r.b[1] + r.b[3]).max(y + h));
                if gap > 0.15 * h.max(r.b[3]) { r.t.push(' '); }
                r.t.push_str(s.t.trim());
                r.b = [nx, ny, nr - nx, nb - ny];
            }
            None => rows.push(LayerLine { b: s.b, t: s.t.trim().to_string() }),
        }
    }
    rows
}

/// The resolution a PDF page image was drawn at, read from the file saved next to it.
pub fn page_dpi(page_png: &Path) -> f32 { load_layer(page_png).map(|l| l.dpi).filter(|d| *d > 0.0).unwrap_or(PDF_DPI) }

pub fn load_layer(page_png: &Path) -> Option<TextLayer> {
    serde_json::from_slice(&std::fs::read(page_png.with_extension("json")).ok()?).ok()
}

// Selectable text lines of a page, converted from PDF points (origin bottom-left) to page-image pixels.
// TODO(local AI): rotated pages (/Rotate) and non-zero crop boxes are not handled yet.
pub(super) fn page_text_layer(page: &pdfium_render::prelude::PdfPage, scale: f32) -> TextLayer {
    use pdfium_render::prelude::*;
    // pdfium's ready-made text "segments" are often one word each, and can even join words from two
    // different lines. So read single characters instead and build the runs ourselves: consecutive
    // characters stay in one run while they share a baseline and sit close together.
    struct Run { l: f32, b: f32, r: f32, t: f32, s: String } // PDF points, y up
    fn flush(cur: &mut Option<Run>, out: &mut Vec<Run>) { if let Some(r) = cur.take() { out.push(r); } }
    let page_h = page.height().value;
    let mut runs: Vec<Run> = vec![];
    if let Ok(text) = page.text() {
        let mut cur: Option<Run> = None;
        for ch in text.chars().iter() {
            let Some(c) = ch.unicode_char() else { continue };
            if c == '\r' || c == '\n' { flush(&mut cur, &mut runs); continue; }
            if c.is_whitespace() { if let Some(u) = cur.as_mut() { if !u.s.ends_with(' ') { u.s.push(' '); } } continue; }
            if c.is_control() { continue; }
            let Ok(r) = ch.loose_bounds() else { continue };
            let (l, rt, tp, bt) = (r.left().value, r.right().value, r.top().value, r.bottom().value);
            let joins = cur.as_ref().is_some_and(|u| {
                let h = u.t - u.b;
                (tp.min(u.t) - bt.max(u.b)) > 0.5 * (tp - bt).min(h) && l - u.r > -0.8 * h && l - u.r < MAX_WORD_GAP * h
            });
            if joins {
                let u = cur.as_mut().unwrap();
                if l - u.r > 0.15 * (u.t - u.b) && !u.s.ends_with(' ') { u.s.push(' '); }
                u.l = u.l.min(l); u.r = u.r.max(rt); u.b = u.b.min(bt); u.t = u.t.max(tp); u.s.push(c);
            } else {
                flush(&mut cur, &mut runs);
                cur = Some(Run { l, b: bt, r: rt, t: tp, s: c.to_string() });
            }
        }
        flush(&mut cur, &mut runs);
    }
    let lines = runs.into_iter().filter(|u| !u.s.trim().is_empty())
        .map(|u| LayerLine { b: [u.l * scale, (page_h - u.t) * scale, (u.r - u.l) * scale, (u.t - u.b) * scale], t: u.s.trim().to_string() })
        .collect();
    let has_images = page.objects().iter().any(|o| matches!(o.object_type(), PdfPageObjectType::Image | PdfPageObjectType::XObjectForm));
    TextLayer { lines, has_images, dpi: 0.0 }
}

/// No page image gets more pixels than this on its longest side. A PDF page can be up to 200 inches wide; drawn at
/// `PDF_DPI` that would be tens of thousands of pixels and the app would run out of memory.
pub const MAX_SIDE_PX: f32 = 6000.0;

/// Pixels per PDF point for a page of the given size in points: `PDF_DPI` normally, less when the page is so large that
/// the image would be wider or taller than `MAX_SIDE_PX`.
pub fn page_scale(width_pt: f32, height_pt: f32) -> f32 {
    let (want, longest) = (PDF_DPI / 72.0, width_pt.max(height_pt));
    if longest * want <= MAX_SIDE_PX { want } else { MAX_SIDE_PX / longest }
}

pub fn rasterise_pdf(pdf: &Path, lib_dir: &Path, out_dir: &Path) -> Result<Vec<String>, String> {
    use pdfium_render::prelude::*;
    let lib_name = Pdfium::pdfium_platform_library_name_at_path(lib_dir);
    let pdfium = Pdfium::new(
        Pdfium::bind_to_library(lib_name.clone())
            .or_else(|_| Pdfium::bind_to_library(lib_dir.join(lib_name.file_name().unwrap_or_default())))
            .or_else(|_| Pdfium::bind_to_system_library())
            .map_err(|e| format!("pdfium library not found in {}: {e}", lib_dir.display()))?,
    );
    let doc = pdfium.load_pdf_from_file(pdf, None).map_err(|e| match e {
        PdfiumError::PdfiumLibraryInternalError(PdfiumInternalError::PasswordError) => "This PDF is password protected. Remove the password first.".to_string(),
        e => e.to_string(),
    })?;
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let mut v = vec![];
    for (i, page) in doc.pages().iter().enumerate() {
        let n = i + 1;
        let scale = page_scale(page.width().value, page.height().value);
        // The two maximums are a safety net for rounding; `page_scale` already keeps the page inside them.
        let cfg = PdfRenderConfig::new().scale_page_by_factor(scale).set_maximum_width(MAX_SIDE_PX as i32).set_maximum_height(MAX_SIDE_PX as i32);
        let bitmap = page.render_with_config(&cfg).map_err(|e| format!("Page {n} could not be drawn: {e}"))?;
        let img = bitmap.as_image();
        let p = out_dir.join(format!("page-{:03}.png", n));
        img.save(&p).map_err(|e| format!("Page {n} could not be saved: {e}"))?;
        let mut layer = page_text_layer(&page, scale);
        layer.dpi = scale * 72.0;
        if let Ok(j) = serde_json::to_vec(&layer) { let _ = std::fs::write(p.with_extension("json"), j); }
        v.push(p.to_string_lossy().into_owned());
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::testutil::*;

    fn ll(x: f32, y: f32, w: f32, t: &str) -> LayerLine { LayerLine { b: [x, y, w, 20.0], t: t.into() } }

    #[test]
    fn layer_words_merge_into_one_line() {
        let m = merge_layer_lines(&[ll(100.0, 50.0, 40.0, "Hello"), ll(150.0, 52.0, 50.0, "brave"), ll(210.0, 50.0, 40.0, "world")]);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].t, "Hello brave world");
        assert_eq!(m[0].b[0], 100.0);
    }

    #[test]
    fn layer_pieces_without_a_gap_are_glued() {
        let m = merge_layer_lines(&[ll(100.0, 50.0, 40.0, "Hel"), ll(140.5, 50.0, 30.0, "lo")]);
        assert_eq!(m[0].t, "Hello");
    }

    #[test]
    fn layer_columns_and_lines_stay_apart() {
        // two columns (gap of 100) on one row, plus a second row below the first
        let m = merge_layer_lines(&[ll(50.0, 50.0, 100.0, "left"), ll(250.0, 50.0, 100.0, "right"), ll(50.0, 80.0, 100.0, "below")]);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn large_pages_get_a_lower_resolution() {
        let normal = PDF_DPI / 72.0;
        assert_eq!(page_scale(595.0, 842.0), normal, "A4 keeps the normal resolution");
        assert_eq!(page_scale(842.0, 595.0), normal);
        let s = page_scale(14400.0, 14400.0); // 200 x 200 inches
        assert!(14400.0 * s <= MAX_SIDE_PX + 0.01 && s < normal, "{s}");
        let tall = page_scale(100.0, 20000.0);
        assert!(20000.0 * tall <= MAX_SIDE_PX + 0.01, "the longest side decides");
    }

    #[test]
    fn normal_pdf_pages_keep_the_normal_resolution() {
        let Some(lib) = pdfium_dir() else { return };
        let d = temp_dir("a4");
        write_pdf(&[image::RgbaImage::from_pixel(100, 140, Rgba([255, 255, 255, 255]))], 18.0, &d.join("small.pdf")).unwrap(); // 400 x 560 pt
        let pages = rasterise_pdf(&d.join("small.pdf"), &lib, &d.join("pages")).unwrap();
        assert_eq!(page_dpi(Path::new(&pages[0])), PDF_DPI);
        let im = image::open(&pages[0]).unwrap();
        assert_eq!((im.width(), im.height()), (1111, 1556)); // 400 * 200/72 and 560 * 200/72, rounded
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_password_protected_pdf_gets_a_clear_message() {
        let Some(lib) = pdfium_dir() else { return };
        let d = temp_dir("locked");
        let f = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/password-protected.pdf");
        let e = rasterise_pdf(&f, &lib, &d).unwrap_err();
        assert_eq!(e, "This PDF is password protected. Remove the password first.");
        let _ = std::fs::remove_dir_all(&d);
    }
}
