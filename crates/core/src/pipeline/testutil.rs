//! Helpers shared by the tests of several modules.

use super::*;

pub(super) fn line(x: f32, y: f32, w: f32, h: f32, t: &str, fg: [u8; 3]) -> Line {
    Line { b: [x, y, w, h], t: t.into(), c: 0.9, fg, bg: [255; 3], layer: false }
}

/// The fonts and pdfium that `npm run models` downloads. They are bundled into the app from src-tauri/resources, so the tests of the
/// core read them from there.
pub(super) fn resources_dir() -> std::path::PathBuf { Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src-tauri/resources") }

pub(super) fn test_font() -> FontArc {
    let p = resources_dir().join("fonts/NotoSans-Regular.ttf");
    FontArc::try_from_vec(std::fs::read(p).unwrap()).unwrap()
}

// ---- text colour ----
/// A grey page with a "flag" on it: black, red and yellow stripes. White vertical bars (each 6 px wide, 16 px apart) stand
/// in for white lettering across the stripes, the case where the colour around the box says nothing about the text.
pub(super) fn striped_page(bars: [u8; 3]) -> RgbImage {
    let mut im = RgbImage::from_pixel(400, 200, Rgb([237, 237, 237]));
    for (y0, y1, c) in [(20, 70, [0u8, 0, 0]), (70, 130, [220, 0, 0]), (130, 180, [250, 205, 0])] {
        draw_filled_rect_mut(&mut im, Rect::at(50, y0).of_size(300, (y1 - y0) as u32), Rgb(c));
    }
    for k in 0..14 { draw_filled_rect_mut(&mut im, Rect::at(80 + 16 * k, 60).of_size(6, 80), Rgb(bars)); }
    im
}

// ---- erasing overlapping blocks as one ----
pub(super) fn blk(id: u32, bbox: [f32; 4], bg: [u8; 3]) -> Block {
    Block { id, bbox, text: String::new(), translated: None, conf: 1.0, font_px: 20.0, fg: [255; 3], bg }
}

/// A fresh, empty folder under the system temp dir, removed again by the caller.
pub(super) fn temp_dir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("lt-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The pdfium library that `npm run models` puts in resources/pdfium. The tests that need it are skipped without it.
pub(super) fn pdfium_dir() -> Option<std::path::PathBuf> {
    let d = resources_dir().join("pdfium");
    let lib = pdfium_render::prelude::Pdfium::pdfium_platform_library_name_at_path(&d);
    if lib.exists() { Some(d) } else { eprintln!("pdfium not found in {}; test skipped", d.display()); None }
}

/// A `TextDetector` that returns the same stored lines for every image (or an error), and counts how often it was asked.
/// With it, detection is tested without model files.
pub(super) struct FakeDetector { lines: Vec<Line>, error: Option<String>, pub(super) calls: std::cell::Cell<usize> }

impl FakeDetector {
    pub(super) fn with(lines: Vec<Line>) -> Self { Self { lines, error: None, calls: Default::default() } }
    pub(super) fn failing(message: &str) -> Self { Self { lines: vec![], error: Some(message.into()), calls: Default::default() } }
}

impl TextDetector for FakeDetector {
    fn lines(&self, _: &RgbImage) -> Result<Vec<Line>, String> {
        self.calls.set(self.calls.get() + 1);
        match &self.error { Some(e) => Err(e.clone()), None => Ok(self.lines.clone()) }
    }
}

/// Writes an image to a file in a fresh temp folder and returns (folder, file), for the tests that read a page from a path.
pub(super) fn page_file(name: &str, im: &RgbImage) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = temp_dir(name);
    let file = dir.join("page.png");
    im.save(&file).unwrap();
    (dir, file)
}
