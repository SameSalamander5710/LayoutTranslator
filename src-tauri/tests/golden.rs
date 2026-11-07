//! Golden-image tests: whole pages go through the erase and the typesetting (`pipeline::render`) and the result is compared
//! with a stored PNG. The unit tests check the logic; these check what a person would see, so a change to a tuning constant
//! (see "Tuning constants" in PLAN.md) shows which pages it changes.
//!
//! The pages are drawn in code (no artwork, no binary input files); only the expected results are stored, in `tests/golden/`.
//!
//! * A page differs: the test writes the new result to the temp folder, names the file and says how big the difference is.
//!   Look at both images. If the new one is better, or the change is intended: `UPDATE_GOLDEN=1 cargo test --test golden`
//!   and commit the changed PNGs.
//! * The fonts come from `npm run models`.
//!
//! The same machine gives the same pixels; the tolerance only covers rounding differences between platforms.

use ab_glyph::{FontArc, PxScale};
use image::{Rgb, RgbImage, RgbaImage};
use imageproc::{drawing::{draw_filled_rect_mut, draw_text_mut}, rect::Rect};
use layout_translator::pipeline::{render, Block};
use std::path::{Path, PathBuf};

const MEAN_TOLERANCE: f64 = 0.25; // mean difference per pixel, in grey levels (0 to 255)
const BIG_DIFF: u8 = 32; // a pixel is "different" when one of its channels is this much off ...
const BIG_SHARE: f64 = 0.0005; // ... and at most this share of the pixels may be

fn font_path() -> PathBuf { Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/fonts/NotoSans-Regular.ttf") }

fn font() -> FontArc {
    FontArc::try_from_vec(std::fs::read(font_path()).expect("the font is missing: run `npm run models`")).unwrap()
}

fn rect(im: &mut RgbImage, x: i32, y: i32, w: u32, h: u32, c: [u8; 3]) { draw_filled_rect_mut(im, Rect::at(x, y).of_size(w, h), Rgb(c)); }

fn text(im: &mut RgbImage, f: &FontArc, x: i32, y: i32, px: f32, c: [u8; 3], s: &str) { draw_text_mut(im, Rgb(c), x, y, PxScale::from(px), f, s); }

fn block(id: u32, bbox: [f32; 4], text: &str, translated: &str, fg: [u8; 3], bg: [u8; 3]) -> Block {
    Block { id, bbox, text: text.into(), translated: Some(translated.into()), conf: 1.0, font_px: bbox[3] * 0.8, fg, bg }
}

/// Saves the page, renders it with the blocks, removes the file again.
fn render_page(name: &str, page: &RgbImage, blocks: &[Block]) -> RgbaImage {
    let dir = std::env::temp_dir().join(format!("lt-golden-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("page.png");
    page.save(&file).unwrap();
    let out = render(&file, &font_path(), blocks).expect("render failed");
    let _ = std::fs::remove_dir_all(&dir);
    out
}

/// None when the images are the same within the tolerance, else a sentence that says how they differ.
fn differs(expected: &RgbaImage, actual: &RgbaImage) -> Option<String> {
    if expected.dimensions() != actual.dimensions() {
        return Some(format!("the size is {:?}, expected {:?}", actual.dimensions(), expected.dimensions()));
    }
    let (mut sum, mut big) = (0u64, 0u64);
    for (a, b) in expected.pixels().zip(actual.pixels()) {
        let d: Vec<u8> = (0..3).map(|k| a[k].abs_diff(b[k])).collect();
        sum += d.iter().map(|v| *v as u64).sum::<u64>();
        if d.iter().any(|v| *v >= BIG_DIFF) { big += 1; }
    }
    let n = (expected.width() * expected.height()) as f64;
    let (mean, share) = (sum as f64 / (3.0 * n), big as f64 / n);
    (mean > MEAN_TOLERANCE || share > BIG_SHARE)
        .then(|| format!("mean difference {mean:.3} grey levels (allowed {MEAN_TOLERANCE}), {:.3}% of the pixels differ by {BIG_DIFF} or more (allowed {:.3}%)", share * 100.0, BIG_SHARE * 100.0))
}

fn check(name: &str, actual: &RgbaImage) {
    let expected_file = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(format!("{name}.png"));
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        actual.save(&expected_file).unwrap();
        eprintln!("updated {}", expected_file.display());
        return;
    }
    let expected = image::open(&expected_file)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}. Create it with UPDATE_GOLDEN=1 cargo test --test golden", expected_file.display()))
        .to_rgba8();
    if let Some(why) = differs(&expected, actual) {
        let new_file = std::env::temp_dir().join(format!("lt-golden-{name}.actual.png"));
        actual.save(&new_file).unwrap();
        panic!("page \"{name}\" changed: {why}.\n  expected: {}\n  new:      {}\n  Look at both. If the change is intended: UPDATE_GOLDEN=1 cargo test --test golden", expected_file.display(), new_file.display());
    }
}

/// A flag-like page: black, red and gold stripes on a grey page, white and gold and black lettering across them. The boxes
/// are loose, like detected ones, and two of them overlap. Covers: one fill for overlapping blocks, dense lettering that keeps
/// the stripes it was written on, three text colours, the font that fits the box.
fn poster() -> (RgbImage, Vec<Block>) {
    let f = font();
    let mut im = RgbImage::from_pixel(480, 300, Rgb([237, 237, 237]));
    rect(&mut im, 60, 30, 360, 70, [10, 10, 10]);
    rect(&mut im, 60, 100, 360, 80, [220, 0, 0]);
    rect(&mut im, 60, 180, 360, 70, [250, 205, 0]);
    text(&mut im, &f, 140, 44, 40.0, [250, 205, 0], "TAG DER");
    text(&mut im, &f, 76, 104, 72.0, [255, 255, 255], "Deutschen");
    text(&mut im, &f, 110, 178, 64.0, [20, 20, 20], "Einheit");
    text(&mut im, &f, 60, 262, 20.0, [30, 30, 30], "Let's join hands to celebrate");
    let blocks = vec![
        block(1, [130.0, 38.0, 220.0, 58.0], "TAG DER", "DAY OF", [250, 205, 0], [10, 10, 10]),
        block(2, [66.0, 102.0, 340.0, 86.0], "Deutschen", "German", [255, 255, 255], [220, 0, 0]),
        block(3, [100.0, 184.0, 280.0, 66.0], "Einheit", "Unity", [20, 20, 20], [250, 205, 0]),
        block(4, [56.0, 258.0, 340.0, 30.0], "Let's join hands to celebrate", "Join us to celebrate together", [30, 30, 30], [237, 237, 237]),
    ];
    (im, blocks)
}

/// A plain scan: black text on white. Covers: the stroke fill (thin text on a flat page), a block of several lines whose
/// translation is much longer (wrapping, a smaller font), a heading, a small grey line.
fn scan() -> (RgbImage, Vec<Block>) {
    let f = font();
    let mut im = RgbImage::from_pixel(480, 300, Rgb([255, 255, 255]));
    for (i, l) in ["The quick brown fox jumps", "over the lazy dog and runs", "far away into the woods."].iter().enumerate() {
        text(&mut im, &f, 30, 30 + 26 * i as i32, 22.0, [0, 0, 0], l);
    }
    text(&mut im, &f, 30, 150, 36.0, [0, 0, 0], "Chapter One");
    text(&mut im, &f, 30, 262, 14.0, [90, 90, 90], "page 1 of 3");
    let blocks = vec![
        block(1, [26.0, 26.0, 340.0, 80.0], "The quick brown fox jumps over the lazy dog and runs far away into the woods.",
              "Der schnelle braune Fuchs springt über den faulen Hund und läuft weit weg in den Wald, wo er sich unter dem alten Baum ausruht.", [0, 0, 0], [255, 255, 255]),
        block(2, [26.0, 146.0, 250.0, 48.0], "Chapter One", "Kapitel Eins", [0, 0, 0], [255, 255, 255]),
        block(3, [26.0, 258.0, 110.0, 22.0], "page 1 of 3", "Seite 1 von 3", [90, 90, 90], [255, 255, 255]),
    ];
    (im, blocks)
}

/// Light text on a busy, patterned background (a checker of two blues with a diagonal ramp). The stroke mask cannot be
/// trusted there, so this covers the blurred fill, and light text on a dark page.
fn dark() -> (RgbImage, Vec<Block>) {
    let f = font();
    let mut im = RgbImage::new(480, 240);
    for (x, y, p) in im.enumerate_pixels_mut() {
        let checker = ((x / 5 + y / 5) % 2) as i32 * 34;
        let ramp = ((x + y) / 8) as i32 % 30;
        *p = Rgb([(28 + checker + ramp) as u8, (34 + ramp) as u8, (84 + checker) as u8]);
    }
    text(&mut im, &f, 40, 50, 48.0, [240, 240, 200], "Night market");
    text(&mut im, &f, 40, 130, 26.0, [240, 240, 200], "open until midnight");
    let blocks = vec![
        block(1, [34.0, 44.0, 320.0, 66.0], "Night market", "Nachtmarkt", [240, 240, 200], [50, 54, 110]),
        block(2, [34.0, 124.0, 280.0, 40.0], "open until midnight", "geöffnet bis Mitternacht", [240, 240, 200], [50, 54, 110]),
    ];
    (im, blocks)
}

#[test]
fn the_poster_keeps_its_stripes_and_gets_the_translation() { let (im, b) = poster(); check("poster", &render_page("poster", &im, &b)); }

#[test]
fn the_plain_scan_is_cleaned_and_the_long_translation_is_wrapped() { let (im, b) = scan(); check("scan", &render_page("scan", &im, &b)); }

#[test]
fn light_text_on_a_busy_background_is_replaced_by_a_soft_blur() { let (im, b) = dark(); check("dark", &render_page("dark", &im, &b)); }

#[test]
fn a_page_without_blocks_is_unchanged() {
    let (im, _) = scan();
    let out = render_page("empty", &im, &[]);
    let same = RgbaImage::from_fn(im.width(), im.height(), |x, y| { let p = im.get_pixel(x, y); image::Rgba([p[0], p[1], p[2], 255]) });
    assert_eq!(differs(&same, &out), None);
}

#[test]
fn the_comparison_notices_a_changed_page_and_accepts_the_same_page() {
    let (im, mut blocks) = poster();
    let a = render_page("cmp-a", &im, &blocks);
    assert_eq!(differs(&a, &a), None);
    blocks[1].translated = Some("Germany".into()); // another word in the big title
    let b = render_page("cmp-b", &im, &blocks);
    assert!(differs(&a, &b).is_some(), "a different title must show as a difference");
    assert!(differs(&a, &RgbaImage::new(10, 10)).unwrap().contains("size"));
}
