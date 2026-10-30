use ab_glyph::{FontArc, PxScale};
use base64::Engine as _;
use futures_util::stream::{self, StreamExt};
use image::{DynamicImage, Rgb, RgbImage, Rgba};
use imageproc::{drawing::{draw_filled_rect_mut, draw_text_mut, text_size}, rect::Rect};
use oar_ocr::prelude::*;
use oar_ocr::oarocr::OAROCR;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

pub const PDF_DPI: f32 = 200.0;
pub type Engine = OAROCR;

/// Reads an image and turns it upright using its EXIF orientation tag. Phone photos are often stored sideways with a tag
/// that says how to turn them; `image::open` and oar-ocr's `load_image` decode the pixels only, so such a photo would be
/// read and drawn on its side. Every place that reads a page image must use this.
pub fn load_oriented(path: &Path) -> Result<DynamicImage, String> {
    use image::{metadata::Orientation, ImageDecoder, ImageReader};
    let err = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
    let mut decoder = ImageReader::open(path).map_err(|e| err(&e))?.with_guessed_format().map_err(|e| err(&e))?.into_decoder().map_err(|e| err(&e))?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder).map_err(|e| err(&e))?;
    img.apply_orientation(orientation);
    Ok(img)
}

/// Locks a mutex even if another thread panicked while holding it. Both caches only hold finished entries, so their data
/// is still valid after such a panic; with a plain `unwrap()` one panic would make every later call panic too, until the
/// app is restarted.
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> { m.lock().unwrap_or_else(|e| e.into_inner()) }

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

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Block {
    pub id: u32, pub bbox: [f32; 4], pub text: String, pub translated: Option<String>,
    pub conf: f32, pub font_px: f32, pub fg: [u8; 3], pub bg: [u8; 3],
}

// ---------- OCR (PaddleOCR via oar-ocr) ----------
struct Line { b: [f32; 4], t: String, c: f32, fg: [u8; 3], bg: [u8; 3] }

pub fn ocr(img: &Path, dir: &Path, cache: &EngineCache) -> Result<Vec<Block>, String> {
    let im = load_oriented(img)?.to_rgb8();
    Ok(group(ocr_lines(&im, dir, cache)?))
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
                let mut det = oar_ocr::domain::TextDetectionConfig::default();
                det.score_threshold = 0.25; // default 0.3: lower catches faint/small lines
                det.box_threshold = 0.5; // default 0.6: lower keeps smaller boxes
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
                c: c,
                fg,
                bg,
            });
        }
    }
    Ok(lines)
}

// ---------- Hybrid run for PDFs: selectable text first, OCR only for the rest ----------
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
const MAX_WORD_GAP: f32 = 0.9;

/// Joins text pieces that sit on the same line and close together (words, font changes, etc.) into whole
/// lines. Pieces from different columns or table cells stay apart because their gap is too large.
fn merge_layer_lines(src: &[LayerLine]) -> Vec<LayerLine> {
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

// Fraction of box `a` that lies inside box `b`.
fn overlap_frac(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = (a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0]);
    let h = (a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1]);
    if w <= 0.0 || h <= 0.0 || a[2] * a[3] <= 0.0 { 0.0 } else { w * h / (a[2] * a[3]) }
}

/// Returns (blocks, number of lines taken from the text layer, whether OCR ran).
/// 1. Selectable text lines become blocks directly (no OCR).
/// 2. If the page has images (or no selectable text at all), OCR runs on a copy of the page with the
///    selectable text painted over, so only the remaining text is read. OCR lines that still fall
///    on selectable text are dropped.
/// 3. All lines are grouped into blocks together.
/// With `allow_ocr == false` only step 1 runs (selectable text only; images and scanned text are ignored).
pub fn ocr_hybrid(img: &Path, dir: &Path, cache: &EngineCache, layer: &TextLayer, allow_ocr: bool) -> Result<(Vec<Block>, usize, bool), String> {
    let im = load_oriented(img)?.to_rgb8();
    let mut lines: Vec<Line> = merge_layer_lines(&layer.lines).iter().filter(|l| !l.t.trim().is_empty() && l.b[2] >= 1.0 && l.b[3] >= 1.0).map(|l| {
        let (fg, bg) = box_colors(&im, l.b[0], l.b[1], l.b[2], l.b[3]);
        Line { b: l.b, t: l.t.trim().to_string(), c: 1.0, fg, bg }
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
    Ok((group(lines), n_layer, run_ocr))
}

/// Text identification method chosen in the UI:
/// - "hybrid": selectable PDF text first, OCR for the rest (plain images are just OCR'd)
/// - "ocr":    OCR only, the PDF text layer is ignored
/// - "text":   selectable PDF text only, nothing is OCR'd
/// Returns (blocks, lines taken from the text layer, whether OCR ran).
pub fn detect(img: &Path, dir: &Path, cache: &EngineCache, mode: &str) -> Result<(Vec<Block>, usize, bool), String> {
    let layer = if mode == "ocr" { None } else { load_layer(img) };
    match layer {
        Some(l) => ocr_hybrid(img, dir, cache, &l, mode != "text"),
        None if mode == "text" => Err("Selectable-text mode only works on PDFs. Choose another detection method for images.".into()),
        None => ocr(img, dir, cache).map(|b| (b, 0, true)),
    }
}

// Merge neighbouring lines into paragraph blocks. Lines only merge when they share
// column (overlap), font size (height ratio), colour, and a small vertical gap —
// so a differently-sized or differently-coloured line gets its own box.
fn group(mut ls: Vec<Line>) -> Vec<Block> {
    ls.sort_by(|a, b| a.b[1].total_cmp(&b.b[1]));
    let mut out: Vec<(Block, u32, f32)> = vec![]; // block, line count, sum of line heights
    for l in ls {
        let [x, y, w, h] = l.b;
        let k = out.iter().rposition(|(b, n, s)| {
            let [bx, by, bw, bh] = b.bbox;
            let avg = s / *n as f32;
            let min_h = h.min(avg);
            let min_w = w.min(bw);
            let overlap = (x + w).min(bx + bw) - x.max(bx);
            let gap = y - (by + bh);
            overlap > 0.5 * min_w
                && h / avg > 0.85 && h / avg < 1.18 // same font size
                && gap > -0.3 * min_h && gap < 0.5 * min_h // close lines
                && color_dist(&l.fg, &b.fg) < 90 // same text colour
        });
        match k {
            Some(i) => {
                let (b, n, s) = &mut out[i];
                let [bx, by, bw, bh] = b.bbox;
                let (nx, ny) = (bx.min(x), by.min(y));
                b.bbox = [nx, ny, (bx + bw).max(x + w) - nx, (by + bh).max(y + h) - ny];
                b.text = format!("{} {}", b.text, l.t); b.conf = b.conf.min(l.c); *n += 1; *s += h;
            }
            None => out.push((Block { id: 0, bbox: l.b, text: l.t, translated: None, conf: l.c, font_px: 0.0, fg: l.fg, bg: l.bg }, 1, h)),
        }
    }
    out.into_iter().map(|(mut b, n, s)| { b.font_px = s / n as f32 * 0.85; b }).collect()
}

fn color_dist(a: &[u8; 3], b: &[u8; 3]) -> i32 {
    (0..3).map(|c| (a[c] as i32 - b[c] as i32).abs()).sum()
}

/// Background colour (median of a ring around the box) and text colour of a box.
/// The text colour comes from `text_colour`; if that cannot tell text from background, the old rule is used: the mean of
/// the 10% of pixels that differ most from the ring colour.
fn box_colors(img: &RgbImage, x: f32, y: f32, bw: f32, bh: f32) -> ([u8; 3], [u8; 3]) {
    let (w, h) = img.dimensions();
    let cl = |v: f32, m: u32| (v.max(0.) as u32).min(m - 1);
    let (x0, y0, x1, y1) = (cl(x - 4., w), cl(y - 4., h), cl(x + bw + 4., w), cl(y + bh + 4., h));
    let mut ring: Vec<Rgb<u8>> = vec![];
    for xx in x0..=x1 { ring.push(*img.get_pixel(xx, y0)); ring.push(*img.get_pixel(xx, y1)); }
    for yy in y0..=y1 { ring.push(*img.get_pixel(x0, yy)); ring.push(*img.get_pixel(x1, yy)); }
    let mut bg = [0u8; 3];
    for c in 0..3 { let mut v: Vec<u8> = ring.iter().map(|p| p[c]).collect(); v.sort(); bg[c] = v[v.len() / 2]; }
    if let Some(fg) = text_colour(img, x, y, bw, bh) { return (fg, bg); }
    let d = |p: &Rgb<u8>| (0..3).map(|c| (p[c] as i32 - bg[c] as i32).abs()).sum::<i32>();
    let mut inner: Vec<(i32, Rgb<u8>)> = vec![];
    for yy in cl(y, h)..=cl(y + bh, h) { for xx in cl(x, w)..=cl(x + bw, w) { let p = *img.get_pixel(xx, yy); inner.push((d(&p), p)); } }
    inner.sort_by(|a, c| c.0.cmp(&a.0));
    let n = (inner.len() / 10).max(1);
    let mut fg = [0u8; 3];
    for c in 0..3 { fg[c] = (inner[..n].iter().map(|(_, p)| p[c] as u32).sum::<u32>() / n as u32) as u8; }
    (fg, bg)
}

/// The colour of the page behind the text, for every pixel of a rectangle: the median of a window around the pixel. The
/// window is wider than a letter stroke (it grows with the text height), so the strokes drop out of the median and the
/// stripe, the gradient or the photo behind them stays. It is computed once per `g * g` cell (about 1500 cells at most) from a sparse set of window pixels, so
/// a box as big as the page costs about as much as a few lines of text. `text_h` is the height of the text (it sets the window).
/// `skip` leaves out the pixels within the given colour distance of a colour (the text colour, once it is known).
struct PageColours { x0: i32, y0: i32, g: usize, nx: usize, cells: Vec<[u8; 3]> }
impl PageColours {
    fn new(w: u32, h: u32, get: &dyn Fn(u32, u32) -> [u8; 3], x0: u32, y0: u32, x1: u32, y1: u32, text_h: f32, skip: Option<([u8; 3], i32)>) -> PageColours {
        let (cw, ch) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
        let g = (((cw * ch) as f32 / 1500.0).sqrt().ceil() as usize).max(3);
        let r = ((text_h * 0.75) as i32).clamp(6, 30);
        let step = (r / 6).max(2);
        let (nx, ny) = ((cw + g - 1) / g, (ch + g - 1) / g);
        let mut cells: Vec<[u8; 3]> = Vec::with_capacity(nx * ny);
        let mut win: [Vec<u8>; 3] = Default::default();
        for cy in 0..ny { for cx in 0..nx {
            let mx = ((x0 as usize + cx * g + g / 2).min(x1 as usize)) as i32;
            let my = ((y0 as usize + cy * g + g / 2).min(y1 as usize)) as i32;
            for v in win.iter_mut() { v.clear(); }
            let mut yy = (my - r).max(0);
            while yy <= (my + r).min(h as i32 - 1) {
                let mut xx = (mx - r).max(0);
                while xx <= (mx + r).min(w as i32 - 1) {
                    let p = get(xx as u32, yy as u32);
                    // pixels that already have the text colour are left out, so lettering thicker than half the window cannot become the "background"
                    if skip.map_or(true, |(c, tol)| (0..3).map(|k| (p[k] as i32 - c[k] as i32).abs()).sum::<i32>() >= tol) { for c in 0..3 { win[c].push(p[c]); } }
                    xx += step;
                }
                yy += step;
            }
            if win[0].is_empty() { // the whole window is text-coloured: nothing better than its own colour is known
                let p = get(mx as u32, my as u32);
                cells.push([p[0], p[1], p[2]]);
                continue;
            }
            let mut m = [0u8; 3];
            for c in 0..3 { let n = win[c].len() / 2; m[c] = *win[c].select_nth_unstable(n).1; }
            cells.push(m);
        } }
        PageColours { x0: x0 as i32, y0: y0 as i32, g, nx, cells }
    }
    fn at(&self, x: i32, y: i32) -> [u8; 3] { self.cells[((y - self.y0) as usize / self.g) * self.nx + (x - self.x0) as usize / self.g] }
}

/// The colour of the text in a box, found without trusting the colour around the box.
///
/// The ring rule fails when the text sits on a busy or multi-coloured background: white letters on a red stripe look
/// "closer" to a grey page margin than the red does, so red is reported. Here every pixel is compared with the page colour
/// right behind it instead (`PageColours`). The pixels that differ most from their own surroundings are the text; the most
/// common colour among them is the answer. Returns `None` when nothing stands out, or when "text" would be most of the box
/// (the median then followed the text, not the page).
fn text_colour(img: &RgbImage, x: f32, y: f32, bw: f32, bh: f32) -> Option<[u8; 3]> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 || x >= w as f32 || y >= h as f32 { return None; }
    let (x0, y0) = (x.max(0.0) as u32, y.max(0.0) as u32);
    let (x1, y1) = (((x + bw) as u32).min(w - 1), ((y + bh) as u32).min(h - 1));
    if x1 <= x0 + 1 || y1 <= y0 + 1 { return None; }
    let get = |a: u32, c: u32| { let p = img.get_pixel(a, c); [p[0], p[1], p[2]] };
    let page = PageColours::new(w, h, &get, x0, y0, x1, y1, bh, None);
    let mut px: Vec<(i32, [u8; 3])> = Vec::with_capacity(((x1 - x0 + 1) * (y1 - y0 + 1)) as usize);
    for j in y0..=y1 { for i in x0..=x1 {
        let (p, b) = (get(i, j), page.at(i as i32, j as i32));
        px.push(((0..3).map(|c| (p[c] as i32 - b[c] as i32).abs()).sum(), p));
    } }
    let mut ds: Vec<i32> = px.iter().map(|p| p.0).collect();
    let k = (ds.len() * 98 / 100).min(ds.len() - 1);
    let dmax = *ds.select_nth_unstable(k).1; // the 98th percentile: a few speckles must not set the scale
    if dmax < 30 { return None; } // flat box, nothing stands out
    if px.iter().filter(|p| p.0 > dmax / 2).count() * 10 > px.len() * 6 { return None; } // "text" is most of the box: not trustworthy
    // Antialiased edge pixels are mixtures of text and page colour, so only the clearest 30% of the contrast range votes.
    let sel: Vec<[u8; 3]> = px.iter().filter(|p| p.0 >= dmax * 7 / 10).map(|p| p.1).collect();
    if sel.len() < 3 { return None; }
    // Most common colour: a 16 x 16 x 16 histogram, scored with its 26 neighbouring bins so a colour split by a bin edge still wins.
    let mut hist = [0u32; 4096];
    for c in &sel { hist[((c[0] >> 4) as usize) << 8 | ((c[1] >> 4) as usize) << 4 | (c[2] >> 4) as usize] += 1; }
    let score = |a: usize, b: usize, c: usize| -> u32 {
        let mut t = 0;
        for i in a.saturating_sub(1)..=(a + 1).min(15) { for j in b.saturating_sub(1)..=(b + 1).min(15) { for k in c.saturating_sub(1)..=(c + 1).min(15) { t += hist[i << 8 | j << 4 | k]; } } }
        t
    };
    let (mut best, mut best_s) = ((0usize, 0usize, 0usize), 0u32);
    for a in 0..16 { for b in 0..16 { for c in 0..16 { let sc = score(a, b, c); if sc > best_s { best_s = sc; best = (a, b, c); } } } }
    let near = |c: &[u8; 3]| ((c[0] >> 4) as i32 - best.0 as i32).abs() <= 1 && ((c[1] >> 4) as i32 - best.1 as i32).abs() <= 1 && ((c[2] >> 4) as i32 - best.2 as i32).abs() <= 1;
    let (mut sum, mut n) = ([0u32; 3], 0u32);
    for c in sel.iter().filter(|c| near(c)) { for k in 0..3 { sum[k] += c[k] as u32; } n += 1; }
    if n == 0 { return None; }
    Some([(sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8])
}

pub fn colors(img: &RgbImage, b: &mut Block) {
    let [x, y, bw, bh] = b.bbox;
    (b.fg, b.bg) = box_colors(img, x, y, bw, bh);
}

// ---------- Translation: local llama.cpp server (OpenAI-compatible endpoint, no key, any model name) ----------
/// Settings from the "Translation options" panel. Everything here is sent to the model as instructions.
#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase", default)]
pub struct TranslateOptions {
    pub tone: String,         // "", "formal" or "informal"
    pub glossary: String,     // one entry per line: `term = translation`; a line with only a term means "keep unchanged"
    pub instructions: String, // free text
    pub use_context: bool,    // send the whole page's text first, as context
    pub max_parallel: usize,  // how many requests may run at once (llama-server -np); 1 = one at a time
    pub reuse_cache: bool,    // reuse the results of an earlier run of the same, unchanged page
}
impl Default for TranslateOptions {
    fn default() -> Self { Self { tone: String::new(), glossary: String::new(), instructions: String::new(), use_context: true, max_parallel: 1, reuse_cache: true } }
}

/// False for text that a translation could only damage: numbers, dates, prices and symbols (no letters at all), URLs,
/// e-mail addresses, and one-word codes with digits such as `A12-B34` or `ISO9001`. A word that happens to contain a
/// digit (`10-Jahres-Plan`) and any text in a non-ASCII script (`第12章`) is still translated.
pub fn is_translatable(text: &str) -> bool {
    let t = text.trim();
    if !t.chars().any(char::is_alphabetic) { return false; }
    if t.chars().any(char::is_whitespace) { return true; }
    let lower = t.to_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("www.") { return false; }
    if let Some((user, host)) = t.split_once('@') { if !user.is_empty() && host.contains('.') && !host.ends_with('.') { return false; } }
    if t.is_ascii() && t.chars().any(|c| c.is_ascii_digit()) {
        // a code unless it holds a real word: a run of 4 or more letters that is not all capitals
        return t.split(|c: char| !c.is_alphabetic()).any(|run| run.chars().count() >= 4 && run.chars().any(char::is_lowercase));
    }
    true
}

/// Parses the glossary box. Separators: `=`, `=>`, `->`, `→` or a tab. Blank lines and lines starting with `#` are ignored.
pub fn parse_glossary(text: &str) -> Vec<(String, Option<String>)> {
    text.lines().filter_map(|l| {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') { return None; }
        for sep in ["=>", "->", "→", "=", "\t"] {
            if let Some((a, b)) = l.split_once(sep) {
                let (a, b) = (a.trim(), b.trim());
                return if a.is_empty() { None } else { Some((a.to_string(), if b.is_empty() { None } else { Some(b.to_string()) })) };
            }
        }
        Some((l.to_string(), None))
    }).collect()
}

const CONTEXT_MAX_CHARS: usize = 6000; // keeps the context inside small local-model windows
const BATCH_MAX_BLOCKS: usize = 25;
const BATCH_MAX_CHARS: usize = 3000;

fn system_prompt(target: &str, o: &TranslateOptions, page_text: &str) -> String {
    let mut s = format!("You are a translation engine. Translate every string in the JSON array into {target}. Return ONLY a JSON array of strings with the same length and order. Keep numbers, names and line breaks.");
    match o.tone.as_str() {
        "formal" => s.push_str("\nUse a formal tone (for example formal forms of address)."),
        "informal" => s.push_str("\nUse an informal, familiar tone (for example informal forms of address)."),
        _ => {}
    }
    // Only glossary terms that occur on this page are sent, so a long glossary does not eat the context window.
    let hay = page_text.to_lowercase();
    let terms: Vec<_> = parse_glossary(&o.glossary).into_iter().filter(|(t, _)| hay.contains(&t.to_lowercase())).collect();
    if !terms.is_empty() {
        s.push_str("\nGlossary. Whenever the source term appears, use exactly this translation. Terms marked keep must stay unchanged:");
        for (t, tr) in terms {
            match tr { Some(tr) => s.push_str(&format!("\n- {t} => {tr}")), None => s.push_str(&format!("\n- {t} => (keep unchanged)")) }
        }
    }
    let ins = o.instructions.trim();
    if !ins.is_empty() { s.push_str(&format!("\nAdditional instructions from the user: {ins}")); }
    s
}

/// The messages that come before every request for one page: the instructions and, optionally, the page's full text as
/// context. Because the prefix is identical for every request on the page, llama.cpp can reuse its cached prompt.
fn prefix(target: &str, o: &TranslateOptions, page_text: &str) -> Vec<serde_json::Value> {
    let mut v = vec![serde_json::json!({"role": "system", "content": system_prompt(target, o, page_text)})];
    if o.use_context && !page_text.trim().is_empty() {
        let ctx: String = page_text.chars().take(CONTEXT_MAX_CHARS).collect();
        v.push(serde_json::json!({"role": "user", "content": format!("For context only, here is the full text of the page that the strings come from. Do not translate it yet.\n\n{ctx}")}));
        v.push(serde_json::json!({"role": "assistant", "content": "Understood. Send me the JSON array of strings to translate."}));
    }
    v
}

/// Cleans up the endpoint address typed by the user: surrounding spaces and a trailing `/` go, and `http://` is added when
/// there is no scheme (`127.0.0.1:8080/v1` is accepted). Spaces inside the address and other schemes are refused with a
/// message, because the network error for them says nothing useful.
pub fn endpoint_url(base: &str) -> Result<String, String> {
    let t = base.trim().trim_end_matches('/');
    if t.is_empty() { return Err("Enter the endpoint URL, for example http://127.0.0.1:8080/v1".into()); }
    if t.chars().any(char::is_whitespace) { return Err("The endpoint URL must not contain spaces".into()); }
    match t.split_once("://") {
        Some((scheme, _)) if scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https") => Ok(t.to_string()),
        Some(_) => Err("URL must start with http:// or https://".into()),
        None => Ok(format!("http://{t}")),
    }
}

/// A server that is still loading its model answers 503, and a server that restarts resets the connection. Both are
/// retried a few times before the request counts as failed. (Stop still works: it drops the whole future, also the wait.)
const RETRIES: usize = 3;
const RETRY_WAIT: std::time::Duration = if cfg!(test) { std::time::Duration::from_millis(50) } else { std::time::Duration::from_secs(2) };

/// True for a connection that was reset or closed under the request (not for "nothing is listening", which is a final answer).
fn is_transient(e: &reqwest::Error) -> bool {
    let mut cur: Option<&(dyn std::error::Error + 'static)> = Some(e);
    while let Some(err) = cur {
        if let Some(io) = err.downcast_ref::<std::io::Error>() {
            use std::io::ErrorKind::*;
            if matches!(io.kind(), ConnectionReset | ConnectionAborted | BrokenPipe | UnexpectedEof) { return true; }
        }
        if err.to_string().contains("connection closed before message completed") { return true; }
        cur = err.source();
    }
    false
}

/// A reply limit, generous on purpose: a model that loops would otherwise write until its context is full, and with no
/// timeout the page would wait until the user presses Stop. A reply that is cut off still falls back to one block at a time.
fn max_tokens(texts: &[&str]) -> usize {
    (texts.iter().map(|t| t.chars().count()).sum::<usize>() * 4 + 256).clamp(256, 8192)
}

async fn call(client: &reqwest::Client, prefix: &[serde_json::Value], texts: &[&str], base: &str) -> Result<Vec<String>, String> {
    let mut messages = prefix.to_vec();
    messages.push(serde_json::json!({"role": "user", "content": serde_json::to_string(texts).unwrap()}));
    let body = serde_json::json!({"model": "default", "temperature": 0.2, "max_tokens": max_tokens(texts), "messages": messages});
    let url = format!("{}/chat/completions", base.trim_end_matches('/'));
    let mut attempt = 0;
    let r = loop {
        match client.post(&url).json(&body).send().await {
            Ok(r) if r.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE && attempt < RETRIES => {}
            Ok(r) => break r,
            Err(e) if is_transient(&e) && attempt < RETRIES => {}
            Err(e) => return Err(format!("Cannot reach {base}: {e}")),
        }
        attempt += 1;
        tokio::time::sleep(RETRY_WAIT).await;
    };
    let status = r.status();
    if !status.is_success() {
        // llama-server explains the problem in the body (for example "context too small"), so show the start of it.
        let text = r.text().await.unwrap_or_default();
        let text: String = text.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(200).collect();
        return Err(if text.is_empty() { format!("Endpoint returned HTTP {status}") } else { format!("Endpoint returned HTTP {status}: {text}") });
    }
    let j: serde_json::Value = r.json().await.map_err(|e| e.to_string())?;
    let c = j["choices"][0]["message"]["content"].as_str().ok_or("Response had no message content")?;
    extract_json_array(c)
}

/// Pulls the JSON array of strings out of a model reply. Reasoning models print `<think>...</think>` first, and chatty
/// models add a sentence or a ``` fence around the array, so the reply is cleaned up in steps: drop the reasoning,
/// strip fences, then (if it still does not parse) try every `[` up to the last `]` as the start of the array.
/// Reads a JSON array as a list of strings. A model sometimes writes a number or true/false where a string belongs
/// (`["Hello", 2]`); those become text instead of failing the whole reply. Anything else (null, nested arrays) is refused.
fn parse_strings(s: &str) -> Option<Vec<String>> {
    let items: Vec<serde_json::Value> = serde_json::from_str(s).ok()?;
    items.into_iter().map(|v| match v {
        serde_json::Value::String(t) => Some(t),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }).collect()
}

pub fn extract_json_array(raw: &str) -> Result<Vec<String>, String> {
    let s = match raw.rfind("</think>") { Some(p) => &raw[p + "</think>".len()..], None => raw };
    let s = s.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
    if let Some(v) = parse_strings(s) { return Ok(v); }
    if let Some(end) = s.rfind(']') {
        for (start, _) in s.match_indices('[') {
            if start < end { if let Some(v) = parse_strings(&s[start..=end]) { return Ok(v); } }
        }
    }
    Err(format!("Model did not return a JSON array. Reply began: {}", s.chars().take(80).collect::<String>()))
}

/// The one HTTP client for all translation requests of the app, so connections stay open between requests and pages.
/// There is deliberately no overall timeout: a slow local model may need minutes for a page.
fn http() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| reqwest::Client::builder().tcp_keepalive(std::time::Duration::from_secs(30)).build().unwrap_or_default())
}

/// Result of the "Test connection" button.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnInfo { pub message: String, pub slots: Option<u32>, pub warning: Option<String> }

/// Checks that the server answers at `base` (e.g. http://127.0.0.1:8080/v1). llama-server's `/props` page, when available,
/// also tells how many parallel slots it was started with (-np) and how many tokens of context each slot has.
pub async fn test_connection(base: &str) -> Result<ConnInfo, String> {
    let base = endpoint_url(base)?;
    let base = base.as_str();
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(5)).build().map_err(|e| e.to_string())?;
    let r = client.get(format!("{base}/models")).send().await.map_err(|e| {
        if e.is_timeout() { "Timed out after 5 seconds".to_string() }
        else if e.is_connect() { format!("Cannot connect to {base}. Is llama-server running?") }
        else { e.to_string() }
    })?;
    if !r.status().is_success() { return Err(format!("Server answered HTTP {}. Check that the URL ends with /v1", r.status())); }
    let j: serde_json::Value = r.json().await.unwrap_or_default();
    let id = j["data"][0]["id"].as_str().or(j["models"][0]["model"].as_str()).or(j["models"][0]["name"].as_str());
    let mut message = match id { Some(id) => format!("Connected. Model: {id}"), None => "Connected.".to_string() };
    let (mut slots, mut warning) = (None, None);
    let root = base.strip_suffix("/v1").unwrap_or(base);
    if let Ok(r) = client.get(format!("{root}/props")).send().await {
        if let Ok(p) = r.json::<serde_json::Value>().await {
            slots = p["total_slots"].as_u64().map(|n| n as u32);
            let ctx = p["default_generation_settings"]["n_ctx"].as_u64();
            if let Some(n) = slots { message.push_str(&format!(" · {n} slot{}", if n == 1 { "" } else { "s" })); }
            if let Some(c) = ctx {
                message.push_str(&format!(" · {c} tokens per slot"));
                if c < 3500 { warning = Some(format!("only {c} tokens per slot. Page context may not fit: start llama-server with a larger -c (about 4096 x -np)")); }
            }
        }
    }
    Ok(ConnInfo { message, slots, warning })
}

/// A reply is only a translation if it is not blank (a blank would erase the block on the page).
fn non_blank(s: String) -> Option<String> { if s.trim().is_empty() { None } else { Some(s) } }

/// Translates one chunk of strings. If the request fails (for example the page context does not fit into a slot's
/// context window) it is retried without the page text; if the model returns the wrong number of strings, the chunk is
/// translated one string at a time. A string that still gets no usable reply comes back as `None` (the block keeps its
/// source text) instead of failing the whole page; only if nothing at all could be translated is the error returned.
async fn translate_chunk(client: &reqwest::Client, pre: &[serde_json::Value], pre_min: &[serde_json::Value], texts: &[String], base: &str)
    -> Result<Vec<Option<String>>, String> {
    let refs: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
    let mut used = pre;
    let mut r = call(client, used, &refs, base).await;
    if r.is_err() && used.len() > pre_min.len() { used = pre_min; r = call(client, used, &refs, base).await; }
    match r {
        Ok(v) if v.len() == refs.len() => Ok(v.into_iter().map(non_blank).collect()),
        _ => {
            let (mut v, mut last_err) = (vec![], None);
            for t in &refs {
                match call(client, used, &[t], base).await {
                    Ok(r) => v.push(r.into_iter().next().and_then(non_blank)),
                    Err(e) => { last_err = Some(e); v.push(None); }
                }
            }
            if v.iter().all(Option::is_none) { if let Some(e) = last_err { return Err(e); } }
            Ok(v)
        }
    }
}

/// Memory of finished translations, so running an unchanged page again costs no model time. It lives in memory only (gone
/// when the app closes). The key covers the whole page, never single lines: the same line can translate differently in a
/// different scene (see `cache_key`).
#[derive(Default)]
pub struct TranslationCache(Mutex<HashMap<String, HashMap<u32, String>>>);
const CACHE_MAX_PAGES: usize = 256;

/// Identifies "this page, translated this way": every block's id and text plus everything that shapes the result (target
/// language, server, model, tone, glossary, instructions, whether page context is sent). Box positions and colours are
/// left out, so moving or resizing a box does not forget the translations.
fn cache_key(blocks: &[Block], target: &str, base: &str, model: &str, o: &TranslateOptions) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (target, base.trim().trim_end_matches('/'), model, &o.tone, &o.glossary, &o.instructions, o.use_context).hash(&mut h);
    for b in blocks { (b.id, b.text.trim()).hash(&mut h); }
    format!("{:016x}", h.finish())
}

/// The model the server has loaded (best effort), so that switching models on the same address does not serve old results.
async fn model_id(client: &reqwest::Client, base: &str) -> String {
    let r = client.get(format!("{}/models", base.trim_end_matches('/'))).timeout(std::time::Duration::from_secs(3)).send().await;
    let Ok(r) = r else { return String::new() };
    let j: serde_json::Value = r.json().await.unwrap_or_default();
    j["data"][0]["id"].as_str().or(j["models"][0]["model"].as_str()).unwrap_or("").to_string()
}

/// Translates the blocks of ONE page. Returns (blocks taken from the cache, blocks that got no translation).
/// The model first sees the whole page's text as context (if enabled), then the numbered lines to translate.
/// `only` limits translation to those block ids; those are always translated afresh, never taken from the cache.
/// With `max_parallel` > 1 the page is cut into small chunks (about two per slot) and up to `max_parallel` requests run at
/// once; each request carries the same context, so the results do not depend on the order they finish in.
pub async fn translate(blocks: &mut Vec<Block>, target: &str, base: &str, opts: &TranslateOptions, only: Option<&[u32]>, cache: &TranslationCache)
    -> Result<(usize, usize), String> {
    let par = opts.max_parallel.clamp(1, 16);
    let client = http();
    let base = endpoint_url(base)?;
    let base = base.as_str();
    let key = cache_key(blocks, target, base, &model_id(client, base).await, opts);
    // Blocks that are only numbers, URLs, e-mail addresses or codes stay as they are. A block the user asks for by id is always translated.
    let want: Vec<usize> = blocks.iter().enumerate()
        .filter(|(_, b)| !b.text.trim().is_empty() && only.map_or_else(|| is_translatable(&b.text), |o| o.contains(&b.id))).map(|(i, _)| i).collect();
    let (mut idx, mut cached) = (vec![], 0usize);
    if opts.reuse_cache && only.is_none() {
        let c = lock(&cache.0);
        let hit = c.get(&key);
        for &i in &want {
            match hit.and_then(|m| m.get(&blocks[i].id)) { Some(t) => { blocks[i].translated = Some(t.clone()); cached += 1; } None => idx.push(i) }
        }
    } else {
        idx = want;
    }
    if idx.is_empty() { return Ok((cached, 0)); }
    let page_text = blocks.iter().map(|b| b.text.trim()).filter(|t| !t.is_empty()).collect::<Vec<_>>().join("\n\n");
    let pre = prefix(target, opts, &page_text);
    let pre_min = prefix(target, &TranslateOptions { use_context: false, ..opts.clone() }, &page_text);
    let total: usize = idx.iter().map(|&i| blocks[i].text.len()).sum();
    let mut batches: Vec<Vec<usize>> = vec![];
    if par > 1 && total > 0 {
        // About two chunks per slot, balanced by text length, never bigger than one sequential batch would be.
        let mut n = (par * 2).min(idx.len()).max(total.div_ceil(BATCH_MAX_CHARS)).max(idx.len().div_ceil(BATCH_MAX_BLOCKS)).min(idx.len());
        if n > par { n = (n.div_ceil(par) * par).min(idx.len()); } // whole waves of `par` requests
        let mut chunks: Vec<Vec<usize>> = vec![vec![]; n];
        let mut cum = 0usize;
        for &i in &idx {
            let len = blocks[i].text.len();
            chunks[(((cum + len / 2) * n) / total).min(n - 1)].push(i);
            cum += len;
        }
        batches = chunks.into_iter().filter(|c| !c.is_empty()).collect();
    } else {
        let (mut cur, mut chars) = (vec![], 0usize);
        for &i in &idx {
            let n = blocks[i].text.len();
            if !cur.is_empty() && (cur.len() >= BATCH_MAX_BLOCKS || chars + n > BATCH_MAX_CHARS) { batches.push(std::mem::take(&mut cur)); chars = 0; }
            cur.push(i); chars += n;
        }
        if !cur.is_empty() { batches.push(cur); }
    }
    let jobs: Vec<(Vec<usize>, Vec<String>)> = batches.into_iter().map(|b| { let t = b.iter().map(|&i| blocks[i].text.clone()).collect(); (b, t) }).collect();
    let (pre, pre_min) = (&pre, &pre_min);
    let mut results = stream::iter(jobs).map(|(b, texts)| async move {
        let out = translate_chunk(client, pre, pre_min, &texts, base).await?;
        Ok::<_, String>((b, out))
    }).buffer_unordered(par);
    let mut fresh: Vec<(u32, String)> = vec![];
    while let Some(r) = results.next().await {
        let (b, out) = r?;
        for (&i, t) in b.iter().zip(out) {
            if let Some(t) = t {
                blocks[i].translated = Some(t.clone());
                fresh.push((blocks[i].id, t));
            }
        }
    }
    let missing = idx.len() - fresh.len();
    if !fresh.is_empty() {
        let mut c = lock(&cache.0);
        if c.len() >= CACHE_MAX_PAGES && !c.contains_key(&key) { c.clear(); }
        c.entry(key).or_default().extend(fresh);
    }
    Ok((cached, missing))
}

// ---------- Box editing helpers ----------
/// Re-estimates text and background colour for a box the user moved or resized.
pub fn sample_colors(img: &Path, bbox: [f32; 4]) -> Result<([u8; 3], [u8; 3]), String> {
    let im = load_oriented(img)?.to_rgb8();
    Ok(box_colors(&im, bbox[0], bbox[1], bbox[2], bbox[3]))
}

/// Builds a block for a box the user drew: selectable PDF text under the box if there is any (unless mode is "ocr"),
/// otherwise OCR on just that area (unless mode is "text"). The text may come back empty; the user can type it in.
pub fn read_region(img: &Path, dir: &Path, cache: &EngineCache, bbox: [f32; 4], mode: &str) -> Result<Block, String> {
    let im = load_oriented(img)?.to_rgb8();
    let (iw, ih) = im.dimensions();
    let x0 = bbox[0].max(0.0).min(iw as f32 - 1.0);
    let y0 = bbox[1].max(0.0).min(ih as f32 - 1.0);
    let region = [x0, y0, bbox[2].min(iw as f32 - x0).max(1.0), bbox[3].min(ih as f32 - y0).max(1.0)];
    let mut found: Vec<(f32, f32, String, f32)> = vec![]; // y, x, text, line height
    let mut conf = 1.0f32;
    if mode != "ocr" {
        if let Some(layer) = load_layer(img) {
            for r in merge_layer_lines(&layer.lines) {
                if overlap_frac(&r.b, &region) > 0.5 { found.push((r.b[1], r.b[0], r.t.clone(), r.b[3])); }
            }
        }
    }
    if found.is_empty() && mode != "text" {
        let crop = image::imageops::crop_imm(&im, region[0] as u32, region[1] as u32, region[2] as u32, region[3] as u32).to_image();
        for o in ocr_lines(&crop, dir, cache)? { conf = conf.min(o.c); found.push((o.b[1], o.b[0], o.t.clone(), o.b[3])); }
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let text = found.iter().map(|f| f.2.as_str()).collect::<Vec<_>>().join(" ");
    let font_px = if found.is_empty() { region[3] * 0.6 } else { found.iter().map(|f| f.3).sum::<f32>() / found.len() as f32 * 0.85 };
    let mut b = Block { id: 0, bbox: region, text, translated: None, conf: if found.is_empty() { 0.0 } else { conf }, font_px, fg: [0; 3], bg: [255; 3] };
    colors(&im, &mut b);
    Ok(b)
}

// ---------- Erase + typeset ----------
/// Cuts one word (no whitespace inside) at the places where a line may end, following the Unicode line breaking rules
/// (UAX #14): Japanese and Chinese text can break between characters but never before closing punctuation like `、` or `。`,
/// and Latin letters, digits and the like stay together. Thai has no break rules without a dictionary, so it comes back whole.
fn break_segments(word: &str) -> Vec<&str> {
    let (mut out, mut start) = (vec![], 0usize);
    for (idx, _) in unicode_linebreak::linebreaks(word) {
        if idx > start { out.push(&word[start..idx]); start = idx; }
    }
    if start < word.len() { out.push(&word[start..]); }
    out
}

/// Fits `text` into lines of at most `maxw` pixels. Words are separated by spaces (several spaces count as one); inside a
/// word, lines may also break where `break_segments` allows. A piece that is wider than a whole line is cut between
/// grapheme clusters, so a letter never loses its accents or vowel marks.
pub fn wrap(f: &FontArc, px: f32, text: &str, maxw: u32) -> Vec<String> {
    use unicode_segmentation::UnicodeSegmentation;
    let width = |s: &str| text_size(PxScale::from(px), f, s).0;
    let mut out = vec![];
    for para in text.split('\n') {
        // Common case while fitting: the whole paragraph is one line, so skip the piece-by-piece measuring.
        let joined = para.split_whitespace().collect::<Vec<_>>().join(" ");
        if width(&joined) <= maxw { out.push(joined); continue; }
        let mut line = String::new();
        for word in para.split_whitespace() {
            for (n, seg) in break_segments(word).into_iter().enumerate() {
                let cand = if line.is_empty() { seg.to_string() } else if n == 0 { format!("{line} {seg}") } else { format!("{line}{seg}") };
                if width(&cand) <= maxw { line = cand; continue; }
                if !line.is_empty() { out.push(std::mem::take(&mut line)); }
                if width(seg) <= maxw { line = seg.to_string(); continue; }
                // Wider than the box: break it between grapheme clusters.
                let mut piece = String::new();
                for g in seg.graphemes(true) {
                    if !piece.is_empty() && width(&format!("{piece}{g}")) > maxw { out.push(std::mem::take(&mut piece)); }
                    piece.push_str(g);
                }
                line = piece;
            }
        }
        out.push(line);
    }
    out
}

/// Per-character data for ONE font size: glyph id, advance, and ink top/bottom (None if the glyph has no outline, like a
/// space). Ink top/bottom do not depend on the x position, so one entry serves every occurrence of the character.
type GlyphCache = HashMap<char, (ab_glyph::GlyphId, f32, Option<(f32, f32)>)>;

// Ink extent of a line relative to the draw origin y, replicating imageproc's layout
// (baseline at y + ascent). Returns (ink top, ink bottom, advance width).
pub fn text_geom(f: &FontArc, px: f32, text: &str) -> (f32, f32, f32) { text_geom_cached(f, px, text, &mut GlyphCache::new()) }

/// Same as `text_geom`, but looks glyphs up in `cache` (which must only ever be used with this `px`). Building outlines is
/// the expensive part, and `fits` measures the same few characters over and over.
fn text_geom_cached(f: &FontArc, px: f32, text: &str, cache: &mut GlyphCache) -> (f32, f32, f32) {
    use ab_glyph::{point, Font as _, GlyphId, ScaleFont};
    let font = f.as_scaled(PxScale::from(px));
    let mut w = 0f32;
    let (mut top, mut bot) = (f32::MAX, f32::MIN);
    let mut last: Option<GlyphId> = None;
    for c in text.chars() {
        let (gid, adv, ink) = *cache.entry(c).or_insert_with(|| {
            let gid = font.glyph_id(c);
            let glyph = gid.with_scale_and_position(PxScale::from(px), point(0.0, font.ascent()));
            let ink = font.outline_glyph(glyph).map(|g| { let bb = g.px_bounds(); (bb.min.y, bb.max.y) });
            (gid, font.h_advance(gid), ink)
        });
        w += adv;
        if let Some((t, b)) = ink {
            if let Some(l) = last { w += font.kern(gid, l); }
            last = Some(gid);
            top = top.min(t);
            bot = bot.max(b);
        }
    }
    if top == f32::MAX { (0.0, 0.0, w) } else { (top, bot, w) }
}

// Line spacing: full em height so consecutive lines never overlap.
// height_unscaled is in font units, so scale by px / units_per_em.
pub fn line_spacing(f: &FontArc, px: f32) -> f32 {
    use ab_glyph::Font as _;
    let upem = f.units_per_em().unwrap_or(1000.0);
    f.height_unscaled() as f32 / upem * px
}

fn fits(f: &FontArc, text: &str, w: u32, h: u32, px: f32) -> bool {
    let lines = wrap(f, px, text, w);
    let ls = line_spacing(f, px);
    let mut cache = GlyphCache::new();
    let (mut top, mut bot) = (f32::MAX, f32::MIN);
    for (i, l) in lines.iter().enumerate() {
        if l.is_empty() { continue; }
        let (t, b, wpx) = text_geom_cached(f, px, l, &mut cache);
        if wpx > w as f32 { return false; }
        top = top.min(i as f32 * ls + t);
        bot = bot.max(i as f32 * ls + b);
    }
    if top == f32::MAX { return true; }
    bot - top <= h as f32
}

// Largest px where the wrapped text fits inside w×h (min 3px, best effort).
// Feasibility is monotonic in px (bigger px → wider lines → more lines → taller),
// so bisection over [3, hi] (stopped at 0.25 px) finds the boundary. hi bounds px because even a
// single line's ink must fit in h.
pub fn fit_px(f: &FontArc, text: &str, w: u32, h: u32) -> f32 {
    const MIN_PX: f32 = 3.0;
    let (top, bot, _) = text_geom(f, 100.0, text);
    let per_px = (bot - top) / 100.0;
    if per_px <= 0.0 || w < 1 || h < 1 { return MIN_PX; }
    if !fits(f, text, w, h, MIN_PX) { return MIN_PX; } // best effort
    let hi = (h as f32 / per_px).max(MIN_PX);
    if fits(f, text, w, h, hi) { return hi; }
    let (mut lo, mut hi) = (MIN_PX, hi);
    while hi - lo > 0.25 { // a quarter of a pixel is far below what is visible; 24 fixed steps were not needed
        let mid = (lo + hi) / 2.0;
        if fits(f, text, w, h, mid) { lo = mid; } else { hi = mid; }
    }
    lo
}

pub fn render(img: &Path, font_path: &Path, blocks: &[Block]) -> Result<image::RgbaImage, String> {
    let mut im = load_oriented(img)?.to_rgba8();
    let font = FontArc::try_from_vec(std::fs::read(font_path).map_err(|e| format!("Font missing ({}): {e}", font_path.display()))?).map_err(|e| e.to_string())?;
    // Pass 1: erase every original text area. All patches are computed from the untouched page (in parallel, one block
    // per task) and then applied in block order, so an erase can never wipe translated text of an overlapping block.
    let patches: Vec<Patch> = blocks.par_iter().filter_map(|b| erase_patch(&im, b)).collect();
    for p in &patches { p.apply(&mut im); }
    // Pass 2: lay out the translated text of every block (in parallel; this is where the font size is searched), then draw
    // only the glyphs (no box, no background), so overlapping boxes stay readable.
    let placed: Vec<Placed> = blocks.par_iter().filter_map(|b| layout(&font, b)).collect();
    for pl in &placed {
        for (i, l) in pl.lines.iter().enumerate() {
            if l.is_empty() { continue; }
            draw_text_mut(&mut im, Rgba([pl.fg[0], pl.fg[1], pl.fg[2], 255]), pl.x, (pl.y0 + i as f32 * pl.ls) as i32, PxScale::from(pl.px), &font, l);
        }
    }
    Ok(im)
}

/// The translated text of one block, fitted into its box and ready to draw.
struct Placed { x: i32, y0: f32, ls: f32, px: f32, lines: Vec<String>, fg: [u8; 3] }

fn layout(font: &FontArc, b: &Block) -> Option<Placed> {
    let [x, y, w, h] = b.bbox;
    let t = b.translated.as_deref().unwrap_or(&b.text);
    if t.trim().is_empty() || w < 4.0 || h < 4.0 { return None; }
    let px = fit_px(font, t, w as u32, h as u32);
    let lines = wrap(font, px, t, w as u32);
    let ls = line_spacing(font, px);
    let (mut top, mut bot) = (f32::MAX, f32::MIN);
    for (i, l) in lines.iter().enumerate() {
        if l.is_empty() { continue; }
        let (t2, b2, _) = text_geom(font, px, l);
        top = top.min(i as f32 * ls + t2);
        bot = bot.max(i as f32 * ls + b2);
    }
    if top == f32::MAX { return None; }
    let y0 = y + (h - (bot - top)) / 2.0 - top; // vertically centred
    Some(Placed { x: x as i32, y0, ls, px, lines, fg: b.fg })
}

pub fn png_data_url(im: &DynamicImage) -> Result<String, String> {
    let mut buf = std::io::Cursor::new(vec![]);
    im.write_to(&mut buf, image::ImageFormat::Png).map_err(|e| e.to_string())?;
    Ok(format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(buf.into_inner())))
}

// ---------- Inpainting (classical, no model) ----------
// Masks the text strokes (relative to a background colour that follows gradients), then fills them by diffusing the
// surrounding colours inward. Falls back to painting the smooth background when the mask covers too much of the box.
// TODO(local AI): optional upgrade path is a LaMa ONNX model for photos and textured backgrounds.
/// Number of diffusion passes. The mask is only thin text strokes, so a pixel is never far from known colour; 20 passes
/// differ from 60 by at most a few grey levels (see `fewer_passes_look_the_same`) and are three times faster.
const RELAX_PASSES: usize = 20;

/// When more than this share of a block's crop looks like text, the stroke mask cannot be trusted (photo, pattern, two
/// backgrounds side by side). The crop is then rebuilt as a blurred fill from the pixels around it (`frosted_patch`).
const FROST_COVER: f32 = 0.45;
/// How fast `push_pull` trusts a coarse cell that is only partly known. 1.0 is the textbook value; higher keeps colours
/// from two different backgrounds apart for longer instead of averaging them to grey.
const PULL_TRUST: f32 = 4.0;

/// What erasing one block changes in the page: the new colour of some pixels.
struct Patch(Vec<(u32, u32, [u8; 3])>);
impl Patch {
    fn apply(&self, im: &mut image::RgbaImage) {
        for (a, c, v) in &self.0 { im.put_pixel(*a, *c, Rgba([v[0], v[1], v[2], 255])); }
    }
}

/// The colour of the page behind the text at every pixel of the crop, as a tilted plane `a + b*x + c*y` per colour channel,
/// fitted by least squares to the pixels that are not text. A pixel counts as "not text" when it is close to the current
/// fit; the fit is repeated three times with a tighter limit (150, 80, 40) and starts from the block's background colour.
/// Text, and the text of a neighbouring box, is far from the fit, so it does not pull the plane. On a flat background the
/// result is that one colour; on a linear gradient it follows the gradient.
fn background_plane(im: &image::RgbaImage, x0: u32, y0: u32, cw: usize, ch: usize, bg: [u8; 3]) -> Vec<[f32; 3]> {
    let at = |i: usize, j: usize| { let q = im.get_pixel(x0 + i as u32, y0 + j as u32); [q[0] as f64, q[1] as f64, q[2] as f64] };
    let mut coef = [[0f64; 3]; 3]; // per channel: a, b, c
    for k in 0..3 { coef[k][0] = bg[k] as f64; }
    for tol in [150.0f64, 80.0, 40.0] {
        let (mut n, mut sx, mut sy, mut sxx, mut sxy, mut syy) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        let (mut sv, mut sxv, mut syv) = ([0f64; 3], [0f64; 3], [0f64; 3]);
        for j in 0..ch { for i in 0..cw {
            let q = at(i, j);
            let off: f64 = (0..3).map(|k| (q[k] - (coef[k][0] + coef[k][1] * i as f64 + coef[k][2] * j as f64)).abs()).sum();
            if off > tol { continue; }
            let (x, y) = (i as f64, j as f64);
            n += 1.0; sx += x; sy += y; sxx += x * x; sxy += x * y; syy += y * y;
            for k in 0..3 { sv[k] += q[k]; sxv[k] += x * q[k]; syv[k] += y * q[k]; }
        } }
        let m = [[n, sx, sy], [sx, sxx, sxy], [sy, sxy, syy]];
        let det = |m: &[[f64; 3]; 3]| m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let d = det(&m);
        if n < 12.0 || d.abs() < 1e-6 { break; } // too few or too flat a set of pixels to fit a plane: keep the last fit
        for k in 0..3 {
            let rhs = [sv[k], sxv[k], syv[k]];
            for col in 0..3 { // Cramer's rule: replace one column by the right-hand side
                let mut mc = m; for r in 0..3 { mc[r][col] = rhs[r]; }
                coef[k][col] = det(&mc) / d;
            }
        }
    }
    let mut out = Vec::with_capacity(cw * ch);
    for j in 0..ch { for i in 0..cw {
        let mut c = [0f32; 3];
        for k in 0..3 { c[k] = ((coef[k][0] + coef[k][1] * i as f64 + coef[k][2] * j as f64) as f32).clamp(0.0, 255.0); }
        out.push(c);
    } }
    out
}

/// Works out how to erase one block, reading the page but not changing it, so many blocks can be done at the same time.
fn erase_patch(im: &image::RgbaImage, b: &Block) -> Option<Patch> { erase_patch_with(im, b, RELAX_PASSES) }

fn erase_patch_with(im: &image::RgbaImage, b: &Block, passes: usize) -> Option<Patch> {
    let (iw, ih) = im.dimensions();
    let [x, y, w, h] = b.bbox;
    let p = 3i32;
    let x0 = (x as i32 - p).max(0); let y0 = (y as i32 - p).max(0);
    let x1 = ((x + w) as i32 + p).min(iw as i32 - 1); let y1 = ((y + h) as i32 + p).min(ih as i32 - 1);
    if x1 <= x0 || y1 <= y0 { return None; }
    let (cw, ch) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
    let px = |i: usize, j: usize| (x0 as u32 + i as u32, y0 as u32 + j as u32);
    let bgs = background_plane(im, x0 as u32, y0 as u32, cw, ch, b.bg);
    // A pixel belongs to the text if it differs from the background behind it by more than a share of the text contrast.
    // Dark text on a light page gives the old fixed limit of 60; faint text gets a lower limit, but never below 24 (noise).
    let contrast: i32 = (0..3).map(|k| (b.fg[k] as i32 - b.bg[k] as i32).abs()).sum();
    let limit = (contrast / 4).clamp(24, 60);
    let mut m = vec![false; cw * ch];
    for j in 0..ch { for i in 0..cw {
        let (a, c) = px(i, j); let q = im.get_pixel(a, c); let g = bgs[j * cw + i];
        m[j * cw + i] = (0..3).map(|k| (q[k] as f32 - g[k]).abs()).sum::<f32>() > limit as f32;
    } }
    let mut md = m.clone(); // dilate by 2 px to swallow anti-aliased edges
    for j in 0..ch { for i in 0..cw { if m[j * cw + i] {
        for dj in -2i32..=2 { for di in -2i32..=2 {
            let (a, c) = (i as i32 + di, j as i32 + dj);
            if a >= 0 && c >= 0 && (a as usize) < cw && (c as usize) < ch { md[c as usize * cw + a as usize] = true; }
        } }
    } } }
    let to_px = |k: usize, v: [f32; 3]| { let (a, c) = px(k % cw, k / cw); (a, c, [v[0].round() as u8, v[1].round() as u8, v[2].round() as u8]) };
    let cover = md.iter().filter(|v| **v).count() as f32 / (cw * ch) as f32;
    // Too much is marked as text (a photo, a pattern, two backgrounds): stroke inpainting would only smear it. Rebuild the
    // whole crop as a blurred fill of the colours around it. Only if that is impossible, paint the flat background plane.
    if cover > FROST_COVER {
        if let Some(p) = frosted_patch(im, b.bg, x0, y0, x1, y1) { return Some(p); }
        return Some(Patch((0..cw * ch).map(|k| to_px(k, bgs[k])).collect()));
    }
    let mut buf: Vec<[f32; 3]> = (0..cw * ch).map(|k| {
        if md[k] { bgs[k] } else { let (a, c) = px(k % cw, k / cw); let q = im.get_pixel(a, c); [q[0] as f32, q[1] as f32, q[2] as f32] }
    }).collect();
    for _ in 0..passes { for j in 0..ch { for i in 0..cw { if md[j * cw + i] {
        let n = [(i.saturating_sub(1), j), ((i + 1).min(cw - 1), j), (i, j.saturating_sub(1)), (i, (j + 1).min(ch - 1))];
        for k in 0..3 { buf[j * cw + i][k] = n.iter().map(|(a, c)| buf[c * cw + a][k]).sum::<f32>() / 4.0; }
    } } } }
    Some(Patch((0..cw * ch).filter(|&k| md[k]).map(|k| to_px(k, buf[k])).collect()))
}

/// Blurred replacement for the crop `x0..=x1, y0..=y1`: the crop is rebuilt only from the pixels around it (so no text
/// can leak back in), and then softened. Colours from different backgrounds on both sides of the text run into each other.
fn frosted_patch(im: &image::RgbaImage, bg: [u8; 3], x0: i32, y0: i32, x1: i32, y1: i32) -> Option<Patch> {
    let (iw, ih) = im.dimensions();
    let ctx = ((y1 - y0) / 2).clamp(8, 40); // how much surrounding page the fill may look at
    let (gx0, gy0) = ((x0 - ctx).max(0), (y0 - ctx).max(0));
    let (gx1, gy1) = ((x1 + ctx).min(iw as i32 - 1), (y1 + ctx).min(ih as i32 - 1));
    let (gw, gh) = ((gx1 - gx0 + 1) as usize, (gy1 - gy0 + 1) as usize);
    let mut rgb = Vec::with_capacity(gw * gh);
    let mut known = Vec::with_capacity(gw * gh);
    for j in 0..gh { for i in 0..gw {
        let (a, c) = (gx0 + i as i32, gy0 + j as i32);
        let q = im.get_pixel(a as u32, c as u32);
        rgb.push([q[0] as f32, q[1] as f32, q[2] as f32]);
        known.push(a < x0 || a > x1 || c < y0 || c > y1);
    } }
    if !known.iter().any(|k| *k) { return None; }
    // Take the tilted background plane out first and fill only what is left (the residual). A flat or linear background
    // then comes back exactly, and the pyramid only has to carry the real detail: photo texture, a second background.
    let plane = background_plane(im, gx0 as u32, gy0 as u32, gw, gh, bg);
    let res: Vec<[f32; 3]> = (0..gw * gh).map(|k| [0, 1, 2].map(|c| rgb[k][c] - plane[k][c])).collect();
    let mut g = push_pull(&res, &known, gw, gh);
    for k in 0..gw * gh {
        // real pixels around the hole stay real, so the edge is seamless
        g[k] = if known[k] { rgb[k] } else { [0, 1, 2].map(|c| (g[k][c] + plane[k][c]).clamp(0.0, 255.0)) };
    }
    let r = ((y1 - y0) / 6).clamp(2, 10) as usize;
    for _ in 0..2 { box_blur(&mut g, gw, gh, r); } // two box blurs are close to a Gaussian
    let px = (0..gw * gh).filter(|&k| !known[k]).map(|k| {
        let v = g[k];
        ((gx0 + (k % gw) as i32) as u32, (gy0 + (k / gw) as i32) as u32, [v[0].round().clamp(0., 255.) as u8, v[1].round().clamp(0., 255.) as u8, v[2].round().clamp(0., 255.) as u8])
    }).collect();
    Some(Patch(px))
}

/// Fills the unknown pixels of a `w * h` colour grid from the known ones ("push-pull" / normalised convolution on an
/// image pyramid). Known colours are averaged into ever coarser levels (push); then each level is filled from the
/// coarser one below it (pull). Cost is about 1.3 passes over the grid, no iteration, and no model.
fn push_pull(rgb: &[[f32; 3]], known: &[bool], w: usize, h: usize) -> Vec<[f32; 3]> {
    // Every cell holds [r*weight, g*weight, b*weight, weight]; averaging four cells keeps colour and weight in step.
    let base: Vec<[f32; 4]> = rgb.iter().zip(known).map(|(c, &k)| if k { [c[0], c[1], c[2], 1.0] } else { [0.0; 4] }).collect();
    let mut lv: Vec<(Vec<[f32; 4]>, usize, usize)> = vec![(base, w, h)];
    while lv.last().map_or(false, |l| l.1 > 1 || l.2 > 1) {
        let (src, sw, sh) = lv.last().unwrap();
        let (dw, dh) = ((sw + 1) / 2, (sh + 1) / 2);
        let mut dst = vec![[0f32; 4]; dw * dh];
        for y in 0..*sh { for x in 0..*sw {
            let s = src[y * sw + x];
            let d = &mut dst[(y / 2) * dw + x / 2];
            for k in 0..4 { d[k] += s[k] * 0.25; }
        } }
        lv.push((dst, dw, dh));
    }
    let own = |c: &[f32; 4], fallback: [f32; 3]| if c[3] > 1e-6 { [c[0] / c[3], c[1] / c[3], c[2] / c[3]] } else { fallback };
    let mut cur = vec![own(&lv[lv.len() - 1].0[0], [127.0; 3])];
    for l in (0..lv.len() - 1).rev() {
        let (src, sw, sh) = &lv[l];
        let (cw, ch) = (lv[l + 1].1, lv[l + 1].2);
        let mut next = Vec::with_capacity(sw * sh);
        for y in 0..*sh { for x in 0..*sw {
            // bilinear sample of the coarser level at this pixel's centre
            let fx = ((x as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (cw - 1) as f32);
            let fy = ((y as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (ch - 1) as f32);
            let (ix, iy) = (fx as usize, fy as usize);
            let (jx, jy) = ((ix + 1).min(cw - 1), (iy + 1).min(ch - 1));
            let (tx, ty) = (fx - ix as f32, fy - iy as f32);
            let mut up = [0f32; 3];
            for k in 0..3 {
                let top = cur[iy * cw + ix][k] * (1.0 - tx) + cur[iy * cw + jx][k] * tx;
                let bot = cur[jy * cw + ix][k] * (1.0 - tx) + cur[jy * cw + jx][k] * tx;
                up[k] = top * (1.0 - ty) + bot * ty;
            }
            let c = &src[y * sw + x];
            let (a, o) = ((c[3] * PULL_TRUST).clamp(0.0, 1.0), own(c, up)); // a cell with a quarter of its pixels known is trusted fully
            next.push([0, 1, 2].map(|k| o[k] * a + up[k] * (1.0 - a)));
        } }
        cur = next;
    }
    cur
}

/// Box blur along x, then y, with edge pixels repeated.
fn box_blur(g: &mut [[f32; 3]], w: usize, h: usize, r: usize) {
    let mut tmp = g.to_vec();
    for y in 0..h { for x in 0..w {
        let (a, b) = (x.saturating_sub(r), (x + r).min(w - 1));
        let mut s = [0f32; 3];
        for i in a..=b { for k in 0..3 { s[k] += g[y * w + i][k]; } }
        for k in 0..3 { tmp[y * w + x][k] = s[k] / (b - a + 1) as f32; }
    } }
    for y in 0..h { for x in 0..w {
        let (a, b) = (y.saturating_sub(r), (y + r).min(h - 1));
        let mut s = [0f32; 3];
        for j in a..=b { for k in 0..3 { s[k] += tmp[j * w + x][k]; } }
        for k in 0..3 { g[y * w + x][k] = s[k] / (b - a + 1) as f32; }
    } }
}

// ---------- PDF in (pdfium) ----------
// Selectable text lines of a page, converted from PDF points (origin bottom-left) to page-image pixels.
// TODO(local AI): rotated pages (/Rotate) and non-zero crop boxes are not handled yet.
fn page_text_layer(page: &pdfium_render::prelude::PdfPage, scale: f32) -> TextLayer {
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
            let joins = cur.as_ref().map_or(false, |u| {
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

// ---------- PDF out: minimal writer, one JPEG image per page (text is not selectable) ----------
fn put(o: &mut Vec<u8>, offs: &mut Vec<usize>, id: usize, body: &[u8]) {
    offs.push(o.len());
    o.extend(format!("{id} 0 obj\n").as_bytes()); o.extend(body); o.extend(b"\nendobj\n");
}
/// Writes a file so that the old one survives a failure: the bytes go to `<name>.tmp` in the same folder first and replace
/// the target with one rename only when everything was written. A full disk or a crash halfway leaves the earlier file
/// (the user's previous export) untouched.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut name = path.file_name().ok_or_else(|| format!("{} is not a file name", path.display()))?.to_os_string();
    name.push(".tmp");
    let tmp = path.with_file_name(name);
    if let Err(e) = std::fs::write(&tmp, bytes) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("Cannot write {}: {e}", path.display()));
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("Cannot write {}: {e}", path.display())
    })
}

/// Saves a rendered page as PNG (encoded in memory first, then written with `write_atomic`).
pub fn save_png(im: &image::RgbaImage, path: &Path) -> Result<(), String> {
    let mut buf = std::io::Cursor::new(vec![]);
    im.write_to(&mut buf, image::ImageFormat::Png).map_err(|e| e.to_string())?;
    write_atomic(path, &buf.into_inner())
}

/// One finished page for the PDF writer. It keeps only the JPEG (about 0.3 to 1 MB) instead of the decoded image (about
/// 15 MB for an A4 page at 200 DPI), so a long document can be exported one page at a time with little memory.
pub struct JpegPage { pub w: u32, pub h: u32, pub dpi: f32, pub jpg: Vec<u8> }

/// Encodes a rendered page. `dpi` is the resolution the image was made at; it sets the page size in the PDF.
pub fn jpeg_page(im: &image::RgbaImage, dpi: f32) -> Result<JpegPage, String> {
    use image::buffer::ConvertBuffer;
    let rgb: RgbImage = im.convert(); // drops the alpha channel without cloning the RGBA copy first
    let mut jpg = vec![];
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpg, 90).encode_image(&rgb).map_err(|e| e.to_string())?;
    Ok(JpegPage { w: im.width(), h: im.height(), dpi, jpg })
}

/// Convenience for callers that already hold every page image (tests). The app itself calls `jpeg_page` once per page.
pub fn write_pdf(pages: &[image::RgbaImage], dpi: f32, out: &Path) -> Result<(), String> {
    let pages = pages.iter().map(|im| jpeg_page(im, dpi)).collect::<Result<Vec<_>, _>>()?;
    write_pdf_pages(&pages, out)
}

pub fn write_pdf_pages(pages: &[JpegPage], out: &Path) -> Result<(), String> {
    let (mut o, mut offs) = (b"%PDF-1.4\n".to_vec(), vec![]);
    let kids: Vec<String> = (0..pages.len()).map(|i| format!("{} 0 R", 3 + 3 * i)).collect();
    put(&mut o, &mut offs, 1, b"<</Type/Catalog/Pages 2 0 R>>");
    put(&mut o, &mut offs, 2, format!("<</Type/Pages/Count {}/Kids[{}]>>", pages.len(), kids.join(" ")).as_bytes());
    for (i, pg) in pages.iter().enumerate() {
        let (pw, ph) = (pg.w as f32 * 72.0 / pg.dpi, pg.h as f32 * 72.0 / pg.dpi);
        let (pid, cid, iid) = (3 + 3 * i, 4 + 3 * i, 5 + 3 * i);
        put(&mut o, &mut offs, pid, format!("<</Type/Page/Parent 2 0 R/MediaBox[0 0 {pw:.2} {ph:.2}]/Resources<</XObject<</Im0 {iid} 0 R>>>>/Contents {cid} 0 R>>").as_bytes());
        let c = format!("q {pw:.2} 0 0 {ph:.2} 0 0 cm /Im0 Do Q");
        put(&mut o, &mut offs, cid, format!("<</Length {}>>\nstream\n{c}\nendstream", c.len()).as_bytes());
        let mut b = format!("<</Type/XObject/Subtype/Image/Width {}/Height {}/ColorSpace/DeviceRGB/BitsPerComponent 8/Filter/DCTDecode/Length {}>>\nstream\n", pg.w, pg.h, pg.jpg.len()).into_bytes();
        b.extend(&pg.jpg); b.extend(b"\nendstream");
        put(&mut o, &mut offs, iid, &b);
    }
    let xref = o.len();
    o.extend(format!("xref\n0 {}\n0000000000 65535 f \n", offs.len() + 1).as_bytes());
    for off in &offs { o.extend(format!("{off:010} 00000 n \n").as_bytes()); }
    o.extend(format!("trailer\n<</Size {}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF", offs.len() + 1).as_bytes());
    write_atomic(out, &o)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(x: f32, y: f32, w: f32, h: f32, t: &str, fg: [u8; 3]) -> Line {
        Line { b: [x, y, w, h], t: t.into(), c: 0.9, fg, bg: [255; 3] }
    }

    fn test_font() -> FontArc {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/fonts/NotoSans-Regular.ttf");
        FontArc::try_from_vec(std::fs::read(p).unwrap()).unwrap()
    }

    #[test]
    fn group_merges_same_size_close_lines() {
        let ls = vec![
            line(10., 10., 100., 20., "foo", [10, 10, 10]),
            line(10., 35., 100., 20., "bar", [12, 12, 12]),
        ];
        let b = group(ls);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].text, "foo bar");
    }

    #[test]
    fn group_splits_different_sizes() {
        let ls = vec![
            line(10., 10., 100., 20., "small", [10, 10, 10]),
            line(10., 35., 100., 40., "big", [10, 10, 10]),
        ];
        assert_eq!(group(ls).len(), 2);
    }

    #[test]
    fn group_splits_different_colours() {
        let ls = vec![
            line(10., 10., 100., 20., "black", [10, 10, 10]),
            line(10., 35., 100., 20., "red", [200, 30, 30]),
        ];
        assert_eq!(group(ls).len(), 2);
    }

    #[test]
    fn group_keeps_far_apart_lines() {
        let ls = vec![
            line(10., 10., 100., 20., "a", [10, 10, 10]),
            line(10., 80., 100., 20., "b", [10, 10, 10]),
        ];
        assert_eq!(group(ls).len(), 2);
    }

    #[test]
    fn wrap_breaks_long_words() {
        let font = test_font();
        let lines = wrap(&font, 20.0, "Einheitsfeier", 30);
        assert!(lines.len() > 1);
        for l in &lines {
            assert!(text_size(PxScale::from(20.0), &font, l).0 <= 30);
        }
    }

    #[test]
    fn wrap_japanese_keeps_punctuation_off_line_start() {
        let font = test_font();
        let text = "これは日本語の文章です。次の文章もあります、そして続きます。「引用」も入ります。";
        let lines = wrap(&font, 20.0, text, 100);
        assert!(lines.len() > 2, "{lines:?}");
        assert_eq!(lines.concat(), text, "no character lost, no space added");
        for l in &lines {
            assert!(!l.starts_with(['、', '。', '」']), "line starts with closing punctuation: {l}");
            assert!(text_size(PxScale::from(20.0), &font, l).0 <= 100, "line too wide: {l}");
        }
    }

    #[test]
    fn wrap_keeps_latin_words_whole_inside_cjk_text() {
        let font = test_font();
        let wa = text_size(PxScale::from(20.0), &font, "abcdefg").0;
        let lines = wrap(&font, 20.0, "日本語のテストabcdefg日本語のテスト", wa + 5);
        assert!(lines.iter().any(|l| l.contains("abcdefg")), "{lines:?}");
        assert_eq!(lines.concat(), "日本語のテストabcdefg日本語のテスト");
    }

    #[test]
    fn wrap_never_splits_thai_vowel_marks_from_their_letter() {
        let font = test_font();
        let text = "สวัสดีครับขอบคุณมาก";
        let lines = wrap(&font, 20.0, text, 30);
        assert!(lines.len() > 1);
        assert_eq!(lines.concat(), text);
        for l in &lines { assert!(!l.starts_with(|c: char| matches!(c, '\u{0E31}' | '\u{0E34}'..='\u{0E3A}' | '\u{0E47}'..='\u{0E4E}')), "mark starts a line: {l}"); }
    }

    #[test]
    fn wrap_latin_text_as_before() {
        let font = test_font();
        let lines = wrap(&font, 20.0, "The Day of German Unity is a public holiday", 150);
        assert_eq!(lines.join(" "), "The Day of German Unity is a public holiday");
        assert!(lines.len() > 1);
        for l in &lines { assert!(text_size(PxScale::from(20.0), &font, l).0 <= 150); }
    }

    #[test]
    fn wrap_keeps_paragraphs_and_joins_spaces() {
        let font = test_font();
        assert_eq!(wrap(&font, 10.0, "hello   world\n\nbye", 1000), vec!["hello world", "", "bye"]);
    }

    #[test]
    fn glyph_cache_gives_the_same_geometry() {
        let font = test_font();
        let mut cache = GlyphCache::new();
        for line in ["Tag der Deutschen Einheit", "Einheit Einheit", "Agg gA", "", "  "] {
            assert_eq!(text_geom_cached(&font, 23.0, line, &mut cache), text_geom(&font, 23.0, line), "line {line:?}");
        }
    }

    #[test]
    fn fit_px_result_fits_box() {
        let font = test_font();
        let text = "The Day of German Unity is a public holiday.";
        let (w, h) = (200u32, 60u32);
        let px = fit_px(&font, text, w, h);
        let lines = wrap(&font, px, text, w);
        let ls = line_spacing(&font, px);
        let (mut top, mut bot) = (f32::MAX, f32::MIN);
        for (i, l) in lines.iter().enumerate() {
            if l.is_empty() { continue; }
            let (t, b, wpx) = text_geom(&font, px, l);
            assert!(wpx <= w as f32, "line {i} too wide: {wpx}");
            top = top.min(i as f32 * ls + t);
            bot = bot.max(i as f32 * ls + b);
        }
        assert!(bot - top <= h as f32, "text too tall: {} > {h}", bot - top);
    }

    #[test]
    fn fit_px_is_maximal_and_uses_wrapping() {
        let font = test_font();
        // Tall narrow box + long text: the best fit wraps to several lines and
        // uses a larger px than the text could on a single line.
        let text = "word word word word word word word word word";
        let (w, h) = (120u32, 200u32);
        let px = fit_px(&font, text, w, h);
        assert!(fits(&font, text, w, h, px), "chosen px {px} must fit");
        assert!(!fits(&font, text, w, h, px * 1.1), "px*1.1 must not fit, so {px} is maximal");
        // It must beat the single-line size (the old bug returned the 1-line px).
        let (_, _, width100) = text_geom(&font, 100.0, text);
        let one_line_px = w as f32 / width100 * 100.0;
        assert!(px > one_line_px, "px {px} should exceed single-line px {one_line_px} via wrapping");
    }

    // The chosen px must be maximal even when the best fit is a single line
    // (guards against the search returning a px smaller than the true max).
    #[test]
    fn fit_px_maximal_single_line() {
        let font = test_font();
        for (text, w, h) in [
            ("FOUSLOND ORGANIZATION", 265u32, 19u32),
            ("PRESENTS", 77u32, 14u32),
            ("OCTOBER 3RD EXCELLA PARK", 430u32, 30u32),
        ] {
            let px = fit_px(&font, text, w, h);
            assert!(fits(&font, text, w, h, px), "{text}: chosen px {px} must fit");
            assert!(!fits(&font, text, w, h, px * 1.05), "{text}: px*1.05 must not fit, so {px} is maximal");
        }
    }

    // A page with two lines of text on a gradient, and the matching blocks.
    fn text_page(font: &FontArc, size: f32) -> (image::RgbaImage, Vec<Block>) {
        let mut im = image::RgbaImage::new(520, 140);
        for (x, y, p) in im.enumerate_pixels_mut() { let v = 200 + (x * 40 / 520) as u8; *p = Rgba([v, v, 255 - (y / 8) as u8, 255]); }
        let mut blocks = vec![];
        for (n, (x, y)) in [(10u32, 10u32), (10, 70)].into_iter().enumerate() {
            draw_text_mut(&mut im, Rgba([20, 20, 20, 255]), x as i32, y as i32, PxScale::from(size), font, "Sample text");
            blocks.push(Block { id: n as u32 + 1, bbox: [x as f32, y as f32, 500.0, size * 1.3], text: "Sample text".into(), translated: None,
                conf: 1.0, font_px: size, fg: [20, 20, 20], bg: [215, 215, 235] });
        }
        (im, blocks)
    }

    #[test]
    fn parallel_erase_matches_sequential() {
        let font = test_font();
        let (page, blocks) = text_page(&font, 28.0);
        let mut seq = page.clone();
        for b in &blocks { if let Some(p) = erase_patch(&seq, b) { p.apply(&mut seq); } }
        let mut par = page.clone();
        let patches: Vec<Patch> = blocks.par_iter().filter_map(|b| erase_patch(&page, b)).collect();
        for p in &patches { p.apply(&mut par); }
        assert!(seq == par, "blocks that do not overlap must give the same page either way");
        assert!(seq != page, "something must have been erased");
    }

    #[test]
    fn write_pdf_makes_a_well_formed_file() {
        let pages = [image::RgbaImage::from_pixel(200, 100, Rgba([10, 120, 220, 255])), image::RgbaImage::from_pixel(80, 120, Rgba([250, 250, 250, 255]))];
        let path = std::env::temp_dir().join(format!("layout-translator-test-{}.pdf", std::process::id()));
        write_pdf(&pages, 72.0, &path).unwrap();
        let data = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let find = |needle: &[u8], from: usize| data[from..].windows(needle.len()).position(|w| w == needle).map(|p| p + from);
        assert!(data.starts_with(b"%PDF-1.4") && data.ends_with(b"%%EOF"));
        // `startxref` must point at the table, and every table entry at its object (2 fixed objects + 3 per page = 8).
        let sx = find(b"startxref\n", 0).unwrap() + "startxref\n".len();
        let xref_at: usize = String::from_utf8_lossy(&data[sx..]).lines().next().unwrap().parse().unwrap();
        let header = b"xref\n0 9\n0000000000 65535 f \n";
        assert!(data[xref_at..].starts_with(header), "startxref does not point to the xref table");
        for id in 1..=8usize {
            let entry = &data[xref_at + header.len() + (id - 1) * 20..][..20];
            let off: usize = std::str::from_utf8(&entry[..10]).unwrap().parse().unwrap();
            assert!(data[off..].starts_with(format!("{id} 0 obj\n").as_bytes()), "xref entry {id} points to the wrong place");
        }
        assert!(find(b"/Count 2", 0).is_some());
        assert!(find(b"MediaBox[0 0 200.00 100.00]", 0).is_some() && find(b"MediaBox[0 0 80.00 120.00]", 0).is_some(), "page size must follow the dpi");
        let jpegs = data.windows(3).filter(|w| *w == [0xFF, 0xD8, 0xFF]).count();
        assert_eq!(jpegs, 2, "one JPEG per page");
    }

    #[test]
    fn faint_text_is_erased() {
        let font = test_font();
        let mut im = image::RgbaImage::from_pixel(300, 60, Rgba([255, 255, 255, 255]));
        draw_text_mut(&mut im, Rgba([236, 236, 236, 255]), 10, 10, PxScale::from(28.0), &font, "Faint text");
        let b = Block { id: 1, bbox: [10.0, 10.0, 250.0, 36.0], text: "Faint text".into(), translated: None, conf: 1.0, font_px: 28.0, fg: [236; 3], bg: [255; 3] };
        assert!(im.pixels().filter(|p| p[0] < 245).count() > 50, "the test page must hold faint text");
        let mut out = im.clone();
        erase_patch(&im, &b).unwrap().apply(&mut out);
        let worst = out.pixels().map(|p| 255 - p[0] as i32).max().unwrap();
        assert!(worst <= 3, "faint text is still visible (grey level {worst} below the page)");
    }

    #[test]
    fn erase_follows_a_gradient() {
        let font = test_font();
        let g = |x: u32| (120 + x * 120 / 520) as i32; // 120 at the left edge, 240 at the right
        let mut im = image::RgbaImage::new(520, 60);
        for (x, _, p) in im.enumerate_pixels_mut() { let v = g(x) as u8; *p = Rgba([v, v, v, 255]); }
        draw_text_mut(&mut im, Rgba([10, 10, 10, 255]), 10, 10, PxScale::from(28.0), &font, "Gradient text");
        let b = Block { id: 1, bbox: [10.0, 10.0, 500.0, 36.0], text: "Gradient text".into(), translated: None, conf: 1.0, font_px: 28.0, fg: [10; 3], bg: [180; 3] };
        let mut out = im.clone();
        erase_patch(&im, &b).unwrap().apply(&mut out);
        let worst = (0..60u32).flat_map(|y| (0..520u32).map(move |x| (x, y))).map(|(x, y)| (out.get_pixel(x, y)[0] as i32 - g(x)).abs()).max().unwrap();
        assert!(worst <= 8, "the erased area differs from the gradient by {worst} grey levels");
    }

    #[test]
    fn text_of_a_tightly_packed_neighbour_does_not_hide_the_text() {
        let font = test_font();
        let mut im = image::RgbaImage::from_pixel(400, 100, Rgba([255, 255, 255, 255]));
        draw_text_mut(&mut im, Rgba([20, 20, 20, 255]), 10, 10, PxScale::from(28.0), &font, "First line of text");
        draw_text_mut(&mut im, Rgba([20, 20, 20, 255]), 10, 42, PxScale::from(28.0), &font, "Second line of text");
        let b = Block { id: 1, bbox: [10.0, 10.0, 380.0, 34.0], text: "First line of text".into(), translated: None, conf: 1.0, font_px: 28.0, fg: [20; 3], bg: [255; 3] };
        let mut out = im.clone();
        erase_patch(&im, &b).unwrap().apply(&mut out);
        let worst = (0..400u32).flat_map(|x| (10..44u32).map(move |y| (x, y))).map(|(x, y)| 255 - out.get_pixel(x, y)[0] as i32).max().unwrap();
        assert!(worst <= 3, "the first line is not fully erased (grey level {worst} below the page)");
    }

    #[test]
    fn fewer_passes_look_the_same() {
        let font = test_font();
        for size in [28.0, 90.0] {
            let (page, blocks) = text_page(&font, size);
            let (mut a, mut b) = (page.clone(), page.clone());
            for bl in &blocks {
                if let Some(p) = erase_patch_with(&page, bl, 60) { p.apply(&mut a); }
                if let Some(p) = erase_patch_with(&page, bl, RELAX_PASSES) { p.apply(&mut b); }
            }
            let worst = a.pixels().zip(b.pixels()).flat_map(|(p, q)| (0..3).map(move |k| (p[k] as i32 - q[k] as i32).abs())).max().unwrap();
            assert!(worst <= 20, "size {size}: 20 passes differ from 60 by {worst} grey levels");
        }
    }

    /// A solid white bar stands in for bold text that covers most of its box (the case that used to become a flat block).
    fn bar_on(mut im: image::RgbaImage, b: &Block) -> image::RgbaImage {
        let [x, y, w, h] = b.bbox;
        draw_filled_rect_mut(&mut im, Rect::at(x as i32, y as i32).of_size(w as u32, h as u32), Rgba([255, 255, 255, 255]));
        im
    }

    #[test]
    fn push_pull_reproduces_a_constant() {
        let (w, h) = (37usize, 21usize);
        let rgb = vec![[10.0, 20.0, 30.0]; w * h];
        let known: Vec<bool> = (0..w * h).map(|k| (k % w < 5) || (k % w > 30) || (k / w < 4) || (k / w > 16)).collect();
        for p in push_pull(&rgb, &known, w, h) { for (c, v) in p.iter().zip([10.0, 20.0, 30.0]) { assert!((c - v).abs() < 0.01); } }
    }

    #[test]
    fn frosted_fill_keeps_two_backgrounds_apart() {
        let (navy, gold) = ([30u8, 60, 120], [235u8, 200, 90]);
        let mut im = image::RgbaImage::new(400, 100);
        for (x, _, p) in im.enumerate_pixels_mut() { let c = if x < 200 { navy } else { gold }; *p = Rgba([c[0], c[1], c[2], 255]); }
        let b = Block { id: 1, bbox: [60.0, 30.0, 280.0, 40.0], text: String::new(), translated: None, conf: 1.0, font_px: 30.0, fg: [255; 3], bg: [130, 130, 105] };
        let page = bar_on(im, &b);
        let mut out = page.clone();
        erase_patch(&page, &b).unwrap().apply(&mut out);
        let near = |p: &Rgba<u8>, c: [u8; 3]| (0..3).all(|k| (p[k] as i32 - c[k] as i32).abs() <= 25);
        assert!(near(out.get_pixel(80, 50), navy), "left of the split must stay navy, got {:?}", out.get_pixel(80, 50));
        assert!(near(out.get_pixel(320, 50), gold), "right of the split must stay gold, got {:?}", out.get_pixel(320, 50));
        assert!(out.pixels().all(|p| p[0].min(p[1]).min(p[2]) < 200), "some of the text is left");
    }

    #[test]
    fn frosted_fill_follows_a_photo_instead_of_one_flat_colour() {
        let truth = |x: u32| (128.0 + 100.0 * (x as f32 / 25.0).sin()) as u8;
        let mut im = image::RgbaImage::new(400, 100);
        for (x, _, p) in im.enumerate_pixels_mut() { let v = truth(x); *p = Rgba([v, 255 - v, 128, 255]); }
        let b = Block { id: 1, bbox: [60.0, 30.0, 280.0, 40.0], text: String::new(), translated: None, conf: 1.0, font_px: 30.0, fg: [255; 3], bg: [128; 3] };
        let page = bar_on(im, &b);
        let mut out = page.clone();
        erase_patch(&page, &b).unwrap().apply(&mut out);
        let row: Vec<i32> = (60..340u32).map(|x| out.get_pixel(x, 50)[0] as i32).collect();
        let spread = row.iter().max().unwrap() - row.iter().min().unwrap();
        assert!(spread > 60, "the fill is almost one colour (spread {spread})");
    }

    #[test]
    fn frosted_fill_survives_page_edges_and_a_box_as_big_as_the_page() {
        let im = image::RgbaImage::from_pixel(60, 30, Rgba([40, 90, 160, 255]));
        let corner = Block { id: 1, bbox: [0.0, 0.0, 50.0, 20.0], text: String::new(), translated: None, conf: 1.0, font_px: 20.0, fg: [255; 3], bg: [0; 3] };
        let page = bar_on(im.clone(), &corner);
        let mut out = page.clone();
        erase_patch(&page, &corner).unwrap().apply(&mut out);
        assert!(out.pixels().all(|p| p[0] < 100), "corner text is left");
        let whole = Block { bbox: [0.0, 0.0, 60.0, 30.0], ..corner };
        let page = bar_on(im, &whole);
        assert!(erase_patch(&page, &whole).is_some()); // nothing is known around the hole: must fall back, not panic
    }

    /// A grey page with a "flag" on it: black, red and yellow stripes. White vertical bars (each 6 px wide, 16 px apart) stand
    /// in for white lettering across the stripes, the case where the colour around the box says nothing about the text.
    fn striped_page(bars: [u8; 3]) -> RgbImage {
        let mut im = RgbImage::from_pixel(400, 200, Rgb([237, 237, 237]));
        for (y0, y1, c) in [(20, 70, [0u8, 0, 0]), (70, 130, [220, 0, 0]), (130, 180, [250, 205, 0])] {
            draw_filled_rect_mut(&mut im, Rect::at(50, y0).of_size(300, (y1 - y0) as u32), Rgb(c));
        }
        for k in 0..14 { draw_filled_rect_mut(&mut im, Rect::at(80 + 16 * k, 60).of_size(6, 80), Rgb(bars)); }
        im
    }

    #[test]
    fn white_letters_on_red_black_and_yellow_are_white() {
        let im = striped_page([255, 255, 255]);
        // a tight box and a loose one that reaches the grey margin (the ring colour is grey there)
        for bbox in [[70.0f32, 55.0, 260.0, 95.0], [35.0, 35.0, 330.0, 130.0]] {
            let (fg, _) = box_colors(&im, bbox[0], bbox[1], bbox[2], bbox[3]);
            assert!(fg.iter().all(|c| *c >= 235), "white text came out as {fg:?} for box {bbox:?}");
        }
    }

    #[test]
    fn text_colour_still_finds_dark_gold_and_red_text() {
        for (page, bars, want) in [([255u8, 255, 255], [20u8, 20, 20], [20i32, 20, 20]), ([30, 30, 30], [250, 205, 0], [250, 205, 0]), ([237, 237, 237], [220, 0, 0], [220, 0, 0])] {
            let mut im = RgbImage::from_pixel(300, 80, Rgb(page));
            for k in 0..12 { draw_filled_rect_mut(&mut im, Rect::at(40 + 18 * k, 20).of_size(7, 40), Rgb(bars)); }
            let (fg, _) = box_colors(&im, 30.0, 15.0, 240.0, 50.0);
            assert!((0..3).all(|c| (fg[c] as i32 - want[c]).abs() <= 12), "text {want:?} on {page:?} came out as {fg:?}");
        }
    }

    #[test]
    fn a_box_with_no_text_does_not_break_the_colours() {
        let im = RgbImage::from_pixel(100, 60, Rgb([200, 200, 200]));
        assert!(text_colour(&im, 10.0, 10.0, 50.0, 30.0).is_none());
        let _ = box_colors(&im, 10.0, 10.0, 50.0, 30.0); // falls back to the old rule, no panic
        let _ = box_colors(&im, 90.0, 50.0, 80.0, 40.0); // box reaching past the page edge
    }

    #[test]
    fn overlap_frac_basics() {
        let a = [10.0, 10.0, 20.0, 10.0];
        assert_eq!(overlap_frac(&a, &[0.0, 0.0, 100.0, 100.0]), 1.0); // fully inside
        assert_eq!(overlap_frac(&a, &[200.0, 0.0, 10.0, 10.0]), 0.0); // apart
        assert!((overlap_frac(&a, &[20.0, 10.0, 50.0, 10.0]) - 0.5).abs() < 1e-6); // half covered
    }

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
    fn text_mode_needs_a_text_layer() {
        let r = detect(Path::new("/nonexistent/page.png"), Path::new("/nonexistent"), &EngineCache::default(), "text");
        assert!(r.is_err());
    }

    #[test]
    fn glossary_parsing() {
        let g = parse_glossary("Haus = house\n# comment\n\nAcme GmbH\nA -> B\nX → Y\nEmpty =\n= nothing");
        assert_eq!(g[0], ("Haus".into(), Some("house".into())));
        assert_eq!(g[1], ("Acme GmbH".into(), None));
        assert_eq!(g[2], ("A".into(), Some("B".into())));
        assert_eq!(g[3], ("X".into(), Some("Y".into())));
        assert_eq!(g[4], ("Empty".into(), None));
        assert_eq!(g.len(), 5);
    }

    #[test]
    fn prompt_uses_only_relevant_glossary_terms() {
        let o = TranslateOptions { tone: "formal".into(), glossary: "Haus = house\nKatze = cat".into(), instructions: "Legal text.".into(), use_context: true, ..Default::default() };
        let p = system_prompt("English", &o, "Das Haus ist groß.");
        assert!(p.contains("formal tone") && p.contains("Haus => house") && p.contains("Legal text."));
        assert!(!p.contains("Katze"));
    }

    #[test]
    fn context_turn_is_optional() {
        let mut o = TranslateOptions::default();
        assert_eq!(prefix("English", &o, "Hallo Welt").len(), 3); // system + context + acknowledgement
        o.use_context = false;
        assert_eq!(prefix("English", &o, "Hallo Welt").len(), 1);
    }



    fn cblock(id: u32, text: &str) -> Block {
        Block { id, bbox: [0.0; 4], text: text.into(), translated: None, conf: 1.0, font_px: 10.0, fg: [0; 3], bg: [255; 3] }
    }

    #[test]
    fn cache_key_follows_what_shapes_the_result() {
        let o = TranslateOptions::default();
        let b = vec![cblock(1, "Hallo"), cblock(2, "Welt")];
        let k = cache_key(&b, "English", "http://x/v1", "m1", &o);
        assert_eq!(k, cache_key(&b, "English", "http://x/v1/", "m1", &o)); // trailing slash is not a difference
        let mut moved = b.clone();
        moved[0].bbox = [5.0, 5.0, 50.0, 20.0];
        assert_eq!(k, cache_key(&moved, "English", "http://x/v1", "m1", &o)); // moving a box keeps the cache
        assert_ne!(k, cache_key(&[cblock(1, "Hallo"), cblock(2, "Welt!")], "English", "http://x/v1", "m1", &o));
        assert_ne!(k, cache_key(&b, "French", "http://x/v1", "m1", &o));
        assert_ne!(k, cache_key(&b, "English", "http://x/v1", "m2", &o)); // another model
        assert_ne!(k, cache_key(&b, "English", "http://x/v1", "m1", &TranslateOptions { tone: "formal".into(), ..o.clone() }));
        assert_eq!(k, cache_key(&b, "English", "http://x/v1", "m1", &TranslateOptions { max_parallel: 8, ..o })); // speed settings do not matter
    }

    #[test]
    fn numbers_urls_and_codes_are_not_translated() {
        for t in ["123", "12.5 %", "$1,234.00", "12/03/2024", "—", "3 – 4", "https://example.com/a", "WWW.Example.com", "me@example.com",
                  "A12-B34", "ISO9001", "PN-2045-X", "  42  "] {
            assert!(!is_translatable(t), "{t:?} should be skipped");
        }
        for t in ["Rechnung", "Invoice no. 5", "10-Jahres-Plan", "Modell3", "第12章", "Seite 3 von 4", "USB-C", "Visit https://example.com", "ä"] {
            assert!(is_translatable(t), "{t:?} should be translated");
        }
    }

    #[test]
    fn json_array_extraction() {
        let want = vec!["Hello".to_string(), "World".to_string()];
        assert_eq!(extract_json_array(r#"["Hello","World"]"#).unwrap(), want);
        assert_eq!(extract_json_array("```json\n[\"Hello\",\"World\"]\n```").unwrap(), want);
        assert_eq!(extract_json_array("<think>The user wants [\"x\"]...</think>\n[\"Hello\",\"World\"]").unwrap(), want);
        assert_eq!(extract_json_array("Here is the translated JSON:\n```json\n[\"Hello\",\"World\"]\n```\nHope this helps!").unwrap(), want);
        assert_eq!(extract_json_array("[Note] Sure: [\"Hello\",\"World\"]").unwrap(), want); // a [ in the preamble
        assert!(extract_json_array("I cannot translate this.").is_err());
        assert!(extract_json_array("[\"cut off").is_err());
    }

    /// A fresh, empty folder under the system temp dir, removed again by the caller.
    fn temp_dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("lt-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
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

    #[tokio::test]
    async fn a_poisoned_cache_lock_does_not_break_translate() {
        let cache = TranslationCache::default();
        // Panic on another thread while it holds the lock, which poisons the mutex.
        let _ = std::thread::scope(|s| s.spawn(|| { let _g = cache.0.lock().unwrap(); panic!("poison the lock on purpose"); }).join());
        assert!(cache.0.is_poisoned());
        // Only numbers: nothing is sent to a model, but the cache lock is taken to look for earlier results.
        let mut blocks = vec![Block { id: 1, bbox: [0.0, 0.0, 10.0, 10.0], text: "12345".into(), translated: None, conf: 1.0, font_px: 10.0, fg: [0; 3], bg: [255; 3] }];
        let r = translate(&mut blocks, "English", "http://127.0.0.1:9/v1", &TranslateOptions::default(), None, &cache).await;
        assert_eq!(r.unwrap(), (0, 0));
    }

    #[test]
    fn exif_orientation_is_applied() {
        // The file is stored 32x16 (left half red, right half blue) with the tag "rotate 90 degrees clockwise".
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/exif-orientation-6.jpg");
        let im = load_oriented(&p).unwrap().to_rgb8();
        assert_eq!(im.dimensions(), (16, 32), "turned upright");
        let (top, bottom) = (im.get_pixel(8, 4).0, im.get_pixel(8, 28).0);
        assert!(top[0] > 200 && top[2] < 60, "top is the red half: {top:?}");
        assert!(bottom[2] > 200 && bottom[0] < 60, "bottom is the blue half: {bottom:?}");
        // The plain decoder leaves it on its side, which is the bug being fixed.
        let raw = image::open(&p).unwrap();
        assert_eq!((raw.width(), raw.height()), (32, 16));
    }

    #[test]
    fn images_without_a_tag_are_unchanged() {
        let d = temp_dir("plain");
        let p = d.join("a.png");
        RgbImage::from_pixel(30, 10, Rgb([1, 2, 3])).save(&p).unwrap();
        let im = load_oriented(&p).unwrap();
        assert_eq!((im.width(), im.height()), (30, 10));
        assert!(load_oriented(&d.join("missing.png")).unwrap_err().contains("missing.png"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn pages_can_be_encoded_one_at_a_time() {
        let a = jpeg_page(&image::RgbaImage::from_pixel(200, 100, Rgba([10, 120, 220, 255])), 72.0).unwrap();
        let b = jpeg_page(&image::RgbaImage::from_pixel(80, 120, Rgba([250, 250, 250, 255])), 144.0).unwrap();
        assert_eq!((a.w, a.h), (200, 100));
        assert!(a.jpg.starts_with(&[0xFF, 0xD8, 0xFF]) && a.jpg.len() < 5_000, "a JPEG, far smaller than the 80 KB of pixels");
        let path = temp_dir("pages").join("two.pdf");
        write_pdf_pages(&[a, b], &path).unwrap();
        let data = std::fs::read(&path).unwrap();
        let has = |n: &[u8]| data.windows(n.len()).any(|w| w == n);
        assert!(has(b"/Count 2") && has(b"MediaBox[0 0 200.00 100.00]") && has(b"MediaBox[0 0 40.00 60.00]"), "each page uses its own dpi");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// The pdfium library that `npm run models` puts in resources/pdfium. The tests that need it are skipped without it.
    fn pdfium_dir() -> Option<std::path::PathBuf> {
        let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/pdfium");
        let lib = pdfium_render::prelude::Pdfium::pdfium_platform_library_name_at_path(&d);
        if lib.exists() { Some(d) } else { eprintln!("pdfium not found in {}; test skipped", d.display()); None }
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
    fn a_200_inch_pdf_page_opens_small_and_exports_at_full_size() {
        let Some(lib) = pdfium_dir() else { return };
        let d = temp_dir("huge");
        // A 100 x 100 pixel image at 0.5 dpi is a page of 14400 x 14400 points = 200 x 200 inches.
        write_pdf(&[image::RgbaImage::from_pixel(100, 100, Rgba([30, 90, 200, 255]))], 0.5, &d.join("huge.pdf")).unwrap();
        let pages = rasterise_pdf(&d.join("huge.pdf"), &lib, &d.join("pages")).unwrap();
        assert_eq!(pages.len(), 1);
        let im = image::open(&pages[0]).unwrap();
        assert!(im.width() <= 6000 && im.height() <= 6000, "{}x{}", im.width(), im.height());
        let dpi = page_dpi(Path::new(&pages[0]));
        assert!(dpi > 20.0 && dpi < 40.0, "recorded dpi {dpi}");
        // Export: the page must come out at its real physical size again.
        let jp = jpeg_page(&im.to_rgba8(), dpi).unwrap();
        write_pdf_pages(&[jp], &d.join("back.pdf")).unwrap();
        let text = String::from_utf8_lossy(&std::fs::read(d.join("back.pdf")).unwrap()).into_owned();
        let w: f32 = text.split("MediaBox[0 0 ").nth(1).unwrap().split(' ').next().unwrap().parse().unwrap();
        assert!((w - 14400.0).abs() < 20.0, "exported width {w} pt");
        let _ = std::fs::remove_dir_all(&d);
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

    // ----- R8: translation endpoint -----

    #[test]
    fn endpoint_urls_are_cleaned_up() {
        assert_eq!(endpoint_url("  http://127.0.0.1:8080/v1/  ").unwrap(), "http://127.0.0.1:8080/v1");
        assert_eq!(endpoint_url("127.0.0.1:8080/v1").unwrap(), "http://127.0.0.1:8080/v1", "scheme added");
        assert_eq!(endpoint_url("localhost:8080").unwrap(), "http://localhost:8080");
        assert_eq!(endpoint_url("HTTPS://example.com/v1").unwrap(), "HTTPS://example.com/v1");
        assert!(endpoint_url("http://127.0.0.1 :8080/v1").unwrap_err().contains("spaces"));
        assert!(endpoint_url("ftp://host/v1").unwrap_err().contains("http://"));
        assert!(endpoint_url("   ").is_err());
    }

    #[test]
    fn numbers_and_booleans_in_a_reply_become_text() {
        assert_eq!(extract_json_array(r#"["Hello", 2]"#).unwrap(), ["Hello", "2"]);
        assert_eq!(extract_json_array(r#"Here: [1.5, true, "x"]"#).unwrap(), ["1.5", "true", "x"]);
        assert!(extract_json_array(r#"["a", null]"#).is_err());
        assert!(extract_json_array(r#"[["a"]]"#).is_err());
    }

    #[test]
    fn the_reply_limit_follows_the_input_and_stays_in_range() {
        assert_eq!(max_tokens(&[]), 256);
        assert_eq!(max_tokens(&["abcde"]), 20 + 256);
        assert_eq!(max_tokens(&["日本語"]), 12 + 256, "counts characters, not bytes");
        assert_eq!(max_tokens(&[&"x".repeat(100_000)]), 8192);
    }

    /// Answers the given (status, body) pairs, one per request in order, and records each request it received. Status 0 hangs up.
    async fn mock_server(replies: Vec<(u16, String)>) -> (String, Arc<Mutex<Vec<String>>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(vec![]));
        let log = seen.clone();
        tokio::spawn(async move {
            for (status, body) in replies {
                let (mut sock, _) = listener.accept().await.unwrap();
                let (mut buf, mut tmp) = (vec![], [0u8; 4096]);
                loop {
                    let n = sock.read(&mut tmp).await.unwrap();
                    if n == 0 { break; }
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(h) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..h]).to_lowercase();
                        let len = head.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0);
                        if buf.len() >= h + 4 + len { break; }
                    }
                }
                log.lock().unwrap().push(String::from_utf8_lossy(&buf).into_owned());
                if status == 0 { continue; } // status 0: hang up without answering (the socket is dropped here)
                let resp = format!("HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            }
        });
        (url, seen)
    }
    const OK_BODY: &str = r#"{"choices":[{"message":{"content":"[\"Hallo\"]"}}]}"#;

    #[tokio::test]
    async fn a_loading_server_is_retried_until_it_answers() {
        let (url, seen) = mock_server(vec![(503, "Loading model".into()), (503, "Loading model".into()), (200, OK_BODY.into())]).await;
        let r = call(&reqwest::Client::new(), &[], &["Hello"], &url).await;
        assert_eq!(r.unwrap(), ["Hallo"]);
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 3, "two 503 answers, then the real one");
        assert!(seen[2].contains("\"max_tokens\":276"), "reply limit is sent: {}", seen[2]);
    }

    #[tokio::test]
    async fn a_server_that_stays_unavailable_gives_up_with_its_message() {
        let (url, seen) = mock_server(vec![(503, "model is still loading".into()); 4]).await;
        let e = call(&reqwest::Client::new(), &[], &["Hello"], &url).await.unwrap_err();
        assert!(e.contains("503") && e.contains("model is still loading"), "{e}");
        assert_eq!(seen.lock().unwrap().len(), 4, "one try and three retries");
    }

    #[tokio::test]
    async fn an_error_answer_shows_the_start_of_its_body() {
        let body = format!(r#"{{"error":{{"message":"context too small {}"}}}}"#, "x".repeat(500));
        let (url, seen) = mock_server(vec![(400, body)]).await;
        let e = call(&reqwest::Client::new(), &[], &["Hello"], &url).await.unwrap_err();
        assert!(e.starts_with("Endpoint returned HTTP 400 Bad Request: ") && e.contains("context too small"), "{e}");
        assert!(e.chars().count() < 260, "the body is cut at 200 characters");
        assert_eq!(seen.lock().unwrap().len(), 1, "a 400 is final, no retry");
    }

    #[tokio::test]
    async fn a_connection_that_is_cut_is_retried() {
        let (url, seen) = mock_server(vec![(0, String::new()), (200, OK_BODY.into())]).await;
        let r = call(&reqwest::Client::new(), &[], &["Hello"], &url).await;
        assert_eq!(r.unwrap(), ["Hallo"]);
        assert_eq!(seen.lock().unwrap().len(), 2);
    }

    // ----- R7: atomic writes -----

    #[test]
    fn write_atomic_writes_and_replaces() {
        let d = temp_dir("atomic");
        let f = d.join("out.pdf");
        write_atomic(&f, b"first").unwrap();
        assert_eq!(std::fs::read(&f).unwrap(), b"first");
        write_atomic(&f, b"second, longer").unwrap();
        assert_eq!(std::fs::read(&f).unwrap(), b"second, longer");
        assert!(!d.join("out.pdf.tmp").exists(), "no temporary file is left behind");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_failed_write_leaves_the_old_file_alone() {
        let d = temp_dir("atomic-fail");
        let f = d.join("out.pdf");
        std::fs::write(&f, b"the user's earlier export").unwrap();
        // The temporary name is taken by a folder, so writing it fails (works the same on every OS, also when run as admin).
        std::fs::create_dir(d.join("out.pdf.tmp")).unwrap();
        assert!(write_atomic(&f, b"new").unwrap_err().contains("out.pdf"));
        assert_eq!(std::fs::read(&f).unwrap(), b"the user's earlier export");
        assert!(d.join("out.pdf.tmp").is_dir(), "what was in the way is not deleted");
        // A missing folder is an error too, not a panic.
        assert!(write_atomic(&d.join("nope").join("x.png"), b"x").is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn png_and_pdf_exports_go_through_the_atomic_writer() {
        let d = temp_dir("export");
        let im = image::RgbaImage::from_pixel(8, 6, Rgba([1, 2, 3, 255]));
        save_png(&im, &d.join("p.png")).unwrap();
        assert_eq!(image::open(d.join("p.png")).unwrap().to_rgba8(), im);
        write_pdf(&[im], 72.0, &d.join("p.pdf")).unwrap();
        assert!(std::fs::read(d.join("p.pdf")).unwrap().starts_with(b"%PDF-1.4"));
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 2, "only the two results, no .tmp files");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn language_names_cannot_leave_the_models_folder() {
        for ok in ["en", "ja", "zh-CN", "chinese_traditional", "My Language 2"] { assert!(valid_lang_name(ok), "{ok}"); }
        for bad in ["../x", "..", "a/b", "a\\b", "..\\x", "C:", "C:\\x", "/etc", "a..b", "", "a\0b"] { assert!(!valid_lang_name(bad), "{bad:?}"); }
    }
}
