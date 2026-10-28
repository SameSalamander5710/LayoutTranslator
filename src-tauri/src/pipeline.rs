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
    let im = load_image(img).map_err(|e| e.to_string())?;
    Ok(group(ocr_lines(&im, dir, cache)?))
}

// Runs the OCR engine on an in-memory image and returns the raw text lines (not yet grouped).
fn ocr_lines(im: &RgbImage, dir: &Path, cache: &EngineCache) -> Result<Vec<Line>, String> {
    let p = |n: &str| dir.join(n).to_string_lossy().to_string();
    let key = dir.to_string_lossy().to_string();
    let engine = {
        let mut m = cache.0.lock().unwrap();
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
    for r in &res[0].text_regions {
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
// page-NNN.json (pixel coordinates at PDF_DPI). The OCR step reads that file if it exists.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct LayerLine { pub b: [f32; 4], pub t: String }
#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct TextLayer { pub lines: Vec<LayerLine>, pub has_images: bool }

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
    let im = load_image(img).map_err(|e| e.to_string())?;
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

// Estimate background (median of a ring around the box) and text colour (mean of the most contrasting pixels).
fn box_colors(img: &RgbImage, x: f32, y: f32, bw: f32, bh: f32) -> ([u8; 3], [u8; 3]) {
    let (w, h) = img.dimensions();
    let cl = |v: f32, m: u32| (v.max(0.) as u32).min(m - 1);
    let (x0, y0, x1, y1) = (cl(x - 4., w), cl(y - 4., h), cl(x + bw + 4., w), cl(y + bh + 4., h));
    let mut ring: Vec<Rgb<u8>> = vec![];
    for xx in x0..=x1 { ring.push(*img.get_pixel(xx, y0)); ring.push(*img.get_pixel(xx, y1)); }
    for yy in y0..=y1 { ring.push(*img.get_pixel(x0, yy)); ring.push(*img.get_pixel(x1, yy)); }
    let mut bg = [0u8; 3];
    for c in 0..3 { let mut v: Vec<u8> = ring.iter().map(|p| p[c]).collect(); v.sort(); bg[c] = v[v.len() / 2]; }
    let d = |p: &Rgb<u8>| (0..3).map(|c| (p[c] as i32 - bg[c] as i32).abs()).sum::<i32>();
    let mut inner: Vec<(i32, Rgb<u8>)> = vec![];
    for yy in cl(y, h)..=cl(y + bh, h) { for xx in cl(x, w)..=cl(x + bw, w) { let p = *img.get_pixel(xx, yy); inner.push((d(&p), p)); } }
    inner.sort_by(|a, c| c.0.cmp(&a.0));
    let n = (inner.len() / 10).max(1);
    let mut fg = [0u8; 3];
    for c in 0..3 { fg[c] = (inner[..n].iter().map(|(_, p)| p[c] as u32).sum::<u32>() / n as u32) as u8; }
    (fg, bg)
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

async fn call(client: &reqwest::Client, prefix: &[serde_json::Value], texts: &[&str], base: &str) -> Result<Vec<String>, String> {
    let mut messages = prefix.to_vec();
    messages.push(serde_json::json!({"role": "user", "content": serde_json::to_string(texts).unwrap()}));
    let body = serde_json::json!({"model": "default", "temperature": 0.2, "messages": messages});
    let r = client.post(format!("{}/chat/completions", base.trim_end_matches('/'))).json(&body).send().await
        .map_err(|e| format!("Cannot reach {base}: {e}"))?;
    if !r.status().is_success() { return Err(format!("Endpoint returned HTTP {}", r.status())); }
    let j: serde_json::Value = r.json().await.map_err(|e| e.to_string())?;
    let c = j["choices"][0]["message"]["content"].as_str().ok_or("Response had no message content")?;
    extract_json_array(c)
}

/// Pulls the JSON array of strings out of a model reply. Reasoning models print `<think>...</think>` first, and chatty
/// models add a sentence or a ``` fence around the array, so the reply is cleaned up in steps: drop the reasoning,
/// strip fences, then (if it still does not parse) try every `[` up to the last `]` as the start of the array.
pub fn extract_json_array(raw: &str) -> Result<Vec<String>, String> {
    let s = match raw.rfind("</think>") { Some(p) => &raw[p + "</think>".len()..], None => raw };
    let s = s.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
    if let Ok(v) = serde_json::from_str::<Vec<String>>(s) { return Ok(v); }
    if let Some(end) = s.rfind(']') {
        for (start, _) in s.match_indices('[') {
            if start < end { if let Ok(v) = serde_json::from_str::<Vec<String>>(&s[start..=end]) { return Ok(v); } }
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
    let base = base.trim().trim_end_matches('/');
    if !(base.starts_with("http://") || base.starts_with("https://")) { return Err("URL must start with http:// or https://".into()); }
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
    let key = cache_key(blocks, target, base, &model_id(client, base).await, opts);
    let want: Vec<usize> = blocks.iter().enumerate()
        .filter(|(_, b)| !b.text.trim().is_empty() && only.map_or(true, |o| o.contains(&b.id))).map(|(i, _)| i).collect();
    let (mut idx, mut cached) = (vec![], 0usize);
    if opts.reuse_cache && only.is_none() {
        let c = cache.0.lock().unwrap();
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
        let mut c = cache.0.lock().unwrap();
        if c.len() >= CACHE_MAX_PAGES && !c.contains_key(&key) { c.clear(); }
        c.entry(key).or_default().extend(fresh);
    }
    Ok((cached, missing))
}

// ---------- Box editing helpers ----------
/// Re-estimates text and background colour for a box the user moved or resized.
pub fn sample_colors(img: &Path, bbox: [f32; 4]) -> Result<([u8; 3], [u8; 3]), String> {
    let im = image::open(img).map_err(|e| e.to_string())?.to_rgb8();
    Ok(box_colors(&im, bbox[0], bbox[1], bbox[2], bbox[3]))
}

/// Builds a block for a box the user drew: selectable PDF text under the box if there is any (unless mode is "ocr"),
/// otherwise OCR on just that area (unless mode is "text"). The text may come back empty; the user can type it in.
pub fn read_region(img: &Path, dir: &Path, cache: &EngineCache, bbox: [f32; 4], mode: &str) -> Result<Block, String> {
    let im = load_image(img).map_err(|e| e.to_string())?;
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
pub fn wrap(f: &FontArc, px: f32, text: &str, maxw: u32) -> Vec<String> {
    let mut out = vec![];
    for para in text.split('\n') {
        // Common case while fitting: the whole paragraph is one line, so skip the word-by-word measuring.
        let joined = para.split_whitespace().collect::<Vec<_>>().join(" ");
        if text_size(PxScale::from(px), f, &joined).0 <= maxw { out.push(joined); continue; }
        let mut line = String::new();
        for w in para.split_whitespace() {
            if text_size(PxScale::from(px), f, w).0 > maxw {
                // Word wider than the box: break it character by character.
                if !line.is_empty() { out.push(std::mem::take(&mut line)); }
                let mut piece = String::new();
                for ch in w.chars() {
                    let t = format!("{piece}{ch}");
                    if !piece.is_empty() && text_size(PxScale::from(px), f, &t).0 > maxw { out.push(std::mem::take(&mut piece)); }
                    piece.push(ch);
                }
                line = piece;
                continue;
            }
            let t = if line.is_empty() { w.to_string() } else { format!("{line} {w}") };
            if text_size(PxScale::from(px), f, &t).0 > maxw && !line.is_empty() { out.push(std::mem::take(&mut line)); line = w.to_string(); } else { line = t; }
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
    let mut im = image::open(img).map_err(|e| e.to_string())?.to_rgba8();
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
// Masks the text strokes, then fills them by diffusing the surrounding colours inward.
// Falls back to a flat background fill when the mask covers too much of the box.
// TODO(local AI): optional upgrade path is a LaMa ONNX model for photos and textured backgrounds.
/// Number of diffusion passes. The mask is only thin text strokes, so a pixel is never far from known colour; 20 passes
/// differ from 60 by at most a few grey levels (see `fewer_passes_look_the_same`) and are three times faster.
const RELAX_PASSES: usize = 20;

/// What erasing one block changes in the page: a flat fill, or the inpainted colour of the masked pixels.
enum Patch {
    Fill { x0: i32, y0: i32, w: u32, h: u32, rgb: [u8; 3] },
    Pixels(Vec<(u32, u32, [u8; 3])>),
}
impl Patch {
    fn apply(&self, im: &mut image::RgbaImage) {
        match self {
            Patch::Fill { x0, y0, w, h, rgb } => draw_filled_rect_mut(im, Rect::at(*x0, *y0).of_size(*w, *h), Rgba([rgb[0], rgb[1], rgb[2], 255])),
            Patch::Pixels(px) => for (a, c, v) in px { im.put_pixel(*a, *c, Rgba([v[0], v[1], v[2], 255])); },
        }
    }
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
    let mut m = vec![false; cw * ch];
    for j in 0..ch { for i in 0..cw {
        let (a, c) = px(i, j); let q = im.get_pixel(a, c);
        m[j * cw + i] = (0..3).map(|k| (q[k] as i32 - b.bg[k] as i32).abs()).sum::<i32>() > 60; // TODO(local AI): make adaptive for gradients
    } }
    let mut md = m.clone(); // dilate by 2 px to swallow anti-aliased edges
    for j in 0..ch { for i in 0..cw { if m[j * cw + i] {
        for dj in -2i32..=2 { for di in -2i32..=2 {
            let (a, c) = (i as i32 + di, j as i32 + dj);
            if a >= 0 && c >= 0 && (a as usize) < cw && (c as usize) < ch { md[c as usize * cw + a as usize] = true; }
        } }
    } } }
    let cover = md.iter().filter(|v| **v).count() as f32 / (cw * ch) as f32;
    if cover > 0.45 { return Some(Patch::Fill { x0, y0, w: cw as u32, h: ch as u32, rgb: b.bg }); }
    let mut buf: Vec<[f32; 3]> = (0..cw * ch).map(|k| {
        if md[k] { [b.bg[0] as f32, b.bg[1] as f32, b.bg[2] as f32] } else { let (a, c) = px(k % cw, k / cw); let q = im.get_pixel(a, c); [q[0] as f32, q[1] as f32, q[2] as f32] }
    }).collect();
    for _ in 0..passes { for j in 0..ch { for i in 0..cw { if md[j * cw + i] {
        let n = [(i.saturating_sub(1), j), ((i + 1).min(cw - 1), j), (i, j.saturating_sub(1)), (i, (j + 1).min(ch - 1))];
        for k in 0..3 { buf[j * cw + i][k] = n.iter().map(|(a, c)| buf[c * cw + a][k]).sum::<f32>() / 4.0; }
    } } } }
    Some(Patch::Pixels((0..cw * ch).filter(|&k| md[k]).map(|k| { let (a, c) = px(k % cw, k / cw); (a, c, [buf[k][0] as u8, buf[k][1] as u8, buf[k][2] as u8]) }).collect()))
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
    TextLayer { lines, has_images }
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
    let doc = pdfium.load_pdf_from_file(pdf, None).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let cfg = PdfRenderConfig::new().scale_page_by_factor(PDF_DPI / 72.0);
    let mut v = vec![];
    for (i, page) in doc.pages().iter().enumerate() {
        let bitmap = page.render_with_config(&cfg).map_err(|e| e.to_string())?;
        let img = bitmap.as_image();
        let p = out_dir.join(format!("page-{:03}.png", i + 1));
        img.save(&p).map_err(|e| e.to_string())?;
        let layer = page_text_layer(&page, PDF_DPI / 72.0);
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
pub fn write_pdf(pages: &[image::RgbaImage], dpi: f32, out: &Path) -> Result<(), String> {
    let (mut o, mut offs) = (b"%PDF-1.4\n".to_vec(), vec![]);
    let kids: Vec<String> = (0..pages.len()).map(|i| format!("{} 0 R", 3 + 3 * i)).collect();
    put(&mut o, &mut offs, 1, b"<</Type/Catalog/Pages 2 0 R>>");
    put(&mut o, &mut offs, 2, format!("<</Type/Pages/Count {}/Kids[{}]>>", pages.len(), kids.join(" ")).as_bytes());
    for (i, im) in pages.iter().enumerate() {
        let (pw, ph) = (im.width() as f32 * 72.0 / dpi, im.height() as f32 * 72.0 / dpi);
        let mut jpg = vec![];
        let rgb = DynamicImage::ImageRgba8(im.clone()).to_rgb8();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpg, 90).encode_image(&rgb).map_err(|e| e.to_string())?;
        let (pid, cid, iid) = (3 + 3 * i, 4 + 3 * i, 5 + 3 * i);
        put(&mut o, &mut offs, pid, format!("<</Type/Page/Parent 2 0 R/MediaBox[0 0 {pw:.2} {ph:.2}]/Resources<</XObject<</Im0 {iid} 0 R>>>>/Contents {cid} 0 R>>").as_bytes());
        let c = format!("q {pw:.2} 0 0 {ph:.2} 0 0 cm /Im0 Do Q");
        put(&mut o, &mut offs, cid, format!("<</Length {}>>\nstream\n{c}\nendstream", c.len()).as_bytes());
        let mut b = format!("<</Type/XObject/Subtype/Image/Width {}/Height {}/ColorSpace/DeviceRGB/BitsPerComponent 8/Filter/DCTDecode/Length {}>>\nstream\n", im.width(), im.height(), jpg.len()).into_bytes();
        b.extend(&jpg); b.extend(b"\nendstream");
        put(&mut o, &mut offs, iid, &b);
    }
    let xref = o.len();
    o.extend(format!("xref\n0 {}\n0000000000 65535 f \n", offs.len() + 1).as_bytes());
    for off in &offs { o.extend(format!("{off:010} 00000 n \n").as_bytes()); }
    o.extend(format!("trailer\n<</Size {}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF", offs.len() + 1).as_bytes());
    std::fs::write(out, o).map_err(|e| e.to_string())
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
}
