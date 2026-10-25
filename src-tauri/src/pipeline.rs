use ab_glyph::{FontArc, PxScale};
use base64::Engine;
use image::{DynamicImage, Rgb, RgbImage, Rgba};
use imageproc::{drawing::{draw_filled_rect_mut, draw_text_mut, text_size}, rect::Rect};
use oar_ocr::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub const PDF_DPI: f32 = 200.0;
// TODO(local AI): confirm the exact engine type name exported by oar-ocr (assumed OAROCR).
pub type Engine = OAROCR;
/// Caches built OCR engines per model folder so only the first run pays the load cost.
#[derive(Default)]
pub struct EngineCache(Mutex<HashMap<String, Arc<Engine>>>);

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Block {
    pub id: u32, pub bbox: [f32; 4], pub text: String, pub translated: Option<String>,
    pub conf: f32, pub font_px: f32, pub fg: [u8; 3], pub bg: [u8; 3],
}

// ---------- OCR (PaddleOCR via oar-ocr) ----------
struct Line { b: [f32; 4], t: String, c: f32 }

pub fn ocr(img: &Path, dir: &Path, cache: &EngineCache) -> Result<Vec<Block>, String> {
    let p = |n: &str| dir.join(n).to_string_lossy().to_string();
    let engine = {
        let mut m = cache.0.lock().unwrap();
        match m.get(&p("")) {
            Some(e) => e.clone(),
            None => {
                let e = Arc::new(OAROCRBuilder::new(p("det.onnx"), p("rec.onnx"), p("dict.txt")).build().map_err(|e| e.to_string())?);
                m.insert(p(""), e.clone());
                e
            }
        }
    };
    let im = load_image(img).map_err(|e| e.to_string())?;
    let res = engine.predict(vec![im]).map_err(|e| e.to_string())?;
    let mut lines = vec![];
    for r in &res[0].text_regions {
        if let Some((t, c)) = r.text_with_confidence() {
            // ADJUST HERE if the compiler disagrees: this assumes a polygon with `.points` having `.x`/`.y`.
            let pts = &r.bounding_box.points;
            let (x0, x1) = pts.iter().fold((f32::MAX, f32::MIN), |a, q| (a.0.min(q.x), a.1.max(q.x)));
            let (y0, y1) = pts.iter().fold((f32::MAX, f32::MIN), |a, q| (a.0.min(q.y), a.1.max(q.y)));
            lines.push(Line { b: [x0, y0, x1 - x0, y1 - y0], t: t.to_string(), c: c as f32 });
        }
    }
    Ok(group(lines))
}

// Merge neighbouring lines (same column, similar height, small gap) into paragraph blocks.
fn group(mut ls: Vec<Line>) -> Vec<Block> {
    ls.sort_by(|a, b| a.b[1].total_cmp(&b.b[1]));
    let mut out: Vec<(Block, u32, f32)> = vec![]; // block, line count, sum of line heights
    for l in ls {
        let [x, y, w, h] = l.b;
        let k = out.iter().rposition(|(b, n, s)| {
            let [bx, by, bw, bh] = b.bbox;
            let avg = s / *n as f32;
            x < bx + bw && x + w > bx && (y - (by + bh)) > -h * 0.3 && (y - (by + bh)) < h * 0.8 && h / avg > 0.7 && h / avg < 1.4
        });
        match k {
            Some(i) => {
                let (b, n, s) = &mut out[i];
                let [bx, by, bw, bh] = b.bbox;
                let (nx, ny) = (bx.min(x), by.min(y));
                b.bbox = [nx, ny, (bx + bw).max(x + w) - nx, (by + bh).max(y + h) - ny];
                b.text = format!("{} {}", b.text, l.t); b.conf = b.conf.min(l.c); *n += 1; *s += h;
            }
            None => out.push((Block { id: 0, bbox: l.b, text: l.t, translated: None, conf: l.c, font_px: 0.0, fg: [0; 3], bg: [255; 3] }, 1, h)),
        }
    }
    out.into_iter().map(|(mut b, n, s)| { b.font_px = s / n as f32 * 0.85; b }).collect()
}

// Estimate background (median of a ring around the box) and text colour (mean of the most contrasting pixels).
pub fn colors(img: &RgbImage, b: &mut Block) {
    let (w, h) = img.dimensions();
    let [x, y, bw, bh] = b.bbox;
    let cl = |v: f32, m: u32| (v.max(0.) as u32).min(m - 1);
    let (x0, y0, x1, y1) = (cl(x - 4., w), cl(y - 4., h), cl(x + bw + 4., w), cl(y + bh + 4., h));
    let mut ring: Vec<Rgb<u8>> = vec![];
    for xx in x0..=x1 { ring.push(*img.get_pixel(xx, y0)); ring.push(*img.get_pixel(xx, y1)); }
    for yy in y0..=y1 { ring.push(*img.get_pixel(x0, yy)); ring.push(*img.get_pixel(x1, yy)); }
    for c in 0..3 { let mut v: Vec<u8> = ring.iter().map(|p| p[c]).collect(); v.sort(); b.bg[c] = v[v.len() / 2]; }
    let d = |p: &Rgb<u8>| (0..3).map(|c| (p[c] as i32 - b.bg[c] as i32).abs()).sum::<i32>();
    let mut inner: Vec<(i32, Rgb<u8>)> = vec![];
    for yy in cl(y, h)..=cl(y + bh, h) { for xx in cl(x, w)..=cl(x + bw, w) { let p = *img.get_pixel(xx, yy); inner.push((d(&p), p)); } }
    inner.sort_by(|a, c| c.0.cmp(&a.0));
    let n = (inner.len() / 10).max(1);
    for c in 0..3 { b.fg[c] = (inner[..n].iter().map(|(_, p)| p[c] as u32).sum::<u32>() / n as u32) as u8; }
}

// ---------- Translation: any OpenAI-compatible endpoint (llama.cpp's llama-server, or a hosted API with a key) ----------
async fn call(texts: &[&str], target: &str, base: &str, model: &str, key: &str) -> Result<Vec<String>, String> {
    let body = serde_json::json!({
        "model": if model.is_empty() { "default" } else { model }, "temperature": 0.2,
        "messages": [
            {"role": "system", "content": format!("You are a translation engine. Translate every string in the JSON array into {target}. Return ONLY a JSON array of strings with the same length and order. Keep numbers, names and line breaks.")},
            {"role": "user", "content": serde_json::to_string(texts).unwrap()}]});
    let mut rq = reqwest::Client::new().post(format!("{}/chat/completions", base.trim_end_matches('/'))).json(&body);
    if !key.is_empty() { rq = rq.bearer_auth(key); }
    let r = rq.send().await.map_err(|e| format!("Cannot reach {base}: {e}"))?;
    if !r.status().is_success() { return Err(format!("Endpoint returned HTTP {}", r.status())); }
    let j: serde_json::Value = r.json().await.map_err(|e| e.to_string())?;
    let c = j["choices"][0]["message"]["content"].as_str().ok_or("Response had no message content")?;
    let c = c.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
    serde_json::from_str::<Vec<String>>(c).map_err(|e| format!("Model did not return a JSON array: {e}"))
}

pub async fn translate(blocks: &mut Vec<Block>, target: &str, base: &str, model: &str, key: &str) -> Result<(), String> {
    let texts: Vec<&str> = blocks.iter().map(|b| b.text.as_str()).collect();
    let out = match call(&texts, target, base, model, key).await {
        Ok(v) if v.len() == texts.len() => v,
        _ => { // batch failed or lost lines: retry one block at a time
            let mut v = vec![];
            for t in &texts { v.push(call(&[t], target, base, model, key).await?.into_iter().next().unwrap_or_default()); }
            v
        }
    };
    for (b, t) in blocks.iter_mut().zip(out) { b.translated = Some(t); }
    Ok(())
}

// ---------- Erase + typeset ----------
fn wrap(f: &FontArc, px: f32, text: &str, maxw: u32) -> Vec<String> {
    let mut out = vec![];
    for para in text.split('\n') {
        let mut line = String::new();
        for w in para.split_whitespace() {
            let t = if line.is_empty() { w.to_string() } else { format!("{line} {w}") };
            if text_size(PxScale::from(px), f, &t).0 > maxw && !line.is_empty() { out.push(std::mem::take(&mut line)); line = w.to_string(); } else { line = t; }
        }
        out.push(line);
    }
    out
}

pub fn render(img: &Path, font_path: &Path, blocks: &[Block]) -> Result<image::RgbaImage, String> {
    let mut im = image::open(img).map_err(|e| e.to_string())?.to_rgba8();
    let font = FontArc::try_from_vec(std::fs::read(font_path).map_err(|e| format!("Font missing ({}): {e}", font_path.display()))?).map_err(|e| e.to_string())?;
    for b in blocks {
        let [x, y, w, h] = b.bbox;
        erase(&mut im, b);
        let t = b.translated.as_deref().unwrap_or(&b.text);
        let mut px = b.font_px * 1.05;
        let mut lines = wrap(&font, px, t, w as u32);
        while px > 7.0 && lines.len() as f32 * px * 1.2 > h + px * 0.3 { px -= 0.5; lines = wrap(&font, px, t, w as u32); }
        for (i, l) in lines.iter().enumerate() {
            draw_text_mut(&mut im, Rgba([b.fg[0], b.fg[1], b.fg[2], 255]), x as i32, (y + i as f32 * px * 1.2) as i32, PxScale::from(px), &font, l);
        }
    }
    Ok(im)
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
fn erase(im: &mut image::RgbaImage, b: &Block) {
    let (iw, ih) = im.dimensions();
    let [x, y, w, h] = b.bbox;
    let p = 3i32;
    let x0 = (x as i32 - p).max(0); let y0 = (y as i32 - p).max(0);
    let x1 = ((x + w) as i32 + p).min(iw as i32 - 1); let y1 = ((y + h) as i32 + p).min(ih as i32 - 1);
    if x1 <= x0 || y1 <= y0 { return; }
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
    if cover > 0.45 {
        draw_filled_rect_mut(im, Rect::at(x0, y0).of_size(cw as u32, ch as u32), Rgba([b.bg[0], b.bg[1], b.bg[2], 255]));
        return;
    }
    let mut buf: Vec<[f32; 3]> = (0..cw * ch).map(|k| {
        if md[k] { [b.bg[0] as f32, b.bg[1] as f32, b.bg[2] as f32] } else { let (a, c) = px(k % cw, k / cw); let q = im.get_pixel(a, c); [q[0] as f32, q[1] as f32, q[2] as f32] }
    }).collect();
    for _ in 0..60 { for j in 0..ch { for i in 0..cw { if md[j * cw + i] {
        let n = [(i.saturating_sub(1), j), ((i + 1).min(cw - 1), j), (i, j.saturating_sub(1)), (i, (j + 1).min(ch - 1))];
        for k in 0..3 { buf[j * cw + i][k] = n.iter().map(|(a, c)| buf[c * cw + a][k]).sum::<f32>() / 4.0; }
    } } } }
    for k in 0..cw * ch { if md[k] { let (a, c) = px(k % cw, k / cw); im.put_pixel(a, c, Rgba([buf[k][0] as u8, buf[k][1] as u8, buf[k][2] as u8, 255])); } }
}

// ---------- PDF in (pdfium) ----------
pub fn rasterise_pdf(pdf: &Path, lib_dir: &Path, out_dir: &Path) -> Result<Vec<String>, String> {
    use pdfium_render::prelude::*;
    // TODO(local AI): verify these pdfium-render calls against the installed version (bind fn names, as_image() return type).
    let lib = Pdfium::pdfium_platform_library_name_at_path(&lib_dir.to_string_lossy().to_string());
    let pdfium = Pdfium::new(Pdfium::bind_to_library(lib).map_err(|e| format!("pdfium library not found in {}: {e}", lib_dir.display()))?);
    let doc = pdfium.load_pdf_from_file(pdf, None).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let cfg = PdfRenderConfig::new().scale_page_by_factor(PDF_DPI / 72.0);
    let mut v = vec![];
    for (i, page) in doc.pages().iter().enumerate() {
        let img = page.render_with_config(&cfg).map_err(|e| e.to_string())?.as_image();
        let p = out_dir.join(format!("page-{:03}.png", i + 1));
        img.save(&p).map_err(|e| e.to_string())?;
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
