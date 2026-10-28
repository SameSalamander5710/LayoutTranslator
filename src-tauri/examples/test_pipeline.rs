// Test harness for the pipeline: ocr, translate, render, geom modes.
use layout_translator::pipeline::{self, Block};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() { usage(); return; }
    match args[0].as_str() {
        "ocr" => ocr(&args[1..]),
        "translate" => translate(&args[1..]),
        "render" => render(&args[1..]),
        "geom" => geom(&args[1..]),
        _ => usage(),
    }
}

fn usage() {
    eprintln!("modes:");
    eprintln!("  ocr <image> <model_dir> <out_prefix>");
    eprintln!("  translate <blocks.json> <target> <endpoint> <out_blocks.json>");
    eprintln!("  render <image> <blocks.json> <font> <out.png>");
    eprintln!("  geom <font>");
    std::process::exit(2);
}

fn ocr(a: &[String]) {
    if a.len() < 3 { usage(); }
    let img = Path::new(&a[0]);
    let dir = Path::new(&a[1]);
    let cache = pipeline::EngineCache::default();
    let mut blocks = pipeline::ocr(img, dir, &cache).expect("ocr failed");
    let rgb = image::open(img).expect("open image").to_rgb8();
    for (i, b) in blocks.iter_mut().enumerate() {
        b.id = i as u32 + 1;
        pipeline::colors(&rgb, b);
    }
    println!("{} blocks:", blocks.len());
    for b in &blocks {
        println!(
            "#{} bbox=[{:.0},{:.0},{:.0},{:.0}] font_px={:.1} fg={:?} bg={:?} conf={:.2} {:?}",
            b.id, b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3], b.font_px, b.fg, b.bg, b.conf, b.text
        );
    }
    let json = format!("{}_blocks.json", a[2]);
    std::fs::write(&json, serde_json::to_string_pretty(&blocks).unwrap()).unwrap();
    println!("blocks saved to {json}");
    let mut im = image::open(img).expect("open image").to_rgba8();
    for b in &blocks {
        let [x, y, w, h] = b.bbox;
        imageproc::drawing::draw_hollow_rect_mut(
            &mut im,
            imageproc::rect::Rect::at(x as i32, y as i32).of_size(w as u32, h as u32),
            image::Rgba([0, 255, 0, 255]),
        );
    }
    let png = format!("{}_boxes.png", a[2]);
    im.save(&png).unwrap();
    println!("boxes saved to {png}");
}

fn translate(a: &[String]) {
    if a.len() < 4 { usage(); }
    let blocks: Vec<Block> = serde_json::from_str(&std::fs::read_to_string(&a[0]).unwrap()).unwrap();
    let mut blocks = blocks;
    let cache = pipeline::TranslationCache::default();
    let (cached, missing) = tauri::async_runtime::block_on(pipeline::translate(
        &mut blocks,
        &a[1], &a[2], &pipeline::TranslateOptions::default(), None, &cache,
    ))
    .expect("translate failed");
    println!("{cached} from cache, {missing} without translation");
    std::fs::write(&a[3], serde_json::to_string_pretty(&blocks).unwrap()).unwrap();
    for b in &blocks {
        println!("#{} {:?} => {:?}", b.id, b.text, b.translated);
    }
    println!("saved {}", a[3]);
}

fn render(a: &[String]) {
    if a.len() < 4 { usage(); }
    let blocks: Vec<Block> = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let font = ab_glyph::FontArc::try_from_vec(std::fs::read(&a[2]).unwrap()).unwrap();
    // Per-block fit diagnostics: does the typeset text stay inside the box?
    for b in &blocks {
        let [x, y, w, h] = b.bbox;
        let t = b.translated.as_deref().unwrap_or(&b.text);
        if t.trim().is_empty() || w < 4.0 || h < 4.0 { continue; }
        let px = layout_translator::pipeline::fit_px(&font, t, w as u32, h as u32);
        let lines = layout_translator::pipeline::wrap(&font, px, t, w as u32);
        let ls = layout_translator::pipeline::line_spacing(&font, px);
        let (mut top, mut bot) = (f32::MAX, f32::MIN);
        for (i, l) in lines.iter().enumerate() {
            if l.is_empty() { continue; }
            let (t2, b2, _) = layout_translator::pipeline::text_geom(&font, px, l);
            top = top.min(i as f32 * ls + t2);
            bot = bot.max(i as f32 * ls + b2);
        }
        let ok = top != f32::MAX && bot - top <= h && lines.iter().all(|l| layout_translator::pipeline::text_geom(&font, px, l).2 <= w as f32);
        println!("#{} px={px:.1} lines={} ink_h={:.1}/box_h={:.0} x={:.0} y={:.0} fit={ok} {:?}", b.id, lines.len(), bot - top, h, x, y, t);
    }
    let im = pipeline::render(Path::new(&a[0]), Path::new(&a[2]), &blocks).expect("render failed");
    im.save(&a[3]).unwrap();
    println!("saved {}", a[3]);
}

fn geom(a: &[String]) {
    if a.len() < 1 { usage(); }
    let font = ab_glyph::FontArc::try_from_vec(std::fs::read(&a[0]).unwrap()).unwrap();
    for s in ["Ag", "A", "g", "ß", "Tag der Deutschen Einheit"] {
        let (t, b, w) = pipeline::text_geom(&font, 100.0, s);
        println!("{s:?}: top={t:.2} bot={b:.2} width={w:.2} (px=100)");
    }
    // Empirical ground truth: draw "Ag" at (50, 100) on a blank canvas, scan ink rows.
    let mut im = image::RgbaImage::new(400, 400);
    imageproc::drawing::draw_text_mut(&mut im, image::Rgba([0, 0, 0, 255]), 50, 100, ab_glyph::PxScale::from(100.0), &font, "Ag");
    let (mut top, mut bot) = (400i32, -1i32);
    for y in 0..400 {
        for x in 0..400 {
            if im.get_pixel(x, y)[3] > 0 { top = top.min(y as i32); bot = bot.max(y as i32); }
        }
    }
    println!("empirical 'Ag' drawn at y=100: ink top={top} bot={bot} (offsets {:.2} / {:.2})", top - 100, bot - 100);
}
