//! Typesetting the translation into the boxes: line breaking, text size that fits, and the page render (erase, then draw).

use super::*;

/// Cuts one word (no whitespace inside) at the places where a line may end, following the Unicode line breaking rules
/// (UAX #14): Japanese and Chinese text can break between characters but never before closing punctuation like `、` or `。`,
/// and Latin letters, digits and the like stay together. Thai has no break rules without a dictionary, so it comes back whole.
pub(super) fn break_segments(word: &str) -> Vec<&str> {
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
pub(super) type GlyphCache = HashMap<char, (ab_glyph::GlyphId, f32, Option<(f32, f32)>)>;

// Ink extent of a line relative to the draw origin y, replicating imageproc's layout
// (baseline at y + ascent). Returns (ink top, ink bottom, advance width).
pub fn text_geom(f: &FontArc, px: f32, text: &str) -> (f32, f32, f32) { text_geom_cached(f, px, text, &mut GlyphCache::new()) }

/// Same as `text_geom`, but looks glyphs up in `cache` (which must only ever be used with this `px`). Building outlines is
/// the expensive part, and `fits` measures the same few characters over and over.
pub(super) fn text_geom_cached(f: &FontArc, px: f32, text: &str, cache: &mut GlyphCache) -> (f32, f32, f32) {
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
    f.height_unscaled() / upem * px
}

pub(super) fn fits(f: &FontArc, text: &str, w: u32, h: u32, px: f32) -> bool {
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
    // Pass 1: erase every original text area. Blocks that overlap are erased together as one group (see `erase_groups`), so
    // the blurred fill has no seam where they meet. All patches are computed from the untouched page (in parallel, one group
    // per task) and applied afterwards, so an erase can never wipe translated text of an overlapping block.
    let patches = erase_patches(&im, blocks);
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
pub(super) struct Placed { x: i32, y0: f32, ls: f32, px: f32, lines: Vec<String>, fg: [u8; 3] }

/// The font that is used when the language needs none of the others, and when a font file is missing.
const LATIN_FONT: &str = "NotoSans-Regular.ttf";

/// The font file for the language the translation is written in, taken from the "target language" field: free text such as
/// "Japanese", "ja" or "Chinese (Traditional)", in any case. This is the only place that knows which font belongs to which
/// language. Scripts that need no special font get the Latin one.
pub fn font_file_name(target: &str) -> &'static str {
    let t = target.trim().to_lowercase();
    match t.as_str() {
        "ja" | "jpn" | "japanese" => "NotoSansJP-Regular.ttf",
        "ko" | "kor" | "korean" => "NotoSansKR-Regular.ttf",
        "zh" | "zh-cn" | "zh-tw" | "zho" | "chi" => "NotoSansSC-Regular.ttf",
        _ if t.starts_with("chinese") => "NotoSansSC-Regular.ttf", // "Chinese (Simplified)", "Chinese (Traditional)", ...
        _ => LATIN_FONT,
    }
}

/// The path of that font in `fonts_dir`. If the file is not there (a trimmed install), the Latin font is used, so the page
/// still renders (the letters the Latin font lacks come out as missing-glyph boxes, which is better than an error).
pub fn resolve_font(fonts_dir: &Path, target: &str) -> PathBuf {
    let font = fonts_dir.join(font_file_name(target));
    if font.exists() { font } else { fonts_dir.join(LATIN_FONT) }
}

/// Renders a page for the UI: returns the PNG data URL. With `out` the PNG is also written to that file.
pub fn render_preview(img: &Path, font: &Path, blocks: &[Block], out: Option<&Path>) -> Result<String, String> {
    let im = render(img, font, blocks)?;
    if let Some(o) = out { save_png(&im, o)?; }
    png_data_url(&DynamicImage::ImageRgba8(im))
}

pub(super) fn layout(font: &FontArc, b: &Block) -> Option<Placed> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::testutil::*;

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

    #[test]
    fn the_font_follows_the_language_of_the_translation() {
        for (target, font) in [
            ("Japanese", "NotoSansJP-Regular.ttf"), ("ja", "NotoSansJP-Regular.ttf"), ("  JAPANESE ", "NotoSansJP-Regular.ttf"),
            ("Korean", "NotoSansKR-Regular.ttf"), ("ko", "NotoSansKR-Regular.ttf"),
            ("Chinese (Simplified)", "NotoSansSC-Regular.ttf"), ("Chinese (Traditional)", "NotoSansSC-Regular.ttf"),
            ("chinese", "NotoSansSC-Regular.ttf"), ("zh-TW", "NotoSansSC-Regular.ttf"),
            ("English", "NotoSans-Regular.ttf"), ("German", "NotoSans-Regular.ttf"), ("", "NotoSans-Regular.ttf"), ("Klingon", "NotoSans-Regular.ttf"),
        ] { assert_eq!(font_file_name(target), font, "for the target language {target:?}"); }
    }

    #[test]
    fn a_missing_font_file_falls_back_to_the_latin_font() {
        let dir = temp_dir("fonts");
        std::fs::write(dir.join("NotoSans-Regular.ttf"), b"x").unwrap();
        assert_eq!(resolve_font(&dir, "Japanese"), dir.join("NotoSans-Regular.ttf")); // no Japanese font installed
        std::fs::write(dir.join("NotoSansJP-Regular.ttf"), b"x").unwrap();
        assert_eq!(resolve_font(&dir, "Japanese"), dir.join("NotoSansJP-Regular.ttf"));
        assert_eq!(resolve_font(&dir, "French"), dir.join("NotoSans-Regular.ttf"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
