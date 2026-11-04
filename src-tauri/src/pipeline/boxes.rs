//! The boxes of a detected page: lines are grouped into blocks, blocks are cleaned up (shrunk to their text, nested ones merged), and a region the user draws is read.

use super::*;

// Fraction of box `a` that lies inside box `b`.
pub(super) fn overlap_frac(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = (a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0]);
    let h = (a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1]);
    if w <= 0.0 || h <= 0.0 || a[2] * a[3] <= 0.0 { 0.0 } else { w * h / (a[2] * a[3]) }
}

// Merge neighbouring lines into paragraph blocks. Lines only merge when they share
// column (overlap), font size (height ratio), colour, and a small vertical gap —
// so a differently-sized or differently-coloured line gets its own box.
#[cfg(test)]
pub(super) fn group(ls: Vec<Line>) -> Vec<Block> { group_tagged(ls).into_iter().map(|(b, _)| b).collect() }

/// `group`, and for each block whether one of its lines came from the PDF text layer.
pub(super) fn group_tagged(mut ls: Vec<Line>) -> Vec<(Block, bool)> {
    ls.sort_by(|a, b| a.b[1].total_cmp(&b.b[1]));
    let mut out: Vec<(Block, u32, f32, bool)> = vec![]; // block, line count, sum of line heights, has a text-layer line
    for l in ls {
        let [x, y, w, h] = l.b;
        let k = out.iter().rposition(|(b, n, s, _)| {
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
                let (b, n, s, lay) = &mut out[i];
                let [bx, by, bw, bh] = b.bbox;
                let (nx, ny) = (bx.min(x), by.min(y));
                b.bbox = [nx, ny, (bx + bw).max(x + w) - nx, (by + bh).max(y + h) - ny];
                b.text = format!("{} {}", b.text, l.t); b.conf = b.conf.min(l.c); *n += 1; *s += h; *lay |= l.layer;
            }
            None => out.push((Block { id: 0, bbox: l.b, text: l.t, translated: None, conf: l.c, font_px: 0.0, fg: l.fg, bg: l.bg }, 1, h, l.layer)),
        }
    }
    out.into_iter().map(|(mut b, n, s, lay)| { b.font_px = s / n as f32 * 0.85; (b, lay) }).collect()
}

/// A box is not shrunk below this share of its height: that would mean the text colour was wrong.
pub(super) const TIGHT_KEEP: f32 = 0.25;

/// A box whose text has fewer pixels than this is left alone.
pub(super) const TIGHT_MIN_PIXELS: u32 = 12;

/// How far a pixel may be from the text colour (sum of the three channel differences) and still count as text.
pub(super) const TIGHT_FG_TOL: i32 = 90;

/// A block is merged into another one when at least this share of its area lies inside it.
pub(super) const NESTED_FRAC: f32 = 0.9;

/// The detector draws its boxes taller than the text (and on posters much taller: background art gets inside). This
/// shrinks the box of a block, from the top and the bottom, to the rows of its own text. A text pixel has the text colour
/// `b.fg` (within `TIGHT_FG_TOL`) and differs clearly from the page colour behind it (`PageColours`, without the
/// text-coloured pixels). Areas of the text colour that run out of the box (the light margin of a poster beside a white
/// title) are page, not text, and are dropped: they are found by flooding the text-coloured pixels from the border of the
/// box (plus 3 px), where real letters do not reach. Rows with only a few text pixels (scratches in the background) do not
/// count. Only the rows are used: they do not depend on the colour of every letter (a line can have a second colour
/// that is not `b.fg`, and thin script strokes are hardly ever as pure as the text colour), so no text can be cut off
/// sideways, and text of another colour above or below does not hold the box open. The box never grows. If almost no text
/// is found, or the height would shrink to a quarter, the colour was probably wrong and the box is left alone.
/// The typeset text is as big as fits the box, so a box of the height of the text gives text of the original size.
pub(super) fn tighten(im: &RgbImage, b: &mut Block) {
    let [x, y, w, h] = b.bbox;
    let (iw, ih) = im.dimensions();
    if iw == 0 || ih == 0 || x >= iw as f32 || y >= ih as f32 { return; }
    let (x0, y0) = (x.max(0.0) as u32, y.max(0.0) as u32);
    let (x1, y1) = (((x + w) as u32).min(iw - 1), ((y + h) as u32).min(ih - 1));
    if x1 <= x0 + 1 || y1 <= y0 + 1 { return; }
    let (ex0, ey0, ex1, ey1) = (x0.saturating_sub(3), y0.saturating_sub(3), (x1 + 3).min(iw - 1), (y1 + 3).min(ih - 1)); // box plus 3 px
    let (ew, eh) = ((ex1 - ex0 + 1) as usize, (ey1 - ey0 + 1) as usize);
    let get = |a: u32, c: u32| { let p = im.get_pixel(a, c); [p[0], p[1], p[2]] };
    // Text-coloured pixels stay out of the page colour: lettering (and a margin of the same colour) must not become the "page".
    // Text that is hardly different from its page is then not found at all, and the box stays as it is.
    let page = PageColours::new(iw, ih, &get, (ex0, ey0, ex1, ey1), (y1 - y0) as f32, Some((b.fg, 60)));
    let like: Vec<bool> = (0..ew * eh).map(|k| color_dist(&get(ex0 + (k % ew) as u32, ey0 + (k / ew) as u32), &b.fg) <= TIGHT_FG_TOL).collect();
    let mut out = vec![false; ew * eh]; // text-coloured pixels connected to the border of the crop
    let mut todo: Vec<usize> = (0..ew * eh).filter(|k| (k % ew == 0 || k % ew == ew - 1 || k / ew == 0 || k / ew == eh - 1) && like[*k]).collect();
    for k in &todo { out[*k] = true; }
    while let Some(k) = todo.pop() {
        let (i, j) = (k % ew, k / ew);
        for (a, c) in [(i.wrapping_sub(1), j), (i + 1, j), (i, j.wrapping_sub(1)), (i, j + 1)] {
            if a < ew && c < eh && like[c * ew + a] && !out[c * ew + a] { out[c * ew + a] = true; todo.push(c * ew + a); }
        }
    }
    let (cw, ch) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
    let (mut rows, mut total) = (vec![0u32; ch], 0u32);
    for (j, row) in rows.iter_mut().enumerate() { for i in 0..cw {
        let (a, c) = (x0 + i as u32, y0 + j as u32);
        let k = (c - ey0) as usize * ew + (a - ex0) as usize;
        if like[k] && !out[k] && color_dist(&get(a, c), &page.at(a as i32, c as i32)) >= 30 { *row += 1; total += 1; }
    } }
    if total < TIGHT_MIN_PIXELS { return; }
    let need = ((cw as f32 * 0.02) as u32).max(2);
    let (Some(r0), Some(r1)) = (rows.iter().position(|c| *c >= need), rows.iter().rposition(|c| *c >= need)) else { return };
    let pad = 2.0 + 0.05 * (r1 - r0 + 1) as f32;
    let ny0 = (y0 as f32 + r0 as f32 - pad).max(y);
    let ny1 = (y0 as f32 + r1 as f32 + 1.0 + pad).min(y + h);
    if ny1 - ny0 < TIGHT_KEEP * h { return; }
    b.bbox = [x, ny0, w, ny1 - ny0];
}

/// Merges a block into another one when at least `NESTED_FRAC` of its area lies inside it: the bigger box is widened to
/// cover the smaller one completely and the smaller block is dropped. The bigger block keeps its text (it already holds
/// the text of the part inside it); only if that text is empty is the smaller block's text taken. This repeats until no
/// pair is left, so a chain of boxes inside boxes ends as one.
pub(super) fn merge_nested(mut bs: Vec<Block>) -> Vec<Block> {
    let area = |b: &Block| b.bbox[2] * b.bbox[3];
    loop {
        let mut pair = None; // (small, big)
        'search: for i in 0..bs.len() { for j in 0..bs.len() {
            // `i` is the smaller one; of two equal boxes the later one is dropped
            if i != j && (area(&bs[i]), j) <= (area(&bs[j]), i) && overlap_frac(&bs[i].bbox, &bs[j].bbox) >= NESTED_FRAC { pair = Some((i, j)); break 'search; }
        } }
        let Some((i, j)) = pair else { return bs };
        let small = bs.remove(i);
        let j = if j > i { j - 1 } else { j };
        let [sx, sy, sw, sh] = small.bbox;
        let [bx, by, bw, bh] = bs[j].bbox;
        let (nx, ny) = (bx.min(sx), by.min(sy));
        bs[j].bbox = [nx, ny, (bx + bw).max(sx + sw) - nx, (by + bh).max(sy + sh) - ny];
        if bs[j].text.trim().is_empty() { bs[j].text = small.text; }
    }
}

/// What a freshly detected page gets before the user sees it: boxes from OCR are shrunk to their text (`tighten`; boxes
/// built from the PDF text layer are exact already and are left as they are), then nested boxes are merged (`merge_nested`,
/// text-layer boxes included). The page is only detected once (and again on Reset), and boxes the user draws or edits do
/// not pass through here, so those stay exactly as they were set.
pub(super) fn refine(im: &RgbImage, blocks: Vec<(Block, bool)>) -> Vec<Block> {
    merge_nested(blocks.into_iter().map(|(mut b, from_layer)| { if !from_layer { tighten(im, &mut b); } b }).collect())
}

/// Re-estimates text and background colour for a box the user moved or resized.
pub fn sample_colors(img: &Path, bbox: [f32; 4]) -> Result<([u8; 3], [u8; 3]), String> {
    let im = load_oriented(img)?.to_rgb8();
    Ok(box_colors(&im, bbox[0], bbox[1], bbox[2], bbox[3]))
}

/// Builds a block for a box the user drew: selectable PDF text under the box if there is any (unless mode is "ocr"),
/// otherwise OCR on just that area (unless mode is "text"). The text may come back empty; the user can type it in.
pub fn read_region(img: &Path, dir: &Path, cache: &EngineCache, bbox: [f32; 4], mode: &str) -> Result<Block, String> {
    read_region_with(&OcrEngine { dir, cache }, img, bbox, mode)
}

fn read_region_with(det: &dyn TextDetector, img: &Path, bbox: [f32; 4], mode: &str) -> Result<Block, String> {
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
        for o in det.lines(&crop)? { conf = conf.min(o.c); found.push((o.b[1], o.b[0], o.t.clone(), o.b[3])); }
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let text = found.iter().map(|f| f.2.as_str()).collect::<Vec<_>>().join(" ");
    let font_px = if found.is_empty() { region[3] * 0.6 } else { found.iter().map(|f| f.3).sum::<f32>() / found.len() as f32 * 0.85 };
    let mut b = Block { id: 0, bbox: region, text, translated: None, conf: if found.is_empty() { 0.0 } else { conf }, font_px, fg: [0; 3], bg: [255; 3] };
    colors(&im, &mut b);
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::testutil::*;

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

    // ---- tighter boxes and nested boxes (detection only) ----
    fn text_block(id: u32, bbox: [f32; 4], text: &str) -> Block { Block { text: text.into(), ..blk(id, bbox, [0; 3]) } }

    #[test]
    fn a_loose_box_shrinks_to_the_rows_of_its_lettering_and_never_grows() {
        let mut page = striped_page([255, 255, 255]); // bars: x 80..294, y 60..140; grey margin left of x 50 and right of x 350
        for (x, y) in [(40, 40), (50, 160), (360, 100)] { page.put_pixel(x, y, Rgb([255, 255, 255])); } // single scratches
        let loose = [35.0f32, 35.0, 330.0, 130.0];
        let mut b = blk(1, loose, [0; 3]);
        (b.fg, b.bg) = box_colors(&page, loose[0], loose[1], loose[2], loose[3]);
        tighten(&page, &mut b);
        let [x, y, w, h] = b.bbox;
        assert!((52.0..=60.0).contains(&y) && y + h >= 140.0 && y + h <= 148.0, "y range {y}..{}", y + h);
        assert_eq!((x, w), (loose[0], loose[2]), "the width is not touched");
        assert!(y >= loose[1] && y + h <= loose[1] + loose[3], "the box grew");
    }

    #[test]
    fn a_line_with_a_second_colour_is_not_cut_off_sideways() {
        // "OCTOBER 3RD" in red, then "EXCELLA PARK" in gold, on a grey page; the block has one colour (the first line's)
        let mut page = RgbImage::from_pixel(400, 100, Rgb([237, 237, 237]));
        for k in 0..6 { draw_filled_rect_mut(&mut page, Rect::at(40 + 16 * k, 30).of_size(7, 30), Rgb([220, 0, 0])); }
        for k in 0..6 { draw_filled_rect_mut(&mut page, Rect::at(160 + 16 * k, 30).of_size(7, 30), Rgb([222, 181, 22])); }
        let loose = [20.0f32, 15.0, 340.0, 60.0];
        let mut b = blk(1, loose, [237; 3]);
        (b.fg, b.bg) = ([220, 0, 0], [237; 3]);
        tighten(&page, &mut b);
        assert!(b.bbox[0] + b.bbox[2] >= 255.0, "the gold part was cut off: box ends at {}", b.bbox[0] + b.bbox[2]);
        assert!(b.bbox[3] < loose[3] - 10.0, "the height should still shrink ({})", b.bbox[3]);
    }

    #[test]
    fn text_of_another_colour_above_does_not_hold_the_box_open() {
        let mut page = RgbImage::from_pixel(400, 140, Rgb([30, 40, 90]));
        for k in 0..12 { draw_filled_rect_mut(&mut page, Rect::at(60 + 18 * k, 20).of_size(7, 24), Rgb([240, 200, 20])); } // a neighbour, gold
        for k in 0..12 { draw_filled_rect_mut(&mut page, Rect::at(60 + 18 * k, 80).of_size(7, 24), Rgb([255, 255, 255])); } // this block, white
        let mut b = blk(1, [40.0, 10.0, 330.0, 120.0], [30, 40, 90]);
        (b.fg, b.bg) = ([255; 3], [30, 40, 90]);
        tighten(&page, &mut b);
        assert!(b.bbox[1] >= 70.0 && b.bbox[1] + b.bbox[3] <= 114.0, "rows {}..{}", b.bbox[1], b.bbox[1] + b.bbox[3]);
    }

    #[test]
    fn a_box_without_clear_text_is_left_alone() {
        let flat = RgbImage::from_pixel(200, 100, Rgb([200, 200, 200]));
        let mut b = blk(1, [20.0, 20.0, 150.0, 60.0], [200; 3]);
        b.fg = [0; 3];
        tighten(&flat, &mut b);
        assert_eq!(b.bbox, [20.0, 20.0, 150.0, 60.0]); // nothing stands out
        let mut specks = flat.clone();
        draw_filled_rect_mut(&mut specks, Rect::at(100, 50).of_size(3, 3), Rgb([0, 0, 0])); // 9 pixels: too few to be text
        tighten(&specks, &mut b);
        assert_eq!(b.bbox, [20.0, 20.0, 150.0, 60.0]);
        let mut off = blk(2, [150.0, 90.0, 200.0, 60.0], [200; 3]); // reaches past the page edge
        off.fg = [0; 3];
        tighten(&flat, &mut off); // no panic
    }

    #[test]
    fn only_boxes_from_ocr_are_tightened_but_all_boxes_can_merge() {
        let page = striped_page([255, 255, 255]);
        let loose = [35.0f32, 35.0, 330.0, 130.0];
        let mut b = blk(1, loose, [0; 3]);
        (b.fg, b.bg) = box_colors(&page, loose[0], loose[1], loose[2], loose[3]);
        let out = refine(&page, vec![(b.clone(), true)]);
        assert_eq!(out[0].bbox, loose, "a text-layer box must stay as it is");
        let out = refine(&page, vec![(b.clone(), false)]);
        assert!(out[0].bbox[3] < loose[3] - 20.0, "an OCR box must be tightened");
        // a text-layer box that holds an OCR box is still merged with it
        let big = Block { text: "layer".into(), ..b.clone() };
        let small = Block { id: 2, text: "ocr".into(), bbox: [100.0, 70.0, 60.0, 30.0], ..b };
        let out = refine(&page, vec![(big, true), (small, false)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "layer");
    }

    #[test]
    fn a_block_that_is_nearly_inside_another_is_merged_into_it() {
        let big = text_block(1, [0.0, 0.0, 100.0, 50.0], "whole paragraph");
        // 95% of the small box is inside: the big box is widened by the missing half pixel
        let out = merge_nested(vec![big.clone(), text_block(2, [90.5, 10.0, 10.0, 10.0], "paragraph")]);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].id, out[0].text.as_str(), out[0].bbox), (1, "whole paragraph", [0.0, 0.0, 100.5, 50.0]));
        // the order in the list does not matter
        let out = merge_nested(vec![text_block(2, [90.5, 10.0, 10.0, 10.0], "paragraph"), big.clone()]);
        assert_eq!((out.len(), out[0].id), (1, 1));
        // half inside: two blocks stay
        assert_eq!(merge_nested(vec![big.clone(), text_block(2, [95.0, 10.0, 10.0, 10.0], "x")]).len(), 2);
        // inside, but only 80%
        assert_eq!(merge_nested(vec![big.clone(), text_block(2, [92.0, 10.0, 10.0, 10.0], "x")]).len(), 2);
    }

    #[test]
    fn nested_merging_follows_chains_and_handles_duplicates_and_empty_text() {
        let a = text_block(1, [0.0, 0.0, 200.0, 100.0], "c");
        let b = text_block(2, [10.0, 10.0, 100.0, 50.0], "b");
        let c = text_block(3, [20.0, 20.0, 30.0, 20.0], "a");
        let out = merge_nested(vec![c, b, a]);
        assert_eq!((out.len(), out[0].id, out[0].bbox), (1, 1, [0.0, 0.0, 200.0, 100.0]));
        let twins = merge_nested(vec![text_block(1, [5.0, 5.0, 40.0, 20.0], "t"), text_block(2, [5.0, 5.0, 40.0, 20.0], "t")]);
        assert_eq!((twins.len(), twins[0].id), (1, 1)); // one of two identical boxes goes, the first stays
        let empty = merge_nested(vec![text_block(1, [0.0, 0.0, 100.0, 50.0], " "), text_block(2, [10.0, 10.0, 20.0, 10.0], "inner")]);
        assert_eq!((empty.len(), empty[0].text.as_str()), (1, "inner")); // an empty big block takes the text of the small one
        assert_eq!(merge_nested(vec![]).len(), 0);
        assert_eq!(merge_nested(vec![text_block(1, [0.0, 0.0, 10.0, 10.0], "a"), text_block(2, [50.0, 50.0, 10.0, 10.0], "b")]).len(), 2);
    }

    #[test]
    fn a_grouped_block_knows_if_it_holds_a_text_layer_line() {
        let mut l2 = line(10.0, 32.0, 200.0, 20.0, "two", [0; 3]);
        l2.layer = true;
        let g = group_tagged(vec![line(10.0, 10.0, 200.0, 20.0, "one", [0; 3]), l2, line(10.0, 300.0, 200.0, 20.0, "far", [0; 3])]);
        assert_eq!(g.len(), 2);
        assert_eq!((g[0].0.text.as_str(), g[0].1), ("one two", true)); // the two lines merged, one is from the layer
        assert_eq!((g[1].0.text.as_str(), g[1].1), ("far", false));
        assert!(!group_tagged(vec![line(10.0, 10.0, 200.0, 20.0, "one", [0; 3])])[0].1);
    }

    #[test]
    fn overlap_frac_basics() {
        let a = [10.0, 10.0, 20.0, 10.0];
        assert_eq!(overlap_frac(&a, &[0.0, 0.0, 100.0, 100.0]), 1.0); // fully inside
        assert_eq!(overlap_frac(&a, &[200.0, 0.0, 10.0, 10.0]), 0.0); // apart
        assert!((overlap_frac(&a, &[20.0, 10.0, 50.0, 10.0]) - 0.5).abs() < 1e-6); // half covered
    }

    #[test]
    fn a_region_is_read_through_the_detector_and_its_lines_come_in_reading_order() {
        let (dir, file) = page_file("region-read", &RgbImage::from_pixel(300, 200, Rgb([255, 255, 255])));
        let mut low = line(20.0, 60.0, 80.0, 20.0, "second", [0; 3]);
        low.c = 0.6;
        let det = FakeDetector::with(vec![low, line(20.0, 10.0, 80.0, 20.0, "first", [0; 3])]); // not in reading order
        let b = read_region_with(&det, &file, [10.0, 10.0, 200.0, 100.0], "ocr").unwrap();
        assert_eq!(b.text, "first second");
        assert_eq!(b.bbox, [10.0, 10.0, 200.0, 100.0]); // the box is the region the user drew
        assert!((b.conf - 0.6).abs() < 1e-6, "the confidence is the lowest of the lines: {}", b.conf);
        assert_eq!(det.calls.get(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_region_with_no_text_gives_an_empty_box_and_text_mode_never_asks_the_detector() {
        let (dir, file) = page_file("region-empty", &RgbImage::from_pixel(300, 200, Rgb([255, 255, 255])));
        let det = FakeDetector::with(vec![]);
        let b = read_region_with(&det, &file, [10.0, 10.0, 100.0, 50.0], "ocr").unwrap();
        assert_eq!((b.text.as_str(), b.conf), ("", 0.0));
        let det = FakeDetector::with(vec![line(0.0, 0.0, 50.0, 20.0, "x", [0; 3])]);
        let b = read_region_with(&det, &file, [10.0, 10.0, 100.0, 50.0], "text").unwrap();
        assert_eq!((b.text.as_str(), det.calls.get()), ("", 0));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
