//! Erasing the original text (classical, no model): stroke fill, blurred fill for busy backgrounds, and the grouping of overlapping blocks.

use super::*;

// Masks the text strokes (relative to a background colour that follows gradients), then fills them by diffusing the
// surrounding colours inward. Falls back to painting the smooth background when the mask covers too much of the box.
// TODO(local AI): optional upgrade path is a LaMa ONNX model for photos and textured backgrounds.
/// Number of diffusion passes. The mask is only thin text strokes, so a pixel is never far from known colour; 20 passes
/// differ from 60 by at most a few grey levels (see `fewer_passes_look_the_same`) and are three times faster.
pub(super) const RELAX_PASSES: usize = 20;

/// When more than this share of a block's crop differs clearly from the page colour behind it, the stroke mask cannot be
/// trusted (a noisy photo, a pattern). The crop is then rebuilt as a blurred fill from the pixels around it (`frosted_fill`).
/// Plain text covers 20 to 35% of its box, big bold lettering up to about 50%.
pub(super) const FROST_COVER: f32 = 0.6;

/// When the text mask (after growing it by 2 px) covers more than this share of a block's crop, the diffusion fill cannot
/// reach the middle of the holes in `RELAX_PASSES` steps. The fill is then done by `push_pull` over the real pixels around
/// the strokes (inside the box and outside it). This fill is not blurred, so the page keeps its detail between the letters.
pub(super) const DENSE_COVER: f32 = 0.5;

/// How fast `push_pull` trusts a coarse cell that is only partly known. 1.0 is the textbook value; higher keeps colours
/// from two different backgrounds apart for longer instead of averaging them to grey.
pub(super) const PULL_TRUST: f32 = 4.0;

/// What erasing one block changes in the page: the new colour of some pixels.
pub(super) struct Patch(Vec<(u32, u32, [u8; 3])>);

impl Patch {
    pub(super) fn apply(&self, im: &mut image::RgbaImage) {
        for (a, c, v) in &self.0 { im.put_pixel(*a, *c, Rgba([v[0], v[1], v[2], 255])); }
    }
}

/// The colour of the page behind the text at every pixel of the crop, as a tilted plane `a + b*x + c*y` per colour channel,
/// fitted by least squares to the pixels that are not text. A pixel counts as "not text" when it is close to the current
/// fit; the fit is repeated three times with a tighter limit (150, 80, 40) and starts from the block's background colour.
/// Text, and the text of a neighbouring box, is far from the fit, so it does not pull the plane. On a flat background the
/// result is that one colour; on a linear gradient it follows the gradient.
pub(super) fn background_plane(im: &image::RgbaImage, x0: u32, y0: u32, cw: usize, ch: usize, bg: [u8; 3]) -> Vec<[f32; 3]> {
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

/// How far past its box the erase of a block may reach. Two blocks whose reach overlaps are erased together.
pub(super) const ERASE_PAD: i32 = 3;

/// Works out how to erase one block, reading the page but not changing it, so many blocks can be done at the same time.
#[cfg(test)]
pub(super) fn erase_patch(im: &image::RgbaImage, b: &Block) -> Option<Patch> { erase_patch_with(im, b, RELAX_PASSES) }

#[cfg(test)]
pub(super) fn erase_patch_with(im: &image::RgbaImage, b: &Block, passes: usize) -> Option<Patch> { erase_group_with(im, &[b], &[], passes) }

/// What the erase has learnt about one block: its crop of the page, which pixels of the crop look like text, and whether
/// the crop is too busy for stroke inpainting (`frosty`: it gets the blurred fill instead).
pub(super) struct Crop { x0: i32, y0: i32, cw: usize, ch: usize, bgs: Vec<[f32; 3]>, md: Vec<bool>, frosty: bool, dense: bool, bg: [u8; 3] }

pub(super) fn analyse_crop(im: &image::RgbaImage, b: &Block) -> Option<Crop> {
    let (iw, ih) = im.dimensions();
    let [x, y, w, h] = b.bbox;
    let p = ERASE_PAD;
    let x0 = (x as i32 - p).max(0); let y0 = (y as i32 - p).max(0);
    let x1 = ((x + w) as i32 + p).min(iw as i32 - 1); let y1 = ((y + h) as i32 + p).min(ih as i32 - 1);
    if x1 <= x0 || y1 <= y0 { return None; }
    let (cw, ch) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
    // The page colour behind every pixel of the crop: the median of a window around it, so it follows gradients, photos and
    // stripes of several colours (a single tilted plane cannot describe those, and would call half the stripes "text").
    let get = |a: u32, c: u32| { let q = im.get_pixel(a, c); [q[0], q[1], q[2]] };
    let contrast: i32 = (0..3).map(|k| (b.fg[k] as i32 - b.bg[k] as i32).abs()).sum();
    // With a clear text colour, pixels of that colour stay out of the median: heavy lettering (stems wider than half of the
    // window, a solid bar) would otherwise become the "page colour" and stop looking like text.
    let skip = if contrast >= 120 { Some((b.fg, 60)) } else { None };
    let page = PageColours::new(iw, ih, &get, (x0 as u32, y0 as u32, x1 as u32, y1 as u32), h, skip);
    let bgs: Vec<[f32; 3]> = (0..cw * ch).map(|k| { let c = page.at(x0 + (k % cw) as i32, y0 + (k / cw) as i32); [c[0] as f32, c[1] as f32, c[2] as f32] }).collect();
    // A pixel belongs to the text if it differs from the background behind it by more than a share of the text contrast.
    // Dark text on a light page gives the old fixed limit of 60; faint text gets a lower limit, but never below 24 (noise).
    let limit = (contrast / 4).clamp(24, 60);
    let mut m = vec![false; cw * ch];
    for j in 0..ch { for i in 0..cw {
        let q = im.get_pixel(x0 as u32 + i as u32, y0 as u32 + j as u32); let g = bgs[j * cw + i];
        m[j * cw + i] = (0..3).map(|k| (q[k] as f32 - g[k]).abs()).sum::<f32>() > limit as f32;
    } }
    let mut md = m.clone(); // dilate by 2 px to swallow anti-aliased edges
    for j in 0..ch { for i in 0..cw { if m[j * cw + i] {
        for dj in -2i32..=2 { for di in -2i32..=2 {
            let (a, c) = (i as i32 + di, j as i32 + dj);
            if a >= 0 && c >= 0 && (a as usize) < cw && (c as usize) < ch { md[c as usize * cw + a as usize] = true; }
        } }
    } } }
    // Too much is marked as text (a photo, a pattern, two backgrounds): stroke inpainting would only smear it. This is judged
    // on the raw mask: the dilated one covers most of any dense line of small text, even on a flat page.
    let share = |v: &[bool]| v.iter().filter(|x| **x).count() as f32 / (cw * ch) as f32;
    Some(Crop { x0, y0, cw, ch, bgs, frosty: share(&m) > FROST_COVER, dense: share(&md) > DENSE_COVER, md, bg: b.bg })
}

/// Groups the blocks of a page for erasing. Two blocks are in one group when their crops overlap or touch (directly or
/// through a chain of blocks). Each group is erased in one go: erased one after the other, the blurred fill of the first
/// block would be pasted over by the fill of the second, and the seam between the two would show.
/// Returns indices into `blocks`, in the order of each group's first block.
pub(super) fn erase_groups(blocks: &[Block]) -> Vec<Vec<usize>> {
    let pad = ERASE_PAD as f32;
    let r: Vec<[f32; 4]> = blocks.iter().map(|b| [b.bbox[0] - pad, b.bbox[1] - pad, b.bbox[0] + b.bbox[2] + pad, b.bbox[1] + b.bbox[3] + pad]).collect();
    let mut root: Vec<usize> = (0..blocks.len()).collect();
    fn find(root: &mut [usize], mut i: usize) -> usize { while root[i] != i { root[i] = root[root[i]]; i = root[i]; } i }
    for i in 0..blocks.len() { for j in i + 1..blocks.len() {
        if r[i][0] < r[j][2] && r[j][0] < r[i][2] && r[i][1] < r[j][3] && r[j][1] < r[i][3] {
            let (a, b) = (find(&mut root, i), find(&mut root, j));
            if a != b { root[a.max(b)] = a.min(b); }
        }
    } }
    let mut out: Vec<Vec<usize>> = vec![];
    let mut slot: HashMap<usize, usize> = HashMap::new();
    for i in 0..blocks.len() {
        let k = find(&mut root, i);
        let n = *slot.entry(k).or_insert_with(|| { out.push(vec![]); out.len() - 1 });
        out[n].push(i);
    }
    out
}

/// The erase patches of a whole page, one per group of overlapping blocks (see `erase_groups`), computed in parallel from
/// the untouched page. Groups never overlap, so the patches can be applied in any order.
pub(super) fn erase_patches(im: &image::RgbaImage, blocks: &[Block]) -> Vec<Patch> {
    erase_groups(blocks).par_iter().filter_map(|g| {
        let members: Vec<&Block> = g.iter().map(|&i| &blocks[i]).collect();
        // Boxes of the other blocks hold other text: the blurred fill must not take its colours from there.
        let others: Vec<[f32; 4]> = blocks.iter().enumerate().filter(|(i, _)| !g.contains(i)).map(|(_, b)| b.bbox).collect();
        erase_group_with(im, &members, &others, RELAX_PASSES)
    }).collect()
}

/// Erases a group of blocks with one fill. The hole is everything the blocks need replaced (the stroke pixels of a plain
/// block, the whole crop of a busy one), and it is filled once, so there is no seam where the blocks overlap.
/// `others` are the boxes of blocks outside the group (their pixels are never used as source colours).
pub(super) fn erase_group_with(im: &image::RgbaImage, blocks: &[&Block], others: &[[f32; 4]], passes: usize) -> Option<Patch> {
    let crops: Vec<Crop> = blocks.iter().filter_map(|b| analyse_crop(im, b)).collect();
    if crops.is_empty() { return None; }
    let gx0 = crops.iter().map(|c| c.x0).min()?; let gy0 = crops.iter().map(|c| c.y0).min()?;
    let gx1 = crops.iter().map(|c| c.x0 + c.cw as i32 - 1).max()?; let gy1 = crops.iter().map(|c| c.y0 + c.ch as i32 - 1).max()?;
    let (gw, gh) = ((gx1 - gx0 + 1) as usize, (gy1 - gy0 + 1) as usize);
    let at = |k: usize| im.get_pixel(gx0 as u32 + (k % gw) as u32, gy0 as u32 + (k / gw) as u32);
    let to_px = |k: usize, v: [f32; 3]| (gx0 as u32 + (k % gw) as u32, gy0 as u32 + (k / gw) as u32, [v[0].round().clamp(0., 255.) as u8, v[1].round().clamp(0., 255.) as u8, v[2].round().clamp(0., 255.) as u8]);
    let (mut hole, mut frost) = (vec![false; gw * gh], vec![false; gw * gh]);
    let mut plane = vec![[0f32; 3]; gw * gh]; // the background plane of the first crop that covers a pixel
    let mut seen = vec![false; gw * gh];
    for c in &crops { for j in 0..c.ch { for i in 0..c.cw {
        let k = (c.y0 - gy0) as usize * gw + (c.x0 - gx0) as usize + j * gw + i;
        if !seen[k] { seen[k] = true; plane[k] = c.bgs[j * c.cw + i]; }
        if c.frosty { hole[k] = true; frost[k] = true; } else if c.md[j * c.cw + i] { hole[k] = true; }
    } } }
    if crops.iter().any(|c| c.frosty || c.dense) {
        // At least one busy or dense block: the group's whole hole comes from one pyramid fill (blurred for the busy
        // blocks). If nothing around it can be used, paint the smooth background.
        let vals = frosted_fill(im, (gx0, gy0, gx1, gy1), &hole, &frost, others, crops[0].bg);
        return Some(Patch((0..gw * gh).filter(|&k| hole[k]).map(|k| to_px(k, vals.as_ref().map_or(plane[k], |v| v[k]))).collect()));
    }
    let mut buf: Vec<[f32; 3]> = (0..gw * gh).map(|k| if hole[k] { plane[k] } else { let q = at(k); [q[0] as f32, q[1] as f32, q[2] as f32] }).collect();
    for _ in 0..passes { for j in 0..gh { for i in 0..gw { if hole[j * gw + i] {
        let n = [(i.saturating_sub(1), j), ((i + 1).min(gw - 1), j), (i, j.saturating_sub(1)), (i, (j + 1).min(gh - 1))];
        buf[j * gw + i] = [0, 1, 2].map(|k| n.iter().map(|(a, c)| buf[c * gw + a][k]).sum::<f32>() / 4.0);
    } } } }
    Some(Patch((0..gw * gh).filter(|&k| hole[k]).map(|k| to_px(k, buf[k])).collect()))
}

/// The blurred ("frosted glass") fill for the group grid `g = (x0, y0, x1, y1)`. `hole` marks the pixels to replace and
/// `frost` those of them that also get the blur (the others are thin strokes of plain blocks: they only need filling).
/// The fill is rebuilt only from the pixels around the hole (so no text can leak back in), minus the boxes in `others`.
/// Returns one colour per pixel of the group grid, or `None` when no pixel is left to copy colours from.
pub(super) fn frosted_fill(im: &image::RgbaImage, g: (i32, i32, i32, i32), hole: &[bool], frost: &[bool], others: &[[f32; 4]], bg: [u8; 3]) -> Option<Vec<[f32; 3]>> {
    let (gx0, gy0, gx1, gy1) = g;
    let (iw, ih) = im.dimensions();
    let (gw, gh) = ((gx1 - gx0 + 1) as usize, (gy1 - gy0 + 1) as usize);
    let ctx = ((gy1 - gy0) / 2).clamp(8, 40); // how much surrounding page the fill may look at
    let (ex0, ey0) = ((gx0 - ctx).max(0), (gy0 - ctx).max(0));
    let (ex1, ey1) = ((gx1 + ctx).min(iw as i32 - 1), (gy1 + ctx).min(ih as i32 - 1));
    let (ew, eh) = ((ex1 - ex0 + 1) as usize, (ey1 - ey0 + 1) as usize);
    let near: Vec<[f32; 4]> = others.iter().copied().filter(|o| o[0] <= ex1 as f32 && o[0] + o[2] >= ex0 as f32 && o[1] <= ey1 as f32 && o[1] + o[3] >= ey0 as f32).collect();
    let mut rgb = Vec::with_capacity(ew * eh);
    let mut known = Vec::with_capacity(ew * eh);
    for j in 0..eh { for i in 0..ew {
        let (a, c) = (ex0 + i as i32, ey0 + j as i32);
        let q = im.get_pixel(a as u32, c as u32);
        rgb.push([q[0] as f32, q[1] as f32, q[2] as f32]);
        let in_hole = a >= gx0 && a <= gx1 && c >= gy0 && c <= gy1 && hole[(c - gy0) as usize * gw + (a - gx0) as usize];
        let in_other = near.iter().any(|o| a as f32 >= o[0] && a as f32 <= o[0] + o[2] && c as f32 >= o[1] && c as f32 <= o[1] + o[3]);
        known.push(!in_hole && !in_other);
    } }
    if !known.iter().any(|k| *k) { return None; }
    // Take the tilted background plane out first and fill only what is left (the residual). A flat or linear background
    // then comes back exactly, and the pyramid only has to carry the real detail: photo texture, a second background.
    let plane = background_plane(im, ex0 as u32, ey0 as u32, ew, eh, bg);
    let res: Vec<[f32; 3]> = (0..ew * eh).map(|k| [0, 1, 2].map(|c| rgb[k][c] - plane[k][c])).collect();
    let mut filled = push_pull(&res, &known, ew, eh);
    for k in 0..ew * eh {
        // real pixels around the hole stay real, so the edge is seamless
        filled[k] = if known[k] { rgb[k] } else { [0, 1, 2].map(|c| (filled[k][c] + plane[k][c]).clamp(0.0, 255.0)) };
    }
    let mut blurred = filled.clone();
    let r = ((gy1 - gy0) / 6).clamp(2, 10) as usize;
    for _ in 0..2 { box_blur(&mut blurred, ew, eh, r); } // two box blurs are close to a Gaussian
    let mut out = vec![[0f32; 3]; gw * gh];
    for j in 0..gh { for i in 0..gw {
        let k = j * gw + i;
        if !hole[k] { continue; }
        let e = (gy0 - ey0) as usize * ew + (gx0 - ex0) as usize + j * ew + i;
        out[k] = if frost[k] { blurred[e] } else { filled[e] };
    } }
    Some(out)
}

/// Fills the unknown pixels of a `w * h` colour grid from the known ones ("push-pull" / normalised convolution on an
/// image pyramid). Known colours are averaged into ever coarser levels (push); then each level is filled from the
/// coarser one below it (pull). Cost is about 1.3 passes over the grid, no iteration, and no model.
pub(super) fn push_pull(rgb: &[[f32; 3]], known: &[bool], w: usize, h: usize) -> Vec<[f32; 3]> {
    // Every cell holds [r*weight, g*weight, b*weight, weight]; averaging four cells keeps colour and weight in step.
    let base: Vec<[f32; 4]> = rgb.iter().zip(known).map(|(c, &k)| if k { [c[0], c[1], c[2], 1.0] } else { [0.0; 4] }).collect();
    let mut lv: Vec<(Vec<[f32; 4]>, usize, usize)> = vec![(base, w, h)];
    while lv.last().is_some_and(|l| l.1 > 1 || l.2 > 1) {
        let (src, sw, sh) = lv.last().unwrap();
        let (dw, dh) = (sw.div_ceil(2), sh.div_ceil(2));
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
pub(super) fn box_blur(g: &mut [[f32; 3]], w: usize, h: usize, r: usize) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::testutil::*;

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

    #[test]
    fn blocks_that_overlap_or_touch_are_one_group() {
        let b = [blk(1, [10.0, 10.0, 100.0, 30.0], [0; 3]), blk(2, [60.0, 30.0, 100.0, 30.0], [0; 3]), blk(3, [10.0, 200.0, 50.0, 20.0], [0; 3]), blk(4, [159.0, 55.0, 40.0, 20.0], [0; 3])];
        // 1 overlaps 2; 4 starts 1 px after the end of 2, inside the erase reach, so the chain 1-2-4 is one group; 3 is alone
        assert_eq!(erase_groups(&b), vec![vec![0, 1, 3], vec![2]]);
        assert_eq!(erase_groups(&[]), Vec::<Vec<usize>>::new());
    }

    /// The largest colour jump between two neighbouring pixels inside `r` (x, y, w, h).
    fn max_step(im: &image::RgbaImage, r: [u32; 4]) -> i32 {
        let mut m = 0;
        for y in r[1]..r[1] + r[3] - 1 { for x in r[0]..r[0] + r[2] - 1 {
            let (a, b, c) = (im.get_pixel(x, y), im.get_pixel(x + 1, y), im.get_pixel(x, y + 1));
            for k in 0..3 { m = m.max((a[k] as i32 - b[k] as i32).abs()).max((a[k] as i32 - c[k] as i32).abs()); }
        } }
        m
    }

    #[test]
    fn overlapping_busy_blocks_are_filled_without_a_seam() {
        let truth = |x: u32| (128.0 + 100.0 * (x as f32 / 25.0).sin()) as u8;
        let mut im = image::RgbaImage::new(420, 140);
        for (x, _, p) in im.enumerate_pixels_mut() { let v = truth(x); *p = Rgba([v, 255 - v, 128, 255]); }
        let blocks = vec![blk(1, [60.0, 30.0, 200.0, 40.0], [128; 3]), blk(2, [160.0, 50.0, 200.0, 40.0], [128; 3])];
        let mut page = im.clone();
        for b in &blocks { page = bar_on(page, b); }
        let mut out = page.clone();
        for p in erase_patches(&page, &blocks) { p.apply(&mut out); }
        assert!(out.pixels().all(|p| p[0].min(p[1]).min(p[2]) < 200), "some of the text is left");
        // the pattern itself changes by at most ~4 per pixel; a seam from pasting one fill over another is much larger
        let step = max_step(&out, [165, 52, 90, 16]); // inside the part where the two blocks overlap
        assert!(step <= 10, "a seam of {step} grey levels between neighbouring pixels");
    }

    #[test]
    fn dense_lettering_keeps_the_stripes_it_was_written_on() {
        let page = striped_page([255, 255, 255]);
        let rgba = DynamicImage::ImageRgb8(page).to_rgba8();
        let mut b = blk(1, [70.0, 55.0, 260.0, 95.0], [237; 3]);
        (b.fg, b.bg) = box_colors(&striped_page([255, 255, 255]), 70.0, 55.0, 260.0, 95.0); // the colours the app would find
        let mut out = rgba.clone();
        for p in erase_patches(&rgba, std::slice::from_ref(&b)) { p.apply(&mut out); }
        let near = |x: u32, y: u32, c: [i32; 3], tol: i32| { let p = out.get_pixel(x, y); (0..3).all(|k| (p[k] as i32 - c[k]).abs() <= tol) };
        for x in (83..320).step_by(16) { // the middle of every former bar
            assert!(near(x, 64, [0, 0, 0], 90), "black stripe lost at {x}: {:?}", out.get_pixel(x, 64));
            assert!(near(x, 100, [220, 0, 0], 90), "red stripe lost at {x}: {:?}", out.get_pixel(x, 100));
            assert!(near(x, 135, [250, 205, 0], 90), "yellow stripe lost at {x}: {:?}", out.get_pixel(x, 135));
        }
    }
}
