//! Text colour and page colour of a box.

use super::*;

pub(super) fn color_dist(a: &[u8; 3], b: &[u8; 3]) -> i32 {
    (0..3).map(|c| (a[c] as i32 - b[c] as i32).abs()).sum()
}

/// Background colour (median of a ring around the box) and text colour of a box.
/// The text colour comes from `text_colour`; if that cannot tell text from background, the old rule is used: the mean of
/// the 10% of pixels that differ most from the ring colour.
pub(super) fn box_colors(img: &RgbImage, x: f32, y: f32, bw: f32, bh: f32) -> ([u8; 3], [u8; 3]) {
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
    inner.sort_by_key(|p| std::cmp::Reverse(p.0));
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
pub(super) struct PageColours { x0: i32, y0: i32, g: usize, nx: usize, cells: Vec<[u8; 3]> }

impl PageColours {
    pub(super) fn new(w: u32, h: u32, get: &dyn Fn(u32, u32) -> [u8; 3], rect: (u32, u32, u32, u32), text_h: f32, skip: Option<([u8; 3], i32)>) -> PageColours {
        let (x0, y0, x1, y1) = rect; // the rectangle (inclusive) to compute the colours for
        let (cw, ch) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
        let g = (((cw * ch) as f32 / 1500.0).sqrt().ceil() as usize).max(3);
        let r = ((text_h * 0.75) as i32).clamp(6, 30);
        let step = (r / 6).max(2);
        let (nx, ny) = (cw.div_ceil(g), ch.div_ceil(g));
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
                    if skip.is_none_or(|(c, tol)| (0..3).map(|k| (p[k] as i32 - c[k] as i32).abs()).sum::<i32>() >= tol) { for c in 0..3 { win[c].push(p[c]); } }
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
    pub(super) fn at(&self, x: i32, y: i32) -> [u8; 3] { self.cells[((y - self.y0) as usize / self.g) * self.nx + (x - self.x0) as usize / self.g] }
}

/// The colour of the text in a box, found without trusting the colour around the box.
///
/// The ring rule fails when the text sits on a busy or multi-coloured background: white letters on a red stripe look
/// "closer" to a grey page margin than the red does, so red is reported. Here every pixel is compared with the page colour
/// right behind it instead (`PageColours`). The pixels that differ most from their own surroundings are the text; the most
/// common colour among them is the answer. Returns `None` when nothing stands out, or when "text" would be most of the box
/// (the median then followed the text, not the page).
pub(super) fn text_colour(img: &RgbImage, x: f32, y: f32, bw: f32, bh: f32) -> Option<[u8; 3]> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 || x >= w as f32 || y >= h as f32 { return None; }
    let (x0, y0) = (x.max(0.0) as u32, y.max(0.0) as u32);
    let (x1, y1) = (((x + bw) as u32).min(w - 1), ((y + bh) as u32).min(h - 1));
    if x1 <= x0 + 1 || y1 <= y0 + 1 { return None; }
    let get = |a: u32, c: u32| { let p = img.get_pixel(a, c); [p[0], p[1], p[2]] };
    let page = PageColours::new(w, h, &get, (x0, y0, x1, y1), bh, None);
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::testutil::*;

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
}
