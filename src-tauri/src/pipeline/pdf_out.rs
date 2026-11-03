//! Writing a PDF: one JPEG image per page (the text is not selectable).

use super::*;

pub(super) fn put(o: &mut Vec<u8>, offs: &mut Vec<usize>, id: usize, body: &[u8]) {
    offs.push(o.len());
    o.extend(format!("{id} 0 obj\n").as_bytes()); o.extend(body); o.extend(b"\nendobj\n");
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

/// Renders every page and writes them as one PDF. One page at a time: the page is rendered, only its JPEG is kept, and
/// the decoded image is dropped before the next page is rendered. Pages with no blocks are exported as the original.
pub fn export_pdf(pages: &[(&Path, &[Block])], font: &Path, out: &Path) -> Result<(), String> {
    let mut done = Vec::with_capacity(pages.len());
    for (path, blocks) in pages {
        let im = render(path, font, blocks)?;
        // Pages that were too large were drawn at a lower resolution; the real one keeps the exported page at its real size.
        done.push(jpeg_page(&im, page_dpi(path))?);
    }
    write_pdf_pages(&done, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::testutil::*;

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
}
