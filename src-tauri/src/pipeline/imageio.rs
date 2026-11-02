//! Reading and writing page images: load with the EXIF turn applied, PNG data URLs for the UI, atomic writes.

use super::*;

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

pub fn png_data_url(im: &DynamicImage) -> Result<String, String> {
    let mut buf = std::io::Cursor::new(vec![]);
    im.write_to(&mut buf, image::ImageFormat::Png).map_err(|e| e.to_string())?;
    Ok(format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(buf.into_inner())))
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

/// Saves an image taken from the clipboard (straight RGBA, `width * height * 4` bytes) as a new PNG in `dir` and returns its
/// path. Transparent areas are flattened onto white, because the page is read as plain RGB afterwards.
pub fn save_clipboard_png(width: usize, height: usize, rgba: &[u8], dir: &Path) -> Result<PathBuf, String> {
    let bad = || "The clipboard image is empty or damaged".to_string();
    if width == 0 || height == 0 || rgba.len() != width * height * 4 { return Err(bad()); }
    let mut im = image::RgbaImage::from_raw(width as u32, height as u32, rgba.to_vec()).ok_or_else(bad)?;
    for p in im.pixels_mut() {
        let a = p[3] as u32;
        for c in 0..3 { p[c] = ((p[c] as u32 * a + 255 * (255 - a)) / 255) as u8; }
        p[3] = 255;
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("Could not create {}: {e}", dir.display()))?;
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let path = dir.join(format!("pasted-{stamp}.png"));
    save_png(&im, &path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::testutil::*;

    #[test]
    fn a_clipboard_image_is_saved_as_png_on_white() {
        let dir = temp_dir("clipboard");
        let rgba = [255u8, 0, 0, 255, 0, 0, 255, 0]; // one opaque red pixel, one fully transparent blue pixel
        let path = save_clipboard_png(2, 1, &rgba, &dir).unwrap();
        let im = image::open(&path).unwrap().to_rgb8();
        assert_eq!((im.width(), im.height()), (2, 1));
        assert_eq!(im.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(im.get_pixel(1, 0).0, [255, 255, 255]);
        assert!(save_clipboard_png(2, 2, &rgba, &dir).is_err()); // the byte count does not match the size
        assert!(save_clipboard_png(0, 0, &[], &dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
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
}
