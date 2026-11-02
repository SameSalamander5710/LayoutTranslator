//! The pipeline: detect text, translate it, erase the original and typeset the translation.
//! One file per concern; this file holds what they share (the `Block` type, the imports) and re-exports the public
//! functions, so callers keep writing `pipeline::detect`, `pipeline::render` and so on.

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
use std::path::{Path, PathBuf};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

pub const PDF_DPI: f32 = 200.0;

pub type Engine = OAROCR;

/// Locks a mutex even if another thread panicked while holding it. Both caches only hold finished entries, so their data
/// is still valid after such a panic; with a plain `unwrap()` one panic would make every later call panic too, until the
/// app is restarted.
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> { m.lock().unwrap_or_else(|e| e.into_inner()) }

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Block {
    pub id: u32, pub bbox: [f32; 4], pub text: String, pub translated: Option<String>,
    pub conf: f32, pub font_px: f32, pub fg: [u8; 3], pub bg: [u8; 3],
}

struct Line { b: [f32; 4], t: String, c: f32, fg: [u8; 3], bg: [u8; 3], layer: bool } // `layer`: from the PDF text layer, not from OCR

mod boxes;
mod colour;
mod erase;
mod imageio;
mod ocr;
mod pdf_in;
mod pdf_out;
mod translate;
mod typeset;
#[cfg(test)]
mod testutil;

pub use boxes::*;
pub use colour::*;
use erase::*;
pub use imageio::*;
pub use ocr::*;
pub use pdf_in::*;
pub use pdf_out::*;
pub use translate::*;
pub use typeset::*;
