#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use layout_translator::pipeline::{self, Block, EngineCache};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};
use tokio_util::sync::CancellationToken;

// Model folders, searched in order. The first one sits next to the app (the "app folder");
// the second is a per-user fallback in case the app folder is read-only.
fn roots(app: &AppHandle) -> Vec<PathBuf> {
    let mut v = vec![];
    if let Ok(r) = app.path().resource_dir() { v.push(r.join("models")); }
    if let Ok(d) = app.path().app_data_dir() { v.push(d.join("models")); }
    v
}
fn has_model(d: &Path) -> bool { ["det.onnx", "rec.onnx", "dict.txt"].iter().all(|f| d.join(f).exists()) }
fn lang_dir(app: &AppHandle, lang: &str) -> Result<PathBuf, String> {
    roots(app).into_iter().map(|r| r.join(lang)).find(|d| has_model(d))
        .ok_or(format!("No model folder for '{lang}'. Expected models/{lang}/ with det.onnx, rec.onnx and dict.txt"))
}

#[tauri::command]
fn list_languages(app: AppHandle) -> Vec<String> {
    let mut v: Vec<String> = roots(&app).iter().filter_map(|r| std::fs::read_dir(r).ok())
        .flat_map(|rd| rd.flatten()).filter(|e| has_model(&e.path()))
        .map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    v.sort(); v.dedup(); v
}

#[tauri::command]
fn open_models_dir(app: AppHandle) -> Result<(), String> {
    let d = roots(&app).into_iter().next().ok_or("no models folder")?;
    std::fs::create_dir_all(&d).ok();
    std::process::Command::new(if cfg!(windows) { "explorer" } else { "open" }).arg(d)
        .spawn().map(|_| ()).map_err(|e| e.to_string())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Page { width: u32, height: u32, original: String, blocks: Vec<Block>, layer_lines: usize, ocr_ran: bool }

#[tauri::command]
async fn run_ocr(app: AppHandle, cache: tauri::State<'_, EngineCache>, path: String, lang: String, mode: Option<String>) -> Result<Page, String> {
    // mode: "hybrid" (default), "ocr" or "text" (see pipeline::detect). Text-only mode needs no OCR model.
    let mode = mode.unwrap_or_else(|| "hybrid".into());
    let dir = if mode == "text" { PathBuf::new() } else { lang_dir(&app, &lang)? };
    let p = PathBuf::from(&path);
    let p2 = p.clone();
    let dir2 = dir.clone();
    let cache = cache.inner().clone(); // the app-wide cache, so the OCR models are loaded once
    let (mut blocks, layer_lines, ocr_ran) = tauri::async_runtime::spawn_blocking(move || pipeline::detect(&p2, &dir2, &cache, &mode))
        .await
        .map_err(|e| e.to_string())??;
    let im = image::open(&p).map_err(|e| e.to_string())?;
    let rgb = im.to_rgb8();
    for (i, b) in blocks.iter_mut().enumerate() {
        b.id = i as u32 + 1;
        pipeline::colors(&rgb, b);
    }
    Ok(Page { width: im.width(), height: im.height(), original: pipeline::png_data_url(&im)?, blocks, layer_lines, ocr_ran })
}

/// The cancel token of the translation request that is running. The Stop button fires it (see `cancel_run`).
#[derive(Default)]
struct RunControl(std::sync::Mutex<CancellationToken>);

/// Stop button: ends the translation request in progress. The HTTP request is dropped, which closes its connection.
#[tauri::command]
fn cancel_run(ctl: tauri::State<'_, RunControl>) { ctl.0.lock().unwrap().cancel(); }

#[tauri::command]
async fn translate_blocks(
    cache: tauri::State<'_, pipeline::TranslationCache>,
    ctl: tauri::State<'_, RunControl>,
    mut blocks: Vec<Block>,
    target: String,
    endpoint: String,
    options: Option<pipeline::TranslateOptions>,
    only: Option<Vec<u32>>,
) -> Result<TranslateOut, String> {
    let options = options.unwrap_or_default();
    let token = { let mut t = ctl.0.lock().unwrap(); *t = CancellationToken::new(); t.clone() }; // a fresh token per request
    let (cached, missing) = token
        .run_until_cancelled(pipeline::translate(&mut blocks, &target, &endpoint, &options, only.as_deref(), &cache))
        .await
        .ok_or_else(|| "Cancelled".to_string())??;
    Ok(TranslateOut { blocks, cached, missing })
}

/// The page's blocks, plus how many came from the cache and how many got no translation (they keep their source text).
#[derive(Serialize)]
struct TranslateOut { blocks: Vec<Block>, cached: usize, missing: usize }

// Box editing: read the text of a box the user drew (selectable text first, then OCR) ...
#[tauri::command]
async fn read_region(app: AppHandle, cache: tauri::State<'_, EngineCache>, path: String, lang: String, mode: Option<String>, bbox: [f32; 4]) -> Result<Block, String> {
    let mode = mode.unwrap_or_else(|| "hybrid".into());
    let dir = if mode == "text" { PathBuf::new() } else { lang_dir(&app, &lang)? };
    let p = PathBuf::from(&path);
    let cache = cache.inner().clone();
    tauri::async_runtime::spawn_blocking(move || pipeline::read_region(&p, &dir, &cache, bbox, &mode))
        .await
        .map_err(|e| e.to_string())?
}

// ... and re-sample the colours of a box the user moved or resized.
#[tauri::command]
async fn sample_colors(path: String, bbox: [f32; 4]) -> Result<([u8; 3], [u8; 3]), String> {
    pipeline::sample_colors(Path::new(&path), bbox)
}

#[tauri::command]
async fn test_connection(endpoint: String) -> Result<pipeline::ConnInfo, String> { pipeline::test_connection(&endpoint).await }

// Renders the translated page. Returns a PNG data URL for preview; also writes `out` when given.
fn pick_font(app: &AppHandle, lang: &str) -> std::path::PathBuf {
    let dir = app.path().resource_dir().unwrap_or_default().join("fonts");
    match lang.to_lowercase().as_str() {
        "ja" | "jpn" | "japanese" => dir.join("NotoSansJP-Regular.ttf"),
        "zh" | "zh-cn" | "chinese" | "zh-tw" | "chinese-traditional" => dir.join("NotoSansSC-Regular.ttf"),
        "ko" | "korean" => dir.join("NotoSansKR-Regular.ttf"),
        _ => dir.join("NotoSans-Regular.ttf"),
    }
}

#[tauri::command]
async fn render_page(app: AppHandle, path: String, blocks: Vec<Block>, out: Option<String>, lang: Option<String>) -> Result<String, String> {
    let lang = lang.unwrap_or_default();
    let font = pick_font(&app, &lang);
    if !font.exists() {
        let fallback = app.path().resource_dir().map_err(|e| e.to_string())?.join("fonts/NotoSans-Regular.ttf");
        let im = pipeline::render(Path::new(&path), &fallback, &blocks)?;
        if let Some(o) = out { im.save(o).map_err(|e| e.to_string())?; }
        return pipeline::png_data_url(&image::DynamicImage::ImageRgba8(im));
    }
    let im = pipeline::render(Path::new(&path), &font, &blocks)?;
    if let Some(o) = out { im.save(o).map_err(|e| e.to_string())?; }
    pipeline::png_data_url(&image::DynamicImage::ImageRgba8(im))
}

// PDFs are rasterised to page PNGs in the app cache; images pass through as a single page.
#[tauri::command]
async fn open_document(app: AppHandle, path: String) -> Result<Vec<String>, String> {
    if !path.to_lowercase().ends_with(".pdf") { return Ok(vec![path]); }
    let lib = app.path().resource_dir().map_err(|e| e.to_string())?.join("pdfium");
    let out = app.path().app_cache_dir().map_err(|e| e.to_string())?.join("pages");
    let _ = std::fs::remove_dir_all(&out);
    pipeline::rasterise_pdf(Path::new(&path), &lib, &out)
}

#[tauri::command]
async fn preview(path: String) -> Result<String, String> {
    pipeline::png_data_url(&image::open(&path).map_err(|e| e.to_string())?)
}

#[derive(Deserialize)]
struct PageIn { path: String, blocks: Vec<Block> }

// Pages with no blocks (not processed) are exported as the original.
#[tauri::command]
async fn export_pdf(app: AppHandle, pages: Vec<PageIn>, out: String, lang: Option<String>) -> Result<(), String> {
    let lang = lang.unwrap_or_default();
    let font = pick_font(&app, &lang);
    let font = if font.exists() { font } else { app.path().resource_dir().map_err(|e| e.to_string())?.join("fonts/NotoSans-Regular.ttf") };
    let ims = pages.iter().map(|p| pipeline::render(Path::new(&p.path), &font, &p.blocks)).collect::<Result<Vec<_>, _>>()?;
    pipeline::write_pdf(&ims, pipeline::PDF_DPI, Path::new(&out))
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(EngineCache::default())
        .manage(pipeline::TranslationCache::default())
        .manage(RunControl::default())
        .invoke_handler(tauri::generate_handler![list_languages, open_models_dir, run_ocr, translate_blocks, cancel_run, read_region, sample_colors, test_connection, render_page, open_document, preview, export_pdf])
        .run(tauri::generate_context!())
        .expect("error while running app");
}
