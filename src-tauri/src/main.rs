#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use layout_translator::error::AppError;
use layout_translator::launcher;
use layout_translator::pipeline::{self, Block, EngineCache};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};
use tokio_util::sync::CancellationToken;

// Runs slow work (OCR, rendering, PDF, image decoding, the clipboard) on a blocking thread, so the async runtime stays free for the
// window and for other commands. The `String` error of the work and a panic of the thread both come back as an `AppError`.
async fn blocking<T, F>(work: F) -> Result<T, AppError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(work).await.map_err(app_err)?.map_err(AppError::from)
}

fn app_err(e: impl std::fmt::Display) -> AppError { AppError::from(e.to_string()) }

// Model folders, searched in order. The first one sits next to the app (the "app folder");
// the second is a per-user fallback in case the app folder is read-only.
fn roots(app: &AppHandle) -> Vec<PathBuf> {
    let mut v = vec![];
    if let Ok(r) = app.path().resource_dir() { v.push(r.join("models")); }
    if let Ok(d) = app.path().app_data_dir() { v.push(d.join("models")); }
    v
}
fn has_model(d: &Path) -> bool { pipeline::model_problem(d).is_none() }
fn lang_dir(app: &AppHandle, lang: &str) -> Result<PathBuf, AppError> {
    if !pipeline::valid_lang_name(lang) { return Err(format!("'{lang}' is not a valid language name").into()); }
    let dirs: Vec<PathBuf> = roots(app).into_iter().map(|r| r.join(lang)).collect();
    if let Some(d) = dirs.iter().find(|d| has_model(d)) { return Ok(d.clone()); }
    // A folder exists but is incomplete: say which file is the problem instead of a vague "not found".
    if let Some((d, why)) = dirs.iter().filter(|d| d.is_dir()).find_map(|d| pipeline::model_problem(d).map(|w| (d, w))) {
        return Err(AppError::no_model(format!("The model folder {} cannot be used: {why}. Run `npm run models` again, or replace the file.", d.display())));
    }
    Err(AppError::no_model(format!("No model folder for '{lang}'. Expected models/{lang}/ with det.onnx, rec.onnx and dict.txt")))
}

#[tauri::command]
fn list_languages(app: AppHandle) -> Vec<String> {
    let mut v: Vec<String> = roots(&app).iter().filter_map(|r| std::fs::read_dir(r).ok())
        .flat_map(|rd| rd.flatten()).filter(|e| has_model(&e.path()))
        .map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| pipeline::valid_lang_name(n)).collect(); // never list a name lang_dir would refuse
    v.sort(); v.dedup(); v
}

#[tauri::command]
fn open_models_dir(app: AppHandle) -> Result<(), AppError> {
    let d = roots(&app).into_iter().next().ok_or("no models folder")?;
    std::fs::create_dir_all(&d).ok();
    Ok(launcher::open_folder(&d)?)
}

// ---- Server launcher: the command that starts the translation server (see launcher.rs) ----

/// The places for the command file: next to the program, then the per-user folder (only that one on macOS).
fn command_files(app: &AppHandle) -> Vec<PathBuf> {
    launcher::command_file_candidates(app.path().resource_dir().ok().as_deref(), app.path().app_data_dir().ok().as_deref(), !cfg!(target_os = "macos"))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ServerCommand { command: String, path: String }
impl From<launcher::StoredCommand> for ServerCommand {
    fn from(s: launcher::StoredCommand) -> Self { Self { command: s.command, path: s.path.display().to_string() } }
}

#[tauri::command]
fn get_server_command(app: AppHandle) -> Result<ServerCommand, AppError> { Ok(launcher::load_command(&command_files(&app))?.into()) }

#[tauri::command]
fn save_server_command(app: AppHandle, command: String) -> Result<ServerCommand, AppError> { Ok(launcher::save_command(&command_files(&app), &command)?.into()) }

/// Runs the saved command in a new terminal window. It takes no argument on purpose: only the saved command can be started.
#[tauri::command]
async fn launch_server(app: AppHandle) -> Result<(), AppError> {
    let saved = launcher::load_command(&command_files(&app))?;
    Ok(launcher::launch(&saved.command, &launcher::Env::current())?)
}

#[tauri::command]
fn open_server_command_folder(app: AppHandle) -> Result<(), AppError> {
    let file = launcher::load_command(&command_files(&app))?.path;
    let dir = file.parent().ok_or("The command file has no folder")?;
    std::fs::create_dir_all(dir).ok();
    Ok(launcher::open_folder(dir)?)
}

#[tauri::command]
async fn run_ocr(app: AppHandle, cache: tauri::State<'_, EngineCache>, path: String, lang: String, mode: Option<String>) -> Result<pipeline::Page, AppError> {
    // mode: "hybrid" (default), "ocr" or "text" (see pipeline::detect). Text-only mode needs no OCR model.
    let mode = mode.unwrap_or_else(|| "hybrid".into());
    let dir = if mode == "text" { PathBuf::new() } else { lang_dir(&app, &lang)? };
    let cache = cache.inner().clone(); // the app-wide cache, so the OCR models are loaded once
    blocking(move || pipeline::detect_page(Path::new(&path), &dir, &cache, &mode)).await
}

/// The cancel token of the translation request that is running. The Stop button fires it (see `cancel_run`).
#[derive(Default)]
struct RunControl(std::sync::Mutex<CancellationToken>);

/// Stop button: ends the translation request in progress. The HTTP request is dropped, which closes its connection.
#[tauri::command]
fn cancel_run(ctl: tauri::State<'_, RunControl>) { ctl.0.lock().unwrap_or_else(|e| e.into_inner()).cancel(); } // a poisoned lock is still a usable token

#[tauri::command]
async fn translate_blocks(
    cache: tauri::State<'_, pipeline::TranslationCache>,
    ctl: tauri::State<'_, RunControl>,
    mut blocks: Vec<Block>,
    target: String,
    endpoint: String,
    options: Option<pipeline::TranslateOptions>,
    only: Option<Vec<u32>>,
) -> Result<TranslateOut, AppError> {
    let options = options.unwrap_or_default();
    let token = { let mut t = ctl.0.lock().unwrap_or_else(|e| e.into_inner()); *t = CancellationToken::new(); t.clone() }; // a fresh token per request
    let (cached, missing) = token
        .run_until_cancelled(pipeline::translate(&mut blocks, &target, &endpoint, &options, only.as_deref(), &cache))
        .await
        .ok_or_else(AppError::cancelled)?
        .map_err(AppError::server)?;
    Ok(TranslateOut { blocks, cached, missing })
}

/// The page's blocks, plus how many came from the cache and how many got no translation (they keep their source text).
#[derive(Serialize)]
struct TranslateOut { blocks: Vec<Block>, cached: usize, missing: usize }

// Box editing: read the text of a box the user drew (selectable text first, then OCR) ...
#[tauri::command]
async fn read_region(app: AppHandle, cache: tauri::State<'_, EngineCache>, path: String, lang: String, mode: Option<String>, bbox: [f32; 4]) -> Result<Block, AppError> {
    let mode = mode.unwrap_or_else(|| "hybrid".into());
    let dir = if mode == "text" { PathBuf::new() } else { lang_dir(&app, &lang)? };
    let p = PathBuf::from(&path);
    let cache = cache.inner().clone();
    blocking(move || pipeline::read_region(&p, &dir, &cache, bbox, &mode)).await
}

// ... and re-sample the colours of a box the user moved or resized.
#[tauri::command]
async fn sample_colors(path: String, bbox: [f32; 4]) -> Result<([u8; 3], [u8; 3]), AppError> {
    blocking(move || pipeline::sample_colors(Path::new(&path), bbox)).await
}

#[tauri::command]
async fn test_connection(endpoint: String) -> Result<pipeline::ConnInfo, AppError> { pipeline::test_connection(&endpoint).await.map_err(AppError::server) }

fn fonts_dir(app: &AppHandle) -> Result<PathBuf, AppError> {
    Ok(app.path().resource_dir().map_err(app_err)?.join("fonts"))
}

// Renders the translated page. Returns a PNG data URL for preview; also writes `out` when given. `target` is the language of the
// translation (it decides the font, see pipeline::font_file_name).
#[tauri::command]
async fn render_page(app: AppHandle, path: String, blocks: Vec<Block>, out: Option<String>, target: Option<String>) -> Result<String, AppError> {
    let font = pipeline::resolve_font(&fonts_dir(&app)?, &target.unwrap_or_default());
    blocking(move || pipeline::render_preview(Path::new(&path), &font, &blocks, out.as_deref().map(Path::new))).await
}

// The page images and text layers of the open PDF live in the app cache, one folder per running app (`pages-<pid>`), so two
// app windows never delete each other's files. The folder is removed when the app starts and when its window closes,
// because the pages of a private document must not stay on disk.
fn pages_dir(app: &AppHandle) -> Result<PathBuf, AppError> {
    Ok(app.path().app_cache_dir().map_err(app_err)?.join(format!("pages-{}", std::process::id())))
}
fn clear_pages(app: &AppHandle) {
    if let Ok(d) = pages_dir(app) { let _ = std::fs::remove_dir_all(d); }
    // the single shared folder that earlier versions used
    if let Ok(c) = app.path().app_cache_dir() { let _ = std::fs::remove_dir_all(c.join("pages")); }
}

// PDFs are rasterised to page PNGs in the app cache; images pass through as a single page.
#[tauri::command]
async fn open_document(app: AppHandle, path: String) -> Result<Vec<String>, AppError> {
    if !path.to_lowercase().ends_with(".pdf") { return Ok(vec![path]); }
    let lib = app.path().resource_dir().map_err(app_err)?.join("pdfium");
    let out = pages_dir(&app)?;
    blocking(move || {
        let _ = std::fs::remove_dir_all(&out);
        pipeline::rasterise_pdf(Path::new(&path), &lib, &out)
    })
    .await
}

// Paste: the image on the clipboard (a screenshot, or an image copied from a browser or an editor) is saved as a PNG next to
// the PDF page images in the app cache, and the UI opens it like any other image. Both the paste button and Ctrl+V use it.
#[tauri::command]
async fn paste_image(app: AppHandle) -> Result<String, AppError> {
    let dir = pages_dir(&app)?;
    blocking(move || {
        let img = arboard::Clipboard::new().and_then(|mut c| c.get_image()).map_err(|e| match e {
            arboard::Error::ContentNotAvailable => "There is no image on the clipboard. Copy an image or take a screenshot first, then paste again.".to_string(),
            e => format!("The clipboard could not be read: {e}"),
        })?;
        pipeline::save_clipboard_png(img.width, img.height, &img.bytes, &dir).map(|p| p.to_string_lossy().into_owned())
    })
    .await
}

#[tauri::command]
async fn preview(path: String) -> Result<String, AppError> {
    blocking(move || pipeline::png_data_url(&pipeline::load_oriented(Path::new(&path))?)).await
}

#[derive(Deserialize)]
struct PageIn { path: String, blocks: Vec<Block> }

// Pages with no blocks (not processed) are exported as the original.
#[tauri::command]
async fn export_pdf(app: AppHandle, pages: Vec<PageIn>, out: String, target: Option<String>) -> Result<(), AppError> {
    let font = pipeline::resolve_font(&fonts_dir(&app)?, &target.unwrap_or_default());
    blocking(move || {
        let pages: Vec<(&Path, &[Block])> = pages.iter().map(|p| (Path::new(&p.path), p.blocks.as_slice())).collect();
        pipeline::export_pdf(&pages, &font, Path::new(&out))
    })
    .await
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            clear_pages(app.handle());
            // A panic that is not inside a command would close the app silently; keep a trace for bug reports.
            if let Ok(d) = app.path().app_data_dir() { layout_translator::crashlog::install(d.join("last-error.log")); }
            // Scripts of the server launcher that a crash left behind
            launcher::sweep_old_scripts(&std::env::temp_dir(), std::time::Duration::from_secs(24 * 3600));
            Ok(())
        })
        .on_window_event(|window, event| { if let tauri::WindowEvent::Destroyed = event { clear_pages(window.app_handle()); } })
        .manage(EngineCache::default())
        .manage(pipeline::TranslationCache::default())
        .manage(RunControl::default())
        .invoke_handler(tauri::generate_handler![list_languages, open_models_dir, get_server_command, save_server_command, launch_server, open_server_command_folder, run_ocr, translate_blocks, cancel_run, read_region, sample_colors, test_connection, render_page, open_document, paste_image, preview, export_pdf])
        .run(tauri::generate_context!())
        .expect("error while running app");
}
