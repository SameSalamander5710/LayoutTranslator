# TODO for the local AI

Nothing in this project has been compiled or run. Work top to bottom. After each item, run the app (or `cargo check` in `src-tauri`) and report what happened before moving on. Search the code for `TODO(local AI)` and `ADJUST HERE` to find the exact spots.

Full rules are in `AGENTS.md`. In short: small steps, explain each change in plain language, give the exact command and the expected result, never add a dependency without checking its latest version and licence, never copy GPL/AGPL code, prefer PowerShell commands. The user has no Rust background.

## A. Make it build
- [ ] A1. Run `npm install`, then `cargo check` inside `src-tauri`. Fix compile errors at the root cause (no `unwrap`/`allow` to silence them).
- [ ] A2. `pipeline.rs`, `ocr()`: the bounding-box access (`r.bounding_box.points`, `.x`, `.y`) is a guess. Check the `oar-ocr` docs (docs.rs) for the real field names.
- [ ] A3. `pipeline.rs`: `pub type Engine = OAROCR` is assumed. Confirm the type returned by `OAROCRBuilder::build()`, and whether `oar-ocr` needs feature flags (for example automatic ONNX Runtime download). Check the newest crate version instead of `0.7`.
- [ ] A4. `OAROCRBuilder::new` is called with `String` arguments; adjust if it wants `&str` or `Path`.
- [ ] A5. `rasterise_pdf()`: verify pdfium-render calls for the installed version (bind function names, whether `as_image()` returns a `Result`, thread-safety feature).
- [ ] A6. Create icons once: `npx tauri icon <square.png>`.
- [ ] A7. Confirm the Tauri v2 permissions in `capabilities/default.json` are enough for the dialog and for the drag-and-drop event; add what is missing.

## B. Models and assets
- [ ] B1. Fill the three English model URLs in `scripts/models.json` (PP-OCRv5 mobile detection, English recognition, matching dictionary, as ONNX). Check that the recognition model and `dict.txt` belong together; a mismatched dictionary gives garbage text. Record licences in `THIRD_PARTY.md`.
- [ ] B2. Run `npm run models`. Verify the pdfium download names and archive layout in the last section of `scripts/fetch-models.mjs`, and that `tar` works on Windows.
- [ ] B3. Noto Sans has no CJK glyphs. Add per-script fonts (for example Noto Sans JP/SC/KR, Arabic) and choose by target language; today Japanese output will show empty boxes.
- [ ] B4. ONNX Runtime must ship with the installer. Find out how `ort` links on Windows and macOS, bundle the DLL/dylib if needed, and test on a clean Windows VM.
- [ ] B5. Test adding a language by hand: create `models/<name>/` (det.onnx, rec.onnx, dict.txt) next to the installed app and confirm it appears in the dropdown after restart. Check that the per-user NSIS install makes that folder writable, and the `app_data_dir/models` fallback works.

## C. Engine cache
- [ ] C1. Confirm the first OCR call is slow and later calls are fast. Check that the cache lock does not freeze the UI while the first model builds (consider building in `spawn_blocking`).
- [ ] C2. Decide whether engines should be dropped when the language changes (memory).

## D. PDF
- [ ] D1. Test with a 1-page, a 10-page and a scanned PDF. Check rasterisation at 200 DPI (`PDF_DPI`), memory use, and speed. Big files rasterise every page up front; consider lazy rendering.
- [ ] D2. Open the exported PDF in Acrobat, Edge and macOS Preview. Validate the hand-written writer (`write_pdf`); the xref entries must be exactly 20 bytes each. Consider replacing it with a library if it is fragile.
- [ ] D3. Output PDFs are page images (text not selectable) and can be large; try lower JPEG quality or an optional "text layer" mode.
- [ ] D4. PDFs with a real text layer are OCR'd like scans. Add the shortcut from PLAN.md (read text runs and boxes with pdfium, skip OCR).
- [ ] D5. Add a cancel button and a progress bar for long documents.

## E. Inpainting
- [ ] E1. `erase()` is a classical diffusion fill, not a neural model. Build a fixtures set (flat background, gradient, coloured band, photo, light-on-dark text) and judge the results.
- [ ] E2. The stroke threshold (`> 60`) assumes a flat background. Make it adaptive for gradients, and reconsider the 45% fallback to a flat fill.
- [ ] E3. Optional: replace it with a LaMa ONNX model through `ort` for photos and textures. Keep the classical fill as the fallback.

## F. Quality
- [ ] F1. Text without spaces (CJK, Thai) does not wrap: `wrap()` only breaks on spaces. Add per-character wrapping and proper shaping (consider `cosmic-text`).
- [ ] F2. Handle rotated or vertical text, and very small text (minimum readable size).
- [ ] F3. Skip translating blocks that are only numbers, URLs or codes.
- [ ] F4. Local models sometimes break the JSON-array reply (extra prose, reasoning tags). Harden the parser in `call()` and consider constrained output (llama.cpp grammar or `response_format`).

## G. Translation endpoint
- [ ] G1. Add a "Test connection" button and clear error messages for unreachable or unauthorised endpoints.
- [ ] G2. The API key is saved in the webview `localStorage` as plain text. Move it to the OS keychain, or at least warn the user.
- [ ] G3. `"csp": null` in `tauri.conf.json` is a development setting. Write a strict CSP for release.
- [ ] G4. Large pages: split block lists into batches so prompts fit the model's context.

## H. Tests
- [ ] H1. Unit tests for `group()`, `wrap()`, `erase()` and `write_pdf()` (check the output parses with pdfium).
- [ ] H2. One end-to-end test per fixture: image in, expected block text out (allow small OCR differences).

## I. Release
- [ ] I1. Commit `Cargo.lock` and `package-lock.json`, switch the workflow back to `npm ci`.
- [ ] I2. Check the macOS Intel runner label (`macos-13`) is still valid; run the workflow with `workflow_dispatch` and fix failures.
- [ ] I3. macOS: confirm pdfium and ONNX Runtime dylibs load from inside the `.app`, and document the unsigned-app workaround (or set up signing/notarization).
- [ ] I4. Windows: document the SmartScreen warning, or set up code signing.
- [x] I5. `PLAN.md` was updated to match the code (Google Translate dropped). Keep it in sync as things change.
