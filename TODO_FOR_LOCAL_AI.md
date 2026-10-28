# TODO for the local AI

Nothing in this project has been compiled or run. Work top to bottom. After each item, run the app (or `cargo check` in `src-tauri`) and report what happened before moving on. Search the code for `TODO(local AI)` and `ADJUST HERE` to find the exact spots.

Full rules are in `AGENTS.md`. In short: small steps, explain each change in plain language, give the exact command and the expected result, never add a dependency without checking its latest version and licence, never copy GPL/AGPL code, prefer PowerShell commands. The user has no Rust background.

## A. Make it build
- [x] A1. Run `npm install`, then `cargo check` inside `src-tauri`. Fixed compile errors (base64/Engine, oar-ocr 0.10 API).
- [x] A2. `pipeline.rs`, `ocr()`: updated to use `BoundingBox` methods `x_min()/y_min()/x_max()/y_max()` for oar-ocr 0.10.
- [x] A3. Updated to `oar-ocr = "0.10"`; added `ort` dependency. `OAROCR` confirmed from builder.
- [x] A4. `OAROCRBuilder::new` still takes `String` args - working as-is.
- [x] A5. pdfium-render 0.8 compiles; API matches current usage.
- [x] A6. Created icons with `npx tauri icon`.
- [x] A7. Capabilities present (`core:default`, `dialog:default`) - may need drag-drop later.

## B. Models and assets
- [x] B1. Filled English PP-OCRv5 URLs (det `pp-ocrv5_mobile_det.onnx`, rec `en_pp-ocrv5_mobile_rec.onnx`, dict `ppocrv5_en_dict.txt` from GreatV/oar-ocr v0.3.0).
- [x] B2. Ran `npm run models` - downloaded font, models, and pdfium (Windows x64). `tar` works on Windows.
- [x] B3. Added Noto Sans JP/SC/KR to fonts, pick by language in render/export. (Arabic was removed again: without shaping its letters do not join.)
- [x] B4. Using `ort` rc.13; ONNX Runtime will be linked via `ort` (may need bundling on some setups). Added note to verify on clean Windows.
- [x] B5. Code supports per-language folders in resource dir and `app_data_dir/models` fallback; `list_languages` scans both. Documented in README. Needs runtime test.

## C. Engine cache
- [x] C1. OCR runs in `spawn_blocking`; cache kept per language dir.
- [x] C2. Kept cached across calls for same lang; can be extended to drop on lang change later.

## D. PDF
- [x] D1. Rasterisation uses pdfium at 200 DPI; loads from resources/pdfium with fallback. Needs runtime tests.
- [x] D2. Using pdfium-render 0.8; writer unchanged - needs validation.
- [ ] D3. Output PDFs are page images (text not selectable) and can be large; try lower JPEG quality or an optional "text layer" mode.
- [x] D4. Hybrid run: selectable PDF text is read with pdfium first, OCR only covers the rest (and is skipped on pages with no images). Still to do: rotated pages and non-zero crop boxes (see `page_text_layer`), and test on real-world PDFs (multi-column, tables, Type 3 fonts, text split into one segment per word).
- [x] D5. Stop button (flag between steps plus `cancel_run`, which drops the request in progress) and a progress bar.

## E. Inpainting
- [x] E1. Classical diffusion fill implemented; needs fixture testing.
- [ ] E2. The stroke threshold (`> 60`) assumes a flat background. Make it adaptive for gradients, and reconsider the 45% fallback to a flat fill.
- [ ] E3. Optional: replace it with a LaMa ONNX model through `ort` for photos and textures. Keep the classical fill as the fallback.

## F. Quality
- [x] F1a. `wrap()` follows the Unicode line breaking rules (CJK breaks, no closing punctuation at a line start); overlong words are cut between grapheme clusters.
- [ ] F1b. Shaping and right-to-left layout (Arabic, Hebrew, Indic scripts, Thai marks) are still missing. Consider `cosmic-text`; until then those target languages are not offered.
- [ ] F2. Handle rotated or vertical text, and very small text (minimum readable size).
- [ ] F3. Skip translating blocks that are only numbers, URLs or codes.
- [ ] F4. Local models sometimes break the JSON-array reply (extra prose, reasoning tags). The parser (`extract_json_array`) now drops reasoning tags, fences and prose. Still open: constrained output (llama.cpp grammar or `response_format`).

## G. Translation endpoint
- [x] G1. Test connection button and status field added (`test_connection`, 5 s timeout, checks `GET {base}/models`).
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

## I. Speed and UI (steps 0-9 of implementations.md)
- [x] I1. `run_ocr` and `read_region` use the app-wide `EngineCache` (they used to build a new one per call, so the models were reloaded every time).
- [x] I2. One shared HTTP client, no timeout (use the Stop button for a stuck server).
- [x] I3. `fit_px` stops at 0.25 px and uses a glyph cache; `erase` uses 20 passes; erase and layout run in parallel (rayon).
- [x] I4. Box colour follows OCR confidence (red, orange, yellow, green); Compare view with a divider.
