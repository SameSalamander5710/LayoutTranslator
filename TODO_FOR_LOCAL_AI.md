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
- [ ] D3. Output PDFs hold one JPEG (quality 90) per page, so the text is not selectable. Still open: an optional text-layer mode.
- [x] D4. Hybrid run: selectable PDF text is read with pdfium first, OCR only covers the rest (and is skipped on pages with no images). Still to do: rotated pages and non-zero crop boxes (see `page_text_layer`), and test on real-world PDFs (multi-column, tables, Type 3 fonts, text split into one segment per word).
- [x] D5. Stop button (flag between steps plus `cancel_run`, which drops the request in progress) and a progress bar.

## E. Inpainting
- [x] E1. Classical diffusion fill implemented; needs fixture testing.
- [x] E2. The stroke threshold follows the text contrast (24 to 60 instead of a fixed 60), and the background colour is a tilted plane fitted to the non-text pixels, so gradients work. When too much looks like text (over 45%), the box is painted with that smooth background instead of a flat colour.
- [ ] E3. Optional: replace it with a LaMa ONNX model through `ort` for photos and textures. Keep the classical fill as the fallback.
- [x] E4. When over 45% of a box looks like text (photo, pattern, two backgrounds), the box is no longer painted with one flat colour. It is rebuilt from the pixels around it (push-pull pyramid fill on the residual of the background plane, then a light blur), so the area looks like frosted glass. See `frosted_patch`, `FROST_COVER`, `PULL_TRUST`. Tune `FROST_COVER` on real pages.

## F. Quality
- [x] F1a. `wrap()` follows the Unicode line breaking rules (CJK breaks, no closing punctuation at a line start); overlong words are cut between grapheme clusters.
- [ ] F1b. Shaping and right-to-left layout (Arabic, Hebrew, Indic scripts, Thai marks) are still missing. Consider `cosmic-text`; until then those target languages are not offered.
- [ ] F2. Handle rotated or vertical text, and very small text (minimum readable size).
- [x] F3. Blocks that are only numbers, URLs, e-mail addresses or codes are not sent to the model (`is_translatable`); a block the user retranslates by hand is always sent.
- [ ] F4. Local models sometimes break the JSON-array reply (extra prose, reasoning tags). The parser (`extract_json_array`) now drops reasoning tags, fences and prose. Still open: constrained output (llama.cpp grammar or `response_format`).

## G. Translation endpoint
- [x] G1. Test connection button and status field added (`test_connection`, 5 s timeout, checks `GET {base}/models`).
- [ ] G3. `"csp": null` in `tauri.conf.json` is a development setting. Write a strict CSP for release.
- [x] G4. Large pages are split into batches (`BATCH_MAX_BLOCKS`, `BATCH_MAX_CHARS`), and a chunk that fails is retried without the page context and then one block at a time.

## H. Tests
- [x] H1. Unit tests cover `group()`, `wrap()`, `erase_patch()` and `write_pdf()` (the test checks the PDF structure: header, `startxref`, every xref entry, page sizes, one JPEG per page). A PDF made by `write_pdf` was also accepted by qpdf, pdfinfo and pypdf. Still open: reading a `write_pdf` file back with pdfium.
- [ ] H2. One end-to-end test per fixture: image in, expected block text out (allow small OCR differences).

## I. Release
- [x] I1. Both lock files are committed. The workflow uses `npm ci` and `cargo ... --locked`, so a release cannot pick up newer crates. Commit `src-tauri/Cargo.lock` whenever `Cargo.toml` changes, or the release build fails.
- [ ] I2. The Intel runner label is now `macos-15-intel` (`macos-13` was retired on 2025-12-04). Still to do: run the workflow once with `workflow_dispatch` and fix failures. GitHub drops Intel macOS after the macOS 15 image is retired in Fall 2027.
- [ ] I3. macOS: confirm pdfium and ONNX Runtime dylibs load from inside the `.app`, and document the unsigned-app workaround (or set up signing/notarization).
- [ ] I4. Windows: document the SmartScreen warning, or set up code signing.
- [x] I5. `PLAN.md` was updated to match the code (Google Translate dropped). Keep it in sync as things change.

## J. Speed and UI (steps 0-9 of implementations.md)
- [x] J1. `run_ocr` and `read_region` use the app-wide `EngineCache` (they used to build a new one per call, so the models were reloaded every time).
- [x] J2. One shared HTTP client, no timeout (use the Stop button for a stuck server).
- [x] J3. `fit_px` stops at 0.25 px and uses a glyph cache; `erase` uses 20 passes; erase and layout run in parallel (rayon).
- [x] J4. Box colour follows OCR confidence (red, orange, yellow, green); Compare view with a divider.
