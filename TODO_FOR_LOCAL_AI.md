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
- [x] E4. When more than 60% of a box differs clearly from the page colour behind it (a noisy photo, a pattern), the box is not painted with one flat colour. It is rebuilt from the pixels around it (push-pull pyramid fill on the residual of the background plane, then a light blur), so the area looks like frosted glass. See `frosted_fill`, `FROST_COVER`, `PULL_TRUST`. Tune `FROST_COVER` on real pages.

## F. Quality
- [x] F1a. `wrap()` follows the Unicode line breaking rules (CJK breaks, no closing punctuation at a line start); overlong words are cut between grapheme clusters.
- [ ] F1b. Shaping and right-to-left layout (Arabic, Hebrew, Indic scripts, Thai marks) are still missing. Consider `cosmic-text`; until then those target languages are not offered.
- [ ] F2. Handle rotated or vertical text, and very small text (minimum readable size).
- [x] F3. Blocks that are only numbers, URLs, e-mail addresses or codes are not sent to the model (`is_translatable`); a block the user retranslates by hand is always sent.
- [ ] F4. Local models sometimes break the JSON-array reply (extra prose, reasoning tags). The parser (`extract_json_array`) now drops reasoning tags, fences and prose. Still open: constrained output (llama.cpp grammar or `response_format`).

## G. Translation endpoint
- [x] G1. Test connection button and status field added (`test_connection`, 5 s timeout, checks `GET {base}/models`).
- [x] G3. `tauri.conf.json` sets a strict Content Security Policy: `default-src 'none'`; `script-src` and `style-src` only `'self'` (Tauri adds a hash for the inline script and a nonce for the inline style by itself); `img-src 'self' data:` (pages travel as data URLs); `connect-src ipc: http://ipc.localhost` (Tauri IPC only; the app never calls the network from the web view, `reqwest` does that in Rust); `base-uri` and `form-action` `'none'`. No `unsafe-inline`, no `unsafe-eval`. `index.html` has no `style=""` attributes (classes instead; helper classes are at the end of the stylesheet) and no inline event handlers. `npm run check:csp` checks both files and runs in CI. Checked in Chromium with the policy as Tauri sends it: the whole UI works with no violation, and inline handlers, `eval`, remote scripts, images and fetches, iframes and `style` attributes are blocked. Still to do: a look at the DevTools console in a real Windows and macOS build (WebView2 and WKWebView).
- [x] G4. Large pages are split into batches (`BATCH_MAX_BLOCKS`, `BATCH_MAX_CHARS`), and a chunk that fails is retried without the page context and then one block at a time.

## H. Tests
- [x] H1. Unit tests cover `group()`, `wrap()`, `erase_patch()` and `write_pdf()` (the test checks the PDF structure: header, `startxref`, every xref entry, page sizes, one JPEG per page). A PDF made by `write_pdf` was also accepted by qpdf, pdfinfo and pypdf. Still open: reading a `write_pdf` file back with pdfium.
- [ ] H2. One end-to-end test per fixture: image in, expected block text out (allow small OCR differences).

## I. Release
- [x] I1. Both lock files are committed. The workflow uses `npm ci` and `cargo ... --locked`, so a release cannot pick up newer crates. Commit `src-tauri/Cargo.lock` whenever `Cargo.toml` changes, or the release build fails.
- [ ] I2. The workflow builds Windows x64 and macOS Apple Silicon. The Intel Mac build was removed: `ort-sys` 2.0.0-rc.13 (pinned exactly by `oar-ocr-core` 0.10.1) has no prebuilt ONNX Runtime for `x86_64-apple-darwin`, and Microsoft's last Intel macOS build is 1.23.2 while `ort` rc.13 needs 1.28. To restore it: build ONNX Runtime 1.28 from source on a `macos-15-intel` runner (cache it), point the build at it with `ORT_LIB_LOCATION`, and add the matrix entry back (`suffix: macos-x64`, `target: x86_64-apple-darwin`). Or wait for an `ort`/`oar-ocr` release that supports Intel again. Still to do: run the workflow once with `workflow_dispatch` and fix failures.
- [ ] I3. macOS: confirm pdfium and ONNX Runtime dylibs load from inside the `.app`, and document the unsigned-app workaround (or set up signing/notarization).
- [ ] I4. Windows: document the SmartScreen warning, or set up code signing.
- [x] I5. `PLAN.md` was updated to match the code (Google Translate dropped). Keep it in sync as things change.
- [ ] I6. Linux x86_64 is built on Ubuntu 22.04 (oldest runner Tauri v2 supports, for the glibc): `.deb`, `.rpm`, `.AppImage` and a portable `.tar.gz` (`bin/` + `lib/Layout Translator/`, the place Tauri looks in on Linux). The build was run once in a sandbox. Still to confirm on a real GitHub run: ONNX Runtime is linked in statically (the workflow fails on unresolved libraries if not), and the app starts on a clean desktop. No Linux ARM build yet (needs a pinned pdfium arm64 asset and an ARM runner).

## J. Speed and UI (steps 0-9 of implementations.md)
- [x] J1. `run_ocr` and `read_region` use the app-wide `EngineCache` (they used to build a new one per call, so the models were reloaded every time).
- [x] J2. One shared HTTP client, no timeout (use the Stop button for a stuck server).
- [x] J3. `fit_px` stops at 0.25 px and uses a glyph cache; `erase` uses 20 passes; erase and layout run in parallel (rayon).
- [x] J4. Box colour follows OCR confidence (red, orange, yellow, green); Compare view with a divider.

## K. Editing, erase and colour
- [x] K1. Text colour (`text_colour`): every pixel is compared with the page colour right behind it (median of a window, `PageColours`) instead of the ring around the box, and the commonest colour among the clearest text pixels is taken. White lettering on a red stripe used to come out red. The old ring rule stays as the fallback.
- [x] K2. Blocks whose crops overlap or touch are erased as one group (`erase_groups`, `erase_group_with`) with one fill, so a blurred fill is never pasted over another one. Other blocks' boxes are never used as source colours. The text mask follows the page colour behind each pixel (text-coloured pixels are left out of the median), the busy test uses the raw mask (`FROST_COVER` 0.6), and dense masks (`DENSE_COVER` 0.5) use the unblurred push-pull fill. Before, the dilated mask made nearly every text line count as busy.
- [x] K3. Detection runs by itself when a file is opened or a page is shown for the first time (no Detect button). Run all skips pages that are already detected, so box edits are kept. The circular-arrow button resets the detection of the page (Undo brings the old boxes back). The box toolbar has icons for add, delete, undo, redo (the last 10 changes per page; Ctrl+Z, Ctrl+Y) and reset.
- [x] K4. Paste an image from the clipboard (button, or Ctrl+V / Cmd+V outside text fields). `paste_image` reads it with `arboard` 3.7 (MIT OR Apache-2.0, `image-data` only), `save_clipboard_png` flattens transparency on white and stores a PNG in the app cache, and the UI opens it like a dropped file.
- [x] K5. A detected page (first detection, Reset) gets its boxes cleaned (`refine`): OCR boxes are shrunk from the top and the bottom to the rows of their own text (`tighten`; text-coloured pixels that run out of the box, like the light margin of a poster, are not text; PDF text-layer boxes are left alone; the width is never changed, so no text of a second colour can be cut off). The typeset text is as big as fits the box, so a box of the height of the text gives text of the original size and a smaller area to erase. Horizontal tightening is not done: on multi-coloured backgrounds it cannot be told apart from text of another colour reliably.
- [x] K6. After that, a block with at least 90% of its area inside another one is merged into it (`merge_nested`, `NESTED_FRAC`): the bigger box is widened to cover it, it keeps its text, and chains collapse. Text-layer boxes are merged too. Boxes the user draws, moves or resizes do not pass through here, so they stay as they were set.

## L. Server launcher
- [x] L1. A **Launch server** button and a pencil icon next to **Test connection**. The pencil opens a text box for a command (Save, Cancel, Show file). The command is a plain text file, `server-command.txt`: next to the program if that folder is writable, else in the per-user app data folder (always there on macOS, to keep the signed `.app` unchanged). `launcher.rs` holds the logic and its tests: the command goes into a temporary script (`.bat`, `.command` or `.sh`, so that `&&` and `|` are read once) that runs in a new, titled terminal window that stays open. Windows: `cmd /C start /WAIT` (the `.bat` is removed when the window closes), macOS: `open -a Terminal`, Linux: `$TERMINAL`, then the list `TERMINALS`. Unix scripts delete themselves, and `sweep_old_scripts` removes leftovers at start. The commands `get_server_command`, `save_server_command`, `launch_server` (no argument: it only starts the saved command) and `open_server_command_folder` are thin wrappers. `open_models_dir` now uses `open_folder`, which also works on Linux (`xdg-open`).
- [ ] L2. Test L1 by hand on Windows (cmd, a path with spaces, `&&`, a window that is closed by hand), on macOS (Terminal) and on a real Linux desktop. The automated tests ran only on Linux with a fake terminal program, and the Windows code path was only compiled by reading the `std` API, not run.
- [ ] L3. Optional: after Launch server, call `test_connection` every few seconds until the server answers, and show the result in the status field. Optional: let the user choose PowerShell on Windows.

## M. Maintenance (from the architecture review; do in this order)
- [x] M1. The 19 clippy warnings are fixed. `ci.yml` runs the tests and clippy (`-D warnings`) on every push and pull request; the Rust version is pinned in `rust-toolchain.toml` (used by `ci.yml` and `release.yml`); Dependabot proposes updates weekly. There is no format check: the code uses a compact hand-written style that rustfmt would rewrite.
- [x] M2. One command changes the version everywhere (`npm run version:set -- x.y.z`) and CI checks that it is the same in six places. `PLAN.md` describes the real `Block`. `AGENTS.md` is shell-neutral. `CHANGELOG.md` records what users can see.
- [x] M3. `pipeline.rs` (2,464 lines) is now `pipeline/` with `ocr`, `pdf_in`, `boxes`, `colour`, `translate`, `typeset`, `erase`, `imageio` and `pdf_out`, a `mod.rs` that re-exports the public functions (so `main.rs` and the example are unchanged), and `testutil.rs` for test helpers shared by several modules. Code was only moved: the only lines that changed are visibility (`pub(super)`), module headers and test wrappers. The largest file is `translate.rs` (545 lines). All 70 pipeline tests moved with the code they test; the total is still 84. Next step in this direction: explicit `use` lines instead of `use super::*;` in each module, once the split has settled.
- [ ] M4. Make `main.rs` thin: a `detect_page` function in the core for what `run_ocr` does after detection; one function that renders with a font and falls back to Noto Sans (now copied in `render_page` and `export_pdf`); choose the font per language from data (`models/<lang>/lang.json`), not from a `match` in Rust plus a second table in the UI.
- [ ] M5. Typed errors at the command boundary (`kind` + `message`), so the UI does not match on text. Replace `Mutex::unwrap()` in `main.rs` with the poison-tolerant helper.
- [ ] M6. Run `render_page`, `export_pdf`, `open_document`, `preview` and `sample_colors` through `spawn_blocking` (one helper), like `run_ocr` does.
- [ ] M7. A one-method `TextDetector` trait with the `oar-ocr` implementation and a fake, so detection (OCR, group, refine) can be tested without model files and the `ort` dependency has one place.
- [ ] M8. Golden-image tests: fixtures (poster, plain scan, light text on dark) through erase and render, compared with stored PNGs within a tolerance. List every tuning constant in one table in `PLAN.md`.
- [ ] M9. Move the pure core into a `core` crate (Cargo workspace, no `tauri`), so `cargo test -p core` needs no GTK or WebKit.
- [ ] M10. Split `index.html` into `styles.css` and ES modules (no bundler; the CSP allows `script-src 'self'`); extend `check-csp.mjs` to every file.
- [ ] M11. Decide where the editing rules live (merge, re-read, colour refresh): one Rust command `apply_edit`, or pure JS functions. Test them with `node --test`.
- [ ] M12. A contract test for `Block` (Rust writes a sample as JSON, a JS test reads it) and JSDoc types with `// @ts-check`. Remove the unused `fontPx` / `font_px`.
- [ ] M13. Measure the time to encode, send and decode a full A4 page at 200 DPI. If it is the bottleneck: binary IPC instead of base64 PNG, and a cache of the last two decoded originals.
- [ ] M14. `LICENSE`, `deny.toml` (licences and advisories) in CI, and a generated `THIRD_PARTY.md` shipped with the app. The project licence is the owner's decision.
- [ ] M15. Make `examples/test_pipeline.rs` build without `tauri` (plain tokio), or turn it into an `#[ignore]` integration test.
- [ ] M16. Release workflow: fail when the tag differs from the version; release notes from `CHANGELOG.md`.
- [ ] M17. Decision records in `docs/adr/`: why `oar-ocr`, what to do if `ort` stalls, why translation is local only, why the erase is classical.
- [ ] M18. Reduce `capabilities/default.json` (`dialog:default`) to the permissions the app uses.
