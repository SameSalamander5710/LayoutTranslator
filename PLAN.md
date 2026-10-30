# LayoutTranslator – plan

A free, local-first desktop app. Drop in an image or PDF, get back the same page with the text translated and the layout kept. Rust core, Tauri UI, Windows first, macOS second, downloadable from GitHub Releases with nothing else to install.

Status: compiles with updated deps (oar-ocr 0.10). English PP-OCRv5 models downloaded, pdfium bundled via script. Icons generated. PaddleOCR via `oar-ocr` with English model bundled and extra languages added as folders; **llama.cpp / any OpenAI-compatible endpoint is the only translation engine (Google Translate was dropped)**; PDF in and out, a classical inpainting fill and an OCR engine cache are implemented in first form. Open items live in `TODO_FOR_LOCAL_AI.md`. Items marked **(verify)** need checking before you depend on them.

---

## 1. What the reference projects teach

| Project | Stack | Useful idea | Watch out for |
|---|---|---|---|
| RustLens | Tauri + Rust, `ocrs`, Ollama, SQLite | Proof that Tauri + Rust OCR + local LLM works on Windows. Models are downloaded by a script with SHA-256 checks and bundled into the NSIS installer. | Screen-region tool, not a layout tool. `ocrs` is Latin-only (see below). MIT. |
| manga-image-translator | Python, PyTorch | The canonical pipeline: detect text, OCR, translate, **inpaint** (erase) the old text, **typeset** the new text. Config-driven, glossary and pre/post replacement dictionaries. | Python plus multi-GB models. **GPL-3.0**: read it, don't copy code unless your app will be GPL. Its README lists Google as "temporarily disabled". |
| pdf2zh-app | Electron, DocLayout-YOLO (ONNX) | Layout-aware PDF translation, Google (free) or any OpenAI-compatible endpoint, page picker. Apache-2.0. | Electron, not Rust. |
| loctran | Python, Tesseract, Ollama | Smart split: PDFs with a real text layer skip OCR; scans get OCR (normal + inverted image to catch light-on-dark). Output is HTML overlay. | **AGPL-3.0** (dual-licensed). Needs Python, Tesseract and Ollama installed. |

Take the pipeline shape and the ideas. Write your own code.

## 2. Pipeline

```
input file
 ├─ image ───────────────────────────────┐
 └─ PDF → per page:                      │
      has text layer? ── yes → read text runs + bboxes with pdfium (no OCR)
                      └─ no  → rasterise at ~200 DPI ─┘
                                  ↓
                  text detection + recognition (OCR)  → lines with boxes + confidence
                                  ↓
                  group lines into paragraph blocks   → TextBlock[]
                                  ↓
                  translate (batched, 1:1 by id)      → block.translated
                                  ↓
                  erase original text                 → clean background
                                  ↓
                  typeset translation into block box  → shrink-to-fit
                                  ↓
                  export PNG / PDF
```

**Data contract** (Rust `struct` with serde, same JSON the UI uses):

```
TextBlock { id, page, bbox:[x,y,w,h], text, translated?, conf, fontPx, weight, fg, bg, angle? }
```

## 3. Technology decisions

### OCR (the hardest choice)
- `ocrs` (pure Rust) is **Latin alphabet only** and describes itself as early preview. Fine for a first spike, not for a general translator.
- **Recommended: PaddleOCR (PP-OCR) models run through ONNX Runtime** using the `ort` crate. The `oar-ocr` crate wraps exactly this (ort + PaddleOCR models) and also has layout analysis. A MNN-based alternative, `rust-paddle-ocr`, lists 10+ scripts including Chinese, Japanese, Korean, Arabic, Cyrillic and Thai. Start with `oar-ocr`; if it is awkward, call `ort` directly.
- **Phase-2 shortcut (verify):** Windows has a built-in OCR API (`Windows.Media.Ocr`, reachable from the `windows` crate) and macOS has Vision. Zero model download, word boxes included, but quality and languages depend on what the OS has installed. Good as an optional "fast" engine behind the same trait.
- Tesseract is possible but means shipping C++ libs and language data; skip.

### PDF (implemented, first draft)
- In: `pdfium-render` binds to a PDFium library shipped in `resources/pdfium/` (downloaded by `npm run models` from `bblanchon/pdfium-binaries`). Each page is rendered to a PNG at 200 DPI and then treated like an image. Text-layer PDFs are still OCR'd (shortcut not built yet).
- Out: a small hand-written PDF writer embeds one JPEG per page. Layout is exact, text is not selectable, files can be large.

### Erase and typeset (implemented, first draft)
- Erase: text strokes are masked and filled by diffusing the surrounding colours inward (classical inpainting, no model). Falls back to a flat background fill when the mask covers too much of the box. Optional later upgrade: LaMa ONNX for photos and textures.
- Typeset: one bundled font (Noto Sans) drawn with `ab_glyph`/`imageproc`, shrink-to-fit inside the original box. Known gaps: no text shaping or right-to-left layout, so Arabic, Hebrew, Persian, Urdu, Hindi, Bengali, Tamil and Thai are not offered as target languages.

### Translation (implemented, first draft)
One engine: any OpenAI-compatible `/v1/chat/completions` endpoint, meaning a local llama.cpp `llama-server` (base URL only; no API key or model name; a Test connection button checks `/models`). The prompt sends a JSON array of block texts and requires a JSON array back; if the count differs, blocks are retried one by one. Google Translate's unofficial endpoint was dropped (undocumented, rate-limited, grey area). Later: bundle `llama-server` as a Tauri sidecar so users need nothing running; add an engine trait if more providers are wanted.

### Models and size
Ship small default OCR models in the installer (tens of MB). Extra languages and the LLM are downloaded on demand into the app data directory with SHA-256 verification (RustLens already does this pattern).

## 4. Repository layout (as built)

```
├─ package.json, README.md, PLAN.md, TODO_FOR_LOCAL_AI.md, AGENTS.md
├─ src/index.html              the real UI (needs Tauri)
├─ scripts/{fetch-models.mjs, models.json}   fonts, OCR models, PDFium
├─ src-tauri/
│  ├─ Cargo.toml, tauri.conf.json, capabilities/default.json
│  ├─ resources/{models/en, fonts, pdfium}   bundled into the installer
│  └─ src/
│     ├─ main.rs        Tauri commands, model-folder lookup
│     └─ pipeline.rs    OCR + cache, grouping, colours, translate, inpaint, typeset, PDF in/out
└─ .github/workflows/release.yml
```
Later, split `pipeline.rs` into modules (`ocr`, `translate`, `render`, `pdf`) with unit tests.

## 5. Phases

1. **Done (draft):** image and PDF in, OCR, translate, erase, typeset, PNG/PDF out, per-language model folders, engine cache.
2. **Next:** make it compile and run (see `TODO_FOR_LOCAL_AI.md`, sections A and B), then quality (sections E and F).
3. **Then:** text-layer PDF shortcut, cancel/progress, tests, CI release on Windows and macOS.
4. **Later:** LaMa inpainting, layout detection, draggable-box editor, bundled llama-server, native OS OCR (Windows.Media.Ocr / macOS Vision) as an optional engine.

## 6. Distribution

- GitHub Actions with `tauri-apps/tauri-action` builds on `windows-latest` and `macos-latest` runners and attaches the installers to a draft release (workflow included). You can build the Mac version without owning a Mac.
- **Unsigned apps show warnings.** Windows SmartScreen warns on unsigned installers; macOS Gatekeeper blocks unsigned/un-notarized apps unless the user right-clicks → Open. Proper macOS distribution needs a paid Apple Developer account for notarization **(verify current price)**. For a hobby release, document the workaround in the README.
- ONNX Runtime must ship next to the app (a DLL on Windows, a dylib on macOS). This is the most likely "works on my machine" trap. Test the installer on a clean machine **(verify how `ort` links on each platform)**.
- macOS: Apple Silicon only for now. An Intel (or universal) build is blocked because no prebuilt ONNX Runtime exists for Intel macOS any more (see TODO_FOR_LOCAL_AI.md I2).

## 7. Honest expectations and risks

- "DeepL quality" layout preservation is hard in general. Expect good results on flat documents, forms, slides, screenshots, signs; weaker on colourful backgrounds, curved or vertical text, dense tables, and mixed fonts.
- Small LLMs sometimes drop, merge or reorder lines. The id-keyed JSON contract and per-block retry are there for this reason.
- OCR errors propagate into translation. Show detected text next to the translation and let the user edit (the UI already does).
- CJK needs separate fonts and per-script OCR models; plan binary size accordingly.
- Licensing: model weights keep their own licences. Keep a `THIRD_PARTY.md`, as RustLens does.

## 8. Rules for local AI

See `AGENTS.md`.

Suggested first prompts:
1. "Read AGENTS.md, PLAN.md and TODO_FOR_LOCAL_AI.md. Walk me through section A on Windows, one step at a time."
2. "Do B1 and B2: help me find the English PP-OCRv5 ONNX model files and fill scripts/models.json."
3. "Do E1: build a fixture set and show me how the inpainting looks on each."
