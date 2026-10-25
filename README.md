# Layout Translator

Drop in an image, get it back with the text translated in the same layout. OCR is PaddleOCR (ONNX) in Rust; translation goes to any OpenAI-compatible endpoint (llama.cpp `llama-server`, or a hosted API with a key). Windows first, macOS via CI. See PLAN.md for the architecture.

## Run it (Windows)
1. Install Rust (rustup, MSVC), Visual Studio Build Tools "Desktop development with C++", Node LTS.
2. Edit `scripts/models.json`: fill in the three English model URLs (see `_help`). Then `npm install` and `npm run models` (also fetches pdfium and the font).
3. Create icons once: `npx tauri icon path\to\any-square.png`.
4. `llama-server -m your-model.gguf --port 8080`, then `npm run tauri dev`.

## Languages
The English model ships in `models/en` next to the app. To add one, create `models/<name>/` there with `det.onnx`, `rec.onnx`, `dict.txt` and restart. The Windows installer is per-user so that folder is writable.

## Status
Images and PDFs in and out, classical inpainting, engine cache. Uncompiled first draft: follow `TODO_FOR_LOCAL_AI.md` step by step (build fixes first, then models, then quality).
