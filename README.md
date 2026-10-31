# Layout Translator [work in progress]

Drop in an image, get it back with the text translated in the same layout. OCR is PaddleOCR (ONNX) in Rust; translation goes to a local llama.cpp `llama-server` (base URL only, no API key or model name needed). Windows first, macOS via CI. See PLAN.md for the architecture.

## Run it (Windows)
1. Install Rust (rustup, MSVC), Visual Studio Build Tools "Desktop development with C++", Node LTS.
2. Edit `scripts/models.json`: fill in the three English model URLs (see `_help`). Then `npm install` and `npm run models` (also fetches pdfium and the font).
3. Create icons once: `npx tauri icon path\to\any-square.png`.
4. `llama-server -m your-model.gguf --port 8080 -np 4 -c 16384`, then `npm run tauri dev`. `-np` is the number of requests the server handles at once; set the same number under "Parallel requests" in the app (the Test connection button reads it for you). The context size `-c` is shared between the slots, so use about 4096 per slot.

## Start the server from the app
Next to **Test connection** is the **Launch server** button. It runs a command that you set, for example the `llama-server` command from step 4 above.
1. Click the pencil icon next to **Launch server**.
2. Paste your command and click **Save**.
3. Click **Launch server**. A terminal window opens and runs the command.
4. Click **Test connection** when the model has loaded.

Close the terminal window to stop the server.

The command is a plain text file named `server-command.txt`. Any text editor can change it, and the **Show file** button opens its folder. The file is next to the program when that folder is writable (the Windows installer and the portable zip). Otherwise it is in the per-user app data folder. On macOS it is always there, because the app folder is inside the signed app.

The terminal depends on the system:
- Windows: `cmd.exe`. The command runs as a batch file, so write a percent sign as `%%`.
- macOS: Terminal.
- Linux: the program in the `TERMINAL` environment variable, or the first of `x-terminal-emulator`, `gnome-terminal`, `konsole`, `xfce4-terminal`, `mate-terminal`, `kitty`, `alacritty`, `lxterminal` and `xterm` that is installed.

The command runs with your user rights, exactly as you wrote it. Do not use a command that you do not trust.

## Languages
The English model ships in `models/en` next to the app. To add one, create `models/<name>/` there with `det.onnx`, `rec.onnx`, `dict.txt` and restart. The Windows installer is per-user so that folder is writable.

## Status
Builds with `cargo check` in `src-tauri`. Models (PP-OCRv5 English) and pdfium downloaded via `npm run models`. Icons generated. Follow `TODO_FOR_LOCAL_AI.md` step by step.

## If the app closes by itself
The release build has no console window. A crash writes its message and a backtrace to `last-error.log` in the app data folder (on Windows `%APPDATA%\com.example.layouttranslator\last-error.log`, on macOS `~/Library/Application Support/com.example.layouttranslator/last-error.log`). Attach that file to a bug report. It is only written when the app crashes. Look through it before sharing it: a panic message can quote text the app was working on.

## Builds and downloads
Every push to `master` runs the `release` workflow (pushes that only change `.md` files are skipped). It builds for Windows (x64) and macOS (Apple Silicon only; there is no Intel Mac build, see `TODO_FOR_LOCAL_AI.md` I2) and replaces the pre-release **Latest build (master)** (tag `nightly`) on the Releases page with:
- the installers: the `..._x64-setup.exe` (Windows) and the `.dmg` files (macOS);
- the portable versions: the zip files with `portable` in the name. Windows: unzip, run `Layout Translator.exe`, nothing is installed (needs the WebView2 runtime that Windows 11 and updated Windows 10 already have). macOS: unzip and open the app.

Pushing a tag such as `v0.1.0` puts the same files into a **draft** release for you to review and publish. All builds are unsigned (Windows: "More info > Run anyway"; macOS: right-click the app > Open). Settings > Actions > General > Workflow permissions must be "Read and write". A manual run (Actions > release > Run workflow) on another branch only keeps the files as workflow artifacts.
