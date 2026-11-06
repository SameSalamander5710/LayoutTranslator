
#TODO: Temporary text snippets moved away from README.md, need to be organised.

## Run it (Windows)
1. Install Rust (rustup, MSVC), Visual Studio Build Tools "Desktop development with C++", Node LTS.
2. Edit `scripts/models.json`: fill in the three English model URLs (see `_help`). Then `npm install` and `npm run models` (also fetches pdfium and the font).
3. Create icons once: `npx tauri icon path\to\any-square.png`.
4. `llama-server -m your-model.gguf --port 9931 -np 4 -c 16384`, then `npm run tauri dev`. `-np` is the number of requests the server handles at once; set the same number under "Parallel requests" in the app (the Test connection button reads it for you). The context size `-c` is shared between the slots, so use about 4096 per slot.


## Status
Builds with `cargo check` in `src-tauri`. Models (PP-OCRv5 English) and pdfium downloaded via `npm run models`. Icons generated. Follow `TODO_FOR_LOCAL_AI.md` step by step.


## Builds and downloads
Every push to `master` runs the `release` workflow (pushes that only change `.md` files are skipped). It builds for Windows (x64), macOS (Apple Silicon only; there is no Intel Mac build, see `TODO_FOR_LOCAL_AI.md` I2) and Linux (x86_64, built on Ubuntu 24.04; needs glibc 2.38 or newer) and replaces the pre-release **Latest build (master)** (tag `nightly`) on the Releases page with:
- the installers: the `..._x64-setup.exe` (Windows), the `.dmg` files (macOS) and the `.deb`, `.rpm` and `.AppImage` files (Linux);
- the portable versions: the zip files with `portable` in the name. Windows: unzip, run `Layout Translator.exe`, nothing is installed (needs the WebView2 runtime that Windows 11 and updated Windows 10 already have). macOS: unzip and open the app. Linux: unpack the `.tar.gz` and run `bin/layout-translator` (needs glibc 2.38 or newer, GTK 3 and WebKitGTK 4.1: Ubuntu 24.04+, Debian 13+ or Fedora 39+); the `.AppImage` is a single file: `chmod +x` it and run it.

Pushing a tag such as `v0.1.0` puts the same files into a **draft** release for you to review and publish. All builds are unsigned (Windows: "More info > Run anyway"; macOS: right-click the app > Open). Settings > Actions > General > Workflow permissions must be "Read and write". A manual run (Actions > release > Run workflow) on another branch only keeps the files as workflow artifacts.


To check/build it yourself: 
Quick dev test (fastest)

- npm install
- npm run models
- npm run tauri dev

Build a release installer locally

- npm run tauri build

Installers will be output to src-tauri/target/release/bundle/ (e.g. nsis/ for Windows .exe, dmg/ for macOS). Unsigned builds will trigger OS warnings (SmartScreen/Gatekeeper). 

Via CI (GitHub Actions)

There's .github/workflows/release.yml using tauri-apps/tauri-action. You can trigger it with workflow_dispatch or push a tag to create a draft release on GitHub's Releases page. 

You'll need Rust (MSVC), Node LTS, and VS Build Tools installed. npm run models must be run first to fetch fonts/models/pdfium.