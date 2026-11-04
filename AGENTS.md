# Rules for AI assistants working on this repo

This is a Tauri v2 app: Rust core in `src-tauri/src`, plain HTML/JS UI in `src/`.

- Read `PLAN.md` (architecture) and `TODO_FOR_LOCAL_AI.md` (work list) first. Work through the list in order and tick items off.
- Explain each change in one or two plain sentences before the code.
- Before changing anything, list every file you'll need and read them in one pass. Only then proceed with the edits.
- Work in small steps. After each step give the exact command to run and what success looks like. Say which shell it is for: PowerShell on Windows, sh on macOS and Linux.
- You favour concise, clean solutions over comprehensive yet complex solutions for risk of over-engineering. Use the STE100 skill for explanations.
- Never add a dependency without saying why and checking its latest version and licence.
- Keep logic in Rust modules with unit tests; Tauri commands stay thin wrappers.
- Never copy code from GPL/AGPL projects.
- Keep the `Block` struct (Rust) and the JSON the UI sends and receives in sync.
- The web view has a strict Content Security Policy (`app.security.csp` in `tauri.conf.json`). In `src/index.html` never add `style="..."` attributes, inline event handlers (`onclick="..."`), `eval`, or resources from the internet. Use CSS classes, `element.onclick = ...` and `element.style.property = ...`. Run `npm run check:csp` after you edit the page or the policy.
- Fix compile errors at the root cause; do not silence them with `unwrap` or `allow`.
- Translation is a local llama.cpp server only (base URL, no API key or model name). Do not add Google Translate or hosted APIs.
- Run `npm run check:csp` after you change `src/index.html`, and `npm run check:version` after you touch a version. Change the app version only with `npm run version:set -- x.y.z`: it is in six places that must agree.
- `ci.yml` runs `cargo test` and `cargo clippy --all-targets -- -D warnings` on every push and pull request. Run both in `src-tauri` before you push.
- Write every change that a user can see in `CHANGELOG.md` (under "Unreleased").
- A Tauri command returns `Result<_, AppError>` (`error.rs`): code below it returns `String` errors, and `?` converts them. Name a kind (`AppError::server(..)`, `AppError::no_model(..)`) only where the UI does something different for it, and in the UI show an error with `errText(e)`. Slow work (OCR, rendering, PDF, big images) goes through `blocking(|| ...)`.
- Test detection, grouping and reading a region with `FakeDetector` (`pipeline/testutil.rs`), not with the OCR models. Only `ocr_lines` may call the OCR crates.
