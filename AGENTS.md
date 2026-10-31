# Rules for AI assistants working on this repo

This is a Tauri v2 app: Rust core in `src-tauri/src`, plain HTML/JS UI in `src/`.

- Read `PLAN.md` (architecture) and `TODO_FOR_LOCAL_AI.md` (work list) first. Work through the list in order and tick items off.
- Explain each change in one or two plain sentences before the code.
- Before changing anything, list every file you'll need and read them in one pass. Only then proceed with the edits.
- Work in small steps. After each step give the exact command to run (PowerShell) and what success looks like.
- You favour concise, clean solutions over comprehensive yet complex solutions for risk of over-engineering. Use the STE100 skill for explanations.
- Never add a dependency without saying why and checking its latest version and licence.
- Keep logic in Rust modules with unit tests; Tauri commands stay thin wrappers.
- Never copy code from GPL/AGPL projects.
- Keep the `Block` struct (Rust) and the JSON the UI sends and receives in sync.
- The web view has a strict Content Security Policy (`app.security.csp` in `tauri.conf.json`). In `src/index.html` never add `style="..."` attributes, inline event handlers (`onclick="..."`), `eval`, or resources from the internet. Use CSS classes, `element.onclick = ...` and `element.style.property = ...`. Run `npm run check:csp` after you edit the page or the policy.
- Fix compile errors at the root cause; do not silence them with `unwrap` or `allow`.
- Translation is a local llama.cpp server only (base URL, no API key or model name). Do not add Google Translate or hosted APIs.
