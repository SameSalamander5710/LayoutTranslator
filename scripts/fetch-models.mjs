// Downloads the bundled font and the default (English) OCR model into src-tauri/resources. Run: npm run models
import { mkdirSync, existsSync, writeFileSync } from "node:fs";
import { readFileSync } from "node:fs";
const cfg = JSON.parse(readFileSync(new URL("./models.json", import.meta.url)));
const root = new URL("../src-tauri/resources/", import.meta.url);
async function get(url, dest) {
  if (existsSync(dest)) return console.log("have", dest.pathname);
  if (!url) throw new Error(`No URL set for ${dest.pathname}. Edit scripts/models.json (see _help).`);
  const r = await fetch(url); if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
  writeFileSync(dest, Buffer.from(await r.arrayBuffer())); console.log("downloaded", dest.pathname);
}
for (const [f, u] of Object.entries(cfg.fonts)) { mkdirSync(new URL("fonts/", root), { recursive: true }); await get(u, new URL("fonts/" + f, root)); }
for (const [lang, files] of Object.entries(cfg.models)) {
  mkdirSync(new URL(`models/${lang}/`, root), { recursive: true });
  for (const [f, u] of Object.entries(files)) await get(u, new URL(`models/${lang}/${f}`, root));
}

// PDFium (from bblanchon/pdfium-binaries) for the current OS. TODO(local AI): verify archive layout and asset names.
import { execFileSync } from "node:child_process";
import { copyFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
const plat = { win32: { x64: "win-x64" }, darwin: { arm64: "mac-arm64", x64: "mac-x64" } }[process.platform]?.[process.arch];
if (!plat) throw new Error(`No pdfium build mapped for ${process.platform}/${process.arch}`);
const lib = process.platform === "win32" ? "bin/pdfium.dll" : "lib/libpdfium.dylib";
const dest = fileURLToPath(new URL("pdfium/", root));
mkdirSync(dest, { recursive: true });
if (!existsSync(dest + lib.split("/").pop())) {
  const tmp = `${tmpdir()}/pdfium-${plat}`; rmSync(tmp, { recursive: true, force: true }); mkdirSync(tmp, { recursive: true });
  const r = await fetch(`https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-${plat}.tgz`);
  if (!r.ok) throw new Error("pdfium download failed: HTTP " + r.status);
  writeFileSync(`${tmp}/p.tgz`, Buffer.from(await r.arrayBuffer()));
  execFileSync("tar", ["-xzf", `${tmp}/p.tgz`, "-C", tmp]);
  copyFileSync(`${tmp}/${lib}`, dest + lib.split("/").pop()); console.log("pdfium ready");
}
