// Downloads the bundled fonts, the default (English) OCR model and pdfium into src-tauri/resources. Run: npm run models
// Every download goes to "<name>.part" first and is renamed only when it is complete (and matches its checksum in
// models.json), so an interrupted run never leaves a broken file under the real name. Failed downloads are retried.
import { mkdirSync, existsSync, readFileSync, writeFileSync, renameSync, rmSync, statSync, copyFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";

const cfg = JSON.parse(readFileSync(new URL("./models.json", import.meta.url)));
const root = fileURLToPath(new URL("../src-tauri/resources/", import.meta.url));
const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const TRIES = 3, WAIT_MS = 2000;

// A file counts as present when it is not empty and, if a checksum is known, matches it.
const good = (path, sha) => existsSync(path) && statSync(path).size > 0 && (!sha || sha256(readFileSync(path)) === sha);

async function download(url, dest, sha) {
  let last;
  for (let attempt = 1; attempt <= TRIES; attempt++) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const buf = Buffer.from(await r.arrayBuffer());
      if (buf.length === 0) throw new Error("empty download");
      const got = sha256(buf);
      if (sha && got !== sha) throw new Error(`checksum mismatch (got ${got})`);
      writeFileSync(dest + ".part", buf);
      renameSync(dest + ".part", dest);
      return got;
    } catch (e) {
      last = e;
      rmSync(dest + ".part", { force: true });
      if (attempt < TRIES) { console.log(`  attempt ${attempt} failed (${e.message}); trying again in ${WAIT_MS / 1000} s`); await sleep(WAIT_MS); }
    }
  }
  throw new Error(`${url}: ${last.message}`);
}

async function get(url, rel) {
  const dest = root + rel, sha = cfg.sha256?.[rel];
  rmSync(dest + ".part", { force: true }); // left over from an interrupted run
  if (good(dest, sha)) return console.log("have", rel);
  if (!url) throw new Error(`No URL set for ${rel}. Edit scripts/models.json (see _help).`);
  console.log("downloading", rel, "<-", url);
  const got = await download(url, dest, sha);
  if (!sha) console.log(`  no checksum in models.json for ${rel}; this download is ${got}`);
}

mkdirSync(root + "fonts/", { recursive: true });
for (const [f, u] of Object.entries(cfg.fonts)) await get(u, "fonts/" + f);
for (const [lang, files] of Object.entries(cfg.models)) {
  mkdirSync(`${root}models/${lang}/`, { recursive: true });
  for (const [f, u] of Object.entries(files)) await get(u, `models/${lang}/${f}`);
}

// PDFium (from bblanchon/pdfium-binaries), pinned to the release in models.json.
const plat = { win32: { x64: "win-x64" }, darwin: { arm64: "mac-arm64", x64: "mac-x64" }, linux: { x64: "linux-x64" } }[process.platform]?.[process.arch];
if (!plat) throw new Error(`No pdfium build mapped for ${process.platform}/${process.arch}`);
const tag = cfg.pdfium?.tag;
if (!tag) throw new Error("scripts/models.json needs pdfium.tag, for example \"chromium/7543\"");
const lib = { win32: "bin/pdfium.dll", darwin: "lib/libpdfium.dylib", linux: "lib/libpdfium.so" }[process.platform];
const pdfDir = root + "pdfium/", libDest = pdfDir + lib.split("/").pop(), marker = pdfDir + "VERSION";
console.log(`pdfium ${tag} (${plat})`);
mkdirSync(pdfDir, { recursive: true });
// The marker records which release the library came from, so changing the pin replaces an older library.
if (good(libDest) && existsSync(marker) && readFileSync(marker, "utf8").trim() === tag) {
  console.log("have", "pdfium/" + libDest.split("/").pop());
} else {
  const tmp = `${tmpdir()}/pdfium-${plat}`;
  rmSync(tmp, { recursive: true, force: true }); mkdirSync(tmp, { recursive: true });
  const url = `https://github.com/bblanchon/pdfium-binaries/releases/download/${tag}/pdfium-${plat}.tgz`;
  console.log("downloading", url);
  await download(url, `${tmp}/p.tgz`, cfg.sha256?.[`pdfium-${plat}`]);
  execFileSync("tar", ["-xzf", `${tmp}/p.tgz`, "-C", tmp]);
  copyFileSync(`${tmp}/${lib}`, libDest + ".part");
  renameSync(libDest + ".part", libDest);
  writeFileSync(marker, tag + "\n");
  console.log("pdfium ready");
}
