// Checks the Content Security Policy (CSP) setup. Run: npm run check:csp   (CI runs it before every build)
//  1. tauri.conf.json must set a policy, and the policy must stay strict.
//  2. src/index.html must not use anything that the policy blocks: inline style attributes, inline event handlers,
//     javascript: URLs, eval, or resources from the internet. Tauri adds a hash for the inline <script> and a nonce
//     for the inline <style> by itself, so those two blocks are allowed.
import { readFileSync } from "node:fs";

const read = (p) => readFileSync(new URL(p, import.meta.url), "utf8");
const problems = [];
const bad = (m) => problems.push(m);

// ---- 1. the policy --------------------------------------------------------------------------------------------
const csp = JSON.parse(read("../src-tauri/tauri.conf.json")).app?.security?.csp;
if (!csp) bad('app.security.csp in src-tauri/tauri.conf.json is empty or null. Set a policy.');
else {
  const dirs = {}; // directive -> list of sources, for the object form and for a policy string
  if (typeof csp === "string") for (const part of csp.split(";")) { const [k, ...v] = part.trim().split(/\s+/); if (k) dirs[k] = v; }
  else for (const [k, v] of Object.entries(csp)) dirs[k] = Array.isArray(v) ? v : String(v).split(/\s+/);
  const has = (d, s) => (dirs[d] ?? []).includes(s);
  if (!dirs["default-src"] || !(has("default-src", "'none'") || has("default-src", "'self'"))) bad("default-src must be 'none' (or 'self').");
  for (const [d, src] of Object.entries(dirs)) {
    for (const s of src) {
      if (/^'unsafe-/.test(s)) bad(`${d} contains ${s}.`);
      if (s === "*" || /^https?:$/.test(s) || /^https?:\/\/\*/.test(s)) bad(`${d} contains the wildcard source ${s}.`);
    }
  }
  for (const d of ["script-src", "style-src"]) if ((dirs[d] ?? ["'self'"]).some((s) => s === "data:" || s === "blob:")) bad(`${d} must not allow data: or blob:.`);
  if (!dirs["connect-src"] || dirs["connect-src"].some((s) => !/^(ipc:|http:\/\/ipc\.localhost|'self')$/.test(s))) bad("connect-src must list only the Tauri IPC addresses: ipc: http://ipc.localhost");
  if (!has("base-uri", "'none'")) bad("base-uri must be 'none'.");
  if (!has("form-action", "'none'")) bad("form-action must be 'none'.");
}

// ---- 2. the page ----------------------------------------------------------------------------------------------
const html = read("../src/index.html").split("\n");
const rules = [
  [/\sstyle\s*=\s*["'\\]/, 'inline style attribute. Use a CSS class (see the helper classes at the end of the stylesheet).'],
  [/setAttribute\(\s*['"]style['"]/, "setAttribute('style', ...). Set element.style.property instead, or use a class."],
  [/<[a-zA-Z][^<>]*\son[a-z]+\s*=/, "inline event handler (onclick=...). Set element.onclick in the script instead."],
  [/javascript:/i, "javascript: URL."],
  [/\beval\s*\(|new\s+Function\s*\(|set(Timeout|Interval)\(\s*['"`]/, "eval, new Function or a string timer."],
  [/<(script|link|img|iframe|source|audio|video)\b[^>]*\s(src|href)\s*=\s*["']?https?:/i, "resource from the internet."],
  [/@import|url\(\s*["']?https?:/i, "CSS import or remote url()."],
  [/<(iframe|object|embed|form)\b/i, "iframe, object, embed or form element."],
];
html.forEach((line, i) => rules.forEach(([re, why]) => { if (re.test(line)) bad(`src/index.html line ${i + 1}: ${why}`); }));

if (problems.length) { console.error("CSP check failed:\n- " + problems.join("\n- ")); process.exit(1); }
console.log("CSP check passed.");
