// One place to read and change the app version. It lives in six places that must agree (and `--locked` builds fail if
// Cargo.lock does not match Cargo.toml):
//   node scripts/version.mjs               check that all of them are the same (CI runs this: npm run check:version)
//   node scripts/version.mjs set 0.2.0     change all of them (npm run version:set -- 0.2.0)
// No dependencies; run it from the repository root.
import { readFileSync, writeFileSync } from 'node:fs';

// Each place: the file, and a pattern with three groups: text before the version, the version, text after it.
const PLACES = [
  { file: 'src-tauri/Cargo.toml', re: /(^\[package\][\s\S]*?^version = ")([^"]+)(")/m },
  { file: 'src-tauri/Cargo.lock', re: /(name = "layout-translator"\r?\nversion = ")([^"]+)(")/ },
  { file: 'src-tauri/tauri.conf.json', re: /("version"\s*:\s*")([^"]+)(")/ },
  { file: 'package.json', re: /("version"\s*:\s*")([^"]+)(")/ },
  { file: 'package-lock.json', re: /("name": "layout-translator",\s*"version": ")([^"]+)(")/g }, // top level and packages[""]
];

const read = (f) => readFileSync(f, 'utf8');
const versionsIn = (text, re) => [...text.matchAll(new RegExp(re.source, re.flags.includes('g') ? re.flags : re.flags + 'g'))].map((m) => m[2]);

const [cmd, wanted] = process.argv.slice(2);

if (cmd === 'set') {
  if (!/^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/.test(wanted ?? '')) {
    console.error('Usage: node scripts/version.mjs set <x.y.z>   (for example 0.2.0 or 0.2.0-beta.1)');
    process.exit(2);
  }
  for (const { file, re } of PLACES) {
    const text = read(file);
    if (versionsIn(text, re).length === 0) throw new Error(`no version found in ${file}`);
    writeFileSync(file, text.replace(re, (_, a, _v, c) => a + wanted + c));
    console.log(`${file}: ${wanted}`);
  }
  console.log('Done. Run the tests, then commit all five files.');
} else if (cmd === undefined) {
  const found = PLACES.flatMap(({ file, re }) => {
    const v = versionsIn(read(file), re);
    return v.length ? v.map((x) => ({ file, version: x })) : [{ file, version: undefined }];
  });
  const first = found[0].version;
  const wrong = found.filter((f) => f.version !== first);
  if (wrong.length) {
    console.error(`The version is not the same everywhere (Cargo.toml says ${first}):`);
    for (const w of wrong) console.error(`  ${w.file}: ${w.version ?? 'not found'}`);
    console.error('Fix it with: npm run version:set -- <x.y.z>');
    process.exit(1);
  }
  console.log(`Version ${first} is the same in ${found.length} places.`);
} else {
  console.error('Usage: node scripts/version.mjs [set <x.y.z>]');
  process.exit(2);
}
