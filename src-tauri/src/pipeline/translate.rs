//! Translation: the local llama.cpp server (OpenAI-compatible endpoint, no key, any model name), prompts, retries and the cache.

use super::*;

/// Settings from the "Translation options" panel. Everything here is sent to the model as instructions.
#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase", default)]
pub struct TranslateOptions {
    pub tone: String,         // "", "formal" or "informal"
    pub glossary: String,     // one entry per line: `term = translation`; a line with only a term means "keep unchanged"
    pub instructions: String, // free text
    pub use_context: bool,    // send the whole page's text first, as context
    pub max_parallel: usize,  // how many requests may run at once (llama-server -np); 1 = one at a time
    pub reuse_cache: bool,    // reuse the results of an earlier run of the same, unchanged page
}

impl Default for TranslateOptions {
    fn default() -> Self { Self { tone: String::new(), glossary: String::new(), instructions: String::new(), use_context: true, max_parallel: 1, reuse_cache: true } }
}

/// False for text that a translation could only damage: numbers, dates, prices and symbols (no letters at all), URLs,
/// e-mail addresses, and one-word codes with digits such as `A12-B34` or `ISO9001`. A word that happens to contain a
/// digit (`10-Jahres-Plan`) and any text in a non-ASCII script (`第12章`) is still translated.
pub fn is_translatable(text: &str) -> bool {
    let t = text.trim();
    if !t.chars().any(char::is_alphabetic) { return false; }
    if t.chars().any(char::is_whitespace) { return true; }
    let lower = t.to_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("www.") { return false; }
    if let Some((user, host)) = t.split_once('@') { if !user.is_empty() && host.contains('.') && !host.ends_with('.') { return false; } }
    if t.is_ascii() && t.chars().any(|c| c.is_ascii_digit()) {
        // a code unless it holds a real word: a run of 4 or more letters that is not all capitals
        return t.split(|c: char| !c.is_alphabetic()).any(|run| run.chars().count() >= 4 && run.chars().any(char::is_lowercase));
    }
    true
}

/// Parses the glossary box. Separators: `=`, `=>`, `->`, `→` or a tab. Blank lines and lines starting with `#` are ignored.
pub fn parse_glossary(text: &str) -> Vec<(String, Option<String>)> {
    text.lines().filter_map(|l| {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') { return None; }
        for sep in ["=>", "->", "→", "=", "\t"] {
            if let Some((a, b)) = l.split_once(sep) {
                let (a, b) = (a.trim(), b.trim());
                return if a.is_empty() { None } else { Some((a.to_string(), if b.is_empty() { None } else { Some(b.to_string()) })) };
            }
        }
        Some((l.to_string(), None))
    }).collect()
}

pub(super) const CONTEXT_MAX_CHARS: usize = 6000; // keeps the context inside small local-model windows

pub(super) const BATCH_MAX_BLOCKS: usize = 25;

pub(super) const BATCH_MAX_CHARS: usize = 3000;

pub(super) fn system_prompt(target: &str, o: &TranslateOptions, page_text: &str) -> String {
    let mut s = format!("You are a translation engine. Translate every string in the JSON array into {target}. Return ONLY a JSON array of strings with the same length and order. Keep numbers, names and line breaks.");
    match o.tone.as_str() {
        "formal" => s.push_str("\nUse a formal tone (for example formal forms of address)."),
        "informal" => s.push_str("\nUse an informal, familiar tone (for example informal forms of address)."),
        _ => {}
    }
    // Only glossary terms that occur on this page are sent, so a long glossary does not eat the context window.
    let hay = page_text.to_lowercase();
    let terms: Vec<_> = parse_glossary(&o.glossary).into_iter().filter(|(t, _)| hay.contains(&t.to_lowercase())).collect();
    if !terms.is_empty() {
        s.push_str("\nGlossary. Whenever the source term appears, use exactly this translation. Terms marked keep must stay unchanged:");
        for (t, tr) in terms {
            match tr { Some(tr) => s.push_str(&format!("\n- {t} => {tr}")), None => s.push_str(&format!("\n- {t} => (keep unchanged)")) }
        }
    }
    let ins = o.instructions.trim();
    if !ins.is_empty() { s.push_str(&format!("\nAdditional instructions from the user: {ins}")); }
    s
}

/// The messages that come before every request for one page: the instructions and, optionally, the page's full text as
/// context. Because the prefix is identical for every request on the page, llama.cpp can reuse its cached prompt.
pub(super) fn prefix(target: &str, o: &TranslateOptions, page_text: &str) -> Vec<serde_json::Value> {
    let mut v = vec![serde_json::json!({"role": "system", "content": system_prompt(target, o, page_text)})];
    if o.use_context && !page_text.trim().is_empty() {
        let ctx: String = page_text.chars().take(CONTEXT_MAX_CHARS).collect();
        v.push(serde_json::json!({"role": "user", "content": format!("For context only, here is the full text of the page that the strings come from. Do not translate it yet.\n\n{ctx}")}));
        v.push(serde_json::json!({"role": "assistant", "content": "Understood. Send me the JSON array of strings to translate."}));
    }
    v
}

/// Cleans up the endpoint address typed by the user: surrounding spaces and a trailing `/` go, and `http://` is added when
/// there is no scheme (`127.0.0.1:8080/v1` is accepted). Spaces inside the address and other schemes are refused with a
/// message, because the network error for them says nothing useful.
pub fn endpoint_url(base: &str) -> Result<String, String> {
    let t = base.trim().trim_end_matches('/');
    if t.is_empty() { return Err("Enter the endpoint URL, for example http://127.0.0.1:8080/v1".into()); }
    if t.chars().any(char::is_whitespace) { return Err("The endpoint URL must not contain spaces".into()); }
    match t.split_once("://") {
        Some((scheme, _)) if scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https") => Ok(t.to_string()),
        Some(_) => Err("URL must start with http:// or https://".into()),
        None => Ok(format!("http://{t}")),
    }
}

/// A server that is still loading its model answers 503, and a server that restarts resets the connection. Both are
/// retried a few times before the request counts as failed. (Stop still works: it drops the whole future, also the wait.)
pub(super) const RETRIES: usize = 3;

pub(super) const RETRY_WAIT: std::time::Duration = if cfg!(test) { std::time::Duration::from_millis(50) } else { std::time::Duration::from_secs(2) };

/// True for a connection that was reset or closed under the request (not for "nothing is listening", which is a final answer).
pub(super) fn is_transient(e: &reqwest::Error) -> bool {
    let mut cur: Option<&(dyn std::error::Error + 'static)> = Some(e);
    while let Some(err) = cur {
        if let Some(io) = err.downcast_ref::<std::io::Error>() {
            use std::io::ErrorKind::*;
            if matches!(io.kind(), ConnectionReset | ConnectionAborted | BrokenPipe | UnexpectedEof) { return true; }
        }
        if err.to_string().contains("connection closed before message completed") { return true; }
        cur = err.source();
    }
    false
}

/// A reply limit, generous on purpose: a model that loops would otherwise write until its context is full, and with no
/// timeout the page would wait until the user presses Stop. A reply that is cut off still falls back to one block at a time.
pub(super) fn max_tokens(texts: &[&str]) -> usize {
    (texts.iter().map(|t| t.chars().count()).sum::<usize>() * 4 + 256).clamp(256, 8192)
}

pub(super) async fn call(client: &reqwest::Client, prefix: &[serde_json::Value], texts: &[&str], base: &str) -> Result<Vec<String>, String> {
    let mut messages = prefix.to_vec();
    messages.push(serde_json::json!({"role": "user", "content": serde_json::to_string(texts).unwrap()}));
    let body = serde_json::json!({"model": "default", "temperature": 0.2, "max_tokens": max_tokens(texts), "messages": messages});
    let url = format!("{}/chat/completions", base.trim_end_matches('/'));
    let mut attempt = 0;
    let r = loop {
        match client.post(&url).json(&body).send().await {
            Ok(r) if r.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE && attempt < RETRIES => {}
            Ok(r) => break r,
            Err(e) if is_transient(&e) && attempt < RETRIES => {}
            Err(e) => return Err(format!("Cannot reach {base}: {e}")),
        }
        attempt += 1;
        tokio::time::sleep(RETRY_WAIT).await;
    };
    let status = r.status();
    if !status.is_success() {
        // llama-server explains the problem in the body (for example "context too small"), so show the start of it.
        let text = r.text().await.unwrap_or_default();
        let text: String = text.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(200).collect();
        return Err(if text.is_empty() { format!("Endpoint returned HTTP {status}") } else { format!("Endpoint returned HTTP {status}: {text}") });
    }
    let j: serde_json::Value = r.json().await.map_err(|e| e.to_string())?;
    let c = j["choices"][0]["message"]["content"].as_str().ok_or("Response had no message content")?;
    extract_json_array(c)
}

/// Pulls the JSON array of strings out of a model reply. Reasoning models print `<think>...</think>` first, and chatty
/// models add a sentence or a ``` fence around the array, so the reply is cleaned up in steps: drop the reasoning,
/// strip fences, then (if it still does not parse) try every `[` up to the last `]` as the start of the array.
/// Reads a JSON array as a list of strings. A model sometimes writes a number or true/false where a string belongs
/// (`["Hello", 2]`); those become text instead of failing the whole reply. Anything else (null, nested arrays) is refused.
pub(super) fn parse_strings(s: &str) -> Option<Vec<String>> {
    let items: Vec<serde_json::Value> = serde_json::from_str(s).ok()?;
    items.into_iter().map(|v| match v {
        serde_json::Value::String(t) => Some(t),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }).collect()
}

pub fn extract_json_array(raw: &str) -> Result<Vec<String>, String> {
    let s = match raw.rfind("</think>") { Some(p) => &raw[p + "</think>".len()..], None => raw };
    let s = s.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
    if let Some(v) = parse_strings(s) { return Ok(v); }
    if let Some(end) = s.rfind(']') {
        for (start, _) in s.match_indices('[') {
            if start < end { if let Some(v) = parse_strings(&s[start..=end]) { return Ok(v); } }
        }
    }
    Err(format!("Model did not return a JSON array. Reply began: {}", s.chars().take(80).collect::<String>()))
}

/// The one HTTP client for all translation requests of the app, so connections stay open between requests and pages.
/// There is deliberately no overall timeout: a slow local model may need minutes for a page.
pub(super) fn http() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| reqwest::Client::builder().tcp_keepalive(std::time::Duration::from_secs(30)).build().unwrap_or_default())
}

/// Result of the "Test connection" button.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnInfo { pub message: String, pub slots: Option<u32>, pub warning: Option<String> }

/// Checks that the server answers at `base` (e.g. http://127.0.0.1:8080/v1). llama-server's `/props` page, when available,
/// also tells how many parallel slots it was started with (-np) and how many tokens of context each slot has.
pub async fn test_connection(base: &str) -> Result<ConnInfo, String> {
    let base = endpoint_url(base)?;
    let base = base.as_str();
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(5)).build().map_err(|e| e.to_string())?;
    let r = client.get(format!("{base}/models")).send().await.map_err(|e| {
        if e.is_timeout() { "Timed out after 5 seconds".to_string() }
        else if e.is_connect() { format!("Cannot connect to {base}. Is llama-server running?") }
        else { e.to_string() }
    })?;
    if !r.status().is_success() { return Err(format!("Server answered HTTP {}. Check that the URL ends with /v1", r.status())); }
    let j: serde_json::Value = r.json().await.unwrap_or_default();
    let id = j["data"][0]["id"].as_str().or(j["models"][0]["model"].as_str()).or(j["models"][0]["name"].as_str());
    let mut message = match id { Some(id) => format!("Connected. Model: {id}"), None => "Connected.".to_string() };
    let (mut slots, mut warning) = (None, None);
    let root = base.strip_suffix("/v1").unwrap_or(base);
    if let Ok(r) = client.get(format!("{root}/props")).send().await {
        if let Ok(p) = r.json::<serde_json::Value>().await {
            slots = p["total_slots"].as_u64().map(|n| n as u32);
            let ctx = p["default_generation_settings"]["n_ctx"].as_u64();
            if let Some(n) = slots { message.push_str(&format!(" · {n} slot{}", if n == 1 { "" } else { "s" })); }
            if let Some(c) = ctx {
                message.push_str(&format!(" · {c} tokens per slot"));
                if c < 3500 { warning = Some(format!("only {c} tokens per slot. Page context may not fit: start llama-server with a larger -c (about 4096 x -np)")); }
            }
        }
    }
    Ok(ConnInfo { message, slots, warning })
}

/// A reply is only a translation if it is not blank (a blank would erase the block on the page).
pub(super) fn non_blank(s: String) -> Option<String> { if s.trim().is_empty() { None } else { Some(s) } }

/// Translates one chunk of strings. If the request fails (for example the page context does not fit into a slot's
/// context window) it is retried without the page text; if the model returns the wrong number of strings, the chunk is
/// translated one string at a time. A string that still gets no usable reply comes back as `None` (the block keeps its
/// source text) instead of failing the whole page; only if nothing at all could be translated is the error returned.
pub(super) async fn translate_chunk(client: &reqwest::Client, pre: &[serde_json::Value], pre_min: &[serde_json::Value], texts: &[String], base: &str)
    -> Result<Vec<Option<String>>, String> {
    let refs: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
    let mut used = pre;
    let mut r = call(client, used, &refs, base).await;
    if r.is_err() && used.len() > pre_min.len() { used = pre_min; r = call(client, used, &refs, base).await; }
    match r {
        Ok(v) if v.len() == refs.len() => Ok(v.into_iter().map(non_blank).collect()),
        _ => {
            let (mut v, mut last_err) = (vec![], None);
            for t in &refs {
                match call(client, used, &[t], base).await {
                    Ok(r) => v.push(r.into_iter().next().and_then(non_blank)),
                    Err(e) => { last_err = Some(e); v.push(None); }
                }
            }
            if v.iter().all(Option::is_none) { if let Some(e) = last_err { return Err(e); } }
            Ok(v)
        }
    }
}

/// Memory of finished translations, so running an unchanged page again costs no model time. It lives in memory only (gone
/// when the app closes). The key covers the whole page, never single lines: the same line can translate differently in a
/// different scene (see `cache_key`).
#[derive(Default)]
pub struct TranslationCache(Mutex<HashMap<String, HashMap<u32, String>>>);

pub(super) const CACHE_MAX_PAGES: usize = 256;

/// Identifies "this page, translated this way": every block's id and text plus everything that shapes the result (target
/// language, server, model, tone, glossary, instructions, whether page context is sent). Box positions and colours are
/// left out, so moving or resizing a box does not forget the translations.
pub(super) fn cache_key(blocks: &[Block], target: &str, base: &str, model: &str, o: &TranslateOptions) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (target, base.trim().trim_end_matches('/'), model, &o.tone, &o.glossary, &o.instructions, o.use_context).hash(&mut h);
    for b in blocks { (b.id, b.text.trim()).hash(&mut h); }
    format!("{:016x}", h.finish())
}

/// The model the server has loaded (best effort), so that switching models on the same address does not serve old results.
pub(super) async fn model_id(client: &reqwest::Client, base: &str) -> String {
    let r = client.get(format!("{}/models", base.trim_end_matches('/'))).timeout(std::time::Duration::from_secs(3)).send().await;
    let Ok(r) = r else { return String::new() };
    let j: serde_json::Value = r.json().await.unwrap_or_default();
    j["data"][0]["id"].as_str().or(j["models"][0]["model"].as_str()).unwrap_or("").to_string()
}

/// Translates the blocks of ONE page. Returns (blocks taken from the cache, blocks that got no translation).
/// The model first sees the whole page's text as context (if enabled), then the numbered lines to translate.
/// `only` limits translation to those block ids; those are always translated afresh, never taken from the cache.
/// With `max_parallel` > 1 the page is cut into small chunks (about two per slot) and up to `max_parallel` requests run at
/// once; each request carries the same context, so the results do not depend on the order they finish in.
pub async fn translate(blocks: &mut [Block], target: &str, base: &str, opts: &TranslateOptions, only: Option<&[u32]>, cache: &TranslationCache)
    -> Result<(usize, usize), String> {
    let par = opts.max_parallel.clamp(1, 16);
    let client = http();
    let base = endpoint_url(base)?;
    let base = base.as_str();
    let key = cache_key(blocks, target, base, &model_id(client, base).await, opts);
    // Blocks that are only numbers, URLs, e-mail addresses or codes stay as they are. A block the user asks for by id is always translated.
    let want: Vec<usize> = blocks.iter().enumerate()
        .filter(|(_, b)| !b.text.trim().is_empty() && only.map_or_else(|| is_translatable(&b.text), |o| o.contains(&b.id))).map(|(i, _)| i).collect();
    let (mut idx, mut cached) = (vec![], 0usize);
    if opts.reuse_cache && only.is_none() {
        let c = lock(&cache.0);
        let hit = c.get(&key);
        for &i in &want {
            match hit.and_then(|m| m.get(&blocks[i].id)) { Some(t) => { blocks[i].translated = Some(t.clone()); cached += 1; } None => idx.push(i) }
        }
    } else {
        idx = want;
    }
    if idx.is_empty() { return Ok((cached, 0)); }
    let page_text = blocks.iter().map(|b| b.text.trim()).filter(|t| !t.is_empty()).collect::<Vec<_>>().join("\n\n");
    let pre = prefix(target, opts, &page_text);
    let pre_min = prefix(target, &TranslateOptions { use_context: false, ..opts.clone() }, &page_text);
    let total: usize = idx.iter().map(|&i| blocks[i].text.len()).sum();
    let mut batches: Vec<Vec<usize>> = vec![];
    if par > 1 && total > 0 {
        // About two chunks per slot, balanced by text length, never bigger than one sequential batch would be.
        let mut n = (par * 2).min(idx.len()).max(total.div_ceil(BATCH_MAX_CHARS)).max(idx.len().div_ceil(BATCH_MAX_BLOCKS)).min(idx.len());
        if n > par { n = (n.div_ceil(par) * par).min(idx.len()); } // whole waves of `par` requests
        let mut chunks: Vec<Vec<usize>> = vec![vec![]; n];
        let mut cum = 0usize;
        for &i in &idx {
            let len = blocks[i].text.len();
            chunks[(((cum + len / 2) * n) / total).min(n - 1)].push(i);
            cum += len;
        }
        batches = chunks.into_iter().filter(|c| !c.is_empty()).collect();
    } else {
        let (mut cur, mut chars) = (vec![], 0usize);
        for &i in &idx {
            let n = blocks[i].text.len();
            if !cur.is_empty() && (cur.len() >= BATCH_MAX_BLOCKS || chars + n > BATCH_MAX_CHARS) { batches.push(std::mem::take(&mut cur)); chars = 0; }
            cur.push(i); chars += n;
        }
        if !cur.is_empty() { batches.push(cur); }
    }
    let jobs: Vec<(Vec<usize>, Vec<String>)> = batches.into_iter().map(|b| { let t = b.iter().map(|&i| blocks[i].text.clone()).collect(); (b, t) }).collect();
    let (pre, pre_min) = (&pre, &pre_min);
    let mut results = stream::iter(jobs).map(|(b, texts)| async move {
        let out = translate_chunk(client, pre, pre_min, &texts, base).await?;
        Ok::<_, String>((b, out))
    }).buffer_unordered(par);
    let mut fresh: Vec<(u32, String)> = vec![];
    while let Some(r) = results.next().await {
        let (b, out) = r?;
        for (&i, t) in b.iter().zip(out) {
            if let Some(t) = t {
                blocks[i].translated = Some(t.clone());
                fresh.push((blocks[i].id, t));
            }
        }
    }
    let missing = idx.len() - fresh.len();
    if !fresh.is_empty() {
        let mut c = lock(&cache.0);
        if c.len() >= CACHE_MAX_PAGES && !c.contains_key(&key) { c.clear(); }
        c.entry(key).or_default().extend(fresh);
    }
    Ok((cached, missing))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glossary_parsing() {
        let g = parse_glossary("Haus = house\n# comment\n\nAcme GmbH\nA -> B\nX → Y\nEmpty =\n= nothing");
        assert_eq!(g[0], ("Haus".into(), Some("house".into())));
        assert_eq!(g[1], ("Acme GmbH".into(), None));
        assert_eq!(g[2], ("A".into(), Some("B".into())));
        assert_eq!(g[3], ("X".into(), Some("Y".into())));
        assert_eq!(g[4], ("Empty".into(), None));
        assert_eq!(g.len(), 5);
    }

    #[test]
    fn prompt_uses_only_relevant_glossary_terms() {
        let o = TranslateOptions { tone: "formal".into(), glossary: "Haus = house\nKatze = cat".into(), instructions: "Legal text.".into(), use_context: true, ..Default::default() };
        let p = system_prompt("English", &o, "Das Haus ist groß.");
        assert!(p.contains("formal tone") && p.contains("Haus => house") && p.contains("Legal text."));
        assert!(!p.contains("Katze"));
    }

    #[test]
    fn context_turn_is_optional() {
        let mut o = TranslateOptions::default();
        assert_eq!(prefix("English", &o, "Hallo Welt").len(), 3); // system + context + acknowledgement
        o.use_context = false;
        assert_eq!(prefix("English", &o, "Hallo Welt").len(), 1);
    }

    fn cblock(id: u32, text: &str) -> Block {
        Block { id, bbox: [0.0; 4], text: text.into(), translated: None, conf: 1.0, font_px: 10.0, fg: [0; 3], bg: [255; 3] }
    }

    #[test]
    fn cache_key_follows_what_shapes_the_result() {
        let o = TranslateOptions::default();
        let b = vec![cblock(1, "Hallo"), cblock(2, "Welt")];
        let k = cache_key(&b, "English", "http://x/v1", "m1", &o);
        assert_eq!(k, cache_key(&b, "English", "http://x/v1/", "m1", &o)); // trailing slash is not a difference
        let mut moved = b.clone();
        moved[0].bbox = [5.0, 5.0, 50.0, 20.0];
        assert_eq!(k, cache_key(&moved, "English", "http://x/v1", "m1", &o)); // moving a box keeps the cache
        assert_ne!(k, cache_key(&[cblock(1, "Hallo"), cblock(2, "Welt!")], "English", "http://x/v1", "m1", &o));
        assert_ne!(k, cache_key(&b, "French", "http://x/v1", "m1", &o));
        assert_ne!(k, cache_key(&b, "English", "http://x/v1", "m2", &o)); // another model
        assert_ne!(k, cache_key(&b, "English", "http://x/v1", "m1", &TranslateOptions { tone: "formal".into(), ..o.clone() }));
        assert_eq!(k, cache_key(&b, "English", "http://x/v1", "m1", &TranslateOptions { max_parallel: 8, ..o })); // speed settings do not matter
    }

    #[test]
    fn numbers_urls_and_codes_are_not_translated() {
        for t in ["123", "12.5 %", "$1,234.00", "12/03/2024", "—", "3 – 4", "https://example.com/a", "WWW.Example.com", "me@example.com",
                  "A12-B34", "ISO9001", "PN-2045-X", "  42  "] {
            assert!(!is_translatable(t), "{t:?} should be skipped");
        }
        for t in ["Rechnung", "Invoice no. 5", "10-Jahres-Plan", "Modell3", "第12章", "Seite 3 von 4", "USB-C", "Visit https://example.com", "ä"] {
            assert!(is_translatable(t), "{t:?} should be translated");
        }
    }

    #[test]
    fn json_array_extraction() {
        let want = vec!["Hello".to_string(), "World".to_string()];
        assert_eq!(extract_json_array(r#"["Hello","World"]"#).unwrap(), want);
        assert_eq!(extract_json_array("```json\n[\"Hello\",\"World\"]\n```").unwrap(), want);
        assert_eq!(extract_json_array("<think>The user wants [\"x\"]...</think>\n[\"Hello\",\"World\"]").unwrap(), want);
        assert_eq!(extract_json_array("Here is the translated JSON:\n```json\n[\"Hello\",\"World\"]\n```\nHope this helps!").unwrap(), want);
        assert_eq!(extract_json_array("[Note] Sure: [\"Hello\",\"World\"]").unwrap(), want); // a [ in the preamble
        assert!(extract_json_array("I cannot translate this.").is_err());
        assert!(extract_json_array("[\"cut off").is_err());
    }

    #[tokio::test]
    async fn a_poisoned_cache_lock_does_not_break_translate() {
        let cache = TranslationCache::default();
        // Panic on another thread while it holds the lock, which poisons the mutex.
        let _ = std::thread::scope(|s| s.spawn(|| { let _g = cache.0.lock().unwrap(); panic!("poison the lock on purpose"); }).join());
        assert!(cache.0.is_poisoned());
        // Only numbers: nothing is sent to a model, but the cache lock is taken to look for earlier results.
        let mut blocks = vec![Block { id: 1, bbox: [0.0, 0.0, 10.0, 10.0], text: "12345".into(), translated: None, conf: 1.0, font_px: 10.0, fg: [0; 3], bg: [255; 3] }];
        let r = translate(&mut blocks, "English", "http://127.0.0.1:9/v1", &TranslateOptions::default(), None, &cache).await;
        assert_eq!(r.unwrap(), (0, 0));
    }

    // ----- R8: translation endpoint -----

    #[test]
    fn endpoint_urls_are_cleaned_up() {
        assert_eq!(endpoint_url("  http://127.0.0.1:8080/v1/  ").unwrap(), "http://127.0.0.1:8080/v1");
        assert_eq!(endpoint_url("127.0.0.1:8080/v1").unwrap(), "http://127.0.0.1:8080/v1", "scheme added");
        assert_eq!(endpoint_url("localhost:8080").unwrap(), "http://localhost:8080");
        assert_eq!(endpoint_url("HTTPS://example.com/v1").unwrap(), "HTTPS://example.com/v1");
        assert!(endpoint_url("http://127.0.0.1 :8080/v1").unwrap_err().contains("spaces"));
        assert!(endpoint_url("ftp://host/v1").unwrap_err().contains("http://"));
        assert!(endpoint_url("   ").is_err());
    }

    #[test]
    fn numbers_and_booleans_in_a_reply_become_text() {
        assert_eq!(extract_json_array(r#"["Hello", 2]"#).unwrap(), ["Hello", "2"]);
        assert_eq!(extract_json_array(r#"Here: [1.5, true, "x"]"#).unwrap(), ["1.5", "true", "x"]);
        assert!(extract_json_array(r#"["a", null]"#).is_err());
        assert!(extract_json_array(r#"[["a"]]"#).is_err());
    }

    #[test]
    fn the_reply_limit_follows_the_input_and_stays_in_range() {
        assert_eq!(max_tokens(&[]), 256);
        assert_eq!(max_tokens(&["abcde"]), 20 + 256);
        assert_eq!(max_tokens(&["日本語"]), 12 + 256, "counts characters, not bytes");
        assert_eq!(max_tokens(&[&"x".repeat(100_000)]), 8192);
    }

    /// Answers the given (status, body) pairs, one per request in order, and records each request it received. Status 0 hangs up.
    async fn mock_server(replies: Vec<(u16, String)>) -> (String, Arc<Mutex<Vec<String>>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(vec![]));
        let log = seen.clone();
        tokio::spawn(async move {
            for (status, body) in replies {
                let (mut sock, _) = listener.accept().await.unwrap();
                let (mut buf, mut tmp) = (vec![], [0u8; 4096]);
                loop {
                    let n = sock.read(&mut tmp).await.unwrap();
                    if n == 0 { break; }
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(h) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..h]).to_lowercase();
                        let len = head.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0);
                        if buf.len() >= h + 4 + len { break; }
                    }
                }
                log.lock().unwrap().push(String::from_utf8_lossy(&buf).into_owned());
                if status == 0 { continue; } // status 0: hang up without answering (the socket is dropped here)
                let resp = format!("HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            }
        });
        (url, seen)
    }

    const OK_BODY: &str = r#"{"choices":[{"message":{"content":"[\"Hallo\"]"}}]}"#;

    #[tokio::test]
    async fn a_loading_server_is_retried_until_it_answers() {
        let (url, seen) = mock_server(vec![(503, "Loading model".into()), (503, "Loading model".into()), (200, OK_BODY.into())]).await;
        let r = call(&reqwest::Client::new(), &[], &["Hello"], &url).await;
        assert_eq!(r.unwrap(), ["Hallo"]);
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 3, "two 503 answers, then the real one");
        assert!(seen[2].contains("\"max_tokens\":276"), "reply limit is sent: {}", seen[2]);
    }

    #[tokio::test]
    async fn a_server_that_stays_unavailable_gives_up_with_its_message() {
        let (url, seen) = mock_server(vec![(503, "model is still loading".into()); 4]).await;
        let e = call(&reqwest::Client::new(), &[], &["Hello"], &url).await.unwrap_err();
        assert!(e.contains("503") && e.contains("model is still loading"), "{e}");
        assert_eq!(seen.lock().unwrap().len(), 4, "one try and three retries");
    }

    #[tokio::test]
    async fn an_error_answer_shows_the_start_of_its_body() {
        let body = format!(r#"{{"error":{{"message":"context too small {}"}}}}"#, "x".repeat(500));
        let (url, seen) = mock_server(vec![(400, body)]).await;
        let e = call(&reqwest::Client::new(), &[], &["Hello"], &url).await.unwrap_err();
        assert!(e.starts_with("Endpoint returned HTTP 400 Bad Request: ") && e.contains("context too small"), "{e}");
        assert!(e.chars().count() < 260, "the body is cut at 200 characters");
        assert_eq!(seen.lock().unwrap().len(), 1, "a 400 is final, no retry");
    }

    #[tokio::test]
    async fn a_connection_that_is_cut_is_retried() {
        let (url, seen) = mock_server(vec![(0, String::new()), (200, OK_BODY.into())]).await;
        let r = call(&reqwest::Client::new(), &[], &["Hello"], &url).await;
        assert_eq!(r.unwrap(), ["Hallo"]);
        assert_eq!(seen.lock().unwrap().len(), 2);
    }
}
