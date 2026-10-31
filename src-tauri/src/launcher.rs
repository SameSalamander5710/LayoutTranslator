//! Starts the translation server with a command that the user writes (for example the `llama-server` command of llama.cpp).
//!
//! The command is kept in a plain text file, `server-command.txt`, that any text editor can change. To start the server
//! the command is written to a temporary script and the script runs in a new, titled terminal window that stays open, so
//! the user sees the output and any error. A script (and not a command line) is used because operators such as `&&`, `|`
//! and `^` would otherwise be read twice, once by the program that opens the window and once by the shell inside it.
//!
//! - Windows: `cmd.exe` runs a temporary `.bat` file in a new console window. The file is removed when the window closes.
//! - macOS: Terminal opens a temporary `.command` file.
//! - Linux: the first terminal program that is found runs a temporary `.sh` file (see `TERMINALS`).
//!
//! On macOS and Linux the script deletes itself when it starts. Old scripts that are left behind (for example after a
//! crash) are removed by `sweep_old_scripts`.
use crate::pipeline::write_atomic;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime};

/// Name of the file that holds the command.
pub const COMMAND_FILE: &str = "server-command.txt";
/// Title of the terminal window.
pub const WINDOW_TITLE: &str = "Layout Translator server";
/// A command longer than this is refused. It is certainly a mistake (for example a pasted file).
const MAX_COMMAND_CHARS: usize = 20_000;
/// Temporary scripts start with this, so that `sweep_old_scripts` knows which files are ours.
const SCRIPT_PREFIX: &str = "layout-translator-server-";

// ---------- The command file ----------

/// Brings what the user typed or what a text editor saved into one form: no byte order mark, `\n` line ends, no spaces or
/// empty lines at the start or the end.
pub fn normalise(text: &str) -> String {
    text.trim_start_matches('\u{feff}').replace("\r\n", "\n").replace('\r', "\n").trim().to_string()
}

/// Refuses a command that cannot be saved or run.
pub fn check(command: &str) -> Result<(), String> {
    if command.contains('\0') { return Err("The command contains a NUL character.".into()); }
    if command.chars().count() > MAX_COMMAND_CHARS { return Err(format!("The command is longer than {MAX_COMMAND_CHARS} characters.")); }
    Ok(())
}

/// The places where the command file may be, in order of preference: the app folder (next to the program, so a portable
/// copy keeps its command) and then the per-user folder. `use_app_dir` is false on macOS: the app folder is inside the
/// signed `.app` bundle, and a changed bundle would no longer pass the signature check.
pub fn command_file_candidates(app_dir: Option<&Path>, user_dir: Option<&Path>, use_app_dir: bool) -> Vec<PathBuf> {
    let mut v = vec![];
    if use_app_dir { if let Some(d) = app_dir { v.push(d.join(COMMAND_FILE)); } }
    if let Some(d) = user_dir { v.push(d.join(COMMAND_FILE)); }
    v
}

/// The saved command and the file it is in (or will be in, when nothing is saved yet).
#[derive(Debug, PartialEq)]
pub struct StoredCommand { pub command: String, pub path: PathBuf }

/// Reads the command from the first file that exists. With no file, the command is empty.
pub fn load_command(candidates: &[PathBuf]) -> Result<StoredCommand, String> {
    for p in candidates {
        if p.is_file() {
            let text = std::fs::read_to_string(p).map_err(|e| format!("Cannot read {}: {e}", p.display()))?;
            return Ok(StoredCommand { command: normalise(&text), path: p.clone() });
        }
    }
    let path = candidates.first().cloned().ok_or("There is no folder for the command file.")?;
    Ok(StoredCommand { command: String::new(), path })
}

/// Saves the command. A file that exists is changed where it is, and never in another place (otherwise the old file would
/// still be the one that is read). With no file yet, the first folder that accepts one is used.
pub fn save_command(candidates: &[PathBuf], command: &str) -> Result<StoredCommand, String> {
    let command = normalise(command);
    check(&command)?;
    let text = format!("{command}\n");
    let existing = candidates.iter().find(|p| p.is_file());
    let tries: Vec<&PathBuf> = match existing { Some(p) => vec![p], None => candidates.iter().collect() };
    let mut last = "There is no folder for the command file.".to_string();
    for p in tries {
        if let Some(dir) = p.parent() { let _ = std::fs::create_dir_all(dir); }
        match write_atomic(p, text.as_bytes()) {
            Ok(()) => return Ok(StoredCommand { command, path: p.clone() }),
            Err(e) => last = e,
        }
    }
    Err(last)
}

// ---------- The scripts ----------

/// Removes the characters from a window title that a batch file or a shell would read as something else.
fn sanitize_title(title: &str) -> String {
    let t: String = title.chars().filter(|c| !"\"'%&|<>^\r\n".contains(*c)).collect();
    let t = t.trim();
    if t.is_empty() { WINDOW_TITLE.to_string() } else { t.to_string() }
}

/// The batch file for Windows. `chcp 65001` makes `cmd.exe` read the rest of the file as UTF-8, so paths with accents work.
/// The file has CRLF line ends. The command is read as batch code, so a percent sign that is not a variable must be written `%%`.
pub fn windows_script(title: &str, command: &str) -> String {
    let body = command.replace("\r\n", "\n").replace('\n', "\r\n");
    format!("@echo off\r\nchcp 65001 >nul\r\ntitle {}\r\n{}\r\n", sanitize_title(title), body)
}

/// The arguments for `cmd.exe`: `start` opens a new console window and `/WAIT` keeps `cmd.exe` until that window closes, so
/// the program knows when the batch file can be removed. `cmd /K` keeps the window open when the command ends.
pub fn windows_command_line(title: &str, script: &Path) -> String {
    format!("/C start /WAIT \"{}\" cmd /K \"{}\"", sanitize_title(title), script.display())
}

/// The script for macOS and Linux. It deletes itself first, sets the window title, runs the command, tells the user how it
/// ended and waits for a key, so the window does not close before the user can read the output. It ends with the exit
/// code of the command.
pub fn unix_script(title: &str, command: &str) -> String {
    format!(
        "#!/usr/bin/env bash\nrm -f -- \"$0\"\nprintf '\\033]0;%s\\007' '{}'\n{}\nstatus=$?\necho\necho \"The command ended with exit code $status.\"\nread -r -n 1 -s -p \"Press any key to close this window...\"\nexit \"$status\"\n",
        sanitize_title(title), command
    )
}

// ---------- Linux terminals ----------

/// Terminal programs for Linux in the order they are tried, with the arguments that come before the program to run.
/// `x-terminal-emulator` is the system default on Debian and Ubuntu.
const TERMINALS: [(&str, &[&str]); 9] = [
    ("x-terminal-emulator", &["-e"]),
    ("gnome-terminal", &["--"]),
    ("konsole", &["-e"]),
    ("xfce4-terminal", &["-x"]),
    ("mate-terminal", &["-x"]),
    ("kitty", &[]),
    ("alacritty", &["-e"]),
    ("lxterminal", &["-e"]),
    ("xterm", &["-e"]),
];

#[cfg(unix)]
fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    p.is_file() && std::fs::metadata(p).map(|m| m.permissions().mode() & 0o111 != 0).unwrap_or(false)
}
#[cfg(not(unix))]
fn is_executable(p: &Path) -> bool { p.is_file() }

/// Finds a program in the folders of a PATH variable. A name with a folder in it (`/usr/bin/xterm`) is used as it is.
fn find_in_path(name: &str, path_var: Option<&OsStr>) -> Option<PathBuf> {
    let p = Path::new(name);
    if p.components().count() > 1 { return is_executable(p).then(|| p.to_path_buf()); }
    std::env::split_paths(path_var?).map(|d| d.join(name)).find(|c| is_executable(c))
}

/// The terminal program and its arguments for a script. `$TERMINAL` (if set) is tried first, then the list `TERMINALS`.
pub fn terminal_command(terminal_var: Option<&str>, path_var: Option<&OsStr>, script: &Path) -> Result<(PathBuf, Vec<OsString>), String> {
    let mut choices: Vec<(String, &[&str])> = vec![];
    if let Some(t) = terminal_var.map(str::trim).filter(|t| !t.is_empty()) { choices.push((t.to_string(), &["-e"])); }
    choices.extend(TERMINALS.iter().map(|(n, a)| (n.to_string(), *a)));
    for (name, args) in choices {
        if let Some(prog) = find_in_path(&name, path_var) {
            let mut v: Vec<OsString> = args.iter().map(OsString::from).collect();
            v.push(script.as_os_str().to_os_string());
            return Ok((prog, v));
        }
    }
    Err("No terminal program was found. Install one (for example xterm) or set the TERMINAL environment variable.".into())
}

// ---------- Starting the server ----------

/// What `launch` needs from the system. A test can give other values.
pub struct Env { pub temp_dir: PathBuf, pub path_var: Option<OsString>, pub terminal_var: Option<String> }
impl Env {
    pub fn current() -> Env { Env { temp_dir: std::env::temp_dir(), path_var: std::env::var_os("PATH"), terminal_var: std::env::var("TERMINAL").ok() } }
}

#[cfg(windows)]
fn windows_command(script: &Path) -> Command {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000; // hides the throw-away cmd.exe; the window of the command is made by `start`
    let mut c = Command::new("cmd");
    c.raw_arg(windows_command_line(WINDOW_TITLE, script)).creation_flags(CREATE_NO_WINDOW);
    c
}
#[cfg(not(windows))]
fn windows_command(_script: &Path) -> Command { Command::new("cmd") } // never used: only called when the system is Windows

fn build_command(script: &Path, env: &Env) -> Result<Command, String> {
    let mut c = if cfg!(windows) {
        windows_command(script)
    } else if cfg!(target_os = "macos") {
        let mut c = Command::new("open");
        c.arg("-a").arg("Terminal").arg(script);
        c
    } else {
        let (prog, args) = terminal_command(env.terminal_var.as_deref(), env.path_var.as_deref(), script)?;
        let mut c = Command::new(prog);
        c.args(args);
        c
    };
    c.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    Ok(c)
}

#[cfg(unix)]
fn private(o: &mut std::fs::OpenOptions) { use std::os::unix::fs::OpenOptionsExt; o.mode(0o700); }
#[cfg(not(unix))]
fn private(_o: &mut std::fs::OpenOptions) {}

/// Writes a new script that only the user can run. `create_new` refuses a file or link that is already there.
fn write_script(path: &Path, text: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create_new(true);
    private(&mut o);
    o.open(path)?.write_all(text.as_bytes())
}

/// Opens a terminal window and runs the command in it. Returns when the window is started, not when the command ends.
pub fn launch(command: &str, env: &Env) -> Result<(), String> {
    let command = normalise(command);
    check(&command)?;
    if command.is_empty() { return Err("No server command is set. Click the pencil icon next to Launch server and enter one.".into()); }
    let ext = if cfg!(windows) { "bat" } else if cfg!(target_os = "macos") { "command" } else { "sh" };
    let stamp = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let script = env.temp_dir.join(format!("{SCRIPT_PREFIX}{}-{stamp}.{ext}", std::process::id()));
    let mut cmd = build_command(&script, env)?; // before the file is written, so a missing terminal leaves nothing behind
    let text = if cfg!(windows) { windows_script(WINDOW_TITLE, &command) } else { unix_script(WINDOW_TITLE, &command) };
    write_script(&script, &text).map_err(|e| format!("Cannot write {}: {e}", script.display()))?;
    let mut child = cmd.spawn().map_err(|e| { let _ = std::fs::remove_file(&script); format!("Cannot start the terminal: {e}") })?;
    // Waiting also collects the finished process. On Windows `start /WAIT` ends when the window closes, and then the file goes.
    std::thread::spawn(move || { let _ = child.wait(); if cfg!(windows) { let _ = std::fs::remove_file(&script); } });
    Ok(())
}

/// Removes scripts of this program that are older than `max_age`: leftovers of a crash, or of a terminal that never ran them.
pub fn sweep_old_scripts(dir: &Path, max_age: Duration) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let name = e.file_name();
        let old = e.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).map_or(false, |age| age > max_age);
        if name.to_string_lossy().starts_with(SCRIPT_PREFIX) && old { let _ = std::fs::remove_file(e.path()); }
    }
}

/// Opens a folder in the file manager of the system.
pub fn open_folder(dir: &Path) -> Result<(), String> {
    let program = if cfg!(windows) { "explorer" } else if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
    let mut child = Command::new(program).arg(dir).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn()
        .map_err(|e| format!("Cannot open {} with {program}: {e}", dir.display()))?;
    std::thread::spawn(move || { let _ = child.wait(); });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, empty folder under the system temp dir.
    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("lt-launcher-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A small program that only exists for the test (a shell script with the execute permission).
    #[cfg(unix)]
    fn fake_program(dir: &Path, name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let p = dir.join(name);
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    // ----- the command file -----

    #[test]
    fn normalise_makes_one_form_of_every_text() {
        assert_eq!(normalise("\u{feff}  llama-server -m a.gguf\r\n--port 8080 \r\n\r\n"), "llama-server -m a.gguf\n--port 8080");
        assert_eq!(normalise("a\rb"), "a\nb");
        assert_eq!(normalise(" \n "), "");
    }

    #[test]
    fn check_refuses_nul_and_very_long_commands() {
        assert!(check("llama-server").is_ok() && check("").is_ok());
        assert!(check("a\0b").unwrap_err().contains("NUL"));
        assert!(check(&"x".repeat(MAX_COMMAND_CHARS + 1)).unwrap_err().contains("longer"));
    }

    #[test]
    fn the_app_folder_comes_first_except_on_macos() {
        let (a, u) = (Path::new("/app"), Path::new("/user"));
        assert_eq!(command_file_candidates(Some(a), Some(u), true), [a.join(COMMAND_FILE), u.join(COMMAND_FILE)]);
        assert_eq!(command_file_candidates(Some(a), Some(u), false), [u.join(COMMAND_FILE)]);
        assert_eq!(command_file_candidates(None, Some(u), true), [u.join(COMMAND_FILE)]);
        assert!(command_file_candidates(None, None, true).is_empty());
    }

    #[test]
    fn a_command_is_saved_as_plain_text_and_read_back() {
        let d = temp_dir("save");
        let c = vec![d.join("app").join(COMMAND_FILE), d.join("user").join(COMMAND_FILE)];
        assert_eq!(load_command(&c).unwrap(), StoredCommand { command: String::new(), path: c[0].clone() }, "nothing saved yet");
        let s = save_command(&c, "  llama-server -m \"C:\\my models\\a.gguf\"\r\n  --port 8080\r\n").unwrap();
        assert_eq!(s.path, c[0], "the first folder that accepts the file");
        assert_eq!(std::fs::read_to_string(&s.path).unwrap(), "llama-server -m \"C:\\my models\\a.gguf\"\n  --port 8080\n", "human readable");
        assert_eq!(load_command(&c).unwrap(), s);
        // an empty command clears it
        assert_eq!(save_command(&c, "  ").unwrap().command, "");
        assert_eq!(load_command(&c).unwrap().command, "");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_folder_that_refuses_the_file_is_skipped_and_an_existing_file_is_kept_where_it_is() {
        let d = temp_dir("fallback");
        std::fs::write(d.join("blocker"), b"a file where a folder should be").unwrap();
        let (bad, good) = (d.join("blocker").join(COMMAND_FILE), d.join("user").join(COMMAND_FILE));
        let c = vec![bad.clone(), good.clone()];
        assert_eq!(save_command(&c, "one").unwrap().path, good, "the app folder cannot be written, so the user folder is used");
        // The file now exists in the second place. A later save changes that file, even if the first place works again.
        std::fs::remove_file(d.join("blocker")).unwrap();
        let c = vec![d.join("app").join(COMMAND_FILE), good.clone()];
        assert_eq!(save_command(&c, "two").unwrap().path, good);
        assert_eq!(load_command(&c).unwrap().command, "two");
        // An existing file that cannot be written is an error. No second copy is made that the next read would ignore.
        std::fs::create_dir(d.join("user").join(format!("{COMMAND_FILE}.tmp"))).unwrap(); // blocks the temporary file
        assert!(save_command(&c, "three").is_err());
        assert_eq!(load_command(&c).unwrap().command, "two");
        assert!(!d.join("app").join(COMMAND_FILE).exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    // ----- the scripts -----

    #[test]
    fn the_windows_script_is_a_batch_file() {
        let s = windows_script("My \"server\" & 100%", "llama-server -m \"C:\\modèles\\a.gguf\" ^\n  --port 8080 && echo ok");
        assert_eq!(s, "@echo off\r\nchcp 65001 >nul\r\ntitle My server  100\r\nllama-server -m \"C:\\modèles\\a.gguf\" ^\r\n  --port 8080 && echo ok\r\n");
        assert_eq!(windows_script("", "x"), format!("@echo off\r\nchcp 65001 >nul\r\ntitle {WINDOW_TITLE}\r\nx\r\n"), "an empty title becomes the default");
    }

    #[test]
    fn the_windows_command_line_opens_a_console_and_waits_for_it() {
        let l = windows_command_line(WINDOW_TITLE, Path::new(r"C:\Users\Jo Doe\AppData\Local\Temp\x.bat"));
        assert_eq!(l, r#"/C start /WAIT "Layout Translator server" cmd /K "C:\Users\Jo Doe\AppData\Local\Temp\x.bat""#);
    }

    #[test]
    fn the_unix_script_deletes_itself_and_keeps_the_window_open() {
        let s = unix_script("It's a title", "echo one\necho two");
        assert!(s.starts_with("#!/usr/bin/env bash\nrm -f -- \"$0\"\n"));
        assert!(s.contains("printf '\\033]0;%s\\007' 'Its a title'\n"), "no quote can end the title early: {s}");
        assert!(s.contains("\necho one\necho two\nstatus=$?\n") && s.trim_end().ends_with("read -r -n 1 -s -p \"Press any key to close this window...\"\nexit \"$status\""));
    }

    #[cfg(unix)]
    #[test]
    fn the_unix_script_really_runs_the_command_and_removes_itself() {
        let d = temp_dir("run");
        let (script, marker) = (d.join("s.sh"), d.join("marker"));
        // Two commands and a pipe: all of it must run in the one shell, in order.
        let cmd = format!("echo first > '{m}' && echo second | tr a-z A-Z >> '{m}'", m = marker.display());
        write_script(&script, &unix_script(WINDOW_TITLE, &cmd)).unwrap();
        let st = Command::new("bash").arg(&script).stdin(Stdio::null()).stdout(Stdio::null()).status().unwrap();
        assert!(st.success(), "the exit code of the command is the exit code of the script");
        assert_eq!(std::fs::read_to_string(&marker).unwrap(), "first\nSECOND\n");
        assert!(!script.exists(), "the script deleted itself");
        assert!(write_script(&d.join("marker"), "x").is_err(), "an existing file is never overwritten");
        let failing = d.join("f.sh");
        write_script(&failing, &unix_script(WINDOW_TITLE, "exit 3")).unwrap();
        assert_eq!(Command::new("bash").arg(&failing).stdin(Stdio::null()).stdout(Stdio::null()).status().unwrap().code(), Some(3));
        let _ = std::fs::remove_dir_all(&d);
    }

    // ----- Linux terminals -----

    #[cfg(unix)]
    #[test]
    fn the_first_terminal_in_the_list_is_used_with_its_own_arguments() {
        let d = temp_dir("terms");
        for n in ["xterm", "konsole", "gnome-terminal"] { fake_program(&d, n, "true"); }
        std::fs::write(d.join("kitty"), "not executable").unwrap(); // a file without the execute permission is skipped
        let path = std::env::join_paths([d.as_path()]).unwrap();
        let script = Path::new("/tmp/s.sh");
        let (prog, args) = terminal_command(None, Some(&path), script).unwrap();
        assert_eq!((prog, args), (d.join("gnome-terminal"), vec![OsString::from("--"), script.into()]));
        std::fs::remove_file(d.join("gnome-terminal")).unwrap();
        assert_eq!(terminal_command(None, Some(&path), script).unwrap(), (d.join("konsole"), vec![OsString::from("-e"), script.into()]));
        std::fs::remove_file(d.join("konsole")).unwrap();
        assert_eq!(terminal_command(None, Some(&path), script).unwrap().0, d.join("xterm"), "kitty has no execute permission");
        // $TERMINAL wins, by name or by path
        fake_program(&d, "myterm", "true");
        assert_eq!(terminal_command(Some("myterm"), Some(&path), script).unwrap().0, d.join("myterm"));
        assert_eq!(terminal_command(Some(d.join("myterm").to_str().unwrap()), None, script).unwrap().0, d.join("myterm"));
        assert_eq!(terminal_command(Some("  "), Some(&path), script).unwrap().0, d.join("xterm"), "an empty value is ignored");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn no_terminal_gives_a_clear_message() {
        let e = terminal_command(None, Some(OsStr::new("")), Path::new("/tmp/s.sh")).unwrap_err();
        assert!(e.contains("No terminal program") && e.contains("TERMINAL"), "{e}");
        assert!(terminal_command(None, None, Path::new("/tmp/s.sh")).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn launch_runs_the_command_in_a_terminal_and_leaves_no_script_behind() {
        let d = temp_dir("launch");
        let (bin, tmp, marker, log) = (d.join("bin"), d.join("tmp"), d.join("marker"), d.join("args.log"));
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(&tmp).unwrap();
        // A "terminal": it records its arguments, drops the -e, and runs the script the way a real one would.
        fake_program(&bin, "x-terminal-emulator", &format!("printf '%s\\n' \"$@\" > '{}'\nshift\nexec \"$@\"", log.display()));
        let env = Env { temp_dir: tmp.clone(), path_var: Some(std::env::join_paths([bin.as_path()]).unwrap()), terminal_var: None };
        launch(&format!("echo started > '{}' && echo done >> '{}'", marker.display(), marker.display()), &env).unwrap();
        let t = std::time::Instant::now();
        while std::fs::read_to_string(&marker).map_or(true, |s| !s.contains("done")) && t.elapsed() < Duration::from_secs(10) { std::thread::sleep(Duration::from_millis(50)); }
        assert_eq!(std::fs::read_to_string(&marker).unwrap(), "started\ndone\n");
        let args = std::fs::read_to_string(&log).unwrap();
        assert!(args.starts_with("-e\n") && args.contains(SCRIPT_PREFIX) && args.trim_end().ends_with(".sh"), "{args}");
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(std::fs::read_dir(&tmp).unwrap().count(), 0, "the script removed itself");
        // Errors: nothing to run, and no terminal. Neither leaves a script in the temp folder.
        assert!(launch("  ", &env).unwrap_err().contains("No server command"));
        let none = Env { temp_dir: tmp.clone(), path_var: Some(OsString::new()), terminal_var: None };
        assert!(launch("echo x", &none).unwrap_err().contains("No terminal program"));
        assert!(launch("a\0b", &env).unwrap_err().contains("NUL"));
        assert_eq!(std::fs::read_dir(&tmp).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn only_old_scripts_of_this_program_are_swept() {
        let d = temp_dir("sweep");
        let old = SystemTime::now() - Duration::from_secs(3 * 86_400);
        let make = |name: &str, t: SystemTime| { let f = std::fs::File::create(d.join(name)).unwrap(); f.set_modified(t).unwrap(); };
        make(&format!("{SCRIPT_PREFIX}1-1.sh"), old);
        make(&format!("{SCRIPT_PREFIX}2-2.bat"), SystemTime::now());
        make("other-file.sh", old);
        sweep_old_scripts(&d, Duration::from_secs(86_400));
        let mut left: Vec<String> = std::fs::read_dir(&d).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        left.sort();
        assert_eq!(left, [format!("{SCRIPT_PREFIX}2-2.bat"), "other-file.sh".to_string()]);
        sweep_old_scripts(&d.join("missing"), Duration::from_secs(1)); // a missing folder is not an error
        let _ = std::fs::remove_dir_all(&d);
    }
}
