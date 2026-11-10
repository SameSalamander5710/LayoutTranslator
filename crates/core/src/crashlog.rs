//! Writes a panic to a log file. The release build has no console (`windows_subsystem = "windows"`), so without this an
//! unexpected panic closes the app without a trace and there is nothing to attach to a bug report.
use std::io::Write;
use std::path::{Path, PathBuf};

/// Files above this size are started afresh, so the log cannot grow without limit.
const MAX_LOG_BYTES: u64 = 512 * 1024;

/// From now on every panic is appended to `file` (message, place, thread and a backtrace) before the previous panic hook
/// runs. Failing to write the log is ignored: the panic itself matters more.
pub fn install(file: PathBuf) {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        write_entry(&file, &info.to_string());
        previous(info);
    }));
}

fn write_entry(file: &Path, what: &str) {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let thread = std::thread::current();
    let text = format!(
        "--- {} {} (unix time {secs}), thread {}\n{what}\nbacktrace:\n{}\n\n",
        env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"), thread.name().unwrap_or("<unnamed>"), std::backtrace::Backtrace::force_capture(),
    );
    if let Some(dir) = file.parent() { let _ = std::fs::create_dir_all(dir); }
    if std::fs::metadata(file).map(|m| m.len() > MAX_LOG_BYTES).unwrap_or(false) { let _ = std::fs::remove_file(file); }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(file) { let _ = f.write_all(text.as_bytes()); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_is_written_to_the_log_file() {
        let dir = std::env::temp_dir().join(format!("lt-crashlog-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let file = dir.join("sub").join("last-error.log"); // the folder does not exist yet
        install(file.clone());
        let r = std::panic::catch_unwind(|| { panic!("forced panic for the test"); });
        let _ = std::panic::take_hook(); // back to the default hook for the other tests
        assert!(r.is_err());
        let log = std::fs::read_to_string(&file).expect("the log file was created");
        assert!(log.contains("forced panic for the test") && log.contains("crashlog.rs") && log.contains("backtrace:"), "{log}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
