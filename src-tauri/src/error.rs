//! The error that the Tauri commands send to the UI: a `kind` the UI can branch on, and a `message` for the person.
//! Code below the commands (the pipeline, the launcher) keeps returning plain `String` errors; `From<String>` turns
//! those into `AppError` with the kind `Other`, so `?` works in a command. A command names a kind only where the UI
//! does something different for it.

use serde::Serialize;

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ErrorKind {
    /// The user pressed Stop.
    Cancelled,
    /// The OCR model of the language is missing or cannot be used.
    NoModel,
    /// The translation server could not be reached, or its answer could not be used.
    Server,
    /// Anything else (files, bad input, ...): only the message helps.
    Other,
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct AppError {
    pub kind: ErrorKind,
    pub message: String,
}

impl AppError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self { Self { kind, message: message.into() } }
    pub fn cancelled() -> Self { Self::new(ErrorKind::Cancelled, "Cancelled") }
    pub fn no_model(message: impl Into<String>) -> Self { Self::new(ErrorKind::NoModel, message) }
    pub fn server(message: impl Into<String>) -> Self { Self::new(ErrorKind::Server, message) }
}

impl From<String> for AppError { fn from(message: String) -> Self { Self::new(ErrorKind::Other, message) } }
impl From<&str> for AppError { fn from(message: &str) -> Self { Self::new(ErrorKind::Other, message) } }

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.message) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ui_receives_the_kind_and_the_message_with_these_names() {
        let json = serde_json::to_string(&AppError::cancelled()).unwrap();
        assert_eq!(json, r#"{"kind":"cancelled","message":"Cancelled"}"#);
        for (kind, name) in [(ErrorKind::NoModel, "noModel"), (ErrorKind::Server, "server"), (ErrorKind::Other, "other")] {
            let v = serde_json::to_value(AppError::new(kind, "x")).unwrap();
            assert_eq!(v["kind"], name);
        }
    }

    #[test]
    fn a_plain_string_error_becomes_the_kind_other_with_the_same_text() {
        let e: AppError = String::from("disk full").into();
        assert_eq!((e.kind, e.message.as_str()), (ErrorKind::Other, "disk full"));
        let e: AppError = "no models folder".into();
        assert_eq!(e.kind, ErrorKind::Other);
        assert_eq!(e.to_string(), "no models folder");
    }

    #[test]
    fn question_mark_converts_string_errors_in_a_function_that_returns_app_error() {
        fn inner() -> Result<(), String> { Err("from the pipeline".into()) }
        fn command() -> Result<(), AppError> { inner()?; Ok(()) }
        assert_eq!(command().unwrap_err().message, "from the pipeline");
    }
}
