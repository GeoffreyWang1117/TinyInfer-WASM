use std::fmt;
use wasm_bindgen::JsValue;

/// Error types for TinyInfer
#[derive(Debug, Clone)]
pub enum TinyInferError {
    /// Invalid tensor shape
    InvalidShape(String),

    /// Invalid tensor data
    InvalidData(String),

    /// Unsupported operation
    UnsupportedOp(String),

    /// Model loading error
    ModelLoadError(String),

    /// Inference error
    InferenceError(String),

    /// Memory allocation error
    MemoryError(String),

    /// Other errors
    Other(String),
}

impl fmt::Display for TinyInferError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TinyInferError::InvalidShape(msg) => write!(f, "Invalid shape: {}", msg),
            TinyInferError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            TinyInferError::UnsupportedOp(msg) => write!(f, "Unsupported operation: {}", msg),
            TinyInferError::ModelLoadError(msg) => write!(f, "Model load error: {}", msg),
            TinyInferError::InferenceError(msg) => write!(f, "Inference error: {}", msg),
            TinyInferError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            TinyInferError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for TinyInferError {}

/// Convert TinyInferError to JsValue for WASM interop
impl From<TinyInferError> for JsValue {
    fn from(err: TinyInferError) -> Self {
        JsValue::from_str(&err.to_string())
    }
}

/// Result type alias
pub type Result<T> = std::result::Result<T, TinyInferError>;
