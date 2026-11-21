use thiserror::Error;

pub type Result<T> = std::result::Result<T, SmallTideError>;

#[derive(Debug, Error)]
pub enum SmallTideError {
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("backend error ({backend}): {message}")]
    Backend { backend: &'static str, message: String },

    #[error("out of VRAM: requested {requested} bytes, available {available} bytes")]
    OutOfVram { requested: usize, available: usize },

    #[error("out of host RAM budget: requested {requested} bytes, budget {budget} bytes")]
    OutOfRamBudget { requested: usize, budget: usize },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid model format: {0}")]
    InvalidModel(String),

    #[error("unsupported operation: {0}")]
    Unsupported(String),
}
