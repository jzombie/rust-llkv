use std::io;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("Invalid argument: {0}")]
    InvalidArgumentError(String),

    #[error("Storage key not found")]
    NotFound,

    #[error("An internal operation failed: {0}")]
    Internal(String),
}
