use std::{fmt, io};
use thiserror::Error;

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

    #[error("{0}")]
    CatalogError(String),

    #[error("Constraint Error: {0}")]
    ConstraintError(String),

    #[error("{0}")]
    TransactionContextError(String),

    #[error("An internal operation failed: {0}")]
    Internal(String),

    #[error("expression cast error: {0}")]
    ExprCast(String),

    #[error("predicate build error: {0}")]
    PredicateBuild(String),

    #[error("table id {0} is reserved for system catalogs")]
    ReservedTableId(u16),
}

impl Error {
    #[inline]
    pub fn expr_cast<E: fmt::Display>(err: E) -> Self {
        Error::ExprCast(err.to_string())
    }

    #[inline]
    pub fn predicate_build<E: fmt::Display>(err: E) -> Self {
        Error::PredicateBuild(err.to_string())
    }

    #[inline]
    pub fn reserved_table_id(table_id: impl Into<u16>) -> Self {
        Error::ReservedTableId(table_id.into())
    }
}
