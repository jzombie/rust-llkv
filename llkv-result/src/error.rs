use std::{fmt, io};
use thiserror::Error;

/// Unified error type for all LLKV operations.
///
/// This enum encompasses all failure modes across the LLKV stack, from low-level storage
/// errors to high-level transaction conflicts. Each variant includes context-specific
/// information to help diagnose and handle the error appropriately.
///
/// # Error Handling Strategy
///
/// Errors propagate upward through the call stack using Rust's `?` operator. At API boundaries
/// (e.g., SQL interface), errors are typically converted to user-friendly messages. Internal
/// code can match on specific variants for fine-grained error handling.
///
/// # Thread Safety
///
/// `Error` implements `Send` and `Sync`, allowing errors to be safely passed between threads.
/// This is important for concurrent query execution and transaction processing.
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error during file or disk operations.
    ///
    /// This error wraps standard library I/O errors and typically occurs during:
    /// - Opening or creating database files
    /// - Reading or writing pages to disk
    /// - Flushing data to persistent storage
    ///
    /// The underlying `io::Error` provides detailed information about the failure
    /// (e.g., permission denied, disk full, file not found).
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Arrow library error during columnar data operations.
    ///
    /// This error occurs when:
    /// - Serializing or deserializing Arrow RecordBatches
    /// - Converting between Arrow data types
    /// - Building Arrow arrays with invalid data
    /// - Schema mismatches during batch operations
    ///
    /// Arrow is the underlying columnar memory format used by LLKV, so these errors
    /// typically indicate data format incompatibilities or memory allocation failures.
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    /// Invalid user input or API parameter.
    ///
    /// This error indicates a problem with arguments passed to LLKV APIs:
    /// - Invalid SQL syntax (e.g., malformed queries)
    /// - Type mismatches (e.g., comparing incompatible column types)
    /// - Out-of-range values (e.g., negative row counts)
    /// - Malformed identifiers (e.g., invalid table names)
    /// - Schema violations (e.g., missing required columns)
    ///
    /// The message string provides specific details about what was invalid and why.
    ///
    /// # Recovery
    ///
    /// These errors are typically recoverable—fix the input and retry the operation.
    #[error("Invalid argument: {0}")]
    InvalidArgumentError(String),

    /// Storage key or entity not found.
    ///
    /// This error occurs when attempting to access a non-existent:
    /// - Table (by name or ID)
    /// - Column (by name or field ID)
    /// - Row (by row_id)
    /// - Chunk or page (by physical key)
    /// - Descriptor or metadata entry
    ///
    /// This is a common error when queries reference dropped tables or columns,
    /// or when low-level storage operations fail to locate expected data.
    ///
    /// # Recovery
    ///
    /// For user-facing errors, present a "table/column not found" message.
    /// For internal errors, this may indicate data corruption or a bug.
    #[error("Storage key not found")]
    NotFound,

    /// Catalog metadata error.
    ///
    /// This error indicates a problem with system catalog operations:
    /// - Corruption in catalog tables (`__llkv_catalog_*`)
    /// - Inconsistency between catalog and actual stored data
    /// - Failure to read or update table/column metadata
    /// - Schema evolution issues
    ///
    /// Catalog errors are serious as they affect the database's understanding of
    /// its own structure. They may require manual inspection or repair.
    #[error("{0}")]
    CatalogError(String),

    /// Data constraint violation.
    ///
    /// This error occurs when an operation would violate data integrity rules:
    /// - Primary key conflicts (duplicate key on insert)
    /// - Type constraints (inserting wrong type into column)
    /// - NOT NULL constraint violations
    /// - Foreign key violations (if implemented)
    /// - Check constraint failures (if implemented)
    ///
    /// The message describes which constraint was violated and what data caused the issue.
    ///
    /// # Recovery
    ///
    /// These errors are expected during normal operation (e.g., duplicate key inserts).
    /// The application should handle them gracefully and inform the user.
    #[error("Constraint Error: {0}")]
    ConstraintError(String),

    /// Transaction execution or isolation error.
    ///
    /// This error occurs during MVCC transaction processing:
    /// - Transaction conflicts during commit (write-write conflicts)
    /// - Isolation violations (attempting to see uncommitted data)
    /// - Transaction abort requests
    /// - Invalid transaction state transitions
    /// - Exceeding transaction limits
    ///
    /// These errors are part of normal transaction processing and indicate that
    /// a transaction must be retried or aborted.
    #[error("{0}")]
    TransactionContextError(String),

    /// Internal error indicating a bug or unexpected state.
    ///
    /// This error should never occur during normal operation. It indicates:
    /// - Violated internal invariants
    /// - Unexpected state transitions
    /// - Logic errors in LLKV code
    /// - Data structure corruption
    ///
    /// If you encounter this error, it likely indicates a bug in LLKV that should
    /// be reported with reproduction steps.
    ///
    /// # Debugging
    ///
    /// The message includes details about what assertion failed or what unexpected
    /// state was encountered. Enable debug logging for more context.
    #[error("An internal operation failed: {0}")]
    Internal(String),

    /// Expression type casting error.
    ///
    /// This error occurs when evaluating SQL expressions that require type conversions:
    /// - Invalid casts (e.g., string to integer when string isn't numeric)
    /// - Unsupported type conversions
    /// - Overflow during numeric conversions
    ///
    /// This is typically a user error (invalid SQL expression) rather than a bug.
    #[error("expression cast error: {0}")]
    ExprCast(String),

    /// Predicate or filter construction error.
    ///
    /// This error occurs when building scan predicates or filter expressions:
    /// - Type mismatches in comparison operations
    /// - Unsupported predicate types for columnar scans
    /// - Invalid filter push-down operations
    ///
    /// These errors typically occur during query planning when attempting to
    /// optimize filters into storage-layer scans.
    #[error("predicate build error: {0}")]
    PredicateBuild(String),

    /// Attempted to use a reserved table ID for a user table.
    ///
    /// Currently, only table ID 0 is reserved for the system catalog.
    /// This error prevents accidental corruption of system metadata.
    ///
    /// User tables receive IDs starting from 1 and incrementing.
    #[error("table id {0} is reserved for system catalogs")]
    ReservedTableId(u16),
}

impl Error {
    /// Create an expression cast error from any displayable error.
    ///
    /// This is a convenience method for converting other error types into
    /// [`Error::ExprCast`] while preserving the original error message.
    ///
    /// # Examples
    ///
    /// ```
    /// use llkv_result::Error;
    ///
    /// fn parse_number(input: &str) -> Result<u32, Error> {
    ///     input.parse::<u32>().map_err(Error::expr_cast)
    /// }
    ///
    /// assert_eq!(parse_number("42").unwrap(), 42);
    /// assert!(matches!(parse_number("abc"), Err(Error::ExprCast(_))));
    /// ```
    #[inline]
    pub fn expr_cast<E: fmt::Display>(err: E) -> Self {
        Error::ExprCast(err.to_string())
    }

    /// Create a predicate build error from any displayable error.
    ///
    /// This is a convenience method for converting other error types into
    /// [`Error::PredicateBuild`] while preserving the original error message.
    ///
    /// # Examples
    ///
    /// ```
    /// use llkv_result::Error;
    ///
    /// fn unsupported_predicate() -> Result<(), Error> {
    ///     let io_err = std::io::Error::new(std::io::ErrorKind::Other, "not supported");
    ///     Err(Error::predicate_build(io_err))
    /// }
    ///
    /// let err = unsupported_predicate().unwrap_err();
    /// assert!(matches!(err, Error::PredicateBuild(msg) if msg.contains("not supported")));
    /// ```
    #[inline]
    pub fn predicate_build<E: fmt::Display>(err: E) -> Self {
        Error::PredicateBuild(err.to_string())
    }

    /// Create a reserved table ID error.
    ///
    /// Currently only table ID 0 is reserved. This method creates an error
    /// when user code attempts to use a reserved ID.
    #[inline]
    pub fn reserved_table_id(table_id: impl Into<u16>) -> Self {
        Error::ReservedTableId(table_id.into())
    }
}
