//! Transaction kind types.

/// Transaction kind enum representing transaction control operations.
#[derive(Clone, Debug)]
pub enum TransactionKind {
    /// Begin a new transaction
    Begin,
    /// Commit the current transaction
    Commit,
    /// Rollback the current transaction
    Rollback,
}
