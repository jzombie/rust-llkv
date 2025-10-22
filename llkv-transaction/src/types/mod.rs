//! Transaction types and result enums.
//!
//! This module contains the core type definitions for transaction operations,
//! including result types and catalog snapshot interfaces.

pub mod catalog;
pub mod kind;
pub mod result;

// Re-export commonly used items
pub use catalog::TransactionCatalogSnapshot;
pub use kind::TransactionKind;
pub use result::TransactionResult;
