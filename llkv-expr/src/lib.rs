//! Shared expression utilities for building typed predicates over Arrow data.
//!
//! The crate exposes three building blocks:
//! - [`expr`] defines a logical predicate AST that is independent of concrete
//!   Arrow scalar types.
//! - [`literal`] holds untyped literal values plus conversion helpers.
//! - [`typed_predicate`] turns logical operators into type-aware predicates that
//!   can be evaluated without additional allocations.
//!
//! All modules are re-exported from the crate root so downstream users can pull
//! the pieces they need with a single import.
pub mod expr;
pub use expr::*;

// Note: For API simplicity these are also exported out of `expr`.
pub mod literal;
pub mod typed_predicate;
pub mod decimal;

pub use decimal::*;
