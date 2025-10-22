//! Expression and query plan translation utilities.
//!
//! This module handles translation from string-based expressions to field-ID-based expressions
//! that can be executed against the storage layer. It also manages projection building and
//! schema construction for query results.

pub mod expression;
pub mod projection;
pub mod schema;

// Re-export commonly used items
pub use expression::{
    full_table_scan_filter, resolve_field_id_from_schema, translate_predicate,
    translate_predicate_with, translate_scalar, translate_scalar_with,
};
pub use projection::{build_projected_columns, build_wildcard_projections};
pub use schema::schema_for_projections;
