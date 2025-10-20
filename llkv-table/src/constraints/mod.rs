//! Unified constraint module for constraint management.
//!
//! This module provides a single interface for all constraint operations:
//! - `types`: Constraint types and definitions (ConstraintRecord, ConstraintKind, etc.)
//! - `service`: ConstraintService for enforcing constraints
//! - `validation`: Validation functions for uniqueness, foreign keys, checks, etc.
//!
//! External code should use this `constraint` module instead of accessing individual components.

mod types;
mod service;
mod validation;

// Re-export constraint types
pub use types::{
    CheckConstraint, ConstraintExpressionRef, ConstraintId, ConstraintKind, ConstraintRecord,
    ConstraintState, ForeignKeyAction, ForeignKeyConstraint, PrimaryKeyConstraint,
    UniqueConstraint, decode_constraint_row_id, encode_constraint_row_id,
};

// Re-export ConstraintService and related types
pub use service::{
    ConstraintService, ForeignKeyChildRowsFetch, ForeignKeyParentRowsFetch, ForeignKeyRowFetch,
    InsertColumnConstraint, InsertMultiColumnUnique, InsertUniqueColumn,
};

// Re-export validation functions and types
pub use validation::{
    ConstraintColumnInfo, ForeignKeyColumn, ForeignKeyTableInfo, UniqueKey, ValidatedForeignKey,
    build_composite_unique_key, ensure_multi_column_unique, ensure_primary_key,
    ensure_single_column_unique, unique_key_component, validate_check_constraints,
    validate_foreign_key_rows, validate_foreign_keys,
};
