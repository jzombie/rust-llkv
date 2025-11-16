//! Unified constraint module for constraint management.
//!
//! This module provides a single interface for all constraint operations:
//! - `types`: Constraint types and definitions (ConstraintRecord, ConstraintKind, etc.)
//! - `service`: ConstraintService for enforcing constraints
//! - `validation`: Validation functions for uniqueness, foreign keys, checks, etc.
//!
//! External code should use this `constraint` module instead of accessing individual components.

mod service;
mod types;
mod validation;

pub use types::{
    CheckConstraint, ConstraintEnforcementMode, ConstraintExpressionRef, ConstraintId,
    ConstraintKind, ConstraintRecord, ConstraintState, ForeignKeyAction, ForeignKeyConstraint,
    PrimaryKeyConstraint, UniqueConstraint, decode_constraint_row_id, encode_constraint_row_id,
};

pub use service::{
    ConstraintService, ForeignKeyChildRowsFetch, ForeignKeyParentRowsFetch, ForeignKeyRowFetch,
    InsertColumnConstraint, InsertMultiColumnUnique, InsertUniqueColumn,
};

pub use validation::{
    ConstraintColumnInfo, ForeignKeyColumn, ForeignKeyTableInfo, UniqueKey, ValidatedForeignKey,
    build_composite_unique_key, column_in_foreign_keys, column_in_multi_column_unique,
    column_in_primary_or_unique, ensure_multi_column_unique, ensure_primary_key,
    ensure_single_column_unique, unique_key_component, validate_alter_table_operation,
    validate_check_constraints, validate_foreign_key_rows, validate_foreign_keys,
};
