//! Common data types for the LLKV toolkit.
//!
//! This crate hosts the core scalar types used throughout the system,
//! decoupled from the expression AST (`llkv-expr`) and compute kernels (`llkv-compute`).

pub mod canonical;
pub mod decimal;
pub mod ids;
pub mod interval;
pub mod literal;
pub mod query_context;

pub use canonical::{CanonicalRow, CanonicalScalar};
pub use decimal::{DecimalError, DecimalValue};
pub use ids::{
    FieldId, LogicalFieldId, LogicalStorageNamespace, ROW_ID_FIELD_ID, RowId, TableId, lfid,
    rid_col, rid_table,
};
pub use interval::IntervalValue;
pub use literal::{FromLiteral, Literal, LiteralCastError, LiteralExt};
pub use query_context::{QueryContext, QueryContextHandle};
