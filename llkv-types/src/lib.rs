//! Common data types for the LLKV toolkit.
//!
//! This crate hosts the core scalar types used throughout the system,
//! decoupled from the expression AST (`llkv-expr`) and compute kernels (`llkv-compute`).

pub mod decimal;
pub mod ids;
pub mod interval;
pub mod literal;

pub use decimal::{DecimalError, DecimalValue};
pub use ids::{FieldId, LogicalFieldId, Namespace, ROW_ID_FIELD_ID, RowId, TableId};
pub use interval::IntervalValue;
pub use literal::Literal;
