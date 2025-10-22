//! Planner data structures shared between SQL parsing and execution.
//!
//! The crate exposes:
//! - [`plan_graph`] for the serialized DAG representation exchanged with tooling.
//! - [`plans`] for high-level logical plan structures emitted by the SQL layer.
//! - [`validation`] helpers that enforce naming and schema invariants while plans
//!   are being assembled.
//! - [`conversion`] utilities for converting SQL AST nodes to Plan types.
//!
//! Modules are re-exported so downstream crates can `use llkv_plan::*` when they
//! only need a subset of the functionality.
#![forbid(unsafe_code)]

pub mod conversion;
pub mod plan_graph;
pub mod plans;
pub mod validation;

pub use conversion::{extract_rows_from_range, plan_value_from_sql_expr, plan_value_from_sql_value, RangeSelectRows};
pub use plan_graph::*;
pub use plans::*;
