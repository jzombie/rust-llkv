//! Table abstraction and system catalog for LLKV.
//!
//! This crate provides the [`Table`] type, which builds on [`llkv-column-map`]'s
//! columnar storage to offer a higher-level, schema-aware interface. It includes:
//!
//! - **[`Table`]**: Schema-aware table abstraction with append, scan, and schema operations
//! - **[`SysCatalog`]**: System catalog (table 0) that stores table and column metadata
//! - **[`TableMeta`]** and **[`ColMeta`]**: Metadata structures for tables and columns
//! - **Schema management**: Arrow schema integration with field ID tracking
//! - **Scan operations**: Projection, filtering, ordering, and computed columns
//! - **MVCC integration**: Automatic handling of `created_by` and `deleted_by` columns
//!
//! # Architecture
//!
//! Tables use [`ColumnStore`](llkv_column_map::ColumnStore) for physical storage but add:
//! - Schema validation and enforcement
//! - Field ID assignment and tracking
//! - System catalog for metadata persistence
//! - MVCC column management
//! - Row ID filtering (for transaction visibility)
//!
//! # Table IDs
//!
//! - **Table 0**: Reserved for the system catalog (stores [`TableMeta`] and [`ColMeta`])
//! - **Tables 1+**: User tables
//!
//! See [`CATALOG_TABLE_ID`] and [`is_reserved_table_id`](reserved::is_reserved_table_id).
//!
//! # System Catalog
//!
//! The [`SysCatalog`] stores table metadata in table 0, which is a reserved system table.
#![forbid(unsafe_code)]

pub mod catalog;
pub mod constants;
pub mod constraints;
pub mod gather;
pub mod metadata;
mod planner;
pub mod reserved;
mod scalar_eval;
pub mod schema_ext;
mod sys_catalog;
pub mod expr {
    pub use llkv_expr::expr::*;
}

pub mod table;
pub mod types;

pub mod stream;

pub use constraints::{
    CheckConstraint, ConstraintExpressionRef, ConstraintId, ConstraintKind, ConstraintRecord,
    ConstraintState, ForeignKeyAction, ForeignKeyConstraint, PrimaryKeyConstraint,
    UniqueConstraint, decode_constraint_row_id, encode_constraint_row_id,
};
pub use metadata::MetadataManager;
pub use reserved::CATALOG_TABLE_ID;
pub use stream::{ColumnStream, ColumnStreamBatch};
pub use sys_catalog::{
    ColMeta, MultiColumnUniqueEntryMeta, SysCatalog, TableMeta, TableMultiColumnUniqueMeta,
};
pub use table::Table;
pub use types::{FieldId, RowId};

pub use planner::plan_graph::{
    PLAN_GRAPH_VERSION, PlanAnnotations, PlanEdge, PlanEdgeMetadata, PlanExpression, PlanField,
    PlanGraph, PlanGraphBuilder, PlanGraphError, PlanGraphResult, PlanGraphVersion, PlanInput,
    PlanNode, PlanNodeId, PlanOperator,
};
