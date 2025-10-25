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
pub mod ddl;
pub mod gather;
pub mod metadata;
mod planner;
pub mod reserved;
pub mod resolvers;
mod scalar_eval;
pub mod schema_ext;
mod sys_catalog;
pub mod view;
pub mod expr {
    pub use llkv_expr::expr::*;
}

pub mod table;
pub mod types;

pub mod stream;

pub use catalog::{CatalogManager, CreateTableResult, FieldResolver, TableCatalogSnapshot};
pub use catalog::{SingleColumnIndexDescriptor, SingleColumnIndexRegistration};
pub use constraints::{
    CheckConstraint, ConstraintColumnInfo, ConstraintExpressionRef, ConstraintId, ConstraintKind,
    ConstraintRecord, ConstraintService, ConstraintState, ForeignKeyAction,
    ForeignKeyChildRowsFetch, ForeignKeyColumn, ForeignKeyConstraint, ForeignKeyParentRowsFetch,
    ForeignKeyRowFetch, ForeignKeyTableInfo, InsertColumnConstraint, InsertMultiColumnUnique,
    InsertUniqueColumn, PrimaryKeyConstraint, UniqueConstraint, UniqueKey, ValidatedForeignKey,
    build_composite_unique_key, column_in_foreign_keys, column_in_multi_column_unique,
    column_in_primary_or_unique, decode_constraint_row_id, encode_constraint_row_id,
    ensure_multi_column_unique, ensure_primary_key, ensure_single_column_unique,
    unique_key_component, validate_alter_table_operation, validate_check_constraints,
    validate_foreign_key_rows, validate_foreign_keys,
};
pub use ddl::CatalogDdl;
pub use metadata::MultiColumnUniqueRegistration;
pub use metadata::{ForeignKeyDescriptor, MetadataManager};
pub use reserved::CATALOG_TABLE_ID;
pub use resolvers::{canonical_table_name, resolve_table_name};
pub use stream::{ColumnStream, ColumnStreamBatch};
pub use sys_catalog::{
    ColMeta, CustomTypeMeta, MultiColumnIndexEntryMeta, SingleColumnIndexEntryMeta, SysCatalog,
    TableMeta, TableMultiColumnIndexMeta, TableSingleColumnIndexMeta,
};
pub use table::Table;
pub use types::{FieldId, ROW_ID_FIELD_ID, RowId, TableColumn, TableId};
pub use view::{ForeignKeyView, TableConstraintSummaryView, TableView};

pub use planner::plan_graph::{
    PLAN_GRAPH_VERSION, PlanAnnotations, PlanEdge, PlanEdgeMetadata, PlanExpression, PlanField,
    PlanGraph, PlanGraphBuilder, PlanGraphError, PlanGraphResult, PlanGraphVersion, PlanInput,
    PlanNode, PlanNodeId, PlanOperator,
};
pub use scalar_eval::{NumericArray, NumericArrayMap, NumericKernels};
