//! Executor table, schema, and column types.

use arrow::record_batch::RecordBatch;
use croaring::Treemap;
use llkv_plan::PlanValue;
use llkv_plan::physical::table::ExecutionTable;
use llkv_plan::schema::{PlanColumn, PlanSchema};
use llkv_storage::pager::Pager;
use llkv_table::types::FieldId;
use simd_r_drive_entry_handle::EntryHandle;
use std::any::Any;
use std::fmt;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};

use crate::types::StorageTable;
use llkv_column_map::store::GatherNullPolicy;
use llkv_expr::Expr;
use llkv_scan::{ScanProjection, ScanStreamOptions};
use llkv_types::LogicalFieldId;

/// Executor's view of a table, including schema and metadata.
///
/// This wraps the underlying `llkv_table::Table` and provides executor-specific
/// metadata like next row ID tracking and multi-column unique constraints.
pub struct ExecutorTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Underlying physical table from llkv-table
    pub table: Arc<llkv_table::table::Table<P>>,
    /// Storage adapter used by the executor; initially wraps `table`.
    pub storage: Arc<dyn StorageTable<P>>,
    /// Executor-level schema with field ID mappings
    pub schema: Arc<PlanSchema>,
    /// Next available row ID for inserts
    pub next_row_id: AtomicU64,
    /// Total number of rows in the table
    pub total_rows: AtomicU64,
    /// Multi-column unique constraints
    pub multi_column_uniques: RwLock<Vec<ExecutorMultiColumnUnique>>,
}

impl<P> ExecutorTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Get a snapshot of multi-column unique constraints.
    pub fn multi_column_uniques(&self) -> Vec<ExecutorMultiColumnUnique> {
        self.multi_column_uniques.read().unwrap().clone()
    }

    /// Get the storage abstraction for scans/joins.
    pub fn storage(&self) -> Arc<dyn StorageTable<P>> {
        Arc::clone(&self.storage)
    }

    /// Convenience accessor for the table's identifier.
    pub fn table_id(&self) -> llkv_table::types::TableId {
        self.storage.table_id()
    }

    /// Collect row ids matching a predicate using the storage abstraction.
    pub fn filter_row_ids<'expr>(
        &self,
        filter_expr: &Expr<'expr, FieldId>,
    ) -> llkv_result::Result<Treemap> {
        self.storage.filter_row_ids(filter_expr)
    }

    /// Stream specific columns for a set of row ids using the storage abstraction.
    pub fn stream_columns<'table>(
        &'table self,
        logical_fields: impl Into<Arc<[LogicalFieldId]>>,
        row_ids: impl Into<llkv_scan::row_stream::RowIdSource>,
        policy: GatherNullPolicy,
    ) -> llkv_result::Result<llkv_table::table::TableScanStream<'table, P>> {
        self.table.stream_columns(logical_fields, row_ids, policy)
    }

    /// Replace all multi-column unique constraints.
    pub fn set_multi_column_uniques(&self, uniques: Vec<ExecutorMultiColumnUnique>) {
        *self.multi_column_uniques.write().unwrap() = uniques;
    }

    /// Add a new multi-column unique constraint if it doesn't already exist.
    pub fn add_multi_column_unique(&self, unique: ExecutorMultiColumnUnique) {
        let mut guard = self.multi_column_uniques.write().unwrap();
        if !guard
            .iter()
            .any(|existing| existing.column_indices == unique.column_indices)
        {
            guard.push(unique);
        }
    }
}

impl<P> ExecutionTable<P> for ExecutorTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn schema(&self) -> Arc<PlanSchema> {
        Arc::clone(&self.schema)
    }

    fn table_id(&self) -> llkv_types::ids::TableId {
        self.storage.table_id()
    }

    fn scan_stream(
        &self,
        projections: &[ScanProjection],
        predicate: &Expr<'static, FieldId>,
        options: ScanStreamOptions<P>,
        callback: &mut dyn FnMut(RecordBatch),
    ) -> Result<(), String> {
        self.table
            .scan_stream(projections, predicate, options, callback)
            .map_err(|e| e.to_string())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Simplified row batch for export/import operations.
///
/// This represents tabular data as a list of column names and a 2D vector of values.
/// Each inner vector represents one row, and values are in the same order as the column names.
pub struct ExecutorRowBatch {
    /// Column names in order
    pub columns: Vec<String>,
    /// Rows of data, where each row is a vector of values matching the column order
    pub rows: Vec<Vec<PlanValue>>,
}

/// Multi-column unique constraint metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutorMultiColumnUnique {
    /// Optional name for the unique constraint
    pub index_name: Option<String>,
    /// Indices of columns in the schema that form the unique constraint
    pub column_indices: Vec<usize>,
}

pub type ExecutorColumn = PlanColumn;
pub type ExecutorSchema = PlanSchema;

impl<P> fmt::Debug for ExecutorTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutorTable")
            .field("schema", &self.schema)
            .field("next_row_id", &self.next_row_id)
            .field("row_count", &self.total_rows)
            .finish_non_exhaustive()
    }
}
