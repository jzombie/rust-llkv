//! Executor table, schema, and column types.

use arrow::datatypes::DataType;
use llkv_plan::PlanValue;
use llkv_storage::pager::Pager;
use llkv_table::types::FieldId;
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};

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
    /// Executor-level schema with field ID mappings
    pub schema: Arc<ExecutorSchema>,
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

/// Schema representation for the executor.
///
/// Maintains column metadata and provides efficient name-to-column lookups.
pub struct ExecutorSchema {
    /// Ordered list of columns
    pub columns: Vec<ExecutorColumn>,
    /// Fast lookup from normalized column name to index
    pub lookup: FxHashMap<String, usize>,
}

impl ExecutorSchema {
    /// Resolve a column by name (case-insensitive).
    pub fn resolve(&self, name: &str) -> Option<&ExecutorColumn> {
        let normalized = name.to_ascii_lowercase();
        self.lookup
            .get(&normalized)
            .and_then(|idx| self.columns.get(*idx))
    }

    /// Get the field ID of the first column, if any.
    pub fn first_field_id(&self) -> Option<FieldId> {
        self.columns.first().map(|col| col.field_id)
    }

    /// Find a column by its field ID.
    pub fn column_by_field_id(&self, field_id: FieldId) -> Option<&ExecutorColumn> {
        self.columns.iter().find(|col| col.field_id == field_id)
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

/// Column metadata used by the executor.
#[derive(Clone)]
pub struct ExecutorColumn {
    /// Column name
    pub name: String,
    /// Arrow data type
    pub data_type: DataType,
    /// Whether the column allows NULL values
    pub nullable: bool,
    /// Whether this column is part of the primary key
    pub primary_key: bool,
    /// Whether this column has a unique constraint
    pub unique: bool,
    /// Storage-level field ID
    pub field_id: FieldId,
    /// Optional CHECK constraint expression
    pub check_expr: Option<String>,
}

/// Multi-column unique constraint metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutorMultiColumnUnique {
    /// Optional name for the unique constraint
    pub index_name: Option<String>,
    /// Indices of columns in the schema that form the unique constraint
    pub column_indices: Vec<usize>,
}
