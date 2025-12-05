use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};

use arrow::datatypes::DataType;
use croaring::Treemap;
use llkv_plan::PlanValue;
use llkv_plan::translation::SchemaView;
use llkv_plan::schema::{PlanColumn, PlanSchema};
use llkv_result::Result as LlkvResult;
use llkv_storage::pager::Pager;
use llkv_table::types::FieldId;
use llkv_types::{LogicalFieldId, TableId};
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;

use crate::types::StorageTable;

/// Executor-side view of a table, carrying schema metadata and scan helpers.
pub struct ExecutorTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Physical table handle
    pub table: Arc<llkv_table::table::Table<P>>,
    /// Storage abstraction used by scans and joins
    pub storage: Arc<dyn StorageTable<P>>,
    /// Logical schema used by the executor
    pub schema: Arc<ExecutorSchema>,
    /// Next row id (runtime-managed)
    pub next_row_id: AtomicU64,
    /// Total number of rows (runtime-managed)
    pub total_rows: AtomicU64,
    /// Multi-column unique constraints
    pub multi_column_uniques: RwLock<Vec<ExecutorMultiColumnUnique>>,
}

impl<P> ExecutorTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn table_id(&self) -> TableId {
        self.storage.table_id()
    }

    pub fn multi_column_uniques(&self) -> Vec<ExecutorMultiColumnUnique> {
        self.multi_column_uniques.read().expect("multi-column uniques lock poisoned").clone()
    }

    pub fn set_multi_column_uniques(&self, uniques: Vec<ExecutorMultiColumnUnique>) {
        *self.multi_column_uniques.write().expect("multi-column uniques lock poisoned") = uniques;
    }

    pub fn add_multi_column_unique(&self, unique: ExecutorMultiColumnUnique) {
        let mut guard = self.multi_column_uniques.write().expect("multi-column uniques lock poisoned");
        if !guard.iter().any(|u| u.column_indices == unique.column_indices) {
            guard.push(unique);
        }
    }

    pub fn filter_row_ids<'expr>(
        &self,
        filter_expr: &llkv_expr::Expr<'expr, FieldId>,
    ) -> LlkvResult<Treemap> {
        self.storage.filter_row_ids(filter_expr)
    }

    pub fn stream_columns<'table>(
        &'table self,
        logical_fields: impl Into<Arc<[LogicalFieldId]>>,
        row_ids: impl Into<llkv_scan::row_stream::RowIdSource>,
        policy: llkv_column_map::store::GatherNullPolicy,
    ) -> LlkvResult<llkv_table::table::TableScanStream<'table, P>> {
        self.table.stream_columns(logical_fields, row_ids, policy)
    }

    pub fn storage(&self) -> Arc<dyn StorageTable<P>> {
        Arc::clone(&self.storage)
    }
}

/// Column metadata understood by the executor.
#[derive(Clone, Debug)]
pub struct ExecutorColumn {
    pub name: String,
    pub data_type: DataType,
    pub is_nullable: bool,
    pub is_primary_key: bool,
    pub is_unique: bool,
    pub default_value: Option<PlanValue>,
    pub field_id: FieldId,
    pub check_expr: Option<String>,
}

/// Schema wrapper with fast name lookups.
#[derive(Debug, Clone)]
pub struct ExecutorSchema {
    pub columns: Vec<ExecutorColumn>,
    pub name_to_index: FxHashMap<String, usize>,
}

impl ExecutorSchema {
    pub fn column_by_name(&self, name: &str) -> Option<&ExecutorColumn> {
        self.name_to_index
            .get(&name.to_ascii_lowercase())
            .and_then(|idx| self.columns.get(*idx))
    }

    pub fn resolve(&self, name: &str) -> Option<&ExecutorColumn> {
        self.name_to_index
            .get(&name.to_ascii_lowercase())
            .and_then(|idx| self.columns.get(*idx))
    }

    pub fn first_field_id(&self) -> Option<FieldId> {
        self.columns.first().map(|c| c.field_id)
    }

    pub fn column_by_field_id(&self, field_id: FieldId) -> Option<&ExecutorColumn> {
        self.columns.iter().find(|c| c.field_id == field_id)
    }

    pub fn to_plan_schema(&self) -> PlanSchema {
        let mut name_to_index = FxHashMap::default();
        let mut columns = Vec::with_capacity(self.columns.len());

        for (idx, col) in self.columns.iter().enumerate() {
            name_to_index.insert(col.name.to_ascii_lowercase(), idx);
            columns.push(PlanColumn {
                name: col.name.clone(),
                data_type: col.data_type.clone(),
                field_id: col.field_id,
                is_nullable: col.is_nullable,
                is_primary_key: col.is_primary_key,
                is_unique: col.is_unique,
                default_value: col.default_value.clone(),
                check_expr: col.check_expr.clone(),
            });
        }

        PlanSchema { columns, name_to_index }
    }
}

impl SchemaView for ExecutorSchema {
    fn field_id_by_name(&self, name: &str) -> Option<llkv_types::FieldId> {
        self.column_by_name(name).map(|c| c.field_id)
    }
}


/// Lightweight row batch used by admin helpers.
#[derive(Debug, Clone)]
pub struct ExecutorRowBatch {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<PlanValue>>, // row-major for ease of use in admin paths
}

/// Multi-column unique constraint descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutorMultiColumnUnique {
    pub index_name: Option<String>,
    pub column_indices: Vec<usize>,
}
