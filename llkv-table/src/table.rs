use croaring::Treemap;
use std::cmp;
use std::mem;
use std::sync::Arc;
use std::sync::RwLock;

use arrow::array::{
    Array, ArrayRef, OffsetSizeTrait, RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Schema, UInt64Type};
use std::collections::HashMap;

use crate::constants::STREAM_BATCH_ROWS;
use llkv_column_map::ColumnStore;
use llkv_column_map::ScanBuilder;
use llkv_column_map::store::scan::ScanOptions;
use llkv_column_map::store::scan::filter::{FilterDispatch, FilterPrimitive, Utf8Filter};
use llkv_column_map::store::{GatherNullPolicy, IndexKind, MultiGatherContext, ROW_ID_COLUMN_NAME};
use llkv_column_map::{
    llkv_for_each_arrow_boolean, llkv_for_each_arrow_numeric, llkv_for_each_arrow_string,
};
use llkv_storage::pager::{MemPager, Pager};
use llkv_types::ids::{LogicalFieldId, TableId};
use simd_r_drive_entry_handle::EntryHandle;

use crate::ROW_ID_FIELD_ID;
use crate::reserved::is_reserved_table_id;
use crate::sys_catalog::{ColMeta, SysCatalog, TableMeta};
use crate::types::{FieldId, RowId};
use llkv_compute::analysis::PredicateFusionCache;
use llkv_compute::program::{OwnedFilter, OwnedOperator, ProgramCompiler};
use llkv_compute::rowid::RowIdFilter as RowIdBitmapFilter;
use llkv_expr::literal::FromLiteral;
use llkv_expr::typed_predicate::{
    Predicate, PredicateValue, build_bool_predicate, build_fixed_width_predicate,
    build_var_width_predicate,
};
use llkv_expr::{Expr, Operator};
use llkv_result::{Error, Result as LlkvResult};
use llkv_scan::execute::execute_scan;
use llkv_scan::row_stream::{
    ColumnProjectionInfo, ProjectionEval, RowIdSource, RowStreamBuilder, ScanRowStream,
};
pub use llkv_scan::{
    RowIdFilter, ScanOrderDirection, ScanOrderSpec, ScanOrderTransform, ScanProjection,
    ScanStorage, ScanStreamOptions,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::ops::Bound;

/// Cached information about which system columns exist in the table schema.
/// This avoids repeated string comparisons in hot paths like append().
#[derive(Debug, Clone, Copy)]
struct MvccColumnCache {
    has_created_by: bool,
    has_deleted_by: bool,
}

/// Handle for data operations on a table.
///
pub struct Table<P = MemPager>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    store: Arc<ColumnStore<P>>,
    table_id: TableId,
    /// Cache of MVCC column presence. Initialized lazily on first schema() call.
    /// None means not yet initialized.
    mvcc_cache: RwLock<Option<MvccColumnCache>>,
}

pub type TableScanStream<'table, P> = ScanRowStream<'table, P, Table<P>>;

impl<P> Table<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Create a new table from column specifications.
    ///
    /// Coordinates metadata persistence, catalog registration, and storage initialization.
    pub fn create_from_columns(
        display_name: &str,
        canonical_name: &str,
        columns: &[llkv_plan::PlanColumnSpec],
        metadata: Arc<crate::metadata::MetadataManager<P>>,
        catalog: Arc<crate::catalog::TableCatalog>,
        store: Arc<ColumnStore<P>>,
    ) -> LlkvResult<crate::catalog::CreateTableResult<P>> {
        let service = crate::catalog::CatalogManager::new(metadata, catalog, store);
        service.create_table_from_columns(display_name, canonical_name, columns)
    }

    /// Create a new table from an Arrow schema (for CREATE TABLE AS SELECT).
    pub fn create_from_schema(
        display_name: &str,
        canonical_name: &str,
        schema: &arrow::datatypes::Schema,
        metadata: Arc<crate::metadata::MetadataManager<P>>,
        catalog: Arc<crate::catalog::TableCatalog>,
        store: Arc<ColumnStore<P>>,
    ) -> LlkvResult<crate::catalog::CreateTableResult<P>> {
        let service = crate::catalog::CatalogManager::new(metadata, catalog, store);
        service.create_table_from_schema(display_name, canonical_name, schema)
    }

    /// Internal constructor: wrap a table ID with column store access.
    ///
    /// **This is for internal crate use only.** User code should create tables via
    /// `CatalogService::create_table_*()`. For tests, use `Table::from_id()`.
    #[doc(hidden)]
    pub fn from_id(table_id: TableId, pager: Arc<P>) -> LlkvResult<Self> {
        if is_reserved_table_id(table_id) {
            return Err(Error::ReservedTableId(table_id));
        }

        tracing::trace!(
            "Table::from_id: Opening table_id={} with pager at {:p}",
            table_id,
            &*pager
        );
        let store = ColumnStore::open(pager)?;
        Ok(Self {
            store: Arc::new(store),
            table_id,
            mvcc_cache: RwLock::new(None),
        })
    }

    /// Internal constructor: wrap a table ID with a shared column store.
    ///
    /// **This is for internal crate use only.** Preferred over `from_id()` when
    /// multiple tables share the same store. For tests, use `Table::from_id_with_store()`.
    #[doc(hidden)]
    pub fn from_id_and_store(table_id: TableId, store: Arc<ColumnStore<P>>) -> LlkvResult<Self> {
        if is_reserved_table_id(table_id) {
            return Err(Error::ReservedTableId(table_id));
        }

        Ok(Self {
            store,
            table_id,
            mvcc_cache: RwLock::new(None),
        })
    }

    /// Register a persisted sort index for the specified user column.
    pub fn register_sort_index(&self, field_id: FieldId) -> LlkvResult<()> {
        let logical_field_id = LogicalFieldId::for_user(self.table_id, field_id);
        self.store
            .register_index(logical_field_id, IndexKind::Sort)?;
        Ok(())
    }

    /// Remove a persisted sort index for the specified user column if it exists.
    pub fn unregister_sort_index(&self, field_id: FieldId) -> LlkvResult<()> {
        let logical_field_id = LogicalFieldId::for_user(self.table_id, field_id);
        match self
            .store
            .unregister_index(logical_field_id, IndexKind::Sort)
        {
            Ok(()) | Err(Error::NotFound) => Ok(()),
            Err(err) => Err(err),
        }
    }

    /// List the persisted index kinds registered for the given user column.
    pub fn list_registered_indexes(&self, field_id: FieldId) -> LlkvResult<Vec<IndexKind>> {
        let logical_field_id = LogicalFieldId::for_user(self.table_id, field_id);
        match self.store.list_persisted_indexes(logical_field_id) {
            Ok(kinds) => Ok(kinds),
            Err(Error::NotFound) => Ok(Vec::new()),
            Err(err) => Err(err),
        }
    }

    /// Get or initialize the MVCC column cache from the provided schema.
    /// This is an optimization to avoid repeated string comparisons in append().
    fn get_mvcc_cache(&self, schema: &Arc<Schema>) -> MvccColumnCache {
        // Fast path: check if cache is already initialized
        {
            let cache_read = self.mvcc_cache.read().unwrap();
            if let Some(cache) = *cache_read {
                return cache;
            }
        }

        // Slow path: initialize cache from schema
        let has_created_by = schema
            .fields()
            .iter()
            .any(|f| f.name() == llkv_column_map::store::CREATED_BY_COLUMN_NAME);
        let has_deleted_by = schema
            .fields()
            .iter()
            .any(|f| f.name() == llkv_column_map::store::DELETED_BY_COLUMN_NAME);

        let cache = MvccColumnCache {
            has_created_by,
            has_deleted_by,
        };

        // Store in cache for future calls
        *self.mvcc_cache.write().unwrap() = Some(cache);

        cache
    }

    /// Append a [`RecordBatch`] to the table.
    ///
    /// The batch must include:
    /// - A `row_id` column (type `UInt64`) with unique row identifiers
    /// - `field_id` metadata for each user column, mapping to this table's field IDs
    ///
    /// # MVCC Columns
    ///
    /// If the batch includes `created_by` or `deleted_by` columns, they are automatically
    /// assigned the correct [`LogicalFieldId`] for this table's MVCC metadata.
    ///
    /// # Field ID Mapping
    ///
    /// Each column's `field_id` metadata is converted to a [`LogicalFieldId`] by combining
    /// it with this table's ID. This ensures columns from different tables don't collide
    /// in the underlying storage.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The batch is missing the `row_id` column
    /// - Any user column is missing `field_id` metadata
    /// - Field IDs are invalid or malformed
    /// - The underlying storage operation fails
    pub fn append(&self, batch: &RecordBatch) -> LlkvResult<()> {
        use arrow::array::UInt64Builder;

        // Check if MVCC columns already exist in the batch using cache
        // This avoids repeated string comparisons on every append
        let cache = self.get_mvcc_cache(&batch.schema());
        let has_created_by = cache.has_created_by;
        let has_deleted_by = cache.has_deleted_by;

        let mut new_fields = Vec::with_capacity(batch.schema().fields().len() + 2);
        let mut new_columns: Vec<Arc<dyn Array>> = Vec::with_capacity(batch.columns().len() + 2);

        for (idx, field) in batch.schema().fields().iter().enumerate() {
            let maybe_field_id = field.metadata().get(crate::constants::FIELD_ID_META_KEY);
            // System columns (row_id, MVCC columns) don't need field_id metadata
            if maybe_field_id.is_none()
                && (field.name() == ROW_ID_COLUMN_NAME
                    || field.name() == llkv_column_map::store::CREATED_BY_COLUMN_NAME
                    || field.name() == llkv_column_map::store::DELETED_BY_COLUMN_NAME)
            {
                if field.name() == ROW_ID_COLUMN_NAME {
                    new_fields.push(field.as_ref().clone());
                    new_columns.push(batch.column(idx).clone());
                } else {
                    let lfid = if field.name() == llkv_column_map::store::CREATED_BY_COLUMN_NAME {
                        LogicalFieldId::for_mvcc_created_by(self.table_id)
                    } else {
                        LogicalFieldId::for_mvcc_deleted_by(self.table_id)
                    };

                    let mut metadata = field.metadata().clone();
                    let lfid_val: u64 = lfid.into();
                    metadata.insert(
                        crate::constants::FIELD_ID_META_KEY.to_string(),
                        lfid_val.to_string(),
                    );

                    let new_field =
                        Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                            .with_metadata(metadata);
                    new_fields.push(new_field);
                    new_columns.push(batch.column(idx).clone());
                }
                continue;
            }

            let raw_field_id = maybe_field_id
                .ok_or_else(|| {
                    llkv_result::Error::Internal(format!(
                        "Field '{}' is missing a valid '{}' in its metadata.",
                        field.name(),
                        crate::constants::FIELD_ID_META_KEY
                    ))
                })?
                .parse::<u64>()
                .map_err(|err| {
                    llkv_result::Error::Internal(format!(
                        "Field '{}' contains an invalid '{}': {}",
                        field.name(),
                        crate::constants::FIELD_ID_META_KEY,
                        err
                    ))
                })?;

            if raw_field_id > FieldId::MAX as u64 {
                return Err(llkv_result::Error::Internal(format!(
                    "Field '{}' expected user FieldId (<= {}) but got logical id '{}'",
                    field.name(),
                    FieldId::MAX,
                    raw_field_id
                )));
            }

            let user_field_id = raw_field_id as FieldId;
            let logical_field_id = LogicalFieldId::for_user(self.table_id, user_field_id);

            // Store the fully-qualified logical field id in the metadata we hand to the
            // column store so descriptors are registered under the correct table id.
            let lfid = logical_field_id;
            let mut new_metadata = field.metadata().clone();
            let lfid_val: u64 = lfid.into();
            new_metadata.insert(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                lfid_val.to_string(),
            );

            let new_field =
                Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                    .with_metadata(new_metadata);
            new_fields.push(new_field);
            new_columns.push(batch.column(idx).clone());

            // Ensure the catalog remembers the human-friendly column name for
            // this field so callers of `Table::schema()` (and other metadata
            // consumers) can recover it later. The CSV ingest path (and other
            // writers) may only supply the `field_id` metadata on the batch,
            // so defensively persist the column name when absent.
            let need_meta = match self
                .catalog()
                .get_cols_meta(self.table_id, &[user_field_id])
            {
                metas if metas.is_empty() => true,
                metas => metas[0].as_ref().and_then(|m| m.name.as_ref()).is_none(),
            };

            if need_meta {
                let meta = ColMeta {
                    col_id: user_field_id,
                    name: Some(field.name().to_string()),
                    flags: 0,
                    default: None,
                };
                self.catalog().put_col_meta(self.table_id, &meta);
            }
        }

        // Inject MVCC columns if they don't exist
        // For non-transactional appends (e.g., CSV ingest), we use TXN_ID_AUTO_COMMIT (1)
        // which is treated as "committed by system" and always visible.
        // Use TXN_ID_NONE (0) for deleted_by to indicate "not deleted".
        const TXN_ID_AUTO_COMMIT: u64 = 1;
        const TXN_ID_NONE: u64 = 0;
        let row_count = batch.num_rows();

        if !has_created_by {
            let mut created_by_builder = UInt64Builder::with_capacity(row_count);
            for _ in 0..row_count {
                created_by_builder.append_value(TXN_ID_AUTO_COMMIT);
            }
            let created_by_lfid = LogicalFieldId::for_mvcc_created_by(self.table_id);
            let mut metadata = HashMap::new();
            let lfid_val: u64 = created_by_lfid.into();
            metadata.insert(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                lfid_val.to_string(),
            );
            new_fields.push(
                Field::new(
                    llkv_column_map::store::CREATED_BY_COLUMN_NAME,
                    DataType::UInt64,
                    false,
                )
                .with_metadata(metadata),
            );
            new_columns.push(Arc::new(created_by_builder.finish()));
        }

        if !has_deleted_by {
            let mut deleted_by_builder = UInt64Builder::with_capacity(row_count);
            for _ in 0..row_count {
                deleted_by_builder.append_value(TXN_ID_NONE);
            }
            let deleted_by_lfid = LogicalFieldId::for_mvcc_deleted_by(self.table_id);
            let mut metadata = HashMap::new();
            let lfid_val: u64 = deleted_by_lfid.into();
            metadata.insert(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                lfid_val.to_string(),
            );
            new_fields.push(
                Field::new(
                    llkv_column_map::store::DELETED_BY_COLUMN_NAME,
                    DataType::UInt64,
                    false,
                )
                .with_metadata(metadata),
            );
            new_columns.push(Arc::new(deleted_by_builder.finish()));
        }

        let new_schema = Arc::new(Schema::new(new_fields));
        let namespaced_batch = RecordBatch::try_new(new_schema, new_columns)?;

        tracing::trace!(
            table_id = self.table_id,
            num_columns = namespaced_batch.num_columns(),
            num_rows = namespaced_batch.num_rows(),
            "Attempting append to table"
        );

        if let Err(err) = self.store.append(&namespaced_batch) {
            let batch_field_ids: Vec<LogicalFieldId> = namespaced_batch
                .schema()
                .fields()
                .iter()
                .filter_map(|f| f.metadata().get(crate::constants::FIELD_ID_META_KEY))
                .filter_map(|s| s.parse::<u64>().ok())
                .map(LogicalFieldId::from)
                .collect();

            // Check which fields are missing from the catalog
            let missing_fields: Vec<LogicalFieldId> = batch_field_ids
                .iter()
                .filter(|&&field_id| !self.store.has_field(field_id))
                .copied()
                .collect();

            tracing::error!(
                table_id = self.table_id,
                error = ?err,
                batch_field_ids = ?batch_field_ids,
                missing_from_catalog = ?missing_fields,
                "Append failed - some fields missing from catalog"
            );
            return Err(err);
        }
        Ok(())
    }

    /// Stream one or more projected columns as a sequence of RecordBatches.
    ///
    /// - Avoids `concat` and large materializations.
    /// - Uses the same filter machinery as the old `scan` to produce
    ///   `row_ids`.
    /// - Splits `row_ids` into fixed-size windows and gathers rows per
    ///   window to form a small `RecordBatch` that is sent to `on_batch`.
    pub fn scan_stream<'a, I, T, F>(
        &self,
        projections: I,
        filter_expr: &Expr<'a, FieldId>,
        options: ScanStreamOptions<P>,
        on_batch: F,
    ) -> LlkvResult<()>
    where
        I: IntoIterator<Item = T>,
        T: Into<ScanProjection>,
        F: FnMut(RecordBatch),
    {
        let stream_projections: Vec<ScanProjection> =
            projections.into_iter().map(|p| p.into()).collect();
        self.scan_stream_with_exprs(&stream_projections, filter_expr, options, on_batch)
    }

    /// Stream projections using fully resolved expression inputs.
    ///
    /// Callers that already parsed expressions into [`ScanProjection`] values can
    /// use this entry point to skip the iterator conversion performed by
    /// [`Self::scan_stream`]. The execution semantics and callbacks are identical.
    pub fn scan_stream_with_exprs<'a, F>(
        &self,
        projections: &[ScanProjection],
        filter_expr: &Expr<'a, FieldId>,
        options: ScanStreamOptions<P>,
        on_batch: F,
    ) -> LlkvResult<()>
    where
        F: FnMut(RecordBatch),
    {
        let mut cb = on_batch;
        execute_scan(
            self,
            self.table_id,
            projections,
            filter_expr,
            options,
            &mut cb,
        )
    }

    pub fn filter_row_ids<'a>(&self, filter_expr: &Expr<'a, FieldId>) -> LlkvResult<Treemap> {
        let source = self.collect_row_ids_for_table(filter_expr)?;
        Ok(match source {
            RowIdSource::Bitmap(b) => b,
            RowIdSource::Vector(v) => Treemap::from_iter(v),
        })
    }

    #[inline]
    pub fn catalog(&self) -> SysCatalog<'_, P> {
        SysCatalog::new(&self.store)
    }

    #[inline]
    pub fn get_table_meta(&self) -> Option<TableMeta> {
        self.catalog().get_table_meta(self.table_id)
    }

    #[inline]
    pub fn get_cols_meta(&self, col_ids: &[FieldId]) -> Vec<Option<ColMeta>> {
        self.catalog().get_cols_meta(self.table_id, col_ids)
    }

    /// Build and return an Arrow `Schema` that describes this table.
    ///
    /// The returned schema includes the `row_id` field first, followed by
    /// user fields. Each user field has its `field_id` stored in the field
    /// metadata (under the "field_id" key) and the name is taken from the
    /// catalog when available or falls back to `col_<id>`.
    pub fn schema(&self) -> LlkvResult<Arc<Schema>> {
        // Collect logical fields for this table and sort by field id.
        let mut logical_fields = self.store.user_field_ids_for_table(self.table_id);
        logical_fields.sort_by_key(|lfid| lfid.field_id());

        let field_ids: Vec<FieldId> = logical_fields.iter().map(|lfid| lfid.field_id()).collect();
        let metas = self.get_cols_meta(&field_ids);

        let mut fields: Vec<Field> = Vec::with_capacity(1 + field_ids.len());
        // Add row_id first
        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

        for (idx, lfid) in logical_fields.into_iter().enumerate() {
            let fid = lfid.field_id();
            let dtype = self.store.data_type(lfid)?;
            let name = metas
                .get(idx)
                .and_then(|m| m.as_ref().and_then(|meta| meta.name.clone()))
                .unwrap_or_else(|| format!("col_{}", fid));

            let mut metadata: HashMap<String, String> = HashMap::new();
            metadata.insert(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                fid.to_string(),
            );

            fields.push(Field::new(&name, dtype.clone(), true).with_metadata(metadata));
        }

        Ok(Arc::new(Schema::new(fields)))
    }

    /// Return the table schema formatted as an Arrow RecordBatch suitable
    /// for pretty printing. The batch has three columns: `name` (Utf8),
    /// `field_id` (UInt32) and `data_type` (Utf8).
    pub fn schema_recordbatch(&self) -> LlkvResult<RecordBatch> {
        let schema = self.schema()?;
        let fields = schema.fields();

        let mut names: Vec<String> = Vec::with_capacity(fields.len());
        let mut fids: Vec<u32> = Vec::with_capacity(fields.len());
        let mut dtypes: Vec<String> = Vec::with_capacity(fields.len());

        for field in fields.iter() {
            names.push(field.name().to_string());
            let fid = field
                .metadata()
                .get(crate::constants::FIELD_ID_META_KEY)
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0u32);
            fids.push(fid);
            dtypes.push(format!("{:?}", field.data_type()));
        }

        // Build Arrow arrays
        let name_array: ArrayRef = Arc::new(StringArray::from(names));
        let fid_array: ArrayRef = Arc::new(UInt32Array::from(fids));
        let dtype_array: ArrayRef = Arc::new(StringArray::from(dtypes));

        let rb_schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new(crate::constants::FIELD_ID_META_KEY, DataType::UInt32, false),
            Field::new("data_type", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(rb_schema, vec![name_array, fid_array, dtype_array])?;
        Ok(batch)
    }

    /// Create a streaming view over the provided row IDs for the specified logical fields.
    pub fn stream_columns<'table>(
        &'table self,
        logical_fields: impl Into<Arc<[LogicalFieldId]>>,
        row_ids: impl Into<RowIdSource>,
        policy: GatherNullPolicy,
    ) -> LlkvResult<TableScanStream<'table, P>> {
        let logical_fields: Arc<[LogicalFieldId]> = logical_fields.into();
        let mut projection_evals = Vec::with_capacity(logical_fields.len());
        let mut schema_fields = Vec::with_capacity(logical_fields.len());
        let mut unique_index: FxHashMap<LogicalFieldId, usize> = FxHashMap::default();
        let mut unique_lfids: Vec<LogicalFieldId> = Vec::new();

        for &lfid in logical_fields.iter() {
            let dtype = self.store.data_type(lfid)?;
            if let std::collections::hash_map::Entry::Vacant(entry) = unique_index.entry(lfid) {
                entry.insert(unique_lfids.len());
                unique_lfids.push(lfid);
            }

            let field_name = self
                .catalog()
                .get_cols_meta(self.table_id, &[lfid.field_id()])
                .into_iter()
                .flatten()
                .next()
                .and_then(|meta| meta.name)
                .unwrap_or_else(|| format!("col_{}", lfid.field_id()));

            projection_evals.push(ProjectionEval::Column(ColumnProjectionInfo {
                logical_field_id: lfid,
                data_type: dtype.clone(),
                output_name: field_name.clone(),
            }));
            schema_fields.push(Field::new(field_name, dtype, true));
        }

        let schema = Arc::new(Schema::new(schema_fields));
        let passthrough_fields = vec![None; projection_evals.len()];
        let row_source = row_ids.into();

        RowStreamBuilder::new(
            self,
            self.table_id,
            Arc::clone(&schema),
            Arc::new(unique_lfids),
            Arc::new(projection_evals),
            Arc::new(passthrough_fields),
            Arc::new(unique_index),
            Arc::new(FxHashSet::default()),
            false,
            policy,
            row_source,
            STREAM_BATCH_ROWS,
            true,
        )
        .build()
    }

    pub fn store(&self) -> &ColumnStore<P> {
        &self.store
    }

    #[inline]
    pub fn table_id(&self) -> TableId {
        self.table_id
    }

    /// Return the total number of rows for a given user column id in this table.
    ///
    /// This delegates to the ColumnStore descriptor for the logical field that
    /// corresponds to (table_id, col_id) and returns the persisted total_row_count.
    pub fn total_rows_for_col(&self, col_id: FieldId) -> llkv_result::Result<u64> {
        let lfid = LogicalFieldId::for_user(self.table_id, col_id);
        self.store.total_rows_for_field(lfid)
    }

    /// Return the total number of rows for this table.
    ///
    /// Prefer reading the dedicated row-id shadow column if present; otherwise
    /// fall back to inspecting any persisted user column descriptor.
    pub fn total_rows(&self) -> llkv_result::Result<u64> {
        use llkv_column_map::store::rowid_fid;
        let rid_lfid = rowid_fid(LogicalFieldId::for_user(self.table_id, 0));
        // Try the row-id shadow column first
        match self.store.total_rows_for_field(rid_lfid) {
            Ok(n) => Ok(n),
            Err(_) => {
                // Fall back to scanning the catalog for any user-data column
                self.store.total_rows_for_table(self.table_id)
            }
        }
    }
}

macro_rules! impl_row_id_ignore_chunk {
    (
        $_base:ident,
        $chunk:ident,
        $_chunk_with_rids:ident,
        $_run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $chunk(&mut self, _values: &$array_ty) {}
    };
}

macro_rules! impl_row_id_ignore_sorted_run {
    (
        $_base:ident,
        $_chunk:ident,
        $_chunk_with_rids:ident,
        $run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $run(&mut self, _values: &$array_ty, _start: usize, _len: usize) {}
    };
}

macro_rules! impl_row_id_collect_chunk_with_rids {
    (
        $_base:ident,
        $_chunk:ident,
        $chunk_with_rids:ident,
        $_run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $chunk_with_rids(&mut self, _: &$array_ty, row_ids: &UInt64Array) {
            self.extend_from_array(row_ids);
        }
    };
}

macro_rules! impl_row_id_collect_sorted_run_with_rids {
    (
        $_base:ident,
        $_chunk:ident,
        $_chunk_with_rids:ident,
        $_run:ident,
        $run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $run_with_rids(
            &mut self,
            _: &$array_ty,
            row_ids: &UInt64Array,
            start: usize,
            len: usize,
        ) {
            self.extend_from_slice(row_ids, start, len);
        }
    };
}

macro_rules! impl_row_id_stream_chunk_with_rids {
    (
        $_base:ident,
        $_chunk:ident,
        $chunk_with_rids:ident,
        $_run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $chunk_with_rids(&mut self, _: &$array_ty, row_ids: &UInt64Array) {
            self.extend_from_array(row_ids);
        }
    };
}

macro_rules! impl_row_id_stream_sorted_run_with_rids {
    (
        $_base:ident,
        $_chunk:ident,
        $_chunk_with_rids:ident,
        $_run:ident,
        $run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $run_with_rids(
            &mut self,
            _: &$array_ty,
            row_ids: &UInt64Array,
            start: usize,
            len: usize,
        ) {
            self.extend_sorted_run(row_ids, start, len);
        }
    };
}

#[derive(Default)]
struct RowIdScanCollector {
    row_ids: Treemap,
}

impl RowIdScanCollector {
    fn extend_from_array(&mut self, row_ids: &UInt64Array) {
        for idx in 0..row_ids.len() {
            self.row_ids.add(row_ids.value(idx));
        }
    }

    fn extend_from_slice(&mut self, row_ids: &UInt64Array, start: usize, len: usize) {
        if len == 0 {
            return;
        }
        let end = (start + len).min(row_ids.len());
        for idx in start..end {
            self.row_ids.add(row_ids.value(idx));
        }
    }

    fn into_inner(self) -> Treemap {
        self.row_ids
    }
}

impl llkv_column_map::scan::PrimitiveVisitor for RowIdScanCollector {
    llkv_for_each_arrow_numeric!(impl_row_id_ignore_chunk);
    llkv_for_each_arrow_boolean!(impl_row_id_ignore_chunk);
    llkv_for_each_arrow_string!(impl_row_id_ignore_chunk);
}

impl llkv_column_map::scan::PrimitiveSortedVisitor for RowIdScanCollector {
    llkv_for_each_arrow_numeric!(impl_row_id_ignore_sorted_run);
    llkv_for_each_arrow_boolean!(impl_row_id_ignore_sorted_run);
    llkv_for_each_arrow_string!(impl_row_id_ignore_sorted_run);
}

impl llkv_column_map::scan::PrimitiveWithRowIdsVisitor for RowIdScanCollector {
    llkv_for_each_arrow_numeric!(impl_row_id_collect_chunk_with_rids);
    llkv_for_each_arrow_boolean!(impl_row_id_collect_chunk_with_rids);
    llkv_for_each_arrow_string!(impl_row_id_collect_chunk_with_rids);
}

impl llkv_column_map::scan::PrimitiveSortedWithRowIdsVisitor for RowIdScanCollector {
    llkv_for_each_arrow_numeric!(impl_row_id_collect_sorted_run_with_rids);
    llkv_for_each_arrow_boolean!(impl_row_id_collect_sorted_run_with_rids);
    llkv_for_each_arrow_string!(impl_row_id_collect_sorted_run_with_rids);

    fn null_run(&mut self, row_ids: &UInt64Array, start: usize, len: usize) {
        self.extend_from_slice(row_ids, start, len);
    }
}

struct RowIdChunkEmitter<'a> {
    chunk_size: usize,
    buffer: Vec<RowId>,
    reverse_sorted_runs: bool,
    on_chunk: &'a mut dyn FnMut(Vec<RowId>) -> LlkvResult<()>,
    error: Option<Error>,
}

impl<'a> RowIdChunkEmitter<'a> {
    fn new(
        chunk_size: usize,
        reverse_sorted_runs: bool,
        on_chunk: &'a mut dyn FnMut(Vec<RowId>) -> LlkvResult<()>,
    ) -> Self {
        let chunk_size = cmp::max(1, chunk_size);
        Self {
            chunk_size,
            buffer: Vec::with_capacity(chunk_size),
            reverse_sorted_runs,
            on_chunk,
            error: None,
        }
    }

    fn extend_from_array(&mut self, row_ids: &UInt64Array) {
        if self.error.is_some() {
            return;
        }
        for idx in 0..row_ids.len() {
            self.push(row_ids.value(idx));
        }
    }

    fn extend_from_slice(&mut self, row_ids: &UInt64Array, start: usize, len: usize) {
        if self.error.is_some() || len == 0 {
            return;
        }
        let end = (start + len).min(row_ids.len());
        for idx in start..end {
            self.push(row_ids.value(idx));
        }
    }

    fn extend_sorted_run(&mut self, row_ids: &UInt64Array, start: usize, len: usize) {
        if self.reverse_sorted_runs {
            if self.error.is_some() || len == 0 {
                return;
            }
            let mut idx = (start + len).min(row_ids.len());
            while idx > start {
                idx -= 1;
                self.push(row_ids.value(idx));
            }
        } else {
            self.extend_from_slice(row_ids, start, len);
        }
    }

    fn push(&mut self, value: RowId) {
        if self.error.is_some() {
            return;
        }
        self.buffer.push(value);
        if self.buffer.len() >= self.chunk_size {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.error.is_some() || self.buffer.is_empty() {
            return;
        }
        let chunk = mem::take(&mut self.buffer);
        if let Err(err) = (self.on_chunk)(chunk) {
            self.error = Some(err);
        }
    }

    fn finish(mut self) -> LlkvResult<()> {
        self.flush();
        match self.error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

impl<'a> llkv_column_map::scan::PrimitiveVisitor for RowIdChunkEmitter<'a> {
    llkv_for_each_arrow_numeric!(impl_row_id_ignore_chunk);
    llkv_for_each_arrow_boolean!(impl_row_id_ignore_chunk);
}

impl<'a> llkv_column_map::scan::PrimitiveSortedVisitor for RowIdChunkEmitter<'a> {
    llkv_for_each_arrow_numeric!(impl_row_id_ignore_sorted_run);
    llkv_for_each_arrow_boolean!(impl_row_id_ignore_sorted_run);
}

impl<'a> llkv_column_map::scan::PrimitiveWithRowIdsVisitor for RowIdChunkEmitter<'a> {
    llkv_for_each_arrow_numeric!(impl_row_id_stream_chunk_with_rids);
    llkv_for_each_arrow_boolean!(impl_row_id_stream_chunk_with_rids);
}

impl<'a> llkv_column_map::scan::PrimitiveSortedWithRowIdsVisitor for RowIdChunkEmitter<'a> {
    llkv_for_each_arrow_numeric!(impl_row_id_stream_sorted_run_with_rids);
    llkv_for_each_arrow_boolean!(impl_row_id_stream_sorted_run_with_rids);

    fn null_run(&mut self, row_ids: &UInt64Array, start: usize, len: usize) {
        self.extend_sorted_run(row_ids, start, len);
    }
}

impl<P> ScanStorage<P> for Table<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn table_id(&self) -> TableId {
        self.table_id
    }

    fn field_data_type(&self, fid: LogicalFieldId) -> LlkvResult<DataType> {
        self.store.data_type(fid)
    }

    fn total_rows(&self) -> LlkvResult<u64> {
        Table::total_rows(self)
    }

    fn prepare_gather_context(
        &self,
        logical_fields: &[LogicalFieldId],
    ) -> LlkvResult<MultiGatherContext> {
        self.store.prepare_gather_context(logical_fields)
    }

    fn gather_row_window_with_context(
        &self,
        logical_fields: &[LogicalFieldId],
        row_ids: &[u64],
        null_policy: GatherNullPolicy,
        ctx: Option<&mut MultiGatherContext>,
    ) -> LlkvResult<RecordBatch> {
        self.store
            .gather_row_window_with_context(logical_fields, row_ids, null_policy, ctx)
    }

    fn filter_row_ids<'expr>(&self, filter_expr: &Expr<'expr, FieldId>) -> LlkvResult<Treemap> {
        Table::filter_row_ids(self, filter_expr)
    }

    fn filter_leaf(&self, filter: &OwnedFilter) -> LlkvResult<Treemap> {
        let source = self.collect_row_ids_for_filter(filter)?;
        Ok(match source {
            RowIdSource::Bitmap(b) => b,
            RowIdSource::Vector(v) => Treemap::from_iter(v),
        })
    }

    fn filter_fused(
        &self,
        field_id: FieldId,
        filters: &[OwnedFilter],
        cache: &PredicateFusionCache,
    ) -> LlkvResult<RowIdSource> {
        self.collect_fused_predicates(field_id, filters, cache)
    }

    fn all_row_ids(&self) -> LlkvResult<Treemap> {
        self.compute_table_row_ids()
    }

    fn sorted_row_ids_full_table(&self, order_spec: ScanOrderSpec) -> LlkvResult<Option<Vec<u64>>> {
        self.collect_full_table_sorted_row_ids(order_spec)
    }

    fn stream_row_ids(
        &self,
        chunk_size: usize,
        on_chunk: &mut dyn FnMut(Vec<u64>) -> LlkvResult<()>,
    ) -> LlkvResult<()> {
        self.stream_table_row_ids(chunk_size, on_chunk)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<P> Table<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn collect_row_ids_for_table<'expr>(
        &self,
        filter_expr: &Expr<'expr, FieldId>,
    ) -> LlkvResult<RowIdSource> {
        let fusion_cache = PredicateFusionCache::from_expr(filter_expr);
        let mut all_rows_cache: FxHashMap<FieldId, Treemap> = FxHashMap::default();
        let filter_arc = Arc::new(filter_expr.clone());
        let programs = ProgramCompiler::new(filter_arc).compile()?;
        llkv_scan::predicate::collect_row_ids_for_program(
            self,
            &programs,
            &fusion_cache,
            &mut all_rows_cache,
        )
    }

    fn collect_row_ids_for_filter(&self, filter: &OwnedFilter) -> LlkvResult<RowIdSource> {
        if filter.field_id == ROW_ID_FIELD_ID {
            let op = filter.op.to_operator();
            let row_ids = self.collect_row_ids_for_rowid_filter(&op)?;
            return Ok(RowIdSource::Bitmap(row_ids));
        }

        let filter_lfid = LogicalFieldId::for_user(self.table_id, filter.field_id);
        let dtype = self.store.data_type(filter_lfid)?;

        match &filter.op {
            OwnedOperator::IsNotNull => {
                let mut cache = FxHashMap::default();
                let non_null = self.collect_all_row_ids_for_field(filter.field_id, &mut cache)?;
                return Ok(RowIdSource::Bitmap(non_null));
            }
            OwnedOperator::IsNull => {
                let all_row_ids = self.compute_table_row_ids()?;
                if all_row_ids.is_empty() {
                    return Ok(RowIdSource::Bitmap(Treemap::new()));
                }
                let mut cache = FxHashMap::default();
                let non_null = self.collect_all_row_ids_for_field(filter.field_id, &mut cache)?;
                let null_ids = all_row_ids - non_null;
                return Ok(RowIdSource::Bitmap(null_ids));
            }
            _ => {}
        }

        if let OwnedOperator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        } = &filter.op
        {
            let all_rows = self.compute_table_row_ids()?;
            return Ok(RowIdSource::Bitmap(all_rows));
        }

        let op = filter.op.to_operator();
        let row_ids = match &dtype {
            DataType::Utf8 => self.collect_matching_row_ids_string::<i32>(filter_lfid, &op),
            DataType::LargeUtf8 => self.collect_matching_row_ids_string::<i64>(filter_lfid, &op),
            DataType::Boolean => self.collect_matching_row_ids_bool(filter_lfid, &op),
            other => llkv_column_map::with_integer_arrow_type!(
                other.clone(),
                |ArrowTy| self.collect_matching_row_ids::<ArrowTy>(filter_lfid, &op),
                Err(Error::Internal(format!(
                    "Filtering on type {:?} is not supported",
                    other
                )))
            ),
        }?;

        Ok(RowIdSource::Bitmap(row_ids))
    }

    fn collect_fused_predicates(
        &self,
        _field_id: FieldId,
        filters: &[OwnedFilter],
        _cache: &PredicateFusionCache,
    ) -> LlkvResult<RowIdSource> {
        let mut result: Option<Treemap> = None;

        for filter in filters {
            let rows = match self.collect_row_ids_for_filter(filter)? {
                RowIdSource::Bitmap(b) => b,
                RowIdSource::Vector(v) => Treemap::from_iter(v),
            };

            result = Some(match result {
                Some(acc) => acc & rows,
                None => rows,
            });

            if let Some(ref r) = result {
                if r.is_empty() {
                    return Ok(RowIdSource::Bitmap(Treemap::new()));
                }
            }
        }

        Ok(RowIdSource::Bitmap(result.unwrap_or_default()))
    }

    fn collect_all_row_ids_for_field(
        &self,
        field_id: FieldId,
        cache: &mut FxHashMap<FieldId, Treemap>,
    ) -> LlkvResult<Treemap> {
        if let Some(rows) = cache.get(&field_id) {
            return Ok(rows.clone());
        }

        let lfid = LogicalFieldId::for_user(self.table_id, field_id);
        let mut collector = RowIdScanCollector::default();
        ScanBuilder::new(self.store(), lfid)
            .options(ScanOptions {
                with_row_ids: true,
                ..Default::default()
            })
            .run(&mut collector)?;

        let rows = collector.into_inner();
        cache.insert(field_id, rows.clone());
        Ok(rows)
    }

    fn collect_matching_row_ids<T>(
        &self,
        lfid: LogicalFieldId,
        op: &Operator,
    ) -> LlkvResult<Treemap>
    where
        T: ArrowPrimitiveType
            + FilterPrimitive<Native = <T as ArrowPrimitiveType>::Native>
            + FilterDispatch<Value = <T as ArrowPrimitiveType>::Native>,
        <T as ArrowPrimitiveType>::Native: PartialOrd + Copy + FromLiteral + PredicateValue,
    {
        let predicate = build_fixed_width_predicate::<T>(op).map_err(Error::predicate_build)?;
        let row_ids =
            <T as FilterPrimitive>::run_nullable_filter(self.store(), lfid, |v| match v {
                Some(val) => predicate.matches(PredicateValue::borrowed(&val)),
                None => false,
            })
            .map_err(Error::from)?;
        Ok(Treemap::from_iter(row_ids))
    }

    fn collect_matching_row_ids_string<O>(
        &self,
        lfid: LogicalFieldId,
        op: &Operator,
    ) -> LlkvResult<Treemap>
    where
        O: OffsetSizeTrait + llkv_column_map::store::scan::StringContainsKernel,
    {
        let predicate = build_var_width_predicate(op).map_err(Error::predicate_build)?;
        let row_ids =
            Utf8Filter::<O>::run_filter(self.store(), lfid, &predicate).map_err(Error::from)?;
        Ok(Treemap::from_iter(row_ids))
    }

    fn collect_matching_row_ids_bool(
        &self,
        lfid: LogicalFieldId,
        op: &Operator,
    ) -> LlkvResult<Treemap> {
        let predicate = build_bool_predicate(op).map_err(Error::predicate_build)?;

        let row_ids = arrow::datatypes::BooleanType::run_nullable_filter(
            self.store(),
            lfid,
            |val: Option<bool>| match val {
                Some(v) => predicate.matches(&v),
                None => false,
            },
        )
        .map_err(Error::from)?;
        Ok(Treemap::from_iter(row_ids))
    }

    fn collect_row_ids_for_rowid_filter(&self, op: &Operator<'_>) -> LlkvResult<Treemap> {
        let all_row_ids = self.compute_table_row_ids()?;
        if all_row_ids.is_empty() {
            return Ok(Treemap::new());
        }
        RowIdBitmapFilter::filter_by_operator(&all_row_ids, op)
    }

    fn collect_full_table_sorted_row_ids(
        &self,
        order_spec: ScanOrderSpec,
    ) -> LlkvResult<Option<Vec<u64>>> {
        use llkv_column_map::store::rowid_fid;

        if !matches!(
            order_spec.transform,
            ScanOrderTransform::IdentityInt64
                | ScanOrderTransform::IdentityInt32
                | ScanOrderTransform::IdentityUtf8
        ) {
            return Ok(None);
        }

        let lfid = LogicalFieldId::for_user(self.table_id, order_spec.field_id);
        let dtype = match self.store.data_type(lfid) {
            Ok(dt) => dt,
            Err(Error::NotFound) => return Ok(None),
            Err(err) => return Err(err),
        };

        if !Self::order_transform_matches_dtype(order_spec.transform, &dtype) {
            return Ok(None);
        }

        let mut ordered: Vec<u64> = Vec::new();

        if let Ok(total_rows) = self.total_rows() {
            if let Ok(cap) = usize::try_from(total_rows) {
                ordered.reserve(cap);
            }
        }

        let mut on_chunk = |chunk: Vec<RowId>| -> LlkvResult<()> {
            ordered.extend(chunk);
            Ok(())
        };
        let reverse_sorted_runs = matches!(order_spec.direction, ScanOrderDirection::Descending);
        let mut emitter = RowIdChunkEmitter::new(
            STREAM_BATCH_ROWS,
            reverse_sorted_runs,
            &mut on_chunk,
        );
        let options = ScanOptions {
            sorted: true,
            reverse: matches!(order_spec.direction, ScanOrderDirection::Descending),
            with_row_ids: true,
            include_nulls: true,
            nulls_first: order_spec.nulls_first,
            anchor_row_id_field: Some(rowid_fid(lfid)),
            ..Default::default()
        };

        match ScanBuilder::new(self.store(), lfid)
            .options(options)
            .run(&mut emitter)
        {
            Ok(()) => emitter.finish()?,
            Err(Error::NotFound) => return Ok(None),
            Err(err) => return Err(err),
        }

        Ok(Some(ordered))
    }

    fn order_transform_matches_dtype(transform: ScanOrderTransform, dtype: &DataType) -> bool {
        match transform {
            ScanOrderTransform::IdentityInt64 => matches!(dtype, DataType::Int64),
            ScanOrderTransform::IdentityInt32 => matches!(dtype, DataType::Int32),
            ScanOrderTransform::IdentityUtf8 => matches!(dtype, DataType::Utf8),
            ScanOrderTransform::CastUtf8ToInteger => false,
        }
    }

    fn compute_table_row_ids(&self) -> LlkvResult<Treemap> {
        use llkv_column_map::store::rowid_fid;

        if let Some(rows) = self.collect_row_ids_from_mvcc()? {
            return Ok(rows);
        }

        let fields = self.store.user_field_ids_for_table(self.table_id);
        if fields.is_empty() {
            return Ok(Treemap::new());
        }

        let expected = self
            .store
            .total_rows_for_table(self.table_id)
            .unwrap_or_default();

        if expected > 0
            && let Some(&first_field) = fields.first()
        {
            let rid_shadow = rowid_fid(first_field);
            let mut collector = RowIdScanCollector::default();

            match ScanBuilder::new(self.store(), rid_shadow)
                .options(ScanOptions {
                    with_row_ids: true,
                    ..Default::default()
                })
                .run(&mut collector)
            {
                Ok(_) => {
                    let row_ids = collector.into_inner();
                    if row_ids.cardinality() == expected {
                        return Ok(row_ids);
                    }
                }
                Err(llkv_result::Error::NotFound) => {}
                Err(_) => {}
            }
        }

        let mut collected = Treemap::new();

        for lfid in fields.clone() {
            let mut collector = RowIdScanCollector::default();
            ScanBuilder::new(self.store(), lfid)
                .options(ScanOptions {
                    with_row_ids: true,
                    ..Default::default()
                })
                .run(&mut collector)?;
            let rows = collector.into_inner();
            collected.or_inplace(&rows);

            if expected > 0 && collected.cardinality() >= expected {
                break;
            }
        }

        Ok(collected)
    }

    fn collect_row_ids_from_mvcc(&self) -> LlkvResult<Option<Treemap>> {
        let Some(rows) = self.fetch_mvcc_row_ids()? else {
            return Ok(None);
        };
        Ok(Some(Treemap::from_iter(rows)))
    }

    fn stream_table_row_ids(
        &self,
        chunk_size: usize,
        on_chunk: &mut dyn FnMut(Vec<RowId>) -> LlkvResult<()>,
    ) -> LlkvResult<()> {
        use llkv_column_map::store::rowid_fid;

        if self.try_stream_row_ids_from_mvcc(chunk_size, on_chunk)? {
            return Ok(());
        }

        let fields = self.store.user_field_ids_for_table(self.table_id);
        if fields.is_empty() {
            return Ok(());
        }

        let Some(&first_field) = fields.first() else {
            return Ok(());
        };

        let rid_shadow = rowid_fid(first_field);
        let mut emitter = RowIdChunkEmitter::new(chunk_size, false, on_chunk);
        let scan_result = ScanBuilder::new(self.store(), rid_shadow)
            .options(ScanOptions {
                with_row_ids: true,
                ..Default::default()
            })
            .run(&mut emitter);

        match scan_result {
            Ok(()) => emitter.finish(),
            Err(Error::NotFound) => {
                let _ = emitter.finish();
                let all_rows = self.compute_table_row_ids()?;
                if all_rows.is_empty() {
                    return Ok(());
                }

                let chunk_cap = cmp::max(1, chunk_size);
                let mut chunk = Vec::with_capacity(chunk_cap);
                for row_id in all_rows.iter() {
                    chunk.push(row_id);
                    if chunk.len() >= chunk_cap {
                        (on_chunk)(mem::take(&mut chunk))?;
                    }
                }
                if !chunk.is_empty() {
                    (on_chunk)(mem::take(&mut chunk))?;
                }
                Ok(())
            }
            Err(err) => {
                let _ = emitter.finish();
                Err(err)
            }
        }
    }

    fn try_stream_row_ids_from_mvcc(
        &self,
        chunk_size: usize,
        on_chunk: &mut dyn FnMut(Vec<RowId>) -> LlkvResult<()>,
    ) -> LlkvResult<bool> {
        let Some(rows) = self.fetch_mvcc_row_ids()? else {
            return Ok(false);
        };
        if rows.is_empty() {
            return Ok(true);
        }

        let chunk_cap = chunk_size.max(1);
        let mut chunk = Vec::with_capacity(chunk_cap);
        for row_id in rows {
            chunk.push(row_id);
            if chunk.len() >= chunk_cap {
                (on_chunk)(mem::take(&mut chunk))?;
            }
        }
        if !chunk.is_empty() {
            (on_chunk)(chunk)?;
        }
        Ok(true)
    }

    fn fetch_mvcc_row_ids(&self) -> LlkvResult<Option<Vec<RowId>>> {
        let created_lfid = LogicalFieldId::for_mvcc_created_by(self.table_id);
        match self
            .store
            .filter_row_ids::<UInt64Type>(created_lfid, &Predicate::All)
        {
            Ok(rows) => Ok(Some(rows)),
            Err(Error::NotFound) => Ok(None),
            Err(err) => Err(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reserved::CATALOG_TABLE_ID;
    use crate::types::RowId;
    use arrow::array::Array;
    use arrow::array::ArrayRef;
    use arrow::array::{
        BinaryArray, Float32Array, Float64Array, Int16Array, Int32Array, StringArray, UInt8Array,
        UInt32Array, UInt64Array,
    };
    use arrow::compute::{cast, max, min, sum, unary};
    use arrow::datatypes::DataType;
    use llkv_column_map::ColumnStore;
    use llkv_column_map::store::{GatherNullPolicy, Projection};
    use llkv_expr::{BinaryOp, CompareOp, Filter, Operator, ScalarExpr};
    use std::collections::HashMap;
    use std::ops::Bound;

    fn setup_test_table() -> Table {
        let pager = Arc::new(MemPager::default());
        setup_test_table_with_pager(&pager)
    }

    fn setup_test_table_with_pager(pager: &Arc<MemPager>) -> Table {
        let table = Table::from_id(1, Arc::clone(pager)).unwrap();
        const COL_A_U64: FieldId = 10;
        const COL_B_BIN: FieldId = 11;
        const COL_C_I32: FieldId = 12;
        const COL_D_F64: FieldId = 13;
        const COL_E_F32: FieldId = 14;

        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("a_u64", DataType::UInt64, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_A_U64.to_string(),
            )])),
            Field::new("b_bin", DataType::Binary, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_B_BIN.to_string(),
            )])),
            Field::new("c_i32", DataType::Int32, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_C_I32.to_string(),
            )])),
            Field::new("d_f64", DataType::Float64, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_D_F64.to_string(),
            )])),
            Field::new("e_f32", DataType::Float32, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_E_F32.to_string(),
            )])),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 3, 4])),
                Arc::new(UInt64Array::from(vec![100, 200, 300, 200])),
                Arc::new(BinaryArray::from(vec![
                    b"foo" as &[u8],
                    b"bar",
                    b"baz",
                    b"qux",
                ])),
                Arc::new(Int32Array::from(vec![10, 20, 30, 20])),
                Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5, 2.5])),
                Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0, 2.0])),
            ],
        )
        .unwrap();

        table.append(&batch).unwrap();
        table
    }

    fn gather_single(
        store: &ColumnStore<MemPager>,
        field_id: LogicalFieldId,
        row_ids: &[u64],
    ) -> ArrayRef {
        store
            .gather_rows(&[field_id], row_ids, GatherNullPolicy::ErrorOnMissing)
            .unwrap()
            .column(0)
            .clone()
    }

    fn pred_expr<'a>(filter: Filter<'a, FieldId>) -> Expr<'a, FieldId> {
        Expr::Pred(filter)
    }

    fn proj(table: &Table, field_id: FieldId) -> Projection {
        Projection::from(LogicalFieldId::for_user(table.table_id, field_id))
    }

    fn proj_alias<S: Into<String>>(table: &Table, field_id: FieldId, alias: S) -> Projection {
        Projection::with_alias(LogicalFieldId::for_user(table.table_id, field_id), alias)
    }

    #[test]
    fn table_new_rejects_reserved_table_id() {
        let result = Table::from_id(CATALOG_TABLE_ID, Arc::new(MemPager::default()));
        assert!(matches!(
            result,
            Err(Error::ReservedTableId(id)) if id == CATALOG_TABLE_ID
        ));
    }

    #[test]
    fn test_append_rejects_logical_field_id_in_metadata() {
        // Create a table and build a schema where the column's metadata
        // contains a fully-qualified LogicalFieldId (u64). Append should
        // reject this and require a plain user FieldId instead.
        let table = Table::from_id(7, Arc::new(MemPager::default())).unwrap();

        const USER_FID: FieldId = 42;
        // Build a logical id (namespaced) and put its numeric value into metadata
        let logical: LogicalFieldId = LogicalFieldId::for_user(table.table_id(), USER_FID);
        let logical_val: u64 = logical.into();

        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("bad", DataType::UInt64, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                logical_val.to_string(),
            )])),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![1u64, 2u64])),
                Arc::new(UInt64Array::from(vec![10u64, 20u64])),
            ],
        )
        .unwrap();

        let res = table.append(&batch);
        assert!(matches!(res, Err(Error::Internal(_))));
    }

    #[test]
    fn test_scan_with_u64_filter() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let expr = pred_expr(Filter {
            field_id: COL_A_U64,
            op: Operator::Equals(200.into()),
        });

        let mut vals: Vec<Option<i32>> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_C_I32)],
                &expr,
                ScanStreamOptions::default(),
                |b| {
                    let a = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                    vals.extend(
                        (0..a.len()).map(|i| if a.is_null(i) { None } else { Some(a.value(i)) }),
                    );
                },
            )
            .unwrap();
        assert_eq!(vals, vec![Some(20), Some(20)]);
    }

    #[test]
    fn test_scan_with_string_filter() {
        let pager = Arc::new(MemPager::default());
        let table = Table::from_id(500, Arc::clone(&pager)).unwrap();

        const COL_STR: FieldId = 42;
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("name", DataType::Utf8, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_STR.to_string(),
            )])),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 3, 4])),
                Arc::new(StringArray::from(vec!["alice", "bob", "albert", "carol"])),
            ],
        )
        .unwrap();
        table.append(&batch).unwrap();

        let expr = pred_expr(Filter {
            field_id: COL_STR,
            op: Operator::starts_with("al".to_string(), true),
        });

        let mut collected: Vec<Option<String>> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_STR)],
                &expr,
                ScanStreamOptions::default(),
                |b| {
                    let arr = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                    collected.extend(arr.iter().map(|v| v.map(|s| s.to_string())));
                },
            )
            .unwrap();

        assert_eq!(
            collected,
            vec![Some("alice".to_string()), Some("albert".to_string())]
        );
    }

    #[test]
    fn test_table_reopen_with_shared_pager() {
        const TABLE_ALPHA: TableId = 42;
        const TABLE_BETA: TableId = 43;
        const TABLE_GAMMA: TableId = 44;
        const COL_ALPHA_U64: FieldId = 100;
        const COL_ALPHA_I32: FieldId = 101;
        const COL_ALPHA_U32: FieldId = 102;
        const COL_ALPHA_I16: FieldId = 103;
        const COL_BETA_U64: FieldId = 200;
        const COL_BETA_U8: FieldId = 201;
        const COL_GAMMA_I16: FieldId = 300;

        let pager = Arc::new(MemPager::default());

        let alpha_rows: Vec<RowId> = vec![1, 2, 3, 4];
        let alpha_vals_u64: Vec<u64> = vec![10, 20, 30, 40];
        let alpha_vals_i32: Vec<i32> = vec![-5, 15, 25, 35];
        let alpha_vals_u32: Vec<u32> = vec![7, 11, 13, 17];
        let alpha_vals_i16: Vec<i16> = vec![-2, 4, -6, 8];

        let beta_rows: Vec<RowId> = vec![101, 102, 103];
        let beta_vals_u64: Vec<u64> = vec![900, 901, 902];
        let beta_vals_u8: Vec<u8> = vec![1, 2, 3];

        let gamma_rows: Vec<RowId> = vec![501, 502];
        let gamma_vals_i16: Vec<i16> = vec![123, -321];

        // First session: create tables and write data.
        {
            let table = Table::from_id(TABLE_ALPHA, Arc::clone(&pager)).unwrap();
            let schema = Arc::new(Schema::new(vec![
                Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
                Field::new("alpha_u64", DataType::UInt64, false).with_metadata(HashMap::from([(
                    crate::constants::FIELD_ID_META_KEY.to_string(),
                    COL_ALPHA_U64.to_string(),
                )])),
                Field::new("alpha_i32", DataType::Int32, false).with_metadata(HashMap::from([(
                    crate::constants::FIELD_ID_META_KEY.to_string(),
                    COL_ALPHA_I32.to_string(),
                )])),
                Field::new("alpha_u32", DataType::UInt32, false).with_metadata(HashMap::from([(
                    crate::constants::FIELD_ID_META_KEY.to_string(),
                    COL_ALPHA_U32.to_string(),
                )])),
                Field::new("alpha_i16", DataType::Int16, false).with_metadata(HashMap::from([(
                    crate::constants::FIELD_ID_META_KEY.to_string(),
                    COL_ALPHA_I16.to_string(),
                )])),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt64Array::from(alpha_rows.clone())),
                    Arc::new(UInt64Array::from(alpha_vals_u64.clone())),
                    Arc::new(Int32Array::from(alpha_vals_i32.clone())),
                    Arc::new(UInt32Array::from(alpha_vals_u32.clone())),
                    Arc::new(Int16Array::from(alpha_vals_i16.clone())),
                ],
            )
            .unwrap();
            table.append(&batch).unwrap();
        }

        {
            let table = Table::from_id(TABLE_BETA, Arc::clone(&pager)).unwrap();
            let schema = Arc::new(Schema::new(vec![
                Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
                Field::new("beta_u64", DataType::UInt64, false).with_metadata(HashMap::from([(
                    crate::constants::FIELD_ID_META_KEY.to_string(),
                    COL_BETA_U64.to_string(),
                )])),
                Field::new("beta_u8", DataType::UInt8, false).with_metadata(HashMap::from([(
                    crate::constants::FIELD_ID_META_KEY.to_string(),
                    COL_BETA_U8.to_string(),
                )])),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt64Array::from(beta_rows.clone())),
                    Arc::new(UInt64Array::from(beta_vals_u64.clone())),
                    Arc::new(UInt8Array::from(beta_vals_u8.clone())),
                ],
            )
            .unwrap();
            table.append(&batch).unwrap();
        }

        {
            let table = Table::from_id(TABLE_GAMMA, Arc::clone(&pager)).unwrap();
            let schema = Arc::new(Schema::new(vec![
                Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
                Field::new("gamma_i16", DataType::Int16, false).with_metadata(HashMap::from([(
                    crate::constants::FIELD_ID_META_KEY.to_string(),
                    COL_GAMMA_I16.to_string(),
                )])),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt64Array::from(gamma_rows.clone())),
                    Arc::new(Int16Array::from(gamma_vals_i16.clone())),
                ],
            )
            .unwrap();
            table.append(&batch).unwrap();
        }

        // Second session: reopen each table and ensure schema and values are intact.
        {
            let table = Table::from_id(TABLE_ALPHA, Arc::clone(&pager)).unwrap();
            let store = table.store();

            let expectations: &[(FieldId, DataType)] = &[
                (COL_ALPHA_U64, DataType::UInt64),
                (COL_ALPHA_I32, DataType::Int32),
                (COL_ALPHA_U32, DataType::UInt32),
                (COL_ALPHA_I16, DataType::Int16),
            ];

            for &(col, ref ty) in expectations {
                let lfid = LogicalFieldId::for_user(TABLE_ALPHA, col);
                assert_eq!(store.data_type(lfid).unwrap(), *ty);
                let arr = gather_single(store, lfid, &alpha_rows);
                match ty {
                    DataType::UInt64 => {
                        let arr = arr.as_any().downcast_ref::<UInt64Array>().unwrap();
                        assert_eq!(arr.values(), alpha_vals_u64.as_slice());
                    }
                    DataType::Int32 => {
                        let arr = arr.as_any().downcast_ref::<Int32Array>().unwrap();
                        assert_eq!(arr.values(), alpha_vals_i32.as_slice());
                    }
                    DataType::UInt32 => {
                        let arr = arr.as_any().downcast_ref::<UInt32Array>().unwrap();
                        assert_eq!(arr.values(), alpha_vals_u32.as_slice());
                    }
                    DataType::Int16 => {
                        let arr = arr.as_any().downcast_ref::<Int16Array>().unwrap();
                        assert_eq!(arr.values(), alpha_vals_i16.as_slice());
                    }
                    other => panic!("unexpected dtype {other:?}"),
                }
            }
        }

        {
            let table = Table::from_id(TABLE_BETA, Arc::clone(&pager)).unwrap();
            let store = table.store();

            let lfid_u64 = LogicalFieldId::for_user(TABLE_BETA, COL_BETA_U64);
            assert_eq!(store.data_type(lfid_u64).unwrap(), DataType::UInt64);
            let arr_u64 = gather_single(store, lfid_u64, &beta_rows);
            let arr_u64 = arr_u64.as_any().downcast_ref::<UInt64Array>().unwrap();
            assert_eq!(arr_u64.values(), beta_vals_u64.as_slice());

            let lfid_u8 = LogicalFieldId::for_user(TABLE_BETA, COL_BETA_U8);
            assert_eq!(store.data_type(lfid_u8).unwrap(), DataType::UInt8);
            let arr_u8 = gather_single(store, lfid_u8, &beta_rows);
            let arr_u8 = arr_u8.as_any().downcast_ref::<UInt8Array>().unwrap();
            assert_eq!(arr_u8.values(), beta_vals_u8.as_slice());
        }

        {
            let table = Table::from_id(TABLE_GAMMA, Arc::clone(&pager)).unwrap();
            let store = table.store();
            let lfid = LogicalFieldId::for_user(TABLE_GAMMA, COL_GAMMA_I16);
            assert_eq!(store.data_type(lfid).unwrap(), DataType::Int16);
            let arr = gather_single(store, lfid, &gamma_rows);
            let arr = arr.as_any().downcast_ref::<Int16Array>().unwrap();
            assert_eq!(arr.values(), gamma_vals_i16.as_slice());
        }
    }

    #[test]
    fn test_scan_with_i32_filter() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = pred_expr(Filter {
            field_id: COL_C_I32,
            op: Operator::Equals(20.into()),
        });

        let mut vals: Vec<Option<u64>> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_A_U64)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    let a = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
                    vals.extend(
                        (0..a.len()).map(|i| if a.is_null(i) { None } else { Some(a.value(i)) }),
                    );
                },
            )
            .unwrap();
        assert_eq!(vals, vec![Some(200), Some(200)]);
    }

    #[test]
    fn test_scan_with_greater_than_filter() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = pred_expr(Filter {
            field_id: COL_C_I32,
            op: Operator::GreaterThan(15.into()),
        });

        let mut vals: Vec<Option<u64>> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_A_U64)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    let a = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
                    vals.extend(
                        (0..a.len()).map(|i| if a.is_null(i) { None } else { Some(a.value(i)) }),
                    );
                },
            )
            .unwrap();
        assert_eq!(vals, vec![Some(200), Some(300), Some(200)]);
    }

    #[test]
    fn test_scan_with_range_filter() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = pred_expr(Filter {
            field_id: COL_A_U64,
            op: Operator::Range {
                lower: Bound::Included(150.into()),
                upper: Bound::Excluded(300.into()),
            },
        });

        let mut vals: Vec<Option<i32>> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_C_I32)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    let a = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                    vals.extend(
                        (0..a.len()).map(|i| if a.is_null(i) { None } else { Some(a.value(i)) }),
                    );
                },
            )
            .unwrap();
        assert_eq!(vals, vec![Some(20), Some(20)]);
    }

    #[test]
    fn test_filtered_scan_sum_kernel() {
        // Trade-off note:
        // - We use Arrow's sum kernel per batch, then add the partial sums.
        // - This preserves Arrow null semantics and avoids concat.
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;

        let filter = pred_expr(Filter {
            field_id: COL_A_U64,
            op: Operator::Range {
                lower: Bound::Included(150.into()),
                upper: Bound::Excluded(300.into()),
            },
        });

        let mut total: u128 = 0;
        table
            .scan_stream(
                &[proj(&table, COL_A_U64)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    let a = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
                    if let Some(part) = sum(a) {
                        total += part as u128;
                    }
                },
            )
            .unwrap();

        assert_eq!(total, 400);
    }

    #[test]
    fn test_filtered_scan_sum_i32_kernel() {
        // Trade-off note:
        // - Per-batch sum + accumulate avoids building one big Array.
        // - For tiny batches overhead may match manual loops, but keeps
        //   Arrow semantics exact.
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let candidates = [100.into(), 300.into()];
        let filter = pred_expr(Filter {
            field_id: COL_A_U64,
            op: Operator::In(&candidates),
        });

        let mut total: i64 = 0;
        table
            .scan_stream(
                &[proj(&table, COL_C_I32)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    let a = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                    if let Some(part) = sum(a) {
                        total += part as i64;
                    }
                },
            )
            .unwrap();
        assert_eq!(total, 40);
    }

    #[test]
    fn test_filtered_scan_min_max_kernel() {
        // Trade-off note:
        // - min/max are computed per batch and folded. This preserves
        //   Arrow's null behavior and avoids concat.
        // - Be mindful of NaN semantics if extended to floats later.
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let candidates = [100.into(), 300.into()];
        let filter = pred_expr(Filter {
            field_id: COL_A_U64,
            op: Operator::In(&candidates),
        });

        let mut mn: Option<i32> = None;
        let mut mx: Option<i32> = None;
        table
            .scan_stream(
                &[proj(&table, COL_C_I32)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    let a = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

                    if let Some(part_min) = min(a) {
                        mn = Some(mn.map_or(part_min, |m| m.min(part_min)));
                    }
                    if let Some(part_max) = max(a) {
                        mx = Some(mx.map_or(part_max, |m| m.max(part_max)));
                    }
                },
            )
            .unwrap();
        assert_eq!(mn, Some(10));
        assert_eq!(mx, Some(30));
    }

    #[test]
    fn test_filtered_scan_float64_column() {
        let table = setup_test_table();
        const COL_D_F64: FieldId = 13;

        let filter = pred_expr(Filter {
            field_id: COL_D_F64,
            op: Operator::GreaterThan(2.0_f64.into()),
        });

        let mut got = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_D_F64)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    let arr = b.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
                    for i in 0..arr.len() {
                        if arr.is_valid(i) {
                            got.push(arr.value(i));
                        }
                    }
                },
            )
            .unwrap();

        assert_eq!(got, vec![2.5, 3.5, 2.5]);
    }

    #[test]
    fn test_filtered_scan_float32_in_operator() {
        let table = setup_test_table();
        const COL_E_F32: FieldId = 14;

        let candidates = [2.0_f32.into(), 3.0_f32.into()];
        let filter = pred_expr(Filter {
            field_id: COL_E_F32,
            op: Operator::In(&candidates),
        });

        let mut vals: Vec<Option<f32>> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_E_F32)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    let arr = b.column(0).as_any().downcast_ref::<Float32Array>().unwrap();
                    vals.extend((0..arr.len()).map(|i| {
                        if arr.is_null(i) {
                            None
                        } else {
                            Some(arr.value(i))
                        }
                    }));
                },
            )
            .unwrap();

        let collected: Vec<f32> = vals.into_iter().flatten().collect();
        assert_eq!(collected, vec![2.0, 3.0, 2.0]);
    }

    #[test]
    fn test_scan_stream_and_expression() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;
        const COL_E_F32: FieldId = 14;

        let expr = Expr::all_of(vec![
            Filter {
                field_id: COL_C_I32,
                op: Operator::GreaterThan(15.into()),
            },
            Filter {
                field_id: COL_A_U64,
                op: Operator::LessThan(250.into()),
            },
        ]);

        let mut vals: Vec<Option<f32>> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_E_F32)],
                &expr,
                ScanStreamOptions::default(),
                |b| {
                    let arr = b.column(0).as_any().downcast_ref::<Float32Array>().unwrap();
                    vals.extend((0..arr.len()).map(|i| {
                        if arr.is_null(i) {
                            None
                        } else {
                            Some(arr.value(i))
                        }
                    }));
                },
            )
            .unwrap();

        assert_eq!(vals, vec![Some(2.0_f32), Some(2.0_f32)]);
    }

    #[test]
    fn test_scan_stream_or_expression() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let expr = Expr::any_of(vec![
            Filter {
                field_id: COL_C_I32,
                op: Operator::Equals(10.into()),
            },
            Filter {
                field_id: COL_C_I32,
                op: Operator::Equals(30.into()),
            },
        ]);

        let mut vals: Vec<Option<u64>> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_A_U64)],
                &expr,
                ScanStreamOptions::default(),
                |b| {
                    let arr = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
                    vals.extend((0..arr.len()).map(|i| {
                        if arr.is_null(i) {
                            None
                        } else {
                            Some(arr.value(i))
                        }
                    }));
                },
            )
            .unwrap();

        assert_eq!(vals, vec![Some(100), Some(300)]);
    }

    #[test]
    fn test_scan_stream_not_predicate() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let expr = Expr::not(pred_expr(Filter {
            field_id: COL_C_I32,
            op: Operator::Equals(20.into()),
        }));

        let mut vals: Vec<Option<u64>> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_A_U64)],
                &expr,
                ScanStreamOptions::default(),
                |b| {
                    let arr = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
                    vals.extend((0..arr.len()).map(|i| {
                        if arr.is_null(i) {
                            None
                        } else {
                            Some(arr.value(i))
                        }
                    }));
                },
            )
            .unwrap();

        assert_eq!(vals, vec![Some(100), Some(300)]);
    }

    #[test]
    fn test_scan_stream_not_and_expression() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let expr = Expr::not(Expr::all_of(vec![
            Filter {
                field_id: COL_A_U64,
                op: Operator::GreaterThan(150.into()),
            },
            Filter {
                field_id: COL_C_I32,
                op: Operator::LessThan(40.into()),
            },
        ]));

        let mut vals: Vec<Option<u64>> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_A_U64)],
                &expr,
                ScanStreamOptions::default(),
                |b| {
                    let arr = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
                    vals.extend((0..arr.len()).map(|i| {
                        if arr.is_null(i) {
                            None
                        } else {
                            Some(arr.value(i))
                        }
                    }));
                },
            )
            .unwrap();

        assert_eq!(vals, vec![Some(100)]);
    }

    #[test]
    fn test_scan_stream_include_nulls_toggle() {
        let pager = Arc::new(MemPager::default());
        let table = setup_test_table_with_pager(&pager);
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;
        const COL_B_BIN: FieldId = 11;
        const COL_D_F64: FieldId = 13;
        const COL_E_F32: FieldId = 14;

        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("a_u64", DataType::UInt64, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_A_U64.to_string(),
            )])),
            Field::new("b_bin", DataType::Binary, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_B_BIN.to_string(),
            )])),
            Field::new("c_i32", DataType::Int32, true).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_C_I32.to_string(),
            )])),
            Field::new("d_f64", DataType::Float64, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_D_F64.to_string(),
            )])),
            Field::new("e_f32", DataType::Float32, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_E_F32.to_string(),
            )])),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![5, 6])),
                Arc::new(UInt64Array::from(vec![500, 600])),
                Arc::new(BinaryArray::from(vec![
                    Some(&b"new"[..]),
                    Some(&b"alt"[..]),
                ])),
                Arc::new(Int32Array::from(vec![Some(40), None])),
                Arc::new(Float64Array::from(vec![5.5, 6.5])),
                Arc::new(Float32Array::from(vec![5.0, 6.0])),
            ],
        )
        .unwrap();
        table.append(&batch).unwrap();

        let filter = pred_expr(Filter {
            field_id: COL_A_U64,
            op: Operator::GreaterThan(450.into()),
        });

        let mut default_vals: Vec<Option<i32>> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_C_I32)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    let arr = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                    default_vals.extend((0..arr.len()).map(|i| {
                        if arr.is_null(i) {
                            None
                        } else {
                            Some(arr.value(i))
                        }
                    }));
                },
            )
            .unwrap();
        assert_eq!(default_vals, vec![Some(40)]);

        let mut include_null_vals: Vec<Option<i32>> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_C_I32)],
                &filter,
                ScanStreamOptions {
                    include_nulls: true,
                    order: None,
                    row_id_filter: None,
                    include_row_ids: true,
                },
                |b| {
                    let arr = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

                    let mut paired_vals: Vec<(Option<i32>, Option<f64>)> = Vec::new();
                    table
                        .scan_stream(
                            &[proj(&table, COL_C_I32), proj(&table, COL_D_F64)],
                            &filter,
                            ScanStreamOptions::default(),
                            |b| {
                                assert_eq!(b.num_columns(), 2);
                                let c_arr =
                                    b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                                let d_arr =
                                    b.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
                                for i in 0..b.num_rows() {
                                    let c_val = if c_arr.is_null(i) {
                                        None
                                    } else {
                                        Some(c_arr.value(i))
                                    };
                                    let d_val = if d_arr.is_null(i) {
                                        None
                                    } else {
                                        Some(d_arr.value(i))
                                    };
                                    paired_vals.push((c_val, d_val));
                                }
                            },
                        )
                        .unwrap();
                    assert_eq!(paired_vals, vec![(Some(40), Some(5.5)), (None, Some(6.5))]);
                    include_null_vals.extend((0..arr.len()).map(|i| {
                        if arr.is_null(i) {
                            None
                        } else {
                            Some(arr.value(i))
                        }
                    }));
                },
            )
            .unwrap();
        assert_eq!(include_null_vals, vec![Some(40), None]);
    }

    #[test]
    fn test_filtered_scan_int_sqrt_float64() {
        // Trade-off note:
        // - We cast per batch and apply a compute unary kernel for sqrt.
        // - This keeps processing streaming and avoids per-value loops.
        // - `unary` operates on `PrimitiveArray<T>`; cast and downcast to
        //   `Float64Array` first.
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = pred_expr(Filter {
            field_id: COL_C_I32,
            op: Operator::GreaterThan(15.into()),
        });

        let mut got: Vec<f64> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_A_U64)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    let casted = cast(b.column(0), &arrow::datatypes::DataType::Float64).unwrap();
                    let f64_arr = casted.as_any().downcast_ref::<Float64Array>().unwrap();

                    // unary::<Float64Type, _, Float64Type>(...)
                    let sqrt_arr = unary::<
                        arrow::datatypes::Float64Type,
                        _,
                        arrow::datatypes::Float64Type,
                    >(f64_arr, |v: f64| v.sqrt());

                    for i in 0..sqrt_arr.len() {
                        if !sqrt_arr.is_null(i) {
                            got.push(sqrt_arr.value(i));
                        }
                    }
                },
            )
            .unwrap();

        let expected = [200_f64.sqrt(), 300_f64.sqrt(), 200_f64.sqrt()];
        assert_eq!(got, expected);
    }

    #[test]
    fn test_multi_field_kernels_with_filters() {
        // Trade-off note:
        // - All reductions use per-batch kernels + accumulation to stay
        //   streaming. No concat or whole-column materialization.
        use arrow::array::{Int16Array, UInt8Array, UInt32Array};

        let table = Table::from_id(2, Arc::new(MemPager::default())).unwrap();

        const COL_A_U64: FieldId = 20;
        const COL_D_U32: FieldId = 21;
        const COL_E_I16: FieldId = 22;
        const COL_F_U8: FieldId = 23;
        const COL_C_I32: FieldId = 24;

        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("a_u64", DataType::UInt64, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_A_U64.to_string(),
            )])),
            Field::new("d_u32", DataType::UInt32, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_D_U32.to_string(),
            )])),
            Field::new("e_i16", DataType::Int16, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_E_I16.to_string(),
            )])),
            Field::new("f_u8", DataType::UInt8, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_F_U8.to_string(),
            )])),
            Field::new("c_i32", DataType::Int32, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                COL_C_I32.to_string(),
            )])),
        ]));

        // Data: 5 rows. We will filter c_i32 >= 20 -> keep rows 2..5.
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(UInt64Array::from(vec![100, 225, 400, 900, 1600])),
                Arc::new(UInt32Array::from(vec![7, 8, 9, 10, 11])),
                Arc::new(Int16Array::from(vec![-3, 0, 3, -6, 6])),
                Arc::new(UInt8Array::from(vec![2, 4, 6, 8, 10])),
                Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50])),
            ],
        )
        .unwrap();

        table.append(&batch).unwrap();

        // Filter: c_i32 >= 20.
        let filter = pred_expr(Filter {
            field_id: COL_C_I32,
            op: Operator::GreaterThanOrEquals(20.into()),
        });

        // 1) SUM over d_u32 (per-batch sum + accumulate).
        let mut d_sum: u128 = 0;
        table
            .scan_stream(
                &[proj(&table, COL_D_U32)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    let a = b.column(0).as_any().downcast_ref::<UInt32Array>().unwrap();
                    if let Some(part) = sum(a) {
                        d_sum += part as u128;
                    }
                },
            )
            .unwrap();
        assert_eq!(d_sum, (8 + 9 + 10 + 11) as u128);

        // 2) MIN over e_i16 (per-batch min + fold).
        let mut e_min: Option<i16> = None;
        table
            .scan_stream(
                &[proj(&table, COL_E_I16)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    let a = b.column(0).as_any().downcast_ref::<Int16Array>().unwrap();
                    if let Some(part_min) = min(a) {
                        e_min = Some(e_min.map_or(part_min, |m| m.min(part_min)));
                    }
                },
            )
            .unwrap();
        assert_eq!(e_min, Some(-6));

        // 3) MAX over f_u8 (per-batch max + fold).
        let mut f_max: Option<u8> = None;
        table
            .scan_stream(
                &[proj(&table, COL_F_U8)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    let a = b
                        .column(0)
                        .as_any()
                        .downcast_ref::<arrow::array::UInt8Array>()
                        .unwrap();
                    if let Some(part_max) = max(a) {
                        f_max = Some(f_max.map_or(part_max, |m| m.max(part_max)));
                    }
                },
            )
            .unwrap();
        assert_eq!(f_max, Some(10));

        // 4) SQRT over a_u64 (cast to f64, then unary sqrt per batch).
        let mut got: Vec<f64> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_A_U64)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    let casted = cast(b.column(0), &arrow::datatypes::DataType::Float64).unwrap();
                    let f64_arr = casted.as_any().downcast_ref::<Float64Array>().unwrap();
                    let sqrt_arr = unary::<
                        arrow::datatypes::Float64Type,
                        _,
                        arrow::datatypes::Float64Type,
                    >(f64_arr, |v: f64| v.sqrt());

                    for i in 0..sqrt_arr.len() {
                        if !sqrt_arr.is_null(i) {
                            got.push(sqrt_arr.value(i));
                        }
                    }
                },
            )
            .unwrap();
        let expected = [15.0_f64, 20.0, 30.0, 40.0];
        assert_eq!(got, expected);
    }

    #[test]
    fn test_scan_with_in_filter() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        // IN now uses untyped literals, too.
        let candidates = [10.into(), 30.into()];
        let filter = pred_expr(Filter {
            field_id: COL_C_I32,
            op: Operator::In(&candidates),
        });

        let mut vals: Vec<Option<u64>> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_A_U64)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    let a = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
                    vals.extend(
                        (0..a.len()).map(|i| if a.is_null(i) { None } else { Some(a.value(i)) }),
                    );
                },
            )
            .unwrap();
        assert_eq!(vals, vec![Some(100), Some(300)]);
    }

    #[test]
    fn test_scan_stream_single_column_batches() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        // Filter c_i32 == 20 -> two rows; stream a_u64 in batches of <= N.
        let filter = pred_expr(Filter {
            field_id: COL_C_I32,
            op: Operator::Equals(20.into()),
        });

        let mut seen_cols = Vec::<u64>::new();
        table
            .scan_stream(
                &[proj(&table, COL_A_U64)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    assert_eq!(b.num_columns(), 1);
                    let a = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
                    // No kernel needed; just collect values for shape assertions.
                    for i in 0..a.len() {
                        if !a.is_null(i) {
                            seen_cols.push(a.value(i));
                        }
                    }
                },
            )
            .unwrap();

        // In fixture, c_i32 == 20 corresponds to a_u64 values [200, 200].
        assert_eq!(seen_cols, vec![200, 200]);
    }

    #[test]
    fn test_scan_with_multiple_projection_columns() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = pred_expr(Filter {
            field_id: COL_C_I32,
            op: Operator::Equals(20.into()),
        });

        let expected_names = [COL_A_U64.to_string(), COL_C_I32.to_string()];

        let mut combined: Vec<(Option<u64>, Option<i32>)> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_A_U64), proj(&table, COL_C_I32)],
                &filter,
                ScanStreamOptions::default(),
                |b| {
                    assert_eq!(b.num_columns(), 2);
                    assert_eq!(b.schema().field(0).name(), &expected_names[0]);
                    assert_eq!(b.schema().field(1).name(), &expected_names[1]);

                    let a = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
                    let c = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
                    for i in 0..b.num_rows() {
                        let left = if a.is_null(i) { None } else { Some(a.value(i)) };
                        let right = if c.is_null(i) { None } else { Some(c.value(i)) };
                        combined.push((left, right));
                    }
                },
            )
            .unwrap();

        assert_eq!(combined, vec![(Some(200), Some(20)), (Some(200), Some(20))]);
    }

    #[test]
    fn test_scan_stream_projection_validation() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = pred_expr(Filter {
            field_id: COL_C_I32,
            op: Operator::Equals(20.into()),
        });

        let empty: [Projection; 0] = [];
        let result = table.scan_stream(&empty, &filter, ScanStreamOptions::default(), |_batch| {});
        assert!(matches!(result, Err(Error::InvalidArgumentError(_))));

        // Duplicate projections are allowed: the same column will be
        // gathered once and duplicated in the output in the requested
        // order. Verify the call succeeds and produces two identical
        // columns per batch.
        let duplicate = [
            proj(&table, COL_A_U64),
            proj_alias(&table, COL_A_U64, "alias_a"),
        ];
        let mut collected = Vec::<u64>::new();
        table
            .scan_stream(&duplicate, &filter, ScanStreamOptions::default(), |b| {
                assert_eq!(b.num_columns(), 2);
                assert_eq!(b.schema().field(0).name(), &COL_A_U64.to_string());
                assert_eq!(b.schema().field(1).name(), "alias_a");
                let a0 = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
                let a1 = b.column(1).as_any().downcast_ref::<UInt64Array>().unwrap();
                for i in 0..b.num_rows() {
                    if !a0.is_null(i) {
                        collected.push(a0.value(i));
                    }
                    if !a1.is_null(i) {
                        collected.push(a1.value(i));
                    }
                }
            })
            .unwrap();
        // Two matching rows, two columns per row -> four values.
        assert_eq!(collected, vec![200, 200, 200, 200]);
    }

    #[test]
    fn test_scan_stream_computed_projection() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;

        let projections = [
            ScanProjection::column(proj(&table, COL_A_U64)),
            ScanProjection::computed(
                ScalarExpr::binary(
                    ScalarExpr::column(COL_A_U64),
                    BinaryOp::Multiply,
                    ScalarExpr::literal(2),
                ),
                "a_times_two",
            ),
        ];

        let filter = pred_expr(Filter {
            field_id: COL_A_U64,
            op: Operator::GreaterThanOrEquals(0.into()),
        });

        let mut computed: Vec<(u64, f64)> = Vec::new();
        table
            .scan_stream_with_exprs(&projections, &filter, ScanStreamOptions::default(), |b| {
                assert_eq!(b.num_columns(), 2);
                let base = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
                let comp = b.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
                for i in 0..b.num_rows() {
                    if base.is_null(i) || comp.is_null(i) {
                        continue;
                    }
                    computed.push((base.value(i), comp.value(i)));
                }
            })
            .unwrap();

        let expected = vec![(100, 200.0), (200, 400.0), (300, 600.0), (200, 400.0)];
        assert_eq!(computed, expected);
    }

    #[test]
    fn test_scan_stream_multi_column_filter_compare() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let expr = Expr::Compare {
            left: ScalarExpr::binary(
                ScalarExpr::column(COL_A_U64),
                BinaryOp::Add,
                ScalarExpr::column(COL_C_I32),
            ),
            op: CompareOp::Gt,
            right: ScalarExpr::literal(220_i64),
        };

        let mut vals: Vec<Option<u64>> = Vec::new();
        table
            .scan_stream(
                &[proj(&table, COL_A_U64)],
                &expr,
                ScanStreamOptions::default(),
                |b| {
                    let col = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
                    for i in 0..b.num_rows() {
                        vals.push(if col.is_null(i) {
                            None
                        } else {
                            Some(col.value(i))
                        });
                    }
                },
            )
            .unwrap();

        assert_eq!(vals.into_iter().flatten().collect::<Vec<_>>(), vec![300]);
    }
}
