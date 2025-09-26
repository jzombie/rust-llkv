use std::cmp;
use std::fmt;
use std::ops::Bound;
use std::sync::Arc;

use crate::types::TableId;

use arrow::array::{Array, ArrayRef, PrimitiveArray, RecordBatch, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::{ArrowPrimitiveType, Field, Schema};

use llkv_column_map::store::{FilterPrimitive, ROW_ID_COLUMN_NAME};
use llkv_column_map::{
    ColumnStore, scan,
    types::{LogicalFieldId, Namespace},
};
use llkv_storage::pager::{MemPager, Pager};
use simd_r_drive_entry_handle::EntryHandle;

use crate::expr::{Filter, LiteralCastError, Operator, bound_to_native, literal_to_native};
use crate::sys_catalog::{CATALOG_TID, ColMeta, SysCatalog, TableMeta};
use crate::types::FieldId;
use llkv_expr::literal::FromLiteral;

// TODO: Extract to constants
/// Default max rows per streamed batch. Tune as needed.
const STREAM_BATCH_ROWS: usize = 8192;

// TODO: Move to `llkv-expr`?
enum RangeLimit<T> {
    Included(T),
    Excluded(T),
}

// TODO: Move to `llkv-expr`?
enum Predicate<T>
where
    T: ArrowPrimitiveType + FilterPrimitive,
{
    All,
    Equals(T::Native),
    GreaterThan(T::Native),
    GreaterThanOrEquals(T::Native),
    LessThan(T::Native),
    LessThanOrEquals(T::Native),
    Range {
        lower: Option<RangeLimit<T::Native>>,
        upper: Option<RangeLimit<T::Native>>,
    },
    In(Vec<T::Native>),
}

impl<T> Predicate<T>
where
    T: ArrowPrimitiveType + FilterPrimitive,
{
    fn build(op: &Operator<'_>) -> Result<Predicate<T>, TableError>
    where
        T: ArrowPrimitiveType + FilterPrimitive,
        T::Native: FromLiteral + Copy,
    {
        match op {
            Operator::Equals(lit) => Ok(Predicate::Equals(literal_to_native::<T::Native>(lit)?)),
            Operator::GreaterThan(lit) => {
                Ok(Predicate::GreaterThan(literal_to_native::<T::Native>(lit)?))
            }
            Operator::GreaterThanOrEquals(lit) => {
                Ok(Predicate::GreaterThanOrEquals(literal_to_native::<
                    T::Native,
                >(lit)?))
            }
            Operator::LessThan(lit) => {
                Ok(Predicate::LessThan(literal_to_native::<T::Native>(lit)?))
            }
            Operator::LessThanOrEquals(lit) => Ok(Predicate::LessThanOrEquals(
                literal_to_native::<T::Native>(lit)?,
            )),
            Operator::Range { lower, upper } => {
                let lb = match bound_to_native::<T>(lower)? {
                    Bound::Unbounded => None,
                    Bound::Included(v) => Some(RangeLimit::Included(v)),
                    Bound::Excluded(v) => Some(RangeLimit::Excluded(v)),
                };
                let ub = match bound_to_native::<T>(upper)? {
                    Bound::Unbounded => None,
                    Bound::Included(v) => Some(RangeLimit::Included(v)),
                    Bound::Excluded(v) => Some(RangeLimit::Excluded(v)),
                };
                if lb.is_none() && ub.is_none() {
                    Ok(Predicate::All)
                } else {
                    Ok(Predicate::Range {
                        lower: lb,
                        upper: ub,
                    })
                }
            }
            Operator::In(values) => {
                let mut natives = Vec::with_capacity(values.len());
                for lit in *values {
                    natives.push(literal_to_native::<T::Native>(lit)?);
                }
                Ok(Predicate::In(natives))
            }
            // Pattern/string ops are not supported in this numeric predicate path.
            _ => Err(TableError::Internal(
                "Filter operator does not contain a supported typed value".to_string(),
            )),
        }
    }

    fn matches(&self, value: T::Native) -> bool {
        match self {
            Predicate::All => true,
            Predicate::Equals(target) => value == *target,
            Predicate::GreaterThan(target) => value > *target,
            Predicate::GreaterThanOrEquals(target) => value >= *target,
            Predicate::LessThan(target) => value < *target,
            Predicate::LessThanOrEquals(target) => value <= *target,
            Predicate::Range { lower, upper } => {
                if let Some(limit) = lower {
                    match limit {
                        RangeLimit::Included(bound) => {
                            if value < *bound {
                                return false;
                            }
                        }
                        RangeLimit::Excluded(bound) => {
                            if value <= *bound {
                                return false;
                            }
                        }
                    }
                }
                if let Some(limit) = upper {
                    match limit {
                        RangeLimit::Included(bound) => {
                            if value > *bound {
                                return false;
                            }
                        }
                        RangeLimit::Excluded(bound) => {
                            if value >= *bound {
                                return false;
                            }
                        }
                    }
                }
                true
            }
            Predicate::In(values) => values.contains(&value),
        }
    }
}

#[inline]
fn lfid_for(table_id: TableId, column_id: FieldId) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_table_id(table_id)
        .with_field_id(column_id)
        .with_namespace(Namespace::UserData)
}

// TODO: Migrate to `llkv-result`
#[derive(Debug)]
pub enum TableError {
    ColumnMap(llkv_result::Error),
    Arrow(arrow::error::ArrowError),
    ExprCast(LiteralCastError),
    ReservedTableId(TableId),
    Internal(String),
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableError::ColumnMap(err) => write!(f, "column map error: {err}"),
            TableError::Arrow(err) => write!(f, "arrow error: {err}"),
            TableError::ExprCast(err) => write!(f, "expression cast error: {err}"),
            TableError::ReservedTableId(table_id) => {
                write!(f, "table id {table_id} is reserved for system catalogs")
            }
            TableError::Internal(msg) => write!(f, "internal table error: {msg}"),
        }
    }
}

impl std::error::Error for TableError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TableError::ColumnMap(err) => Some(err),
            TableError::Arrow(err) => Some(err),
            TableError::ExprCast(err) => Some(err),
            TableError::ReservedTableId(_) => None,
            TableError::Internal(_) => None,
        }
    }
}

impl From<llkv_result::Error> for TableError {
    fn from(e: llkv_result::Error) -> Self {
        TableError::ColumnMap(e)
    }
}

impl From<arrow::error::ArrowError> for TableError {
    fn from(e: arrow::error::ArrowError) -> Self {
        TableError::Arrow(e)
    }
}

impl From<LiteralCastError> for TableError {
    fn from(e: LiteralCastError) -> Self {
        TableError::ExprCast(e)
    }
}

pub struct Table<P = MemPager>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    store: ColumnStore<P>,
    table_id: TableId,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ScanStreamOptions {
    pub include_nulls: bool,
}

impl<P> Table<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn new(table_id: TableId, pager: Arc<P>) -> Result<Self, TableError> {
        if table_id == CATALOG_TID {
            return Err(TableError::ReservedTableId(table_id));
        }

        let store = ColumnStore::open(pager)?;
        Ok(Self { store, table_id })
    }

    pub fn append(&self, batch: &RecordBatch) -> Result<(), llkv_result::Error> {
        let mut new_fields = Vec::with_capacity(batch.schema().fields().len());
        for field in batch.schema().fields() {
            if field.name() == ROW_ID_COLUMN_NAME {
                new_fields.push(field.as_ref().clone());
                continue;
            }

            let user_field_id: FieldId = field
                .metadata()
                .get("field_id")
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| {
                    llkv_result::Error::Internal(format!(
                        "Field '{}' is missing a valid 'field_id' in its \
                         metadata.",
                        field.name()
                    ))
                })?;

            let lfid = lfid_for(self.table_id, user_field_id);
            let mut new_metadata = field.metadata().clone();
            let lfid_val: u64 = lfid.into();
            new_metadata.insert("field_id".to_string(), lfid_val.to_string());

            let new_field =
                Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                    .with_metadata(new_metadata);
            new_fields.push(new_field);
        }

        let new_schema = Arc::new(Schema::new(new_fields));
        let namespaced_batch = RecordBatch::try_new(new_schema, batch.columns().to_vec())?;
        self.store.append(&namespaced_batch)
    }

    /// Stream one projected column as a sequence of RecordBatches.
    ///
    /// - Avoids `concat` and large materializations.
    /// - Uses the scan builder to stream logical batches directly from
    ///   column-map without allocating whole-column materializations.
    /// - Applies predicate filtering inside column-map and emits compact
    ///   Arrow batches that contain the requested projection only.
    pub fn scan_stream<'a, F>(
        &self,
        projection_col: FieldId,
        filter: &Filter<'a, FieldId>,
        on_batch: F,
    ) -> Result<(), TableError>
    where
        F: FnMut(RecordBatch),
    {
        self.scan_stream_with_options(
            &[projection_col],
            filter,
            ScanStreamOptions::default(),
            on_batch,
        )
    }

    pub fn scan_stream_with_options<'a, F>(
        &self,
        projection_cols: &[FieldId],
        filter: &Filter<'a, FieldId>,
        options: ScanStreamOptions,
        mut on_batch: F,
    ) -> Result<(), TableError>
    where
        F: FnMut(RecordBatch),
    {
        if projection_cols.is_empty() {
            return Err(TableError::Internal(
                "scan_stream_with_options requires at least one projection column".into(),
            ));
        }

        let filter_lfid = lfid_for(self.table_id, filter.field_id);
        let dtype = self.store.data_type(filter_lfid)?;

        let mut projection_lfids: Vec<LogicalFieldId> = Vec::with_capacity(projection_cols.len());
        let mut out_fields: Vec<Field> = Vec::with_capacity(projection_cols.len());
        for &col_id in projection_cols {
            let proj_lfid = lfid_for(self.table_id, col_id);
            let proj_dtype = self.store.data_type(proj_lfid)?;
            projection_lfids.push(proj_lfid);
            out_fields.push(Field::new(col_id.to_string(), proj_dtype.clone(), true));
        }
        let out_schema = Arc::new(Schema::new(out_fields));

        let mut builder_columns: Vec<LogicalFieldId> =
            Vec::with_capacity(1 + projection_lfids.len());
        builder_columns.push(filter_lfid);
        for &lfid in &projection_lfids {
            if !builder_columns.contains(&lfid) {
                builder_columns.push(lfid);
            }
        }

        let mut scan_opts = scan::ScanOptions::default();
        scan_opts.include_nulls = options.include_nulls;

        llkv_column_map::with_integer_arrow_type!(
            dtype.clone(),
            |ArrowTy| {
                let predicate = Predicate::<ArrowTy>::build(&filter.op)?;
                let builder = scan::ScanBuilder::with_columns(&self.store, &builder_columns)
                    .options(scan_opts);

                builder
                    .project(|mp_batch| {
                        process_multi_projection_batch::<ArrowTy, _>(
                            filter_lfid,
                            &projection_lfids,
                            mp_batch,
                            &predicate,
                            options.include_nulls,
                            &out_schema,
                            &mut on_batch,
                        )
                    })
                    .map_err(TableError::from)?;

                Ok(())
            },
            Err(TableError::Internal(format!(
                "Filtering on type {:?} is not supported",
                dtype
            ))),
        )
    }

    #[inline]
    pub fn catalog(&self) -> SysCatalog<'_, P> {
        SysCatalog::new(&self.store)
    }

    #[inline]
    pub fn put_table_meta(&self, meta: &TableMeta) {
        debug_assert_eq!(meta.table_id, self.table_id);
        self.catalog().put_table_meta(meta);
    }

    #[inline]
    pub fn get_table_meta(&self) -> Option<TableMeta> {
        self.catalog().get_table_meta(self.table_id)
    }

    #[inline]
    pub fn put_col_meta(&self, meta: &ColMeta) {
        self.catalog().put_col_meta(self.table_id, meta);
    }

    #[inline]
    pub fn get_cols_meta(&self, col_ids: &[FieldId]) -> Vec<Option<ColMeta>> {
        self.catalog().get_cols_meta(self.table_id, col_ids)
    }

    pub fn store(&self) -> &ColumnStore<P> {
        &self.store
    }
}

fn process_multi_projection_batch<ArrowTy, F>(
    filter_lfid: LogicalFieldId,
    projection_lfids: &[LogicalFieldId],
    batch: scan::MultiProjectionBatch,
    predicate: &Predicate<ArrowTy>,
    include_nulls: bool,
    out_schema: &Arc<Schema>,
    on_batch: &mut F,
) -> Result<(), llkv_result::Error>
where
    ArrowTy: ArrowPrimitiveType + FilterPrimitive,
    ArrowTy::Native: FromLiteral + Copy,
    F: FnMut(RecordBatch),
{
    let driver_pos = batch
        .columns
        .iter()
        .position(|fid| *fid == filter_lfid)
        .ok_or_else(|| llkv_result::Error::Internal("project: filter column missing".into()))?;
    let driver_column = batch.record_batch.column(1 + driver_pos).clone();
    let filter_values = driver_column
        .as_any()
        .downcast_ref::<PrimitiveArray<ArrowTy>>()
        .ok_or_else(|| {
            llkv_result::Error::Internal("project: filter column type mismatch".into())
        })?;

    let mut indices: Vec<u32> = Vec::new();
    for (idx, maybe_value) in filter_values.iter().enumerate() {
        if let Some(value) = maybe_value {
            if predicate.matches(value) {
                indices.push(idx as u32);
            }
        }
    }

    if indices.is_empty() {
        return Ok(());
    }

    let indices_array = UInt32Array::from(indices);
    let mut projected: Vec<ArrayRef> = Vec::with_capacity(projection_lfids.len());
    for &proj_lfid in projection_lfids {
        let projection_pos = batch
            .columns
            .iter()
            .position(|fid| *fid == proj_lfid)
            .ok_or_else(|| {
                llkv_result::Error::Internal("project: projection column missing".into())
            })?;
        let col_idx = 1 + projection_pos;
        let column = batch.record_batch.column(col_idx).clone();
        let taken =
            take(column.as_ref(), &indices_array, None).map_err(llkv_result::Error::from)?;
        projected.push(taken);
    }

    if projected.is_empty() {
        return Ok(());
    }

    if !include_nulls {
        let row_count = projected[0].len();
        let mut keep: Vec<u32> = Vec::with_capacity(row_count);
        for idx in 0..row_count {
            if projected.iter().any(|col| col.is_null(idx)) {
                continue;
            }
            keep.push(idx as u32);
        }

        if keep.len() != row_count {
            if keep.is_empty() {
                return Ok(());
            }
            let keep_indices = UInt32Array::from(keep);
            for col in projected.iter_mut() {
                let filtered =
                    take(col.as_ref(), &keep_indices, None).map_err(llkv_result::Error::from)?;
                *col = filtered;
            }
        }
    }

    let total_len = projected[0].len();
    if total_len == 0 {
        return Ok(());
    }

    let mut offset = 0;
    while offset < total_len {
        let chunk_len = cmp::min(STREAM_BATCH_ROWS, total_len - offset);
        if chunk_len == 0 {
            break;
        }
        let mut chunk_columns: Vec<ArrayRef> = Vec::with_capacity(projected.len());
        for col in &projected {
            chunk_columns.push(col.slice(offset, chunk_len));
        }
        if chunk_columns.is_empty() {
            break;
        }
        let batch = RecordBatch::try_new(out_schema.clone(), chunk_columns)
            .map_err(llkv_result::Error::from)?;
        on_batch(batch);
        offset += chunk_len;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, ArrayRef, Int32Array, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use llkv_storage::pager::MemPager;
    use std::collections::HashMap;

    const TABLE_ID_SMALL: TableId = 1;
    const TABLE_ID_LARGE: TableId = 2;
    const COL_A_U64: FieldId = 10;
    const COL_B_U64: FieldId = 12;
    const COL_C_I32: FieldId = 11;
    const COL_BIG_U64: FieldId = 42;

    fn collect_batches<F>(table: &Table, proj: FieldId, filter: &Filter<'_, FieldId>, mut f: F)
    where
        F: FnMut(RecordBatch),
    {
        table.scan_stream(proj, filter, |batch| f(batch)).unwrap();
    }

    fn setup_small_table() -> Table {
        let pager = Arc::new(MemPager::default());
        let table = Table::new(TABLE_ID_SMALL, Arc::clone(&pager)).unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("a_u64", DataType::UInt64, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_A_U64.to_string(),
            )])),
            Field::new("c_i32", DataType::Int32, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_C_I32.to_string(),
            )])),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 3, 4])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![100, 200, 300, 200])) as ArrayRef,
                Arc::new(Int32Array::from(vec![10, 20, 30, 20])) as ArrayRef,
            ],
        )
        .unwrap();

        table.append(&batch).unwrap();
        table
    }

    fn setup_table_with_null_projection() -> Table {
        let pager = Arc::new(MemPager::default());
        let table = Table::new(TABLE_ID_SMALL, Arc::clone(&pager)).unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("a_u64", DataType::UInt64, true).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_A_U64.to_string(),
            )])),
            Field::new("b_u64", DataType::UInt64, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_B_U64.to_string(),
            )])),
            Field::new("c_i32", DataType::Int32, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_C_I32.to_string(),
            )])),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 3])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![Some(100), None, Some(300)])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![1000, 2000, 3000])) as ArrayRef,
                Arc::new(Int32Array::from(vec![10, 20, 30])) as ArrayRef,
            ],
        )
        .unwrap();

        table.append(&batch).unwrap();
        table
    }

    #[test]
    fn scan_stream_filters_and_preserves_order() {
        let table = setup_small_table();

        let filter = Filter {
            field_id: COL_C_I32,
            op: Operator::Equals(20.into()),
        };

        let mut batches: Vec<RecordBatch> = Vec::new();
        collect_batches(&table, COL_A_U64, &filter, |batch| batches.push(batch));

        assert!(!batches.is_empty(), "expected at least one batch");

        let mut collected = Vec::new();
        for batch in batches {
            assert_eq!(batch.num_columns(), 1);
            let values = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            collected.extend(values.iter().flatten());
        }

        assert_eq!(collected, vec![200, 200]);
    }

    #[test]
    fn scan_stream_handles_null_projection_values() {
        let table = setup_table_with_null_projection();

        let filter = Filter {
            field_id: COL_C_I32,
            op: Operator::GreaterThanOrEquals(0.into()),
        };

        let mut non_null_values = Vec::new();
        collect_batches(&table, COL_A_U64, &filter, |batch| {
            assert_eq!(batch.num_columns(), 1);
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(column.null_count(), 0);
            non_null_values.extend(column.iter().flatten());
        });

        assert_eq!(non_null_values, vec![100, 300]);

        let mut values_with_nulls: Vec<Option<u64>> = Vec::new();
        let include_null_options = ScanStreamOptions {
            include_nulls: true,
        };
        table
            .scan_stream_with_options(&[COL_A_U64], &filter, include_null_options, |batch| {
                assert_eq!(batch.num_columns(), 1);
                let column = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                values_with_nulls.extend(column.iter());
            })
            .expect("scan_stream_with_options should succeed");

        assert_eq!(values_with_nulls, vec![Some(100), None, Some(300)]);

        // Verify multi-column streaming keeps values aligned when one column contains nulls.
        let mut multi_column_a: Vec<Option<u64>> = Vec::new();
        let mut multi_column_b: Vec<u64> = Vec::new();

        table
            .scan_stream_with_options(
                &[COL_A_U64, COL_B_U64],
                &filter,
                include_null_options,
                |batch| {
                    assert_eq!(batch.num_columns(), 2);
                    let col_a = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .unwrap();
                    let col_b = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .unwrap();
                    for idx in 0..batch.num_rows() {
                        if col_a.is_null(idx) {
                            multi_column_a.push(None);
                        } else {
                            multi_column_a.push(Some(col_a.value(idx)));
                        }
                        multi_column_b.push(col_b.value(idx));
                    }
                },
            )
            .expect("multi-column scan_stream_with_options should succeed");

        assert_eq!(multi_column_a, vec![Some(100), None, Some(300)]);
        assert_eq!(multi_column_b, vec![1000, 2000, 3000]);
    }

    fn setup_large_table() -> Table {
        let pager = Arc::new(MemPager::default());
        let table = Table::new(TABLE_ID_LARGE, Arc::clone(&pager)).unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("big_u64", DataType::UInt64, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_BIG_U64.to_string(),
            )])),
        ]));

        let first_chunk: Vec<u64> = (0..6000).collect();
        let second_chunk: Vec<u64> = (6000..12000).collect();

        let batch_a = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(first_chunk.clone())) as ArrayRef,
                Arc::new(UInt64Array::from_iter(first_chunk.iter().map(|v| v + 10))) as ArrayRef,
            ],
        )
        .unwrap();
        let batch_b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(second_chunk.clone())) as ArrayRef,
                Arc::new(UInt64Array::from_iter(second_chunk.iter().map(|v| v + 10))) as ArrayRef,
            ],
        )
        .unwrap();

        table.append(&batch_a).unwrap();
        table.append(&batch_b).unwrap();
        table
    }

    #[test]
    fn scan_stream_emits_sequential_batches() {
        let table = setup_large_table();

        let filter = Filter {
            field_id: COL_BIG_U64,
            op: Operator::GreaterThanOrEquals(0.into()),
        };

        let mut batches = Vec::new();
        collect_batches(&table, COL_BIG_U64, &filter, |batch| batches.push(batch));

        assert!(batches.len() > 1, "expected multiple batches");

        let mut expected_value = 10u64;
        for batch in batches {
            assert_eq!(batch.num_columns(), 1);
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert!(column.len() <= STREAM_BATCH_ROWS);
            for value in column.iter().flatten() {
                assert_eq!(value, expected_value);
                expected_value += 1;
            }
        }

        assert_eq!(expected_value, 12010);
    }
}
