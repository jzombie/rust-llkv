use std::cmp;
use std::fmt;
use std::ops::Bound;
use std::sync::Arc;

use crate::types::TableId;

use arrow::array::{Array, ArrayRef, Int32Array, PrimitiveArray, RecordBatch, UInt64Array};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Schema};

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

fn collect_matching_row_ids<T, P>(
    store: &ColumnStore<P>,
    field_id: LogicalFieldId,
    op: &Operator<'_>,
) -> Result<Vec<u64>, TableError>
where
    T: ArrowPrimitiveType + FilterPrimitive,
    T::Native: FromLiteral + Copy,
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let predicate = Predicate::<T>::build(op)?;
    store
        .filter_row_ids::<T, _>(field_id, move |value| predicate.matches(value))
        .map_err(TableError::from)
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

    /// Stream a single projected column as a sequence of RecordBatches.
    ///
    /// - Avoids `concat` and large materializations.
    /// - Uses the same filter machinery as the old `scan` to produce
    ///   `row_ids`.
    /// - Splits `row_ids` into fixed-size windows and gathers rows per
    ///   window to form a small `RecordBatch` that is sent to `on_batch`.
    ///
    /// Notes:
    /// - Currently supports exactly 1 projected column. This avoids
    ///   cross-column chunk alignment problems without changing
    ///   column-map APIs. We can generalize once a chunk-aligned
    ///   multi-column scan is exposed upstream.
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
            projection_col,
            filter,
            ScanStreamOptions::default(),
            on_batch,
        )
    }

    pub fn scan_stream_with_options<'a, F>(
        &self,
        projection_col: FieldId,
        filter: &Filter<'a, FieldId>,
        options: ScanStreamOptions,
        mut on_batch: F,
    ) -> Result<(), TableError>
    where
        F: FnMut(RecordBatch),
    {
        let filter_lfid = lfid_for(self.table_id, filter.field_id);
        let anchor_lfid =
            lfid_for(self.table_id, filter.field_id).with_namespace(Namespace::RowIdShadow);
        let dtype = self.store.data_type(filter_lfid)?;

        let row_ids = llkv_column_map::with_integer_arrow_type!(
            dtype.clone(),
            |ArrowTy| collect_matching_row_ids::<ArrowTy, P>(&self.store, filter_lfid, &filter.op),
            Err(TableError::Internal(format!(
                "Filtering on type {:?} is not supported",
                dtype
            ))),
        )?;

        // If nothing matches, emit nothing.
        if row_ids.is_empty() {
            return Ok(());
        }

        // Prepare schema for the single projected column.
        let proj_lfid = lfid_for(self.table_id, projection_col);
        let proj_dtype = self.store.data_type(proj_lfid)?;
        let proj_field = Field::new(projection_col.to_string(), proj_dtype.clone(), true);
        let out_schema = Arc::new(Schema::new(vec![proj_field]));

        // Stream in fixed-size windows of row_ids.
        let mut start = 0usize;
        while start < row_ids.len() {
            let end = cmp::min(start + STREAM_BATCH_ROWS, row_ids.len());
            let window = &row_ids[start..end];

            // Gather only this window's rows for the projected column.
            let arr = if options.include_nulls {
                Some(
                    self.store
                        .gather_rows_with_nulls(proj_lfid, window, Some(anchor_lfid))
                        .map_err(TableError::from)?,
                )
            } else {
                match self.store.gather_rows(proj_lfid, window) {
                    Ok(values) => Some(values),
                    Err(err) => Some(match err {
                        llkv_result::Error::Internal(_) | llkv_result::Error::NotFound => self
                            .store
                            .gather_rows_with_nulls(proj_lfid, window, Some(anchor_lfid))
                            .map_err(TableError::from)?,
                        _ => return Err(TableError::from(err)),
                    }),
                }
            };

            let Some(arr) = arr else {
                start = end;
                continue;
            };

            let maybe_arr = if options.include_nulls {
                Some(arr)
            } else {
                drop_nulls(arr)
            };

            let Some(col) = maybe_arr else {
                start = end;
                continue;
            };

            if col.is_empty() {
                start = end;
                continue;
            }

            let batch = RecordBatch::try_new(out_schema.clone(), vec![col])?;
            on_batch(batch);

            start = end;
        }

        Ok(())
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

fn drop_nulls(arr: ArrayRef) -> Option<ArrayRef> {
    if arr.null_count() == 0 {
        return Some(arr);
    }

    fn filter_primitive<T>(array: &PrimitiveArray<T>) -> Option<ArrayRef>
    where
        T: ArrowPrimitiveType,
    {
        let values: Vec<T::Native> = array.iter().flatten().collect();
        if values.is_empty() {
            None
        } else {
            Some(Arc::new(PrimitiveArray::<T>::from_iter_values(values)) as ArrayRef)
        }
    }

    match arr.data_type() {
        DataType::UInt64 => {
            let prim = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<arrow::datatypes::UInt64Type>>()
                .unwrap();
            filter_primitive(prim)
        }
        DataType::UInt32 => {
            let prim = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<arrow::datatypes::UInt32Type>>()
                .unwrap();
            filter_primitive(prim)
        }
        DataType::UInt16 => {
            let prim = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<arrow::datatypes::UInt16Type>>()
                .unwrap();
            filter_primitive(prim)
        }
        DataType::UInt8 => {
            let prim = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<arrow::datatypes::UInt8Type>>()
                .unwrap();
            filter_primitive(prim)
        }
        DataType::Int64 => {
            let prim = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<arrow::datatypes::Int64Type>>()
                .unwrap();
            filter_primitive(prim)
        }
        DataType::Int32 => {
            let prim = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<arrow::datatypes::Int32Type>>()
                .unwrap();
            filter_primitive(prim)
        }
        DataType::Int16 => {
            let prim = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<arrow::datatypes::Int16Type>>()
                .unwrap();
            filter_primitive(prim)
        }
        DataType::Int8 => {
            let prim = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<arrow::datatypes::Int8Type>>()
                .unwrap();
            filter_primitive(prim)
        }
        DataType::Float64 => {
            let prim = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<arrow::datatypes::Float64Type>>()
                .unwrap();
            filter_primitive(prim)
        }
        DataType::Float32 => {
            let prim = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<arrow::datatypes::Float32Type>>()
                .unwrap();
            filter_primitive(prim)
        }
        _ => Some(arr),
    }
}

pub struct ColumnData {
    pub data: ArrayRef,
    pub data_type: DataType,
}

// FIX: Move helper methods into a trait to satisfy Rust's orphan rule
pub trait ColumnStoreTestExt {
    fn get_column_for_test(&self, field_id: LogicalFieldId) -> Result<ColumnData, TableError>;
}

impl ColumnStoreTestExt for ColumnStore<MemPager> {
    fn get_column_for_test(&self, field_id: LogicalFieldId) -> Result<ColumnData, TableError> {
        struct ColVisitor {
            chunks: Vec<ArrayRef>,
            data_type: Option<DataType>,
        }
        impl scan::PrimitiveVisitor for ColVisitor {
            fn u64_chunk(&mut self, a: &UInt64Array) {
                self.data_type.get_or_insert(DataType::UInt64);
                self.chunks.push(Arc::new(a.clone()));
            }
            fn i32_chunk(&mut self, a: &Int32Array) {
                self.data_type.get_or_insert(DataType::Int32);
                self.chunks.push(Arc::new(a.clone()));
            }
            fn f64_chunk(&mut self, a: &arrow::array::Float64Array) {
                self.data_type.get_or_insert(DataType::Float64);
                self.chunks.push(Arc::new(a.clone()));
            }
            fn f32_chunk(&mut self, a: &arrow::array::Float32Array) {
                self.data_type.get_or_insert(DataType::Float32);
                self.chunks.push(Arc::new(a.clone()));
            }
            // FIX: The trait `PrimitiveVisitor` does not have a `binary_chunk`
            // method. This must be handled by downcasting a generic
            // `array_chunk` if needed, or by adding the method to the trait in
            // `column-map`. For now, we remove it to allow compilation.
            // fn binary_chunk(&mut self, a: &arrow::array::BinaryArray) {
            //     self.data_type.get_or_insert(DataType::Binary);
            //     self.chunks.push(Arc::new(a.clone()));
            // }
        }
        impl scan::PrimitiveWithRowIdsVisitor for ColVisitor {}
        impl scan::PrimitiveSortedVisitor for ColVisitor {}
        impl scan::PrimitiveSortedWithRowIdsVisitor for ColVisitor {}

        let mut visitor = ColVisitor {
            chunks: vec![],
            data_type: None,
        };
        self.scan(field_id, Default::default(), &mut visitor)?;

        if visitor.chunks.is_empty() {
            return Err(TableError::Internal(
                "Column not found or is empty".to_string(),
            ));
        }

        // NOTE: This helper still concatenates for tests. Production code
        // uses streaming via `scan_stream`.
        let refs: Vec<&dyn Array> = visitor.chunks.iter().map(|c| c.as_ref()).collect();
        let combined = arrow::compute::concat(&refs)?;

        Ok(ColumnData {
            data: combined,
            data_type: visitor.data_type.unwrap(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sys_catalog::CATALOG_TID;
    use crate::types::RowId;
    use arrow::array::{
        BinaryArray, Float32Array, Float64Array, Int16Array, Int32Array, UInt8Array, UInt32Array,
        UInt64Array,
    };
    use arrow::compute::{cast, max, min, sum, unary};
    use std::collections::HashMap;

    fn setup_test_table() -> Table {
        let pager = Arc::new(MemPager::default());
        setup_test_table_with_pager(&pager)
    }

    fn setup_test_table_with_pager(pager: &Arc<MemPager>) -> Table {
        let table = Table::new(1, Arc::clone(pager)).unwrap();
        const COL_A_U64: FieldId = 10;
        const COL_B_BIN: FieldId = 11;
        const COL_C_I32: FieldId = 12;
        const COL_D_F64: FieldId = 13;
        const COL_E_F32: FieldId = 14;

        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("a_u64", DataType::UInt64, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_A_U64.to_string(),
            )])),
            Field::new("b_bin", DataType::Binary, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_B_BIN.to_string(),
            )])),
            Field::new("c_i32", DataType::Int32, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_C_I32.to_string(),
            )])),
            Field::new("d_f64", DataType::Float64, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_D_F64.to_string(),
            )])),
            Field::new("e_f32", DataType::Float32, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
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

    /// Helper to collect a streamed column into a Vec<Option<T>> using a
    /// provided extractor function. Avoids concat in tests while keeping
    /// assertions simple.
    fn collect_streamed_values<T, FExtract>(
        table: &Table,
        proj: FieldId,
        filter: &Filter<'_, FieldId>,
        mut extract: FExtract,
    ) -> Vec<Option<T>>
    where
        T: Copy,
        FExtract: FnMut(&RecordBatch) -> Vec<Option<T>>,
    {
        let mut out = Vec::<Option<T>>::new();
        table
            .scan_stream(proj, filter, |b| {
                out.extend(extract(&b));
            })
            .unwrap();
        out
    }

    fn collect_streamed_values_with_options<T, FExtract>(
        table: &Table,
        proj: FieldId,
        filter: &Filter<'_, FieldId>,
        options: ScanStreamOptions,
        mut extract: FExtract,
    ) -> Vec<Option<T>>
    where
        T: Copy,
        FExtract: FnMut(&RecordBatch) -> Vec<Option<T>>,
    {
        let mut out = Vec::<Option<T>>::new();
        table
            .scan_stream_with_options(proj, filter, options, |b| {
                out.extend(extract(&b));
            })
            .unwrap();
        out
    }

    #[test]
    fn table_new_rejects_reserved_table_id() {
        let result = Table::new(CATALOG_TID, Arc::new(MemPager::default()));
        assert!(matches!(
            result,
            Err(TableError::ReservedTableId(id)) if id == CATALOG_TID
        ));
    }

    #[test]
    fn test_scan_with_u64_filter() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = Filter {
            field_id: COL_A_U64,
            op: Operator::Equals(200.into()),
        };

        let vals = collect_streamed_values::<i32, _>(&table, COL_C_I32, &filter, |b| {
            let a = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            (0..a.len())
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect::<Vec<_>>()
        });
        assert_eq!(vals, vec![Some(20), Some(20)]);
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

        let beta_rows: Vec<u64> = vec![101, 102, 103];
        let beta_vals_u64: Vec<u64> = vec![900, 901, 902];
        let beta_vals_u8: Vec<u8> = vec![1, 2, 3];

        let gamma_rows: Vec<u64> = vec![501, 502];
        let gamma_vals_i16: Vec<i16> = vec![123, -321];

        // First session: create tables and write data.
        {
            let table = Table::new(TABLE_ALPHA, Arc::clone(&pager)).unwrap();
            let schema =
                Arc::new(Schema::new(vec![
                    Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
                    Field::new("alpha_u64", DataType::UInt64, false).with_metadata(HashMap::from(
                        [("field_id".to_string(), COL_ALPHA_U64.to_string())],
                    )),
                    Field::new("alpha_i32", DataType::Int32, false).with_metadata(HashMap::from([
                        ("field_id".to_string(), COL_ALPHA_I32.to_string()),
                    ])),
                    Field::new("alpha_u32", DataType::UInt32, false).with_metadata(HashMap::from(
                        [("field_id".to_string(), COL_ALPHA_U32.to_string())],
                    )),
                    Field::new("alpha_i16", DataType::Int16, false).with_metadata(HashMap::from([
                        ("field_id".to_string(), COL_ALPHA_I16.to_string()),
                    ])),
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
            let table = Table::new(TABLE_BETA, Arc::clone(&pager)).unwrap();
            let schema = Arc::new(Schema::new(vec![
                Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
                Field::new("beta_u64", DataType::UInt64, false).with_metadata(HashMap::from([(
                    "field_id".to_string(),
                    COL_BETA_U64.to_string(),
                )])),
                Field::new("beta_u8", DataType::UInt8, false).with_metadata(HashMap::from([(
                    "field_id".to_string(),
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
            let table = Table::new(TABLE_GAMMA, Arc::clone(&pager)).unwrap();
            let schema = Arc::new(Schema::new(vec![
                Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
                Field::new("gamma_i16", DataType::Int16, false).with_metadata(HashMap::from([(
                    "field_id".to_string(),
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
            let table = Table::new(TABLE_ALPHA, Arc::clone(&pager)).unwrap();
            let store = table.store();

            let expectations: &[(FieldId, DataType)] = &[
                (COL_ALPHA_U64, DataType::UInt64),
                (COL_ALPHA_I32, DataType::Int32),
                (COL_ALPHA_U32, DataType::UInt32),
                (COL_ALPHA_I16, DataType::Int16),
            ];

            for &(col, ref ty) in expectations {
                let lfid = lfid_for(TABLE_ALPHA, col);
                assert_eq!(store.data_type(lfid).unwrap(), *ty);
                let arr = store.gather_rows(lfid, &alpha_rows).unwrap();
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
            let table = Table::new(TABLE_BETA, Arc::clone(&pager)).unwrap();
            let store = table.store();

            let lfid_u64 = lfid_for(TABLE_BETA, COL_BETA_U64);
            assert_eq!(store.data_type(lfid_u64).unwrap(), DataType::UInt64);
            let arr_u64 = store.gather_rows(lfid_u64, &beta_rows).unwrap();
            let arr_u64 = arr_u64.as_any().downcast_ref::<UInt64Array>().unwrap();
            assert_eq!(arr_u64.values(), beta_vals_u64.as_slice());

            let lfid_u8 = lfid_for(TABLE_BETA, COL_BETA_U8);
            assert_eq!(store.data_type(lfid_u8).unwrap(), DataType::UInt8);
            let arr_u8 = store.gather_rows(lfid_u8, &beta_rows).unwrap();
            let arr_u8 = arr_u8.as_any().downcast_ref::<UInt8Array>().unwrap();
            assert_eq!(arr_u8.values(), beta_vals_u8.as_slice());
        }

        {
            let table = Table::new(TABLE_GAMMA, Arc::clone(&pager)).unwrap();
            let store = table.store();
            let lfid = lfid_for(TABLE_GAMMA, COL_GAMMA_I16);
            assert_eq!(store.data_type(lfid).unwrap(), DataType::Int16);
            let arr = store.gather_rows(lfid, &gamma_rows).unwrap();
            let arr = arr.as_any().downcast_ref::<Int16Array>().unwrap();
            assert_eq!(arr.values(), gamma_vals_i16.as_slice());
        }
    }

    #[test]
    fn test_scan_with_i32_filter() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = Filter {
            field_id: COL_C_I32,
            op: Operator::Equals(20.into()),
        };

        let vals = collect_streamed_values::<u64, _>(&table, COL_A_U64, &filter, |b| {
            let a = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
            (0..a.len())
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect::<Vec<_>>()
        });
        assert_eq!(vals, vec![Some(200), Some(200)]);
    }

    #[test]
    fn test_scan_with_greater_than_filter() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = Filter {
            field_id: COL_C_I32,
            op: Operator::GreaterThan(15.into()),
        };

        let vals = collect_streamed_values::<u64, _>(&table, COL_A_U64, &filter, |b| {
            let a = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
            (0..a.len())
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect::<Vec<_>>()
        });
        assert_eq!(vals, vec![Some(200), Some(300), Some(200)]);
    }

    #[test]
    fn test_scan_with_range_filter() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = Filter {
            field_id: COL_A_U64,
            op: Operator::Range {
                lower: Bound::Included(150.into()),
                upper: Bound::Excluded(300.into()),
            },
        };

        let vals = collect_streamed_values::<i32, _>(&table, COL_C_I32, &filter, |b| {
            let a = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            (0..a.len())
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect::<Vec<_>>()
        });
        assert_eq!(vals, vec![Some(20), Some(20)]);
    }

    #[test]
    fn test_filtered_scan_sum_kernel() {
        // Trade-off note:
        // - We use Arrow's sum kernel per batch, then add the partial sums.
        // - This preserves Arrow null semantics and avoids concat.
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;

        let filter = Filter {
            field_id: COL_A_U64,
            op: Operator::Range {
                lower: Bound::Included(150.into()),
                upper: Bound::Excluded(300.into()),
            },
        };

        let mut total: u128 = 0;
        table
            .scan_stream(COL_A_U64, &filter, |b| {
                let a = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
                if let Some(part) = sum(a) {
                    total += part as u128;
                }
            })
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
        let filter = Filter {
            field_id: COL_A_U64,
            op: Operator::In(&candidates),
        };

        let mut total: i64 = 0;
        table
            .scan_stream(COL_C_I32, &filter, |b| {
                let a = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                if let Some(part) = sum(a) {
                    total += part as i64;
                }
            })
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
        let filter = Filter {
            field_id: COL_A_U64,
            op: Operator::In(&candidates),
        };

        let mut mn: Option<i32> = None;
        let mut mx: Option<i32> = None;
        table
            .scan_stream(COL_C_I32, &filter, |b| {
                let a = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

                if let Some(part_min) = min(a) {
                    mn = Some(mn.map_or(part_min, |m| m.min(part_min)));
                }
                if let Some(part_max) = max(a) {
                    mx = Some(mx.map_or(part_max, |m| m.max(part_max)));
                }
            })
            .unwrap();
        assert_eq!(mn, Some(10));
        assert_eq!(mx, Some(30));
    }

    #[test]
    fn test_filtered_scan_float64_column() {
        let table = setup_test_table();
        const COL_D_F64: FieldId = 13;

        let filter = Filter {
            field_id: COL_D_F64,
            op: Operator::GreaterThan(2.0_f64.into()),
        };

        let mut got = Vec::new();
        table
            .scan_stream(COL_D_F64, &filter, |b| {
                let arr = b.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        got.push(arr.value(i));
                    }
                }
            })
            .unwrap();

        assert_eq!(got, vec![2.5, 3.5, 2.5]);
    }

    #[test]
    fn test_filtered_scan_float32_in_operator() {
        let table = setup_test_table();
        const COL_E_F32: FieldId = 14;

        let candidates = [2.0_f32.into(), 3.0_f32.into()];
        let filter = Filter {
            field_id: COL_E_F32,
            op: Operator::In(&candidates),
        };

        let vals = collect_streamed_values::<f32, _>(&table, COL_E_F32, &filter, |b| {
            let arr = b.column(0).as_any().downcast_ref::<Float32Array>().unwrap();
            (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i))
                    }
                })
                .collect()
        });

        let collected: Vec<f32> = vals.into_iter().flatten().collect();
        assert_eq!(collected, vec![2.0, 3.0, 2.0]);
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
                "field_id".to_string(),
                COL_A_U64.to_string(),
            )])),
            Field::new("b_bin", DataType::Binary, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_B_BIN.to_string(),
            )])),
            Field::new("c_i32", DataType::Int32, true).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_C_I32.to_string(),
            )])),
            Field::new("d_f64", DataType::Float64, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_D_F64.to_string(),
            )])),
            Field::new("e_f32", DataType::Float32, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_E_F32.to_string(),
            )])),
        ]));

        let batch = RecordBatch::try_new(
            schema,
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

        let filter = Filter {
            field_id: COL_A_U64,
            op: Operator::GreaterThan(450.into()),
        };

        let default_vals = collect_streamed_values::<i32, _>(&table, COL_C_I32, &filter, |b| {
            let arr = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i))
                    }
                })
                .collect::<Vec<_>>()
        });
        assert_eq!(default_vals, vec![Some(40)]);

        let include_null_vals = collect_streamed_values_with_options::<i32, _>(
            &table,
            COL_C_I32,
            &filter,
            ScanStreamOptions {
                include_nulls: true,
            },
            |b| {
                let arr = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            None
                        } else {
                            Some(arr.value(i))
                        }
                    })
                    .collect::<Vec<_>>()
            },
        );
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

        let filter = Filter {
            field_id: COL_C_I32,
            op: Operator::GreaterThan(15.into()),
        };

        let mut got: Vec<f64> = Vec::new();
        table
            .scan_stream(COL_A_U64, &filter, |b| {
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
            })
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

        let table = Table::new(2, Arc::new(MemPager::default())).unwrap();

        const COL_A_U64: FieldId = 20;
        const COL_D_U32: FieldId = 21;
        const COL_E_I16: FieldId = 22;
        const COL_F_U8: FieldId = 23;
        const COL_C_I32: FieldId = 24;

        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("a_u64", DataType::UInt64, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_A_U64.to_string(),
            )])),
            Field::new("d_u32", DataType::UInt32, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_D_U32.to_string(),
            )])),
            Field::new("e_i16", DataType::Int16, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_E_I16.to_string(),
            )])),
            Field::new("f_u8", DataType::UInt8, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                COL_F_U8.to_string(),
            )])),
            Field::new("c_i32", DataType::Int32, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
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
        let filter = Filter {
            field_id: COL_C_I32,
            op: Operator::GreaterThanOrEquals(20.into()),
        };

        // 1) SUM over d_u32 (per-batch sum + accumulate).
        let mut d_sum: u128 = 0;
        table
            .scan_stream(COL_D_U32, &filter, |b| {
                let a = b.column(0).as_any().downcast_ref::<UInt32Array>().unwrap();
                if let Some(part) = sum(a) {
                    d_sum += part as u128;
                }
            })
            .unwrap();
        assert_eq!(d_sum, (8 + 9 + 10 + 11) as u128);

        // 2) MIN over e_i16 (per-batch min + fold).
        let mut e_min: Option<i16> = None;
        table
            .scan_stream(COL_E_I16, &filter, |b| {
                let a = b.column(0).as_any().downcast_ref::<Int16Array>().unwrap();
                if let Some(part_min) = min(a) {
                    e_min = Some(e_min.map_or(part_min, |m| m.min(part_min)));
                }
            })
            .unwrap();
        assert_eq!(e_min, Some(-6));

        // 3) MAX over f_u8 (per-batch max + fold).
        let mut f_max: Option<u8> = None;
        table
            .scan_stream(COL_F_U8, &filter, |b| {
                let a = b
                    .column(0)
                    .as_any()
                    .downcast_ref::<arrow::array::UInt8Array>()
                    .unwrap();
                if let Some(part_max) = max(a) {
                    f_max = Some(f_max.map_or(part_max, |m| m.max(part_max)));
                }
            })
            .unwrap();
        assert_eq!(f_max, Some(10));

        // 4) SQRT over a_u64 (cast to f64, then unary sqrt per batch).
        let mut got: Vec<f64> = Vec::new();
        table
            .scan_stream(COL_A_U64, &filter, |b| {
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
            })
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
        let filter = Filter {
            field_id: COL_C_I32,
            op: Operator::In(&candidates),
        };

        let vals = collect_streamed_values::<u64, _>(&table, COL_A_U64, &filter, |b| {
            let a = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
            (0..a.len())
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect::<Vec<_>>()
        });
        assert_eq!(vals, vec![Some(100), Some(300)]);
    }

    #[test]
    fn test_scan_stream_single_column_batches() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        // Filter c_i32 == 20 -> two rows; stream a_u64 in batches of <= N.
        let filter = Filter {
            field_id: COL_C_I32,
            op: Operator::Equals(20.into()),
        };

        let mut seen = Vec::<u64>::new();
        table
            .scan_stream(COL_A_U64, &filter, |b| {
                assert_eq!(b.num_columns(), 1);
                let a = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
                // No kernel needed; just collect values for shape assertions.
                for i in 0..a.len() {
                    if !a.is_null(i) {
                        seen.push(a.value(i));
                    }
                }
            })
            .unwrap();

        // In fixture, c_i32 == 20 corresponds to a_u64 values [200, 200].
        assert_eq!(seen, vec![200, 200]);
    }

    #[test]
    fn test_scan_with_multiple_projection_columns() {
        // With streaming, scan each column independently using the same
        // filter and compare results. This keeps the core API streaming-
        // friendly without multi-column chunk alignment.
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = Filter {
            field_id: COL_C_I32,
            op: Operator::Equals(20.into()),
        };

        let a_vals = collect_streamed_values::<u64, _>(&table, COL_A_U64, &filter, |b| {
            let a = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
            (0..a.len())
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect::<Vec<_>>()
        });
        assert_eq!(a_vals, vec![Some(200), Some(200)]);

        let c_vals = collect_streamed_values::<i32, _>(&table, COL_C_I32, &filter, |b| {
            let a = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            (0..a.len())
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect::<Vec<_>>()
        });
        assert_eq!(c_vals, vec![Some(20), Some(20)]);
    }
}
