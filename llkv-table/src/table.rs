use std::ops::Bound;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int32Array, PrimitiveArray, RecordBatch, UInt64Array};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Int32Type, Schema, UInt64Type};
use arrow_array::{Datum, Scalar};

use llkv_column_map::store::rowid_fid;
use llkv_column_map::{
    ColumnStore,
    scan::{self, ScanBuilder},
    storage::pager::MemPager,
    types::{LogicalFieldId, Namespace},
};

use crate::expr::{Filter, Operator};
use crate::sys_catalog::{ColMeta, SysCatalog, TableMeta};
use crate::types::FieldId;

type DynScalar = Scalar<Arc<dyn arrow_array::Array>>;

fn scalar_data_type(s: &DynScalar) -> DataType {
    let (array, _) = s.get();
    array.data_type().clone()
}

fn scalar_to_native<T: ArrowPrimitiveType>(scalar: &DynScalar) -> Result<T::Native, CmError> {
    let (array, _) = scalar.get();
    let typed = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .ok_or_else(|| {
            CmError::Internal(format!(
                "Mismatched type for filter value. Expected {}",
                std::any::type_name::<T::Native>()
            ))
        })?;
    if typed.is_null(0) {
        return Err(CmError::Internal("Filter value is null".to_string()));
    }
    Ok(typed.value(0))
}

fn bound_to_native<T: ArrowPrimitiveType>(
    bound: &Bound<DynScalar>,
) -> Result<Bound<T::Native>, CmError>
where
    T::Native: Copy,
{
    Ok(match bound {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(scalar) => Bound::Included(scalar_to_native::<T>(scalar)?),
        Bound::Excluded(scalar) => Bound::Excluded(scalar_to_native::<T>(scalar)?),
    })
}

fn scalar_bound_data_type(bound: &Bound<DynScalar>) -> Option<DataType> {
    match bound {
        Bound::Unbounded => None,
        Bound::Included(s) | Bound::Excluded(s) => Some(scalar_data_type(s)),
    }
}

fn filter_data_type(op: &Operator<'_>) -> Result<DataType, CmError> {
    match op {
        Operator::Equals(s)
        | Operator::GreaterThan(s)
        | Operator::LessThan(s)
        | Operator::GreaterThanOrEquals(s)
        | Operator::LessThanOrEquals(s) => Ok(scalar_data_type(s)),
        Operator::Range { lower, upper } => {
            let lower_ty = scalar_bound_data_type(lower);
            let upper_ty = scalar_bound_data_type(upper);
            match (lower_ty, upper_ty) {
                (None, None) => Err(CmError::Internal(
                    "Range filter requires at least one concrete bound".to_string(),
                )),
                (Some(dt), None) | (None, Some(dt)) => Ok(dt),
                (Some(l), Some(u)) if l == u => Ok(l),
                (Some(l), Some(u)) => Err(CmError::Internal(format!(
                    "Mismatched range bound types: {:?} vs {:?}",
                    l, u
                ))),
            }
        }
        Operator::In(values) => {
            let first = values.first().ok_or_else(|| {
                CmError::Internal("IN operator requires at least one value".to_string())
            })?;
            let dtype = scalar_data_type(first);
            let all_match = values.iter().all(|value| scalar_data_type(value) == dtype);
            if all_match {
                Ok(dtype)
            } else {
                Err(CmError::Internal(
                    "IN operator values must share the same type".to_string(),
                ))
            }
        }
        _ => Err(CmError::Internal(
            "Filter operator does not contain a typed value".to_string(),
        )),
    }
}

enum RangeLimit<T> {
    Included(T),
    Excluded(T),
}

enum Predicate<T>
where
    T: ArrowPrimitiveType,
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
    T: ArrowPrimitiveType,
{
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

fn build_predicate<T>(op: &Operator<'_>) -> Result<Predicate<T>, CmError>
where
    T: ArrowPrimitiveType,
{
    match op {
        Operator::Equals(scalar) => Ok(Predicate::Equals(scalar_to_native::<T>(scalar)?)),
        Operator::GreaterThan(scalar) => Ok(Predicate::GreaterThan(scalar_to_native::<T>(scalar)?)),
        Operator::GreaterThanOrEquals(scalar) => Ok(Predicate::GreaterThanOrEquals(
            scalar_to_native::<T>(scalar)?,
        )),
        Operator::LessThan(scalar) => Ok(Predicate::LessThan(scalar_to_native::<T>(scalar)?)),
        Operator::LessThanOrEquals(scalar) => {
            Ok(Predicate::LessThanOrEquals(scalar_to_native::<T>(scalar)?))
        }
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
            for scalar in *values {
                natives.push(scalar_to_native::<T>(scalar)?);
            }
            Ok(Predicate::In(natives))
        }
        _ => Err(CmError::Internal(
            "Filter operator does not contain a supported typed value".to_string(),
        )),
    }
}

struct FilterRowCollector<T>
where
    T: ArrowPrimitiveType,
{
    predicate: Predicate<T>,
    row_ids: Vec<u64>,
}

impl<T> FilterRowCollector<T>
where
    T: ArrowPrimitiveType,
{
    fn new(predicate: Predicate<T>) -> Self {
        Self {
            predicate,
            row_ids: Vec::new(),
        }
    }

    fn into_row_ids(self) -> Vec<u64> {
        self.row_ids
    }
}

impl<T> scan::PrimitiveVisitor for FilterRowCollector<T>
where
    T: ArrowPrimitiveType,
{
    fn u64_chunk(&mut self, _: &UInt64Array) {}
    fn u32_chunk(&mut self, _: &arrow::array::UInt32Array) {}
    fn u16_chunk(&mut self, _: &arrow::array::UInt16Array) {}
    fn u8_chunk(&mut self, _: &arrow::array::UInt8Array) {}
    fn i64_chunk(&mut self, _: &arrow::array::Int64Array) {}
    fn i32_chunk(&mut self, _: &Int32Array) {}
    fn i16_chunk(&mut self, _: &arrow::array::Int16Array) {}
    fn i8_chunk(&mut self, _: &arrow::array::Int8Array) {}
}

impl<T> scan::PrimitiveSortedVisitor for FilterRowCollector<T>
where
    T: ArrowPrimitiveType,
{
    fn u64_run(&mut self, _: &UInt64Array, _: usize, _: usize) {}
    fn u32_run(&mut self, _: &arrow::array::UInt32Array, _: usize, _: usize) {}
    fn u16_run(&mut self, _: &arrow::array::UInt16Array, _: usize, _: usize) {}
    fn u8_run(&mut self, _: &arrow::array::UInt8Array, _: usize, _: usize) {}
    fn i64_run(&mut self, _: &arrow::array::Int64Array, _: usize, _: usize) {}
    fn i32_run(&mut self, _: &Int32Array, _: usize, _: usize) {}
    fn i16_run(&mut self, _: &arrow::array::Int16Array, _: usize, _: usize) {}
    fn i8_run(&mut self, _: &arrow::array::Int8Array, _: usize, _: usize) {}
}

impl<T> scan::PrimitiveSortedWithRowIdsVisitor for FilterRowCollector<T> where T: ArrowPrimitiveType {}

macro_rules! impl_filter_with_rids {
    ($ty:ty, $method:ident, $arr:ty) => {
        impl scan::PrimitiveWithRowIdsVisitor for FilterRowCollector<$ty> {
            fn $method(&mut self, v: &$arr, r: &UInt64Array) {
                let len = v.len();
                debug_assert_eq!(len, r.len());
                for i in 0..len {
                    if v.is_null(i) {
                        continue;
                    }
                    let value = v.value(i);
                    if self.predicate.matches(value) {
                        self.row_ids.push(r.value(i));
                    }
                }
            }
        }
    };
}

impl_filter_with_rids!(UInt64Type, u64_chunk_with_rids, UInt64Array);
impl_filter_with_rids!(
    arrow::datatypes::UInt32Type,
    u32_chunk_with_rids,
    arrow::array::UInt32Array
);
impl_filter_with_rids!(
    arrow::datatypes::UInt16Type,
    u16_chunk_with_rids,
    arrow::array::UInt16Array
);
impl_filter_with_rids!(
    arrow::datatypes::UInt8Type,
    u8_chunk_with_rids,
    arrow::array::UInt8Array
);
impl_filter_with_rids!(
    arrow::datatypes::Int64Type,
    i64_chunk_with_rids,
    arrow::array::Int64Array
);
impl_filter_with_rids!(Int32Type, i32_chunk_with_rids, Int32Array);
impl_filter_with_rids!(
    arrow::datatypes::Int16Type,
    i16_chunk_with_rids,
    arrow::array::Int16Array
);
impl_filter_with_rids!(
    arrow::datatypes::Int8Type,
    i8_chunk_with_rids,
    arrow::array::Int8Array
);

fn collect_matching_row_ids<T>(
    store: &ColumnStore<MemPager>,
    field_id: LogicalFieldId,
    op: &Operator<'_>,
) -> Result<Vec<u64>, CmError>
where
    T: ArrowPrimitiveType,
    FilterRowCollector<T>: scan::PrimitiveWithRowIdsVisitor,
{
    let predicate = build_predicate::<T>(op)?;
    let mut collector = FilterRowCollector::<T>::new(predicate);
    ScanBuilder::new(store, field_id)
        .with_row_ids(rowid_fid(field_id))
        .run(&mut collector)?;
    Ok(collector.into_row_ids())
}

#[inline]
fn lfid_for(table_id: u32, column_id: FieldId) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_table_id(table_id)
        .with_field_id(column_id)
        .with_namespace(Namespace::UserData)
}

// TODO: Remove?
#[derive(Clone, Debug, Default)]
pub struct TableCfg {}

#[derive(Debug)]
pub enum CmError {
    ColumnMap(llkv_column_map::error::Error),
    Arrow(arrow::error::ArrowError),
    Internal(String),
}

impl From<llkv_column_map::error::Error> for CmError {
    fn from(e: llkv_column_map::error::Error) -> Self {
        CmError::ColumnMap(e)
    }
}

impl From<arrow::error::ArrowError> for CmError {
    fn from(e: arrow::error::ArrowError) -> Self {
        CmError::Arrow(e)
    }
}

pub struct Table {
    store: ColumnStore<MemPager>,
    #[allow(dead_code)]
    cfg: TableCfg,
    table_id: u32,
}

impl Table {
    pub fn new(table_id: u32, cfg: TableCfg) -> Self {
        let pager = Arc::new(MemPager::default());
        let store = ColumnStore::open(pager).unwrap();
        Self {
            store,
            cfg,
            table_id,
        }
    }

    pub fn append(&self, batch: &RecordBatch) -> Result<(), llkv_column_map::error::Error> {
        let mut new_fields = Vec::with_capacity(batch.schema().fields().len());
        for field in batch.schema().fields() {
            if field.name() == "row_id" {
                new_fields.push(field.as_ref().clone());
                continue;
            }

            let user_field_id: FieldId = field
                .metadata()
                .get("field_id")
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| {
                    llkv_column_map::error::Error::Internal(format!(
                        "Field '{}' is missing a valid 'field_id' in its metadata.",
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

    pub fn scan<'a>(
        &self,
        projection: &[FieldId],
        filter: &Filter<'a, FieldId>,
    ) -> Result<RecordBatch, CmError> {
        let filter_lfid = lfid_for(self.table_id, filter.field_id);

        let dtype = filter_data_type(&filter.op)?;

        let row_ids = match dtype {
            DataType::UInt64 => {
                collect_matching_row_ids::<UInt64Type>(&self.store, filter_lfid, &filter.op)?
            }
            DataType::UInt32 => collect_matching_row_ids::<arrow::datatypes::UInt32Type>(
                &self.store,
                filter_lfid,
                &filter.op,
            )?,
            DataType::UInt16 => collect_matching_row_ids::<arrow::datatypes::UInt16Type>(
                &self.store,
                filter_lfid,
                &filter.op,
            )?,
            DataType::UInt8 => collect_matching_row_ids::<arrow::datatypes::UInt8Type>(
                &self.store,
                filter_lfid,
                &filter.op,
            )?,
            DataType::Int64 => collect_matching_row_ids::<arrow::datatypes::Int64Type>(
                &self.store,
                filter_lfid,
                &filter.op,
            )?,
            DataType::Int32 => {
                collect_matching_row_ids::<Int32Type>(&self.store, filter_lfid, &filter.op)?
            }
            DataType::Int16 => collect_matching_row_ids::<arrow::datatypes::Int16Type>(
                &self.store,
                filter_lfid,
                &filter.op,
            )?,
            DataType::Int8 => collect_matching_row_ids::<arrow::datatypes::Int8Type>(
                &self.store,
                filter_lfid,
                &filter.op,
            )?,
            other => {
                return Err(CmError::Internal(format!(
                    "Filtering on type {:?} is not supported",
                    other
                )));
            }
        };

        let mut projected_columns: Vec<ArrayRef> = Vec::with_capacity(projection.len());
        let mut projected_fields: Vec<Field> = Vec::with_capacity(projection.len());

        for &field_id_to_project in projection {
            let lfid_to_project = lfid_for(self.table_id, field_id_to_project);
            let dtype = self.store.data_type(lfid_to_project)?;
            let column = self.store.gather_rows(lfid_to_project, &row_ids)?;

            projected_fields.push(Field::new(field_id_to_project.to_string(), dtype, true));
            projected_columns.push(column);
        }

        let schema = Arc::new(Schema::new(projected_fields));
        Ok(RecordBatch::try_new(schema, projected_columns)?)
    }

    #[inline]
    pub fn catalog(&self) -> SysCatalog<'_> {
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
    pub fn get_cols_meta(&self, col_ids: &[u32]) -> Vec<Option<ColMeta>> {
        self.catalog().get_cols_meta(self.table_id, col_ids)
    }

    pub fn store(&self) -> &ColumnStore<MemPager> {
        &self.store
    }
}

pub struct ColumnData {
    pub data: ArrayRef,
    pub data_type: DataType,
}

// FIX: Move helper methods into a trait to satisfy Rust's orphan rule
pub trait ColumnStoreTestExt {
    fn get_column_for_test(&self, field_id: LogicalFieldId) -> Result<ColumnData, CmError>;
}

impl ColumnStoreTestExt for ColumnStore<MemPager> {
    fn get_column_for_test(&self, field_id: LogicalFieldId) -> Result<ColumnData, CmError> {
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
            // FIX: The trait `PrimitiveVisitor` does not have a `binary_chunk` method.
            // This must be handled by downcasting a generic `array_chunk` if needed,
            // or by adding the method to the trait in `column-map`. For now, we remove it
            // to allow compilation.
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
            return Err(CmError::Internal(
                "Column not found or is empty".to_string(),
            ));
        }

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
    use arrow::array::{BinaryArray, Int32Array, UInt64Array};
    use std::collections::HashMap;

    fn scalar_from_u64(value: u64) -> DynScalar {
        Scalar::new(Arc::new(UInt64Array::from(vec![value])) as ArrayRef)
    }

    fn scalar_from_i32(value: i32) -> DynScalar {
        Scalar::new(Arc::new(Int32Array::from(vec![value])) as ArrayRef)
    }

    fn setup_test_table() -> Table {
        let table = Table::new(1, TableCfg::default());
        const COL_A_U64: FieldId = 10;
        const COL_B_BIN: FieldId = 11;
        const COL_C_I32: FieldId = 12;

        let schema = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
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
            ],
        )
        .unwrap();

        table.append(&batch).unwrap();
        table
    }

    #[test]
    fn test_scan_with_u64_filter() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = Filter {
            field_id: COL_A_U64,
            op: Operator::Equals(scalar_from_u64(200)),
        };
        let projection = vec![COL_C_I32];
        let result_batch = table.scan(&projection, &filter).unwrap();

        assert_eq!(result_batch.num_columns(), 1);
        let values = result_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let collected: Vec<Option<i32>> = (0..values.len())
            .map(|i| {
                if values.is_null(i) {
                    None
                } else {
                    Some(values.value(i))
                }
            })
            .collect();
        assert_eq!(collected, vec![Some(20), Some(20)]);
    }

    #[test]
    fn test_scan_with_i32_filter() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = Filter {
            field_id: COL_C_I32,
            op: Operator::Equals(scalar_from_i32(20)),
        };
        let projection = vec![COL_A_U64];
        let result_batch = table.scan(&projection, &filter).unwrap();

        assert_eq!(result_batch.num_columns(), 1);
        let values = result_batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let collected: Vec<Option<u64>> = (0..values.len())
            .map(|i| {
                if values.is_null(i) {
                    None
                } else {
                    Some(values.value(i))
                }
            })
            .collect();
        assert_eq!(collected, vec![Some(200), Some(200)]);
    }

    #[test]
    fn test_scan_with_greater_than_filter() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = Filter {
            field_id: COL_C_I32,
            op: Operator::GreaterThan(scalar_from_i32(15)),
        };
        let projection = vec![COL_A_U64];
        let result_batch = table.scan(&projection, &filter).unwrap();

        assert_eq!(result_batch.num_columns(), 1);
        let values = result_batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let collected: Vec<Option<u64>> = (0..values.len())
            .map(|i| {
                if values.is_null(i) {
                    None
                } else {
                    Some(values.value(i))
                }
            })
            .collect();
        assert_eq!(collected, vec![Some(200), Some(300), Some(200)]);
    }

    #[test]
    fn test_scan_with_range_filter() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = Filter {
            field_id: COL_A_U64,
            op: Operator::Range {
                lower: Bound::Included(scalar_from_u64(150)),
                upper: Bound::Excluded(scalar_from_u64(300)),
            },
        };
        let projection = vec![COL_C_I32];
        let result_batch = table.scan(&projection, &filter).unwrap();

        assert_eq!(result_batch.num_columns(), 1);
        let values = result_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let collected: Vec<Option<i32>> = (0..values.len())
            .map(|i| {
                if values.is_null(i) {
                    None
                } else {
                    Some(values.value(i))
                }
            })
            .collect();
        assert_eq!(collected, vec![Some(20), Some(20)]);
    }

    #[test]
    fn test_scan_with_in_filter() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let candidates = [scalar_from_i32(10), scalar_from_i32(30)];
        let filter = Filter {
            field_id: COL_C_I32,
            op: Operator::In(&candidates),
        };
        let projection = vec![COL_A_U64];
        let result_batch = table.scan(&projection, &filter).unwrap();

        assert_eq!(result_batch.num_columns(), 1);
        let values = result_batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let collected: Vec<Option<u64>> = (0..values.len())
            .map(|i| {
                if values.is_null(i) {
                    None
                } else {
                    Some(values.value(i))
                }
            })
            .collect();
        assert_eq!(collected, vec![Some(100), Some(300)]);
    }

    #[test]
    fn test_scan_with_multiple_projection_columns() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        let filter = Filter {
            field_id: COL_C_I32,
            op: Operator::Equals(scalar_from_i32(20)),
        };
        let projection = vec![COL_A_U64, COL_C_I32];
        let batch = table.scan(&projection, &filter).unwrap();

        assert_eq!(batch.num_columns(), 2);
        let a_vals = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let a_collected: Vec<Option<u64>> = (0..a_vals.len())
            .map(|i| {
                if a_vals.is_null(i) {
                    None
                } else {
                    Some(a_vals.value(i))
                }
            })
            .collect();
        assert_eq!(a_collected, vec![Some(200), Some(200)]);

        let c_vals = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let c_collected: Vec<Option<i32>> = (0..c_vals.len())
            .map(|i| {
                if c_vals.is_null(i) {
                    None
                } else {
                    Some(c_vals.value(i))
                }
            })
            .collect();
        assert_eq!(c_collected, vec![Some(20), Some(20)]);
    }
}
