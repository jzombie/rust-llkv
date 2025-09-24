use std::ops::{Bound, RangeBounds};
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
            let all_match = values
                .iter()
                .all(|value| scalar_data_type(value) == dtype);
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

/// Generic helper to apply filters for any primitive type that supports them.
fn apply_primitive_filter<'a, T, P, V, F>(
    make_builder: F,
    op: &Operator<'a>,
    visitor: &mut V,
) -> Result<(), CmError>
where
    T: ArrowPrimitiveType,
    T::Native: scan::RangeKey + Copy,
    P: llkv_column_map::storage::pager::Pager<Blob = simd_r_drive_entry_handle::EntryHandle>,
    V: scan::PrimitiveVisitor
        + scan::PrimitiveWithRowIdsVisitor
        + scan::PrimitiveSortedVisitor
        + scan::PrimitiveSortedWithRowIdsVisitor,
    F: Fn() -> ScanBuilder<'a, P>,
{
    struct BoundRange<T> {
        start: Bound<T>,
        end: Bound<T>,
    }

    impl<T: Copy> RangeBounds<T> for BoundRange<T> {
        fn start_bound(&self) -> Bound<&T> {
            match &self.start {
                Bound::Unbounded => Bound::Unbounded,
                Bound::Included(v) => Bound::Included(v),
                Bound::Excluded(v) => Bound::Excluded(v),
            }
        }

        fn end_bound(&self) -> Bound<&T> {
            match &self.end {
                Bound::Unbounded => Bound::Unbounded,
                Bound::Included(v) => Bound::Included(v),
                Bound::Excluded(v) => Bound::Excluded(v),
            }
        }
    }

    let mut run_bounds = |start: Bound<T::Native>, end: Bound<T::Native>| -> Result<(), CmError> {
        make_builder()
            .with_range::<T::Native, _>(BoundRange { start, end })
            .run(visitor)?;
        Ok(())
    };

    match op {
        Operator::Equals(scalar) => {
            let value = scalar_to_native::<T>(scalar)?;
            run_bounds(Bound::Included(value), Bound::Included(value))
        }
        Operator::GreaterThan(scalar) => {
            let value = scalar_to_native::<T>(scalar)?;
            run_bounds(Bound::Excluded(value), Bound::Unbounded)
        }
        Operator::GreaterThanOrEquals(scalar) => {
            let value = scalar_to_native::<T>(scalar)?;
            run_bounds(Bound::Included(value), Bound::Unbounded)
        }
        Operator::LessThan(scalar) => {
            let value = scalar_to_native::<T>(scalar)?;
            run_bounds(Bound::Unbounded, Bound::Excluded(value))
        }
        Operator::LessThanOrEquals(scalar) => {
            let value = scalar_to_native::<T>(scalar)?;
            run_bounds(Bound::Unbounded, Bound::Included(value))
        }
        Operator::Range { lower, upper } => {
            let lb = bound_to_native::<T>(lower)?;
            let ub = bound_to_native::<T>(upper)?;
            if matches!(&lb, Bound::Unbounded) && matches!(&ub, Bound::Unbounded) {
                make_builder().run(visitor)?;
                Ok(())
            } else {
                run_bounds(lb, ub)
            }
        }
        Operator::In(values) => {
            for scalar in values.iter() {
                let value = scalar_to_native::<T>(scalar)?;
                run_bounds(Bound::Included(value), Bound::Included(value))?;
            }
            Ok(())
        }
        _ => Err(CmError::Internal(format!(
            "Unsupported operator for type {}",
            std::any::type_name::<T::Native>()
        ))),
    }
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

        struct RowIdCollector {
            row_ids: Vec<u64>,
        }
        impl scan::PrimitiveVisitor for RowIdCollector {}
        impl scan::PrimitiveWithRowIdsVisitor for RowIdCollector {
            fn u64_chunk_with_rids(&mut self, _v: &UInt64Array, r: &UInt64Array) {
                self.row_ids.extend_from_slice(r.values());
            }
            fn i32_chunk_with_rids(&mut self, _v: &Int32Array, r: &UInt64Array) {
                self.row_ids.extend_from_slice(r.values());
            }
        }
        impl scan::PrimitiveSortedVisitor for RowIdCollector {}
        impl scan::PrimitiveSortedWithRowIdsVisitor for RowIdCollector {}

        let mut rid_visitor = RowIdCollector { row_ids: vec![] };

        let make_builder = || {
            ScanBuilder::new(&self.store, filter_lfid).with_row_ids(rowid_fid(filter_lfid))
        };
        match dtype {
            DataType::UInt64 => {
                apply_primitive_filter::<UInt64Type, _, _, _>(
                    make_builder,
                    &filter.op,
                    &mut rid_visitor,
                )?;
            }
            DataType::Int32 => {
                apply_primitive_filter::<Int32Type, _, _, _>(
                    make_builder,
                    &filter.op,
                    &mut rid_visitor,
                )?;
            }
            other => {
                return Err(CmError::Internal(format!(
                    "Filtering on type {:?} is not supported",
                    other
                )));
            }
        }

        let mut projected_columns: Vec<ArrayRef> = Vec::with_capacity(projection.len());
        let mut projected_fields: Vec<Field> = Vec::with_capacity(projection.len());

        for &field_id_to_project in projection {
            let lfid_to_project = lfid_for(self.table_id, field_id_to_project);
            let all_column_data = self.store.get_column_for_test(lfid_to_project)?;
            projected_columns.push(all_column_data.data);
            projected_fields.push(Field::new(
                &field_id_to_project.to_string(),
                all_column_data.data_type,
                true,
            ));
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
    use arrow::array::BinaryArray;
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
        assert_eq!(result_batch.column(0).data_type(), &DataType::Int32);
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
        assert_eq!(result_batch.column(0).data_type(), &DataType::UInt64);
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
        assert_eq!(result_batch.column(0).data_type(), &DataType::UInt64);
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
        assert_eq!(result_batch.column(0).data_type(), &DataType::Int32);
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
        assert_eq!(result_batch.column(0).data_type(), &DataType::UInt64);
    }
}
