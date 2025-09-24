use std::ops::Bound;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int32Array, RecordBatch, UInt64Array};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Int32Type, Schema, UInt64Type};

use llkv_column_map::store::FilterPrimitive;
use llkv_column_map::{
    ColumnStore, scan,
    storage::pager::MemPager,
    types::{LogicalFieldId, Namespace},
};

use crate::expr::{Filter, Literal, LiteralCastError, Operator, literal_to_native};
use crate::sys_catalog::{ColMeta, SysCatalog, TableMeta};
use crate::types::FieldId;

// TODO: Move to `llkv-expr`
/// Convert a bound of `Literal` into a bound of `T::Native`.
fn bound_to_native<T>(bound: &Bound<Literal>) -> Result<Bound<T::Native>, TableError>
where
    T: ArrowPrimitiveType + FilterPrimitive,
    T::Native: TryFrom<i128> + Copy,
{
    Ok(match bound {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(l) => Bound::Included(literal_to_native::<T::Native>(l)?),
        Bound::Excluded(l) => Bound::Excluded(literal_to_native::<T::Native>(l)?),
    })
}

enum RangeLimit<T> {
    Included(T),
    Excluded(T),
}

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

fn build_predicate<T>(op: &Operator<'_>) -> Result<Predicate<T>, TableError>
where
    T: ArrowPrimitiveType + FilterPrimitive,
    T::Native: TryFrom<i128> + Copy,
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
        Operator::LessThan(lit) => Ok(Predicate::LessThan(literal_to_native::<T::Native>(lit)?)),
        Operator::LessThanOrEquals(lit) => Ok(Predicate::LessThanOrEquals(literal_to_native::<
            T::Native,
        >(lit)?)),
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

fn collect_matching_row_ids<T>(
    store: &ColumnStore<MemPager>,
    field_id: LogicalFieldId,
    op: &Operator<'_>,
) -> Result<Vec<u64>, TableError>
where
    T: ArrowPrimitiveType + FilterPrimitive,
    T::Native: TryFrom<i128> + Copy,
{
    let predicate = build_predicate::<T>(op)?;
    store
        .filter_row_ids::<T, _>(field_id, move |value| predicate.matches(value))
        .map_err(TableError::from)
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
pub enum TableError {
    ColumnMap(llkv_column_map::error::Error),
    Arrow(arrow::error::ArrowError),
    ExprCast(LiteralCastError),
    Internal(String),
}

impl From<llkv_column_map::error::Error> for TableError {
    fn from(e: llkv_column_map::error::Error) -> Self {
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

    pub fn scan<'a>(
        &self,
        projection: &[FieldId],
        filter: &Filter<'a, FieldId>,
    ) -> Result<RecordBatch, TableError> {
        let filter_lfid = lfid_for(self.table_id, filter.field_id);

        // Determine dtype from the store. Operators now carry untyped
        // `Literal`s, so we dispatch based on the column's dtype.
        let dtype = self.store.data_type(filter_lfid)?;

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
            // Float types are not supported by FilterPrimitive in column-map.
            DataType::Float64 | DataType::Float32 => {
                return Err(TableError::Internal(
                    "Filtering on float columns is not supported".to_string(),
                ));
            }
            other => {
                return Err(TableError::Internal(format!(
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
    use arrow::compute::sum;
    use std::collections::HashMap;

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
            op: Operator::Equals(200.into()),
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
            op: Operator::Equals(20.into()),
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
            op: Operator::GreaterThan(15.into()),
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
                lower: Bound::Included(150.into()),
                upper: Bound::Excluded(300.into()),
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
    fn test_filtered_scan_sum_kernel() {
        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;

        let filter = Filter {
            field_id: COL_A_U64,
            op: Operator::Range {
                lower: Bound::Included(150.into()),
                upper: Bound::Excluded(300.into()),
            },
        };
        let projection = vec![COL_A_U64];
        let batch = table.scan(&projection, &filter).unwrap();

        assert_eq!(batch.num_columns(), 1);
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        // Only the two values equal to 200 should be included by the filter
        // range.
        let total = sum(values).expect("non-empty sum");
        assert_eq!(total, 400);
    }

    #[test]
    fn test_filtered_scan_sum_i32_kernel() {
        use arrow::compute::sum;

        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        // Same IN filter on a_u64 to select rows with c_i32 = [10, 30].
        let candidates = [100.into(), 300.into()];
        let filter = Filter {
            field_id: COL_A_U64,
            op: Operator::In(&candidates),
        };

        let projection = vec![COL_C_I32];
        let batch = table.scan(&projection, &filter).unwrap();

        assert_eq!(batch.num_columns(), 1);
        let c_vals = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        // Sum over the filtered c_i32 column: 10 + 30 = 40.
        let total = sum(c_vals).expect("non-empty sum");
        assert_eq!(total, 40);
    }

    #[test]
    fn test_filtered_scan_min_max_kernel() {
        use arrow::compute::{max, min};

        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        // Filter rows where a_u64 IN {100, 300}. That selects row_ids with
        // c_i32 values [10, 30] in our fixture.
        let candidates = [100.into(), 300.into()];
        let filter = Filter {
            field_id: COL_A_U64,
            op: Operator::In(&candidates),
        };

        let projection = vec![COL_C_I32];
        let batch = table.scan(&projection, &filter).unwrap();

        assert_eq!(batch.num_columns(), 1);
        let c_vals = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        // min/max should be 10 and 30 respectively.
        let mn = min(c_vals).expect("non-empty min");
        let mx = max(c_vals).expect("non-empty max");
        assert_eq!(mn, 10);
        assert_eq!(mx, 30);
    }

    #[test]
    fn test_filtered_scan_int_sqrt_float64() {
        use arrow::array::Float64Array;
        use arrow::compute::cast;

        let table = setup_test_table();
        const COL_A_U64: FieldId = 10;
        const COL_C_I32: FieldId = 12;

        // Filter rows where c_i32 > 15: picks rows with a_u64 = [200, 300, 200]
        let filter = Filter {
            field_id: COL_C_I32,
            op: Operator::GreaterThan(15.into()),
        };

        // Project the a_u64 column, then compute sqrt(a_u64) as f64.
        let projection = vec![COL_A_U64];
        let batch = table.scan(&projection, &filter).unwrap();

        assert_eq!(batch.num_columns(), 1);

        // Cast UInt64 -> Float64 for math
        let a_as_f64 = cast(batch.column(0), &arrow::datatypes::DataType::Float64).unwrap();
        let a_as_f64 = a_as_f64.as_any().downcast_ref::<Float64Array>().unwrap();

        // Apply sqrt in pure Rust over Option<f64> iterator
        let sqrt_vals = (0..a_as_f64.len()).map(|i| {
            if a_as_f64.is_null(i) {
                None
            } else {
                let v = a_as_f64.value(i);
                Some(v.sqrt())
            }
        });

        let sqrt_arr = Float64Array::from_iter(sqrt_vals);

        // Expected: sqrt([200, 300, 200])
        let expected = [200_f64.sqrt(), 300_f64.sqrt(), 200_f64.sqrt()];

        let got: Vec<Option<f64>> = (0..sqrt_arr.len())
            .map(|i| {
                if sqrt_arr.is_null(i) {
                    None
                } else {
                    Some(sqrt_arr.value(i))
                }
            })
            .collect();

        assert_eq!(got, expected.iter().copied().map(Some).collect::<Vec<_>>());
    }

    #[test]
    fn test_multi_field_kernels_with_filters() {
        use arrow::array::{Float64Array, Int16Array, UInt8Array, UInt32Array};
        use arrow::compute::{cast, max, min, sum};
        use std::collections::HashMap;

        let table = Table::new(2, TableCfg::default());

        // Field ids for this test table.
        const COL_A_U64: FieldId = 20;
        const COL_D_U32: FieldId = 21;
        const COL_E_I16: FieldId = 22;
        const COL_F_U8: FieldId = 23;
        const COL_C_I32: FieldId = 24;

        // Schema: row_id, a_u64, d_u32, e_i16, f_u8, c_i32.
        let schema = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
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

        // 1) SUM over d_u32 under the filter.
        let proj_d = vec![COL_D_U32];
        let batch_d = table.scan(&proj_d, &filter).unwrap();
        let d_vals = batch_d
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let d_sum = sum(d_vals).expect("non-empty sum");
        assert_eq!(d_sum, 8 + 9 + 10 + 11);

        // 2) MIN over e_i16 under the filter.
        let proj_e = vec![COL_E_I16];
        let batch_e = table.scan(&proj_e, &filter).unwrap();
        let e_vals = batch_e
            .column(0)
            .as_any()
            .downcast_ref::<Int16Array>()
            .unwrap();
        let e_min = min(e_vals).expect("non-empty min");
        assert_eq!(e_min, -6);

        // 3) MAX over f_u8 under the filter.
        let proj_f = vec![COL_F_U8];
        let batch_f = table.scan(&proj_f, &filter).unwrap();
        let f_vals = batch_f
            .column(0)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap();
        let f_max = max(f_vals).expect("non-empty max");
        assert_eq!(f_max, 10);

        // 4) SQRT over a_u64 under the filter. Cast to Float64, then sqrt.
        let proj_a = vec![COL_A_U64];
        let batch_a = table.scan(&proj_a, &filter).unwrap();
        let a_as_f64 = cast(batch_a.column(0), &arrow::datatypes::DataType::Float64).unwrap();
        let a_as_f64 = a_as_f64.as_any().downcast_ref::<Float64Array>().unwrap();
        let sqrt_vals = (0..a_as_f64.len()).map(|i| {
            if a_as_f64.is_null(i) {
                None
            } else {
                Some(a_as_f64.value(i).sqrt())
            }
        });
        let sqrt_arr = Float64Array::from_iter(sqrt_vals);

        // Expected sqrt for filtered rows: [225, 400, 900, 1600].
        let expected = [15.0_f64, 20.0, 30.0, 40.0];
        let got: Vec<Option<f64>> = (0..sqrt_arr.len())
            .map(|i| {
                if sqrt_arr.is_null(i) {
                    None
                } else {
                    Some(sqrt_arr.value(i))
                }
            })
            .collect();
        assert_eq!(got, expected.iter().copied().map(Some).collect::<Vec<_>>());
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
            op: Operator::Equals(20.into()),
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
