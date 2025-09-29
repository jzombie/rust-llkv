use std::sync::Arc;

use crate::planner::TablePlanner;
use crate::types::TableId;

use arrow::array::RecordBatch;
use arrow::datatypes::{Field, Schema};

use llkv_column_map::store::{Projection, ROW_ID_COLUMN_NAME};
use llkv_column_map::{ColumnStore, types::LogicalFieldId};
use llkv_storage::pager::{MemPager, Pager};
use simd_r_drive_entry_handle::EntryHandle;

use crate::sys_catalog::{CATALOG_TID, ColMeta, SysCatalog, TableMeta};
use crate::types::FieldId;
use llkv_expr::{Expr, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};

pub struct Table<P = MemPager>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    store: ColumnStore<P>,
    table_id: TableId,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ScanStreamOptions {
    /// Preserve null rows emitted by the projected columns when `true`.
    /// When `false`, the scan gatherer drops rows where all projected
    /// columns are null or missing before yielding batches. This keeps
    /// the table scan column-oriented while delegating row-level
    /// filtering to the column-map layer.
    pub include_nulls: bool,
}

#[derive(Clone, Debug)]
pub enum ScanProjection {
    Column(Projection),
    Computed {
        expr: ScalarExpr<FieldId>,
        alias: String,
    },
}

impl ScanProjection {
    pub fn column<P: Into<Projection>>(proj: P) -> Self {
        Self::Column(proj.into())
    }

    pub fn computed<S: Into<String>>(expr: ScalarExpr<FieldId>, alias: S) -> Self {
        Self::Computed {
            expr,
            alias: alias.into(),
        }
    }
}

impl From<Projection> for ScanProjection {
    fn from(value: Projection) -> Self {
        ScanProjection::Column(value)
    }
}

impl From<&Projection> for ScanProjection {
    fn from(value: &Projection) -> Self {
        ScanProjection::Column(value.clone())
    }
}

impl From<&ScanProjection> for ScanProjection {
    fn from(value: &ScanProjection) -> Self {
        value.clone()
    }
}

impl<P> Table<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn new(table_id: TableId, pager: Arc<P>) -> LlkvResult<Self> {
        if table_id == CATALOG_TID {
            return Err(Error::reserved_table_id(table_id));
        }

        let store = ColumnStore::open(pager)?;
        Ok(Self { store, table_id })
    }

    pub fn append(&self, batch: &RecordBatch) -> LlkvResult<()> {
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

            let lfid = LogicalFieldId::for_user(self.table_id, user_field_id);
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
        options: ScanStreamOptions,
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

    pub fn scan_stream_with_exprs<'a, F>(
        &self,
        projections: &[ScanProjection],
        filter_expr: &Expr<'a, FieldId>,
        options: ScanStreamOptions,
        on_batch: F,
    ) -> LlkvResult<()>
    where
        F: FnMut(RecordBatch),
    {
        TablePlanner::new(self).scan_stream_with_exprs(projections, filter_expr, options, on_batch)
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

    #[inline]
    pub fn table_id(&self) -> TableId {
        self.table_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sys_catalog::CATALOG_TID;
    use crate::types::RowId;
    use arrow::array::Array;
    use arrow::array::ArrayRef;
    use arrow::array::{
        BinaryArray, Float32Array, Float64Array, Int16Array, Int32Array, UInt8Array, UInt32Array,
        UInt64Array,
    };
    use arrow::compute::{cast, max, min, sum, unary};
    use arrow::datatypes::DataType;
    use llkv_column_map::ColumnStore;
    use llkv_column_map::store::GatherNullPolicy;
    use llkv_expr::{BinaryOp, CompareOp, Filter, Operator, ScalarExpr};
    use std::collections::HashMap;
    use std::ops::Bound;

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
        let result = Table::new(CATALOG_TID, Arc::new(MemPager::default()));
        assert!(matches!(
            result,
            Err(Error::ReservedTableId(id)) if id == CATALOG_TID
        ));
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
            let table = Table::new(TABLE_BETA, Arc::clone(&pager)).unwrap();
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
            let table = Table::new(TABLE_GAMMA, Arc::clone(&pager)).unwrap();
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
