use std::cmp;
use std::ops::Bound;
use std::sync::Arc;

use crate::constants::STREAM_BATCH_ROWS;
use crate::scalar_eval::{NumericArrayMap, NumericKernels};
use crate::types::TableId;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Schema};

use llkv_column_map::store::{FilterPrimitive, GatherNullPolicy, Projection, ROW_ID_COLUMN_NAME};
use llkv_column_map::{
    ColumnStore,
    types::{LogicalFieldId, Namespace},
};
use llkv_storage::pager::{MemPager, Pager};
use simd_r_drive_entry_handle::EntryHandle;

use crate::sys_catalog::{CATALOG_TID, ColMeta, SysCatalog, TableMeta};
use crate::types::FieldId;
use llkv_expr::typed_predicate::build_predicate;
use llkv_expr::{CompareOp, Expr, Filter, Operator, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};
use rustc_hash::{FxHashMap, FxHashSet};

fn collect_matching_row_ids<T, P>(
    store: &ColumnStore<P>,
    field_id: LogicalFieldId,
    op: &Operator<'_>,
) -> LlkvResult<Vec<u64>>
where
    T: ArrowPrimitiveType + FilterPrimitive,
    T::Native: llkv_expr::literal::FromLiteral + Copy,
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let predicate = build_predicate::<T>(op).map_err(Error::predicate_build)?;
    store.filter_row_ids::<T, _>(field_id, move |value| predicate.matches(value))
}

fn normalize_row_ids(mut row_ids: Vec<u64>) -> Vec<u64> {
    row_ids.sort_unstable();
    row_ids.dedup();
    row_ids
}

fn intersect_sorted(left: Vec<u64>, right: Vec<u64>) -> Vec<u64> {
    let mut result = Vec::with_capacity(left.len().min(right.len()));
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        let lv = left[i];
        let rv = right[j];
        if lv == rv {
            result.push(lv);
            i += 1;
            j += 1;
        } else if lv < rv {
            i += 1;
        } else {
            j += 1;
        }
    }
    result
}

fn union_sorted(left: Vec<u64>, right: Vec<u64>) -> Vec<u64> {
    if left.is_empty() {
        return right;
    }
    if right.is_empty() {
        return left;
    }

    let mut result = Vec::with_capacity(left.len() + right.len());
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        let lv = left[i];
        let rv = right[j];
        if lv == rv {
            result.push(lv);
            i += 1;
            j += 1;
        } else if lv < rv {
            result.push(lv);
            i += 1;
        } else {
            result.push(rv);
            j += 1;
        }
    }

    while i < left.len() {
        result.push(left[i]);
        i += 1;
    }
    while j < right.len() {
        result.push(right[j]);
        j += 1;
    }

    result.dedup();
    result
}

fn difference_sorted(base: Vec<u64>, subtract: Vec<u64>) -> Vec<u64> {
    if base.is_empty() || subtract.is_empty() {
        return base;
    }

    let mut result = Vec::with_capacity(base.len());
    let mut i = 0;
    let mut j = 0;
    while i < base.len() && j < subtract.len() {
        let bv = base[i];
        let sv = subtract[j];
        if bv == sv {
            i += 1;
            j += 1;
        } else if bv < sv {
            result.push(bv);
            i += 1;
        } else {
            j += 1;
        }
    }
    while i < base.len() {
        result.push(base[i]);
        i += 1;
    }
    result
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
    /// Preserve null rows emitted by the projected columns when `true`.
    /// When `false`, the scan gatherer drops rows where all projected
    /// columns are null or missing before yielding batches. This keeps
    /// the table scan column-oriented while delegating row-level
    /// filtering to the column-map layer.
    pub include_nulls: bool,
}

#[derive(Clone, Debug)]
pub enum StreamProjection {
    Column(Projection),
    Computed {
        expr: ScalarExpr<FieldId>,
        alias: String,
    },
}

impl StreamProjection {
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

impl From<Projection> for StreamProjection {
    fn from(value: Projection) -> Self {
        StreamProjection::Column(value)
    }
}

#[derive(Clone)]
struct ColumnProjectionInfo {
    logical_field_id: LogicalFieldId,
    data_type: DataType,
    output_name: String,
}

#[derive(Clone)]
struct ComputedProjectionInfo {
    expr: ScalarExpr<FieldId>,
    alias: String,
}

#[derive(Clone)]
enum ProjectionEval {
    Column(ColumnProjectionInfo),
    Computed(ComputedProjectionInfo),
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
    pub fn scan_stream<'a, F>(
        &self,
        projections: &[Projection],
        filter_expr: &Expr<'a, FieldId>,
        options: ScanStreamOptions,
        on_batch: F,
    ) -> LlkvResult<()>
    where
        F: FnMut(RecordBatch),
    {
        let stream_projections: Vec<StreamProjection> = projections
            .iter()
            .cloned()
            .map(StreamProjection::from)
            .collect();
        self.scan_stream_with_exprs(&stream_projections, filter_expr, options, on_batch)
    }

    pub fn scan_stream_with_exprs<'a, F>(
        &self,
        projections: &[StreamProjection],
        filter_expr: &Expr<'a, FieldId>,
        options: ScanStreamOptions,
        mut on_batch: F,
    ) -> LlkvResult<()>
    where
        F: FnMut(RecordBatch),
    {
        if projections.is_empty() {
            return Err(Error::InvalidArgumentError(
                "scan_stream requires at least one projection".into(),
            ));
        }

        let mut projection_evals: Vec<ProjectionEval> = Vec::with_capacity(projections.len());
        let mut unique_index: FxHashMap<LogicalFieldId, usize> = FxHashMap::default();
        unique_index.reserve(projections.len());
        let mut unique_lfids: Vec<LogicalFieldId> = Vec::with_capacity(projections.len());
        let mut numeric_fields: FxHashSet<FieldId> = FxHashSet::default();

        for proj in projections {
            match proj {
                StreamProjection::Column(p) => {
                    let lfid = p.logical_field_id;
                    if lfid.table_id() != self.table_id {
                        return Err(Error::InvalidArgumentError(format!(
                            "Projection targets table {} but scan_stream is on table {}",
                            lfid.table_id(),
                            self.table_id
                        )));
                    }
                    if lfid.namespace() != Namespace::UserData {
                        return Err(Error::InvalidArgumentError(format!(
                            "Projection {:?} must target user data namespace",
                            lfid
                        )));
                    }
                    let dtype = self.store.data_type(lfid)?;
                    if let std::collections::hash_map::Entry::Vacant(e) = unique_index.entry(lfid)
                    {
                        e.insert(unique_lfids.len());
                        unique_lfids.push(lfid);
                    }
                    let fallback = lfid.field_id().to_string();
                    let output_name = p
                        .alias
                        .clone()
                        .unwrap_or_else(|| fallback.clone());
                    projection_evals.push(ProjectionEval::Column(ColumnProjectionInfo {
                        logical_field_id: lfid,
                        data_type: dtype,
                        output_name,
                    }));
                }
                StreamProjection::Computed { expr, alias } => {
                    if alias.trim().is_empty() {
                        return Err(Error::InvalidArgumentError(
                            "Computed projection requires a non-empty alias".into(),
                        ));
                    }
                    let mut fields_set: FxHashSet<FieldId> = FxHashSet::default();
                    NumericKernels::collect_fields(expr, &mut fields_set);
                    if fields_set.is_empty() {
                        return Err(Error::InvalidArgumentError(
                            "Computed projection must reference at least one column".into(),
                        ));
                    }
                    let mut ordered_fields: Vec<FieldId> = fields_set.into_iter().collect();
                    ordered_fields.sort_unstable();
                    for field_id in &ordered_fields {
                        let lfid = LogicalFieldId::for_user(self.table_id, *field_id);
                        if lfid.namespace() != Namespace::UserData {
                            return Err(Error::InvalidArgumentError(format!(
                                "Computed projection field {:?} must target user data namespace",
                                lfid
                            )));
                        }
                        if let std::collections::hash_map::Entry::Vacant(e) =
                            unique_index.entry(lfid)
                        {
                            e.insert(unique_lfids.len());
                            unique_lfids.push(lfid);
                        }
                        numeric_fields.insert(*field_id);
                    }
                    projection_evals.push(ProjectionEval::Computed(ComputedProjectionInfo {
                        expr: expr.clone(),
                        alias: alias.clone(),
                    }));
                }
            }
        }

        let mut all_rows_cache: FxHashMap<FieldId, Vec<u64>> = FxHashMap::default();
        let row_ids = self.collect_row_ids_for_expr(filter_expr, &mut all_rows_cache)?;

        if row_ids.is_empty() {
            return Ok(());
        }

        let fields: Vec<Field> = projection_evals
            .iter()
            .map(|eval| match eval {
                ProjectionEval::Column(info) => {
                    Field::new(info.output_name.clone(), info.data_type.clone(), true)
                }
                ProjectionEval::Computed(info) => {
                    Field::new(info.alias.clone(), DataType::Float64, true)
                }
            })
            .collect();
        let out_schema = Arc::new(Schema::new(fields));

        let mut start = 0usize;
        let null_policy = if options.include_nulls {
            GatherNullPolicy::IncludeNulls
        } else {
            GatherNullPolicy::DropNulls
        };
        while start < row_ids.len() {
            let end = cmp::min(start + STREAM_BATCH_ROWS, row_ids.len());
            let window = &row_ids[start..end];

            let gathered =
                self.store
                    .gather_rows_multi_with_policy(&unique_lfids, window, null_policy)?;

            if gathered.num_columns() == 0 || gathered.num_rows() == 0 {
                start = end;
                continue;
            }

            let unique_arrays = gathered.columns();
            let numeric_arrays: NumericArrayMap =
                NumericKernels::prepare_numeric_arrays(&unique_lfids, unique_arrays, &numeric_fields)?;

            let mut columns = Vec::with_capacity(projection_evals.len());
            for eval in &projection_evals {
                match eval {
                    ProjectionEval::Column(info) => {
                        let idx = unique_index
                            .get(&info.logical_field_id)
                            .copied()
                            .expect("lfid missing from unique-index");
                        columns.push(Arc::clone(&unique_arrays[idx]));
                    }
                    ProjectionEval::Computed(info) => {
                        let array = NumericKernels::evaluate_batch(
                            &info.expr,
                            gathered.num_rows(),
                            &numeric_arrays,
                        )?;
                        columns.push(array);
                    }
                }
            }

            let batch = RecordBatch::try_new(out_schema.clone(), columns)?;
            on_batch(batch);
            start = end;
        }

        Ok(())
    }

    fn collect_row_ids_for_filter<'a>(&self, filter: &Filter<'a, FieldId>) -> LlkvResult<Vec<u64>> {
        let filter_lfid = LogicalFieldId::for_user(self.table_id, filter.field_id);
        let dtype = self.store.data_type(filter_lfid)?;

        let row_ids = llkv_column_map::with_integer_arrow_type!(
            dtype.clone(),
            |ArrowTy| collect_matching_row_ids::<ArrowTy, P>(&self.store, filter_lfid, &filter.op),
            Err(Error::Internal(format!(
                "Filtering on type {:?} is not supported",
                dtype
            ))),
        )?;

        Ok(normalize_row_ids(row_ids))
    }

    fn collect_row_ids_for_expr<'a>(
        &self,
        expr: &Expr<'a, FieldId>,
        all_rows_cache: &mut FxHashMap<FieldId, Vec<u64>>,
    ) -> LlkvResult<Vec<u64>> {
        match expr {
            Expr::Pred(filter) => self.collect_row_ids_for_filter(filter),
            Expr::Compare { left, op, right } => {
                self.collect_row_ids_for_compare(left, *op, right, all_rows_cache)
            }
            Expr::And(children) => {
                if children.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "AND expression requires at least one predicate".into(),
                    ));
                }
                let mut iter = children.iter();
                let mut acc =
                    self.collect_row_ids_for_expr(iter.next().unwrap(), all_rows_cache)?;
                for child in iter {
                    let next_ids = self.collect_row_ids_for_expr(child, all_rows_cache)?;
                    if acc.is_empty() || next_ids.is_empty() {
                        return Ok(Vec::new());
                    }
                    acc = intersect_sorted(acc, next_ids);
                    if acc.is_empty() {
                        break;
                    }
                }
                Ok(acc)
            }
            Expr::Or(children) => {
                if children.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "OR expression requires at least one predicate".into(),
                    ));
                }
                let mut acc = Vec::new();
                for child in children {
                    let next_ids = self.collect_row_ids_for_expr(child, all_rows_cache)?;
                    if acc.is_empty() {
                        acc = next_ids;
                    } else if !next_ids.is_empty() {
                        acc = union_sorted(acc, next_ids);
                    }
                }
                Ok(acc)
            }
            Expr::Not(inner) => {
                let domain = self.collect_row_ids_domain(inner, all_rows_cache)?;
                if domain.is_empty() {
                    return Ok(Vec::new());
                }
                let matched = self.collect_row_ids_for_expr(inner, all_rows_cache)?;
                Ok(difference_sorted(domain, matched))
            }
        }
    }

    fn collect_row_ids_for_compare(
        &self,
        left: &ScalarExpr<FieldId>,
        op: CompareOp,
        right: &ScalarExpr<FieldId>,
        all_rows_cache: &mut FxHashMap<FieldId, Vec<u64>>,
    ) -> LlkvResult<Vec<u64>> {
        let mut fields = FxHashSet::default();
    NumericKernels::collect_fields(left, &mut fields);
    NumericKernels::collect_fields(right, &mut fields);
        if fields.is_empty() {
            return Err(Error::InvalidArgumentError(
                "Comparison expression must reference at least one column".into(),
            ));
        }
        let mut domain: Option<Vec<u64>> = None;
        let mut ordered_fields: Vec<FieldId> = fields.into_iter().collect();
        ordered_fields.sort_unstable();
        for fid in &ordered_fields {
            let rows = self.collect_all_row_ids_for_field(*fid, all_rows_cache)?;
            domain = Some(match domain {
                Some(existing) => intersect_sorted(existing, rows),
                None => rows,
            });
            if let Some(ref d) = domain {
                if d.is_empty() {
                    return Ok(Vec::new());
                }
            }
        }
        let domain = domain.unwrap_or_default();
        if domain.is_empty() {
            return Ok(domain);
        }
        self.evaluate_compare_over_rows(&domain, &ordered_fields, left, op, right)
    }

    fn evaluate_compare_over_rows(
        &self,
        row_ids: &[u64],
        fields: &[FieldId],
        left: &ScalarExpr<FieldId>,
        op: CompareOp,
        right: &ScalarExpr<FieldId>,
    ) -> LlkvResult<Vec<u64>> {
        if row_ids.is_empty() {
            return Ok(Vec::new());
        }
        let lfids: Vec<LogicalFieldId> = fields
            .iter()
            .map(|fid| LogicalFieldId::for_user(self.table_id, *fid))
            .collect();
        let mut result = Vec::new();
        let numeric_fields: FxHashSet<FieldId> = fields.iter().copied().collect();
        let mut start = 0usize;
        while start < row_ids.len() {
            let end = cmp::min(start + STREAM_BATCH_ROWS, row_ids.len());
            let window = &row_ids[start..end];
            let gathered = self.store.gather_rows_multi_with_policy(
                &lfids,
                window,
                GatherNullPolicy::IncludeNulls,
            )?;
            if gathered.num_rows() == 0 {
                start = end;
                continue;
            }
            let arrays = gathered.columns();
            let numeric_arrays: NumericArrayMap =
                NumericKernels::prepare_numeric_arrays(&lfids, arrays, &numeric_fields)?;
            for (offset, &row_id) in window.iter().enumerate() {
                let left_val = NumericKernels::evaluate_value(left, offset, &numeric_arrays)?;
                let right_val = NumericKernels::evaluate_value(right, offset, &numeric_arrays)?;
                if let (Some(lv), Some(rv)) = (left_val, right_val) {
                    if NumericKernels::compare(op, lv, rv) {
                        result.push(row_id);
                    }
                }
            }
            start = end;
        }
        Ok(result)
    }

    fn collect_row_ids_domain<'a>(
        &self,
        expr: &Expr<'a, FieldId>,
        all_rows_cache: &mut FxHashMap<FieldId, Vec<u64>>,
    ) -> LlkvResult<Vec<u64>> {
        match expr {
            Expr::Pred(filter) => {
                self.collect_all_row_ids_for_field(filter.field_id, all_rows_cache)
            }
            Expr::Compare { left, right, .. } => {
                let mut fields = FxHashSet::default();
                NumericKernels::collect_fields(left, &mut fields);
                NumericKernels::collect_fields(right, &mut fields);
                if fields.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "Comparison expression must reference at least one column".into(),
                    ));
                }
                let mut domain: Option<Vec<u64>> = None;
                let mut ordered_fields: Vec<FieldId> = fields.into_iter().collect();
                ordered_fields.sort_unstable();
                for fid in ordered_fields {
                    let rows = self.collect_all_row_ids_for_field(fid, all_rows_cache)?;
                    domain = Some(match domain {
                        Some(existing) => intersect_sorted(existing, rows),
                        None => rows,
                    });
                    if let Some(ref d) = domain {
                        if d.is_empty() {
                            return Ok(Vec::new());
                        }
                    }
                }
                Ok(domain.unwrap_or_default())
            }
            Expr::And(children) | Expr::Or(children) => {
                let mut acc = Vec::new();
                for child in children {
                    let next_ids = self.collect_row_ids_domain(child, all_rows_cache)?;
                    if acc.is_empty() {
                        acc = next_ids;
                    } else if !next_ids.is_empty() {
                        acc = union_sorted(acc, next_ids);
                    }
                }
                Ok(acc)
            }
            Expr::Not(inner) => self.collect_row_ids_domain(inner, all_rows_cache),
        }
    }

    fn collect_all_row_ids_for_field(
        &self,
        field_id: FieldId,
        cache: &mut FxHashMap<FieldId, Vec<u64>>,
    ) -> LlkvResult<Vec<u64>> {
        if let Some(existing) = cache.get(&field_id) {
            return Ok(existing.clone());
        }

        let filter = Filter {
            field_id,
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
        };
        let ids = self.collect_row_ids_for_filter(&filter)?;
        cache.insert(field_id, ids.clone());
        Ok(ids)
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

pub struct ColumnData {
    pub data: ArrayRef,
    pub data_type: DataType,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sys_catalog::CATALOG_TID;
    use crate::types::RowId;
    use arrow::array::Array;
    use arrow::array::{
        BinaryArray, Float32Array, Float64Array, Int16Array, Int32Array, UInt8Array, UInt32Array,
        UInt64Array,
    };
    use arrow::compute::{cast, max, min, sum, unary};
    use std::collections::HashMap;
    use std::ops::Bound;
    use llkv_expr::BinaryOp;

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
                let arr = store.gather_rows(lfid, &alpha_rows, false).unwrap();
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
            let arr_u64 = store.gather_rows(lfid_u64, &beta_rows, false).unwrap();
            let arr_u64 = arr_u64.as_any().downcast_ref::<UInt64Array>().unwrap();
            assert_eq!(arr_u64.values(), beta_vals_u64.as_slice());

            let lfid_u8 = LogicalFieldId::for_user(TABLE_BETA, COL_BETA_U8);
            assert_eq!(store.data_type(lfid_u8).unwrap(), DataType::UInt8);
            let arr_u8 = store.gather_rows(lfid_u8, &beta_rows, false).unwrap();
            let arr_u8 = arr_u8.as_any().downcast_ref::<UInt8Array>().unwrap();
            assert_eq!(arr_u8.values(), beta_vals_u8.as_slice());
        }

        {
            let table = Table::new(TABLE_GAMMA, Arc::clone(&pager)).unwrap();
            let store = table.store();
            let lfid = LogicalFieldId::for_user(TABLE_GAMMA, COL_GAMMA_I16);
            assert_eq!(store.data_type(lfid).unwrap(), DataType::Int16);
            let arr = store.gather_rows(lfid, &gamma_rows, false).unwrap();
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
            StreamProjection::column(proj(&table, COL_A_U64)),
            StreamProjection::computed(
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
