use std::cmp;
use std::ops::Bound;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    RecordBatch, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Schema};

use llkv_column_map::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanBuilder, ScanOptions,
};
use llkv_column_map::store::{
    FilterPrimitive, FilterResult, FilterRun, GatherNullPolicy, dense_row_runs,
};
use llkv_column_map::types::{LogicalFieldId, Namespace};
use llkv_expr::literal::FromLiteral;
use llkv_expr::typed_predicate::build_predicate;
use llkv_expr::{CompareOp, Expr, Filter, Operator, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;

use llkv_storage::pager::Pager;

use crate::constants::STREAM_BATCH_ROWS;
use crate::scalar_eval::{NumericArrayMap, NumericKernels};
use crate::table::{ScanStreamOptions, StreamProjection, Table};
use crate::types::FieldId;

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

pub(crate) struct TablePlanner<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table: &'a Table<P>,
}

impl<'a, P> TablePlanner<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub(crate) fn new(table: &'a Table<P>) -> Self {
        Self { table }
    }

    pub(crate) fn scan_stream_with_exprs<'expr, F>(
        &self,
        projections: &[StreamProjection],
        filter_expr: &Expr<'expr, FieldId>,
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
                    if lfid.table_id() != self.table.table_id() {
                        return Err(Error::InvalidArgumentError(format!(
                            "Projection targets table {} but scan_stream is on table {}",
                            lfid.table_id(),
                            self.table.table_id()
                        )));
                    }
                    if lfid.namespace() != Namespace::UserData {
                        return Err(Error::InvalidArgumentError(format!(
                            "Projection {:?} must target user data namespace",
                            lfid
                        )));
                    }
                    let dtype = self.table.store().data_type(lfid)?;
                    if let std::collections::hash_map::Entry::Vacant(e) = unique_index.entry(lfid) {
                        e.insert(unique_lfids.len());
                        unique_lfids.push(lfid);
                    }
                    let fallback = lfid.field_id().to_string();
                    let output_name = p.alias.clone().unwrap_or_else(|| fallback.clone());
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
                        let lfid = LogicalFieldId::for_user(self.table.table_id(), *field_id);
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

        if let Expr::Pred(filter) = filter_expr
            && self.try_fast_scan_stream(
                &projection_evals,
                &unique_index,
                &unique_lfids,
                filter,
                &options,
                &mut on_batch,
            )?
        {
            return Ok(());
        }

        let passthrough_fields: Vec<Option<FieldId>> = projection_evals
            .iter()
            .map(|eval| match eval {
                ProjectionEval::Computed(info) => NumericKernels::passthrough_column(&info.expr),
                _ => None,
            })
            .collect();

        let mut fields: Vec<Field> = Vec::with_capacity(projection_evals.len());
        for (idx, eval) in projection_evals.iter().enumerate() {
            match eval {
                ProjectionEval::Column(info) => {
                    fields.push(Field::new(
                        info.output_name.clone(),
                        info.data_type.clone(),
                        true,
                    ));
                }
                ProjectionEval::Computed(info) => {
                    if let Some(fid) = passthrough_fields[idx] {
                        let lfid = LogicalFieldId::for_user(self.table.table_id(), fid);
                        let dtype = self.table.store().data_type(lfid)?;
                        fields.push(Field::new(info.alias.clone(), dtype, true));
                    } else {
                        fields.push(Field::new(info.alias.clone(), DataType::Float64, true));
                    }
                }
            }
        }
        let out_schema = Arc::new(Schema::new(fields));

        if self.try_scan_builder_passthrough(
            &projection_evals,
            &passthrough_fields,
            &out_schema,
            filter_expr,
            &options,
            &mut on_batch,
        )? {
            return Ok(());
        }

        let mut all_rows_cache: FxHashMap<FieldId, Vec<u64>> = FxHashMap::default();
        let row_ids = self.collect_row_ids_for_expr(filter_expr, &mut all_rows_cache)?;

        if row_ids.is_empty() {
            return Ok(());
        }

        let mut gather_ctx = self.table.store().prepare_gather_context(&unique_lfids)?;
        let mut start = 0usize;
        let null_policy = if options.include_nulls {
            GatherNullPolicy::IncludeNulls
        } else {
            GatherNullPolicy::DropNulls
        };
        while start < row_ids.len() {
            let mut end = cmp::min(start + STREAM_BATCH_ROWS, row_ids.len());
            if let Some((_, _, span_max)) = gather_ctx.chunk_span_for_row(row_ids[start]) {
                while end < row_ids.len() && row_ids[end - 1] <= span_max {
                    end += 1;
                    if end - start >= STREAM_BATCH_ROWS {
                        break;
                    }
                }
                while end <= row_ids.len() && row_ids[end - 1] > span_max {
                    end -= 1;
                }
            }
            let end = end.max(start + 1);
            let window = &row_ids[start..end];

            let gathered = self.table.store().gather_rows_with_reusable_context(
                &mut gather_ctx,
                window,
                null_policy,
            )?;

            if gathered.num_columns() == 0 || gathered.num_rows() == 0 {
                start = end;
                continue;
            }

            let unique_arrays = gathered.columns();
            let requires_numeric = projection_evals.iter().zip(passthrough_fields.iter()).any(
                |(eval, passthrough)| {
                    matches!(eval, ProjectionEval::Computed(_)) && passthrough.is_none()
                },
            );

            let numeric_arrays: Option<NumericArrayMap> = if requires_numeric {
                Some(NumericKernels::prepare_numeric_arrays(
                    &unique_lfids,
                    unique_arrays,
                    &numeric_fields,
                )?)
            } else {
                None
            };

            let mut columns = Vec::with_capacity(projection_evals.len());
            for (idx, eval) in projection_evals.iter().enumerate() {
                match eval {
                    ProjectionEval::Column(info) => {
                        let col_idx = unique_index
                            .get(&info.logical_field_id)
                            .copied()
                            .expect("lfid missing from unique-index");
                        columns.push(Arc::clone(&unique_arrays[col_idx]));
                    }
                    ProjectionEval::Computed(info) => {
                        if let Some(fid) = passthrough_fields[idx] {
                            let lfid = LogicalFieldId::for_user(self.table.table_id(), fid);
                            let col_idx = unique_index
                                .get(&lfid)
                                .copied()
                                .expect("passthrough field missing from unique-index");
                            columns.push(Arc::clone(&unique_arrays[col_idx]));
                        } else {
                            let numeric_arrays = numeric_arrays
                                .as_ref()
                                .expect("numeric arrays should exist for computed projection");
                            let array = NumericKernels::evaluate_batch(
                                &info.expr,
                                gathered.num_rows(),
                                numeric_arrays,
                            )?;
                            columns.push(array);
                        }
                    }
                }
            }

            let batch = RecordBatch::try_new(out_schema.clone(), columns)?;
            on_batch(batch);
            start = end;
        }

        Ok(())
    }

    fn try_fast_scan_stream<F>(
        &self,
        projection_evals: &[ProjectionEval],
        unique_index: &FxHashMap<LogicalFieldId, usize>,
        unique_lfids: &[LogicalFieldId],
        filter: &Filter<'_, FieldId>,
        options: &ScanStreamOptions,
        on_batch: &mut F,
    ) -> LlkvResult<bool>
    where
        F: FnMut(RecordBatch),
    {
        if projection_evals
            .iter()
            .any(|eval| !matches!(eval, ProjectionEval::Column(_)))
        {
            return Ok(false);
        }

        let filter_lfid = LogicalFieldId::for_user(self.table.table_id(), filter.field_id);
        let dtype = self.table.store().data_type(filter_lfid)?;
        let dense_runs = match &filter.op {
            Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            } => dense_row_runs(self.table.store(), filter_lfid)?,
            _ => None,
        };

        let runs = if let Some(runs) = dense_runs {
            if runs.is_empty() {
                return Ok(true);
            }
            runs
        } else {
            let result = llkv_column_map::with_integer_arrow_type!(
                dtype.clone(),
                |ArrowTy| self.collect_filter_result::<ArrowTy>(filter_lfid, &filter.op),
                return Ok(false),
            )?;

            if result.total_matches() == 0 {
                return Ok(true);
            }

            if !result.is_dense() {
                return Ok(false);
            }

            match result.into_runs() {
                Some(runs) if runs.is_empty() => return Ok(true),
                Some(runs) => runs,
                None => return Ok(false),
            }
        };

        let mut gather_ctx = self.table.store().prepare_gather_context(unique_lfids)?;
        if gather_ctx.is_empty() {
            return Ok(true);
        }

        let null_policy = if options.include_nulls {
            GatherNullPolicy::IncludeNulls
        } else {
            GatherNullPolicy::DropNulls
        };

        let fields: Vec<Field> = projection_evals
            .iter()
            .map(|eval| match eval {
                ProjectionEval::Column(info) => {
                    Field::new(info.output_name.clone(), info.data_type.clone(), true)
                }
                _ => unreachable!("fast path only handles direct column projections"),
            })
            .collect();
        let schema = Arc::new(Schema::new(fields));

        let mut temp_row_ids: Vec<u64> = Vec::with_capacity(STREAM_BATCH_ROWS);

        for run in runs {
            let mut remaining = run.len;
            let mut current_row = run.start_row_id;

            let run_last = run.start_row_id + (run.len.saturating_sub(1) as u64);
            if gather_ctx.chunk_span_for_row(run_last).is_none() {
                return Ok(false);
            }

            while remaining > 0 {
                let (_, _, chunk_max) = match gather_ctx.chunk_span_for_row(current_row) {
                    Some(span) => span,
                    None => return Ok(false),
                };
                let chunk_limit = (chunk_max - current_row + 1) as usize;
                let batch_rows = remaining.min(chunk_limit).min(STREAM_BATCH_ROWS);

                temp_row_ids.clear();
                temp_row_ids.reserve(batch_rows);
                for offset in 0..batch_rows {
                    temp_row_ids.push(current_row + offset as u64);
                }

                let gathered = match self.table.store().gather_rows_with_reusable_context(
                    &mut gather_ctx,
                    &temp_row_ids,
                    null_policy,
                ) {
                    Ok(batch) => batch,
                    Err(_) => return Ok(false),
                };

                if gathered.num_rows() > 0 {
                    let unique_arrays = gathered.columns();
                    let mut columns = Vec::with_capacity(projection_evals.len());
                    for eval in projection_evals {
                        if let ProjectionEval::Column(info) = eval {
                            let idx = unique_index
                                .get(&info.logical_field_id)
                                .copied()
                                .expect("logical field missing from unique index");
                            columns.push(Arc::clone(&unique_arrays[idx]));
                        }
                    }

                    let batch = RecordBatch::try_new(schema.clone(), columns)?;
                    on_batch(batch);
                }

                current_row += batch_rows as u64;
                remaining -= batch_rows;
            }
        }

        Ok(true)
    }

    fn try_scan_builder_passthrough<'expr, F>(
        &self,
        projection_evals: &[ProjectionEval],
        passthrough_fields: &[Option<FieldId>],
        schema: &Arc<Schema>,
        filter_expr: &Expr<'expr, FieldId>,
        options: &ScanStreamOptions,
        on_batch: &mut F,
    ) -> LlkvResult<bool>
    where
        F: FnMut(RecordBatch),
    {
        if options.include_nulls {
            return Ok(false);
        }
        if projection_evals.len() != 1 {
            return Ok(false);
        }

        let (target_lfid, dtype) = match &projection_evals[0] {
            ProjectionEval::Column(info) => (info.logical_field_id, info.data_type.clone()),
            ProjectionEval::Computed(_info) => {
                let passthrough = match passthrough_fields.first().and_then(|&f| f) {
                    Some(fid) => fid,
                    None => return Ok(false),
                };
                let lfid = LogicalFieldId::for_user(self.table.table_id(), passthrough);
                let dtype = self.table.store().data_type(lfid)?;
                (lfid, dtype)
            }
        };

        if !is_supported_numeric(&dtype) {
            return Ok(false);
        }

        let filter = match filter_expr {
            Expr::Pred(pred) => pred,
            _ => return Ok(false),
        };
        match &filter.op {
            Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            } => {}
            _ => return Ok(false),
        }

        let mut emitter = BuilderBatchEmitter {
            schema: Arc::clone(schema),
            on_batch,
            error: None,
            emitted: false,
        };

        let scan_opts = ScanOptions {
            with_row_ids: false,
            include_nulls: false,
            ..Default::default()
        };

        ScanBuilder::new(self.table.store(), target_lfid)
            .options(scan_opts)
            .run(&mut emitter)?;

        if let Some(err) = emitter.error {
            return Err(err);
        }

        Ok(emitter.emitted)
    }

    fn collect_row_ids_for_filter(&self, filter: &Filter<'_, FieldId>) -> LlkvResult<Vec<u64>> {
        let filter_lfid = LogicalFieldId::for_user(self.table.table_id(), filter.field_id);
        let dtype = self.table.store().data_type(filter_lfid)?;

        if let Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        } = &filter.op
            && let Some(runs) = dense_row_runs(self.table.store(), filter_lfid)?
        {
            return Ok(expand_filter_runs(&runs));
        }

        let row_ids = llkv_column_map::with_integer_arrow_type!(
            dtype.clone(),
            |ArrowTy| self.collect_matching_row_ids::<ArrowTy>(filter_lfid, &filter.op),
            Err(Error::Internal(format!(
                "Filtering on type {:?} is not supported",
                dtype
            ))),
        )?;

        Ok(normalize_row_ids(row_ids))
    }

    fn collect_row_ids_for_expr(
        &self,
        expr: &Expr<'_, FieldId>,
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
            if let Some(ref d) = domain
                && d.is_empty()
            {
                return Ok(Vec::new());
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
            .map(|fid| LogicalFieldId::for_user(self.table.table_id(), *fid))
            .collect();
        let mut result = Vec::new();
        let numeric_fields: FxHashSet<FieldId> = fields.iter().copied().collect();
        let mut start = 0usize;
        while start < row_ids.len() {
            let end = cmp::min(start + STREAM_BATCH_ROWS, row_ids.len());
            let window = &row_ids[start..end];
            let gathered =
                self.table
                    .store()
                    .gather_rows(&lfids, window, GatherNullPolicy::IncludeNulls)?;
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
                if let (Some(lv), Some(rv)) = (left_val, right_val)
                    && NumericKernels::compare(op, lv, rv)
                {
                    result.push(row_id);
                }
            }
            start = end;
        }
        Ok(result)
    }

    fn collect_row_ids_domain(
        &self,
        expr: &Expr<'_, FieldId>,
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
                    if let Some(ref d) = domain
                        && d.is_empty()
                    {
                        return Ok(Vec::new());
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

    fn collect_matching_row_ids<T>(
        &self,
        field_id: LogicalFieldId,
        op: &Operator<'_>,
    ) -> LlkvResult<Vec<u64>>
    where
        T: ArrowPrimitiveType + FilterPrimitive,
        T::Native: FromLiteral + Copy,
    {
        let predicate = build_predicate::<T>(op).map_err(Error::predicate_build)?;
        self.table
            .store()
            .filter_row_ids::<T, _>(field_id, move |value| predicate.matches(value))
    }

    fn collect_filter_result<T>(
        &self,
        field_id: LogicalFieldId,
        op: &Operator<'_>,
    ) -> LlkvResult<FilterResult>
    where
        T: ArrowPrimitiveType + FilterPrimitive,
        T::Native: FromLiteral + Copy,
    {
        let predicate = build_predicate::<T>(op).map_err(Error::predicate_build)?;
        self.table
            .store()
            .filter_matches::<T, _>(field_id, move |value| predicate.matches(value))
    }
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

fn expand_filter_runs(runs: &[FilterRun]) -> Vec<u64> {
    let total: usize = runs.iter().map(|r| r.len).sum();
    let mut row_ids = Vec::with_capacity(total);
    for run in runs {
        let mut row_id = run.start_row_id;
        for _ in 0..run.len {
            row_ids.push(row_id);
            row_id += 1;
        }
    }
    row_ids
}

fn is_supported_numeric(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::UInt64
            | DataType::UInt32
            | DataType::UInt16
            | DataType::UInt8
            | DataType::Int64
            | DataType::Int32
            | DataType::Int16
            | DataType::Int8
            | DataType::Float64
            | DataType::Float32
    )
}

struct BuilderBatchEmitter<'a, F>
where
    F: FnMut(RecordBatch),
{
    schema: Arc<Schema>,
    on_batch: &'a mut F,
    error: Option<Error>,
    emitted: bool,
}

impl<'a, F> BuilderBatchEmitter<'a, F>
where
    F: FnMut(RecordBatch),
{
    fn emit_array(&mut self, array: ArrayRef) {
        if self.error.is_some() {
            return;
        }
        match RecordBatch::try_new(self.schema.clone(), vec![array]) {
            Ok(batch) => {
                (self.on_batch)(batch);
                self.emitted = true;
            }
            Err(err) => self.error = Some(Error::from(err)),
        }
    }

    fn emit_chunk<A>(&mut self, array: &A)
    where
        A: Array + Clone + 'static,
    {
        let array_ref: ArrayRef = Arc::new(array.clone()) as ArrayRef;
        self.emit_array(array_ref);
    }

    fn emit_slice<A>(&mut self, array: &A, start: usize, len: usize)
    where
        A: Array + Clone + 'static,
    {
        let slice = array.slice(start, len);
        self.emit_array(slice);
    }

    fn unsupported(&mut self) {
        if self.error.is_none() {
            self.error = Some(Error::Internal(
                "scan builder emitted unsupported row-id payload".into(),
            ));
        }
    }
}

macro_rules! impl_emit_chunk {
    ($method:ident, $ArrayTy:ty) => {
        fn $method(&mut self, array: &$ArrayTy) {
            self.emit_chunk(array);
        }
    };
}

macro_rules! impl_emit_run {
    ($method:ident, $ArrayTy:ty) => {
        fn $method(&mut self, array: &$ArrayTy, start: usize, len: usize) {
            self.emit_slice(array, start, len);
        }
    };
}

impl<'a, F> PrimitiveVisitor for BuilderBatchEmitter<'a, F>
where
    F: FnMut(RecordBatch),
{
    impl_emit_chunk!(u64_chunk, UInt64Array);
    impl_emit_chunk!(u32_chunk, UInt32Array);
    impl_emit_chunk!(u16_chunk, UInt16Array);
    impl_emit_chunk!(u8_chunk, UInt8Array);
    impl_emit_chunk!(i64_chunk, Int64Array);
    impl_emit_chunk!(i32_chunk, Int32Array);
    impl_emit_chunk!(i16_chunk, Int16Array);
    impl_emit_chunk!(i8_chunk, Int8Array);
    impl_emit_chunk!(f64_chunk, Float64Array);
    impl_emit_chunk!(f32_chunk, Float32Array);
}

impl<'a, F> PrimitiveWithRowIdsVisitor for BuilderBatchEmitter<'a, F>
where
    F: FnMut(RecordBatch),
{
    fn u64_chunk_with_rids(&mut self, _v: &UInt64Array, _r: &UInt64Array) {
        self.unsupported();
    }
    fn u32_chunk_with_rids(&mut self, _v: &UInt32Array, _r: &UInt64Array) {
        self.unsupported();
    }
    fn u16_chunk_with_rids(&mut self, _v: &UInt16Array, _r: &UInt64Array) {
        self.unsupported();
    }
    fn u8_chunk_with_rids(&mut self, _v: &UInt8Array, _r: &UInt64Array) {
        self.unsupported();
    }
    fn i64_chunk_with_rids(&mut self, _v: &Int64Array, _r: &UInt64Array) {
        self.unsupported();
    }
    fn i32_chunk_with_rids(&mut self, _v: &Int32Array, _r: &UInt64Array) {
        self.unsupported();
    }
    fn i16_chunk_with_rids(&mut self, _v: &Int16Array, _r: &UInt64Array) {
        self.unsupported();
    }
    fn i8_chunk_with_rids(&mut self, _v: &Int8Array, _r: &UInt64Array) {
        self.unsupported();
    }
    fn f64_chunk_with_rids(&mut self, _v: &Float64Array, _r: &UInt64Array) {
        self.unsupported();
    }
    fn f32_chunk_with_rids(&mut self, _v: &Float32Array, _r: &UInt64Array) {
        self.unsupported();
    }
}

impl<'a, F> PrimitiveSortedVisitor for BuilderBatchEmitter<'a, F>
where
    F: FnMut(RecordBatch),
{
    impl_emit_run!(u64_run, UInt64Array);
    impl_emit_run!(u32_run, UInt32Array);
    impl_emit_run!(u16_run, UInt16Array);
    impl_emit_run!(u8_run, UInt8Array);
    impl_emit_run!(i64_run, Int64Array);
    impl_emit_run!(i32_run, Int32Array);
    impl_emit_run!(i16_run, Int16Array);
    impl_emit_run!(i8_run, Int8Array);
    impl_emit_run!(f64_run, Float64Array);
    impl_emit_run!(f32_run, Float32Array);
}

impl<'a, F> PrimitiveSortedWithRowIdsVisitor for BuilderBatchEmitter<'a, F>
where
    F: FnMut(RecordBatch),
{
    fn u64_run_with_rids(
        &mut self,
        _v: &UInt64Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
        self.unsupported();
    }
    fn u32_run_with_rids(
        &mut self,
        _v: &UInt32Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
        self.unsupported();
    }
    fn u16_run_with_rids(
        &mut self,
        _v: &UInt16Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
        self.unsupported();
    }
    fn u8_run_with_rids(&mut self, _v: &UInt8Array, _r: &UInt64Array, _start: usize, _len: usize) {
        self.unsupported();
    }
    fn i64_run_with_rids(&mut self, _v: &Int64Array, _r: &UInt64Array, _start: usize, _len: usize) {
        self.unsupported();
    }
    fn i32_run_with_rids(&mut self, _v: &Int32Array, _r: &UInt64Array, _start: usize, _len: usize) {
        self.unsupported();
    }
    fn i16_run_with_rids(&mut self, _v: &Int16Array, _r: &UInt64Array, _start: usize, _len: usize) {
        self.unsupported();
    }
    fn i8_run_with_rids(&mut self, _v: &Int8Array, _r: &UInt64Array, _start: usize, _len: usize) {
        self.unsupported();
    }
    fn f64_run_with_rids(
        &mut self,
        _v: &Float64Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
        self.unsupported();
    }
    fn f32_run_with_rids(
        &mut self,
        _v: &Float32Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
        self.unsupported();
    }
    fn null_run(&mut self, _r: &UInt64Array, _start: usize, _len: usize) {
        self.unsupported();
    }
}
