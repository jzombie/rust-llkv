use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, BooleanBuilder, Int64Array};
use arrow::compute;
use arrow::datatypes::DataType;
use croaring::Treemap;
use llkv_compute::analysis::{PredicateFusionCache, scalar_expr_contains_coalesce};
use llkv_compute::compute_compare;
use llkv_compute::eval::ScalarEvaluator;
use llkv_compute::kernels;
use llkv_compute::program::{
    DomainOp, DomainProgramId, EvalOp, OwnedFilter, OwnedOperator, ProgramSet,
};
use llkv_expr::literal::Literal;
use llkv_expr::{BinaryOp, CompareOp, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};
use llkv_types::{FieldId, LogicalFieldId, ROW_ID_FIELD_ID};
use llkv_threading::with_thread_pool;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::row_stream::RowIdSource;
use crate::{NumericArrayMap, ScanStorage};
use llkv_column_map::store::{GatherNullPolicy, MultiGatherContext};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

// Chunk size for predicate evaluation. Smaller chunks create more parallel tasks.
const CHUNK_SIZE: usize = 4096;

/// Evaluate a compiled predicate program against storage to produce RowIdSource.
pub fn collect_row_ids_for_program<'expr, P, S>(
    storage: &S,
    programs: &ProgramSet<'expr>,
    fusion_cache: &PredicateFusionCache,
    all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
) -> LlkvResult<RowIdSource>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    let mut stack: Vec<RowIdSource> = Vec::new();

    let eval_ops = programs.eval_ops();

    for op in eval_ops {
        match op {
            EvalOp::PushPredicate(filter) => {
                stack.push(RowIdSource::Bitmap(storage.filter_leaf(filter)?));
            }
            EvalOp::PushCompare { left, op, right } => {
                let rows = collect_row_ids_for_compare(storage, left, *op, right, all_rows_cache)?;
                stack.push(RowIdSource::Bitmap(rows));
            }
            EvalOp::PushInList {
                expr,
                list,
                negated,
            } => {
                let rows = collect_row_ids_for_in_list(
                    storage,
                    expr,
                    list.as_slice(),
                    *negated,
                    all_rows_cache,
                )?;
                stack.push(RowIdSource::Bitmap(rows));
            }
            EvalOp::PushIsNull { expr, negated } => {
                let rows = collect_row_ids_for_is_null(storage, expr, *negated, all_rows_cache)?;
                stack.push(RowIdSource::Bitmap(rows));
            }
            EvalOp::PushLiteral(value) => {
                if *value {
                    stack.push(RowIdSource::Bitmap(collect_all_row_ids(
                        storage,
                        all_rows_cache,
                    )?));
                } else {
                    stack.push(RowIdSource::Bitmap(Treemap::new()));
                }
            }
            EvalOp::FusedAnd { field_id, filters } => {
                let rows = storage.filter_fused(*field_id, filters.as_slice(), fusion_cache)?;
                stack.push(rows);
            }
            EvalOp::And { child_count } => {
                if *child_count == 0 {
                    return Err(Error::Internal("AND opcode requires operands".into()));
                }
                let mut acc = stack
                    .pop()
                    .ok_or_else(|| Error::Internal("AND opcode underflow".into()))?;
                for _ in 1..*child_count {
                    let next = stack
                        .pop()
                        .ok_or_else(|| Error::Internal("AND opcode underflow".into()))?;

                    let acc_empty = match &acc {
                        RowIdSource::Bitmap(b) => b.is_empty(),
                        RowIdSource::Vector(v) => v.is_empty(),
                    };

                    if acc_empty {
                        // Short circuit: AND with empty is empty.
                        continue;
                    }

                    acc = match (acc, next) {
                        (RowIdSource::Bitmap(a), RowIdSource::Bitmap(b)) => {
                            RowIdSource::Bitmap(a & b)
                        }
                        (RowIdSource::Bitmap(a), RowIdSource::Vector(b)) => {
                            // Intersect bitmap with vector.
                            // For now, convert vector to bitmap.
                            let b_map = Treemap::from_iter(b);
                            RowIdSource::Bitmap(a & b_map)
                        }
                        (RowIdSource::Vector(a), RowIdSource::Bitmap(b)) => {
                            let a_map = Treemap::from_iter(a);
                            RowIdSource::Bitmap(a_map & b)
                        }
                        (RowIdSource::Vector(a), RowIdSource::Vector(b)) => {
                            // Intersect two sorted vectors.
                            // TODO: Optimize this.
                            let a_map = Treemap::from_iter(a);
                            let b_map = Treemap::from_iter(b);
                            RowIdSource::Bitmap(a_map & b_map)
                        }
                    };
                }
                stack.push(acc);
            }
            EvalOp::Or { child_count } => {
                if *child_count == 0 {
                    return Err(Error::Internal("OR opcode requires operands".into()));
                }
                let mut acc = stack
                    .pop()
                    .ok_or_else(|| Error::Internal("OR opcode underflow".into()))?;
                for _ in 1..*child_count {
                    let next = stack
                        .pop()
                        .ok_or_else(|| Error::Internal("OR opcode underflow".into()))?;

                    acc = match (acc, next) {
                        (RowIdSource::Bitmap(a), RowIdSource::Bitmap(b)) => {
                            RowIdSource::Bitmap(a | b)
                        }
                        (RowIdSource::Bitmap(a), RowIdSource::Vector(b)) => {
                            let b_map = Treemap::from_iter(b);
                            RowIdSource::Bitmap(a | b_map)
                        }
                        (RowIdSource::Vector(a), RowIdSource::Bitmap(b)) => {
                            let a_map = Treemap::from_iter(a);
                            RowIdSource::Bitmap(a_map | b)
                        }
                        (RowIdSource::Vector(a), RowIdSource::Vector(b)) => {
                            let a_map = Treemap::from_iter(a);
                            let b_map = Treemap::from_iter(b);
                            RowIdSource::Bitmap(a_map | b_map)
                        }
                    };
                }
                stack.push(acc);
            }
            EvalOp::Not { domain } => {
                let operand = stack
                    .pop()
                    .ok_or_else(|| Error::Internal("NOT opcode underflow".into()))?;

                let mut domain_cache = FxHashMap::default();
                let domain_rows = evaluate_domain_program(
                    storage,
                    programs,
                    *domain,
                    all_rows_cache,
                    &mut domain_cache,
                )?;

                let result = match operand {
                    RowIdSource::Bitmap(b) => domain_rows.as_ref() - &b,
                    RowIdSource::Vector(v) => domain_rows.as_ref() - &Treemap::from_iter(v),
                };
                stack.push(RowIdSource::Bitmap(result));
            }
        }
    }

    stack
        .pop()
        .ok_or_else(|| Error::Internal("Program stack empty after evaluation".into()))
}

fn collect_all_row_ids<P, S>(
    storage: &S,
    all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
) -> LlkvResult<Treemap>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    if let Some(rows) = all_rows_cache.get(&ROW_ID_FIELD_ID) {
        return Ok(rows.clone());
    }

    let rows = storage.all_row_ids()?;
    all_rows_cache.insert(ROW_ID_FIELD_ID, rows.clone());
    Ok(rows)
}

fn collect_all_row_ids_for_field<P, S>(
    storage: &S,
    field_id: FieldId,
    all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
) -> LlkvResult<Treemap>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    if let Some(rows) = all_rows_cache.get(&field_id) {
        return Ok(rows.clone());
    }

    if field_id == ROW_ID_FIELD_ID {
        let rows = collect_all_row_ids(storage, all_rows_cache)?;
        all_rows_cache.insert(field_id, rows.clone());
        return Ok(rows);
    }

    let filter = OwnedFilter {
        field_id,
        op: OwnedOperator::IsNotNull,
    };
    let mut rows = storage.filter_leaf(&filter)?;

    let null_filter = OwnedFilter {
        field_id,
        op: OwnedOperator::IsNull,
    };
    if let Ok(null_rows) = storage.filter_leaf(&null_filter) {
        rows |= null_rows;
    }

    all_rows_cache.insert(field_id, rows.clone());
    Ok(rows)
}

fn collect_row_ids_for_is_null<P, S>(
    storage: &S,
    expr: &ScalarExpr<FieldId>,
    negated: bool,
    all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
) -> LlkvResult<Treemap>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    if let ScalarExpr::Column(fid) = expr {
        let op = if negated {
            OwnedOperator::IsNotNull
        } else {
            OwnedOperator::IsNull
        };
        let filter = OwnedFilter { field_id: *fid, op };
        return storage.filter_leaf(&filter);
    }

    let mut fields = FxHashSet::default();
    ScalarEvaluator::collect_fields(expr, &mut fields);

    let mut ordered_fields: Vec<FieldId> = fields.into_iter().collect();
    ordered_fields.sort_unstable();
    ordered_fields.dedup();

    let domain = if ordered_fields.is_empty() {
        let arrays: NumericArrayMap = FxHashMap::default();
        let value = ScalarEvaluator::evaluate_value(expr, 0, &arrays)?;
        let is_null = value.data_type() == &DataType::Null || value.is_null(0);
        if (is_null && !negated) || (!is_null && negated) {
            return collect_all_row_ids(storage, all_rows_cache);
        } else {
            return Ok(Treemap::new());
        }
    } else {
        let mut union_rows = Treemap::new();
        for fid in &ordered_fields {
            let rows = collect_all_row_ids_for_field(storage, *fid, all_rows_cache)?;
            union_rows |= rows;
        }
        union_rows
    };

    if domain.is_empty() {
        return Ok(Treemap::new());
    }

    let mut result = Treemap::new();
    let logical_fields: Vec<LogicalFieldId> = ordered_fields
        .iter()
        .map(|&fid| LogicalFieldId::for_user(storage.table_id(), fid))
        .collect();

    let mut ctx = storage.prepare_gather_context(&logical_fields)?;

    let domain_vec: Vec<u64> = domain.iter().collect();
    for chunk in domain_vec.chunks(CHUNK_SIZE) {
        let batch = storage.gather_row_window_with_context(
            &logical_fields,
            chunk,
            GatherNullPolicy::IncludeNulls,
            Some(&mut ctx),
        )?;

        let mut arrays: NumericArrayMap = FxHashMap::default();
        for (i, fid) in ordered_fields.iter().enumerate() {
            arrays.insert(*fid, batch.column(i).clone());
        }

        let value_array = ScalarEvaluator::evaluate_batch(expr, chunk.len(), &arrays)?;

        for (i, row_id) in chunk.iter().enumerate() {
            let is_null = value_array.is_null(i);
            if (is_null && !negated) || (!is_null && negated) {
                result.add(*row_id);
            }
        }
    }

    Ok(result)
}

fn collect_row_ids_for_compare<P, S>(
    storage: &S,
    left: &ScalarExpr<FieldId>,
    op: CompareOp,
    right: &ScalarExpr<FieldId>,
    all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
) -> LlkvResult<Treemap>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    if let Some(filter) = simple_compare_filter(left, op, right) {
        return storage.filter_leaf(&filter);
    }

    let mut fields = FxHashSet::default();
    ScalarEvaluator::collect_fields(left, &mut fields);
    ScalarEvaluator::collect_fields(right, &mut fields);

    if fields.is_empty() {
        return match evaluate_constant_compare(left, op, right)? {
            Some(true) => collect_all_row_ids(storage, all_rows_cache),
            _ => Ok(Treemap::new()),
        };
    }

    let mut ordered_fields: Vec<FieldId> = fields.into_iter().collect();
    ordered_fields.sort_unstable();
    ordered_fields.dedup();

    let requires_full_scan =
        scalar_expr_contains_coalesce(left) || scalar_expr_contains_coalesce(right);

    let domain = if requires_full_scan {
        let mut union_rows = Treemap::new();
        for fid in &ordered_fields {
            let rows = collect_all_row_ids_for_field(storage, *fid, all_rows_cache)?;
            union_rows |= rows;
        }
        union_rows
    } else {
        let mut domain: Option<Treemap> = None;
        for fid in &ordered_fields {
            let rows = collect_all_row_ids_for_field(storage, *fid, all_rows_cache)?;
            domain = Some(match domain {
                Some(existing) => existing & rows,
                None => rows,
            });
            if let Some(ref d) = domain
                && d.is_empty()
            {
                return Ok(Treemap::new());
            }
        }
        domain.unwrap_or_default()
    };

    if domain.is_empty() {
        return Ok(Treemap::new());
    }

    let (matched, _) = evaluate_compare_rows(storage, &ordered_fields, &domain, left, op, right)?;
    Ok(matched)
}

fn collect_row_ids_for_in_list<P, S>(
    storage: &S,
    expr: &ScalarExpr<FieldId>,
    list: &[ScalarExpr<FieldId>],
    negated: bool,
    all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
) -> LlkvResult<Treemap>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    let mut fields = FxHashSet::default();
    ScalarEvaluator::collect_fields(expr, &mut fields);
    for item in list {
        ScalarEvaluator::collect_fields(item, &mut fields);
    }

    if fields.is_empty() {
        let (matched, _) = evaluate_constant_in_list(storage, expr, list, negated, all_rows_cache)?;
        return Ok(matched);
    }

    let mut ordered_fields: Vec<FieldId> = fields.into_iter().collect();
    ordered_fields.sort_unstable();

    let mut domain: Option<Treemap> = None;
    for fid in &ordered_fields {
        let rows = collect_all_row_ids_for_field(storage, *fid, all_rows_cache)?;
        domain = Some(match domain {
            Some(existing) => existing & rows,
            None => rows,
        });
    }
    let domain = domain.unwrap_or_default();

    if domain.is_empty() {
        return Ok(Treemap::new());
    }

    let (matched, _) =
        evaluate_in_list_over_rows(storage, &domain, &ordered_fields, expr, list, negated)?;
    Ok(matched)
}

fn evaluate_in_list_over_rows<P, S>(
    storage: &S,
    domain_rows: &Treemap,
    ordered_fields: &[FieldId],
    expr: &ScalarExpr<FieldId>,
    list: &[ScalarExpr<FieldId>],
    negated: bool,
) -> LlkvResult<(Treemap, Treemap)>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    if domain_rows.is_empty() {
        return Ok((Treemap::new(), Treemap::new()));
    }

    let has_row_id = ordered_fields.contains(&ROW_ID_FIELD_ID);

    let physical_fields: Vec<FieldId> = ordered_fields
        .iter()
        .copied()
        .filter(|fid| *fid != ROW_ID_FIELD_ID)
        .collect();

    let logical_fields: Vec<LogicalFieldId> = physical_fields
        .iter()
        .map(|&fid| LogicalFieldId::for_user(storage.table_id(), fid))
        .collect();

    let domain_vec: Vec<u64> = domain_rows.iter().collect();
    let (local_matched, local_determined, _): (Treemap, Treemap, Option<MultiGatherContext>) =
        with_thread_pool(|| {
            domain_vec
                .par_chunks(CHUNK_SIZE)
                .try_fold(
                    || (Treemap::new(), Treemap::new(), None),
                    |(mut matched, mut determined, mut ctx), chunk| -> LlkvResult<_> {
                        if ctx.is_none() {
                            ctx = Some(storage.prepare_gather_context(&logical_fields)?);
                        }
                        let batch = storage.gather_row_window_with_context(
                            &logical_fields,
                            chunk,
                            GatherNullPolicy::IncludeNulls,
                            ctx.as_mut(),
                        )?;

                        let mut arrays: NumericArrayMap = FxHashMap::default();
                        for (i, fid) in physical_fields.iter().enumerate() {
                            arrays.insert(*fid, batch.column(i).clone());
                        }

                        if has_row_id {
                            let rid_values: Vec<i64> =
                                chunk.iter().map(|row_id| *row_id as i64).collect();
                            let rid_array: ArrayRef = Arc::new(Int64Array::from(rid_values));
                            arrays.insert(ROW_ID_FIELD_ID, rid_array);
                        }

                        let mut target_array =
                            ScalarEvaluator::evaluate_batch(expr, chunk.len(), &arrays)?;
                        let mut acc: Option<BooleanArray> = None;

                        for item in list {
                            let value_array =
                                ScalarEvaluator::evaluate_batch(item, chunk.len(), &arrays)?;
                            let (new_target, new_value) =
                                kernels::coerce_types(&target_array, &value_array, BinaryOp::Add)?;
                            target_array = new_target;
                            let eq_array =
                                arrow::compute::kernels::cmp::eq(&new_value, &target_array)?;
                            acc = match acc {
                                Some(prev) => Some(compute::or_kleene(&prev, &eq_array)?),
                                None => Some(eq_array),
                            };
                        }

                        let mut final_bool = match acc {
                            Some(arr) => arr,
                            None => {
                                let mut builder = BooleanBuilder::with_capacity(chunk.len());
                                for _ in 0..chunk.len() {
                                    builder.append_value(false);
                                }
                                builder.finish()
                            }
                        };

                        if negated {
                            final_bool = compute::not(&final_bool)?;
                        }

                        for (idx, row_id) in chunk.iter().enumerate() {
                            if final_bool.is_null(idx) {
                                continue;
                            }
                            determined.add(*row_id);
                            if final_bool.value(idx) {
                                matched.add(*row_id);
                            }
                        }

                        Ok((matched, determined, ctx))
                    },
                )
                .try_reduce(
                    || (Treemap::new(), Treemap::new(), None),
                    |a, b| {
                        let (mut am, mut ad, actx) = a;
                        let (bm, bd, _bctx) = b;
                        am |= bm;
                        ad |= bd;
                        Ok((am, ad, actx))
                    },
                )
        })?;

    Ok((local_matched, local_determined))
}

fn evaluate_compare_rows<P, S>(
    storage: &S,
    ordered_fields: &[FieldId],
    domain_rows: &Treemap,
    left: &ScalarExpr<FieldId>,
    op: CompareOp,
    right: &ScalarExpr<FieldId>,
) -> LlkvResult<(Treemap, Treemap)>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    if domain_rows.is_empty() {
        return Ok((Treemap::new(), Treemap::new()));
    }

    let has_row_id = ordered_fields.contains(&ROW_ID_FIELD_ID);

    let physical_fields: Vec<FieldId> = ordered_fields
        .iter()
        .copied()
        .filter(|fid| *fid != ROW_ID_FIELD_ID)
        .collect();

    let logical_fields: Vec<LogicalFieldId> = physical_fields
        .iter()
        .map(|&fid| LogicalFieldId::for_user(storage.table_id(), fid))
        .collect();

    let domain_vec: Vec<u64> = domain_rows.iter().collect();
    let (matched, determined, _): (Treemap, Treemap, Option<MultiGatherContext>) =
        with_thread_pool(|| {
            domain_vec
                .par_chunks(CHUNK_SIZE)
                .try_fold(
                    || (Treemap::new(), Treemap::new(), None),
                    |(mut matched, mut determined, mut ctx), chunk| -> LlkvResult<_> {
                        if ctx.is_none() {
                            ctx = Some(storage.prepare_gather_context(&logical_fields)?);
                        }

                        let batch = storage.gather_row_window_with_context(
                            &logical_fields,
                            chunk,
                            GatherNullPolicy::IncludeNulls,
                            ctx.as_mut(),
                        )?;

                        let mut arrays: NumericArrayMap = FxHashMap::default();
                        for (i, fid) in physical_fields.iter().enumerate() {
                            arrays.insert(*fid, batch.column(i).clone());
                        }

                        if has_row_id {
                            let rid_values: Vec<i64> =
                                chunk.iter().map(|row_id| *row_id as i64).collect();
                            let rid_array: ArrayRef = Arc::new(Int64Array::from(rid_values));
                            arrays.insert(ROW_ID_FIELD_ID, rid_array);
                        }

                        let left_vals =
                            ScalarEvaluator::evaluate_batch(left, chunk.len(), &arrays)?;
                        let right_vals =
                            ScalarEvaluator::evaluate_batch(right, chunk.len(), &arrays)?;

                        let cmp_array = compute_compare(&left_vals, op, &right_vals)?;
                        let cmp = cmp_array
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                            .ok_or_else(|| {
                                Error::Internal("compare kernel did not return booleans".into())
                            })?;

                        for (idx, row_id) in chunk.iter().enumerate() {
                            if !left_vals.is_null(idx) && !right_vals.is_null(idx) {
                                determined.add(*row_id);
                            }
                            if cmp.is_null(idx) {
                                continue;
                            }
                            if cmp.value(idx) {
                                matched.add(*row_id);
                            }
                        }

                        Ok((matched, determined, ctx))
                    },
                )
                .try_reduce(
                    || (Treemap::new(), Treemap::new(), None),
                    |a, b| {
                        let (mut am, mut ad, actx) = a;
                        let (bm, bd, _bctx) = b;
                        am |= bm;
                        ad |= bd;
                        Ok((am, ad, actx))
                    },
                )
        })?;

    Ok((matched, determined))
}

fn evaluate_domain_program<P, S>(
    storage: &S,
    programs: &ProgramSet<'_>,
    domain_id: DomainProgramId,
    all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
    cache: &mut FxHashMap<DomainProgramId, Arc<Treemap>>,
) -> LlkvResult<Arc<Treemap>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    if let Some(rows) = cache.get(&domain_id) {
        return Ok(Arc::clone(rows));
    }

    let program = programs
        .domain(domain_id)
        .ok_or_else(|| Error::Internal(format!("missing domain program {domain_id:?}")))?;

    let mut stack: Vec<Treemap> = Vec::new();
    for op in program.ops() {
        match op {
            DomainOp::PushFieldAll(field_id) => {
                stack.push(collect_all_row_ids_for_field(
                    storage,
                    *field_id,
                    all_rows_cache,
                )?);
            }
            DomainOp::PushCompareDomain {
                left,
                right,
                op,
                fields,
            } => {
                if scalar_expr_constant_null(left)? || scalar_expr_constant_null(right)? {
                    stack.push(Treemap::new());
                    continue;
                }
                let rows =
                    collect_compare_domain_rows(storage, left, right, *op, fields, all_rows_cache)?;
                stack.push(rows);
            }
            DomainOp::PushInListDomain {
                expr,
                list,
                fields,
                negated,
            } => {
                if scalar_expr_constant_null(expr)? || list_all_constant_null(list)? {
                    stack.push(Treemap::new());
                    continue;
                }
                let rows = collect_in_list_domain_rows(
                    storage,
                    expr,
                    list,
                    *negated,
                    fields,
                    all_rows_cache,
                )?;
                stack.push(rows);
            }
            DomainOp::Union { child_count } => {
                let mut acc = stack.pop().unwrap();
                for _ in 1..*child_count {
                    let next = stack.pop().unwrap();
                    acc |= next;
                }
                stack.push(acc);
            }
            DomainOp::Intersect { child_count } => {
                let mut acc = stack.pop().unwrap();
                for _ in 1..*child_count {
                    let next = stack.pop().unwrap();
                    acc &= next;
                }
                stack.push(acc);
            }
            DomainOp::PushIsNullDomain {
                expr: _expr,
                fields,
                negated: _negated,
            } => {
                let mut ordered_fields: Vec<FieldId> = fields.to_vec();
                ordered_fields.sort_unstable();
                ordered_fields.dedup();

                if ordered_fields.is_empty() {
                    stack.push(collect_all_row_ids(storage, all_rows_cache)?);
                    continue;
                }

                let mut domain: Option<Treemap> = None;
                for &fid in &ordered_fields {
                    let rows = collect_all_row_ids_for_field(storage, fid, all_rows_cache)?;
                    domain = Some(match domain {
                        Some(existing) => existing & rows,
                        None => rows,
                    });
                }
                stack.push(domain.unwrap_or_default());
            }
            DomainOp::PushAllRows => {
                stack.push(collect_all_row_ids(storage, all_rows_cache)?);
            }
        }
    }

    let result = Arc::new(stack.pop().unwrap_or_default());
    cache.insert(domain_id, Arc::clone(&result));
    Ok(result)
}

fn collect_compare_domain_rows<P, S>(
    storage: &S,
    left: &ScalarExpr<FieldId>,
    right: &ScalarExpr<FieldId>,
    op: CompareOp,
    fields: &[FieldId],
    all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
) -> LlkvResult<Treemap>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    if fields.is_empty() {
        return match evaluate_constant_compare(left, op, right)? {
            Some(_) => collect_all_row_ids(storage, all_rows_cache),
            None => Ok(Treemap::new()),
        };
    }

    let mut ordered_fields: Vec<FieldId> = fields.to_vec();
    ordered_fields.sort_unstable();
    ordered_fields.dedup();

    let mut domain: Option<Treemap> = None;
    for &fid in &ordered_fields {
        let rows = collect_all_row_ids_for_field(storage, fid, all_rows_cache)?;
        domain = Some(match domain {
            Some(existing) => existing & rows,
            None => rows,
        });
    }
    let domain_rows = domain.unwrap_or_default();
    if domain_rows.is_empty() {
        return Ok(Treemap::new());
    }

    let (_, determined) =
        evaluate_compare_rows(storage, &ordered_fields, &domain_rows, left, op, right)?;
    Ok(determined)
}

fn collect_in_list_domain_rows<P, S>(
    storage: &S,
    expr: &ScalarExpr<FieldId>,
    list: &[ScalarExpr<FieldId>],
    negated: bool,
    fields: &[FieldId],
    all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
) -> LlkvResult<Treemap>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    if fields.is_empty() {
        let (_, determined) =
            evaluate_constant_in_list(storage, expr, list, negated, all_rows_cache)?;
        return Ok(determined);
    }

    let mut ordered_fields: Vec<FieldId> = fields.to_vec();
    ordered_fields.sort_unstable();
    ordered_fields.dedup();

    let mut domain: Option<Treemap> = None;
    for &fid in &ordered_fields {
        let rows = collect_all_row_ids_for_field(storage, fid, all_rows_cache)?;
        domain = Some(match domain {
            Some(existing) => existing & rows,
            None => rows,
        });
    }
    let domain_rows = domain.unwrap_or_default();
    if domain_rows.is_empty() {
        return Ok(Treemap::new());
    }

    let (_, determined) =
        evaluate_in_list_over_rows(storage, &domain_rows, &ordered_fields, expr, list, negated)?;
    Ok(determined)
}

fn scalar_expr_constant_null(expr: &ScalarExpr<FieldId>) -> LlkvResult<bool> {
    let mut fields = FxHashSet::default();
    ScalarEvaluator::collect_fields(expr, &mut fields);
    if !fields.is_empty() {
        return Ok(false);
    }

    match ScalarEvaluator::evaluate_constant_literal_expr(expr)? {
        Some(Literal::Null) | None => Ok(true),
        _ => Ok(false),
    }
}

fn list_all_constant_null(list: &[ScalarExpr<FieldId>]) -> LlkvResult<bool> {
    if list.is_empty() {
        return Ok(false);
    }

    for item in list {
        if !scalar_expr_constant_null(item)? {
            return Ok(false);
        }
    }

    Ok(true)
}

fn evaluate_constant_compare(
    left: &ScalarExpr<FieldId>,
    op: CompareOp,
    right: &ScalarExpr<FieldId>,
) -> LlkvResult<Option<bool>> {
    let arrays: NumericArrayMap = FxHashMap::default();
    let left_value = ScalarEvaluator::evaluate_value(left, 0, &arrays)?;
    let right_value = ScalarEvaluator::evaluate_value(right, 0, &arrays)?;

    let cmp_array = compute_compare(&left_value, op, &right_value)?;
    let bool_array = cmp_array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| Error::Internal("compare kernel did not return bools".into()))?;

    if bool_array.is_null(0) {
        Ok(None)
    } else {
        Ok(Some(bool_array.value(0)))
    }
}

fn evaluate_constant_in_list<P, S>(
    storage: &S,
    expr: &ScalarExpr<FieldId>,
    list: &[ScalarExpr<FieldId>],
    negated: bool,
    all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
) -> LlkvResult<(Treemap, Treemap)>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    let arrays: NumericArrayMap = FxHashMap::default();
    let mut target_array = ScalarEvaluator::evaluate_value(expr, 0, &arrays)?;

    if target_array.data_type() == &DataType::Null || target_array.is_null(0) {
        return Ok((Treemap::new(), Treemap::new()));
    }

    let mut matched = false;
    let mut saw_null = false;

    for value_expr in list {
        let value_array = ScalarEvaluator::evaluate_value(value_expr, 0, &arrays)?;
        if value_array.data_type() == &DataType::Null || value_array.is_null(0) {
            saw_null = true;
            continue;
        }

        let (new_target, new_value) =
            kernels::coerce_types(&target_array, &value_array, BinaryOp::Add)?;
        target_array = new_target;
        let cmp_array = arrow::compute::kernels::cmp::eq(&new_value, &target_array)?;
        let bool_array = cmp_array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| Error::Internal("in-list compare did not return bools".into()))?;

        if bool_array.value(0) {
            matched = true;
            break;
        }
    }

    let rows = collect_all_row_ids(storage, all_rows_cache)?;
    let outcome = if matched {
        Some(!negated)
    } else if saw_null {
        None
    } else if negated {
        Some(true)
    } else {
        Some(false)
    };

    match outcome {
        Some(true) => Ok((rows.clone(), rows)),
        Some(false) => Ok((Treemap::new(), rows)),
        None => Ok((Treemap::new(), Treemap::new())),
    }
}

fn simple_compare_filter(
    left: &ScalarExpr<FieldId>,
    op: CompareOp,
    right: &ScalarExpr<FieldId>,
) -> Option<OwnedFilter> {
    match (left, right) {
        (ScalarExpr::Column(field_id), ScalarExpr::Literal(lit)) => {
            compare_op_to_owned(*field_id, op, lit)
        }
        (ScalarExpr::Literal(lit), ScalarExpr::Column(field_id)) => {
            let flipped = match op {
                CompareOp::Eq => Some(CompareOp::Eq),
                CompareOp::NotEq => None,
                CompareOp::Lt => Some(CompareOp::Gt),
                CompareOp::LtEq => Some(CompareOp::GtEq),
                CompareOp::Gt => Some(CompareOp::Lt),
                CompareOp::GtEq => Some(CompareOp::LtEq),
            }?;
            compare_op_to_owned(*field_id, flipped, lit)
        }
        _ => None,
    }
}

fn compare_op_to_owned(field_id: FieldId, op: CompareOp, literal: &Literal) -> Option<OwnedFilter> {
    if matches!(literal, Literal::Null) {
        // Null comparisons propagate null, so fall back to row-wise evaluation.
        return None;
    }

    let op = match op {
        CompareOp::Eq => OwnedOperator::Equals(literal.clone()),
        CompareOp::NotEq => return None,
        CompareOp::Lt => OwnedOperator::LessThan(literal.clone()),
        CompareOp::LtEq => OwnedOperator::LessThanOrEquals(literal.clone()),
        CompareOp::Gt => OwnedOperator::GreaterThan(literal.clone()),
        CompareOp::GtEq => OwnedOperator::GreaterThanOrEquals(literal.clone()),
    };

    Some(OwnedFilter { field_id, op })
}
