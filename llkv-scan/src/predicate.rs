use std::sync::Arc;

use arrow::array::{Array, BooleanArray, BooleanBuilder};
use arrow::datatypes::DataType;
use arrow::compute;
use croaring::Treemap;
use llkv_compute::analysis::PredicateFusionCache;
use llkv_compute::compute_compare;
use llkv_compute::eval::ScalarEvaluator;
use llkv_compute::kernels;
use llkv_compute::program::{
    DomainOp, DomainProgramId, EvalOp, OwnedFilter, OwnedOperator, ProgramSet,
};
use llkv_expr::literal::Literal;
use llkv_expr::{BinaryOp, CompareOp, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};
use llkv_types::{FieldId, LogicalFieldId};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::row_stream::RowIdSource;
use crate::{NumericArrayMap, ScanStorage};
use llkv_column_map::store::GatherNullPolicy;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

const CHUNK_SIZE: usize = 8192;

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
    _all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
) -> LlkvResult<Treemap>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    storage.all_row_ids()
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

    let filter = OwnedFilter {
        field_id,
        op: OwnedOperator::IsNotNull,
    };
    let rows = storage.filter_leaf(&filter)?;
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

    let mut domain: Option<Treemap> = None;
    for fid in &ordered_fields {
        let rows = collect_all_row_ids_for_field(storage, *fid, all_rows_cache)?;
        domain = Some(match domain {
            Some(existing) => existing & rows,
            None => rows,
        });
        if let Some(ref d) = domain {
            if d.is_empty() {
                return Ok(Treemap::new());
            }
        }
    }
    let domain = domain.unwrap_or_default();

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

        let left_val = ScalarEvaluator::evaluate_batch(left, chunk.len(), &arrays)?;
        let right_val = ScalarEvaluator::evaluate_batch(right, chunk.len(), &arrays)?;

        if let Ok(cmp_array) = compute_compare(&left_val, op, &right_val) {
            let cmp = cmp_array.as_any().downcast_ref::<BooleanArray>().unwrap();
            for (i, valid) in cmp.iter().enumerate() {
                if valid.unwrap_or(false) {
                    result.add(chunk[i]);
                }
            }
        }
    }

    Ok(result)
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
        return match evaluate_constant_in_list(expr, list, negated)? {
            Some(true) => collect_all_row_ids(storage, all_rows_cache),
            _ => Ok(Treemap::new()),
        };
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

        let mut target_array = ScalarEvaluator::evaluate_batch(expr, chunk.len(), &arrays)?;
        let mut acc: Option<BooleanArray> = None;

        for item in list {
            let value_array = ScalarEvaluator::evaluate_batch(item, chunk.len(), &arrays)?;
            let (new_target, new_value) =
                kernels::coerce_types(&target_array, &value_array, BinaryOp::Add)?;
            target_array = new_target;
            let eq_array = arrow::compute::kernels::cmp::eq(&new_value, &target_array)?;
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
            if final_bool.value(idx) {
                result.add(*row_id);
            }
        }
    }

    Ok(result)
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
                op: _op,
                fields,
            } => {
                if scalar_expr_constant_null(left)? || scalar_expr_constant_null(right)? {
                    stack.push(Treemap::new());
                    continue;
                }
                let rows = collect_compare_domain_rows(
                    storage,
                    left,
                    right,
                    *_op,
                    fields,
                    all_rows_cache,
                )?;
                stack.push(rows);
            }
            DomainOp::PushInListDomain {
                expr,
                list,
                fields,
                negated: _negated,
            } => {
                if scalar_expr_constant_null(expr)? || list_all_constant_null(list)? {
                    stack.push(Treemap::new());
                    continue;
                }
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
            DomainOp::PushLiteralFalse => {
                stack.push(Treemap::new());
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
    _left: &ScalarExpr<FieldId>,
    _right: &ScalarExpr<FieldId>,
    _op: CompareOp,
    fields: &[FieldId],
    all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
) -> LlkvResult<Treemap>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    if fields.is_empty() {
        return collect_all_row_ids(storage, all_rows_cache);
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
    Ok(domain.unwrap_or_default())
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

fn evaluate_constant_in_list(
    expr: &ScalarExpr<FieldId>,
    list: &[ScalarExpr<FieldId>],
    negated: bool,
) -> LlkvResult<Option<bool>> {
    let arrays: NumericArrayMap = FxHashMap::default();
    let mut target_array = ScalarEvaluator::evaluate_value(expr, 0, &arrays)?;

    if target_array.data_type() == &DataType::Null || target_array.is_null(0) {
        return Ok(None);
    }

    let mut matched = false;
    let mut saw_null = false;

    for value_expr in list {
        let value_array = ScalarEvaluator::evaluate_value(value_expr, 0, &arrays)?;
        if value_array.data_type() == &DataType::Null || value_array.is_null(0) {
            saw_null = true;
            continue;
        }

        let (new_target, new_value) = kernels::coerce_types(&target_array, &value_array, BinaryOp::Add)?;
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

    let outcome = if matched {
        Some(!negated)
    } else if saw_null {
        None
    } else if negated {
        Some(true)
    } else {
        Some(false)
    };

    Ok(outcome)
}
