//! Query execution engine for LLKV.
//!
//! This crate provides the query execution layer that sits between the query planner
//! (`llkv-plan`) and the storage layer (`llkv-table`, `llkv-column-map`).
//!
//! # Module Organization
//!
//! - [`translation`]: Expression and projection translation utilities
//! - [`types`]: Core type definitions (tables, schemas, columns)  
//! - [`insert`]: INSERT operation support (value coercion)
//!
//! The [`QueryExecutor`] and [`SelectExecution`] implementations are defined inline
//! in this module for now, but should be extracted to a dedicated `query` module
//! in a future refactoring.

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Date32Array, Decimal128Array, Decimal128Builder,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, Int64Builder,
    IntervalMonthDayNanoArray, RecordBatch, StringArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array, new_null_array,
};
use arrow::compute::{
    SortColumn, SortOptions, cast, concat_batches, filter_record_batch, lexsort_to_indices, not,
    or_kleene, take,
};
use arrow::datatypes::{DataType, Field, Float64Type, Int32Type, Int64Type, IntervalUnit, Schema};
use arrow::row::{RowConverter, SortField};
use arrow_buffer::IntervalMonthDayNano;
use llkv_aggregate::{AggregateAccumulator, AggregateKind, AggregateSpec, AggregateState};
use llkv_column_map::gather::gather_indices_from_batches;
use llkv_column_map::store::Projection as StoreProjection;
use llkv_expr::SubqueryId;
use llkv_expr::expr::{
    AggregateCall, BinaryOp, CompareOp, Expr as LlkvExpr, Filter, Operator, ScalarExpr,
};
use llkv_expr::literal::{Literal, LiteralExt};
use llkv_expr::typed_predicate::{
    build_bool_predicate, build_fixed_width_predicate, build_var_width_predicate,
};
use llkv_join::cross_join_pair;
use llkv_plan::{
    AggregateExpr, AggregateFunction, CanonicalRow, CompoundOperator, CompoundQuantifier,
    CompoundSelectComponent, CompoundSelectPlan, OrderByPlan, OrderSortType, OrderTarget,
    PlanValue, SelectPlan, SelectProjection, plan_value_from_literal,
};
use llkv_result::Error;
use llkv_storage::pager::Pager;
use llkv_table::table::{
    RowIdFilter, ScanOrderDirection, ScanOrderSpec, ScanOrderTransform, ScanProjection,
    ScanStreamOptions,
};
use llkv_table::types::FieldId;
use llkv_table::{NumericArrayMap, NumericKernels, ROW_ID_FIELD_ID};
use llkv_types::LogicalFieldId;
use llkv_types::decimal::DecimalValue;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;
use std::convert::TryFrom;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::Ordering;

#[cfg(test)]
use std::cell::RefCell;

// ============================================================================
// Module Declarations
// ============================================================================

pub mod insert;
pub mod translation;
pub mod types;

// ============================================================================
// Type Aliases and Re-exports
// ============================================================================

/// Result type for executor operations.
pub type ExecutorResult<T> = Result<T, Error>;

use crate::translation::schema::infer_computed_data_type;
pub use insert::{
    build_array_for_column, normalize_insert_value_for_column, resolve_insert_columns,
};
use llkv_compute::date::{format_date32_literal, parse_date32_literal};
use llkv_compute::scalar::decimal::{
    align_decimal_to_scale, decimal_from_f64, decimal_from_i64, decimal_truthy,
};
use llkv_compute::scalar::interval::{
    compare_interval_values, interval_value_from_arrow, interval_value_to_arrow,
};
pub use llkv_compute::time::current_time_micros;
pub use translation::{
    build_projected_columns, build_wildcard_projections, full_table_scan_filter,
    resolve_field_id_from_schema, schema_for_projections, translate_predicate,
    translate_predicate_with, translate_scalar, translate_scalar_with,
};
pub use types::{
    ExecutorColumn, ExecutorMultiColumnUnique, ExecutorRowBatch, ExecutorSchema, ExecutorTable,
    ExecutorTableProvider,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum GroupKeyValue {
    Null,
    Int(i64),
    Bool(bool),
    String(String),
}

/// Represents the result value from an aggregate computation.
/// Different aggregates return different types (e.g., AVG returns Float64, COUNT returns Int64).
#[derive(Clone, Debug, PartialEq)]
enum AggregateValue {
    Null,
    Int64(i64),
    Float64(f64),
    Decimal128 { value: i128, scale: i8 },
    String(String),
}

impl AggregateValue {
    /// Convert to i64, truncating floats when present.
    fn as_i64(&self) -> Option<i64> {
        match self {
            AggregateValue::Null => None,
            AggregateValue::Int64(v) => Some(*v),
            AggregateValue::Float64(v) => Some(*v as i64),
            AggregateValue::Decimal128 { value, scale } => {
                // Convert Decimal128 to i64 by scaling down
                let divisor = 10_i128.pow(*scale as u32);
                Some((value / divisor) as i64)
            }
            AggregateValue::String(s) => s.parse().ok(),
        }
    }

    /// Convert to f64, promoting integers when necessary.
    #[allow(dead_code)]
    fn as_f64(&self) -> Option<f64> {
        match self {
            AggregateValue::Null => None,
            AggregateValue::Int64(v) => Some(*v as f64),
            AggregateValue::Float64(v) => Some(*v),
            AggregateValue::Decimal128 { value, scale } => {
                // Convert Decimal128 to f64
                let divisor = 10_f64.powi(*scale as i32);
                Some(*value as f64 / divisor)
            }
            AggregateValue::String(s) => s.parse().ok(),
        }
    }
}

fn decimal_exact_i64(decimal: DecimalValue) -> Option<i64> {
    llkv_compute::scalar::decimal::rescale(decimal, 0)
        .ok()
        .and_then(|integral| i64::try_from(integral.raw_value()).ok())
}

struct GroupState {
    batch: RecordBatch,
    row_idx: usize,
}

/// State for a group when computing aggregates
struct GroupAggregateState {
    representative_batch_idx: usize,
    representative_row: usize,
    row_locations: Vec<(usize, usize)>,
}

struct OutputColumn {
    field: Field,
    source: OutputSource,
}

enum OutputSource {
    TableColumn { index: usize },
    Computed { projection_index: usize },
}

// ============================================================================
// Query Logging Helpers
// ============================================================================

#[cfg(test)]
thread_local! {
    static QUERY_LABEL_STACK: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Guard object that pops the current query label when dropped.
pub struct QueryLogGuard {
    _private: (),
}

/// Install a query label for the current thread so that executor logs can
/// annotate diagnostics with the originating SQL statement.
#[cfg(test)]
pub fn push_query_label(label: impl Into<String>) -> QueryLogGuard {
    QUERY_LABEL_STACK.with(|stack| stack.borrow_mut().push(label.into()));
    QueryLogGuard { _private: () }
}

/// Install a query label for the current thread so that executor logs can
/// annotate diagnostics with the originating SQL statement.
///
/// No-op in non-test builds.
#[cfg(not(test))]
#[inline]
pub fn push_query_label(_label: impl Into<String>) -> QueryLogGuard {
    QueryLogGuard { _private: () }
}

#[cfg(test)]
impl Drop for QueryLogGuard {
    fn drop(&mut self) {
        QUERY_LABEL_STACK.with(|stack| {
            let _ = stack.borrow_mut().pop();
        });
    }
}

#[cfg(not(test))]
impl Drop for QueryLogGuard {
    #[inline]
    fn drop(&mut self) {
        // No-op in non-test builds
    }
}

/// Fetch the innermost query label associated with the current execution thread.
#[cfg(test)]
pub fn current_query_label() -> Option<String> {
    QUERY_LABEL_STACK.with(|stack| stack.borrow().last().cloned())
}

/// Fetch the innermost query label associated with the current execution thread.
///
/// Always returns None in non-test builds.
#[cfg(not(test))]
#[inline]
pub fn current_query_label() -> Option<String> {
    None
}

// ============================================================================
// Query Executor - Implementation
// ============================================================================
// TODO: Extract this implementation into a dedicated query/ module

/// Extract a simple column name from a ScalarExpr when possible.
///
/// Returns `Some(column_name)` if the expression is a plain column reference
/// (optionally wrapped in additive identity operators like unary `+`), otherwise returns `None`
/// (indicating a complex expression that needs full evaluation).
///
/// This handles common cases like `col`, `+col`, or `++col`. Unary negation (`-col`)
/// is intentionally treated as a true expression so aggregates such as `SUM(-col)`
/// evaluate the negated values instead of reading the base column directly.
fn try_extract_simple_column<F: AsRef<str>>(expr: &ScalarExpr<F>) -> Option<&str> {
    match expr {
        ScalarExpr::Column(name) => Some(name.as_ref()),
        // Unwrap unary operators to check if there's a column underneath
        ScalarExpr::Binary { left, op, right } => {
            // Check for unary-like patterns: left or right is a literal that acts as identity
            match op {
                BinaryOp::Add => {
                    // Check if one side is zero (identity for addition)
                    if matches!(left.as_ref(), ScalarExpr::Literal(Literal::Int128(0))) {
                        return try_extract_simple_column(right);
                    }
                    if matches!(right.as_ref(), ScalarExpr::Literal(Literal::Int128(0))) {
                        return try_extract_simple_column(left);
                    }
                }
                // Note: We do NOT handle Subtract here because 0 - col is NOT the same as col
                // It needs to be evaluated as a negation
                BinaryOp::Multiply => {
                    // +col might be Multiply(1, col)
                    if matches!(left.as_ref(), ScalarExpr::Literal(Literal::Int128(1))) {
                        return try_extract_simple_column(right);
                    }
                    if matches!(right.as_ref(), ScalarExpr::Literal(Literal::Int128(1))) {
                        return try_extract_simple_column(left);
                    }
                }
                _ => {}
            }
            None
        }
        _ => None,
    }
}

/// Convert a vector of PlanValues to an Arrow array.
///
/// This currently supports Integer, Float, Null, and String values.
/// The array type is inferred from the first non-null value.
fn plan_values_to_arrow_array(values: &[PlanValue]) -> ExecutorResult<ArrayRef> {
    use arrow::array::{
        Date32Array, Decimal128Array, Float64Array, Int64Array, IntervalMonthDayNanoArray,
        StringArray,
    };

    // Infer type from first non-null value
    let mut value_type = None;
    for v in values {
        if !matches!(v, PlanValue::Null) {
            value_type = Some(v);
            break;
        }
    }

    match value_type {
        Some(PlanValue::Decimal(d)) => {
            let precision = d.precision();
            let scale = d.scale();
            let mut builder = Decimal128Array::builder(values.len())
                .with_precision_and_scale(precision, scale)
                .map_err(|e| {
                    Error::InvalidArgumentError(format!(
                        "invalid Decimal128 precision/scale: {}",
                        e
                    ))
                })?;
            for v in values {
                match v {
                    PlanValue::Decimal(d) => builder.append_value(d.raw_value()),
                    PlanValue::Null => builder.append_null(),
                    other => {
                        return Err(Error::InvalidArgumentError(format!(
                            "expected DECIMAL plan value, found {other:?}"
                        )));
                    }
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        Some(PlanValue::Integer(_)) => {
            let int_values: Vec<Option<i64>> = values
                .iter()
                .map(|v| match v {
                    PlanValue::Integer(i) => Ok(Some(*i)),
                    PlanValue::Null => Ok(None),
                    other => Err(Error::InvalidArgumentError(format!(
                        "expected INTEGER plan value, found {other:?}"
                    ))),
                })
                .collect::<Result<_, _>>()?;
            Ok(Arc::new(Int64Array::from(int_values)) as ArrayRef)
        }
        Some(PlanValue::Float(_)) => {
            let float_values: Vec<Option<f64>> = values
                .iter()
                .map(|v| match v {
                    PlanValue::Float(f) => Ok(Some(*f)),
                    PlanValue::Null => Ok(None),
                    PlanValue::Integer(i) => Ok(Some(*i as f64)),
                    other => Err(Error::InvalidArgumentError(format!(
                        "expected FLOAT plan value, found {other:?}"
                    ))),
                })
                .collect::<Result<_, _>>()?;
            Ok(Arc::new(Float64Array::from(float_values)) as ArrayRef)
        }
        Some(PlanValue::String(_)) => {
            let string_values: Vec<Option<&str>> = values
                .iter()
                .map(|v| match v {
                    PlanValue::String(s) => Ok(Some(s.as_str())),
                    PlanValue::Null => Ok(None),
                    other => Err(Error::InvalidArgumentError(format!(
                        "expected STRING plan value, found {other:?}"
                    ))),
                })
                .collect::<Result<_, _>>()?;
            Ok(Arc::new(StringArray::from(string_values)) as ArrayRef)
        }
        Some(PlanValue::Date32(_)) => {
            let date_values: Vec<Option<i32>> = values
                .iter()
                .map(|v| match v {
                    PlanValue::Date32(d) => Ok(Some(*d)),
                    PlanValue::Null => Ok(None),
                    other => Err(Error::InvalidArgumentError(format!(
                        "expected DATE plan value, found {other:?}"
                    ))),
                })
                .collect::<Result<_, _>>()?;
            Ok(Arc::new(Date32Array::from(date_values)) as ArrayRef)
        }
        Some(PlanValue::Interval(_)) => {
            let interval_values: Vec<Option<IntervalMonthDayNano>> = values
                .iter()
                .map(|v| match v {
                    PlanValue::Interval(interval) => Ok(Some(interval_value_to_arrow(*interval))),
                    PlanValue::Null => Ok(None),
                    other => Err(Error::InvalidArgumentError(format!(
                        "expected INTERVAL plan value, found {other:?}"
                    ))),
                })
                .collect::<Result<_, _>>()?;
            Ok(Arc::new(IntervalMonthDayNanoArray::from(interval_values)) as ArrayRef)
        }
        _ => Ok(new_null_array(&DataType::Null, values.len())),
    }
}

// TODO: Does llkv-table resolvers already handle this (and similar)?
/// Resolve a column name to its index using flexible name matching.
///
/// This function handles various column name formats:
/// 1. Exact match (case-insensitive)
/// 2. Unqualified match (e.g., "col0" matches "table.col0" or "alias.col0")
///
/// This is useful when aggregate expressions reference columns with table qualifiers
/// (like "cor0.col0") but the schema has different qualification patterns.
fn resolve_column_name_to_index(
    col_name: &str,
    column_lookup_map: &FxHashMap<String, usize>,
) -> Option<usize> {
    let col_lower = col_name.to_ascii_lowercase();

    // Try exact match first
    if let Some(&idx) = column_lookup_map.get(&col_lower) {
        return Some(idx);
    }

    // Try matching just the column name without table qualifier
    // e.g., "cor0.col0" should match a field ending in ".col0" or exactly "col0"
    let unqualified = col_name
        .rsplit('.')
        .next()
        .unwrap_or(col_name)
        .to_ascii_lowercase();
    column_lookup_map
        .iter()
        .find(|(k, _)| k.ends_with(&format!(".{}", unqualified)) || k == &&unqualified)
        .map(|(_, &idx)| idx)
}

/// Ensure a stored-column projection exists for the given field, reusing prior entries.
fn get_or_insert_column_projection<P>(
    projections: &mut Vec<ScanProjection>,
    cache: &mut FxHashMap<FieldId, usize>,
    table: &ExecutorTable<P>,
    column: &ExecutorColumn,
) -> usize
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    if let Some(existing) = cache.get(&column.field_id) {
        return *existing;
    }

    let projection_index = projections.len();
    let alias = if column.name.is_empty() {
        format!("col{}", column.field_id)
    } else {
        column.name.clone()
    };
    projections.push(ScanProjection::from(StoreProjection::with_alias(
        LogicalFieldId::for_user(table.table.table_id(), column.field_id),
        alias,
    )));
    cache.insert(column.field_id, projection_index);
    projection_index
}

/// Ensure a computed projection exists for the provided expression, returning its index and type.
fn ensure_computed_projection<P>(
    expr: &ScalarExpr<String>,
    table: &ExecutorTable<P>,
    projections: &mut Vec<ScanProjection>,
    cache: &mut FxHashMap<String, (usize, DataType)>,
    alias_counter: &mut usize,
) -> ExecutorResult<(usize, DataType)>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let key = format!("{:?}", expr);
    if let Some((idx, dtype)) = cache.get(&key) {
        return Ok((*idx, dtype.clone()));
    }

    let translated = translate_scalar(expr, table.schema.as_ref(), |name| {
        Error::InvalidArgumentError(format!("unknown column '{}' in aggregate expression", name))
    })?;
    let data_type = infer_computed_data_type(table.schema.as_ref(), &translated)?;
    if data_type == DataType::Null {
        tracing::debug!(
            "ensure_computed_projection inferred Null type for expr: {:?}",
            expr
        );
    }
    let alias = format!("__agg_expr_{}", *alias_counter);
    *alias_counter += 1;
    let projection_index = projections.len();
    projections.push(ScanProjection::computed(translated, alias));
    cache.insert(key, (projection_index, data_type.clone()));
    Ok((projection_index, data_type))
}

/// Query executor that executes SELECT plans.
pub struct QueryExecutor<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    provider: Arc<dyn ExecutorTableProvider<P>>,
}

impl<P> QueryExecutor<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(provider: Arc<dyn ExecutorTableProvider<P>>) -> Self {
        Self { provider }
    }

    pub fn execute_select(&self, plan: SelectPlan) -> ExecutorResult<SelectExecution<P>> {
        self.execute_select_with_filter(plan, None)
    }

    pub fn execute_select_with_filter(
        &self,
        plan: SelectPlan,
        row_filter: Option<std::sync::Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        let limit = plan.limit;
        let offset = plan.offset;

        let execution = if plan.compound.is_some() {
            self.execute_compound_select(plan, row_filter)?
        } else if plan.tables.is_empty() {
            self.execute_select_without_table(plan)?
        } else if !plan.group_by.is_empty() {
            if plan.tables.len() > 1 {
                self.execute_cross_product(plan)?
            } else {
                let table_ref = &plan.tables[0];
                let table = self.provider.get_table(&table_ref.qualified_name())?;
                let display_name = table_ref.qualified_name();
                self.execute_group_by_single_table(table, display_name, plan, row_filter)?
            }
        } else if plan.tables.len() > 1 {
            self.execute_cross_product(plan)?
        } else {
            // Single table query
            let table_ref = &plan.tables[0];
            let table = self.provider.get_table(&table_ref.qualified_name())?;
            let display_name = table_ref.qualified_name();

            if !plan.aggregates.is_empty() {
                self.execute_aggregates(table, display_name, plan, row_filter)?
            } else if self.has_computed_aggregates(&plan) {
                // Handle computed projections that contain embedded aggregates
                self.execute_computed_aggregates(table, display_name, plan, row_filter)?
            } else {
                self.execute_projection(table, display_name, plan, row_filter)?
            }
        };

        Ok(execution.with_limit(limit).with_offset(offset))
    }

    /// Execute a compound SELECT query (UNION, EXCEPT, INTERSECT).
    ///
    /// Evaluates the initial SELECT and each subsequent operation, combining results
    /// according to the specified operator and quantifier. Handles deduplication for
    /// DISTINCT quantifiers using hash-based row encoding.
    ///
    /// # Arguments
    ///
    /// * `plan` - SELECT plan containing compound operations
    /// * `row_filter` - Optional row ID filter to apply to all component queries
    ///
    /// # Implementation Notes
    ///
    /// - UNION ALL: Simple concatenation with no deduplication
    /// - UNION DISTINCT: Hash-based deduplication across all rows
    /// - EXCEPT DISTINCT: Removes right-side rows from left-side results
    /// - INTERSECT DISTINCT: Keeps only rows present in both sides
    /// - EXCEPT ALL: Not yet implemented
    /// - INTERSECT ALL: Not yet implemented
    fn execute_compound_select(
        &self,
        plan: SelectPlan,
        row_filter: Option<std::sync::Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        let order_by = plan.order_by.clone();
        let compound = plan.compound.expect("compound plan should be present");

        let CompoundSelectPlan {
            initial,
            operations,
        } = compound;

        let initial_exec = self.execute_select_with_filter(*initial, row_filter.clone())?;
        let schema = initial_exec.schema();
        let mut rows = initial_exec.into_rows()?;
        let mut distinct_cache: Option<FxHashSet<Vec<u8>>> = None;

        for component in operations {
            let exec = self.execute_select_with_filter(component.plan, row_filter.clone())?;
            let other_schema = exec.schema();
            ensure_schema_compatibility(schema.as_ref(), other_schema.as_ref())?;
            let other_rows = exec.into_rows()?;

            match (component.operator, component.quantifier) {
                (CompoundOperator::Union, CompoundQuantifier::All) => {
                    rows.extend(other_rows);
                    distinct_cache = None;
                }
                (CompoundOperator::Union, CompoundQuantifier::Distinct) => {
                    ensure_distinct_rows(&mut rows, &mut distinct_cache);
                    let cache = distinct_cache
                        .as_mut()
                        .expect("distinct cache should be initialized");
                    for row in other_rows {
                        let key = encode_row(&row);
                        if cache.insert(key) {
                            rows.push(row);
                        }
                    }
                }
                (CompoundOperator::Except, CompoundQuantifier::Distinct) => {
                    ensure_distinct_rows(&mut rows, &mut distinct_cache);
                    let cache = distinct_cache
                        .as_mut()
                        .expect("distinct cache should be initialized");
                    if rows.is_empty() {
                        continue;
                    }
                    let mut remove_keys = FxHashSet::default();
                    for row in other_rows {
                        remove_keys.insert(encode_row(&row));
                    }
                    if remove_keys.is_empty() {
                        continue;
                    }
                    rows.retain(|row| {
                        let key = encode_row(row);
                        if remove_keys.contains(&key) {
                            cache.remove(&key);
                            false
                        } else {
                            true
                        }
                    });
                }
                (CompoundOperator::Except, CompoundQuantifier::All) => {
                    return Err(Error::InvalidArgumentError(
                        "EXCEPT ALL is not supported yet".into(),
                    ));
                }
                (CompoundOperator::Intersect, CompoundQuantifier::Distinct) => {
                    ensure_distinct_rows(&mut rows, &mut distinct_cache);
                    let mut right_keys = FxHashSet::default();
                    for row in other_rows {
                        right_keys.insert(encode_row(&row));
                    }
                    if right_keys.is_empty() {
                        rows.clear();
                        distinct_cache = Some(FxHashSet::default());
                        continue;
                    }
                    let mut new_rows = Vec::new();
                    let mut new_cache = FxHashSet::default();
                    for row in rows.drain(..) {
                        let key = encode_row(&row);
                        if right_keys.contains(&key) && new_cache.insert(key) {
                            new_rows.push(row);
                        }
                    }
                    rows = new_rows;
                    distinct_cache = Some(new_cache);
                }
                (CompoundOperator::Intersect, CompoundQuantifier::All) => {
                    return Err(Error::InvalidArgumentError(
                        "INTERSECT ALL is not supported yet".into(),
                    ));
                }
            }
        }

        let mut batch = rows_to_record_batch(schema.clone(), &rows)?;
        if !order_by.is_empty() && batch.num_rows() > 0 {
            batch = sort_record_batch_with_order(&schema, &batch, &order_by)?;
        }

        Ok(SelectExecution::new_single_batch(
            String::new(),
            schema,
            batch,
        ))
    }

    /// Check if any computed projections contain aggregate functions
    fn has_computed_aggregates(&self, plan: &SelectPlan) -> bool {
        plan.projections.iter().any(|proj| {
            if let SelectProjection::Computed { expr, .. } = proj {
                Self::expr_contains_aggregate(expr)
            } else {
                false
            }
        })
    }

    /// Recursively check if a predicate expression contains aggregates
    fn predicate_contains_aggregate(expr: &llkv_expr::expr::Expr<String>) -> bool {
        match expr {
            llkv_expr::expr::Expr::And(exprs) | llkv_expr::expr::Expr::Or(exprs) => {
                exprs.iter().any(Self::predicate_contains_aggregate)
            }
            llkv_expr::expr::Expr::Not(inner) => Self::predicate_contains_aggregate(inner),
            llkv_expr::expr::Expr::Compare { left, right, .. } => {
                Self::expr_contains_aggregate(left) || Self::expr_contains_aggregate(right)
            }
            llkv_expr::expr::Expr::InList { expr, list, .. } => {
                Self::expr_contains_aggregate(expr)
                    || list.iter().any(|e| Self::expr_contains_aggregate(e))
            }
            llkv_expr::expr::Expr::IsNull { expr, .. } => Self::expr_contains_aggregate(expr),
            llkv_expr::expr::Expr::Literal(_) => false,
            llkv_expr::expr::Expr::Pred(_) => false,
            llkv_expr::expr::Expr::Exists(_) => false,
        }
    }

    /// Recursively check if a scalar expression contains aggregates
    fn expr_contains_aggregate(expr: &ScalarExpr<String>) -> bool {
        match expr {
            ScalarExpr::Aggregate(_) => true,
            ScalarExpr::Binary { left, right, .. } => {
                Self::expr_contains_aggregate(left) || Self::expr_contains_aggregate(right)
            }
            ScalarExpr::Compare { left, right, .. } => {
                Self::expr_contains_aggregate(left) || Self::expr_contains_aggregate(right)
            }
            ScalarExpr::GetField { base, .. } => Self::expr_contains_aggregate(base),
            ScalarExpr::Cast { expr, .. } => Self::expr_contains_aggregate(expr),
            ScalarExpr::Not(expr) => Self::expr_contains_aggregate(expr),
            ScalarExpr::IsNull { expr, .. } => Self::expr_contains_aggregate(expr),
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                operand
                    .as_deref()
                    .map(Self::expr_contains_aggregate)
                    .unwrap_or(false)
                    || branches.iter().any(|(when_expr, then_expr)| {
                        Self::expr_contains_aggregate(when_expr)
                            || Self::expr_contains_aggregate(then_expr)
                    })
                    || else_expr
                        .as_deref()
                        .map(Self::expr_contains_aggregate)
                        .unwrap_or(false)
            }
            ScalarExpr::Coalesce(items) => items.iter().any(Self::expr_contains_aggregate),
            ScalarExpr::Column(_) | ScalarExpr::Literal(_) | ScalarExpr::Random => false,
            ScalarExpr::ScalarSubquery(_) => false,
        }
    }

    fn evaluate_exists_subquery(
        &self,
        context: &mut CrossProductExpressionContext,
        subquery: &llkv_plan::FilterSubquery,
        batch: &RecordBatch,
        row_idx: usize,
    ) -> ExecutorResult<bool> {
        let bindings =
            collect_correlated_bindings(context, batch, row_idx, &subquery.correlated_columns)?;
        let bound_plan = bind_select_plan(&subquery.plan, &bindings)?;
        let execution = self.execute_select(bound_plan)?;
        let mut found = false;
        execution.stream(|inner_batch| {
            if inner_batch.num_rows() > 0 {
                found = true;
            }
            Ok(())
        })?;
        Ok(found)
    }

    fn evaluate_scalar_subquery_literal(
        &self,
        context: &mut CrossProductExpressionContext,
        subquery: &llkv_plan::ScalarSubquery,
        batch: &RecordBatch,
        row_idx: usize,
    ) -> ExecutorResult<Literal> {
        let bindings =
            collect_correlated_bindings(context, batch, row_idx, &subquery.correlated_columns)?;
        self.evaluate_scalar_subquery_with_bindings(subquery, &bindings)
    }

    fn evaluate_scalar_subquery_with_bindings(
        &self,
        subquery: &llkv_plan::ScalarSubquery,
        bindings: &FxHashMap<String, Literal>,
    ) -> ExecutorResult<Literal> {
        let bound_plan = bind_select_plan(&subquery.plan, bindings)?;
        let execution = self.execute_select(bound_plan)?;
        let mut rows_seen: usize = 0;
        let mut result: Option<Literal> = None;
        execution.stream(|inner_batch| {
            if inner_batch.num_columns() != 1 {
                return Err(Error::InvalidArgumentError(
                    "scalar subquery must return exactly one column".into(),
                ));
            }
            let column = inner_batch.column(0).clone();
            for idx in 0..inner_batch.num_rows() {
                if rows_seen >= 1 {
                    return Err(Error::InvalidArgumentError(
                        "scalar subquery produced more than one row".into(),
                    ));
                }
                rows_seen = rows_seen.saturating_add(1);
                result = Some(Literal::from_array_ref(&column, idx)?);
            }
            Ok(())
        })?;

        if rows_seen == 0 {
            Ok(Literal::Null)
        } else {
            result
                .ok_or_else(|| Error::Internal("scalar subquery evaluation missing result".into()))
        }
    }

    fn evaluate_scalar_subquery_numeric(
        &self,
        context: &mut CrossProductExpressionContext,
        subquery: &llkv_plan::ScalarSubquery,
        batch: &RecordBatch,
    ) -> ExecutorResult<ArrayRef> {
        let row_count = batch.num_rows();
        let mut row_job_indices: Vec<usize> = Vec::with_capacity(row_count);
        let mut unique_bindings: Vec<FxHashMap<String, Literal>> = Vec::new();
        let mut key_lookup: FxHashMap<Vec<u8>, usize> = FxHashMap::default();

        for row_idx in 0..row_count {
            let bindings =
                collect_correlated_bindings(context, batch, row_idx, &subquery.correlated_columns)?;

            // Encode the correlated binding set so we can deduplicate identical subqueries.
            let mut plan_values: Vec<PlanValue> =
                Vec::with_capacity(subquery.correlated_columns.len());
            for column in &subquery.correlated_columns {
                let literal = bindings
                    .get(&column.placeholder)
                    .cloned()
                    .unwrap_or(Literal::Null);
                let plan_value = plan_value_from_literal(&literal)?;
                plan_values.push(plan_value);
            }
            let key = encode_row(&plan_values);

            let job_idx = if let Some(&existing) = key_lookup.get(&key) {
                existing
            } else {
                let idx = unique_bindings.len();
                key_lookup.insert(key, idx);
                unique_bindings.push(bindings);
                idx
            };
            row_job_indices.push(job_idx);
        }

        // Execute each unique correlated subquery in parallel on the shared Rayon pool.
        let job_results: Vec<ExecutorResult<Literal>> =
            llkv_column_map::parallel::with_thread_pool(|| {
                let results: Vec<ExecutorResult<Literal>> = unique_bindings
                    .par_iter()
                    .map(|bindings| self.evaluate_scalar_subquery_with_bindings(subquery, bindings))
                    .collect();
                results
            });

        let mut job_literals: Vec<Literal> = Vec::with_capacity(job_results.len());
        for result in job_results {
            job_literals.push(result?);
        }

        let mut values: Vec<Option<f64>> = Vec::with_capacity(row_count);
        let mut all_integer = true;

        for row_idx in 0..row_count {
            let literal = &job_literals[row_job_indices[row_idx]];
            match literal {
                Literal::Null => values.push(None),
                Literal::Int128(value) => {
                    let cast = i64::try_from(*value).map_err(|_| {
                        Error::InvalidArgumentError(
                            "scalar subquery integer result exceeds supported range".into(),
                        )
                    })?;
                    values.push(Some(cast as f64));
                }
                Literal::Float64(value) => {
                    all_integer = false;
                    values.push(Some(*value));
                }
                Literal::Boolean(flag) => {
                    let numeric = if *flag { 1.0 } else { 0.0 };
                    values.push(Some(numeric));
                }
                Literal::Decimal128(decimal) => {
                    if let Some(value) = decimal_exact_i64(*decimal) {
                        values.push(Some(value as f64));
                    } else {
                        all_integer = false;
                        values.push(Some(decimal.to_f64()));
                    }
                }
                Literal::String(_)
                | Literal::Struct(_)
                | Literal::Date32(_)
                | Literal::Interval(_) => {
                    return Err(Error::InvalidArgumentError(
                        "scalar subquery produced non-numeric result in numeric context".into(),
                    ));
                }
            }
        }

        if all_integer {
            let iter = values.into_iter().map(|opt| opt.map(|v| v as i64));
            let array = Int64Array::from_iter(iter);
            Ok(Arc::new(array) as ArrayRef)
        } else {
            let array = Float64Array::from_iter(values);
            Ok(Arc::new(array) as ArrayRef)
        }
    }

    fn evaluate_scalar_subquery_array(
        &self,
        context: &mut CrossProductExpressionContext,
        subquery: &llkv_plan::ScalarSubquery,
        batch: &RecordBatch,
    ) -> ExecutorResult<ArrayRef> {
        let mut values = Vec::with_capacity(batch.num_rows());
        for row_idx in 0..batch.num_rows() {
            let literal =
                self.evaluate_scalar_subquery_literal(context, subquery, batch, row_idx)?;
            values.push(literal);
        }
        literals_to_array(&values)
    }

    fn evaluate_projection_expression(
        &self,
        context: &mut CrossProductExpressionContext,
        expr: &ScalarExpr<String>,
        batch: &RecordBatch,
        scalar_lookup: &FxHashMap<SubqueryId, &llkv_plan::ScalarSubquery>,
    ) -> ExecutorResult<ArrayRef> {
        let translated = translate_scalar(expr, context.schema(), |name| {
            Error::InvalidArgumentError(format!(
                "column '{}' not found in cross product result",
                name
            ))
        })?;

        let mut subquery_ids: FxHashSet<SubqueryId> = FxHashSet::default();
        collect_scalar_subquery_ids(&translated, &mut subquery_ids);

        let mut mapping: FxHashMap<SubqueryId, FieldId> = FxHashMap::default();
        for subquery_id in &subquery_ids {
            let info = scalar_lookup
                .get(subquery_id)
                .ok_or_else(|| Error::Internal("missing scalar subquery metadata".into()))?;
            let field_id = context.allocate_synthetic_field_id()?;
            let numeric = self.evaluate_scalar_subquery_numeric(context, info, batch)?;
            context.numeric_cache.insert(field_id, numeric);
            mapping.insert(*subquery_id, field_id);
        }

        let rewritten = rewrite_scalar_expr_for_subqueries(&translated, &mapping);
        context.evaluate_numeric(&rewritten, batch)
    }

    /// Execute a SELECT without a FROM clause (e.g., SELECT 42, SELECT {'a': 1})
    fn execute_select_without_table(&self, plan: SelectPlan) -> ExecutorResult<SelectExecution<P>> {
        use arrow::array::ArrayRef;
        use arrow::datatypes::Field;

        // Build schema from computed projections
        let mut fields = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        for proj in &plan.projections {
            match proj {
                SelectProjection::Computed { expr, alias } => {
                    let literal =
                        evaluate_constant_scalar_with_aggregates(expr).ok_or_else(|| {
                            Error::InvalidArgumentError(
                                "SELECT without FROM only supports constant expressions".into(),
                            )
                        })?;
                    let (dtype, array) = Self::literal_to_array(&literal)?;

                    fields.push(Field::new(alias.clone(), dtype, true));
                    arrays.push(array);
                }
                _ => {
                    return Err(Error::InvalidArgumentError(
                        "SELECT without FROM only supports computed projections".into(),
                    ));
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));
        let mut batch = RecordBatch::try_new(Arc::clone(&schema), arrays)
            .map_err(|e| Error::Internal(format!("failed to create record batch: {}", e)))?;

        if plan.distinct {
            let mut state = DistinctState::default();
            batch = match distinct_filter_batch(batch, &mut state)? {
                Some(filtered) => filtered,
                None => RecordBatch::new_empty(Arc::clone(&schema)),
            };
        }

        let schema = batch.schema();

        Ok(SelectExecution::new_single_batch(
            String::new(), // No table name
            schema,
            batch,
        ))
    }

    /// Convert a Literal to an Arrow array (recursive for nested structs)
    fn literal_to_array(lit: &llkv_expr::literal::Literal) -> ExecutorResult<(DataType, ArrayRef)> {
        use arrow::array::{
            ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float64Array, Int64Array,
            IntervalMonthDayNanoArray, StringArray, StructArray, new_null_array,
        };
        use arrow::datatypes::{DataType, Field, IntervalUnit};
        use llkv_compute::scalar::interval::interval_value_to_arrow;
        use llkv_expr::literal::Literal;

        match lit {
            Literal::Int128(v) => {
                let val = i64::try_from(*v).unwrap_or(0);
                Ok((
                    DataType::Int64,
                    Arc::new(Int64Array::from(vec![val])) as ArrayRef,
                ))
            }
            Literal::Float64(v) => Ok((
                DataType::Float64,
                Arc::new(Float64Array::from(vec![*v])) as ArrayRef,
            )),
            Literal::Boolean(v) => Ok((
                DataType::Boolean,
                Arc::new(BooleanArray::from(vec![*v])) as ArrayRef,
            )),
            Literal::Decimal128(value) => {
                let iter = std::iter::once(value.raw_value());
                let precision = std::cmp::max(value.precision(), value.scale() as u8);
                let array = Decimal128Array::from_iter_values(iter)
                    .with_precision_and_scale(precision, value.scale())
                    .map_err(|err| {
                        Error::InvalidArgumentError(format!(
                            "failed to build Decimal128 literal array: {err}"
                        ))
                    })?;
                Ok((
                    DataType::Decimal128(precision, value.scale()),
                    Arc::new(array) as ArrayRef,
                ))
            }
            Literal::String(v) => Ok((
                DataType::Utf8,
                Arc::new(StringArray::from(vec![v.clone()])) as ArrayRef,
            )),
            Literal::Date32(v) => Ok((
                DataType::Date32,
                Arc::new(Date32Array::from(vec![*v])) as ArrayRef,
            )),
            Literal::Null => Ok((DataType::Null, new_null_array(&DataType::Null, 1))),
            Literal::Interval(interval) => Ok((
                DataType::Interval(IntervalUnit::MonthDayNano),
                Arc::new(IntervalMonthDayNanoArray::from(vec![
                    interval_value_to_arrow(*interval),
                ])) as ArrayRef,
            )),
            Literal::Struct(struct_fields) => {
                // Build a struct array recursively
                let mut inner_fields = Vec::new();
                let mut inner_arrays = Vec::new();

                for (field_name, field_lit) in struct_fields {
                    let (field_dtype, field_array) = Self::literal_to_array(field_lit)?;
                    inner_fields.push(Field::new(field_name.clone(), field_dtype, true));
                    inner_arrays.push(field_array);
                }

                let struct_array =
                    StructArray::try_new(inner_fields.clone().into(), inner_arrays, None).map_err(
                        |e| Error::Internal(format!("failed to create struct array: {}", e)),
                    )?;

                Ok((
                    DataType::Struct(inner_fields.into()),
                    Arc::new(struct_array) as ArrayRef,
                ))
            }
        }
    }

    /// Execute a cross product query (FROM table1, table2, ...)
    fn execute_cross_product(&self, plan: SelectPlan) -> ExecutorResult<SelectExecution<P>> {
        use arrow::compute::concat_batches;

        if plan.tables.len() < 2 {
            return Err(Error::InvalidArgumentError(
                "cross product requires at least 2 tables".into(),
            ));
        }

        let mut tables_with_handles = Vec::with_capacity(plan.tables.len());
        for table_ref in &plan.tables {
            let qualified_name = table_ref.qualified_name();
            let table = self.provider.get_table(&qualified_name)?;
            tables_with_handles.push((table_ref.clone(), table));
        }

        let display_name = tables_with_handles
            .iter()
            .map(|(table_ref, _)| table_ref.qualified_name())
            .collect::<Vec<_>>()
            .join(",");

        let mut remaining_filter = plan.filter.clone();

        // Try hash join optimization first - this avoids materializing all tables
        // We now attempt hash join even with subqueries, as we can handle non-subquery join predicates separately
        let join_data = if remaining_filter.as_ref().is_some() {
            self.try_execute_hash_join(&plan, &tables_with_handles)?
        } else {
            None
        };

        let current = if let Some((joined, handled_all_predicates)) = join_data {
            // Clear filter only if hash join handled all predicates
            if handled_all_predicates {
                remaining_filter = None;
            }
            joined
        } else {
            // Hash join not applicable - use llkv-join for proper join support or fall back to cartesian product
            let has_joins = !plan.joins.is_empty();

            if has_joins && tables_with_handles.len() == 2 {
                // Use llkv-join for 2-table joins (including LEFT JOIN)
                use llkv_join::{JoinOptions, TableJoinExt};

                let (left_ref, left_table) = &tables_with_handles[0];
                let (right_ref, right_table) = &tables_with_handles[1];

                let join_metadata = plan.joins.first().ok_or_else(|| {
                    Error::InvalidArgumentError("expected join metadata for two-table join".into())
                })?;

                let join_type = match join_metadata.join_type {
                    llkv_plan::JoinPlan::Inner => llkv_join::JoinType::Inner,
                    llkv_plan::JoinPlan::Left => llkv_join::JoinType::Left,
                    llkv_plan::JoinPlan::Right => llkv_join::JoinType::Right,
                    llkv_plan::JoinPlan::Full => llkv_join::JoinType::Full,
                };

                tracing::debug!(
                    "Using llkv-join for {join_type:?} join between {} and {}",
                    left_ref.qualified_name(),
                    right_ref.qualified_name()
                );

                let left_col_count = left_table.schema.columns.len();
                let right_col_count = right_table.schema.columns.len();

                let mut combined_fields = Vec::with_capacity(left_col_count + right_col_count);
                for col in &left_table.schema.columns {
                    combined_fields.push(Field::new(
                        col.name.clone(),
                        col.data_type.clone(),
                        col.nullable,
                    ));
                }
                for col in &right_table.schema.columns {
                    combined_fields.push(Field::new(
                        col.name.clone(),
                        col.data_type.clone(),
                        col.nullable,
                    ));
                }
                let combined_schema = Arc::new(Schema::new(combined_fields));
                let column_counts = vec![left_col_count, right_col_count];
                let table_indices = vec![0, 1];

                let mut join_keys = Vec::new();
                let mut condition_is_trivial = false;
                let mut condition_is_impossible = false;

                if let Some(condition) = join_metadata.on_condition.as_ref() {
                    let plan = build_join_keys_from_condition(
                        condition,
                        left_ref,
                        left_table.as_ref(),
                        right_ref,
                        right_table.as_ref(),
                    )?;
                    join_keys = plan.keys;
                    condition_is_trivial = plan.always_true;
                    condition_is_impossible = plan.always_false;
                }

                if condition_is_impossible {
                    let batches = build_no_match_join_batches(
                        join_type,
                        left_ref,
                        left_table.as_ref(),
                        right_ref,
                        right_table.as_ref(),
                        Arc::clone(&combined_schema),
                    )?;

                    TableCrossProductData {
                        schema: combined_schema,
                        batches,
                        column_counts,
                        table_indices,
                    }
                } else {
                    if !condition_is_trivial
                        && join_metadata.on_condition.is_some()
                        && join_keys.is_empty()
                    {
                        return Err(Error::InvalidArgumentError(
                            "JOIN ON clause must include at least one equality predicate".into(),
                        ));
                    }

                    let mut result_batches = Vec::new();
                    left_table.table.join_stream(
                        &right_table.table,
                        &join_keys,
                        &JoinOptions {
                            join_type,
                            ..Default::default()
                        },
                        |batch| {
                            result_batches.push(batch);
                        },
                    )?;

                    TableCrossProductData {
                        schema: combined_schema,
                        batches: result_batches,
                        column_counts,
                        table_indices,
                    }
                }
            } else if has_joins && tables_with_handles.len() > 2 {
                // Multi-table join: perform sequential pairwise hash joins to avoid full cartesian product

                let join_lookup: FxHashMap<usize, &llkv_plan::JoinMetadata> = plan
                    .joins
                    .iter()
                    .map(|join| (join.left_table_index, join))
                    .collect();

                // Get pushdown filters
                let constraint_map = if let Some(filter_wrapper) = remaining_filter.as_ref() {
                    extract_literal_pushdown_filters(
                        &filter_wrapper.predicate,
                        &tables_with_handles,
                    )
                } else {
                    vec![Vec::new(); tables_with_handles.len()]
                };

                // Start with the first table
                let (first_ref, first_table) = &tables_with_handles[0];
                let first_constraints = constraint_map.first().map(|v| v.as_slice()).unwrap_or(&[]);
                let mut accumulated =
                    collect_table_data(0, first_ref, first_table.as_ref(), first_constraints)?;

                // Join each subsequent table
                for (idx, (right_ref, right_table)) in
                    tables_with_handles.iter().enumerate().skip(1)
                {
                    let right_constraints =
                        constraint_map.get(idx).map(|v| v.as_slice()).unwrap_or(&[]);

                    let join_metadata = join_lookup.get(&(idx - 1)).ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "No join condition found between table {} and {}. Multi-table queries require explicit JOIN syntax.",
                            idx - 1, idx
                        ))
                    })?;

                    let join_type = match join_metadata.join_type {
                        llkv_plan::JoinPlan::Inner => llkv_join::JoinType::Inner,
                        llkv_plan::JoinPlan::Left => llkv_join::JoinType::Left,
                        llkv_plan::JoinPlan::Right => llkv_join::JoinType::Right,
                        llkv_plan::JoinPlan::Full => llkv_join::JoinType::Full,
                    };

                    // Collect right table data with pushdown filters
                    let right_data = collect_table_data(
                        idx,
                        right_ref,
                        right_table.as_ref(),
                        right_constraints,
                    )?;

                    // Extract join condition
                    let condition_expr = join_metadata
                        .on_condition
                        .clone()
                        .unwrap_or(LlkvExpr::Literal(true));

                    // For now, materialize accumulated result and perform join via llkv-join hash join
                    // This avoids full cartesian product while using our existing join implementation
                    let join_batches = execute_hash_join_batches(
                        &accumulated.schema,
                        &accumulated.batches,
                        &right_data.schema,
                        &right_data.batches,
                        &condition_expr,
                        join_type,
                    )?;

                    // Build combined schema
                    let combined_fields: Vec<Field> = accumulated
                        .schema
                        .fields()
                        .iter()
                        .chain(right_data.schema.fields().iter())
                        .map(|f| {
                            Field::new(f.name().clone(), f.data_type().clone(), f.is_nullable())
                        })
                        .collect();
                    let combined_schema = Arc::new(Schema::new(combined_fields));

                    accumulated = TableCrossProductData {
                        schema: combined_schema,
                        batches: join_batches,
                        column_counts: {
                            let mut counts = accumulated.column_counts;
                            counts.push(right_data.schema.fields().len());
                            counts
                        },
                        table_indices: {
                            let mut indices = accumulated.table_indices;
                            indices.push(idx);
                            indices
                        },
                    };
                }

                accumulated
            } else {
                // Fall back to cartesian product for other cases (no joins specified)
                let constraint_map = if let Some(filter_wrapper) = remaining_filter.as_ref() {
                    extract_literal_pushdown_filters(
                        &filter_wrapper.predicate,
                        &tables_with_handles,
                    )
                } else {
                    vec![Vec::new(); tables_with_handles.len()]
                };

                let mut staged: Vec<TableCrossProductData> =
                    Vec::with_capacity(tables_with_handles.len());
                let join_lookup: FxHashMap<usize, &llkv_plan::JoinMetadata> = plan
                    .joins
                    .iter()
                    .map(|join| (join.left_table_index, join))
                    .collect();

                let mut idx = 0usize;
                while idx < tables_with_handles.len() {
                    if let Some(join_metadata) = join_lookup.get(&idx) {
                        if idx + 1 >= tables_with_handles.len() {
                            return Err(Error::Internal(
                                "join metadata references table beyond FROM list".into(),
                            ));
                        }

                        // Only apply the no-match optimisation when the ON predicate is provably false
                        // and the right-hand table is not the start of another join chain. This keeps the
                        // fallback semantics aligned with two-table joins without interfering with more
                        // complex join graphs (which are still handled by the cartesian product path).
                        let overlaps_next_join = join_lookup.contains_key(&(idx + 1));
                        if let Some(condition) = join_metadata.on_condition.as_ref() {
                            let (left_ref, left_table) = &tables_with_handles[idx];
                            let (right_ref, right_table) = &tables_with_handles[idx + 1];
                            let join_plan = build_join_keys_from_condition(
                                condition,
                                left_ref,
                                left_table.as_ref(),
                                right_ref,
                                right_table.as_ref(),
                            )?;
                            if join_plan.always_false && !overlaps_next_join {
                                let join_type = match join_metadata.join_type {
                                    llkv_plan::JoinPlan::Inner => llkv_join::JoinType::Inner,
                                    llkv_plan::JoinPlan::Left => llkv_join::JoinType::Left,
                                    llkv_plan::JoinPlan::Right => llkv_join::JoinType::Right,
                                    llkv_plan::JoinPlan::Full => llkv_join::JoinType::Full,
                                };

                                let left_col_count = left_table.schema.columns.len();
                                let right_col_count = right_table.schema.columns.len();

                                let mut combined_fields =
                                    Vec::with_capacity(left_col_count + right_col_count);
                                for col in &left_table.schema.columns {
                                    combined_fields.push(Field::new(
                                        col.name.clone(),
                                        col.data_type.clone(),
                                        col.nullable,
                                    ));
                                }
                                for col in &right_table.schema.columns {
                                    combined_fields.push(Field::new(
                                        col.name.clone(),
                                        col.data_type.clone(),
                                        col.nullable,
                                    ));
                                }

                                let combined_schema = Arc::new(Schema::new(combined_fields));
                                let batches = build_no_match_join_batches(
                                    join_type,
                                    left_ref,
                                    left_table.as_ref(),
                                    right_ref,
                                    right_table.as_ref(),
                                    Arc::clone(&combined_schema),
                                )?;

                                staged.push(TableCrossProductData {
                                    schema: combined_schema,
                                    batches,
                                    column_counts: vec![left_col_count, right_col_count],
                                    table_indices: vec![idx, idx + 1],
                                });
                                idx += 2;
                                continue;
                            }
                        }
                    }

                    let (table_ref, table) = &tables_with_handles[idx];
                    let constraints = constraint_map.get(idx).map(|v| v.as_slice()).unwrap_or(&[]);
                    staged.push(collect_table_data(
                        idx,
                        table_ref,
                        table.as_ref(),
                        constraints,
                    )?);
                    idx += 1;
                }

                cross_join_all(staged)?
            }
        };

        let TableCrossProductData {
            schema: combined_schema,
            batches: mut combined_batches,
            column_counts,
            table_indices,
        } = current;

        let column_lookup_map = build_cross_product_column_lookup(
            combined_schema.as_ref(),
            &plan.tables,
            &column_counts,
            &table_indices,
        );

        let scalar_lookup: FxHashMap<SubqueryId, &llkv_plan::ScalarSubquery> = plan
            .scalar_subqueries
            .iter()
            .map(|subquery| (subquery.id, subquery))
            .collect();

        if let Some(filter_wrapper) = remaining_filter.as_ref() {
            let mut filter_context = CrossProductExpressionContext::new(
                combined_schema.as_ref(),
                column_lookup_map.clone(),
            )?;
            let translated_filter = translate_predicate(
                filter_wrapper.predicate.clone(),
                filter_context.schema(),
                |name| {
                    Error::InvalidArgumentError(format!(
                        "column '{}' not found in cross product result",
                        name
                    ))
                },
            )?;

            let subquery_lookup: FxHashMap<llkv_expr::SubqueryId, &llkv_plan::FilterSubquery> =
                filter_wrapper
                    .subqueries
                    .iter()
                    .map(|subquery| (subquery.id, subquery))
                    .collect();
            let mut predicate_scalar_ids = FxHashSet::default();
            collect_predicate_scalar_subquery_ids(&translated_filter, &mut predicate_scalar_ids);

            let mut filtered_batches = Vec::with_capacity(combined_batches.len());
            for batch in combined_batches.into_iter() {
                filter_context.reset();
                for subquery_id in &predicate_scalar_ids {
                    let info = scalar_lookup.get(subquery_id).ok_or_else(|| {
                        Error::Internal("missing scalar subquery metadata".into())
                    })?;
                    let array =
                        self.evaluate_scalar_subquery_array(&mut filter_context, info, &batch)?;
                    let accessor = ColumnAccessor::from_array(&array)?;
                    filter_context.register_scalar_subquery_column(*subquery_id, accessor);
                }
                let mask = filter_context.evaluate_predicate_mask(
                    &translated_filter,
                    &batch,
                    |ctx, subquery_expr, row_idx, current_batch| {
                        let subquery = subquery_lookup.get(&subquery_expr.id).ok_or_else(|| {
                            Error::Internal("missing correlated subquery metadata".into())
                        })?;
                        let exists =
                            self.evaluate_exists_subquery(ctx, subquery, current_batch, row_idx)?;
                        let value = if subquery_expr.negated {
                            !exists
                        } else {
                            exists
                        };
                        Ok(Some(value))
                    },
                )?;
                let filtered = filter_record_batch(&batch, &mask).map_err(|err| {
                    Error::InvalidArgumentError(format!(
                        "failed to apply cross product filter: {err}"
                    ))
                })?;
                if filtered.num_rows() > 0 {
                    filtered_batches.push(filtered);
                }
            }
            combined_batches = filtered_batches;
        }

        // GROUP BY takes precedence - it can also have aggregates in projections
        if !plan.group_by.is_empty() {
            return self.execute_group_by_from_batches(
                display_name,
                plan,
                combined_schema,
                combined_batches,
                column_lookup_map,
            );
        }

        if !plan.aggregates.is_empty() {
            return self.execute_cross_product_aggregates(
                Arc::clone(&combined_schema),
                combined_batches,
                &column_lookup_map,
                &plan,
                &display_name,
            );
        }

        if self.has_computed_aggregates(&plan) {
            return self.execute_cross_product_computed_aggregates(
                Arc::clone(&combined_schema),
                combined_batches,
                &column_lookup_map,
                &plan,
                &display_name,
            );
        }

        let mut combined_batch = if combined_batches.is_empty() {
            RecordBatch::new_empty(Arc::clone(&combined_schema))
        } else if combined_batches.len() == 1 {
            combined_batches.pop().unwrap()
        } else {
            concat_batches(&combined_schema, &combined_batches).map_err(|e| {
                Error::Internal(format!(
                    "failed to concatenate cross product batches: {}",
                    e
                ))
            })?
        };

        // Sort if needed (before projection, so we have access to all columns and the lookup map)
        if !plan.order_by.is_empty() {
            let mut resolved_order_by = Vec::with_capacity(plan.order_by.len());
            for order in &plan.order_by {
                let resolved_target = match &order.target {
                    OrderTarget::Column(name) => {
                        let col_name = name.to_ascii_lowercase();
                        if let Some(&idx) = column_lookup_map.get(&col_name) {
                            OrderTarget::Index(idx)
                        } else {
                            // Fallback: try to find in schema directly (for unqualified names)
                            if let Ok(idx) = combined_schema.index_of(name) {
                                OrderTarget::Index(idx)
                            } else {
                                return Err(Error::InvalidArgumentError(format!(
                                    "ORDER BY references unknown column '{}'",
                                    name
                                )));
                            }
                        }
                    }
                    other => other.clone(),
                };
                resolved_order_by.push(llkv_plan::OrderByPlan {
                    target: resolved_target,
                    sort_type: order.sort_type.clone(),
                    ascending: order.ascending,
                    nulls_first: order.nulls_first,
                });
            }

            combined_batch = sort_record_batch_with_order(
                &combined_schema,
                &combined_batch,
                &resolved_order_by,
            )?;
        }

        // Apply SELECT projections if specified
        if !plan.projections.is_empty() {
            let mut selected_fields = Vec::new();
            let mut selected_columns = Vec::new();
            let mut expr_context: Option<CrossProductExpressionContext> = None;

            for proj in &plan.projections {
                match proj {
                    SelectProjection::AllColumns => {
                        // Keep all columns
                        selected_fields = combined_schema.fields().iter().cloned().collect();
                        selected_columns = combined_batch.columns().to_vec();
                        break;
                    }
                    SelectProjection::AllColumnsExcept { exclude } => {
                        // Keep all columns except the excluded ones
                        let exclude_lower: Vec<String> =
                            exclude.iter().map(|e| e.to_ascii_lowercase()).collect();

                        let mut excluded_indices = FxHashSet::default();
                        for excluded_name in &exclude_lower {
                            if let Some(&idx) = column_lookup_map.get(excluded_name) {
                                excluded_indices.insert(idx);
                            }
                        }

                        for (idx, field) in combined_schema.fields().iter().enumerate() {
                            let field_name_lower = field.name().to_ascii_lowercase();
                            if !exclude_lower.contains(&field_name_lower)
                                && !excluded_indices.contains(&idx)
                            {
                                selected_fields.push(field.clone());
                                selected_columns.push(combined_batch.column(idx).clone());
                            }
                        }
                        break;
                    }
                    SelectProjection::Column { name, alias } => {
                        // Find the column by qualified name
                        let col_name = name.to_ascii_lowercase();
                        if let Some(&idx) = column_lookup_map.get(&col_name) {
                            let field = combined_schema.field(idx);
                            let output_name = alias.as_ref().unwrap_or(name).clone();
                            selected_fields.push(Arc::new(arrow::datatypes::Field::new(
                                output_name,
                                field.data_type().clone(),
                                field.is_nullable(),
                            )));
                            selected_columns.push(combined_batch.column(idx).clone());
                        } else {
                            return Err(Error::InvalidArgumentError(format!(
                                "column '{}' not found in cross product result",
                                name
                            )));
                        }
                    }
                    SelectProjection::Computed { expr, alias } => {
                        if expr_context.is_none() {
                            expr_context = Some(CrossProductExpressionContext::new(
                                combined_schema.as_ref(),
                                column_lookup_map.clone(),
                            )?);
                        }
                        let context = expr_context
                            .as_mut()
                            .expect("projection context must be initialized");
                        context.reset();
                        let evaluated = self.evaluate_projection_expression(
                            context,
                            expr,
                            &combined_batch,
                            &scalar_lookup,
                        )?;
                        let field = Arc::new(arrow::datatypes::Field::new(
                            alias.clone(),
                            evaluated.data_type().clone(),
                            true,
                        ));
                        selected_fields.push(field);
                        selected_columns.push(evaluated);
                    }
                }
            }

            let projected_schema = Arc::new(Schema::new(selected_fields));
            combined_batch = RecordBatch::try_new(projected_schema, selected_columns)
                .map_err(|e| Error::Internal(format!("failed to apply projections: {}", e)))?;
        }

        if plan.distinct {
            let mut state = DistinctState::default();
            let source_schema = combined_batch.schema();
            combined_batch = match distinct_filter_batch(combined_batch, &mut state)? {
                Some(filtered) => filtered,
                None => RecordBatch::new_empty(source_schema),
            };
        }

        let schema = combined_batch.schema();

        Ok(SelectExecution::new_single_batch(
            display_name,
            schema,
            combined_batch,
        ))
    }
}

struct JoinKeyBuild {
    keys: Vec<llkv_join::JoinKey>,
    always_true: bool,
    always_false: bool,
}

// Type alias for backward compatibility
#[allow(dead_code)]
type JoinKeyBuildEqualities = JoinKeyBuild;

impl JoinKeyBuild {
    #[allow(dead_code)]
    fn equalities(&self) -> &[llkv_join::JoinKey] {
        &self.keys
    }
}

#[derive(Debug)]
enum JoinConditionAnalysis {
    AlwaysTrue,
    AlwaysFalse,
    EquiPairs(Vec<(String, String)>),
}

fn build_join_keys_from_condition<P>(
    condition: &LlkvExpr<'static, String>,
    left_ref: &llkv_plan::TableRef,
    left_table: &ExecutorTable<P>,
    right_ref: &llkv_plan::TableRef,
    right_table: &ExecutorTable<P>,
) -> ExecutorResult<JoinKeyBuild>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    match analyze_join_condition(condition)? {
        JoinConditionAnalysis::AlwaysTrue => Ok(JoinKeyBuild {
            keys: Vec::new(),
            always_true: true,
            always_false: false,
        }),
        JoinConditionAnalysis::AlwaysFalse => Ok(JoinKeyBuild {
            keys: Vec::new(),
            always_true: false,
            always_false: true,
        }),
        JoinConditionAnalysis::EquiPairs(pairs) => {
            let left_lookup = build_join_column_lookup(left_ref, left_table);
            let right_lookup = build_join_column_lookup(right_ref, right_table);

            let mut keys = Vec::with_capacity(pairs.len());
            for (lhs, rhs) in pairs {
                let (lhs_side, lhs_field) = resolve_join_column(&lhs, &left_lookup, &right_lookup)?;
                let (rhs_side, rhs_field) = resolve_join_column(&rhs, &left_lookup, &right_lookup)?;

                match (lhs_side, rhs_side) {
                    (JoinColumnSide::Left, JoinColumnSide::Right) => {
                        keys.push(llkv_join::JoinKey::new(lhs_field, rhs_field));
                    }
                    (JoinColumnSide::Right, JoinColumnSide::Left) => {
                        keys.push(llkv_join::JoinKey::new(rhs_field, lhs_field));
                    }
                    (JoinColumnSide::Left, JoinColumnSide::Left) => {
                        return Err(Error::InvalidArgumentError(format!(
                            "JOIN condition compares two columns from '{}': '{}' and '{}'",
                            left_ref.display_name(),
                            lhs,
                            rhs
                        )));
                    }
                    (JoinColumnSide::Right, JoinColumnSide::Right) => {
                        return Err(Error::InvalidArgumentError(format!(
                            "JOIN condition compares two columns from '{}': '{}' and '{}'",
                            right_ref.display_name(),
                            lhs,
                            rhs
                        )));
                    }
                }
            }

            Ok(JoinKeyBuild {
                keys,
                always_true: false,
                always_false: false,
            })
        }
    }
}

fn analyze_join_condition(
    expr: &LlkvExpr<'static, String>,
) -> ExecutorResult<JoinConditionAnalysis> {
    match evaluate_constant_join_expr(expr) {
        ConstantJoinEvaluation::Known(true) => {
            return Ok(JoinConditionAnalysis::AlwaysTrue);
        }
        ConstantJoinEvaluation::Known(false) | ConstantJoinEvaluation::Unknown => {
            return Ok(JoinConditionAnalysis::AlwaysFalse);
        }
        ConstantJoinEvaluation::NotConstant => {}
    }
    match expr {
        LlkvExpr::Literal(value) => {
            if *value {
                Ok(JoinConditionAnalysis::AlwaysTrue)
            } else {
                Ok(JoinConditionAnalysis::AlwaysFalse)
            }
        }
        LlkvExpr::And(children) => {
            let mut collected: Vec<(String, String)> = Vec::new();
            for child in children {
                match analyze_join_condition(child)? {
                    JoinConditionAnalysis::AlwaysTrue => {}
                    JoinConditionAnalysis::AlwaysFalse => {
                        return Ok(JoinConditionAnalysis::AlwaysFalse);
                    }
                    JoinConditionAnalysis::EquiPairs(mut pairs) => {
                        collected.append(&mut pairs);
                    }
                }
            }

            if collected.is_empty() {
                Ok(JoinConditionAnalysis::AlwaysTrue)
            } else {
                Ok(JoinConditionAnalysis::EquiPairs(collected))
            }
        }
        LlkvExpr::Compare { left, op, right } => {
            if *op != CompareOp::Eq {
                return Err(Error::InvalidArgumentError(
                    "JOIN ON clause only supports '=' comparisons in optimized path".into(),
                ));
            }
            let left_name = try_extract_simple_column(left).ok_or_else(|| {
                Error::InvalidArgumentError(
                    "JOIN ON clause requires plain column references".into(),
                )
            })?;
            let right_name = try_extract_simple_column(right).ok_or_else(|| {
                Error::InvalidArgumentError(
                    "JOIN ON clause requires plain column references".into(),
                )
            })?;
            Ok(JoinConditionAnalysis::EquiPairs(vec![(
                left_name.to_string(),
                right_name.to_string(),
            )]))
        }
        _ => Err(Error::InvalidArgumentError(
            "JOIN ON expressions must be conjunctions of column equality predicates".into(),
        )),
    }
}

fn compare_literals_with_mode(
    op: CompareOp,
    left: &Literal,
    right: &Literal,
    null_behavior: NullComparisonBehavior,
) -> Option<bool> {
    use std::cmp::Ordering;

    fn ordering_result(ord: Ordering, op: CompareOp) -> bool {
        match op {
            CompareOp::Eq => ord == Ordering::Equal,
            CompareOp::NotEq => ord != Ordering::Equal,
            CompareOp::Lt => ord == Ordering::Less,
            CompareOp::LtEq => ord != Ordering::Greater,
            CompareOp::Gt => ord == Ordering::Greater,
            CompareOp::GtEq => ord != Ordering::Less,
        }
    }

    fn compare_f64(lhs: f64, rhs: f64, op: CompareOp) -> bool {
        match op {
            CompareOp::Eq => lhs == rhs,
            CompareOp::NotEq => lhs != rhs,
            CompareOp::Lt => lhs < rhs,
            CompareOp::LtEq => lhs <= rhs,
            CompareOp::Gt => lhs > rhs,
            CompareOp::GtEq => lhs >= rhs,
        }
    }

    match (left, right) {
        (Literal::Null, _) | (_, Literal::Null) => match null_behavior {
            NullComparisonBehavior::ThreeValuedLogic => None,
        },
        (Literal::Int128(lhs), Literal::Int128(rhs)) => Some(ordering_result(lhs.cmp(rhs), op)),
        (Literal::Float64(lhs), Literal::Float64(rhs)) => Some(compare_f64(*lhs, *rhs, op)),
        (Literal::Int128(lhs), Literal::Float64(rhs)) => Some(compare_f64(*lhs as f64, *rhs, op)),
        (Literal::Float64(lhs), Literal::Int128(rhs)) => Some(compare_f64(*lhs, *rhs as f64, op)),
        (Literal::Boolean(lhs), Literal::Boolean(rhs)) => Some(ordering_result(lhs.cmp(rhs), op)),
        (Literal::String(lhs), Literal::String(rhs)) => Some(ordering_result(lhs.cmp(rhs), op)),
        (Literal::Decimal128(lhs), Literal::Decimal128(rhs)) => {
            llkv_compute::scalar::decimal::compare(*lhs, *rhs)
                .ok()
                .map(|ord| ordering_result(ord, op))
        }
        (Literal::Decimal128(lhs), Literal::Int128(rhs)) => {
            DecimalValue::new(*rhs, 0).ok().and_then(|rhs_dec| {
                llkv_compute::scalar::decimal::compare(*lhs, rhs_dec)
                    .ok()
                    .map(|ord| ordering_result(ord, op))
            })
        }
        (Literal::Int128(lhs), Literal::Decimal128(rhs)) => {
            DecimalValue::new(*lhs, 0).ok().and_then(|lhs_dec| {
                llkv_compute::scalar::decimal::compare(lhs_dec, *rhs)
                    .ok()
                    .map(|ord| ordering_result(ord, op))
            })
        }
        (Literal::Decimal128(lhs), Literal::Float64(rhs)) => {
            Some(compare_f64(lhs.to_f64(), *rhs, op))
        }
        (Literal::Float64(lhs), Literal::Decimal128(rhs)) => {
            Some(compare_f64(*lhs, rhs.to_f64(), op))
        }
        (Literal::Struct(_), _) | (_, Literal::Struct(_)) => None,
        _ => None,
    }
}

fn build_no_match_join_batches<P>(
    join_type: llkv_join::JoinType,
    left_ref: &llkv_plan::TableRef,
    left_table: &ExecutorTable<P>,
    right_ref: &llkv_plan::TableRef,
    right_table: &ExecutorTable<P>,
    combined_schema: Arc<Schema>,
) -> ExecutorResult<Vec<RecordBatch>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    match join_type {
        llkv_join::JoinType::Inner => Ok(Vec::new()),
        llkv_join::JoinType::Left => {
            let left_batches = scan_all_columns_for_join(left_ref, left_table)?;
            let mut results = Vec::new();

            for left_batch in left_batches {
                let row_count = left_batch.num_rows();
                if row_count == 0 {
                    continue;
                }

                let mut columns = Vec::with_capacity(combined_schema.fields().len());
                columns.extend(left_batch.columns().iter().cloned());
                for column in &right_table.schema.columns {
                    columns.push(new_null_array(&column.data_type, row_count));
                }

                let batch =
                    RecordBatch::try_new(Arc::clone(&combined_schema), columns).map_err(|err| {
                        Error::Internal(format!("failed to build LEFT JOIN fallback batch: {err}"))
                    })?;
                results.push(batch);
            }

            Ok(results)
        }
        llkv_join::JoinType::Right => {
            let right_batches = scan_all_columns_for_join(right_ref, right_table)?;
            let mut results = Vec::new();

            for right_batch in right_batches {
                let row_count = right_batch.num_rows();
                if row_count == 0 {
                    continue;
                }

                let mut columns = Vec::with_capacity(combined_schema.fields().len());
                for column in &left_table.schema.columns {
                    columns.push(new_null_array(&column.data_type, row_count));
                }
                columns.extend(right_batch.columns().iter().cloned());

                let batch =
                    RecordBatch::try_new(Arc::clone(&combined_schema), columns).map_err(|err| {
                        Error::Internal(format!("failed to build RIGHT JOIN fallback batch: {err}"))
                    })?;
                results.push(batch);
            }

            Ok(results)
        }
        llkv_join::JoinType::Full => {
            let mut results = Vec::new();

            let left_batches = scan_all_columns_for_join(left_ref, left_table)?;
            for left_batch in left_batches {
                let row_count = left_batch.num_rows();
                if row_count == 0 {
                    continue;
                }

                let mut columns = Vec::with_capacity(combined_schema.fields().len());
                columns.extend(left_batch.columns().iter().cloned());
                for column in &right_table.schema.columns {
                    columns.push(new_null_array(&column.data_type, row_count));
                }

                let batch =
                    RecordBatch::try_new(Arc::clone(&combined_schema), columns).map_err(|err| {
                        Error::Internal(format!(
                            "failed to build FULL JOIN left fallback batch: {err}"
                        ))
                    })?;
                results.push(batch);
            }

            let right_batches = scan_all_columns_for_join(right_ref, right_table)?;
            for right_batch in right_batches {
                let row_count = right_batch.num_rows();
                if row_count == 0 {
                    continue;
                }

                let mut columns = Vec::with_capacity(combined_schema.fields().len());
                for column in &left_table.schema.columns {
                    columns.push(new_null_array(&column.data_type, row_count));
                }
                columns.extend(right_batch.columns().iter().cloned());

                let batch =
                    RecordBatch::try_new(Arc::clone(&combined_schema), columns).map_err(|err| {
                        Error::Internal(format!(
                            "failed to build FULL JOIN right fallback batch: {err}"
                        ))
                    })?;
                results.push(batch);
            }

            Ok(results)
        }
        other => Err(Error::InvalidArgumentError(format!(
            "{other:?} join type is not supported when join predicate is unsatisfiable",
        ))),
    }
}

fn scan_all_columns_for_join<P>(
    table_ref: &llkv_plan::TableRef,
    table: &ExecutorTable<P>,
) -> ExecutorResult<Vec<RecordBatch>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    if table.schema.columns.is_empty() {
        return Err(Error::InvalidArgumentError(format!(
            "table '{}' has no columns; joins require at least one column",
            table_ref.qualified_name()
        )));
    }

    let mut projections = Vec::with_capacity(table.schema.columns.len());
    for column in &table.schema.columns {
        projections.push(ScanProjection::from(StoreProjection::with_alias(
            LogicalFieldId::for_user(table.table.table_id(), column.field_id),
            column.name.clone(),
        )));
    }

    let filter_field = table.schema.first_field_id().unwrap_or(ROW_ID_FIELD_ID);
    let filter_expr = full_table_scan_filter(filter_field);

    let mut batches = Vec::new();
    table.table.scan_stream(
        projections,
        &filter_expr,
        ScanStreamOptions {
            include_nulls: true,
            ..ScanStreamOptions::default()
        },
        |batch| {
            batches.push(batch);
        },
    )?;

    Ok(batches)
}

fn build_join_column_lookup<P>(
    table_ref: &llkv_plan::TableRef,
    table: &ExecutorTable<P>,
) -> FxHashMap<String, FieldId>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut lookup = FxHashMap::default();
    let table_lower = table_ref.table.to_ascii_lowercase();
    let qualified_lower = table_ref.qualified_name().to_ascii_lowercase();
    let display_lower = table_ref.display_name().to_ascii_lowercase();
    let alias_lower = table_ref.alias.as_ref().map(|s| s.to_ascii_lowercase());
    let schema_lower = if table_ref.schema.is_empty() {
        None
    } else {
        Some(table_ref.schema.to_ascii_lowercase())
    };

    for column in &table.schema.columns {
        let base = column.name.to_ascii_lowercase();
        let short = base.rsplit('.').next().unwrap_or(base.as_str()).to_string();

        lookup.entry(short.clone()).or_insert(column.field_id);
        lookup.entry(base.clone()).or_insert(column.field_id);

        lookup
            .entry(format!("{table_lower}.{short}"))
            .or_insert(column.field_id);

        if display_lower != table_lower {
            lookup
                .entry(format!("{display_lower}.{short}"))
                .or_insert(column.field_id);
        }

        if qualified_lower != table_lower {
            lookup
                .entry(format!("{qualified_lower}.{short}"))
                .or_insert(column.field_id);
        }

        if let Some(schema) = &schema_lower {
            lookup
                .entry(format!("{schema}.{table_lower}.{short}"))
                .or_insert(column.field_id);
            if display_lower != table_lower {
                lookup
                    .entry(format!("{schema}.{display_lower}.{short}"))
                    .or_insert(column.field_id);
            }
        }

        if let Some(alias) = &alias_lower {
            lookup
                .entry(format!("{alias}.{short}"))
                .or_insert(column.field_id);
        }
    }

    lookup
}

#[derive(Clone, Copy)]
enum JoinColumnSide {
    Left,
    Right,
}

fn resolve_join_column(
    column: &str,
    left_lookup: &FxHashMap<String, FieldId>,
    right_lookup: &FxHashMap<String, FieldId>,
) -> ExecutorResult<(JoinColumnSide, FieldId)> {
    let key = column.to_ascii_lowercase();
    match (left_lookup.get(&key), right_lookup.get(&key)) {
        (Some(&field_id), None) => Ok((JoinColumnSide::Left, field_id)),
        (None, Some(&field_id)) => Ok((JoinColumnSide::Right, field_id)),
        (Some(_), Some(_)) => Err(Error::InvalidArgumentError(format!(
            "join column '{column}' is ambiguous; qualify it with a table name or alias",
        ))),
        (None, None) => Err(Error::InvalidArgumentError(format!(
            "join column '{column}' was not found in either table",
        ))),
    }
}

/// Execute a high-performance hash join between two RecordBatches.
///
/// This implements an optimized hash join algorithm using FxHashMap for maximum performance:
/// 1. Build phase: Hash all rows from the right (build) side by join keys
/// 2. Probe phase: For each left row, look up matches in the hash table and emit joined rows
///
/// Performance characteristics:
/// - O(N + M) time complexity vs O(N × M) for nested loops
/// - Uses FxHashMap (faster than HashMap for integer keys)
/// - Batch output construction for better memory locality
fn execute_hash_join_batches(
    left_schema: &Arc<Schema>,
    left_batches: &[RecordBatch],
    right_schema: &Arc<Schema>,
    right_batches: &[RecordBatch],
    condition: &LlkvExpr<'static, String>,
    join_type: llkv_join::JoinType,
) -> ExecutorResult<Vec<RecordBatch>> {
    // Extract equality conditions from the join predicate
    let equalities = match analyze_join_condition(condition)? {
        JoinConditionAnalysis::AlwaysTrue => {
            // Cross join - return cartesian product
            let mut results = Vec::new();
            for left in left_batches {
                for right in right_batches {
                    results.push(execute_cross_join_batches(left, right)?);
                }
            }
            return Ok(results);
        }
        JoinConditionAnalysis::AlwaysFalse => {
            // No matches - return empty result based on join type
            let combined_fields: Vec<Field> = left_schema
                .fields()
                .iter()
                .chain(right_schema.fields().iter())
                .map(|f| Field::new(f.name().clone(), f.data_type().clone(), f.is_nullable()))
                .collect();
            let combined_schema = Arc::new(Schema::new(combined_fields));

            let mut results = Vec::new();
            match join_type {
                llkv_join::JoinType::Inner
                | llkv_join::JoinType::Semi
                | llkv_join::JoinType::Anti => {
                    results.push(RecordBatch::new_empty(combined_schema));
                }
                llkv_join::JoinType::Left => {
                    for left_batch in left_batches {
                        let row_count = left_batch.num_rows();
                        if row_count == 0 {
                            continue;
                        }
                        let mut columns = Vec::with_capacity(combined_schema.fields().len());
                        columns.extend(left_batch.columns().iter().cloned());
                        for field in right_schema.fields() {
                            columns.push(new_null_array(field.data_type(), row_count));
                        }
                        results.push(
                            RecordBatch::try_new(Arc::clone(&combined_schema), columns)
                                .map_err(|err| {
                                    Error::Internal(format!(
                                        "failed to materialize LEFT JOIN null-extension batch: {err}"
                                    ))
                                })?,
                        );
                    }
                    if results.is_empty() {
                        results.push(RecordBatch::new_empty(combined_schema));
                    }
                }
                llkv_join::JoinType::Right => {
                    for right_batch in right_batches {
                        let row_count = right_batch.num_rows();
                        if row_count == 0 {
                            continue;
                        }
                        let mut columns = Vec::with_capacity(combined_schema.fields().len());
                        for field in left_schema.fields() {
                            columns.push(new_null_array(field.data_type(), row_count));
                        }
                        columns.extend(right_batch.columns().iter().cloned());
                        results.push(
                            RecordBatch::try_new(Arc::clone(&combined_schema), columns)
                                .map_err(|err| {
                                    Error::Internal(format!(
                                        "failed to materialize RIGHT JOIN null-extension batch: {err}"
                                    ))
                                })?,
                        );
                    }
                    if results.is_empty() {
                        results.push(RecordBatch::new_empty(combined_schema));
                    }
                }
                llkv_join::JoinType::Full => {
                    for left_batch in left_batches {
                        let row_count = left_batch.num_rows();
                        if row_count == 0 {
                            continue;
                        }
                        let mut columns = Vec::with_capacity(combined_schema.fields().len());
                        columns.extend(left_batch.columns().iter().cloned());
                        for field in right_schema.fields() {
                            columns.push(new_null_array(field.data_type(), row_count));
                        }
                        results.push(
                            RecordBatch::try_new(Arc::clone(&combined_schema), columns).map_err(
                                |err| {
                                    Error::Internal(format!(
                                        "failed to materialize FULL JOIN left batch: {err}"
                                    ))
                                },
                            )?,
                        );
                    }

                    for right_batch in right_batches {
                        let row_count = right_batch.num_rows();
                        if row_count == 0 {
                            continue;
                        }
                        let mut columns = Vec::with_capacity(combined_schema.fields().len());
                        for field in left_schema.fields() {
                            columns.push(new_null_array(field.data_type(), row_count));
                        }
                        columns.extend(right_batch.columns().iter().cloned());
                        results.push(
                            RecordBatch::try_new(Arc::clone(&combined_schema), columns).map_err(
                                |err| {
                                    Error::Internal(format!(
                                        "failed to materialize FULL JOIN right batch: {err}"
                                    ))
                                },
                            )?,
                        );
                    }

                    if results.is_empty() {
                        results.push(RecordBatch::new_empty(combined_schema));
                    }
                }
            }

            return Ok(results);
        }
        JoinConditionAnalysis::EquiPairs(pairs) => pairs,
    };

    // Build column lookups
    let mut left_lookup: FxHashMap<String, usize> = FxHashMap::default();
    for (idx, field) in left_schema.fields().iter().enumerate() {
        left_lookup.insert(field.name().to_ascii_lowercase(), idx);
    }

    let mut right_lookup: FxHashMap<String, usize> = FxHashMap::default();
    for (idx, field) in right_schema.fields().iter().enumerate() {
        right_lookup.insert(field.name().to_ascii_lowercase(), idx);
    }

    // Map join keys to column indices
    let mut left_key_indices = Vec::new();
    let mut right_key_indices = Vec::new();

    for (lhs_col, rhs_col) in equalities {
        let lhs_lower = lhs_col.to_ascii_lowercase();
        let rhs_lower = rhs_col.to_ascii_lowercase();

        let (left_idx, right_idx) =
            match (left_lookup.get(&lhs_lower), right_lookup.get(&rhs_lower)) {
                (Some(&l), Some(&r)) => (l, r),
                (Some(_), None) => {
                    if left_lookup.contains_key(&rhs_lower) {
                        return Err(Error::InvalidArgumentError(format!(
                            "Both join columns '{}' and '{}' are from left table",
                            lhs_col, rhs_col
                        )));
                    }
                    return Err(Error::InvalidArgumentError(format!(
                        "Join column '{}' not found in right table",
                        rhs_col
                    )));
                }
                (None, Some(_)) => {
                    if right_lookup.contains_key(&lhs_lower) {
                        return Err(Error::InvalidArgumentError(format!(
                            "Both join columns '{}' and '{}' are from right table",
                            lhs_col, rhs_col
                        )));
                    }
                    return Err(Error::InvalidArgumentError(format!(
                        "Join column '{}' not found in left table",
                        lhs_col
                    )));
                }
                (None, None) => {
                    // Try swapped
                    match (left_lookup.get(&rhs_lower), right_lookup.get(&lhs_lower)) {
                        (Some(&l), Some(&r)) => (l, r),
                        _ => {
                            return Err(Error::InvalidArgumentError(format!(
                                "Join columns '{}' and '{}' not found in either table",
                                lhs_col, rhs_col
                            )));
                        }
                    }
                }
            };

        left_key_indices.push(left_idx);
        right_key_indices.push(right_idx);
    }

    // Build hash table from right side (build phase)
    // Key: hash of join column values, Value: (batch_idx, row_idx)
    let mut hash_table: FxHashMap<Vec<i64>, Vec<(usize, usize)>> = FxHashMap::default();

    for (batch_idx, right_batch) in right_batches.iter().enumerate() {
        let num_rows = right_batch.num_rows();
        if num_rows == 0 {
            continue;
        }

        // Extract key columns for hashing
        let key_columns: Vec<&ArrayRef> = right_key_indices
            .iter()
            .map(|&idx| right_batch.column(idx))
            .collect();

        // Build hash table
        for row_idx in 0..num_rows {
            // Extract key values for this row
            let mut key_values = Vec::with_capacity(key_columns.len());
            let mut has_null = false;

            for col in &key_columns {
                if col.is_null(row_idx) {
                    has_null = true;
                    break;
                }
                // Convert to i64 for hashing (works for most numeric types)
                let value = extract_key_value_as_i64(col, row_idx)?;
                key_values.push(value);
            }

            // Skip NULL keys (SQL semantics: NULL != NULL)
            if has_null {
                continue;
            }

            hash_table
                .entry(key_values)
                .or_default()
                .push((batch_idx, row_idx));
        }
    }

    // Probe phase: scan left side and emit matches
    let mut result_batches = Vec::new();
    let combined_fields: Vec<Field> = left_schema
        .fields()
        .iter()
        .chain(right_schema.fields().iter())
        .map(|f| Field::new(f.name().clone(), f.data_type().clone(), true)) // Make nullable for outer joins
        .collect();
    let combined_schema = Arc::new(Schema::new(combined_fields));

    for left_batch in left_batches {
        let num_rows = left_batch.num_rows();
        if num_rows == 0 {
            continue;
        }

        // Extract left key columns
        let left_key_columns: Vec<&ArrayRef> = left_key_indices
            .iter()
            .map(|&idx| left_batch.column(idx))
            .collect();

        // Track which left rows matched (for outer joins)
        let mut left_matched = vec![false; num_rows];

        // Collect matching row pairs
        let mut left_indices = Vec::new();
        let mut right_refs = Vec::new();

        for (left_row_idx, matched) in left_matched.iter_mut().enumerate() {
            // Extract key for this left row
            let mut key_values = Vec::with_capacity(left_key_columns.len());
            let mut has_null = false;

            for col in &left_key_columns {
                if col.is_null(left_row_idx) {
                    has_null = true;
                    break;
                }
                let value = extract_key_value_as_i64(col, left_row_idx)?;
                key_values.push(value);
            }

            if has_null {
                // NULL keys never match
                continue;
            }

            // Look up matches in hash table
            if let Some(right_rows) = hash_table.get(&key_values) {
                *matched = true;
                for &(right_batch_idx, right_row_idx) in right_rows {
                    left_indices.push(left_row_idx as u32);
                    right_refs.push((right_batch_idx, right_row_idx));
                }
            }
        }

        // Build output batch based on join type
        if !left_indices.is_empty() || join_type == llkv_join::JoinType::Left {
            let output_batch = build_join_output_batch(
                left_batch,
                right_batches,
                &left_indices,
                &right_refs,
                &left_matched,
                &combined_schema,
                join_type,
            )?;

            if output_batch.num_rows() > 0 {
                result_batches.push(output_batch);
            }
        }
    }

    if result_batches.is_empty() {
        result_batches.push(RecordBatch::new_empty(combined_schema));
    }

    Ok(result_batches)
}

/// Extract a column value as i64 for hashing (handles various numeric types)
fn extract_key_value_as_i64(col: &ArrayRef, row_idx: usize) -> ExecutorResult<i64> {
    use arrow::array::*;
    use arrow::datatypes::DataType;

    match col.data_type() {
        DataType::Int8 => Ok(col
            .as_any()
            .downcast_ref::<Int8Array>()
            .unwrap()
            .value(row_idx) as i64),
        DataType::Int16 => Ok(col
            .as_any()
            .downcast_ref::<Int16Array>()
            .unwrap()
            .value(row_idx) as i64),
        DataType::Int32 => Ok(col
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .value(row_idx) as i64),
        DataType::Int64 => Ok(col
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(row_idx)),
        DataType::UInt8 => Ok(col
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .value(row_idx) as i64),
        DataType::UInt16 => Ok(col
            .as_any()
            .downcast_ref::<UInt16Array>()
            .unwrap()
            .value(row_idx) as i64),
        DataType::UInt32 => Ok(col
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .value(row_idx) as i64),
        DataType::UInt64 => {
            let val = col
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(row_idx);
            Ok(val as i64) // May overflow but maintains hash consistency
        }
        DataType::Utf8 => {
            // Hash string to i64
            let s = col
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row_idx);
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            s.hash(&mut hasher);
            Ok(hasher.finish() as i64)
        }
        _ => Err(Error::InvalidArgumentError(format!(
            "Unsupported join key type: {:?}",
            col.data_type()
        ))),
    }
}

/// Build the output batch from matched indices
fn build_join_output_batch(
    left_batch: &RecordBatch,
    right_batches: &[RecordBatch],
    left_indices: &[u32],
    right_refs: &[(usize, usize)],
    left_matched: &[bool],
    combined_schema: &Arc<Schema>,
    join_type: llkv_join::JoinType,
) -> ExecutorResult<RecordBatch> {
    use arrow::array::UInt32Array;
    use arrow::compute::take;

    match join_type {
        llkv_join::JoinType::Inner => {
            // Only matched rows
            let left_indices_array = UInt32Array::from(left_indices.to_vec());

            let mut output_columns = Vec::new();

            // Take from left columns
            for col in left_batch.columns() {
                let taken = take(col.as_ref(), &left_indices_array, None)
                    .map_err(|e| Error::Internal(format!("Failed to take left column: {}", e)))?;
                output_columns.push(taken);
            }

            // Gather from right columns
            for right_col_idx in 0..right_batches[0].num_columns() {
                let mut values = Vec::with_capacity(right_refs.len());
                for &(batch_idx, row_idx) in right_refs {
                    let col = right_batches[batch_idx].column(right_col_idx);
                    values.push((col.clone(), row_idx));
                }

                // Build array from gathered values
                let right_col = gather_from_multiple_batches(
                    &values,
                    right_batches[0].column(right_col_idx).data_type(),
                )?;
                output_columns.push(right_col);
            }

            RecordBatch::try_new(Arc::clone(combined_schema), output_columns)
                .map_err(|e| Error::Internal(format!("Failed to create output batch: {}", e)))
        }
        llkv_join::JoinType::Left => {
            // All left rows + matched right rows (nulls for unmatched)
            let mut output_columns = Vec::new();

            // Include all left columns as-is
            for col in left_batch.columns() {
                output_columns.push(col.clone());
            }

            // Build right columns with nulls for unmatched
            for right_col_idx in 0..right_batches[0].num_columns() {
                let right_col = build_left_join_column(
                    left_matched,
                    right_batches,
                    right_col_idx,
                    left_indices,
                    right_refs,
                )?;
                output_columns.push(right_col);
            }

            RecordBatch::try_new(Arc::clone(combined_schema), output_columns)
                .map_err(|e| Error::Internal(format!("Failed to create left join batch: {}", e)))
        }
        _ => Err(Error::InvalidArgumentError(format!(
            "{:?} join not yet implemented in batch join",
            join_type
        ))),
    }
}

/// Gather values from multiple batches at specified positions using Arrow's take kernel.
///
/// This is MUCH faster than copying individual values because it uses Arrow's vectorized
/// take operation which operates on internal buffers. This is DuckDB-level performance.
fn gather_from_multiple_batches(
    values: &[(ArrayRef, usize)],
    _data_type: &DataType,
) -> ExecutorResult<ArrayRef> {
    use arrow::array::*;
    use arrow::compute::take;

    if values.is_empty() {
        return Ok(new_null_array(&DataType::Null, 0));
    }

    // Optimize: if all values come from the same array, use take directly (vectorized, zero-copy)
    if values.len() > 1 {
        let first_array_ptr = Arc::as_ptr(&values[0].0);
        let all_same_array = values
            .iter()
            .all(|(arr, _)| std::ptr::addr_eq(Arc::as_ptr(arr), first_array_ptr));

        if all_same_array {
            // Fast path: single source array - use Arrow's vectorized take kernel
            // This is SIMD-optimized and operates on buffers, not individual values
            let indices: Vec<u32> = values.iter().map(|(_, idx)| *idx as u32).collect();
            let indices_array = UInt32Array::from(indices);
            return take(values[0].0.as_ref(), &indices_array, None)
                .map_err(|e| Error::Internal(format!("Arrow take failed: {}", e)));
        }
    }

    // Multiple source arrays: concatenate then take (still faster than element-by-element)
    // Build a unified index mapping
    use arrow::compute::concat;

    // Group by unique array pointers
    let mut unique_arrays: Vec<(Arc<dyn Array>, Vec<usize>)> = Vec::new();
    let mut array_map: FxHashMap<*const dyn Array, usize> = FxHashMap::default();

    for (arr, row_idx) in values {
        let ptr = Arc::as_ptr(arr);
        if let Some(&idx) = array_map.get(&ptr) {
            unique_arrays[idx].1.push(*row_idx);
        } else {
            let idx = unique_arrays.len();
            array_map.insert(ptr, idx);
            unique_arrays.push((Arc::clone(arr), vec![*row_idx]));
        }
    }

    // If only one unique array, use fast path
    if unique_arrays.len() == 1 {
        let (arr, indices) = &unique_arrays[0];
        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let indices_array = UInt32Array::from(indices_u32);
        return take(arr.as_ref(), &indices_array, None)
            .map_err(|e| Error::Internal(format!("Arrow take failed: {}", e)));
    }

    // Multiple arrays: concat first, then take
    let arrays_to_concat: Vec<&dyn Array> =
        unique_arrays.iter().map(|(arr, _)| arr.as_ref()).collect();

    let concatenated = concat(&arrays_to_concat)
        .map_err(|e| Error::Internal(format!("Arrow concat failed: {}", e)))?;

    // Build adjusted indices for the concatenated array
    let mut offset = 0;
    let mut adjusted_indices = Vec::with_capacity(values.len());
    for (arr, _) in &unique_arrays {
        let arr_len = arr.len();
        for (check_arr, row_idx) in values {
            if Arc::ptr_eq(arr, check_arr) {
                adjusted_indices.push((offset + row_idx) as u32);
            }
        }
        offset += arr_len;
    }

    let indices_array = UInt32Array::from(adjusted_indices);
    take(&concatenated, &indices_array, None)
        .map_err(|e| Error::Internal(format!("Arrow take on concatenated failed: {}", e)))
}

/// Build a column for left join with nulls for unmatched rows
fn build_left_join_column(
    left_matched: &[bool],
    right_batches: &[RecordBatch],
    right_col_idx: usize,
    _left_indices: &[u32],
    _right_refs: &[(usize, usize)],
) -> ExecutorResult<ArrayRef> {
    // This is complex - for now return nulls
    // TODO: Implement proper left join column building
    let data_type = right_batches[0].column(right_col_idx).data_type();
    Ok(new_null_array(data_type, left_matched.len()))
}

/// Execute cross join between two batches
fn execute_cross_join_batches(
    left: &RecordBatch,
    right: &RecordBatch,
) -> ExecutorResult<RecordBatch> {
    let combined_fields: Vec<Field> = left
        .schema()
        .fields()
        .iter()
        .chain(right.schema().fields().iter())
        .map(|f| Field::new(f.name().clone(), f.data_type().clone(), f.is_nullable()))
        .collect();
    let combined_schema = Arc::new(Schema::new(combined_fields));

    cross_join_pair(left, right, &combined_schema)
}

/// Build a temporary in-memory table from RecordBatches for chained joins
#[allow(dead_code)]
fn build_temp_table_from_batches<P>(
    _schema: &Arc<Schema>,
    _batches: &[RecordBatch],
) -> ExecutorResult<llkv_table::Table<P>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    // This function is no longer needed with the new approach
    Err(Error::Internal(
        "build_temp_table_from_batches should not be called".into(),
    ))
}

/// Build join keys from condition for accumulated multi-table join
#[allow(dead_code)]
fn build_join_keys_from_condition_indexed(
    _condition: &LlkvExpr<'static, String>,
    _left_data: &TableCrossProductData,
    _right_data: &TableCrossProductData,
    _left_idx: usize,
    _right_idx: usize,
) -> ExecutorResult<JoinKeyBuild> {
    // This function is no longer needed with the new approach
    Err(Error::Internal(
        "build_join_keys_from_condition_indexed should not be called".into(),
    ))
}

#[cfg(test)]
mod join_condition_tests {
    use super::*;
    use llkv_expr::expr::{CompareOp, ScalarExpr};
    use llkv_expr::literal::Literal;

    #[test]
    fn analyze_detects_simple_equality() {
        let expr = LlkvExpr::Compare {
            left: ScalarExpr::Column("t1.col".into()),
            op: CompareOp::Eq,
            right: ScalarExpr::Column("t2.col".into()),
        };

        match analyze_join_condition(&expr).expect("analysis succeeds") {
            JoinConditionAnalysis::EquiPairs(pairs) => {
                assert_eq!(pairs, vec![("t1.col".to_string(), "t2.col".to_string())]);
            }
            other => panic!("unexpected analysis result: {other:?}"),
        }
    }

    #[test]
    fn analyze_handles_literal_true() {
        let expr = LlkvExpr::Literal(true);
        assert!(matches!(
            analyze_join_condition(&expr).expect("analysis succeeds"),
            JoinConditionAnalysis::AlwaysTrue
        ));
    }

    #[test]
    fn analyze_rejects_non_equality() {
        let expr = LlkvExpr::Compare {
            left: ScalarExpr::Column("t1.col".into()),
            op: CompareOp::Gt,
            right: ScalarExpr::Column("t2.col".into()),
        };
        assert!(analyze_join_condition(&expr).is_err());
    }

    #[test]
    fn analyze_handles_constant_is_not_null() {
        let expr = LlkvExpr::IsNull {
            expr: ScalarExpr::Literal(Literal::Null),
            negated: true,
        };

        assert!(matches!(
            analyze_join_condition(&expr).expect("analysis succeeds"),
            JoinConditionAnalysis::AlwaysFalse
        ));
    }

    #[test]
    fn analyze_handles_not_applied_to_is_not_null() {
        let expr = LlkvExpr::Not(Box::new(LlkvExpr::IsNull {
            expr: ScalarExpr::Literal(Literal::Int128(86)),
            negated: true,
        }));

        assert!(matches!(
            analyze_join_condition(&expr).expect("analysis succeeds"),
            JoinConditionAnalysis::AlwaysFalse
        ));
    }

    #[test]
    fn analyze_literal_is_null_is_always_false() {
        let expr = LlkvExpr::IsNull {
            expr: ScalarExpr::Literal(Literal::Int128(1)),
            negated: false,
        };

        assert!(matches!(
            analyze_join_condition(&expr).expect("analysis succeeds"),
            JoinConditionAnalysis::AlwaysFalse
        ));
    }

    #[test]
    fn analyze_not_null_comparison_is_always_false() {
        let expr = LlkvExpr::Not(Box::new(LlkvExpr::Compare {
            left: ScalarExpr::Literal(Literal::Null),
            op: CompareOp::Lt,
            right: ScalarExpr::Column("t2.col".into()),
        }));

        assert!(matches!(
            analyze_join_condition(&expr).expect("analysis succeeds"),
            JoinConditionAnalysis::AlwaysFalse
        ));
    }
}

#[cfg(test)]
mod cross_join_batch_tests {
    use super::*;
    use arrow::array::Int32Array;

    #[test]
    fn execute_cross_join_batches_emits_full_cartesian_product() {
        let left_schema = Arc::new(Schema::new(vec![Field::new("l", DataType::Int32, false)]));
        let right_schema = Arc::new(Schema::new(vec![Field::new("r", DataType::Int32, false)]));

        let left_batch = RecordBatch::try_new(
            Arc::clone(&left_schema),
            vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef],
        )
        .expect("left batch");
        let right_batch = RecordBatch::try_new(
            Arc::clone(&right_schema),
            vec![Arc::new(Int32Array::from(vec![10, 20, 30])) as ArrayRef],
        )
        .expect("right batch");

        let result = execute_cross_join_batches(&left_batch, &right_batch).expect("cross join");

        assert_eq!(result.num_rows(), 6);
        assert_eq!(result.num_columns(), 2);

        let left_values: Vec<i32> = {
            let array = result
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            (0..array.len()).map(|idx| array.value(idx)).collect()
        };
        let right_values: Vec<i32> = {
            let array = result
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            (0..array.len()).map(|idx| array.value(idx)).collect()
        };

        assert_eq!(left_values, vec![1, 1, 1, 2, 2, 2]);
        assert_eq!(right_values, vec![10, 20, 30, 10, 20, 30]);
    }
}

impl<P> QueryExecutor<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn execute_cross_product_aggregates(
        &self,
        combined_schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
        column_lookup_map: &FxHashMap<String, usize>,
        plan: &SelectPlan,
        display_name: &str,
    ) -> ExecutorResult<SelectExecution<P>> {
        if !plan.scalar_subqueries.is_empty() {
            return Err(Error::InvalidArgumentError(
                "scalar subqueries not supported in aggregate joins".into(),
            ));
        }

        let mut specs: Vec<AggregateSpec> = Vec::with_capacity(plan.aggregates.len());
        let mut spec_to_projection: Vec<Option<usize>> = Vec::with_capacity(plan.aggregates.len());

        for aggregate in &plan.aggregates {
            match aggregate {
                AggregateExpr::CountStar { alias, distinct } => {
                    specs.push(AggregateSpec {
                        alias: alias.clone(),
                        kind: AggregateKind::Count {
                            field_id: None,
                            distinct: *distinct,
                        },
                    });
                    spec_to_projection.push(None);
                }
                AggregateExpr::Column {
                    column,
                    alias,
                    function,
                    distinct,
                } => {
                    let key = column.to_ascii_lowercase();
                    let column_index = *column_lookup_map.get(&key).ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "unknown column '{column}' in aggregate"
                        ))
                    })?;
                    let field = combined_schema.field(column_index);
                    let kind = match function {
                        AggregateFunction::Count => AggregateKind::Count {
                            field_id: Some(column_index as u32),
                            distinct: *distinct,
                        },
                        AggregateFunction::SumInt64 => {
                            let input_type = Self::validate_aggregate_type(
                                Some(field.data_type().clone()),
                                "SUM",
                                &[DataType::Int64, DataType::Float64],
                            )?;
                            AggregateKind::Sum {
                                field_id: column_index as u32,
                                data_type: input_type,
                                distinct: *distinct,
                            }
                        }
                        AggregateFunction::TotalInt64 => {
                            let input_type = Self::validate_aggregate_type(
                                Some(field.data_type().clone()),
                                "TOTAL",
                                &[DataType::Int64, DataType::Float64],
                            )?;
                            AggregateKind::Total {
                                field_id: column_index as u32,
                                data_type: input_type,
                                distinct: *distinct,
                            }
                        }
                        AggregateFunction::MinInt64 => {
                            let input_type = Self::validate_aggregate_type(
                                Some(field.data_type().clone()),
                                "MIN",
                                &[DataType::Int64, DataType::Float64],
                            )?;
                            AggregateKind::Min {
                                field_id: column_index as u32,
                                data_type: input_type,
                            }
                        }
                        AggregateFunction::MaxInt64 => {
                            let input_type = Self::validate_aggregate_type(
                                Some(field.data_type().clone()),
                                "MAX",
                                &[DataType::Int64, DataType::Float64],
                            )?;
                            AggregateKind::Max {
                                field_id: column_index as u32,
                                data_type: input_type,
                            }
                        }
                        AggregateFunction::CountNulls => AggregateKind::CountNulls {
                            field_id: column_index as u32,
                        },
                        AggregateFunction::GroupConcat => AggregateKind::GroupConcat {
                            field_id: column_index as u32,
                            distinct: *distinct,
                            separator: ",".to_string(),
                        },
                    };

                    specs.push(AggregateSpec {
                        alias: alias.clone(),
                        kind,
                    });
                    spec_to_projection.push(Some(column_index));
                }
            }
        }

        if specs.is_empty() {
            return Err(Error::InvalidArgumentError(
                "aggregate query requires at least one aggregate expression".into(),
            ));
        }

        let mut states = Vec::with_capacity(specs.len());
        for (idx, spec) in specs.iter().enumerate() {
            states.push(AggregateState {
                alias: spec.alias.clone(),
                accumulator: AggregateAccumulator::new_with_projection_index(
                    spec,
                    spec_to_projection[idx],
                    None,
                )?,
                override_value: None,
            });
        }

        for batch in &batches {
            for state in &mut states {
                state.update(batch)?;
            }
        }

        let mut fields = Vec::with_capacity(states.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(states.len());
        for state in states {
            let (field, array) = state.finalize()?;
            fields.push(Arc::new(field));
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        let mut batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;

        if plan.distinct {
            let mut distinct_state = DistinctState::default();
            batch = match distinct_filter_batch(batch, &mut distinct_state)? {
                Some(filtered) => filtered,
                None => RecordBatch::new_empty(Arc::clone(&schema)),
            };
        }

        if !plan.order_by.is_empty() && batch.num_rows() > 0 {
            batch = sort_record_batch_with_order(&schema, &batch, &plan.order_by)?;
        }

        Ok(SelectExecution::new_single_batch(
            display_name.to_string(),
            schema,
            batch,
        ))
    }

    fn execute_cross_product_computed_aggregates(
        &self,
        combined_schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
        column_lookup_map: &FxHashMap<String, usize>,
        plan: &SelectPlan,
        display_name: &str,
    ) -> ExecutorResult<SelectExecution<P>> {
        let mut aggregate_specs: Vec<(String, AggregateCall<String>)> = Vec::new();
        for projection in &plan.projections {
            match projection {
                SelectProjection::Computed { expr, .. } => {
                    Self::collect_aggregates(expr, &mut aggregate_specs);
                }
                SelectProjection::AllColumns
                | SelectProjection::AllColumnsExcept { .. }
                | SelectProjection::Column { .. } => {
                    return Err(Error::InvalidArgumentError(
                        "non-computed projections not supported with aggregate expressions".into(),
                    ));
                }
            }
        }

        if aggregate_specs.is_empty() {
            return Err(Error::InvalidArgumentError(
                "computed aggregate query requires at least one aggregate expression".into(),
            ));
        }

        let aggregate_values = self.compute_cross_product_aggregate_values(
            &combined_schema,
            &batches,
            column_lookup_map,
            &aggregate_specs,
        )?;

        let mut fields = Vec::with_capacity(plan.projections.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(plan.projections.len());

        for projection in &plan.projections {
            if let SelectProjection::Computed { expr, alias } = projection {
                // Check if this is a simple aggregate expression
                if let ScalarExpr::Aggregate(agg) = expr {
                    let key = format!("{:?}", agg);
                    if let Some(agg_value) = aggregate_values.get(&key) {
                        match agg_value {
                            AggregateValue::Null => {
                                fields.push(Arc::new(Field::new(alias, DataType::Int64, true)));
                                arrays.push(Arc::new(Int64Array::from(vec![None::<i64>])) as ArrayRef);
                            }
                            AggregateValue::Int64(v) => {
                                fields.push(Arc::new(Field::new(alias, DataType::Int64, true)));
                                arrays.push(Arc::new(Int64Array::from(vec![Some(*v)])) as ArrayRef);
                            }
                            AggregateValue::Float64(v) => {
                                fields.push(Arc::new(Field::new(alias, DataType::Float64, true)));
                                arrays
                                    .push(Arc::new(Float64Array::from(vec![Some(*v)])) as ArrayRef);
                            }
                            AggregateValue::Decimal128 { value, scale } => {
                                // Determine precision from the value
                                let precision = if *value == 0 {
                                    1
                                } else {
                                    (*value).abs().to_string().len() as u8
                                };
                                fields.push(Arc::new(Field::new(
                                    alias,
                                    DataType::Decimal128(precision, *scale),
                                    true,
                                )));
                                let array = Decimal128Array::from(vec![Some(*value)])
                                    .with_precision_and_scale(precision, *scale)
                                    .map_err(|e| {
                                        Error::Internal(format!("invalid Decimal128: {}", e))
                                    })?;
                                arrays.push(Arc::new(array) as ArrayRef);
                            }
                            AggregateValue::String(s) => {
                                fields.push(Arc::new(Field::new(alias, DataType::Utf8, true)));
                                arrays
                                    .push(Arc::new(StringArray::from(vec![Some(s.as_str())]))
                                        as ArrayRef);
                            }
                        }
                        continue;
                    }
                }

                // Complex expression - try to evaluate as integer
                let value = Self::evaluate_expr_with_aggregates(expr, &aggregate_values)?;
                fields.push(Arc::new(Field::new(alias, DataType::Int64, true)));
                arrays.push(Arc::new(Int64Array::from(vec![value])) as ArrayRef);
            }
        }

        let schema = Arc::new(Schema::new(fields));
        let mut batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;

        if plan.distinct {
            let mut distinct_state = DistinctState::default();
            batch = match distinct_filter_batch(batch, &mut distinct_state)? {
                Some(filtered) => filtered,
                None => RecordBatch::new_empty(Arc::clone(&schema)),
            };
        }

        if !plan.order_by.is_empty() && batch.num_rows() > 0 {
            batch = sort_record_batch_with_order(&schema, &batch, &plan.order_by)?;
        }

        Ok(SelectExecution::new_single_batch(
            display_name.to_string(),
            schema,
            batch,
        ))
    }

    fn compute_cross_product_aggregate_values(
        &self,
        combined_schema: &Arc<Schema>,
        batches: &[RecordBatch],
        column_lookup_map: &FxHashMap<String, usize>,
        aggregate_specs: &[(String, AggregateCall<String>)],
    ) -> ExecutorResult<FxHashMap<String, AggregateValue>> {
        let mut specs: Vec<AggregateSpec> = Vec::with_capacity(aggregate_specs.len());
        let mut spec_to_projection: Vec<Option<usize>> = Vec::with_capacity(aggregate_specs.len());

        let mut columns_per_batch: Option<Vec<Vec<ArrayRef>>> = None;
        let mut augmented_fields: Option<Vec<Field>> = None;
        let mut owned_batches: Option<Vec<RecordBatch>> = None;
        let mut computed_projection_cache: FxHashMap<String, (usize, DataType)> =
            FxHashMap::default();
        let mut computed_alias_counter: usize = 0;
        let mut expr_context = CrossProductExpressionContext::new(
            combined_schema.as_ref(),
            column_lookup_map.clone(),
        )?;

        let mut ensure_computed_column =
            |expr: &ScalarExpr<String>| -> ExecutorResult<(usize, DataType)> {
                let key = format!("{:?}", expr);
                if let Some((idx, dtype)) = computed_projection_cache.get(&key) {
                    return Ok((*idx, dtype.clone()));
                }

                if columns_per_batch.is_none() {
                    let initial_columns: Vec<Vec<ArrayRef>> = batches
                        .iter()
                        .map(|batch| batch.columns().to_vec())
                        .collect();
                    columns_per_batch = Some(initial_columns);
                }
                if augmented_fields.is_none() {
                    augmented_fields = Some(
                        combined_schema
                            .fields()
                            .iter()
                            .map(|field| field.as_ref().clone())
                            .collect(),
                    );
                }

                let translated = translate_scalar(expr, expr_context.schema(), |name| {
                    Error::InvalidArgumentError(format!(
                        "unknown column '{}' in aggregate expression",
                        name
                    ))
                })?;
                let data_type = infer_computed_data_type(expr_context.schema(), &translated)?;

                if let Some(columns) = columns_per_batch.as_mut() {
                    for (batch_idx, batch) in batches.iter().enumerate() {
                        expr_context.reset();
                        let array = expr_context.materialize_scalar_array(&translated, batch)?;
                        if let Some(batch_columns) = columns.get_mut(batch_idx) {
                            batch_columns.push(array);
                        }
                    }
                }

                let column_index = augmented_fields
                    .as_ref()
                    .map(|fields| fields.len())
                    .unwrap_or_else(|| combined_schema.fields().len());

                let alias = format!("__agg_expr_cp_{}", computed_alias_counter);
                computed_alias_counter += 1;
                augmented_fields
                    .as_mut()
                    .expect("augmented fields initialized")
                    .push(Field::new(&alias, data_type.clone(), true));

                computed_projection_cache.insert(key, (column_index, data_type.clone()));
                Ok((column_index, data_type))
            };

        for (key, agg) in aggregate_specs {
            match agg {
                AggregateCall::CountStar => {
                    specs.push(AggregateSpec {
                        alias: key.clone(),
                        kind: AggregateKind::Count {
                            field_id: None,
                            distinct: false,
                        },
                    });
                    spec_to_projection.push(None);
                }
                AggregateCall::Count { expr, .. }
                | AggregateCall::Sum { expr, .. }
                | AggregateCall::Total { expr, .. }
                | AggregateCall::Avg { expr, .. }
                | AggregateCall::Min(expr)
                | AggregateCall::Max(expr)
                | AggregateCall::CountNulls(expr)
                | AggregateCall::GroupConcat { expr, .. } => {
                    let (column_index, data_type_opt) = if let Some(column) =
                        try_extract_simple_column(expr)
                    {
                        let key_lower = column.to_ascii_lowercase();
                        let column_index = *column_lookup_map.get(&key_lower).ok_or_else(|| {
                            Error::InvalidArgumentError(format!(
                                "unknown column '{column}' in aggregate"
                            ))
                        })?;
                        let field = combined_schema.field(column_index);
                        (column_index, Some(field.data_type().clone()))
                    } else {
                        let (index, dtype) = ensure_computed_column(expr)?;
                        (index, Some(dtype))
                    };

                    let kind = match agg {
                        AggregateCall::Count { distinct, .. } => {
                            let field_id = u32::try_from(column_index).map_err(|_| {
                                Error::InvalidArgumentError(
                                    "aggregate projection index exceeds supported range".into(),
                                )
                            })?;
                            AggregateKind::Count {
                                field_id: Some(field_id),
                                distinct: *distinct,
                            }
                        }
                        AggregateCall::Sum { distinct, .. } => {
                            let input_type = Self::validate_aggregate_type(
                                data_type_opt.clone(),
                                "SUM",
                                &[DataType::Int64, DataType::Float64],
                            )?;
                            let field_id = u32::try_from(column_index).map_err(|_| {
                                Error::InvalidArgumentError(
                                    "aggregate projection index exceeds supported range".into(),
                                )
                            })?;
                            AggregateKind::Sum {
                                field_id,
                                data_type: input_type,
                                distinct: *distinct,
                            }
                        }
                        AggregateCall::Total { distinct, .. } => {
                            let input_type = Self::validate_aggregate_type(
                                data_type_opt.clone(),
                                "TOTAL",
                                &[DataType::Int64, DataType::Float64],
                            )?;
                            let field_id = u32::try_from(column_index).map_err(|_| {
                                Error::InvalidArgumentError(
                                    "aggregate projection index exceeds supported range".into(),
                                )
                            })?;
                            AggregateKind::Total {
                                field_id,
                                data_type: input_type,
                                distinct: *distinct,
                            }
                        }
                        AggregateCall::Avg { distinct, .. } => {
                            let input_type = Self::validate_aggregate_type(
                                data_type_opt.clone(),
                                "AVG",
                                &[DataType::Int64, DataType::Float64],
                            )?;
                            let field_id = u32::try_from(column_index).map_err(|_| {
                                Error::InvalidArgumentError(
                                    "aggregate projection index exceeds supported range".into(),
                                )
                            })?;
                            AggregateKind::Avg {
                                field_id,
                                data_type: input_type,
                                distinct: *distinct,
                            }
                        }
                        AggregateCall::Min(_) => {
                            let input_type = Self::validate_aggregate_type(
                                data_type_opt.clone(),
                                "MIN",
                                &[DataType::Int64, DataType::Float64],
                            )?;
                            let field_id = u32::try_from(column_index).map_err(|_| {
                                Error::InvalidArgumentError(
                                    "aggregate projection index exceeds supported range".into(),
                                )
                            })?;
                            AggregateKind::Min {
                                field_id,
                                data_type: input_type,
                            }
                        }
                        AggregateCall::Max(_) => {
                            let input_type = Self::validate_aggregate_type(
                                data_type_opt.clone(),
                                "MAX",
                                &[DataType::Int64, DataType::Float64],
                            )?;
                            let field_id = u32::try_from(column_index).map_err(|_| {
                                Error::InvalidArgumentError(
                                    "aggregate projection index exceeds supported range".into(),
                                )
                            })?;
                            AggregateKind::Max {
                                field_id,
                                data_type: input_type,
                            }
                        }
                        AggregateCall::CountNulls(_) => {
                            let field_id = u32::try_from(column_index).map_err(|_| {
                                Error::InvalidArgumentError(
                                    "aggregate projection index exceeds supported range".into(),
                                )
                            })?;
                            AggregateKind::CountNulls { field_id }
                        }
                        AggregateCall::GroupConcat {
                            distinct,
                            separator,
                            ..
                        } => {
                            let field_id = u32::try_from(column_index).map_err(|_| {
                                Error::InvalidArgumentError(
                                    "aggregate projection index exceeds supported range".into(),
                                )
                            })?;
                            AggregateKind::GroupConcat {
                                field_id,
                                distinct: *distinct,
                                separator: separator.clone().unwrap_or_else(|| ",".to_string()),
                            }
                        }
                        _ => unreachable!(),
                    };

                    specs.push(AggregateSpec {
                        alias: key.clone(),
                        kind,
                    });
                    spec_to_projection.push(Some(column_index));
                }
            }
        }

        if let Some(columns) = columns_per_batch {
            let fields = augmented_fields.unwrap_or_else(|| {
                combined_schema
                    .fields()
                    .iter()
                    .map(|field| field.as_ref().clone())
                    .collect()
            });
            let augmented_schema = Arc::new(Schema::new(fields));
            let mut new_batches = Vec::with_capacity(columns.len());
            for batch_columns in columns {
                let batch = RecordBatch::try_new(Arc::clone(&augmented_schema), batch_columns)
                    .map_err(|err| {
                        Error::InvalidArgumentError(format!(
                            "failed to materialize aggregate projections: {err}"
                        ))
                    })?;
                new_batches.push(batch);
            }
            owned_batches = Some(new_batches);
        }

        let mut states = Vec::with_capacity(specs.len());
        for (idx, spec) in specs.iter().enumerate() {
            states.push(AggregateState {
                alias: spec.alias.clone(),
                accumulator: AggregateAccumulator::new_with_projection_index(
                    spec,
                    spec_to_projection[idx],
                    None,
                )?,
                override_value: None,
            });
        }

        let batch_iter: &[RecordBatch] = if let Some(ref extended) = owned_batches {
            extended.as_slice()
        } else {
            batches
        };

        for batch in batch_iter {
            for state in &mut states {
                state.update(batch)?;
            }
        }

        let mut results = FxHashMap::default();
        for state in states {
            let (field, array) = state.finalize()?;

            // Try Int64Array first
            if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                if int_array.len() != 1 {
                    return Err(Error::Internal(format!(
                        "Expected single value from aggregate, got {}",
                        int_array.len()
                    )));
                }
                let value = if int_array.is_null(0) {
                    AggregateValue::Null
                } else {
                    AggregateValue::Int64(int_array.value(0))
                };
                results.insert(field.name().to_string(), value);
            }
            // Try Float64Array for AVG
            else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                if float_array.len() != 1 {
                    return Err(Error::Internal(format!(
                        "Expected single value from aggregate, got {}",
                        float_array.len()
                    )));
                }
                let value = if float_array.is_null(0) {
                    AggregateValue::Null
                } else {
                    AggregateValue::Float64(float_array.value(0))
                };
                results.insert(field.name().to_string(), value);
            }
            // Try StringArray for GROUP_CONCAT
            else if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
                if string_array.len() != 1 {
                    return Err(Error::Internal(format!(
                        "Expected single value from aggregate, got {}",
                        string_array.len()
                    )));
                }
                let value = if string_array.is_null(0) {
                    AggregateValue::Null
                } else {
                    AggregateValue::String(string_array.value(0).to_string())
                };
                results.insert(field.name().to_string(), value);
            }
            // Try Decimal128Array for SUM/AVG on Decimal columns
            else if let Some(decimal_array) = array.as_any().downcast_ref::<Decimal128Array>() {
                if decimal_array.len() != 1 {
                    return Err(Error::Internal(format!(
                        "Expected single value from aggregate, got {}",
                        decimal_array.len()
                    )));
                }
                let value = if decimal_array.is_null(0) {
                    AggregateValue::Null
                } else {
                    AggregateValue::Decimal128 {
                        value: decimal_array.value(0),
                        scale: decimal_array.scale(),
                    }
                };
                results.insert(field.name().to_string(), value);
            } else {
                return Err(Error::Internal(format!(
                    "Unexpected array type from aggregate: {:?}",
                    array.data_type()
                )));
            }
        }

        Ok(results)
    }

    /// Attempt to optimize a multi-table query using hash joins instead of cartesian product.
    ///
    /// This replaces the O(n₁×n₂×...×nₖ) backtracking algorithm with O(n₁+n₂+...+nₖ)
    /// hash join execution. For two-table joins, applies a single hash join. For N-way joins,
    /// performs left-associative pairwise joins: ((T₁ ⋈ T₂) ⋈ T₃) ⋈ ... ⋈ Tₙ.
    ///
    /// # Arguments
    ///
    /// * `plan` - The SELECT plan containing table references and filter predicates
    /// * `tables_with_handles` - Vector of (TableRef, ExecutorTable) pairs for all tables in the query
    ///
    /// # Returns
    ///
    /// * `Ok(Some((data, handled_all)))` - Join optimization succeeded, returning joined batches and whether all predicates were handled
    /// * `Ok(None)` - Optimization cannot be applied (falls back to cartesian product)
    /// * `Err(...)` - Join execution failed
    fn try_execute_hash_join(
        &self,
        plan: &SelectPlan,
        tables_with_handles: &[(llkv_plan::TableRef, Arc<ExecutorTable<P>>)],
    ) -> ExecutorResult<Option<(TableCrossProductData, bool)>> {
        let query_label_opt = current_query_label();
        let query_label = query_label_opt.as_deref().unwrap_or("<unknown query>");

        // Validate preconditions for hash join optimization
        let filter_wrapper = match &plan.filter {
            Some(filter) => filter,
            None => {
                tracing::debug!(
                    "join_opt[{query_label}]: skipping optimization – no filter present"
                );
                return Ok(None);
            }
        };

        // Note: We now allow subqueries in filters. Join predicates will be extracted
        // and used for hash join, while subquery predicates will be handled post-join.

        // Check if we can optimize despite explicit joins.
        // We can optimize if all joins are INNER joins, as they are commutative/associative.
        // For comma-joins (implicit joins), the parser/planner creates Inner joins with no ON condition,
        // and all filters are in the WHERE clause (plan.filter).
        let all_inner_joins = plan
            .joins
            .iter()
            .all(|j| j.join_type == llkv_plan::JoinPlan::Inner);

        if !plan.joins.is_empty() && !all_inner_joins {
            tracing::debug!(
                "join_opt[{query_label}]: skipping optimization – explicit non-INNER JOINs present"
            );
            return Ok(None);
        }

        if tables_with_handles.len() < 2 {
            tracing::debug!(
                "join_opt[{query_label}]: skipping optimization – requires at least 2 tables"
            );
            return Ok(None);
        }

        // Build table metadata for join constraint extraction
        let mut table_infos = Vec::with_capacity(tables_with_handles.len());
        for (index, (table_ref, executor_table)) in tables_with_handles.iter().enumerate() {
            let mut column_map = FxHashMap::default();
            for (column_idx, column) in executor_table.schema.columns.iter().enumerate() {
                let column_name = column.name.to_ascii_lowercase();
                column_map.entry(column_name).or_insert(column_idx);
            }
            table_infos.push(TableInfo {
                index,
                table_ref,
                column_map,
            });
        }

        // Extract join constraints from WHERE clause
        let constraint_plan = match extract_join_constraints(
            &filter_wrapper.predicate,
            &table_infos,
        ) {
            Some(plan) => plan,
            None => {
                tracing::debug!(
                    "join_opt[{query_label}]: skipping optimization – predicate parsing failed (contains OR or other unsupported top-level structure)"
                );
                return Ok(None);
            }
        };

        tracing::debug!(
            "join_opt[{query_label}]: constraint extraction succeeded - equalities={}, literals={}, handled={}/{} predicates",
            constraint_plan.equalities.len(),
            constraint_plan.literals.len(),
            constraint_plan.handled_conjuncts,
            constraint_plan.total_conjuncts
        );
        tracing::debug!(
            "join_opt[{query_label}]: attempting hash join with tables={:?} filter={:?}",
            plan.tables
                .iter()
                .map(|t| t.qualified_name())
                .collect::<Vec<_>>(),
            filter_wrapper.predicate,
        );

        // Handle unsatisfiable predicates (e.g., WHERE FALSE)
        if constraint_plan.unsatisfiable {
            tracing::debug!(
                "join_opt[{query_label}]: predicate unsatisfiable – returning empty result"
            );
            let mut combined_fields = Vec::new();
            let mut column_counts = Vec::new();
            for (_table_ref, executor_table) in tables_with_handles {
                for column in &executor_table.schema.columns {
                    combined_fields.push(Field::new(
                        column.name.clone(),
                        column.data_type.clone(),
                        column.nullable,
                    ));
                }
                column_counts.push(executor_table.schema.columns.len());
            }
            let combined_schema = Arc::new(Schema::new(combined_fields));
            let empty_batch = RecordBatch::new_empty(Arc::clone(&combined_schema));
            return Ok(Some((
                TableCrossProductData {
                    schema: combined_schema,
                    batches: vec![empty_batch],
                    column_counts,
                    table_indices: (0..tables_with_handles.len()).collect(),
                },
                true, // Handled all predicates (unsatisfiable predicate consumes everything)
            )));
        }

        // Hash join requires equality predicates
        if constraint_plan.equalities.is_empty() {
            tracing::debug!(
                "join_opt[{query_label}]: skipping optimization – no join equalities found"
            );
            return Ok(None);
        }

        // Note: Literal constraints (e.g., t1.x = 5) are currently ignored in the hash join path.
        // They should ideally be pushed down as pre-filters on individual tables before joining.
        // For now, we'll let the hash join proceed and any literal constraints will be handled
        // by the fallback cartesian product path if needed.
        if !constraint_plan.literals.is_empty() {
            tracing::debug!(
                "join_opt[{query_label}]: found {} literal constraints - proceeding with hash join but may need fallback",
                constraint_plan.literals.len()
            );
        }

        tracing::debug!(
            "join_opt[{query_label}]: hash join optimization applicable with {} equality constraints",
            constraint_plan.equalities.len()
        );

        let mut literal_map: Vec<Vec<ColumnConstraint>> =
            vec![Vec::new(); tables_with_handles.len()];
        for constraint in &constraint_plan.literals {
            let table_idx = match constraint {
                ColumnConstraint::Equality(lit) => lit.column.table,
                ColumnConstraint::InList(in_list) => in_list.column.table,
            };
            if table_idx >= literal_map.len() {
                tracing::debug!(
                    "join_opt[{query_label}]: constraint references unknown table index {}; falling back",
                    table_idx
                );
                return Ok(None);
            }
            tracing::debug!(
                "join_opt[{query_label}]: mapping constraint to table_idx={} (table={})",
                table_idx,
                tables_with_handles[table_idx].0.qualified_name()
            );
            literal_map[table_idx].push(constraint.clone());
        }

        let mut per_table: Vec<Option<TableCrossProductData>> =
            Vec::with_capacity(tables_with_handles.len());
        for (idx, (table_ref, table)) in tables_with_handles.iter().enumerate() {
            let data =
                collect_table_data(idx, table_ref, table.as_ref(), literal_map[idx].as_slice())?;
            per_table.push(Some(data));
        }

        // Determine if we should use llkv-join (when LEFT JOINs are present or for better architecture)
        let has_left_join = plan
            .joins
            .iter()
            .any(|j| j.join_type == llkv_plan::JoinPlan::Left);

        let mut current: Option<TableCrossProductData> = None;

        if has_left_join {
            // LEFT JOIN path: delegate to llkv-join crate which has proper implementation
            tracing::debug!(
                "join_opt[{query_label}]: delegating to llkv-join for LEFT JOIN support"
            );
            // Bail out of hash join optimization - let the fallback path use llkv-join properly
            return Ok(None);
        } else {
            // INNER JOIN path: use existing optimization that can reorder joins
            let mut remaining: Vec<usize> = (0..tables_with_handles.len()).collect();
            let mut used_tables: FxHashSet<usize> = FxHashSet::default();

            while !remaining.is_empty() {
                let next_index = if used_tables.is_empty() {
                    remaining[0]
                } else {
                    match remaining.iter().copied().find(|idx| {
                        table_has_join_with_used(*idx, &used_tables, &constraint_plan.equalities)
                    }) {
                        Some(idx) => idx,
                        None => {
                            tracing::debug!(
                                "join_opt[{query_label}]: no remaining equality links – using cartesian expansion for table index {idx}",
                                idx = remaining[0]
                            );
                            remaining[0]
                        }
                    }
                };

                let position = remaining
                    .iter()
                    .position(|&idx| idx == next_index)
                    .expect("next index present");

                let next_data = per_table[next_index]
                    .take()
                    .ok_or_else(|| Error::Internal("hash join consumed table data twice".into()))?;

                if let Some(current_data) = current.take() {
                    let join_keys = gather_join_keys(
                        &current_data,
                        &next_data,
                        &used_tables,
                        next_index,
                        &constraint_plan.equalities,
                    )?;

                    let joined = if join_keys.is_empty() {
                        tracing::debug!(
                            "join_opt[{query_label}]: joining '{}' via cartesian expansion (no equality keys)",
                            tables_with_handles[next_index].0.qualified_name()
                        );
                        cross_join_table_batches(current_data, next_data)?
                    } else {
                        hash_join_table_batches(
                            current_data,
                            next_data,
                            &join_keys,
                            llkv_join::JoinType::Inner,
                        )?
                    };
                    current = Some(joined);
                } else {
                    current = Some(next_data);
                }

                used_tables.insert(next_index);
                remaining.remove(position);
            }
        }

        if let Some(result) = current {
            let handled_all = constraint_plan.handled_conjuncts == constraint_plan.total_conjuncts;
            tracing::debug!(
                "join_opt[{query_label}]: hash join succeeded across {} tables (handled {}/{} predicates)",
                tables_with_handles.len(),
                constraint_plan.handled_conjuncts,
                constraint_plan.total_conjuncts
            );
            return Ok(Some((result, handled_all)));
        }

        Ok(None)
    }

    fn execute_projection(
        &self,
        table: Arc<ExecutorTable<P>>,
        display_name: String,
        plan: SelectPlan,
        row_filter: Option<std::sync::Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        if plan.having.is_some() {
            return Err(Error::InvalidArgumentError(
                "HAVING requires GROUP BY".into(),
            ));
        }

        let has_filter_subqueries = plan
            .filter
            .as_ref()
            .is_some_and(|filter| !filter.subqueries.is_empty());
        let has_scalar_subqueries = !plan.scalar_subqueries.is_empty();

        if has_filter_subqueries || has_scalar_subqueries {
            return self.execute_projection_with_subqueries(table, display_name, plan, row_filter);
        }

        let table_ref = table.as_ref();
        let constant_filter = plan
            .filter
            .as_ref()
            .and_then(|filter| evaluate_constant_predicate(&filter.predicate));
        let projections = if plan.projections.is_empty() {
            build_wildcard_projections(table_ref)
        } else {
            build_projected_columns(table_ref, &plan.projections)?
        };
        let schema = schema_for_projections(table_ref, &projections)?;

        if let Some(result) = constant_filter {
            match result {
                Some(true) => {
                    // Treat as full table scan by clearing the filter below.
                }
                Some(false) | None => {
                    let batch = RecordBatch::new_empty(Arc::clone(&schema));
                    return Ok(SelectExecution::new_single_batch(
                        display_name,
                        schema,
                        batch,
                    ));
                }
            }
        }

        let (mut filter_expr, mut full_table_scan) = match &plan.filter {
            Some(filter_wrapper) => (
                crate::translation::expression::translate_predicate(
                    filter_wrapper.predicate.clone(),
                    table_ref.schema.as_ref(),
                    |name| Error::InvalidArgumentError(format!("unknown column '{}'", name)),
                )?,
                false,
            ),
            None => {
                let field_id = table_ref.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform wildcard scan".into(),
                    )
                })?;
                (
                    crate::translation::expression::full_table_scan_filter(field_id),
                    true,
                )
            }
        };

        if matches!(constant_filter, Some(Some(true))) {
            let field_id = table_ref.schema.first_field_id().ok_or_else(|| {
                Error::InvalidArgumentError(
                    "table has no columns; cannot perform wildcard scan".into(),
                )
            })?;
            filter_expr = crate::translation::expression::full_table_scan_filter(field_id);
            full_table_scan = true;
        }

        let expanded_order = expand_order_targets(&plan.order_by, &projections)?;

        let mut physical_order: Option<ScanOrderSpec> = None;

        if let Some(first) = expanded_order.first() {
            match &first.target {
                OrderTarget::Column(name) => {
                    if table_ref.schema.resolve(name).is_some() {
                        physical_order = Some(resolve_scan_order(table_ref, &projections, first)?);
                    }
                }
                OrderTarget::Index(position) => match projections.get(*position) {
                    Some(ScanProjection::Column(_)) => {
                        physical_order = Some(resolve_scan_order(table_ref, &projections, first)?);
                    }
                    Some(ScanProjection::Computed { .. }) => {}
                    None => {
                        return Err(Error::InvalidArgumentError(format!(
                            "ORDER BY position {} is out of range",
                            position + 1
                        )));
                    }
                },
                OrderTarget::All => {}
            }
        }

        let options = if let Some(order_spec) = physical_order {
            if row_filter.is_some() {
                tracing::debug!("Applying MVCC row filter with ORDER BY");
            }
            ScanStreamOptions {
                include_nulls: true,
                order: Some(order_spec),
                row_id_filter: row_filter.clone(),
            }
        } else {
            if row_filter.is_some() {
                tracing::debug!("Applying MVCC row filter");
            }
            ScanStreamOptions {
                include_nulls: true,
                order: None,
                row_id_filter: row_filter.clone(),
            }
        };

        Ok(SelectExecution::new_projection(
            display_name,
            schema,
            table,
            projections,
            filter_expr,
            options,
            full_table_scan,
            expanded_order,
            plan.distinct,
        ))
    }

    fn execute_projection_with_subqueries(
        &self,
        table: Arc<ExecutorTable<P>>,
        display_name: String,
        plan: SelectPlan,
        row_filter: Option<std::sync::Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        if plan.having.is_some() {
            return Err(Error::InvalidArgumentError(
                "HAVING requires GROUP BY".into(),
            ));
        }
        let table_ref = table.as_ref();

        let (output_scan_projections, effective_projections): (
            Vec<ScanProjection>,
            Vec<SelectProjection>,
        ) = if plan.projections.is_empty() {
            (
                build_wildcard_projections(table_ref),
                vec![SelectProjection::AllColumns],
            )
        } else {
            (
                build_projected_columns(table_ref, &plan.projections)?,
                plan.projections.clone(),
            )
        };

        let scalar_lookup: FxHashMap<SubqueryId, &llkv_plan::ScalarSubquery> = plan
            .scalar_subqueries
            .iter()
            .map(|subquery| (subquery.id, subquery))
            .collect();

        let base_projections = build_wildcard_projections(table_ref);

        let filter_wrapper_opt = plan.filter.as_ref();

        // Check if the filter contains scalar subqueries that need to be handled
        let mut filter_has_scalar_subqueries = false;
        if let Some(filter_wrapper) = filter_wrapper_opt {
            let translated = crate::translation::expression::translate_predicate(
                filter_wrapper.predicate.clone(),
                table_ref.schema.as_ref(),
                |name| Error::InvalidArgumentError(format!("unknown column '{}'", name)),
            )?;
            let mut scalar_filter_ids = FxHashSet::default();
            collect_predicate_scalar_subquery_ids(&translated, &mut scalar_filter_ids);
            filter_has_scalar_subqueries = !scalar_filter_ids.is_empty();
        }

        let mut translated_filter: Option<llkv_expr::expr::Expr<'static, FieldId>> = None;
        let pushdown_filter = if let Some(filter_wrapper) = filter_wrapper_opt {
            let translated = crate::translation::expression::translate_predicate(
                filter_wrapper.predicate.clone(),
                table_ref.schema.as_ref(),
                |name| Error::InvalidArgumentError(format!("unknown column '{}'", name)),
            )?;
            if !filter_wrapper.subqueries.is_empty() || filter_has_scalar_subqueries {
                translated_filter = Some(translated.clone());
                if filter_has_scalar_subqueries {
                    // If filter has scalar subqueries, use full table scan for pushdown
                    // and evaluate scalar subqueries in the callback
                    let field_id = table_ref.schema.first_field_id().ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "table has no columns; cannot perform scalar subquery projection"
                                .into(),
                        )
                    })?;
                    crate::translation::expression::full_table_scan_filter(field_id)
                } else {
                    // Only EXISTS subqueries, strip them for pushdown
                    strip_exists(&translated)
                }
            } else {
                translated
            }
        } else {
            let field_id = table_ref.schema.first_field_id().ok_or_else(|| {
                Error::InvalidArgumentError(
                    "table has no columns; cannot perform scalar subquery projection".into(),
                )
            })?;
            crate::translation::expression::full_table_scan_filter(field_id)
        };

        let mut base_fields: Vec<Field> = Vec::with_capacity(table_ref.schema.columns.len());
        for column in &table_ref.schema.columns {
            base_fields.push(Field::new(
                column.name.clone(),
                column.data_type.clone(),
                column.nullable,
            ));
        }
        let base_schema = Arc::new(Schema::new(base_fields));
        let base_column_counts = vec![base_schema.fields().len()];
        let base_table_indices = vec![0usize];
        let base_lookup = build_cross_product_column_lookup(
            base_schema.as_ref(),
            &plan.tables,
            &base_column_counts,
            &base_table_indices,
        );

        let mut filter_context = if translated_filter.is_some() {
            Some(CrossProductExpressionContext::new(
                base_schema.as_ref(),
                base_lookup.clone(),
            )?)
        } else {
            None
        };

        // Collect scalar subqueries that appear in the filter
        let mut filter_scalar_subquery_ids = FxHashSet::default();
        if let Some(translated) = translated_filter.as_ref() {
            collect_predicate_scalar_subquery_ids(translated, &mut filter_scalar_subquery_ids);
        }

        // Build lookup for filter scalar subqueries
        let filter_scalar_lookup: FxHashMap<SubqueryId, &llkv_plan::ScalarSubquery> =
            if !filter_scalar_subquery_ids.is_empty() {
                plan.scalar_subqueries
                    .iter()
                    .filter(|subquery| filter_scalar_subquery_ids.contains(&subquery.id))
                    .map(|subquery| (subquery.id, subquery))
                    .collect()
            } else {
                FxHashMap::default()
            };

        let options = ScanStreamOptions {
            include_nulls: true,
            order: None,
            row_id_filter: row_filter.clone(),
        };

        let subquery_lookup: FxHashMap<llkv_expr::SubqueryId, &llkv_plan::FilterSubquery> =
            filter_wrapper_opt
                .map(|wrapper| {
                    wrapper
                        .subqueries
                        .iter()
                        .map(|subquery| (subquery.id, subquery))
                        .collect()
                })
                .unwrap_or_default();

        let mut projected_batches: Vec<RecordBatch> = Vec::new();
        let mut scan_error: Option<Error> = None;

        table.table.scan_stream(
            base_projections.clone(),
            &pushdown_filter,
            options,
            |batch| {
                if scan_error.is_some() {
                    return;
                }
                let effective_batch = if let Some(context) = filter_context.as_mut() {
                    context.reset();

                    // Evaluate scalar subqueries in the filter for this batch
                    for (subquery_id, subquery) in filter_scalar_lookup.iter() {
                        let result_array = match self
                            .evaluate_scalar_subquery_numeric(context, subquery, &batch)
                        {
                            Ok(array) => array,
                            Err(err) => {
                                scan_error = Some(err);
                                return;
                            }
                        };
                        let accessor = match ColumnAccessor::from_numeric_array(&result_array) {
                            Ok(acc) => acc,
                            Err(err) => {
                                scan_error = Some(err);
                                return;
                            }
                        };
                        context
                            .scalar_subquery_columns
                            .insert(*subquery_id, accessor);
                    }
                    let translated = translated_filter
                        .as_ref()
                        .expect("filter context requires translated filter");
                    let mask = match context.evaluate_predicate_mask(
                        translated,
                        &batch,
                        |ctx, subquery_expr, row_idx, current_batch| {
                            let subquery =
                                subquery_lookup.get(&subquery_expr.id).ok_or_else(|| {
                                    Error::Internal("missing correlated subquery metadata".into())
                                })?;
                            let exists = self.evaluate_exists_subquery(
                                ctx,
                                subquery,
                                current_batch,
                                row_idx,
                            )?;
                            let value = if subquery_expr.negated {
                                !exists
                            } else {
                                exists
                            };
                            Ok(Some(value))
                        },
                    ) {
                        Ok(mask) => mask,
                        Err(err) => {
                            scan_error = Some(err);
                            return;
                        }
                    };
                    match filter_record_batch(&batch, &mask) {
                        Ok(filtered) => {
                            if filtered.num_rows() == 0 {
                                return;
                            }
                            filtered
                        }
                        Err(err) => {
                            scan_error = Some(Error::InvalidArgumentError(format!(
                                "failed to apply EXISTS filter: {err}"
                            )));
                            return;
                        }
                    }
                } else {
                    batch.clone()
                };

                if effective_batch.num_rows() == 0 {
                    return;
                }

                let projected = match self.project_record_batch(
                    &effective_batch,
                    &effective_projections,
                    &base_lookup,
                    &scalar_lookup,
                ) {
                    Ok(batch) => batch,
                    Err(err) => {
                        scan_error = Some(Error::InvalidArgumentError(format!(
                            "failed to evaluate projections: {err}"
                        )));
                        return;
                    }
                };
                projected_batches.push(projected);
            },
        )?;

        if let Some(err) = scan_error {
            return Err(err);
        }

        let mut result_batch = if projected_batches.is_empty() {
            let empty_batch = RecordBatch::new_empty(Arc::clone(&base_schema));
            self.project_record_batch(
                &empty_batch,
                &effective_projections,
                &base_lookup,
                &scalar_lookup,
            )?
        } else if projected_batches.len() == 1 {
            projected_batches.pop().unwrap()
        } else {
            let schema = projected_batches[0].schema();
            concat_batches(&schema, &projected_batches).map_err(|err| {
                Error::Internal(format!("failed to combine filtered batches: {err}"))
            })?
        };

        if plan.distinct && result_batch.num_rows() > 0 {
            let mut state = DistinctState::default();
            let schema = result_batch.schema();
            result_batch = match distinct_filter_batch(result_batch, &mut state)? {
                Some(filtered) => filtered,
                None => RecordBatch::new_empty(schema),
            };
        }

        if !plan.order_by.is_empty() && result_batch.num_rows() > 0 {
            let expanded_order = expand_order_targets(&plan.order_by, &output_scan_projections)?;
            if !expanded_order.is_empty() {
                result_batch = sort_record_batch_with_order(
                    &result_batch.schema(),
                    &result_batch,
                    &expanded_order,
                )?;
            }
        }

        let schema = result_batch.schema();

        Ok(SelectExecution::new_single_batch(
            display_name,
            schema,
            result_batch,
        ))
    }

    fn execute_group_by_single_table(
        &self,
        table: Arc<ExecutorTable<P>>,
        display_name: String,
        plan: SelectPlan,
        row_filter: Option<std::sync::Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        if plan
            .filter
            .as_ref()
            .is_some_and(|filter| !filter.subqueries.is_empty())
            || !plan.scalar_subqueries.is_empty()
        {
            return Err(Error::InvalidArgumentError(
                "GROUP BY with subqueries is not supported yet".into(),
            ));
        }

        // Debug: check if we have aggregates in unexpected places
        tracing::debug!(
            "[GROUP BY] Original plan: projections={}, aggregates={}, has_filter={}, has_having={}",
            plan.projections.len(),
            plan.aggregates.len(),
            plan.filter.is_some(),
            plan.having.is_some()
        );

        // For GROUP BY with aggregates, we need a two-phase execution:
        // 1. First phase: fetch all base table data (no projections, no aggregates)
        // 2. Second phase: group rows and compute aggregates
        let mut base_plan = plan.clone();
        base_plan.projections.clear();
        base_plan.aggregates.clear();
        base_plan.scalar_subqueries.clear();
        base_plan.order_by.clear();
        base_plan.distinct = false;
        base_plan.group_by.clear();
        base_plan.value_table_mode = None;
        base_plan.having = None;

        tracing::debug!(
            "[GROUP BY] Base plan: projections={}, aggregates={}, has_filter={}, has_having={}",
            base_plan.projections.len(),
            base_plan.aggregates.len(),
            base_plan.filter.is_some(),
            base_plan.having.is_some()
        );

        // For base scan, we want all columns from the table
        // We build wildcard projections directly to avoid any expression evaluation
        let table_ref = table.as_ref();
        let projections = build_wildcard_projections(table_ref);
        let base_schema = schema_for_projections(table_ref, &projections)?;

        // Build filter if present (should NOT contain aggregates)
        tracing::debug!(
            "[GROUP BY] Building base filter: has_filter={}",
            base_plan.filter.is_some()
        );
        let (filter_expr, full_table_scan) = match &base_plan.filter {
            Some(filter_wrapper) => {
                tracing::debug!(
                    "[GROUP BY] Translating filter predicate: {:?}",
                    filter_wrapper.predicate
                );
                let expr = crate::translation::expression::translate_predicate(
                    filter_wrapper.predicate.clone(),
                    table_ref.schema.as_ref(),
                    |name| {
                        Error::InvalidArgumentError(format!(
                            "Binder Error: does not have a column named '{}'",
                            name
                        ))
                    },
                )?;
                tracing::debug!("[GROUP BY] Translated filter expr: {:?}", expr);
                (expr, false)
            }
            None => {
                // Use first column as dummy for full table scan
                let first_col =
                    table_ref.schema.columns.first().ok_or_else(|| {
                        Error::InvalidArgumentError("Table has no columns".into())
                    })?;
                (full_table_scan_filter(first_col.field_id), true)
            }
        };

        let options = ScanStreamOptions {
            include_nulls: true,
            order: None,
            row_id_filter: row_filter.clone(),
        };

        let execution = SelectExecution::new_projection(
            display_name.clone(),
            Arc::clone(&base_schema),
            Arc::clone(&table),
            projections,
            filter_expr,
            options,
            full_table_scan,
            vec![],
            false,
        );

        let batches = execution.collect()?;

        tracing::debug!(
            "[GROUP BY] Collected {} batches from base scan, total_rows={}",
            batches.len(),
            batches.iter().map(|b| b.num_rows()).sum::<usize>()
        );

        let column_lookup_map = build_column_lookup_map(base_schema.as_ref());

        self.execute_group_by_from_batches(
            display_name,
            plan,
            base_schema,
            batches,
            column_lookup_map,
        )
    }

    fn execute_group_by_from_batches(
        &self,
        display_name: String,
        plan: SelectPlan,
        base_schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
        column_lookup_map: FxHashMap<String, usize>,
    ) -> ExecutorResult<SelectExecution<P>> {
        if plan
            .filter
            .as_ref()
            .is_some_and(|filter| !filter.subqueries.is_empty())
            || !plan.scalar_subqueries.is_empty()
        {
            return Err(Error::InvalidArgumentError(
                "GROUP BY with subqueries is not supported yet".into(),
            ));
        }

        // If there are aggregates with GROUP BY, OR if HAVING contains aggregates, use aggregates path
        // Must check HAVING because aggregates can appear in HAVING even if not in SELECT projections
        let having_has_aggregates = plan
            .having
            .as_ref()
            .map(|h| Self::predicate_contains_aggregate(h))
            .unwrap_or(false);

        tracing::debug!(
            "[GROUP BY PATH] aggregates={}, has_computed={}, having_has_agg={}",
            plan.aggregates.len(),
            self.has_computed_aggregates(&plan),
            having_has_aggregates
        );

        if !plan.aggregates.is_empty()
            || self.has_computed_aggregates(&plan)
            || having_has_aggregates
        {
            tracing::debug!("[GROUP BY PATH] Taking aggregates path");
            return self.execute_group_by_with_aggregates(
                display_name,
                plan,
                base_schema,
                batches,
                column_lookup_map,
            );
        }

        let mut key_indices = Vec::with_capacity(plan.group_by.len());
        for column in &plan.group_by {
            let key = column.to_ascii_lowercase();
            let index = column_lookup_map.get(&key).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "column '{}' not found in GROUP BY input",
                    column
                ))
            })?;
            key_indices.push(*index);
        }

        let sample_batch = batches
            .first()
            .cloned()
            .unwrap_or_else(|| RecordBatch::new_empty(Arc::clone(&base_schema)));

        let output_columns = self.build_group_by_output_columns(
            &plan,
            base_schema.as_ref(),
            &column_lookup_map,
            &sample_batch,
        )?;

        let constant_having = plan.having.as_ref().and_then(evaluate_constant_predicate);

        if let Some(result) = constant_having
            && !result.unwrap_or(false)
        {
            let fields: Vec<Field> = output_columns
                .iter()
                .map(|output| output.field.clone())
                .collect();
            let schema = Arc::new(Schema::new(fields));
            let batch = RecordBatch::new_empty(Arc::clone(&schema));
            return Ok(SelectExecution::new_single_batch(
                display_name,
                schema,
                batch,
            ));
        }

        let translated_having = if plan.having.is_some() && constant_having.is_none() {
            let having = plan.having.clone().expect("checked above");
            // Only translate HAVING if it doesn't contain aggregates
            // Aggregates must be evaluated after GROUP BY aggregation
            if Self::predicate_contains_aggregate(&having) {
                None
            } else {
                let temp_context = CrossProductExpressionContext::new(
                    base_schema.as_ref(),
                    column_lookup_map.clone(),
                )?;
                Some(translate_predicate(
                    having,
                    temp_context.schema(),
                    |name| {
                        Error::InvalidArgumentError(format!(
                            "column '{}' not found in GROUP BY result",
                            name
                        ))
                    },
                )?)
            }
        } else {
            None
        };

        let mut group_index: FxHashMap<Vec<GroupKeyValue>, usize> = FxHashMap::default();
        let mut groups: Vec<GroupState> = Vec::new();

        for batch in &batches {
            for row_idx in 0..batch.num_rows() {
                let key = build_group_key(batch, row_idx, &key_indices)?;
                if group_index.contains_key(&key) {
                    continue;
                }
                group_index.insert(key, groups.len());
                groups.push(GroupState {
                    batch: batch.clone(),
                    row_idx,
                });
            }
        }

        let mut rows: Vec<Vec<PlanValue>> = Vec::with_capacity(groups.len());

        for group in &groups {
            if let Some(predicate) = translated_having.as_ref() {
                let mut context = CrossProductExpressionContext::new(
                    group.batch.schema().as_ref(),
                    column_lookup_map.clone(),
                )?;
                context.reset();
                let mut eval = |_ctx: &mut CrossProductExpressionContext,
                                _subquery_expr: &llkv_expr::SubqueryExpr,
                                _row_idx: usize,
                                _current_batch: &RecordBatch|
                 -> ExecutorResult<Option<bool>> {
                    Err(Error::InvalidArgumentError(
                        "HAVING subqueries are not supported yet".into(),
                    ))
                };
                let truths =
                    context.evaluate_predicate_truths(predicate, &group.batch, &mut eval)?;
                let passes = truths
                    .get(group.row_idx)
                    .copied()
                    .flatten()
                    .unwrap_or(false);
                if !passes {
                    continue;
                }
            }

            let mut row: Vec<PlanValue> = Vec::with_capacity(output_columns.len());
            for output in &output_columns {
                match output.source {
                    OutputSource::TableColumn { index } => {
                        let value = llkv_plan::plan_value_from_array(
                            group.batch.column(index),
                            group.row_idx,
                        )?;
                        row.push(value);
                    }
                    OutputSource::Computed { projection_index } => {
                        let expr = match &plan.projections[projection_index] {
                            SelectProjection::Computed { expr, .. } => expr,
                            _ => unreachable!("projection index mismatch for computed column"),
                        };
                        let mut context = CrossProductExpressionContext::new(
                            group.batch.schema().as_ref(),
                            column_lookup_map.clone(),
                        )?;
                        context.reset();
                        let evaluated = self.evaluate_projection_expression(
                            &mut context,
                            expr,
                            &group.batch,
                            &FxHashMap::default(),
                        )?;
                        let value = llkv_plan::plan_value_from_array(&evaluated, group.row_idx)?;
                        row.push(value);
                    }
                }
            }
            rows.push(row);
        }

        let fields: Vec<Field> = output_columns
            .into_iter()
            .map(|output| output.field)
            .collect();
        let schema = Arc::new(Schema::new(fields));

        let mut batch = rows_to_record_batch(Arc::clone(&schema), &rows)?;

        if plan.distinct && batch.num_rows() > 0 {
            let mut state = DistinctState::default();
            batch = match distinct_filter_batch(batch, &mut state)? {
                Some(filtered) => filtered,
                None => RecordBatch::new_empty(Arc::clone(&schema)),
            };
        }

        if !plan.order_by.is_empty() && batch.num_rows() > 0 {
            batch = sort_record_batch_with_order(&schema, &batch, &plan.order_by)?;
        }

        Ok(SelectExecution::new_single_batch(
            display_name,
            schema,
            batch,
        ))
    }

    /// Infer the Arrow data type for a computed expression in GROUP BY context
    fn infer_computed_expression_type(
        expr: &ScalarExpr<String>,
        base_schema: &Schema,
        column_lookup_map: &FxHashMap<String, usize>,
        sample_batch: &RecordBatch,
    ) -> Option<DataType> {
        use llkv_expr::expr::AggregateCall;

        // If it's a simple aggregate, return its result type
        if let ScalarExpr::Aggregate(agg_call) = expr {
            return match agg_call {
                AggregateCall::CountStar
                | AggregateCall::Count { .. }
                | AggregateCall::CountNulls(_) => Some(DataType::Int64),
                AggregateCall::Sum { expr: agg_expr, .. }
                | AggregateCall::Total { expr: agg_expr, .. }
                | AggregateCall::Avg { expr: agg_expr, .. }
                | AggregateCall::Min(agg_expr)
                | AggregateCall::Max(agg_expr) => {
                    // Try recursive inference first to handle complex expressions without sample data
                    if let Some(dtype) =
                        infer_type_recursive(agg_expr, base_schema, column_lookup_map)
                    {
                        return Some(dtype);
                    }

                    // For aggregate functions, infer the type from the expression
                    if let Some(col_name) = try_extract_simple_column(agg_expr) {
                        let idx = resolve_column_name_to_index(col_name, column_lookup_map)?;
                        Some(base_schema.field(idx).data_type().clone())
                    } else {
                        // Complex expression - evaluate to determine scale, but use conservative precision
                        // For SUM/TOTAL/AVG of decimal expressions, the result can grow beyond the sample value
                        if sample_batch.num_rows() > 0 {
                            let mut computed_values = Vec::new();
                            if let Ok(value) =
                                Self::evaluate_expr_with_plan_value_aggregates_and_row(
                                    agg_expr,
                                    &FxHashMap::default(),
                                    Some(sample_batch),
                                    Some(column_lookup_map),
                                    0,
                                )
                            {
                                computed_values.push(value);
                                if let Ok(array) = plan_values_to_arrow_array(&computed_values) {
                                    match array.data_type() {
                                        // If it's Decimal128, use maximum precision to avoid overflow in aggregates
                                        DataType::Decimal128(_, scale) => {
                                            return Some(DataType::Decimal128(38, *scale));
                                        }
                                        // If evaluation returned NULL, fall back to Float64
                                        DataType::Null => {
                                            return Some(DataType::Float64);
                                        }
                                        other => {
                                            return Some(other.clone());
                                        }
                                    }
                                }
                            }
                        }
                        // Unable to determine type, fall back to Float64
                        Some(DataType::Float64)
                    }
                }
                AggregateCall::GroupConcat { .. } => Some(DataType::Utf8),
            };
        }

        // For other expressions, could try evaluating with sample data
        // but for now return None to fall back to Float64
        None
    }

    fn build_group_by_output_columns(
        &self,
        plan: &SelectPlan,
        base_schema: &Schema,
        column_lookup_map: &FxHashMap<String, usize>,
        _sample_batch: &RecordBatch,
    ) -> ExecutorResult<Vec<OutputColumn>> {
        let projections = if plan.projections.is_empty() {
            vec![SelectProjection::AllColumns]
        } else {
            plan.projections.clone()
        };

        let mut columns: Vec<OutputColumn> = Vec::new();

        for (proj_idx, projection) in projections.iter().enumerate() {
            match projection {
                SelectProjection::AllColumns => {
                    for (index, field) in base_schema.fields().iter().enumerate() {
                        columns.push(OutputColumn {
                            field: (**field).clone(),
                            source: OutputSource::TableColumn { index },
                        });
                    }
                }
                SelectProjection::AllColumnsExcept { exclude } => {
                    let exclude_lower: FxHashSet<String> = exclude
                        .iter()
                        .map(|name| name.to_ascii_lowercase())
                        .collect();

                    let mut excluded_indices = FxHashSet::default();
                    for excluded_name in &exclude_lower {
                        if let Some(&idx) = column_lookup_map.get(excluded_name) {
                            excluded_indices.insert(idx);
                        }
                    }

                    for (index, field) in base_schema.fields().iter().enumerate() {
                        if !exclude_lower.contains(&field.name().to_ascii_lowercase())
                            && !excluded_indices.contains(&index)
                        {
                            columns.push(OutputColumn {
                                field: (**field).clone(),
                                source: OutputSource::TableColumn { index },
                            });
                        }
                    }
                }
                SelectProjection::Column { name, alias } => {
                    let lookup_key = name.to_ascii_lowercase();
                    let index = column_lookup_map.get(&lookup_key).ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "column '{}' not found in GROUP BY result",
                            name
                        ))
                    })?;
                    let field = base_schema.field(*index);
                    let field = Field::new(
                        alias.as_ref().unwrap_or(name).clone(),
                        field.data_type().clone(),
                        field.is_nullable(),
                    );
                    columns.push(OutputColumn {
                        field,
                        source: OutputSource::TableColumn { index: *index },
                    });
                }
                SelectProjection::Computed { expr, alias } => {
                    // For GROUP BY with aggregates, we need to infer the type from the expression
                    // If it's a simple aggregate, use the aggregate's result type
                    // Otherwise, we'll need to evaluate a sample to determine the type
                    let inferred_type = Self::infer_computed_expression_type(
                        expr,
                        base_schema,
                        column_lookup_map,
                        _sample_batch,
                    )
                    .unwrap_or(DataType::Float64);
                    let field = Field::new(alias.clone(), inferred_type, true);
                    columns.push(OutputColumn {
                        field,
                        source: OutputSource::Computed {
                            projection_index: proj_idx,
                        },
                    });
                }
            }
        }

        if columns.is_empty() {
            for (index, field) in base_schema.fields().iter().enumerate() {
                columns.push(OutputColumn {
                    field: (**field).clone(),
                    source: OutputSource::TableColumn { index },
                });
            }
        }

        Ok(columns)
    }

    fn project_record_batch(
        &self,
        batch: &RecordBatch,
        projections: &[SelectProjection],
        lookup: &FxHashMap<String, usize>,
        scalar_lookup: &FxHashMap<SubqueryId, &llkv_plan::ScalarSubquery>,
    ) -> ExecutorResult<RecordBatch> {
        if projections.is_empty() {
            return Ok(batch.clone());
        }

        let schema = batch.schema();
        let mut selected_fields: Vec<Arc<Field>> = Vec::new();
        let mut selected_columns: Vec<ArrayRef> = Vec::new();
        let mut expr_context: Option<CrossProductExpressionContext> = None;

        for proj in projections {
            match proj {
                SelectProjection::AllColumns => {
                    selected_fields = schema.fields().iter().cloned().collect();
                    selected_columns = batch.columns().to_vec();
                    break;
                }
                SelectProjection::AllColumnsExcept { exclude } => {
                    let exclude_lower: FxHashSet<String> = exclude
                        .iter()
                        .map(|name| name.to_ascii_lowercase())
                        .collect();

                    let mut excluded_indices = FxHashSet::default();
                    for excluded_name in &exclude_lower {
                        if let Some(&idx) = lookup.get(excluded_name) {
                            excluded_indices.insert(idx);
                        }
                    }

                    for (idx, field) in schema.fields().iter().enumerate() {
                        let column_name = field.name().to_ascii_lowercase();
                        if !exclude_lower.contains(&column_name) && !excluded_indices.contains(&idx)
                        {
                            selected_fields.push(Arc::clone(field));
                            selected_columns.push(batch.column(idx).clone());
                        }
                    }
                    break;
                }
                SelectProjection::Column { name, alias } => {
                    let normalized = name.to_ascii_lowercase();
                    let column_index = lookup.get(&normalized).ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "column '{}' not found in projection",
                            name
                        ))
                    })?;
                    let field = schema.field(*column_index);
                    let output_field = Arc::new(Field::new(
                        alias.as_ref().unwrap_or_else(|| field.name()),
                        field.data_type().clone(),
                        field.is_nullable(),
                    ));
                    selected_fields.push(output_field);
                    selected_columns.push(batch.column(*column_index).clone());
                }
                SelectProjection::Computed { expr, alias } => {
                    if expr_context.is_none() {
                        expr_context = Some(CrossProductExpressionContext::new(
                            schema.as_ref(),
                            lookup.clone(),
                        )?);
                    }
                    let context = expr_context
                        .as_mut()
                        .expect("projection context must be initialized");
                    context.reset();
                    let evaluated =
                        self.evaluate_projection_expression(context, expr, batch, scalar_lookup)?;
                    let field = Arc::new(Field::new(
                        alias.clone(),
                        evaluated.data_type().clone(),
                        true,
                    ));
                    selected_fields.push(field);
                    selected_columns.push(evaluated);
                }
            }
        }

        let projected_schema = Arc::new(Schema::new(selected_fields));
        RecordBatch::try_new(projected_schema, selected_columns)
            .map_err(|e| Error::Internal(format!("failed to apply projections: {}", e)))
    }

    /// Execute GROUP BY with aggregates - computes aggregates per group
    fn execute_group_by_with_aggregates(
        &self,
        display_name: String,
        plan: SelectPlan,
        base_schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
        column_lookup_map: FxHashMap<String, usize>,
    ) -> ExecutorResult<SelectExecution<P>> {
        use llkv_expr::expr::AggregateCall;

        // Extract GROUP BY key indices
        let mut key_indices = Vec::with_capacity(plan.group_by.len());
        for column in &plan.group_by {
            let key = column.to_ascii_lowercase();
            let index = column_lookup_map.get(&key).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "column '{}' not found in GROUP BY input",
                    column
                ))
            })?;
            key_indices.push(*index);
        }

        // Extract all aggregates from computed projections
        let mut aggregate_specs: Vec<(String, AggregateCall<String>)> = Vec::new();
        for proj in &plan.projections {
            if let SelectProjection::Computed { expr, .. } = proj {
                Self::collect_aggregates(expr, &mut aggregate_specs);
            }
        }

        // Also extract aggregates from HAVING clause
        if let Some(having_expr) = &plan.having {
            Self::collect_aggregates_from_predicate(having_expr, &mut aggregate_specs);
        }

        // Build a hash map for groups - collect row indices per group
        let mut group_index: FxHashMap<Vec<GroupKeyValue>, usize> = FxHashMap::default();
        let mut group_states: Vec<GroupAggregateState> = Vec::new();

        // First pass: collect all rows for each group
        for (batch_idx, batch) in batches.iter().enumerate() {
            for row_idx in 0..batch.num_rows() {
                let key = build_group_key(batch, row_idx, &key_indices)?;

                if let Some(&group_idx) = group_index.get(&key) {
                    // Add row to existing group
                    group_states[group_idx]
                        .row_locations
                        .push((batch_idx, row_idx));
                } else {
                    // New group
                    let group_idx = group_states.len();
                    group_index.insert(key, group_idx);
                    group_states.push(GroupAggregateState {
                        representative_batch_idx: batch_idx,
                        representative_row: row_idx,
                        row_locations: vec![(batch_idx, row_idx)],
                    });
                }
            }
        }

        // Second pass: compute aggregates for each group using proper AggregateState
        let mut group_aggregate_values: Vec<FxHashMap<String, PlanValue>> =
            Vec::with_capacity(group_states.len());

        for group_state in &group_states {
            tracing::debug!(
                "[GROUP BY] aggregate group rows={:?}",
                group_state.row_locations
            );
            // Create a mini-batch containing only rows from this group
            let group_batch = {
                let representative_batch = &batches[group_state.representative_batch_idx];
                let schema = representative_batch.schema();

                // Collect row indices per batch while preserving scan order
                let mut per_batch_indices: Vec<(usize, Vec<u64>)> = Vec::new();
                for &(batch_idx, row_idx) in &group_state.row_locations {
                    if let Some((_, indices)) = per_batch_indices
                        .iter_mut()
                        .find(|(idx, _)| *idx == batch_idx)
                    {
                        indices.push(row_idx as u64);
                    } else {
                        per_batch_indices.push((batch_idx, vec![row_idx as u64]));
                    }
                }

                let mut row_index_arrays: Vec<(usize, ArrayRef)> =
                    Vec::with_capacity(per_batch_indices.len());
                for (batch_idx, indices) in per_batch_indices {
                    let index_array: ArrayRef = Arc::new(arrow::array::UInt64Array::from(indices));
                    row_index_arrays.push((batch_idx, index_array));
                }

                let mut arrays: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());

                for col_idx in 0..schema.fields().len() {
                    let column_array = if row_index_arrays.len() == 1 {
                        let (batch_idx, indices) = &row_index_arrays[0];
                        let source_array = batches[*batch_idx].column(col_idx);
                        arrow::compute::take(source_array.as_ref(), indices.as_ref(), None)?
                    } else {
                        let mut partial_arrays: Vec<ArrayRef> =
                            Vec::with_capacity(row_index_arrays.len());
                        for (batch_idx, indices) in &row_index_arrays {
                            let source_array = batches[*batch_idx].column(col_idx);
                            let taken = arrow::compute::take(
                                source_array.as_ref(),
                                indices.as_ref(),
                                None,
                            )?;
                            partial_arrays.push(taken);
                        }
                        let slices: Vec<&dyn arrow::array::Array> =
                            partial_arrays.iter().map(|arr| arr.as_ref()).collect();
                        arrow::compute::concat(&slices)?
                    };
                    arrays.push(column_array);
                }

                let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
                tracing::debug!("[GROUP BY] group batch rows={}", batch.num_rows());
                batch
            };

            // Create AggregateState for each aggregate and compute
            let mut aggregate_values: FxHashMap<String, PlanValue> = FxHashMap::default();

            // We might need to add computed columns to the batch for complex aggregate expressions
            let mut working_batch = group_batch.clone();
            let mut next_temp_col_idx = working_batch.num_columns();

            for (key, agg_call) in &aggregate_specs {
                // Determine the column index and Arrow type that backs this aggregate input
                let (projection_idx, value_type) = match agg_call {
                    AggregateCall::CountStar => (None, None),
                    AggregateCall::Count { expr, .. }
                    | AggregateCall::Sum { expr, .. }
                    | AggregateCall::Total { expr, .. }
                    | AggregateCall::Avg { expr, .. }
                    | AggregateCall::Min(expr)
                    | AggregateCall::Max(expr)
                    | AggregateCall::CountNulls(expr)
                    | AggregateCall::GroupConcat { expr, .. } => {
                        if let Some(col_name) = try_extract_simple_column(expr) {
                            let idx = resolve_column_name_to_index(col_name, &column_lookup_map)
                                .ok_or_else(|| {
                                    Error::InvalidArgumentError(format!(
                                        "column '{}' not found for aggregate",
                                        col_name
                                    ))
                                })?;
                            let field_type = working_batch.schema().field(idx).data_type().clone();
                            (Some(idx), Some(field_type))
                        } else {
                            // Complex expression - evaluate it and add as temporary column
                            let mut computed_values = Vec::with_capacity(working_batch.num_rows());
                            for row_idx in 0..working_batch.num_rows() {
                                let value = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                                    expr,
                                    &FxHashMap::default(),
                                    Some(&working_batch),
                                    Some(&column_lookup_map),
                                    row_idx,
                                )?;
                                computed_values.push(value);
                            }

                            let computed_array = plan_values_to_arrow_array(&computed_values)?;
                            let computed_type = computed_array.data_type().clone();

                            let mut new_columns: Vec<ArrayRef> = working_batch.columns().to_vec();
                            new_columns.push(computed_array);

                            let temp_field = Arc::new(Field::new(
                                format!("__temp_agg_expr_{}", next_temp_col_idx),
                                computed_type.clone(),
                                true,
                            ));
                            let mut new_fields: Vec<Arc<Field>> =
                                working_batch.schema().fields().iter().cloned().collect();
                            new_fields.push(temp_field);
                            let new_schema = Arc::new(Schema::new(new_fields));

                            working_batch = RecordBatch::try_new(new_schema, new_columns)?;

                            let col_idx = next_temp_col_idx;
                            next_temp_col_idx += 1;
                            (Some(col_idx), Some(computed_type))
                        }
                    }
                };

                // Build the AggregateSpec - use dummy field_id since projection_idx will override it
                let spec = Self::build_aggregate_spec_for_cross_product(
                    agg_call,
                    key.clone(),
                    value_type.clone(),
                )?;

                let mut state = llkv_aggregate::AggregateState {
                    alias: key.clone(),
                    accumulator: llkv_aggregate::AggregateAccumulator::new_with_projection_index(
                        &spec,
                        projection_idx,
                        None,
                    )?,
                    override_value: None,
                };

                // Update with the working batch (which may have temporary columns)
                state.update(&working_batch)?;

                // Finalize and extract value
                let (_field, array) = state.finalize()?;
                let value = llkv_plan::plan_value_from_array(&array, 0)?;
                tracing::debug!(
                    "[GROUP BY] aggregate result key={:?} value={:?}",
                    key,
                    value
                );
                aggregate_values.insert(key.clone(), value);
            }

            group_aggregate_values.push(aggregate_values);
        }

        // Build result rows
        let output_columns = self.build_group_by_output_columns(
            &plan,
            base_schema.as_ref(),
            &column_lookup_map,
            batches
                .first()
                .unwrap_or(&RecordBatch::new_empty(Arc::clone(&base_schema))),
        )?;

        let mut rows: Vec<Vec<PlanValue>> = Vec::with_capacity(group_states.len());

        for (group_idx, group_state) in group_states.iter().enumerate() {
            let aggregate_values = &group_aggregate_values[group_idx];
            let representative_batch = &batches[group_state.representative_batch_idx];

            let mut row: Vec<PlanValue> = Vec::with_capacity(output_columns.len());
            for output in &output_columns {
                match output.source {
                    OutputSource::TableColumn { index } => {
                        // Use the representative row from this group
                        let value = llkv_plan::plan_value_from_array(
                            representative_batch.column(index),
                            group_state.representative_row,
                        )?;
                        row.push(value);
                    }
                    OutputSource::Computed { projection_index } => {
                        let expr = match &plan.projections[projection_index] {
                            SelectProjection::Computed { expr, .. } => expr,
                            _ => unreachable!("projection index mismatch for computed column"),
                        };
                        // Evaluate expression with aggregates and row context
                        let value = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                            expr,
                            aggregate_values,
                            Some(representative_batch),
                            Some(&column_lookup_map),
                            group_state.representative_row,
                        )?;
                        row.push(value);
                    }
                }
            }
            rows.push(row);
        }

        // Apply HAVING clause if present
        let filtered_rows = if let Some(having) = &plan.having {
            let mut filtered = Vec::new();
            for (row_idx, row) in rows.iter().enumerate() {
                let aggregate_values = &group_aggregate_values[row_idx];
                let group_state = &group_states[row_idx];
                let representative_batch = &batches[group_state.representative_batch_idx];
                // Evaluate HAVING expression recursively
                let passes = Self::evaluate_having_expr(
                    having,
                    aggregate_values,
                    representative_batch,
                    &column_lookup_map,
                    group_state.representative_row,
                )?;
                // Only include row if HAVING evaluates to TRUE (not FALSE or NULL)
                if matches!(passes, Some(true)) {
                    filtered.push(row.clone());
                }
            }
            filtered
        } else {
            rows
        };

        let fields: Vec<Field> = output_columns
            .into_iter()
            .map(|output| output.field)
            .collect();
        let schema = Arc::new(Schema::new(fields));

        let mut batch = rows_to_record_batch(Arc::clone(&schema), &filtered_rows)?;

        if plan.distinct && batch.num_rows() > 0 {
            let mut state = DistinctState::default();
            batch = match distinct_filter_batch(batch, &mut state)? {
                Some(filtered) => filtered,
                None => RecordBatch::new_empty(Arc::clone(&schema)),
            };
        }

        if !plan.order_by.is_empty() && batch.num_rows() > 0 {
            batch = sort_record_batch_with_order(&schema, &batch, &plan.order_by)?;
        }

        Ok(SelectExecution::new_single_batch(
            display_name,
            schema,
            batch,
        ))
    }

    fn execute_aggregates(
        &self,
        table: Arc<ExecutorTable<P>>,
        display_name: String,
        plan: SelectPlan,
        row_filter: Option<std::sync::Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        let table_ref = table.as_ref();
        let distinct = plan.distinct;
        let mut specs: Vec<AggregateSpec> = Vec::with_capacity(plan.aggregates.len());
        for aggregate in plan.aggregates {
            match aggregate {
                AggregateExpr::CountStar { alias, distinct } => {
                    specs.push(AggregateSpec {
                        alias,
                        kind: AggregateKind::Count {
                            field_id: None,
                            distinct,
                        },
                    });
                }
                AggregateExpr::Column {
                    column,
                    alias,
                    function,
                    distinct,
                } => {
                    let col = table_ref.schema.resolve(&column).ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "unknown column '{}' in aggregate",
                            column
                        ))
                    })?;

                    let kind = match function {
                        AggregateFunction::Count => AggregateKind::Count {
                            field_id: Some(col.field_id),
                            distinct,
                        },
                        AggregateFunction::SumInt64 => {
                            let input_type = Self::validate_aggregate_type(
                                Some(col.data_type.clone()),
                                "SUM",
                                &[DataType::Int64, DataType::Float64],
                            )?;
                            AggregateKind::Sum {
                                field_id: col.field_id,
                                data_type: input_type,
                                distinct,
                            }
                        }
                        AggregateFunction::TotalInt64 => {
                            let input_type = Self::validate_aggregate_type(
                                Some(col.data_type.clone()),
                                "TOTAL",
                                &[DataType::Int64, DataType::Float64],
                            )?;
                            AggregateKind::Total {
                                field_id: col.field_id,
                                data_type: input_type,
                                distinct,
                            }
                        }
                        AggregateFunction::MinInt64 => {
                            let input_type = Self::validate_aggregate_type(
                                Some(col.data_type.clone()),
                                "MIN",
                                &[DataType::Int64, DataType::Float64],
                            )?;
                            AggregateKind::Min {
                                field_id: col.field_id,
                                data_type: input_type,
                            }
                        }
                        AggregateFunction::MaxInt64 => {
                            let input_type = Self::validate_aggregate_type(
                                Some(col.data_type.clone()),
                                "MAX",
                                &[DataType::Int64, DataType::Float64],
                            )?;
                            AggregateKind::Max {
                                field_id: col.field_id,
                                data_type: input_type,
                            }
                        }
                        AggregateFunction::CountNulls => {
                            if distinct {
                                return Err(Error::InvalidArgumentError(
                                    "DISTINCT is not supported for COUNT_NULLS".into(),
                                ));
                            }
                            AggregateKind::CountNulls {
                                field_id: col.field_id,
                            }
                        }
                        AggregateFunction::GroupConcat => AggregateKind::GroupConcat {
                            field_id: col.field_id,
                            distinct,
                            separator: ",".to_string(),
                        },
                    };
                    specs.push(AggregateSpec { alias, kind });
                }
            }
        }

        if specs.is_empty() {
            return Err(Error::InvalidArgumentError(
                "aggregate query requires at least one aggregate expression".into(),
            ));
        }

        let had_filter = plan.filter.is_some();
        let filter_expr = match &plan.filter {
            Some(filter_wrapper) => {
                if !filter_wrapper.subqueries.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "EXISTS subqueries not yet implemented in aggregate queries".into(),
                    ));
                }
                let mut translated = crate::translation::expression::translate_predicate(
                    filter_wrapper.predicate.clone(),
                    table.schema.as_ref(),
                    |name| Error::InvalidArgumentError(format!("unknown column '{}'", name)),
                )?;

                // Check if filter contains scalar subqueries
                let mut filter_scalar_ids = FxHashSet::default();
                collect_predicate_scalar_subquery_ids(&translated, &mut filter_scalar_ids);

                if !filter_scalar_ids.is_empty() {
                    // Evaluate each scalar subquery and replace with literal
                    let filter_scalar_lookup: FxHashMap<SubqueryId, &llkv_plan::ScalarSubquery> =
                        plan.scalar_subqueries
                            .iter()
                            .filter(|subquery| filter_scalar_ids.contains(&subquery.id))
                            .map(|subquery| (subquery.id, subquery))
                            .collect();

                    // Create a minimal context for evaluation (aggregates have no row context)
                    let base_schema = Arc::new(Schema::new(Vec::<Field>::new()));
                    let base_lookup = FxHashMap::default();
                    let mut context =
                        CrossProductExpressionContext::new(base_schema.as_ref(), base_lookup)?;
                    let empty_batch =
                        RecordBatch::new_empty(Arc::new(Schema::new(Vec::<Field>::new())));

                    // Evaluate each scalar subquery to a literal
                    let mut scalar_literals: FxHashMap<SubqueryId, Literal> = FxHashMap::default();
                    for (subquery_id, subquery) in filter_scalar_lookup.iter() {
                        let literal = self.evaluate_scalar_subquery_literal(
                            &mut context,
                            subquery,
                            &empty_batch,
                            0,
                        )?;
                        scalar_literals.insert(*subquery_id, literal);
                    }

                    // Rewrite the filter to replace scalar subqueries with literals
                    translated = rewrite_predicate_scalar_subqueries(translated, &scalar_literals)?;
                }

                translated
            }
            None => {
                let field_id = table.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform aggregate scan".into(),
                    )
                })?;
                crate::translation::expression::full_table_scan_filter(field_id)
            }
        };

        // Build projections and track which projection index each spec uses
        let mut projections = Vec::new();
        let mut spec_to_projection: Vec<Option<usize>> = Vec::with_capacity(specs.len());

        for spec in &specs {
            if let Some(field_id) = spec.kind.field_id() {
                let proj_idx = projections.len();
                spec_to_projection.push(Some(proj_idx));
                projections.push(ScanProjection::from(StoreProjection::with_alias(
                    LogicalFieldId::for_user(table.table.table_id(), field_id),
                    table
                        .schema
                        .column_by_field_id(field_id)
                        .map(|c| c.name.clone())
                        .unwrap_or_else(|| format!("col{field_id}")),
                )));
            } else {
                spec_to_projection.push(None);
            }
        }

        if projections.is_empty() {
            let field_id = table.schema.first_field_id().ok_or_else(|| {
                Error::InvalidArgumentError(
                    "table has no columns; cannot perform aggregate scan".into(),
                )
            })?;
            projections.push(ScanProjection::from(StoreProjection::with_alias(
                LogicalFieldId::for_user(table.table.table_id(), field_id),
                table
                    .schema
                    .column_by_field_id(field_id)
                    .map(|c| c.name.clone())
                    .unwrap_or_else(|| format!("col{field_id}")),
            )));
        }

        let options = ScanStreamOptions {
            include_nulls: true,
            order: None,
            row_id_filter: row_filter.clone(),
        };

        let mut states: Vec<AggregateState> = Vec::with_capacity(specs.len());
        // MVCC Note: We cannot use the total_rows shortcut when MVCC visibility filtering
        // is enabled, because some rows may be invisible due to uncommitted or aborted transactions.
        // Always scan to apply proper visibility rules.
        let mut count_star_override: Option<i64> = None;
        if !had_filter && row_filter.is_none() {
            // Only use shortcut if no filter AND no MVCC row filtering
            let total_rows = table.total_rows.load(Ordering::SeqCst);
            tracing::debug!(
                "[AGGREGATE] Using COUNT(*) shortcut: total_rows={}",
                total_rows
            );
            if total_rows > i64::MAX as u64 {
                return Err(Error::InvalidArgumentError(
                    "COUNT(*) result exceeds supported range".into(),
                ));
            }
            count_star_override = Some(total_rows as i64);
        } else {
            tracing::debug!(
                "[AGGREGATE] NOT using COUNT(*) shortcut: had_filter={}, has_row_filter={}",
                had_filter,
                row_filter.is_some()
            );
        }

        for (idx, spec) in specs.iter().enumerate() {
            states.push(AggregateState {
                alias: spec.alias.clone(),
                accumulator: AggregateAccumulator::new_with_projection_index(
                    spec,
                    spec_to_projection[idx],
                    count_star_override,
                )?,
                override_value: match &spec.kind {
                    AggregateKind::Count { field_id: None, .. } => {
                        tracing::debug!(
                            "[AGGREGATE] CountStar override_value={:?}",
                            count_star_override
                        );
                        count_star_override
                    }
                    _ => None,
                },
            });
        }

        let mut error: Option<Error> = None;
        match table.table.scan_stream(
            projections,
            &filter_expr,
            ScanStreamOptions {
                row_id_filter: row_filter.clone(),
                ..options
            },
            |batch| {
                if error.is_some() {
                    return;
                }
                for state in &mut states {
                    if let Err(err) = state.update(&batch) {
                        error = Some(err);
                        return;
                    }
                }
            },
        ) {
            Ok(()) => {}
            Err(llkv_result::Error::NotFound) => {
                // Treat missing storage keys as an empty result set. This occurs
                // for freshly created tables that have no persisted chunks yet.
            }
            Err(err) => return Err(err),
        }
        if let Some(err) = error {
            return Err(err);
        }

        let mut fields = Vec::with_capacity(states.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(states.len());
        for state in states {
            let (field, array) = state.finalize()?;
            fields.push(field);
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        let mut batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;

        if distinct {
            let mut state = DistinctState::default();
            batch = match distinct_filter_batch(batch, &mut state)? {
                Some(filtered) => filtered,
                None => RecordBatch::new_empty(Arc::clone(&schema)),
            };
        }

        let schema = batch.schema();

        Ok(SelectExecution::new_single_batch(
            display_name,
            schema,
            batch,
        ))
    }

    /// Execute a query where computed projections contain embedded aggregates
    /// This extracts aggregates, computes them, then evaluates the scalar expressions
    fn execute_computed_aggregates(
        &self,
        table: Arc<ExecutorTable<P>>,
        display_name: String,
        plan: SelectPlan,
        row_filter: Option<std::sync::Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        use arrow::array::Int64Array;
        use llkv_expr::expr::AggregateCall;

        let table_ref = table.as_ref();
        let distinct = plan.distinct;

        // First, extract all unique aggregates from the projections
        let mut aggregate_specs: Vec<(String, AggregateCall<String>)> = Vec::new();
        for proj in &plan.projections {
            if let SelectProjection::Computed { expr, .. } = proj {
                Self::collect_aggregates(expr, &mut aggregate_specs);
            }
        }

        // Compute the aggregates using the existing aggregate execution infrastructure
        let filter_predicate = plan
            .filter
            .as_ref()
            .map(|wrapper| {
                if !wrapper.subqueries.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "EXISTS subqueries not yet implemented with aggregates".into(),
                    ));
                }
                Ok(wrapper.predicate.clone())
            })
            .transpose()?;

        let computed_aggregates = self.compute_aggregate_values(
            table.clone(),
            &filter_predicate,
            &aggregate_specs,
            row_filter.clone(),
        )?;

        // Now build the final projections by evaluating expressions with aggregates substituted
        let mut fields = Vec::with_capacity(plan.projections.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(plan.projections.len());

        for proj in &plan.projections {
            match proj {
                SelectProjection::AllColumns | SelectProjection::AllColumnsExcept { .. } => {
                    return Err(Error::InvalidArgumentError(
                        "Wildcard projections not supported with computed aggregates".into(),
                    ));
                }
                SelectProjection::Column { name, alias } => {
                    let col = table_ref.schema.resolve(name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", name))
                    })?;
                    let field_name = alias.as_ref().unwrap_or(name);
                    fields.push(arrow::datatypes::Field::new(
                        field_name,
                        col.data_type.clone(),
                        col.nullable,
                    ));
                    // For regular columns in an aggregate query, we'd need to handle GROUP BY
                    // For now, return an error as this is not supported
                    return Err(Error::InvalidArgumentError(
                        "Regular columns not supported in aggregate queries without GROUP BY"
                            .into(),
                    ));
                }
                SelectProjection::Computed { expr, alias } => {
                    // Check if this is a simple aggregate expression
                    if let ScalarExpr::Aggregate(agg) = expr {
                        let key = format!("{:?}", agg);
                        if let Some(agg_value) = computed_aggregates.get(&key) {
                            match agg_value {
                                AggregateValue::Null => {
                                    fields.push(arrow::datatypes::Field::new(
                                        alias,
                                        DataType::Int64,
                                        true,
                                    ));
                                    arrays
                                        .push(Arc::new(Int64Array::from(vec![None::<i64>]))
                                            as ArrayRef);
                                }
                                AggregateValue::Int64(v) => {
                                    fields.push(arrow::datatypes::Field::new(
                                        alias,
                                        DataType::Int64,
                                        true,
                                    ));
                                    arrays.push(
                                        Arc::new(Int64Array::from(vec![Some(*v)])) as ArrayRef
                                    );
                                }
                                AggregateValue::Float64(v) => {
                                    fields.push(arrow::datatypes::Field::new(
                                        alias,
                                        DataType::Float64,
                                        true,
                                    ));
                                    arrays
                                        .push(Arc::new(Float64Array::from(vec![Some(*v)]))
                                            as ArrayRef);
                                }
                                AggregateValue::Decimal128 { value, scale } => {
                                    // Determine precision from the value
                                    let precision = if *value == 0 {
                                        1
                                    } else {
                                        (*value).abs().to_string().len() as u8
                                    };
                                    fields.push(arrow::datatypes::Field::new(
                                        alias,
                                        DataType::Decimal128(precision, *scale),
                                        true,
                                    ));
                                    let array = Decimal128Array::from(vec![Some(*value)])
                                        .with_precision_and_scale(precision, *scale)
                                        .map_err(|e| {
                                            Error::Internal(format!("invalid Decimal128: {}", e))
                                        })?;
                                    arrays.push(Arc::new(array) as ArrayRef);
                                }
                                AggregateValue::String(s) => {
                                    fields.push(arrow::datatypes::Field::new(
                                        alias,
                                        DataType::Utf8,
                                        true,
                                    ));
                                    arrays
                                        .push(Arc::new(StringArray::from(vec![Some(s.as_str())]))
                                            as ArrayRef);
                                }
                            }
                            continue;
                        }
                    }

                    // Complex expression - try to evaluate as integer
                    let value = Self::evaluate_expr_with_aggregates(expr, &computed_aggregates)?;

                    fields.push(arrow::datatypes::Field::new(alias, DataType::Int64, true));

                    let array = Arc::new(Int64Array::from(vec![value])) as ArrayRef;
                    arrays.push(array);
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));
        let mut batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;

        if distinct {
            let mut state = DistinctState::default();
            batch = match distinct_filter_batch(batch, &mut state)? {
                Some(filtered) => filtered,
                None => RecordBatch::new_empty(Arc::clone(&schema)),
            };
        }

        let schema = batch.schema();

        Ok(SelectExecution::new_single_batch(
            display_name,
            schema,
            batch,
        ))
    }

    /// Build an AggregateSpec for cross product GROUP BY (no field_id metadata required).
    /// Uses dummy field_id=0 since projection_index will override it in new_with_projection_index.
    fn build_aggregate_spec_for_cross_product(
        agg_call: &llkv_expr::expr::AggregateCall<String>,
        alias: String,
        data_type: Option<DataType>,
    ) -> ExecutorResult<llkv_aggregate::AggregateSpec> {
        use llkv_expr::expr::AggregateCall;

        let kind = match agg_call {
            AggregateCall::CountStar => llkv_aggregate::AggregateKind::Count {
                field_id: None,
                distinct: false,
            },
            AggregateCall::Count { distinct, .. } => llkv_aggregate::AggregateKind::Count {
                field_id: Some(0),
                distinct: *distinct,
            },
            AggregateCall::Sum { distinct, .. } => llkv_aggregate::AggregateKind::Sum {
                field_id: 0,
                data_type: Self::validate_aggregate_type(
                    data_type.clone(),
                    "SUM",
                    &[DataType::Int64, DataType::Float64],
                )?,
                distinct: *distinct,
            },
            AggregateCall::Total { distinct, .. } => llkv_aggregate::AggregateKind::Total {
                field_id: 0,
                data_type: Self::validate_aggregate_type(
                    data_type.clone(),
                    "TOTAL",
                    &[DataType::Int64, DataType::Float64],
                )?,
                distinct: *distinct,
            },
            AggregateCall::Avg { distinct, .. } => llkv_aggregate::AggregateKind::Avg {
                field_id: 0,
                data_type: Self::validate_aggregate_type(
                    data_type.clone(),
                    "AVG",
                    &[DataType::Int64, DataType::Float64],
                )?,
                distinct: *distinct,
            },
            AggregateCall::Min(_) => llkv_aggregate::AggregateKind::Min {
                field_id: 0,
                data_type: Self::validate_aggregate_type(
                    data_type.clone(),
                    "MIN",
                    &[DataType::Int64, DataType::Float64],
                )?,
            },
            AggregateCall::Max(_) => llkv_aggregate::AggregateKind::Max {
                field_id: 0,
                data_type: Self::validate_aggregate_type(
                    data_type.clone(),
                    "MAX",
                    &[DataType::Int64, DataType::Float64],
                )?,
            },
            AggregateCall::CountNulls(_) => {
                llkv_aggregate::AggregateKind::CountNulls { field_id: 0 }
            }
            AggregateCall::GroupConcat {
                distinct,
                separator,
                ..
            } => llkv_aggregate::AggregateKind::GroupConcat {
                field_id: 0,
                distinct: *distinct,
                separator: separator.clone().unwrap_or_else(|| ",".to_string()),
            },
        };

        Ok(llkv_aggregate::AggregateSpec { alias, kind })
    }

    /// Validate or coerce input data type for aggregates with centralized, scalable type handling.
    ///
    /// For numeric aggregates (e.g. SUM, AVG, TOTAL):
    /// - Implements SQLite-compatible type coercion automatically
    /// - Int64 and Float64 are used directly
    /// - Utf8 (strings), Boolean, and Date types are coerced to Float64
    /// - String values will be parsed at runtime (non-numeric strings -> 0)
    /// - **Ignores** the `allowed` parameter for these functions
    ///
    /// For other aggregates (e.g. MIN, MAX, COUNT, etc.):
    /// - Only allows types in the `allowed` list
    fn validate_aggregate_type(
        data_type: Option<DataType>,
        func_name: &str,
        allowed: &[DataType],
    ) -> ExecutorResult<DataType> {
        let dt = data_type.ok_or_else(|| {
            Error::Internal(format!(
                "missing input type metadata for {func_name} aggregate"
            ))
        })?;

        // Numeric aggregates (SUM, AVG, TOTAL) and comparison aggregates (MIN, MAX)
        // all support SQLite-style type coercion
        if matches!(func_name, "SUM" | "AVG" | "TOTAL" | "MIN" | "MAX") {
            match dt {
                // Numeric types used directly
                DataType::Int64 | DataType::Float64 | DataType::Decimal128(_, _) => Ok(dt),

                // SQLite-compatible coercion: strings, booleans, dates -> Float64
                // Actual conversion happens in llkv-aggregate::array_value_to_numeric
                DataType::Utf8 | DataType::Boolean | DataType::Date32 => Ok(DataType::Float64),

                // Null type can occur when expression is all NULLs - aggregate result will be NULL
                // Use Float64 as a safe default type for the aggregate accumulator
                DataType::Null => Ok(DataType::Float64),

                _ => Err(Error::InvalidArgumentError(format!(
                    "{func_name} aggregate not supported for column type {:?}",
                    dt
                ))),
            }
        } else {
            // Other aggregates use explicit allowed list
            if allowed.iter().any(|candidate| candidate == &dt) {
                Ok(dt)
            } else {
                Err(Error::InvalidArgumentError(format!(
                    "{func_name} aggregate not supported for column type {:?}",
                    dt
                )))
            }
        }
    }

    /// Collect all aggregate calls from an expression
    fn collect_aggregates(
        expr: &ScalarExpr<String>,
        aggregates: &mut Vec<(String, llkv_expr::expr::AggregateCall<String>)>,
    ) {
        match expr {
            ScalarExpr::Aggregate(agg) => {
                // Create a unique key for this aggregate
                let key = format!("{:?}", agg);
                if !aggregates.iter().any(|(k, _)| k == &key) {
                    aggregates.push((key, agg.clone()));
                }
            }
            ScalarExpr::Binary { left, right, .. } => {
                Self::collect_aggregates(left, aggregates);
                Self::collect_aggregates(right, aggregates);
            }
            ScalarExpr::Compare { left, right, .. } => {
                Self::collect_aggregates(left, aggregates);
                Self::collect_aggregates(right, aggregates);
            }
            ScalarExpr::GetField { base, .. } => {
                Self::collect_aggregates(base, aggregates);
            }
            ScalarExpr::Cast { expr, .. } => {
                Self::collect_aggregates(expr, aggregates);
            }
            ScalarExpr::Not(expr) => {
                Self::collect_aggregates(expr, aggregates);
            }
            ScalarExpr::IsNull { expr, .. } => {
                Self::collect_aggregates(expr, aggregates);
            }
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                if let Some(inner) = operand.as_deref() {
                    Self::collect_aggregates(inner, aggregates);
                }
                for (when_expr, then_expr) in branches {
                    Self::collect_aggregates(when_expr, aggregates);
                    Self::collect_aggregates(then_expr, aggregates);
                }
                if let Some(inner) = else_expr.as_deref() {
                    Self::collect_aggregates(inner, aggregates);
                }
            }
            ScalarExpr::Coalesce(items) => {
                for item in items {
                    Self::collect_aggregates(item, aggregates);
                }
            }
            ScalarExpr::Column(_) | ScalarExpr::Literal(_) | ScalarExpr::Random => {}
            ScalarExpr::ScalarSubquery(_) => {}
        }
    }

    /// Collect aggregates from predicate expressions (Expr, not ScalarExpr)
    fn collect_aggregates_from_predicate(
        expr: &llkv_expr::expr::Expr<String>,
        aggregates: &mut Vec<(String, llkv_expr::expr::AggregateCall<String>)>,
    ) {
        match expr {
            llkv_expr::expr::Expr::Compare { left, right, .. } => {
                Self::collect_aggregates(left, aggregates);
                Self::collect_aggregates(right, aggregates);
            }
            llkv_expr::expr::Expr::And(exprs) | llkv_expr::expr::Expr::Or(exprs) => {
                for e in exprs {
                    Self::collect_aggregates_from_predicate(e, aggregates);
                }
            }
            llkv_expr::expr::Expr::Not(inner) => {
                Self::collect_aggregates_from_predicate(inner, aggregates);
            }
            llkv_expr::expr::Expr::InList {
                expr: test_expr,
                list,
                ..
            } => {
                Self::collect_aggregates(test_expr, aggregates);
                for item in list {
                    Self::collect_aggregates(item, aggregates);
                }
            }
            llkv_expr::expr::Expr::IsNull { expr, .. } => {
                Self::collect_aggregates(expr, aggregates);
            }
            llkv_expr::expr::Expr::Literal(_) => {}
            llkv_expr::expr::Expr::Pred(_) => {}
            llkv_expr::expr::Expr::Exists(_) => {}
        }
    }

    /// Compute the actual values for the aggregates
    fn compute_aggregate_values(
        &self,
        table: Arc<ExecutorTable<P>>,
        filter: &Option<llkv_expr::expr::Expr<'static, String>>,
        aggregate_specs: &[(String, llkv_expr::expr::AggregateCall<String>)],
        row_filter: Option<std::sync::Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<FxHashMap<String, AggregateValue>> {
        use llkv_expr::expr::AggregateCall;

        let table_ref = table.as_ref();
        let mut results =
            FxHashMap::with_capacity_and_hasher(aggregate_specs.len(), Default::default());

        let mut specs: Vec<AggregateSpec> = Vec::with_capacity(aggregate_specs.len());
        let mut spec_to_projection: Vec<Option<usize>> = Vec::with_capacity(aggregate_specs.len());
        let mut projections: Vec<ScanProjection> = Vec::new();
        let mut column_projection_cache: FxHashMap<FieldId, usize> = FxHashMap::default();
        let mut computed_projection_cache: FxHashMap<String, (usize, DataType)> =
            FxHashMap::default();
        let mut computed_alias_counter: usize = 0;

        for (key, agg) in aggregate_specs {
            match agg {
                AggregateCall::CountStar => {
                    specs.push(AggregateSpec {
                        alias: key.clone(),
                        kind: AggregateKind::Count {
                            field_id: None,
                            distinct: false,
                        },
                    });
                    spec_to_projection.push(None);
                }
                AggregateCall::Count { expr, distinct } => {
                    if let Some(col_name) = try_extract_simple_column(expr) {
                        let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                            Error::InvalidArgumentError(format!(
                                "unknown column '{}' in aggregate",
                                col_name
                            ))
                        })?;
                        let projection_index = get_or_insert_column_projection(
                            &mut projections,
                            &mut column_projection_cache,
                            table_ref,
                            col,
                        );
                        specs.push(AggregateSpec {
                            alias: key.clone(),
                            kind: AggregateKind::Count {
                                field_id: Some(col.field_id),
                                distinct: *distinct,
                            },
                        });
                        spec_to_projection.push(Some(projection_index));
                    } else {
                        let (projection_index, _dtype) = ensure_computed_projection(
                            expr,
                            table_ref,
                            &mut projections,
                            &mut computed_projection_cache,
                            &mut computed_alias_counter,
                        )?;
                        let field_id = u32::try_from(projection_index).map_err(|_| {
                            Error::InvalidArgumentError(
                                "aggregate projection index exceeds supported range".into(),
                            )
                        })?;
                        specs.push(AggregateSpec {
                            alias: key.clone(),
                            kind: AggregateKind::Count {
                                field_id: Some(field_id),
                                distinct: *distinct,
                            },
                        });
                        spec_to_projection.push(Some(projection_index));
                    }
                }
                AggregateCall::Sum { expr, distinct } => {
                    let (projection_index, data_type, field_id) =
                        if let Some(col_name) = try_extract_simple_column(expr) {
                            let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                                Error::InvalidArgumentError(format!(
                                    "unknown column '{}' in aggregate",
                                    col_name
                                ))
                            })?;
                            let projection_index = get_or_insert_column_projection(
                                &mut projections,
                                &mut column_projection_cache,
                                table_ref,
                                col,
                            );
                            let data_type = col.data_type.clone();
                            (projection_index, data_type, col.field_id)
                        } else {
                            let (projection_index, inferred_type) = ensure_computed_projection(
                                expr,
                                table_ref,
                                &mut projections,
                                &mut computed_projection_cache,
                                &mut computed_alias_counter,
                            )?;
                            let field_id = u32::try_from(projection_index).map_err(|_| {
                                Error::InvalidArgumentError(
                                    "aggregate projection index exceeds supported range".into(),
                                )
                            })?;
                            (projection_index, inferred_type, field_id)
                        };
                    let normalized_type = Self::validate_aggregate_type(
                        Some(data_type.clone()),
                        "SUM",
                        &[DataType::Int64, DataType::Float64],
                    )?;
                    specs.push(AggregateSpec {
                        alias: key.clone(),
                        kind: AggregateKind::Sum {
                            field_id,
                            data_type: normalized_type,
                            distinct: *distinct,
                        },
                    });
                    spec_to_projection.push(Some(projection_index));
                }
                AggregateCall::Total { expr, distinct } => {
                    let (projection_index, data_type, field_id) =
                        if let Some(col_name) = try_extract_simple_column(expr) {
                            let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                                Error::InvalidArgumentError(format!(
                                    "unknown column '{}' in aggregate",
                                    col_name
                                ))
                            })?;
                            let projection_index = get_or_insert_column_projection(
                                &mut projections,
                                &mut column_projection_cache,
                                table_ref,
                                col,
                            );
                            let data_type = col.data_type.clone();
                            (projection_index, data_type, col.field_id)
                        } else {
                            let (projection_index, inferred_type) = ensure_computed_projection(
                                expr,
                                table_ref,
                                &mut projections,
                                &mut computed_projection_cache,
                                &mut computed_alias_counter,
                            )?;
                            let field_id = u32::try_from(projection_index).map_err(|_| {
                                Error::InvalidArgumentError(
                                    "aggregate projection index exceeds supported range".into(),
                                )
                            })?;
                            (projection_index, inferred_type, field_id)
                        };
                    let normalized_type = Self::validate_aggregate_type(
                        Some(data_type.clone()),
                        "TOTAL",
                        &[DataType::Int64, DataType::Float64],
                    )?;
                    specs.push(AggregateSpec {
                        alias: key.clone(),
                        kind: AggregateKind::Total {
                            field_id,
                            data_type: normalized_type,
                            distinct: *distinct,
                        },
                    });
                    spec_to_projection.push(Some(projection_index));
                }
                AggregateCall::Avg { expr, distinct } => {
                    let (projection_index, data_type, field_id) =
                        if let Some(col_name) = try_extract_simple_column(expr) {
                            let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                                Error::InvalidArgumentError(format!(
                                    "unknown column '{}' in aggregate",
                                    col_name
                                ))
                            })?;
                            let projection_index = get_or_insert_column_projection(
                                &mut projections,
                                &mut column_projection_cache,
                                table_ref,
                                col,
                            );
                            let data_type = col.data_type.clone();
                            (projection_index, data_type, col.field_id)
                        } else {
                            let (projection_index, inferred_type) = ensure_computed_projection(
                                expr,
                                table_ref,
                                &mut projections,
                                &mut computed_projection_cache,
                                &mut computed_alias_counter,
                            )?;
                            tracing::debug!(
                                "AVG aggregate expr={:?} inferred_type={:?}",
                                expr,
                                inferred_type
                            );
                            let field_id = u32::try_from(projection_index).map_err(|_| {
                                Error::InvalidArgumentError(
                                    "aggregate projection index exceeds supported range".into(),
                                )
                            })?;
                            (projection_index, inferred_type, field_id)
                        };
                    let normalized_type = Self::validate_aggregate_type(
                        Some(data_type.clone()),
                        "AVG",
                        &[DataType::Int64, DataType::Float64],
                    )?;
                    specs.push(AggregateSpec {
                        alias: key.clone(),
                        kind: AggregateKind::Avg {
                            field_id,
                            data_type: normalized_type,
                            distinct: *distinct,
                        },
                    });
                    spec_to_projection.push(Some(projection_index));
                }
                AggregateCall::Min(expr) => {
                    let (projection_index, data_type, field_id) =
                        if let Some(col_name) = try_extract_simple_column(expr) {
                            let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                                Error::InvalidArgumentError(format!(
                                    "unknown column '{}' in aggregate",
                                    col_name
                                ))
                            })?;
                            let projection_index = get_or_insert_column_projection(
                                &mut projections,
                                &mut column_projection_cache,
                                table_ref,
                                col,
                            );
                            let data_type = col.data_type.clone();
                            (projection_index, data_type, col.field_id)
                        } else {
                            let (projection_index, inferred_type) = ensure_computed_projection(
                                expr,
                                table_ref,
                                &mut projections,
                                &mut computed_projection_cache,
                                &mut computed_alias_counter,
                            )?;
                            let field_id = u32::try_from(projection_index).map_err(|_| {
                                Error::InvalidArgumentError(
                                    "aggregate projection index exceeds supported range".into(),
                                )
                            })?;
                            (projection_index, inferred_type, field_id)
                        };
                    let normalized_type = Self::validate_aggregate_type(
                        Some(data_type.clone()),
                        "MIN",
                        &[DataType::Int64, DataType::Float64],
                    )?;
                    specs.push(AggregateSpec {
                        alias: key.clone(),
                        kind: AggregateKind::Min {
                            field_id,
                            data_type: normalized_type,
                        },
                    });
                    spec_to_projection.push(Some(projection_index));
                }
                AggregateCall::Max(expr) => {
                    let (projection_index, data_type, field_id) =
                        if let Some(col_name) = try_extract_simple_column(expr) {
                            let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                                Error::InvalidArgumentError(format!(
                                    "unknown column '{}' in aggregate",
                                    col_name
                                ))
                            })?;
                            let projection_index = get_or_insert_column_projection(
                                &mut projections,
                                &mut column_projection_cache,
                                table_ref,
                                col,
                            );
                            let data_type = col.data_type.clone();
                            (projection_index, data_type, col.field_id)
                        } else {
                            let (projection_index, inferred_type) = ensure_computed_projection(
                                expr,
                                table_ref,
                                &mut projections,
                                &mut computed_projection_cache,
                                &mut computed_alias_counter,
                            )?;
                            let field_id = u32::try_from(projection_index).map_err(|_| {
                                Error::InvalidArgumentError(
                                    "aggregate projection index exceeds supported range".into(),
                                )
                            })?;
                            (projection_index, inferred_type, field_id)
                        };
                    let normalized_type = Self::validate_aggregate_type(
                        Some(data_type.clone()),
                        "MAX",
                        &[DataType::Int64, DataType::Float64],
                    )?;
                    specs.push(AggregateSpec {
                        alias: key.clone(),
                        kind: AggregateKind::Max {
                            field_id,
                            data_type: normalized_type,
                        },
                    });
                    spec_to_projection.push(Some(projection_index));
                }
                AggregateCall::CountNulls(expr) => {
                    if let Some(col_name) = try_extract_simple_column(expr) {
                        let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                            Error::InvalidArgumentError(format!(
                                "unknown column '{}' in aggregate",
                                col_name
                            ))
                        })?;
                        let projection_index = get_or_insert_column_projection(
                            &mut projections,
                            &mut column_projection_cache,
                            table_ref,
                            col,
                        );
                        specs.push(AggregateSpec {
                            alias: key.clone(),
                            kind: AggregateKind::CountNulls {
                                field_id: col.field_id,
                            },
                        });
                        spec_to_projection.push(Some(projection_index));
                    } else {
                        let (projection_index, _dtype) = ensure_computed_projection(
                            expr,
                            table_ref,
                            &mut projections,
                            &mut computed_projection_cache,
                            &mut computed_alias_counter,
                        )?;
                        let field_id = u32::try_from(projection_index).map_err(|_| {
                            Error::InvalidArgumentError(
                                "aggregate projection index exceeds supported range".into(),
                            )
                        })?;
                        specs.push(AggregateSpec {
                            alias: key.clone(),
                            kind: AggregateKind::CountNulls { field_id },
                        });
                        spec_to_projection.push(Some(projection_index));
                    }
                }
                AggregateCall::GroupConcat {
                    expr,
                    distinct,
                    separator,
                } => {
                    if let Some(col_name) = try_extract_simple_column(expr) {
                        let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                            Error::InvalidArgumentError(format!(
                                "unknown column '{}' in aggregate",
                                col_name
                            ))
                        })?;
                        let projection_index = get_or_insert_column_projection(
                            &mut projections,
                            &mut column_projection_cache,
                            table_ref,
                            col,
                        );
                        specs.push(AggregateSpec {
                            alias: key.clone(),
                            kind: AggregateKind::GroupConcat {
                                field_id: col.field_id,
                                distinct: *distinct,
                                separator: separator.clone().unwrap_or_else(|| ",".to_string()),
                            },
                        });
                        spec_to_projection.push(Some(projection_index));
                    } else {
                        let (projection_index, _dtype) = ensure_computed_projection(
                            expr,
                            table_ref,
                            &mut projections,
                            &mut computed_projection_cache,
                            &mut computed_alias_counter,
                        )?;
                        let field_id = u32::try_from(projection_index).map_err(|_| {
                            Error::InvalidArgumentError(
                                "aggregate projection index exceeds supported range".into(),
                            )
                        })?;
                        specs.push(AggregateSpec {
                            alias: key.clone(),
                            kind: AggregateKind::GroupConcat {
                                field_id,
                                distinct: *distinct,
                                separator: separator.clone().unwrap_or_else(|| ",".to_string()),
                            },
                        });
                        spec_to_projection.push(Some(projection_index));
                    }
                }
            }
        }

        let filter_expr = match filter {
            Some(expr) => crate::translation::expression::translate_predicate(
                expr.clone(),
                table_ref.schema.as_ref(),
                |name| Error::InvalidArgumentError(format!("unknown column '{}'", name)),
            )?,
            None => {
                let field_id = table_ref.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform aggregate scan".into(),
                    )
                })?;
                crate::translation::expression::full_table_scan_filter(field_id)
            }
        };

        if projections.is_empty() {
            let field_id = table_ref.schema.first_field_id().ok_or_else(|| {
                Error::InvalidArgumentError(
                    "table has no columns; cannot perform aggregate scan".into(),
                )
            })?;
            projections.push(ScanProjection::from(StoreProjection::with_alias(
                LogicalFieldId::for_user(table.table.table_id(), field_id),
                table
                    .schema
                    .column_by_field_id(field_id)
                    .map(|c| c.name.clone())
                    .unwrap_or_else(|| format!("col{field_id}")),
            )));
        }

        let base_options = ScanStreamOptions {
            include_nulls: true,
            order: None,
            row_id_filter: None,
        };

        let count_star_override: Option<i64> = None;

        let mut states: Vec<AggregateState> = Vec::with_capacity(specs.len());
        for (idx, spec) in specs.iter().enumerate() {
            states.push(AggregateState {
                alias: spec.alias.clone(),
                accumulator: AggregateAccumulator::new_with_projection_index(
                    spec,
                    spec_to_projection[idx],
                    count_star_override,
                )?,
                override_value: match &spec.kind {
                    AggregateKind::Count { field_id: None, .. } => count_star_override,
                    _ => None,
                },
            });
        }

        let mut error: Option<Error> = None;
        match table.table.scan_stream(
            projections,
            &filter_expr,
            ScanStreamOptions {
                row_id_filter: row_filter.clone(),
                ..base_options
            },
            |batch| {
                if error.is_some() {
                    return;
                }
                for state in &mut states {
                    if let Err(err) = state.update(&batch) {
                        error = Some(err);
                        return;
                    }
                }
            },
        ) {
            Ok(()) => {}
            Err(llkv_result::Error::NotFound) => {}
            Err(err) => return Err(err),
        }
        if let Some(err) = error {
            return Err(err);
        }

        for state in states {
            let alias = state.alias.clone();
            let (_field, array) = state.finalize()?;

            if let Some(int64_array) = array.as_any().downcast_ref::<arrow::array::Int64Array>() {
                if int64_array.len() != 1 {
                    return Err(Error::Internal(format!(
                        "Expected single value from aggregate, got {}",
                        int64_array.len()
                    )));
                }
                let value = if int64_array.is_null(0) {
                    AggregateValue::Null
                } else {
                    AggregateValue::Int64(int64_array.value(0))
                };
                results.insert(alias, value);
            } else if let Some(float64_array) =
                array.as_any().downcast_ref::<arrow::array::Float64Array>()
            {
                if float64_array.len() != 1 {
                    return Err(Error::Internal(format!(
                        "Expected single value from aggregate, got {}",
                        float64_array.len()
                    )));
                }
                let value = if float64_array.is_null(0) {
                    AggregateValue::Null
                } else {
                    AggregateValue::Float64(float64_array.value(0))
                };
                results.insert(alias, value);
            } else if let Some(string_array) =
                array.as_any().downcast_ref::<arrow::array::StringArray>()
            {
                if string_array.len() != 1 {
                    return Err(Error::Internal(format!(
                        "Expected single value from aggregate, got {}",
                        string_array.len()
                    )));
                }
                let value = if string_array.is_null(0) {
                    AggregateValue::Null
                } else {
                    AggregateValue::String(string_array.value(0).to_string())
                };
                results.insert(alias, value);
            } else if let Some(decimal_array) = array
                .as_any()
                .downcast_ref::<arrow::array::Decimal128Array>()
            {
                if decimal_array.len() != 1 {
                    return Err(Error::Internal(format!(
                        "Expected single value from aggregate, got {}",
                        decimal_array.len()
                    )));
                }
                let value = if decimal_array.is_null(0) {
                    AggregateValue::Null
                } else {
                    AggregateValue::Decimal128 {
                        value: decimal_array.value(0),
                        scale: decimal_array.scale(),
                    }
                };
                results.insert(alias, value);
            } else {
                return Err(Error::Internal(format!(
                    "Unexpected array type from aggregate: {:?}",
                    array.data_type()
                )));
            }
        }

        Ok(results)
    }

    fn evaluate_having_expr(
        expr: &llkv_expr::expr::Expr<String>,
        aggregates: &FxHashMap<String, PlanValue>,
        row_batch: &RecordBatch,
        column_lookup: &FxHashMap<String, usize>,
        row_idx: usize,
    ) -> ExecutorResult<Option<bool>> {
        fn compare_plan_values_for_pred(
            left: &PlanValue,
            right: &PlanValue,
        ) -> Option<std::cmp::Ordering> {
            match (left, right) {
                (PlanValue::Integer(l), PlanValue::Integer(r)) => Some(l.cmp(r)),
                (PlanValue::Float(l), PlanValue::Float(r)) => l.partial_cmp(r),
                (PlanValue::Integer(l), PlanValue::Float(r)) => (*l as f64).partial_cmp(r),
                (PlanValue::Float(l), PlanValue::Integer(r)) => l.partial_cmp(&(*r as f64)),
                (PlanValue::String(l), PlanValue::String(r)) => Some(l.cmp(r)),
                (PlanValue::Interval(l), PlanValue::Interval(r)) => {
                    Some(compare_interval_values(*l, *r))
                }
                _ => None,
            }
        }

        fn evaluate_ordering_predicate<F>(
            value: &PlanValue,
            literal: &Literal,
            predicate: F,
        ) -> ExecutorResult<Option<bool>>
        where
            F: Fn(std::cmp::Ordering) -> bool,
        {
            if matches!(value, PlanValue::Null) {
                return Ok(None);
            }
            let expected = llkv_plan::plan_value_from_literal(literal)?;
            if matches!(expected, PlanValue::Null) {
                return Ok(None);
            }

            match compare_plan_values_for_pred(value, &expected) {
                Some(ordering) => Ok(Some(predicate(ordering))),
                None => Err(Error::InvalidArgumentError(
                    "unsupported HAVING comparison between column value and literal".into(),
                )),
            }
        }

        match expr {
            llkv_expr::expr::Expr::Compare { left, op, right } => {
                let left_val = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                    left,
                    aggregates,
                    Some(row_batch),
                    Some(column_lookup),
                    row_idx,
                )?;
                let right_val = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                    right,
                    aggregates,
                    Some(row_batch),
                    Some(column_lookup),
                    row_idx,
                )?;

                // Coerce numeric types for comparison
                let (left_val, right_val) = match (&left_val, &right_val) {
                    (PlanValue::Integer(i), PlanValue::Float(_)) => {
                        (PlanValue::Float(*i as f64), right_val)
                    }
                    (PlanValue::Float(_), PlanValue::Integer(i)) => {
                        (left_val, PlanValue::Float(*i as f64))
                    }
                    _ => (left_val, right_val),
                };

                match (left_val, right_val) {
                    // NULL comparisons return NULL (represented as None)
                    (PlanValue::Null, _) | (_, PlanValue::Null) => Ok(None),
                    (PlanValue::Integer(l), PlanValue::Integer(r)) => {
                        use llkv_expr::expr::CompareOp;
                        Ok(Some(match op {
                            CompareOp::Eq => l == r,
                            CompareOp::NotEq => l != r,
                            CompareOp::Lt => l < r,
                            CompareOp::LtEq => l <= r,
                            CompareOp::Gt => l > r,
                            CompareOp::GtEq => l >= r,
                        }))
                    }
                    (PlanValue::Float(l), PlanValue::Float(r)) => {
                        use llkv_expr::expr::CompareOp;
                        Ok(Some(match op {
                            CompareOp::Eq => l == r,
                            CompareOp::NotEq => l != r,
                            CompareOp::Lt => l < r,
                            CompareOp::LtEq => l <= r,
                            CompareOp::Gt => l > r,
                            CompareOp::GtEq => l >= r,
                        }))
                    }
                    (PlanValue::Interval(l), PlanValue::Interval(r)) => {
                        use llkv_expr::expr::CompareOp;
                        let ordering = compare_interval_values(l, r);
                        Ok(Some(match op {
                            CompareOp::Eq => ordering == std::cmp::Ordering::Equal,
                            CompareOp::NotEq => ordering != std::cmp::Ordering::Equal,
                            CompareOp::Lt => ordering == std::cmp::Ordering::Less,
                            CompareOp::LtEq => {
                                matches!(
                                    ordering,
                                    std::cmp::Ordering::Less | std::cmp::Ordering::Equal
                                )
                            }
                            CompareOp::Gt => ordering == std::cmp::Ordering::Greater,
                            CompareOp::GtEq => {
                                matches!(
                                    ordering,
                                    std::cmp::Ordering::Greater | std::cmp::Ordering::Equal
                                )
                            }
                        }))
                    }
                    _ => Ok(Some(false)),
                }
            }
            llkv_expr::expr::Expr::Not(inner) => {
                // NOT NULL = NULL, NOT TRUE = FALSE, NOT FALSE = TRUE
                match Self::evaluate_having_expr(
                    inner,
                    aggregates,
                    row_batch,
                    column_lookup,
                    row_idx,
                )? {
                    Some(b) => Ok(Some(!b)),
                    None => Ok(None), // NOT NULL = NULL
                }
            }
            llkv_expr::expr::Expr::InList {
                expr: test_expr,
                list,
                negated,
            } => {
                let test_val = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                    test_expr,
                    aggregates,
                    Some(row_batch),
                    Some(column_lookup),
                    row_idx,
                )?;

                // SQL semantics: test_value IN (NULL, ...) handling
                // - If test_val is NULL, result is always NULL
                if matches!(test_val, PlanValue::Null) {
                    return Ok(None);
                }

                let mut found = false;
                let mut has_null = false;

                for list_item in list {
                    let list_val = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                        list_item,
                        aggregates,
                        Some(row_batch),
                        Some(column_lookup),
                        row_idx,
                    )?;

                    // Track if list contains NULL
                    if matches!(list_val, PlanValue::Null) {
                        has_null = true;
                        continue;
                    }

                    // Coerce for comparison
                    let matches = match (&test_val, &list_val) {
                        (PlanValue::Integer(a), PlanValue::Integer(b)) => a == b,
                        (PlanValue::Float(a), PlanValue::Float(b)) => a == b,
                        (PlanValue::Integer(a), PlanValue::Float(b)) => (*a as f64) == *b,
                        (PlanValue::Float(a), PlanValue::Integer(b)) => *a == (*b as f64),
                        (PlanValue::String(a), PlanValue::String(b)) => a == b,
                        (PlanValue::Interval(a), PlanValue::Interval(b)) => {
                            compare_interval_values(*a, *b) == std::cmp::Ordering::Equal
                        }
                        _ => false,
                    };

                    if matches {
                        found = true;
                        break;
                    }
                }

                // SQL semantics for IN/NOT IN with NULL:
                // - value IN (...): TRUE if match found, FALSE if no match and no NULLs, NULL if no match but has NULLs
                // - value NOT IN (...): FALSE if match found, TRUE if no match and no NULLs, NULL if no match but has NULLs
                if *negated {
                    // NOT IN
                    Ok(if found {
                        Some(false)
                    } else if has_null {
                        None // NULL in list makes NOT IN return NULL
                    } else {
                        Some(true)
                    })
                } else {
                    // IN
                    Ok(if found {
                        Some(true)
                    } else if has_null {
                        None // No match but NULL in list returns NULL
                    } else {
                        Some(false)
                    })
                }
            }
            llkv_expr::expr::Expr::IsNull { expr, negated } => {
                // Evaluate the expression to get its value
                let val = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                    expr,
                    aggregates,
                    Some(row_batch),
                    Some(column_lookup),
                    row_idx,
                )?;

                // IS NULL / IS NOT NULL returns a boolean (not NULL) even when testing NULL
                // NULL IS NULL = TRUE
                // NULL IS NOT NULL = FALSE
                let is_null = matches!(val, PlanValue::Null);
                Ok(Some(if *negated { !is_null } else { is_null }))
            }
            llkv_expr::expr::Expr::Literal(val) => Ok(Some(*val)),
            llkv_expr::expr::Expr::And(exprs) => {
                // AND with NULL: FALSE AND anything = FALSE, TRUE AND NULL = NULL, NULL AND TRUE = NULL
                let mut has_null = false;
                for e in exprs {
                    match Self::evaluate_having_expr(
                        e,
                        aggregates,
                        row_batch,
                        column_lookup,
                        row_idx,
                    )? {
                        Some(false) => return Ok(Some(false)), // Short-circuit on FALSE
                        None => has_null = true,
                        Some(true) => {} // Continue
                    }
                }
                Ok(if has_null { None } else { Some(true) })
            }
            llkv_expr::expr::Expr::Or(exprs) => {
                // OR with NULL: TRUE OR anything = TRUE, FALSE OR NULL = NULL, NULL OR FALSE = NULL
                let mut has_null = false;
                for e in exprs {
                    match Self::evaluate_having_expr(
                        e,
                        aggregates,
                        row_batch,
                        column_lookup,
                        row_idx,
                    )? {
                        Some(true) => return Ok(Some(true)), // Short-circuit on TRUE
                        None => has_null = true,
                        Some(false) => {} // Continue
                    }
                }
                Ok(if has_null { None } else { Some(false) })
            }
            llkv_expr::expr::Expr::Pred(filter) => {
                // Handle Filter predicates (e.g., column IS NULL, column IS NOT NULL)
                // In HAVING context, filters reference columns in the grouped result
                use llkv_expr::expr::Operator;

                let col_name = &filter.field_id;
                let col_idx = column_lookup
                    .get(&col_name.to_ascii_lowercase())
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "column '{}' not found in HAVING context",
                            col_name
                        ))
                    })?;

                let value = llkv_plan::plan_value_from_array(row_batch.column(*col_idx), row_idx)?;

                match &filter.op {
                    Operator::IsNull => Ok(Some(matches!(value, PlanValue::Null))),
                    Operator::IsNotNull => Ok(Some(!matches!(value, PlanValue::Null))),
                    Operator::Equals(expected) => {
                        // NULL comparisons return NULL
                        if matches!(value, PlanValue::Null) {
                            return Ok(None);
                        }
                        // Compare the value with the expected literal
                        let expected_value = llkv_plan::plan_value_from_literal(expected)?;
                        if matches!(expected_value, PlanValue::Null) {
                            return Ok(None);
                        }
                        Ok(Some(value == expected_value))
                    }
                    Operator::GreaterThan(expected) => {
                        evaluate_ordering_predicate(&value, expected, |ordering| {
                            ordering == std::cmp::Ordering::Greater
                        })
                    }
                    Operator::GreaterThanOrEquals(expected) => {
                        evaluate_ordering_predicate(&value, expected, |ordering| {
                            ordering == std::cmp::Ordering::Greater
                                || ordering == std::cmp::Ordering::Equal
                        })
                    }
                    Operator::LessThan(expected) => {
                        evaluate_ordering_predicate(&value, expected, |ordering| {
                            ordering == std::cmp::Ordering::Less
                        })
                    }
                    Operator::LessThanOrEquals(expected) => {
                        evaluate_ordering_predicate(&value, expected, |ordering| {
                            ordering == std::cmp::Ordering::Less
                                || ordering == std::cmp::Ordering::Equal
                        })
                    }
                    _ => {
                        // For other operators, fall back to a general error
                        // These should ideally be translated to Compare expressions instead of Pred
                        Err(Error::InvalidArgumentError(format!(
                            "Operator {:?} not supported for column predicates in HAVING clause",
                            filter.op
                        )))
                    }
                }
            }
            llkv_expr::expr::Expr::Exists(_) => Err(Error::InvalidArgumentError(
                "EXISTS subqueries not supported in HAVING clause".into(),
            )),
        }
    }

    fn evaluate_expr_with_plan_value_aggregates_and_row(
        expr: &ScalarExpr<String>,
        aggregates: &FxHashMap<String, PlanValue>,
        row_batch: Option<&RecordBatch>,
        column_lookup: Option<&FxHashMap<String, usize>>,
        row_idx: usize,
    ) -> ExecutorResult<PlanValue> {
        use llkv_expr::expr::BinaryOp;
        use llkv_expr::literal::Literal;

        match expr {
            ScalarExpr::Literal(Literal::Int128(v)) => Ok(PlanValue::Integer(*v as i64)),
            ScalarExpr::Literal(Literal::Float64(v)) => Ok(PlanValue::Float(*v)),
            ScalarExpr::Literal(Literal::Decimal128(value)) => Ok(PlanValue::Decimal(*value)),
            ScalarExpr::Literal(Literal::Boolean(v)) => {
                Ok(PlanValue::Integer(if *v { 1 } else { 0 }))
            }
            ScalarExpr::Literal(Literal::String(s)) => Ok(PlanValue::String(s.clone())),
            ScalarExpr::Literal(Literal::Date32(days)) => Ok(PlanValue::Date32(*days)),
            ScalarExpr::Literal(Literal::Interval(interval)) => Ok(PlanValue::Interval(*interval)),
            ScalarExpr::Literal(Literal::Null) => Ok(PlanValue::Null),
            ScalarExpr::Literal(Literal::Struct(_)) => Err(Error::InvalidArgumentError(
                "Struct literals not supported in aggregate expressions".into(),
            )),
            ScalarExpr::Column(col_name) => {
                // If row context is provided, look up the column value
                if let (Some(batch), Some(lookup)) = (row_batch, column_lookup) {
                    let col_idx = lookup.get(&col_name.to_ascii_lowercase()).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("column '{}' not found", col_name))
                    })?;
                    llkv_plan::plan_value_from_array(batch.column(*col_idx), row_idx)
                } else {
                    Err(Error::InvalidArgumentError(
                        "Column references not supported in aggregate-only expressions".into(),
                    ))
                }
            }
            ScalarExpr::Compare { left, op, right } => {
                // Evaluate both sides of the comparison
                let left_val = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                    left,
                    aggregates,
                    row_batch,
                    column_lookup,
                    row_idx,
                )?;
                let right_val = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                    right,
                    aggregates,
                    row_batch,
                    column_lookup,
                    row_idx,
                )?;

                // SQL three-valued logic: comparisons involving NULL yield NULL.
                if matches!(left_val, PlanValue::Null) || matches!(right_val, PlanValue::Null) {
                    return Ok(PlanValue::Null);
                }

                // Coerce types for comparison
                let (left_val, right_val) = match (&left_val, &right_val) {
                    (PlanValue::Integer(i), PlanValue::Float(_)) => {
                        (PlanValue::Float(*i as f64), right_val)
                    }
                    (PlanValue::Float(_), PlanValue::Integer(i)) => {
                        (left_val, PlanValue::Float(*i as f64))
                    }
                    _ => (left_val, right_val),
                };

                // Perform the comparison
                let result = match (&left_val, &right_val) {
                    (PlanValue::Integer(l), PlanValue::Integer(r)) => {
                        use llkv_expr::expr::CompareOp;
                        match op {
                            CompareOp::Eq => l == r,
                            CompareOp::NotEq => l != r,
                            CompareOp::Lt => l < r,
                            CompareOp::LtEq => l <= r,
                            CompareOp::Gt => l > r,
                            CompareOp::GtEq => l >= r,
                        }
                    }
                    (PlanValue::Float(l), PlanValue::Float(r)) => {
                        use llkv_expr::expr::CompareOp;
                        match op {
                            CompareOp::Eq => l == r,
                            CompareOp::NotEq => l != r,
                            CompareOp::Lt => l < r,
                            CompareOp::LtEq => l <= r,
                            CompareOp::Gt => l > r,
                            CompareOp::GtEq => l >= r,
                        }
                    }
                    (PlanValue::String(l), PlanValue::String(r)) => {
                        use llkv_expr::expr::CompareOp;
                        match op {
                            CompareOp::Eq => l == r,
                            CompareOp::NotEq => l != r,
                            CompareOp::Lt => l < r,
                            CompareOp::LtEq => l <= r,
                            CompareOp::Gt => l > r,
                            CompareOp::GtEq => l >= r,
                        }
                    }
                    (PlanValue::Interval(l), PlanValue::Interval(r)) => {
                        use llkv_expr::expr::CompareOp;
                        let ordering = compare_interval_values(*l, *r);
                        match op {
                            CompareOp::Eq => ordering == std::cmp::Ordering::Equal,
                            CompareOp::NotEq => ordering != std::cmp::Ordering::Equal,
                            CompareOp::Lt => ordering == std::cmp::Ordering::Less,
                            CompareOp::LtEq => {
                                matches!(
                                    ordering,
                                    std::cmp::Ordering::Less | std::cmp::Ordering::Equal
                                )
                            }
                            CompareOp::Gt => ordering == std::cmp::Ordering::Greater,
                            CompareOp::GtEq => {
                                matches!(
                                    ordering,
                                    std::cmp::Ordering::Greater | std::cmp::Ordering::Equal
                                )
                            }
                        }
                    }
                    _ => false,
                };

                // Return 1 for true, 0 for false (integer representation of boolean)
                Ok(PlanValue::Integer(if result { 1 } else { 0 }))
            }
            ScalarExpr::Not(inner) => {
                let value = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                    inner,
                    aggregates,
                    row_batch,
                    column_lookup,
                    row_idx,
                )?;
                match value {
                    PlanValue::Integer(v) => Ok(PlanValue::Integer(if v != 0 { 0 } else { 1 })),
                    PlanValue::Float(v) => Ok(PlanValue::Integer(if v != 0.0 { 0 } else { 1 })),
                    PlanValue::Null => Ok(PlanValue::Null),
                    other => Err(Error::InvalidArgumentError(format!(
                        "logical NOT does not support value {other:?}"
                    ))),
                }
            }
            ScalarExpr::IsNull { expr, negated } => {
                let value = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                    expr,
                    aggregates,
                    row_batch,
                    column_lookup,
                    row_idx,
                )?;
                let is_null = matches!(value, PlanValue::Null);
                let condition = if is_null { !negated } else { *negated };
                Ok(PlanValue::Integer(if condition { 1 } else { 0 }))
            }
            ScalarExpr::Aggregate(agg) => {
                let key = format!("{:?}", agg);
                aggregates
                    .get(&key)
                    .cloned()
                    .ok_or_else(|| Error::Internal(format!("Aggregate value not found: {}", key)))
            }
            ScalarExpr::Binary { left, op, right } => {
                let left_val = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                    left,
                    aggregates,
                    row_batch,
                    column_lookup,
                    row_idx,
                )?;
                let right_val = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                    right,
                    aggregates,
                    row_batch,
                    column_lookup,
                    row_idx,
                )?;

                match op {
                    BinaryOp::Add
                    | BinaryOp::Subtract
                    | BinaryOp::Multiply
                    | BinaryOp::Divide
                    | BinaryOp::Modulo => {
                        if matches!(&left_val, PlanValue::Null)
                            || matches!(&right_val, PlanValue::Null)
                        {
                            return Ok(PlanValue::Null);
                        }

                        if matches!(left_val, PlanValue::Interval(_))
                            || matches!(right_val, PlanValue::Interval(_))
                        {
                            return Err(Error::InvalidArgumentError(
                                "interval arithmetic not supported in aggregate expressions".into(),
                            ));
                        }

                        // Special case: integer division
                        if matches!(op, BinaryOp::Divide)
                            && let (PlanValue::Integer(lhs), PlanValue::Integer(rhs)) =
                                (&left_val, &right_val)
                        {
                            if *rhs == 0 {
                                return Ok(PlanValue::Null);
                            }

                            if *lhs == i64::MIN && *rhs == -1 {
                                return Ok(PlanValue::Float((*lhs as f64) / (*rhs as f64)));
                            }

                            return Ok(PlanValue::Integer(lhs / rhs));
                        }

                        // Check if either operand is Decimal - if so, use exact Decimal arithmetic
                        let has_decimal = matches!(&left_val, PlanValue::Decimal(_))
                            || matches!(&right_val, PlanValue::Decimal(_));

                        if has_decimal {
                            use llkv_types::decimal::DecimalValue;

                            // Convert both operands to Decimal
                            let left_dec = match &left_val {
                                PlanValue::Integer(i) => DecimalValue::from_i64(*i),
                                PlanValue::Float(_f) => {
                                    // For Float, we lose some precision but maintain the value
                                    return Err(Error::InvalidArgumentError(
                                        "Cannot perform exact decimal arithmetic with Float operands"
                                            .into(),
                                    ));
                                }
                                PlanValue::Decimal(d) => *d,
                                other => {
                                    return Err(Error::InvalidArgumentError(format!(
                                        "Non-numeric value {:?} in binary operation",
                                        other
                                    )));
                                }
                            };

                            let right_dec = match &right_val {
                                PlanValue::Integer(i) => DecimalValue::from_i64(*i),
                                PlanValue::Float(_f) => {
                                    return Err(Error::InvalidArgumentError(
                                        "Cannot perform exact decimal arithmetic with Float operands"
                                            .into(),
                                    ));
                                }
                                PlanValue::Decimal(d) => *d,
                                other => {
                                    return Err(Error::InvalidArgumentError(format!(
                                        "Non-numeric value {:?} in binary operation",
                                        other
                                    )));
                                }
                            };

                            // Perform exact decimal arithmetic
                            let result_dec = match op {
                                BinaryOp::Add => {
                                    llkv_compute::scalar::decimal::add(left_dec, right_dec)
                                        .map_err(|e| {
                                            Error::InvalidArgumentError(format!(
                                                "Decimal addition overflow: {}",
                                                e
                                            ))
                                        })?
                                }
                                BinaryOp::Subtract => {
                                    llkv_compute::scalar::decimal::sub(left_dec, right_dec)
                                        .map_err(|e| {
                                            Error::InvalidArgumentError(format!(
                                                "Decimal subtraction overflow: {}",
                                                e
                                            ))
                                        })?
                                }
                                BinaryOp::Multiply => {
                                    llkv_compute::scalar::decimal::mul(left_dec, right_dec)
                                        .map_err(|e| {
                                            Error::InvalidArgumentError(format!(
                                                "Decimal multiplication overflow: {}",
                                                e
                                            ))
                                        })?
                                }
                                BinaryOp::Divide => {
                                    // Check for division by zero first
                                    if right_dec.raw_value() == 0 {
                                        return Ok(PlanValue::Null);
                                    }
                                    // For division, preserve scale of left operand
                                    let target_scale = left_dec.scale();
                                    llkv_compute::scalar::decimal::div(
                                        left_dec,
                                        right_dec,
                                        target_scale,
                                    )
                                    .map_err(|e| {
                                        Error::InvalidArgumentError(format!(
                                            "Decimal division error: {}",
                                            e
                                        ))
                                    })?
                                }
                                BinaryOp::Modulo => {
                                    return Err(Error::InvalidArgumentError(
                                        "Modulo not supported for Decimal types".into(),
                                    ));
                                }
                                BinaryOp::And
                                | BinaryOp::Or
                                | BinaryOp::BitwiseShiftLeft
                                | BinaryOp::BitwiseShiftRight => unreachable!(),
                            };

                            return Ok(PlanValue::Decimal(result_dec));
                        }

                        // No decimals - use Float or Integer arithmetic as before
                        let left_is_float = matches!(&left_val, PlanValue::Float(_));
                        let right_is_float = matches!(&right_val, PlanValue::Float(_));

                        let left_num = match left_val {
                            PlanValue::Integer(i) => i as f64,
                            PlanValue::Float(f) => f,
                            other => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "Non-numeric value {:?} in binary operation",
                                    other
                                )));
                            }
                        };
                        let right_num = match right_val {
                            PlanValue::Integer(i) => i as f64,
                            PlanValue::Float(f) => f,
                            other => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "Non-numeric value {:?} in binary operation",
                                    other
                                )));
                            }
                        };

                        let result = match op {
                            BinaryOp::Add => left_num + right_num,
                            BinaryOp::Subtract => left_num - right_num,
                            BinaryOp::Multiply => left_num * right_num,
                            BinaryOp::Divide => {
                                if right_num == 0.0 {
                                    return Ok(PlanValue::Null);
                                }
                                left_num / right_num
                            }
                            BinaryOp::Modulo => {
                                if right_num == 0.0 {
                                    return Ok(PlanValue::Null);
                                }
                                left_num % right_num
                            }
                            BinaryOp::And
                            | BinaryOp::Or
                            | BinaryOp::BitwiseShiftLeft
                            | BinaryOp::BitwiseShiftRight => unreachable!(),
                        };

                        if matches!(op, BinaryOp::Divide) {
                            return Ok(PlanValue::Float(result));
                        }

                        if left_is_float || right_is_float {
                            Ok(PlanValue::Float(result))
                        } else {
                            Ok(PlanValue::Integer(result as i64))
                        }
                    }
                    BinaryOp::And => Ok(evaluate_plan_value_logical_and(left_val, right_val)),
                    BinaryOp::Or => Ok(evaluate_plan_value_logical_or(left_val, right_val)),
                    BinaryOp::BitwiseShiftLeft | BinaryOp::BitwiseShiftRight => {
                        if matches!(&left_val, PlanValue::Null)
                            || matches!(&right_val, PlanValue::Null)
                        {
                            return Ok(PlanValue::Null);
                        }

                        // Convert to integers
                        let lhs = match left_val {
                            PlanValue::Integer(i) => i,
                            PlanValue::Float(f) => f as i64,
                            other => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "Non-numeric value {:?} in bitwise shift operation",
                                    other
                                )));
                            }
                        };
                        let rhs = match right_val {
                            PlanValue::Integer(i) => i,
                            PlanValue::Float(f) => f as i64,
                            other => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "Non-numeric value {:?} in bitwise shift operation",
                                    other
                                )));
                            }
                        };

                        // Use wrapping arithmetic like SQLite
                        let result = match op {
                            BinaryOp::BitwiseShiftLeft => lhs.wrapping_shl(rhs as u32),
                            BinaryOp::BitwiseShiftRight => lhs.wrapping_shr(rhs as u32),
                            _ => unreachable!(),
                        };

                        Ok(PlanValue::Integer(result))
                    }
                }
            }
            ScalarExpr::Cast { expr, data_type } => {
                // Evaluate the inner expression and cast it to the target type
                let value = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                    expr,
                    aggregates,
                    row_batch,
                    column_lookup,
                    row_idx,
                )?;

                // Handle NULL values
                if matches!(value, PlanValue::Null) {
                    return Ok(PlanValue::Null);
                }

                // Cast to the target type
                match data_type {
                    DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                        match value {
                            PlanValue::Integer(i) => Ok(PlanValue::Integer(i)),
                            PlanValue::Float(f) => Ok(PlanValue::Integer(f as i64)),
                            PlanValue::String(s) => {
                                s.parse::<i64>().map(PlanValue::Integer).map_err(|_| {
                                    Error::InvalidArgumentError(format!(
                                        "Cannot cast '{}' to integer",
                                        s
                                    ))
                                })
                            }
                            _ => Err(Error::InvalidArgumentError(format!(
                                "Cannot cast {:?} to integer",
                                value
                            ))),
                        }
                    }
                    DataType::Float64 | DataType::Float32 => match value {
                        PlanValue::Integer(i) => Ok(PlanValue::Float(i as f64)),
                        PlanValue::Float(f) => Ok(PlanValue::Float(f)),
                        PlanValue::String(s) => {
                            s.parse::<f64>().map(PlanValue::Float).map_err(|_| {
                                Error::InvalidArgumentError(format!("Cannot cast '{}' to float", s))
                            })
                        }
                        _ => Err(Error::InvalidArgumentError(format!(
                            "Cannot cast {:?} to float",
                            value
                        ))),
                    },
                    DataType::Utf8 | DataType::LargeUtf8 => match value {
                        PlanValue::String(s) => Ok(PlanValue::String(s)),
                        PlanValue::Integer(i) => Ok(PlanValue::String(i.to_string())),
                        PlanValue::Float(f) => Ok(PlanValue::String(f.to_string())),
                        PlanValue::Interval(_) => Err(Error::InvalidArgumentError(
                            "Cannot cast interval to string in aggregate expressions".into(),
                        )),
                        _ => Err(Error::InvalidArgumentError(format!(
                            "Cannot cast {:?} to string",
                            value
                        ))),
                    },
                    DataType::Interval(IntervalUnit::MonthDayNano) => match value {
                        PlanValue::Interval(interval) => Ok(PlanValue::Interval(interval)),
                        _ => Err(Error::InvalidArgumentError(format!(
                            "Cannot cast {:?} to interval",
                            value
                        ))),
                    },
                    DataType::Date32 => match value {
                        PlanValue::Date32(days) => Ok(PlanValue::Date32(days)),
                        PlanValue::String(text) => {
                            let days = parse_date32_literal(&text)?;
                            Ok(PlanValue::Date32(days))
                        }
                        _ => Err(Error::InvalidArgumentError(format!(
                            "Cannot cast {:?} to date",
                            value
                        ))),
                    },
                    _ => Err(Error::InvalidArgumentError(format!(
                        "CAST to {:?} not supported in aggregate expressions",
                        data_type
                    ))),
                }
            }
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                // Evaluate the operand if present (for simple CASE)
                let operand_value = if let Some(op) = operand {
                    Some(Self::evaluate_expr_with_plan_value_aggregates_and_row(
                        op,
                        aggregates,
                        row_batch,
                        column_lookup,
                        row_idx,
                    )?)
                } else {
                    None
                };

                // Evaluate each WHEN/THEN branch
                for (when_expr, then_expr) in branches {
                    let matches = if let Some(ref op_val) = operand_value {
                        // Simple CASE compares using SQL equality semantics: NULL never matches.
                        let when_val = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                            when_expr,
                            aggregates,
                            row_batch,
                            column_lookup,
                            row_idx,
                        )?;
                        Self::simple_case_branch_matches(op_val, &when_val)
                    } else {
                        // Searched CASE: evaluate WHEN as boolean condition
                        let when_val = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                            when_expr,
                            aggregates,
                            row_batch,
                            column_lookup,
                            row_idx,
                        )?;
                        // Treat non-zero as true
                        match when_val {
                            PlanValue::Integer(i) => i != 0,
                            PlanValue::Float(f) => f != 0.0,
                            PlanValue::Null => false,
                            _ => false,
                        }
                    };

                    if matches {
                        return Self::evaluate_expr_with_plan_value_aggregates_and_row(
                            then_expr,
                            aggregates,
                            row_batch,
                            column_lookup,
                            row_idx,
                        );
                    }
                }

                // No branch matched, evaluate ELSE or return NULL
                if let Some(else_e) = else_expr {
                    Self::evaluate_expr_with_plan_value_aggregates_and_row(
                        else_e,
                        aggregates,
                        row_batch,
                        column_lookup,
                        row_idx,
                    )
                } else {
                    Ok(PlanValue::Null)
                }
            }
            ScalarExpr::Coalesce(exprs) => {
                // Return the first non-NULL value
                for expr in exprs {
                    let value = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                        expr,
                        aggregates,
                        row_batch,
                        column_lookup,
                        row_idx,
                    )?;
                    if !matches!(value, PlanValue::Null) {
                        return Ok(value);
                    }
                }
                Ok(PlanValue::Null)
            }
            ScalarExpr::Random => Ok(PlanValue::Float(rand::random::<f64>())),
            ScalarExpr::GetField { .. } => Err(Error::InvalidArgumentError(
                "GetField not supported in aggregate expressions".into(),
            )),
            ScalarExpr::ScalarSubquery(_) => Err(Error::InvalidArgumentError(
                "Scalar subqueries not supported in aggregate expressions".into(),
            )),
        }
    }

    fn simple_case_branch_matches(operand: &PlanValue, candidate: &PlanValue) -> bool {
        if matches!(operand, PlanValue::Null) || matches!(candidate, PlanValue::Null) {
            return false;
        }

        match (operand, candidate) {
            (PlanValue::Integer(left), PlanValue::Integer(right)) => left == right,
            (PlanValue::Integer(left), PlanValue::Float(right)) => (*left as f64) == *right,
            (PlanValue::Float(left), PlanValue::Integer(right)) => *left == (*right as f64),
            (PlanValue::Float(left), PlanValue::Float(right)) => left == right,
            (PlanValue::String(left), PlanValue::String(right)) => left == right,
            (PlanValue::Struct(left), PlanValue::Struct(right)) => left == right,
            (PlanValue::Interval(left), PlanValue::Interval(right)) => {
                compare_interval_values(*left, *right) == std::cmp::Ordering::Equal
            }
            _ => operand == candidate,
        }
    }

    fn evaluate_expr_with_aggregates(
        expr: &ScalarExpr<String>,
        aggregates: &FxHashMap<String, AggregateValue>,
    ) -> ExecutorResult<Option<i64>> {
        use llkv_expr::expr::BinaryOp;
        use llkv_expr::literal::Literal;

        match expr {
            ScalarExpr::Literal(Literal::Int128(v)) => Ok(Some(*v as i64)),
            ScalarExpr::Literal(Literal::Float64(v)) => Ok(Some(*v as i64)),
            ScalarExpr::Literal(Literal::Decimal128(value)) => {
                if let Some(int) = decimal_exact_i64(*value) {
                    Ok(Some(int))
                } else {
                    Ok(Some(value.to_f64() as i64))
                }
            }
            ScalarExpr::Literal(Literal::Boolean(v)) => Ok(Some(if *v { 1 } else { 0 })),
            ScalarExpr::Literal(Literal::String(_)) => Err(Error::InvalidArgumentError(
                "String literals not supported in aggregate expressions".into(),
            )),
            ScalarExpr::Literal(Literal::Date32(days)) => Ok(Some(*days as i64)),
            ScalarExpr::Literal(Literal::Null) => Ok(None),
            ScalarExpr::Literal(Literal::Struct(_)) => Err(Error::InvalidArgumentError(
                "Struct literals not supported in aggregate expressions".into(),
            )),
            ScalarExpr::Literal(Literal::Interval(_)) => Err(Error::InvalidArgumentError(
                "Interval literals not supported in aggregate-only expressions".into(),
            )),
            ScalarExpr::Column(_) => Err(Error::InvalidArgumentError(
                "Column references not supported in aggregate-only expressions".into(),
            )),
            ScalarExpr::Compare { .. } => Err(Error::InvalidArgumentError(
                "Comparisons not supported in aggregate-only expressions".into(),
            )),
            ScalarExpr::Aggregate(agg) => {
                let key = format!("{:?}", agg);
                let value = aggregates.get(&key).ok_or_else(|| {
                    Error::Internal(format!("Aggregate value not found for key: {}", key))
                })?;
                Ok(value.as_i64())
            }
            ScalarExpr::Not(inner) => {
                let value = Self::evaluate_expr_with_aggregates(inner, aggregates)?;
                Ok(value.map(|v| if v != 0 { 0 } else { 1 }))
            }
            ScalarExpr::IsNull { expr, negated } => {
                let value = Self::evaluate_expr_with_aggregates(expr, aggregates)?;
                let is_null = value.is_none();
                Ok(Some(if is_null != *negated { 1 } else { 0 }))
            }
            ScalarExpr::Binary { left, op, right } => {
                let left_val = Self::evaluate_expr_with_aggregates(left, aggregates)?;
                let right_val = Self::evaluate_expr_with_aggregates(right, aggregates)?;

                match op {
                    BinaryOp::Add
                    | BinaryOp::Subtract
                    | BinaryOp::Multiply
                    | BinaryOp::Divide
                    | BinaryOp::Modulo => match (left_val, right_val) {
                        (Some(lhs), Some(rhs)) => {
                            let result = match op {
                                BinaryOp::Add => lhs.checked_add(rhs),
                                BinaryOp::Subtract => lhs.checked_sub(rhs),
                                BinaryOp::Multiply => lhs.checked_mul(rhs),
                                BinaryOp::Divide => {
                                    if rhs == 0 {
                                        return Ok(None);
                                    }
                                    lhs.checked_div(rhs)
                                }
                                BinaryOp::Modulo => {
                                    if rhs == 0 {
                                        return Ok(None);
                                    }
                                    lhs.checked_rem(rhs)
                                }
                                BinaryOp::And
                                | BinaryOp::Or
                                | BinaryOp::BitwiseShiftLeft
                                | BinaryOp::BitwiseShiftRight => unreachable!(),
                            };

                            result.map(Some).ok_or_else(|| {
                                Error::InvalidArgumentError(
                                    "Arithmetic overflow in expression".into(),
                                )
                            })
                        }
                        _ => Ok(None),
                    },
                    BinaryOp::And => Ok(evaluate_option_logical_and(left_val, right_val)),
                    BinaryOp::Or => Ok(evaluate_option_logical_or(left_val, right_val)),
                    BinaryOp::BitwiseShiftLeft | BinaryOp::BitwiseShiftRight => {
                        match (left_val, right_val) {
                            (Some(lhs), Some(rhs)) => {
                                let result = match op {
                                    BinaryOp::BitwiseShiftLeft => {
                                        Some(lhs.wrapping_shl(rhs as u32))
                                    }
                                    BinaryOp::BitwiseShiftRight => {
                                        Some(lhs.wrapping_shr(rhs as u32))
                                    }
                                    _ => unreachable!(),
                                };
                                Ok(result)
                            }
                            _ => Ok(None),
                        }
                    }
                }
            }
            ScalarExpr::Cast { expr, data_type } => {
                let value = Self::evaluate_expr_with_aggregates(expr, aggregates)?;
                match value {
                    Some(v) => Self::cast_aggregate_value(v, data_type).map(Some),
                    None => Ok(None),
                }
            }
            ScalarExpr::GetField { .. } => Err(Error::InvalidArgumentError(
                "GetField not supported in aggregate-only expressions".into(),
            )),
            ScalarExpr::Case { .. } => Err(Error::InvalidArgumentError(
                "CASE not supported in aggregate-only expressions".into(),
            )),
            ScalarExpr::Coalesce(_) => Err(Error::InvalidArgumentError(
                "COALESCE not supported in aggregate-only expressions".into(),
            )),
            ScalarExpr::Random => Ok(Some((rand::random::<f64>() * (i64::MAX as f64)) as i64)),
            ScalarExpr::ScalarSubquery(_) => Err(Error::InvalidArgumentError(
                "Scalar subqueries not supported in aggregate-only expressions".into(),
            )),
        }
    }

    fn cast_aggregate_value(value: i64, data_type: &DataType) -> ExecutorResult<i64> {
        fn ensure_range(value: i64, min: i64, max: i64, ty: &DataType) -> ExecutorResult<i64> {
            if value < min || value > max {
                return Err(Error::InvalidArgumentError(format!(
                    "value {} out of range for CAST target {:?}",
                    value, ty
                )));
            }
            Ok(value)
        }

        match data_type {
            DataType::Int8 => ensure_range(value, i8::MIN as i64, i8::MAX as i64, data_type),
            DataType::Int16 => ensure_range(value, i16::MIN as i64, i16::MAX as i64, data_type),
            DataType::Int32 => ensure_range(value, i32::MIN as i64, i32::MAX as i64, data_type),
            DataType::Int64 => Ok(value),
            DataType::UInt8 => ensure_range(value, 0, u8::MAX as i64, data_type),
            DataType::UInt16 => ensure_range(value, 0, u16::MAX as i64, data_type),
            DataType::UInt32 => ensure_range(value, 0, u32::MAX as i64, data_type),
            DataType::UInt64 => {
                if value < 0 {
                    return Err(Error::InvalidArgumentError(format!(
                        "value {} out of range for CAST target {:?}",
                        value, data_type
                    )));
                }
                Ok(value)
            }
            DataType::Float32 | DataType::Float64 => Ok(value),
            DataType::Boolean => Ok(if value == 0 { 0 } else { 1 }),
            DataType::Null => Err(Error::InvalidArgumentError(
                "CAST to NULL is not supported in aggregate-only expressions".into(),
            )),
            _ => Err(Error::InvalidArgumentError(format!(
                "CAST to {:?} is not supported in aggregate-only expressions",
                data_type
            ))),
        }
    }
}

struct CrossProductExpressionContext {
    schema: Arc<ExecutorSchema>,
    field_id_to_index: FxHashMap<FieldId, usize>,
    numeric_cache: FxHashMap<FieldId, ArrayRef>,
    column_cache: FxHashMap<FieldId, ColumnAccessor>,
    scalar_subquery_columns: FxHashMap<SubqueryId, ColumnAccessor>,
    next_field_id: FieldId,
}

#[derive(Clone)]
enum ColumnAccessor {
    Int64(Arc<Int64Array>),
    Float64(Arc<Float64Array>),
    Boolean(Arc<BooleanArray>),
    Utf8(Arc<StringArray>),
    Date32(Arc<Date32Array>),
    Interval(Arc<IntervalMonthDayNanoArray>),
    Decimal128 {
        array: Arc<Decimal128Array>,
        scale: i8,
    },
    Null(usize),
}

impl ColumnAccessor {
    fn from_array(array: &ArrayRef) -> ExecutorResult<Self> {
        match array.data_type() {
            DataType::Int64 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| Error::Internal("expected Int64 array".into()))?
                    .clone();
                Ok(Self::Int64(Arc::new(typed)))
            }
            DataType::Float64 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| Error::Internal("expected Float64 array".into()))?
                    .clone();
                Ok(Self::Float64(Arc::new(typed)))
            }
            DataType::Boolean => {
                let typed = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| Error::Internal("expected Boolean array".into()))?
                    .clone();
                Ok(Self::Boolean(Arc::new(typed)))
            }
            DataType::Utf8 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| Error::Internal("expected Utf8 array".into()))?
                    .clone();
                Ok(Self::Utf8(Arc::new(typed)))
            }
            DataType::Date32 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .ok_or_else(|| Error::Internal("expected Date32 array".into()))?
                    .clone();
                Ok(Self::Date32(Arc::new(typed)))
            }
            DataType::Interval(IntervalUnit::MonthDayNano) => {
                let typed = array
                    .as_any()
                    .downcast_ref::<IntervalMonthDayNanoArray>()
                    .ok_or_else(|| Error::Internal("expected IntervalMonthDayNano array".into()))?
                    .clone();
                Ok(Self::Interval(Arc::new(typed)))
            }
            DataType::Decimal128(_, scale) => {
                let typed = array
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| Error::Internal("expected Decimal128 array".into()))?
                    .clone();
                Ok(Self::Decimal128 {
                    array: Arc::new(typed),
                    scale: *scale,
                })
            }
            DataType::Null => Ok(Self::Null(array.len())),
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported column type {:?} in cross product filter",
                other
            ))),
        }
    }

    fn from_numeric_array(numeric: &ArrayRef) -> ExecutorResult<Self> {
        let casted = cast(numeric, &DataType::Float64)?;
        let float_array = casted
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("cast to Float64 failed")
            .clone();
        Ok(Self::Float64(Arc::new(float_array)))
    }

    fn len(&self) -> usize {
        match self {
            ColumnAccessor::Int64(array) => array.len(),
            ColumnAccessor::Float64(array) => array.len(),
            ColumnAccessor::Boolean(array) => array.len(),
            ColumnAccessor::Utf8(array) => array.len(),
            ColumnAccessor::Date32(array) => array.len(),
            ColumnAccessor::Interval(array) => array.len(),
            ColumnAccessor::Decimal128 { array, .. } => array.len(),
            ColumnAccessor::Null(len) => *len,
        }
    }

    fn is_null(&self, idx: usize) -> bool {
        match self {
            ColumnAccessor::Int64(array) => array.is_null(idx),
            ColumnAccessor::Float64(array) => array.is_null(idx),
            ColumnAccessor::Boolean(array) => array.is_null(idx),
            ColumnAccessor::Utf8(array) => array.is_null(idx),
            ColumnAccessor::Date32(array) => array.is_null(idx),
            ColumnAccessor::Interval(array) => array.is_null(idx),
            ColumnAccessor::Decimal128 { array, .. } => array.is_null(idx),
            ColumnAccessor::Null(_) => true,
        }
    }

    fn literal_at(&self, idx: usize) -> ExecutorResult<Literal> {
        if self.is_null(idx) {
            return Ok(Literal::Null);
        }
        match self {
            ColumnAccessor::Int64(array) => Ok(Literal::Int128(array.value(idx) as i128)),
            ColumnAccessor::Float64(array) => Ok(Literal::Float64(array.value(idx))),
            ColumnAccessor::Boolean(array) => Ok(Literal::Boolean(array.value(idx))),
            ColumnAccessor::Utf8(array) => Ok(Literal::String(array.value(idx).to_string())),
            ColumnAccessor::Date32(array) => Ok(Literal::Date32(array.value(idx))),
            ColumnAccessor::Interval(array) => Ok(Literal::Interval(interval_value_from_arrow(
                array.value(idx),
            ))),
            ColumnAccessor::Decimal128 { array, .. } => Ok(Literal::Int128(array.value(idx))),
            ColumnAccessor::Null(_) => Ok(Literal::Null),
        }
    }

    fn as_array_ref(&self) -> ArrayRef {
        match self {
            ColumnAccessor::Int64(array) => Arc::clone(array) as ArrayRef,
            ColumnAccessor::Float64(array) => Arc::clone(array) as ArrayRef,
            ColumnAccessor::Boolean(array) => Arc::clone(array) as ArrayRef,
            ColumnAccessor::Utf8(array) => Arc::clone(array) as ArrayRef,
            ColumnAccessor::Date32(array) => Arc::clone(array) as ArrayRef,
            ColumnAccessor::Interval(array) => Arc::clone(array) as ArrayRef,
            ColumnAccessor::Decimal128 { array, .. } => Arc::clone(array) as ArrayRef,
            ColumnAccessor::Null(len) => new_null_array(&DataType::Null, *len),
        }
    }
}

#[derive(Clone)]
enum ValueArray {
    Numeric(ArrayRef),
    Boolean(Arc<BooleanArray>),
    Utf8(Arc<StringArray>),
    Interval(Arc<IntervalMonthDayNanoArray>),
    Null(usize),
}

impl ValueArray {
    fn from_array(array: ArrayRef) -> ExecutorResult<Self> {
        match array.data_type() {
            DataType::Boolean => {
                let typed = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| Error::Internal("expected Boolean array".into()))?
                    .clone();
                Ok(Self::Boolean(Arc::new(typed)))
            }
            DataType::Utf8 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| Error::Internal("expected Utf8 array".into()))?
                    .clone();
                Ok(Self::Utf8(Arc::new(typed)))
            }
            DataType::Interval(IntervalUnit::MonthDayNano) => {
                let typed = array
                    .as_any()
                    .downcast_ref::<IntervalMonthDayNanoArray>()
                    .ok_or_else(|| Error::Internal("expected IntervalMonthDayNano array".into()))?
                    .clone();
                Ok(Self::Interval(Arc::new(typed)))
            }
            DataType::Null => Ok(Self::Null(array.len())),
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Date32
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(_, _) => Ok(Self::Numeric(array)),
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported data type {:?} in cross product expression",
                other
            ))),
        }
    }

    fn len(&self) -> usize {
        match self {
            ValueArray::Numeric(array) => array.len(),
            ValueArray::Boolean(array) => array.len(),
            ValueArray::Utf8(array) => array.len(),
            ValueArray::Interval(array) => array.len(),
            ValueArray::Null(len) => *len,
        }
    }

    fn as_array_ref(&self) -> ArrayRef {
        match self {
            ValueArray::Numeric(arr) => arr.clone(),
            ValueArray::Boolean(arr) => arr.clone() as ArrayRef,
            ValueArray::Utf8(arr) => arr.clone() as ArrayRef,
            ValueArray::Interval(arr) => arr.clone() as ArrayRef,
            ValueArray::Null(len) => new_null_array(&DataType::Null, *len),
        }
    }
}

fn truth_and(lhs: Option<bool>, rhs: Option<bool>) -> Option<bool> {
    match (lhs, rhs) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        (Some(true), None) | (None, Some(true)) | (None, None) => None,
    }
}

fn truth_or(lhs: Option<bool>, rhs: Option<bool>) -> Option<bool> {
    match (lhs, rhs) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        (Some(false), None) | (None, Some(false)) | (None, None) => None,
    }
}

fn truth_not(value: Option<bool>) -> Option<bool> {
    match value {
        Some(true) => Some(false),
        Some(false) => Some(true),
        None => None,
    }
}

fn literal_to_constant_array(literal: &Literal, len: usize) -> ExecutorResult<ArrayRef> {
    match literal {
        Literal::Int128(v) => {
            let value = i64::try_from(*v).unwrap_or(0);
            let values = vec![value; len];
            Ok(Arc::new(Int64Array::from(values)) as ArrayRef)
        }
        Literal::Float64(v) => {
            let values = vec![*v; len];
            Ok(Arc::new(Float64Array::from(values)) as ArrayRef)
        }
        Literal::Boolean(v) => {
            let values = vec![Some(*v); len];
            Ok(Arc::new(BooleanArray::from(values)) as ArrayRef)
        }
        Literal::String(v) => {
            let values: Vec<Option<String>> = (0..len).map(|_| Some(v.clone())).collect();
            Ok(Arc::new(StringArray::from(values)) as ArrayRef)
        }
        Literal::Date32(days) => {
            let values = vec![*days; len];
            Ok(Arc::new(Date32Array::from(values)) as ArrayRef)
        }
        Literal::Decimal128(value) => {
            let iter = std::iter::repeat_n(value.raw_value(), len);
            let array = Decimal128Array::from_iter_values(iter)
                .with_precision_and_scale(value.precision(), value.scale())
                .map_err(|err| {
                    Error::InvalidArgumentError(format!(
                        "failed to synthesize decimal literal array: {err}"
                    ))
                })?;
            Ok(Arc::new(array) as ArrayRef)
        }
        Literal::Interval(interval) => {
            let value = interval_value_to_arrow(*interval);
            let values = vec![value; len];
            Ok(Arc::new(IntervalMonthDayNanoArray::from(values)) as ArrayRef)
        }
        Literal::Null => Ok(new_null_array(&DataType::Null, len)),
        Literal::Struct(_) => Err(Error::InvalidArgumentError(
            "struct literals are not supported in cross product filters".into(),
        )),
    }
}

fn literals_to_array(values: &[Literal]) -> ExecutorResult<ArrayRef> {
    #[derive(Copy, Clone, Eq, PartialEq)]
    enum LiteralArrayKind {
        Null,
        Integer,
        Float,
        Boolean,
        String,
        Date32,
        Interval,
        Decimal,
    }

    if values.is_empty() {
        return Ok(new_null_array(&DataType::Null, 0));
    }

    let mut has_integer = false;
    let mut has_float = false;
    let mut has_decimal = false;
    let mut has_boolean = false;
    let mut has_string = false;
    let mut has_date = false;
    let mut has_interval = false;

    for literal in values {
        match literal {
            Literal::Null => {}
            Literal::Int128(_) => {
                has_integer = true;
            }
            Literal::Float64(_) => {
                has_float = true;
            }
            Literal::Decimal128(_) => {
                has_decimal = true;
            }
            Literal::Boolean(_) => {
                has_boolean = true;
            }
            Literal::String(_) => {
                has_string = true;
            }
            Literal::Date32(_) => {
                has_date = true;
            }
            Literal::Interval(_) => {
                has_interval = true;
            }
            Literal::Struct(_) => {
                return Err(Error::InvalidArgumentError(
                    "struct scalar subquery results are not supported".into(),
                ));
            }
        }
    }

    let mixed_numeric = has_integer as u8 + has_float as u8 + has_decimal as u8;
    if has_string && (has_boolean || has_date || has_interval || mixed_numeric > 0)
        || has_boolean && (has_date || has_interval || mixed_numeric > 0)
        || has_date && (has_interval || mixed_numeric > 0)
        || has_interval && (mixed_numeric > 0)
    {
        return Err(Error::InvalidArgumentError(
            "mixed scalar subquery result types are not supported".into(),
        ));
    }

    let target_kind = if has_string {
        LiteralArrayKind::String
    } else if has_interval {
        LiteralArrayKind::Interval
    } else if has_date {
        LiteralArrayKind::Date32
    } else if has_boolean {
        LiteralArrayKind::Boolean
    } else if has_float {
        LiteralArrayKind::Float
    } else if has_decimal {
        LiteralArrayKind::Decimal
    } else if has_integer {
        LiteralArrayKind::Integer
    } else {
        LiteralArrayKind::Null
    };

    match target_kind {
        LiteralArrayKind::Null => Ok(new_null_array(&DataType::Null, values.len())),
        LiteralArrayKind::Integer => {
            let mut coerced: Vec<Option<i64>> = Vec::with_capacity(values.len());
            for literal in values {
                match literal {
                    Literal::Null => coerced.push(None),
                    Literal::Int128(value) => {
                        let v = i64::try_from(*value).map_err(|_| {
                            Error::InvalidArgumentError(
                                "scalar subquery integer result exceeds supported range".into(),
                            )
                        })?;
                        coerced.push(Some(v));
                    }
                    _ => unreachable!("non-integer value encountered in integer array"),
                }
            }
            let array = Int64Array::from_iter(coerced);
            Ok(Arc::new(array) as ArrayRef)
        }
        LiteralArrayKind::Float => {
            let mut coerced: Vec<Option<f64>> = Vec::with_capacity(values.len());
            for literal in values {
                match literal {
                    Literal::Null => coerced.push(None),
                    Literal::Int128(_) | Literal::Float64(_) | Literal::Decimal128(_) => {
                        let value = literal_to_f64(literal).ok_or_else(|| {
                            Error::InvalidArgumentError(
                                "failed to coerce scalar subquery value to FLOAT".into(),
                            )
                        })?;
                        coerced.push(Some(value));
                    }
                    _ => unreachable!("non-numeric value encountered in float array"),
                }
            }
            let array = Float64Array::from_iter(coerced);
            Ok(Arc::new(array) as ArrayRef)
        }
        LiteralArrayKind::Boolean => {
            let iter = values.iter().map(|literal| match literal {
                Literal::Null => None,
                Literal::Boolean(flag) => Some(*flag),
                _ => unreachable!("non-boolean value encountered in boolean array"),
            });
            let array = BooleanArray::from_iter(iter);
            Ok(Arc::new(array) as ArrayRef)
        }
        LiteralArrayKind::String => {
            let iter = values.iter().map(|literal| match literal {
                Literal::Null => None,
                Literal::String(value) => Some(value.clone()),
                _ => unreachable!("non-string value encountered in string array"),
            });
            let array = StringArray::from_iter(iter);
            Ok(Arc::new(array) as ArrayRef)
        }
        LiteralArrayKind::Date32 => {
            let iter = values.iter().map(|literal| match literal {
                Literal::Null => None,
                Literal::Date32(days) => Some(*days),
                _ => unreachable!("non-date value encountered in date array"),
            });
            let array = Date32Array::from_iter(iter);
            Ok(Arc::new(array) as ArrayRef)
        }
        LiteralArrayKind::Interval => {
            let iter = values.iter().map(|literal| match literal {
                Literal::Null => None,
                Literal::Interval(interval) => Some(interval_value_to_arrow(*interval)),
                _ => unreachable!("non-interval value encountered in interval array"),
            });
            let array = IntervalMonthDayNanoArray::from_iter(iter);
            Ok(Arc::new(array) as ArrayRef)
        }
        LiteralArrayKind::Decimal => {
            let mut target_scale: Option<i8> = None;
            for literal in values {
                if let Literal::Decimal128(value) = literal {
                    target_scale = Some(match target_scale {
                        Some(scale) => scale.max(value.scale()),
                        None => value.scale(),
                    });
                }
            }
            let target_scale = target_scale.expect("decimal literal expected");

            let mut max_precision: u8 = 1;
            let mut aligned: Vec<Option<DecimalValue>> = Vec::with_capacity(values.len());
            for literal in values {
                match literal {
                    Literal::Null => aligned.push(None),
                    Literal::Decimal128(value) => {
                        let adjusted = if value.scale() != target_scale {
                            llkv_compute::scalar::decimal::rescale(*value, target_scale).map_err(
                                |err| {
                                    Error::InvalidArgumentError(format!(
                                        "failed to align decimal scale: {err}"
                                    ))
                                },
                            )?
                        } else {
                            *value
                        };
                        max_precision = max_precision.max(adjusted.precision());
                        aligned.push(Some(adjusted));
                    }
                    Literal::Int128(value) => {
                        let decimal = DecimalValue::new(*value, 0).map_err(|err| {
                            Error::InvalidArgumentError(format!(
                                "failed to build decimal from integer: {err}"
                            ))
                        })?;
                        let decimal = llkv_compute::scalar::decimal::rescale(decimal, target_scale)
                            .map_err(|err| {
                                Error::InvalidArgumentError(format!(
                                    "failed to align integer decimal scale: {err}"
                                ))
                            })?;
                        max_precision = max_precision.max(decimal.precision());
                        aligned.push(Some(decimal));
                    }
                    _ => unreachable!("unexpected literal in decimal array"),
                }
            }

            let mut builder = Decimal128Builder::new()
                .with_precision_and_scale(max_precision, target_scale)
                .map_err(|err| {
                    Error::InvalidArgumentError(format!(
                        "invalid Decimal128 precision/scale: {err}"
                    ))
                })?;
            for value in aligned {
                match value {
                    Some(decimal) => builder.append_value(decimal.raw_value()),
                    None => builder.append_null(),
                }
            }
            let array = builder.finish();
            Ok(Arc::new(array) as ArrayRef)
        }
    }
}

impl CrossProductExpressionContext {
    fn new(schema: &Schema, lookup: FxHashMap<String, usize>) -> ExecutorResult<Self> {
        let mut columns = Vec::with_capacity(schema.fields().len());
        let mut field_id_to_index = FxHashMap::default();
        let mut next_field_id: FieldId = 1;

        for (idx, field) in schema.fields().iter().enumerate() {
            if next_field_id == u32::MAX {
                return Err(Error::Internal(
                    "cross product projection exhausted FieldId space".into(),
                ));
            }

            let executor_column = ExecutorColumn {
                name: field.name().clone(),
                data_type: field.data_type().clone(),
                nullable: field.is_nullable(),
                primary_key: false,
                unique: false,
                field_id: next_field_id,
                check_expr: None,
            };
            let field_id = next_field_id;
            next_field_id = next_field_id.saturating_add(1);

            columns.push(executor_column);
            field_id_to_index.insert(field_id, idx);
        }

        Ok(Self {
            schema: Arc::new(ExecutorSchema { columns, lookup }),
            field_id_to_index,
            numeric_cache: FxHashMap::default(),
            column_cache: FxHashMap::default(),
            scalar_subquery_columns: FxHashMap::default(),
            next_field_id,
        })
    }

    fn schema(&self) -> &ExecutorSchema {
        self.schema.as_ref()
    }

    fn field_id_for_column(&self, name: &str) -> Option<FieldId> {
        self.schema.resolve(name).map(|column| column.field_id)
    }

    fn reset(&mut self) {
        self.numeric_cache.clear();
        self.column_cache.clear();
        self.scalar_subquery_columns.clear();
    }

    fn allocate_synthetic_field_id(&mut self) -> ExecutorResult<FieldId> {
        if self.next_field_id == FieldId::MAX {
            return Err(Error::Internal(
                "cross product projection exhausted FieldId space".into(),
            ));
        }
        let field_id = self.next_field_id;
        self.next_field_id = self.next_field_id.saturating_add(1);
        Ok(field_id)
    }

    fn register_scalar_subquery_column(
        &mut self,
        subquery_id: SubqueryId,
        accessor: ColumnAccessor,
    ) {
        self.scalar_subquery_columns.insert(subquery_id, accessor);
    }

    #[cfg(test)]
    fn evaluate(
        &mut self,
        expr: &ScalarExpr<String>,
        batch: &RecordBatch,
    ) -> ExecutorResult<ArrayRef> {
        let translated = translate_scalar(expr, self.schema.as_ref(), |name| {
            Error::InvalidArgumentError(format!(
                "column '{}' not found in cross product result",
                name
            ))
        })?;

        self.evaluate_numeric(&translated, batch)
    }

    fn evaluate_predicate_mask(
        &mut self,
        expr: &LlkvExpr<'static, FieldId>,
        batch: &RecordBatch,
        mut exists_eval: impl FnMut(
            &mut Self,
            &llkv_expr::SubqueryExpr,
            usize,
            &RecordBatch,
        ) -> ExecutorResult<Option<bool>>,
    ) -> ExecutorResult<BooleanArray> {
        let truths = self.evaluate_predicate_truths(expr, batch, &mut exists_eval)?;
        let mut builder = BooleanBuilder::with_capacity(truths.len());
        for value in truths {
            builder.append_value(value.unwrap_or(false));
        }
        Ok(builder.finish())
    }

    fn evaluate_predicate_truths(
        &mut self,
        expr: &LlkvExpr<'static, FieldId>,
        batch: &RecordBatch,
        exists_eval: &mut impl FnMut(
            &mut Self,
            &llkv_expr::SubqueryExpr,
            usize,
            &RecordBatch,
        ) -> ExecutorResult<Option<bool>>,
    ) -> ExecutorResult<Vec<Option<bool>>> {
        match expr {
            LlkvExpr::Literal(value) => Ok(vec![Some(*value); batch.num_rows()]),
            LlkvExpr::And(children) => {
                if children.is_empty() {
                    return Ok(vec![Some(true); batch.num_rows()]);
                }
                let mut result =
                    self.evaluate_predicate_truths(&children[0], batch, exists_eval)?;
                for child in &children[1..] {
                    let next = self.evaluate_predicate_truths(child, batch, exists_eval)?;
                    for (lhs, rhs) in result.iter_mut().zip(next.into_iter()) {
                        *lhs = truth_and(*lhs, rhs);
                    }
                }
                Ok(result)
            }
            LlkvExpr::Or(children) => {
                if children.is_empty() {
                    return Ok(vec![Some(false); batch.num_rows()]);
                }
                let mut result =
                    self.evaluate_predicate_truths(&children[0], batch, exists_eval)?;
                for child in &children[1..] {
                    let next = self.evaluate_predicate_truths(child, batch, exists_eval)?;
                    for (lhs, rhs) in result.iter_mut().zip(next.into_iter()) {
                        *lhs = truth_or(*lhs, rhs);
                    }
                }
                Ok(result)
            }
            LlkvExpr::Not(inner) => {
                let mut values = self.evaluate_predicate_truths(inner, batch, exists_eval)?;
                for value in &mut values {
                    *value = truth_not(*value);
                }
                Ok(values)
            }
            LlkvExpr::Pred(filter) => self.evaluate_filter_truths(filter, batch),
            LlkvExpr::Compare { left, op, right } => {
                self.evaluate_compare_truths(left, *op, right, batch)
            }
            LlkvExpr::InList {
                expr: target,
                list,
                negated,
            } => self.evaluate_in_list_truths(target, list, *negated, batch),
            LlkvExpr::IsNull { expr, negated } => {
                self.evaluate_is_null_truths(expr, *negated, batch)
            }
            LlkvExpr::Exists(subquery_expr) => {
                let mut values = Vec::with_capacity(batch.num_rows());
                for row_idx in 0..batch.num_rows() {
                    let value = exists_eval(self, subquery_expr, row_idx, batch)?;
                    values.push(value);
                }
                Ok(values)
            }
        }
    }

    fn evaluate_filter_truths(
        &mut self,
        filter: &Filter<FieldId>,
        batch: &RecordBatch,
    ) -> ExecutorResult<Vec<Option<bool>>> {
        let accessor = self.column_accessor(filter.field_id, batch)?;
        let len = accessor.len();

        match &filter.op {
            Operator::IsNull => {
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    out.push(Some(accessor.is_null(idx)));
                }
                Ok(out)
            }
            Operator::IsNotNull => {
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    out.push(Some(!accessor.is_null(idx)));
                }
                Ok(out)
            }
            _ => match accessor {
                ColumnAccessor::Int64(array) => {
                    let predicate = build_fixed_width_predicate::<Int64Type>(&filter.op)
                        .map_err(Error::predicate_build)?;
                    let mut out = Vec::with_capacity(len);
                    for idx in 0..len {
                        if array.is_null(idx) {
                            out.push(None);
                        } else {
                            let value = array.value(idx);
                            out.push(Some(predicate.matches(&value)));
                        }
                    }
                    Ok(out)
                }
                ColumnAccessor::Float64(array) => {
                    let predicate = build_fixed_width_predicate::<Float64Type>(&filter.op)
                        .map_err(Error::predicate_build)?;
                    let mut out = Vec::with_capacity(len);
                    for idx in 0..len {
                        if array.is_null(idx) {
                            out.push(None);
                        } else {
                            let value = array.value(idx);
                            out.push(Some(predicate.matches(&value)));
                        }
                    }
                    Ok(out)
                }
                ColumnAccessor::Boolean(array) => {
                    let predicate =
                        build_bool_predicate(&filter.op).map_err(Error::predicate_build)?;
                    let mut out = Vec::with_capacity(len);
                    for idx in 0..len {
                        if array.is_null(idx) {
                            out.push(None);
                        } else {
                            let value = array.value(idx);
                            out.push(Some(predicate.matches(&value)));
                        }
                    }
                    Ok(out)
                }
                ColumnAccessor::Utf8(array) => {
                    let predicate =
                        build_var_width_predicate(&filter.op).map_err(Error::predicate_build)?;
                    let mut out = Vec::with_capacity(len);
                    for idx in 0..len {
                        if array.is_null(idx) {
                            out.push(None);
                        } else {
                            let value = array.value(idx);
                            out.push(Some(predicate.matches(value)));
                        }
                    }
                    Ok(out)
                }
                ColumnAccessor::Date32(array) => {
                    let predicate = build_fixed_width_predicate::<Int32Type>(&filter.op)
                        .map_err(Error::predicate_build)?;
                    let mut out = Vec::with_capacity(len);
                    for idx in 0..len {
                        if array.is_null(idx) {
                            out.push(None);
                        } else {
                            let value = array.value(idx);
                            out.push(Some(predicate.matches(&value)));
                        }
                    }
                    Ok(out)
                }
                ColumnAccessor::Interval(array) => {
                    let array = array.as_ref();
                    let mut out = Vec::with_capacity(len);
                    for idx in 0..len {
                        if array.is_null(idx) {
                            out.push(None);
                            continue;
                        }
                        let literal =
                            Literal::Interval(interval_value_from_arrow(array.value(idx)));
                        let matches = evaluate_filter_against_literal(&literal, &filter.op)?;
                        out.push(Some(matches));
                    }
                    Ok(out)
                }
                ColumnAccessor::Decimal128 { array, scale } => {
                    // For decimal comparisons, we need to handle them specially
                    // Convert decimals to float for comparison purposes
                    let scale_factor = 10_f64.powi(scale as i32);
                    let mut out = Vec::with_capacity(len);
                    for idx in 0..len {
                        if array.is_null(idx) {
                            out.push(None);
                            continue;
                        }
                        let raw_value = array.value(idx);
                        let decimal_value = raw_value as f64 / scale_factor;
                        let literal = Literal::Float64(decimal_value);
                        let matches = evaluate_filter_against_literal(&literal, &filter.op)?;
                        out.push(Some(matches));
                    }
                    Ok(out)
                }
                ColumnAccessor::Null(len) => Ok(vec![None; len]),
            },
        }
    }

    fn evaluate_compare_truths(
        &mut self,
        left: &ScalarExpr<FieldId>,
        op: CompareOp,
        right: &ScalarExpr<FieldId>,
        batch: &RecordBatch,
    ) -> ExecutorResult<Vec<Option<bool>>> {
        let left_values = self.materialize_value_array(left, batch)?;
        let right_values = self.materialize_value_array(right, batch)?;

        if left_values.len() != right_values.len() {
            return Err(Error::Internal(
                "mismatched compare operand lengths in cross product filter".into(),
            ));
        }

        let len = left_values.len();

        if matches!(left_values, ValueArray::Null(_)) || matches!(right_values, ValueArray::Null(_))
        {
            return Ok(vec![None; len]);
        }

        let lhs_arr = left_values.as_array_ref();
        let rhs_arr = right_values.as_array_ref();

        let result_array = llkv_compute::kernels::compute_compare(&lhs_arr, op, &rhs_arr)?;
        let bool_array = result_array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("compute_compare must return BooleanArray");

        let out: Vec<Option<bool>> = bool_array.iter().collect();
        Ok(out)
    }

    fn evaluate_is_null_truths(
        &mut self,
        expr: &ScalarExpr<FieldId>,
        negated: bool,
        batch: &RecordBatch,
    ) -> ExecutorResult<Vec<Option<bool>>> {
        let values = self.materialize_value_array(expr, batch)?;
        let len = values.len();

        if let ValueArray::Null(len) = values {
            let result = if negated { Some(false) } else { Some(true) };
            return Ok(vec![result; len]);
        }

        let arr = values.as_array_ref();
        let mut out = Vec::with_capacity(len);
        for idx in 0..len {
            let is_null = arr.is_null(idx);
            let result = if negated { !is_null } else { is_null };
            out.push(Some(result));
        }
        Ok(out)
    }

    fn evaluate_in_list_truths(
        &mut self,
        target: &ScalarExpr<FieldId>,
        list: &[ScalarExpr<FieldId>],
        negated: bool,
        batch: &RecordBatch,
    ) -> ExecutorResult<Vec<Option<bool>>> {
        let target_values = self.materialize_value_array(target, batch)?;
        let list_values = list
            .iter()
            .map(|expr| self.materialize_value_array(expr, batch))
            .collect::<ExecutorResult<Vec<_>>>()?;

        let len = target_values.len();
        for values in &list_values {
            if values.len() != len {
                return Err(Error::Internal(
                    "mismatched IN list operand lengths in cross product filter".into(),
                ));
            }
        }

        if matches!(target_values, ValueArray::Null(_)) {
            return Ok(vec![None; len]);
        }

        let target_arr = target_values.as_array_ref();
        let mut combined_result: Option<BooleanArray> = None;

        for candidate in &list_values {
            if matches!(candidate, ValueArray::Null(_)) {
                let nulls = new_null_array(&DataType::Boolean, len);
                let bool_nulls = nulls
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .unwrap()
                    .clone();

                match combined_result {
                    None => combined_result = Some(bool_nulls),
                    Some(prev) => {
                        combined_result = Some(or_kleene(&prev, &bool_nulls)?);
                    }
                }
                continue;
            }

            let candidate_arr = candidate.as_array_ref();

            let cmp =
                llkv_compute::kernels::compute_compare(&target_arr, CompareOp::Eq, &candidate_arr)?;
            let bool_cmp = cmp
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("compute_compare returns BooleanArray")
                .clone();

            match combined_result {
                None => combined_result = Some(bool_cmp),
                Some(prev) => {
                    combined_result = Some(or_kleene(&prev, &bool_cmp)?);
                }
            }
        }

        let final_bool = combined_result.unwrap_or_else(|| {
            let mut builder = BooleanBuilder::new();
            for _ in 0..len {
                builder.append_value(false);
            }
            builder.finish()
        });

        let final_bool = if negated {
            not(&final_bool)?
        } else {
            final_bool
        };

        let out: Vec<Option<bool>> = final_bool.iter().collect();
        Ok(out)
    }

    fn evaluate_numeric(
        &mut self,
        expr: &ScalarExpr<FieldId>,
        batch: &RecordBatch,
    ) -> ExecutorResult<ArrayRef> {
        let mut required = FxHashSet::default();
        collect_field_ids(expr, &mut required);

        let mut arrays = NumericArrayMap::default();
        for field_id in required {
            let numeric = self.numeric_array(field_id, batch)?;
            arrays.insert(field_id, numeric);
        }

        NumericKernels::evaluate_batch(expr, batch.num_rows(), &arrays)
    }

    fn numeric_array(
        &mut self,
        field_id: FieldId,
        batch: &RecordBatch,
    ) -> ExecutorResult<ArrayRef> {
        if let Some(existing) = self.numeric_cache.get(&field_id) {
            return Ok(existing.clone());
        }

        let column_index = *self.field_id_to_index.get(&field_id).ok_or_else(|| {
            Error::Internal("field mapping missing during cross product evaluation".into())
        })?;

        let array_ref = batch.column(column_index).clone();
        self.numeric_cache.insert(field_id, array_ref.clone());
        Ok(array_ref)
    }

    fn column_accessor(
        &mut self,
        field_id: FieldId,
        batch: &RecordBatch,
    ) -> ExecutorResult<ColumnAccessor> {
        if let Some(existing) = self.column_cache.get(&field_id) {
            return Ok(existing.clone());
        }

        let column_index = *self.field_id_to_index.get(&field_id).ok_or_else(|| {
            Error::Internal("field mapping missing during cross product evaluation".into())
        })?;

        let accessor = ColumnAccessor::from_array(batch.column(column_index))?;
        self.column_cache.insert(field_id, accessor.clone());
        Ok(accessor)
    }

    fn materialize_scalar_array(
        &mut self,
        expr: &ScalarExpr<FieldId>,
        batch: &RecordBatch,
    ) -> ExecutorResult<ArrayRef> {
        match expr {
            ScalarExpr::Column(field_id) => {
                let accessor = self.column_accessor(*field_id, batch)?;
                Ok(accessor.as_array_ref())
            }
            ScalarExpr::Literal(literal) => literal_to_constant_array(literal, batch.num_rows()),
            ScalarExpr::Binary { .. } => self.evaluate_numeric(expr, batch),
            ScalarExpr::Compare { .. } => self.evaluate_numeric(expr, batch),
            ScalarExpr::Not(_) => self.evaluate_numeric(expr, batch),
            ScalarExpr::IsNull { .. } => self.evaluate_numeric(expr, batch),
            ScalarExpr::Aggregate(_) => Err(Error::InvalidArgumentError(
                "aggregate expressions are not supported in cross product filters".into(),
            )),
            ScalarExpr::GetField { .. } => Err(Error::InvalidArgumentError(
                "struct field access is not supported in cross product filters".into(),
            )),
            ScalarExpr::Cast { expr, data_type } => {
                let source = self.materialize_scalar_array(expr.as_ref(), batch)?;
                let casted = cast(source.as_ref(), data_type).map_err(|err| {
                    Error::InvalidArgumentError(format!("failed to cast expression: {err}"))
                })?;
                Ok(casted)
            }
            ScalarExpr::Case { .. } => self.evaluate_numeric(expr, batch),
            ScalarExpr::Coalesce(_) => self.evaluate_numeric(expr, batch),
            ScalarExpr::Random => self.evaluate_numeric(expr, batch),
            ScalarExpr::ScalarSubquery(subquery) => {
                let accessor = self
                    .scalar_subquery_columns
                    .get(&subquery.id)
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "scalar subqueries are not supported in cross product filters".into(),
                        )
                    })?
                    .clone();
                Ok(accessor.as_array_ref())
            }
        }
    }

    fn materialize_value_array(
        &mut self,
        expr: &ScalarExpr<FieldId>,
        batch: &RecordBatch,
    ) -> ExecutorResult<ValueArray> {
        let array = self.materialize_scalar_array(expr, batch)?;
        ValueArray::from_array(array)
    }
}

// TODO: Move to llkv-aggregate?
fn collect_field_ids(expr: &ScalarExpr<FieldId>, out: &mut FxHashSet<FieldId>) {
    match expr {
        ScalarExpr::Column(fid) => {
            out.insert(*fid);
        }
        ScalarExpr::Binary { left, right, .. } => {
            collect_field_ids(left, out);
            collect_field_ids(right, out);
        }
        ScalarExpr::Compare { left, right, .. } => {
            collect_field_ids(left, out);
            collect_field_ids(right, out);
        }
        ScalarExpr::Aggregate(call) => match call {
            AggregateCall::CountStar => {}
            AggregateCall::Count { expr, .. }
            | AggregateCall::Sum { expr, .. }
            | AggregateCall::Total { expr, .. }
            | AggregateCall::Avg { expr, .. }
            | AggregateCall::Min(expr)
            | AggregateCall::Max(expr)
            | AggregateCall::CountNulls(expr)
            | AggregateCall::GroupConcat { expr, .. } => {
                collect_field_ids(expr, out);
            }
        },
        ScalarExpr::GetField { base, .. } => collect_field_ids(base, out),
        ScalarExpr::Cast { expr, .. } => collect_field_ids(expr, out),
        ScalarExpr::Not(expr) => collect_field_ids(expr, out),
        ScalarExpr::IsNull { expr, .. } => collect_field_ids(expr, out),
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            if let Some(inner) = operand.as_deref() {
                collect_field_ids(inner, out);
            }
            for (when_expr, then_expr) in branches {
                collect_field_ids(when_expr, out);
                collect_field_ids(then_expr, out);
            }
            if let Some(inner) = else_expr.as_deref() {
                collect_field_ids(inner, out);
            }
        }
        ScalarExpr::Coalesce(items) => {
            for item in items {
                collect_field_ids(item, out);
            }
        }
        ScalarExpr::Literal(_) | ScalarExpr::Random => {}
        ScalarExpr::ScalarSubquery(_) => {}
    }
}

fn strip_exists(expr: &LlkvExpr<'static, FieldId>) -> LlkvExpr<'static, FieldId> {
    match expr {
        LlkvExpr::And(children) => LlkvExpr::And(children.iter().map(strip_exists).collect()),
        LlkvExpr::Or(children) => LlkvExpr::Or(children.iter().map(strip_exists).collect()),
        LlkvExpr::Not(inner) => LlkvExpr::Not(Box::new(strip_exists(inner))),
        LlkvExpr::Pred(filter) => LlkvExpr::Pred(filter.clone()),
        LlkvExpr::Compare { left, op, right } => LlkvExpr::Compare {
            left: left.clone(),
            op: *op,
            right: right.clone(),
        },
        LlkvExpr::InList {
            expr,
            list,
            negated,
        } => LlkvExpr::InList {
            expr: expr.clone(),
            list: list.clone(),
            negated: *negated,
        },
        LlkvExpr::IsNull { expr, negated } => LlkvExpr::IsNull {
            expr: expr.clone(),
            negated: *negated,
        },
        LlkvExpr::Literal(value) => LlkvExpr::Literal(*value),
        LlkvExpr::Exists(_) => LlkvExpr::Literal(true),
    }
}

fn rewrite_predicate_scalar_subqueries(
    expr: LlkvExpr<'static, FieldId>,
    literals: &FxHashMap<SubqueryId, Literal>,
) -> ExecutorResult<LlkvExpr<'static, FieldId>> {
    match expr {
        LlkvExpr::And(children) => {
            let rewritten: ExecutorResult<Vec<_>> = children
                .into_iter()
                .map(|child| rewrite_predicate_scalar_subqueries(child, literals))
                .collect();
            Ok(LlkvExpr::And(rewritten?))
        }
        LlkvExpr::Or(children) => {
            let rewritten: ExecutorResult<Vec<_>> = children
                .into_iter()
                .map(|child| rewrite_predicate_scalar_subqueries(child, literals))
                .collect();
            Ok(LlkvExpr::Or(rewritten?))
        }
        LlkvExpr::Not(inner) => Ok(LlkvExpr::Not(Box::new(
            rewrite_predicate_scalar_subqueries(*inner, literals)?,
        ))),
        LlkvExpr::Pred(filter) => Ok(LlkvExpr::Pred(filter)),
        LlkvExpr::Compare { left, op, right } => Ok(LlkvExpr::Compare {
            left: rewrite_scalar_expr_subqueries(left, literals)?,
            op,
            right: rewrite_scalar_expr_subqueries(right, literals)?,
        }),
        LlkvExpr::InList {
            expr,
            list,
            negated,
        } => Ok(LlkvExpr::InList {
            expr: rewrite_scalar_expr_subqueries(expr, literals)?,
            list: list
                .into_iter()
                .map(|item| rewrite_scalar_expr_subqueries(item, literals))
                .collect::<ExecutorResult<_>>()?,
            negated,
        }),
        LlkvExpr::IsNull { expr, negated } => Ok(LlkvExpr::IsNull {
            expr: rewrite_scalar_expr_subqueries(expr, literals)?,
            negated,
        }),
        LlkvExpr::Literal(value) => Ok(LlkvExpr::Literal(value)),
        LlkvExpr::Exists(subquery) => Ok(LlkvExpr::Exists(subquery)),
    }
}

fn rewrite_scalar_expr_subqueries(
    expr: ScalarExpr<FieldId>,
    literals: &FxHashMap<SubqueryId, Literal>,
) -> ExecutorResult<ScalarExpr<FieldId>> {
    match expr {
        ScalarExpr::ScalarSubquery(subquery) => {
            let literal = literals.get(&subquery.id).ok_or_else(|| {
                Error::Internal(format!(
                    "missing literal for scalar subquery {:?}",
                    subquery.id
                ))
            })?;
            Ok(ScalarExpr::Literal(literal.clone()))
        }
        ScalarExpr::Column(fid) => Ok(ScalarExpr::Column(fid)),
        ScalarExpr::Literal(lit) => Ok(ScalarExpr::Literal(lit)),
        ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
            left: Box::new(rewrite_scalar_expr_subqueries(*left, literals)?),
            op,
            right: Box::new(rewrite_scalar_expr_subqueries(*right, literals)?),
        }),
        ScalarExpr::Compare { left, op, right } => Ok(ScalarExpr::Compare {
            left: Box::new(rewrite_scalar_expr_subqueries(*left, literals)?),
            op,
            right: Box::new(rewrite_scalar_expr_subqueries(*right, literals)?),
        }),
        ScalarExpr::Not(inner) => Ok(ScalarExpr::Not(Box::new(rewrite_scalar_expr_subqueries(
            *inner, literals,
        )?))),
        ScalarExpr::IsNull { expr, negated } => Ok(ScalarExpr::IsNull {
            expr: Box::new(rewrite_scalar_expr_subqueries(*expr, literals)?),
            negated,
        }),
        ScalarExpr::Aggregate(agg) => Ok(ScalarExpr::Aggregate(agg)),
        ScalarExpr::GetField { base, field_name } => Ok(ScalarExpr::GetField {
            base: Box::new(rewrite_scalar_expr_subqueries(*base, literals)?),
            field_name,
        }),
        ScalarExpr::Cast { expr, data_type } => Ok(ScalarExpr::Cast {
            expr: Box::new(rewrite_scalar_expr_subqueries(*expr, literals)?),
            data_type,
        }),
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => Ok(ScalarExpr::Case {
            operand: operand
                .map(|e| rewrite_scalar_expr_subqueries(*e, literals))
                .transpose()?
                .map(Box::new),
            branches: branches
                .into_iter()
                .map(|(when, then)| {
                    Ok((
                        rewrite_scalar_expr_subqueries(when, literals)?,
                        rewrite_scalar_expr_subqueries(then, literals)?,
                    ))
                })
                .collect::<ExecutorResult<_>>()?,
            else_expr: else_expr
                .map(|e| rewrite_scalar_expr_subqueries(*e, literals))
                .transpose()?
                .map(Box::new),
        }),
        ScalarExpr::Coalesce(items) => Ok(ScalarExpr::Coalesce(
            items
                .into_iter()
                .map(|item| rewrite_scalar_expr_subqueries(item, literals))
                .collect::<ExecutorResult<_>>()?,
        )),
        ScalarExpr::Random => Ok(ScalarExpr::Random),
    }
}

fn bind_select_plan(
    plan: &SelectPlan,
    bindings: &FxHashMap<String, Literal>,
) -> ExecutorResult<SelectPlan> {
    if bindings.is_empty() {
        return Ok(plan.clone());
    }

    let projections = plan
        .projections
        .iter()
        .map(|projection| bind_projection(projection, bindings))
        .collect::<ExecutorResult<Vec<_>>>()?;

    let filter = match &plan.filter {
        Some(wrapper) => Some(bind_select_filter(wrapper, bindings)?),
        None => None,
    };

    let aggregates = plan
        .aggregates
        .iter()
        .map(|aggregate| bind_aggregate_expr(aggregate, bindings))
        .collect::<ExecutorResult<Vec<_>>>()?;

    let scalar_subqueries = plan
        .scalar_subqueries
        .iter()
        .map(|subquery| bind_scalar_subquery(subquery, bindings))
        .collect::<ExecutorResult<Vec<_>>>()?;

    if let Some(compound) = &plan.compound {
        let bound_compound = bind_compound_select(compound, bindings)?;
        return Ok(SelectPlan {
            tables: Vec::new(),
            joins: Vec::new(),
            projections: Vec::new(),
            filter: None,
            having: None,
            aggregates: Vec::new(),
            order_by: plan.order_by.clone(),
            distinct: false,
            scalar_subqueries: Vec::new(),
            compound: Some(bound_compound),
            group_by: Vec::new(),
            value_table_mode: None,
            limit: plan.limit,
            offset: plan.offset,
        });
    }

    Ok(SelectPlan {
        tables: plan.tables.clone(),
        joins: plan.joins.clone(),
        projections,
        filter,
        having: plan.having.clone(),
        aggregates,
        order_by: Vec::new(),
        distinct: plan.distinct,
        scalar_subqueries,
        compound: None,
        group_by: plan.group_by.clone(),
        value_table_mode: plan.value_table_mode.clone(),
        limit: plan.limit,
        offset: plan.offset,
    })
}

fn bind_compound_select(
    compound: &CompoundSelectPlan,
    bindings: &FxHashMap<String, Literal>,
) -> ExecutorResult<CompoundSelectPlan> {
    let initial = bind_select_plan(&compound.initial, bindings)?;
    let mut operations = Vec::with_capacity(compound.operations.len());
    for component in &compound.operations {
        let bound_plan = bind_select_plan(&component.plan, bindings)?;
        operations.push(CompoundSelectComponent {
            operator: component.operator.clone(),
            quantifier: component.quantifier.clone(),
            plan: bound_plan,
        });
    }
    Ok(CompoundSelectPlan {
        initial: Box::new(initial),
        operations,
    })
}

fn ensure_schema_compatibility(base: &Schema, other: &Schema) -> ExecutorResult<()> {
    if base.fields().len() != other.fields().len() {
        return Err(Error::InvalidArgumentError(
            "compound SELECT requires matching column counts".into(),
        ));
    }
    for (left, right) in base.fields().iter().zip(other.fields().iter()) {
        if left.data_type() != right.data_type() {
            return Err(Error::InvalidArgumentError(format!(
                "compound SELECT column type mismatch: {} vs {}",
                left.data_type(),
                right.data_type()
            )));
        }
    }
    Ok(())
}

fn ensure_distinct_rows(rows: &mut Vec<Vec<PlanValue>>, cache: &mut Option<FxHashSet<Vec<u8>>>) {
    if cache.is_some() {
        return;
    }
    let mut set = FxHashSet::default();
    let mut deduped: Vec<Vec<PlanValue>> = Vec::with_capacity(rows.len());
    for row in rows.drain(..) {
        let key = encode_row(&row);
        if set.insert(key) {
            deduped.push(row);
        }
    }
    *rows = deduped;
    *cache = Some(set);
}

fn encode_row(row: &[PlanValue]) -> Vec<u8> {
    let mut buf = Vec::new();
    for value in row {
        encode_plan_value(&mut buf, value);
        buf.push(0x1F);
    }
    buf
}

fn encode_plan_value(buf: &mut Vec<u8>, value: &PlanValue) {
    match value {
        PlanValue::Null => buf.push(0),
        PlanValue::Integer(v) => {
            buf.push(1);
            buf.extend_from_slice(&v.to_be_bytes());
        }
        PlanValue::Float(v) => {
            buf.push(2);
            buf.extend_from_slice(&v.to_bits().to_be_bytes());
        }
        PlanValue::Decimal(decimal) => {
            buf.push(7);
            buf.extend_from_slice(&decimal.raw_value().to_be_bytes());
            buf.push(decimal.scale().to_be_bytes()[0]);
        }
        PlanValue::String(s) => {
            buf.push(3);
            let bytes = s.as_bytes();
            let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
            buf.extend_from_slice(&len.to_be_bytes());
            buf.extend_from_slice(bytes);
        }
        PlanValue::Date32(days) => {
            buf.push(5);
            buf.extend_from_slice(&days.to_be_bytes());
        }
        PlanValue::Struct(map) => {
            buf.push(4);
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let len = u32::try_from(entries.len()).unwrap_or(u32::MAX);
            buf.extend_from_slice(&len.to_be_bytes());
            for (key, val) in entries {
                let key_bytes = key.as_bytes();
                let key_len = u32::try_from(key_bytes.len()).unwrap_or(u32::MAX);
                buf.extend_from_slice(&key_len.to_be_bytes());
                buf.extend_from_slice(key_bytes);
                encode_plan_value(buf, val);
            }
        }
        PlanValue::Interval(interval) => {
            buf.push(6);
            buf.extend_from_slice(&interval.months.to_be_bytes());
            buf.extend_from_slice(&interval.days.to_be_bytes());
            buf.extend_from_slice(&interval.nanos.to_be_bytes());
        }
    }
}

fn rows_to_record_batch(
    schema: Arc<Schema>,
    rows: &[Vec<PlanValue>],
) -> ExecutorResult<RecordBatch> {
    let column_count = schema.fields().len();
    let mut columns: Vec<Vec<PlanValue>> = vec![Vec::with_capacity(rows.len()); column_count];
    for row in rows {
        if row.len() != column_count {
            return Err(Error::InvalidArgumentError(
                "compound SELECT produced mismatched column counts".into(),
            ));
        }
        for (idx, value) in row.iter().enumerate() {
            columns[idx].push(value.clone());
        }
    }

    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(column_count);
    for (idx, field) in schema.fields().iter().enumerate() {
        let array = build_array_for_column(field.data_type(), &columns[idx])?;
        arrays.push(array);
    }

    RecordBatch::try_new(schema, arrays).map_err(|err| {
        Error::InvalidArgumentError(format!("failed to materialize compound SELECT: {err}"))
    })
}

fn build_column_lookup_map(schema: &Schema) -> FxHashMap<String, usize> {
    let mut lookup = FxHashMap::default();
    for (idx, field) in schema.fields().iter().enumerate() {
        lookup.insert(field.name().to_ascii_lowercase(), idx);
    }
    lookup
}

fn build_group_key(
    batch: &RecordBatch,
    row_idx: usize,
    key_indices: &[usize],
) -> ExecutorResult<Vec<GroupKeyValue>> {
    let mut values = Vec::with_capacity(key_indices.len());
    for &index in key_indices {
        values.push(group_key_value(batch.column(index), row_idx)?);
    }
    Ok(values)
}

fn group_key_value(array: &ArrayRef, row_idx: usize) -> ExecutorResult<GroupKeyValue> {
    if !array.is_valid(row_idx) {
        return Ok(GroupKeyValue::Null);
    }

    match array.data_type() {
        DataType::Int8 => {
            let values = array
                .as_any()
                .downcast_ref::<Int8Array>()
                .ok_or_else(|| Error::Internal("failed to downcast to Int8Array".into()))?;
            Ok(GroupKeyValue::Int(values.value(row_idx) as i64))
        }
        DataType::Int16 => {
            let values = array
                .as_any()
                .downcast_ref::<Int16Array>()
                .ok_or_else(|| Error::Internal("failed to downcast to Int16Array".into()))?;
            Ok(GroupKeyValue::Int(values.value(row_idx) as i64))
        }
        DataType::Int32 => {
            let values = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| Error::Internal("failed to downcast to Int32Array".into()))?;
            Ok(GroupKeyValue::Int(values.value(row_idx) as i64))
        }
        DataType::Int64 => {
            let values = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| Error::Internal("failed to downcast to Int64Array".into()))?;
            Ok(GroupKeyValue::Int(values.value(row_idx)))
        }
        DataType::UInt8 => {
            let values = array
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| Error::Internal("failed to downcast to UInt8Array".into()))?;
            Ok(GroupKeyValue::Int(values.value(row_idx) as i64))
        }
        DataType::UInt16 => {
            let values = array
                .as_any()
                .downcast_ref::<UInt16Array>()
                .ok_or_else(|| Error::Internal("failed to downcast to UInt16Array".into()))?;
            Ok(GroupKeyValue::Int(values.value(row_idx) as i64))
        }
        DataType::UInt32 => {
            let values = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| Error::Internal("failed to downcast to UInt32Array".into()))?;
            Ok(GroupKeyValue::Int(values.value(row_idx) as i64))
        }
        DataType::UInt64 => {
            let values = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("failed to downcast to UInt64Array".into()))?;
            let value = values.value(row_idx);
            if value > i64::MAX as u64 {
                return Err(Error::InvalidArgumentError(
                    "GROUP BY value exceeds supported integer range".into(),
                ));
            }
            Ok(GroupKeyValue::Int(value as i64))
        }
        DataType::Date32 => {
            let values = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| Error::Internal("failed to downcast to Date32Array".into()))?;
            Ok(GroupKeyValue::Int(values.value(row_idx) as i64))
        }
        DataType::Boolean => {
            let values = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| Error::Internal("failed to downcast to BooleanArray".into()))?;
            Ok(GroupKeyValue::Bool(values.value(row_idx)))
        }
        DataType::Utf8 => {
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::Internal("failed to downcast to StringArray".into()))?;
            Ok(GroupKeyValue::String(values.value(row_idx).to_string()))
        }
        other => Err(Error::InvalidArgumentError(format!(
            "GROUP BY does not support column type {:?}",
            other
        ))),
    }
}

fn evaluate_constant_predicate(expr: &LlkvExpr<'static, String>) -> Option<Option<bool>> {
    match expr {
        LlkvExpr::Literal(value) => Some(Some(*value)),
        LlkvExpr::Not(inner) => {
            let inner_val = evaluate_constant_predicate(inner)?;
            Some(truth_not(inner_val))
        }
        LlkvExpr::And(children) => {
            let mut acc = Some(true);
            for child in children {
                let child_val = evaluate_constant_predicate(child)?;
                acc = truth_and(acc, child_val);
            }
            Some(acc)
        }
        LlkvExpr::Or(children) => {
            let mut acc = Some(false);
            for child in children {
                let child_val = evaluate_constant_predicate(child)?;
                acc = truth_or(acc, child_val);
            }
            Some(acc)
        }
        LlkvExpr::Compare { left, op, right } => {
            let left_literal = evaluate_constant_scalar(left)?;
            let right_literal = evaluate_constant_scalar(right)?;
            Some(compare_literals(*op, &left_literal, &right_literal))
        }
        LlkvExpr::IsNull { expr, negated } => {
            let literal = evaluate_constant_scalar(expr)?;
            let is_null = matches!(literal, Literal::Null);
            Some(Some(if *negated { !is_null } else { is_null }))
        }
        LlkvExpr::InList {
            expr,
            list,
            negated,
        } => {
            let needle = evaluate_constant_scalar(expr)?;
            let mut saw_unknown = false;

            for candidate in list {
                let value = evaluate_constant_scalar(candidate)?;
                match compare_literals(CompareOp::Eq, &needle, &value) {
                    Some(true) => {
                        return Some(Some(!*negated));
                    }
                    Some(false) => {}
                    None => saw_unknown = true,
                }
            }

            if saw_unknown {
                Some(None)
            } else {
                Some(Some(*negated))
            }
        }
        _ => None,
    }
}

enum ConstantJoinEvaluation {
    Known(bool),
    Unknown,
    NotConstant,
}

fn evaluate_constant_join_expr(expr: &LlkvExpr<'static, String>) -> ConstantJoinEvaluation {
    match expr {
        LlkvExpr::Literal(value) => ConstantJoinEvaluation::Known(*value),
        LlkvExpr::And(children) => {
            let mut saw_unknown = false;
            for child in children {
                match evaluate_constant_join_expr(child) {
                    ConstantJoinEvaluation::Known(false) => {
                        return ConstantJoinEvaluation::Known(false);
                    }
                    ConstantJoinEvaluation::Known(true) => {}
                    ConstantJoinEvaluation::Unknown => saw_unknown = true,
                    ConstantJoinEvaluation::NotConstant => {
                        return ConstantJoinEvaluation::NotConstant;
                    }
                }
            }
            if saw_unknown {
                ConstantJoinEvaluation::Unknown
            } else {
                ConstantJoinEvaluation::Known(true)
            }
        }
        LlkvExpr::Or(children) => {
            let mut saw_unknown = false;
            for child in children {
                match evaluate_constant_join_expr(child) {
                    ConstantJoinEvaluation::Known(true) => {
                        return ConstantJoinEvaluation::Known(true);
                    }
                    ConstantJoinEvaluation::Known(false) => {}
                    ConstantJoinEvaluation::Unknown => saw_unknown = true,
                    ConstantJoinEvaluation::NotConstant => {
                        return ConstantJoinEvaluation::NotConstant;
                    }
                }
            }
            if saw_unknown {
                ConstantJoinEvaluation::Unknown
            } else {
                ConstantJoinEvaluation::Known(false)
            }
        }
        LlkvExpr::Not(inner) => match evaluate_constant_join_expr(inner) {
            ConstantJoinEvaluation::Known(value) => ConstantJoinEvaluation::Known(!value),
            ConstantJoinEvaluation::Unknown => ConstantJoinEvaluation::Unknown,
            ConstantJoinEvaluation::NotConstant => ConstantJoinEvaluation::NotConstant,
        },
        LlkvExpr::Compare { left, op, right } => {
            let left_lit = evaluate_constant_scalar(left);
            let right_lit = evaluate_constant_scalar(right);

            if matches!(left_lit, Some(Literal::Null)) || matches!(right_lit, Some(Literal::Null)) {
                // Per SQL three-valued logic, any comparison involving NULL yields UNKNOWN.
                return ConstantJoinEvaluation::Unknown;
            }

            let (Some(left_lit), Some(right_lit)) = (left_lit, right_lit) else {
                return ConstantJoinEvaluation::NotConstant;
            };

            match compare_literals(*op, &left_lit, &right_lit) {
                Some(result) => ConstantJoinEvaluation::Known(result),
                None => ConstantJoinEvaluation::Unknown,
            }
        }
        LlkvExpr::IsNull { expr, negated } => match evaluate_constant_scalar(expr) {
            Some(literal) => {
                let is_null = matches!(literal, Literal::Null);
                let value = if *negated { !is_null } else { is_null };
                ConstantJoinEvaluation::Known(value)
            }
            None => ConstantJoinEvaluation::NotConstant,
        },
        LlkvExpr::InList {
            expr,
            list,
            negated,
        } => {
            let needle = match evaluate_constant_scalar(expr) {
                Some(literal) => literal,
                None => return ConstantJoinEvaluation::NotConstant,
            };

            if matches!(needle, Literal::Null) {
                return ConstantJoinEvaluation::Unknown;
            }

            let mut saw_unknown = false;
            for candidate in list {
                let value = match evaluate_constant_scalar(candidate) {
                    Some(literal) => literal,
                    None => return ConstantJoinEvaluation::NotConstant,
                };

                match compare_literals(CompareOp::Eq, &needle, &value) {
                    Some(true) => {
                        let result = !*negated;
                        return ConstantJoinEvaluation::Known(result);
                    }
                    Some(false) => {}
                    None => saw_unknown = true,
                }
            }

            if saw_unknown {
                ConstantJoinEvaluation::Unknown
            } else {
                let result = *negated;
                ConstantJoinEvaluation::Known(result)
            }
        }
        _ => ConstantJoinEvaluation::NotConstant,
    }
}

enum NullComparisonBehavior {
    ThreeValuedLogic,
}

fn evaluate_constant_scalar(expr: &ScalarExpr<String>) -> Option<Literal> {
    evaluate_constant_scalar_internal(expr, false)
}

fn evaluate_constant_scalar_with_aggregates(expr: &ScalarExpr<String>) -> Option<Literal> {
    evaluate_constant_scalar_internal(expr, true)
}

fn evaluate_constant_scalar_internal(
    expr: &ScalarExpr<String>,
    allow_aggregates: bool,
) -> Option<Literal> {
    match expr {
        ScalarExpr::Literal(lit) => Some(lit.clone()),
        ScalarExpr::Binary { left, op, right } => {
            let left_value = evaluate_constant_scalar_internal(left, allow_aggregates)?;
            let right_value = evaluate_constant_scalar_internal(right, allow_aggregates)?;
            evaluate_binary_literal(*op, &left_value, &right_value)
        }
        ScalarExpr::Cast { expr, data_type } => {
            let value = evaluate_constant_scalar_internal(expr, allow_aggregates)?;
            cast_literal_to_type(&value, data_type)
        }
        ScalarExpr::Not(inner) => {
            let value = evaluate_constant_scalar_internal(inner, allow_aggregates)?;
            match literal_truthiness(&value) {
                Some(true) => Some(Literal::Int128(0)),
                Some(false) => Some(Literal::Int128(1)),
                None => Some(Literal::Null),
            }
        }
        ScalarExpr::IsNull { expr, negated } => {
            let value = evaluate_constant_scalar_internal(expr, allow_aggregates)?;
            let is_null = matches!(value, Literal::Null);
            Some(Literal::Boolean(if *negated { !is_null } else { is_null }))
        }
        ScalarExpr::Coalesce(items) => {
            let mut saw_null = false;
            for item in items {
                match evaluate_constant_scalar_internal(item, allow_aggregates) {
                    Some(Literal::Null) => saw_null = true,
                    Some(value) => return Some(value),
                    None => return None,
                }
            }
            if saw_null { Some(Literal::Null) } else { None }
        }
        ScalarExpr::Compare { left, op, right } => {
            let left_value = evaluate_constant_scalar_internal(left, allow_aggregates)?;
            let right_value = evaluate_constant_scalar_internal(right, allow_aggregates)?;
            match compare_literals(*op, &left_value, &right_value) {
                Some(flag) => Some(Literal::Boolean(flag)),
                None => Some(Literal::Null),
            }
        }
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            if let Some(operand_expr) = operand {
                let operand_value =
                    evaluate_constant_scalar_internal(operand_expr, allow_aggregates)?;
                for (when_expr, then_expr) in branches {
                    let when_value =
                        evaluate_constant_scalar_internal(when_expr, allow_aggregates)?;
                    if let Some(true) = compare_literals(CompareOp::Eq, &operand_value, &when_value)
                    {
                        return evaluate_constant_scalar_internal(then_expr, allow_aggregates);
                    }
                }
            } else {
                for (condition_expr, result_expr) in branches {
                    let condition_value =
                        evaluate_constant_scalar_internal(condition_expr, allow_aggregates)?;
                    match literal_truthiness(&condition_value) {
                        Some(true) => {
                            return evaluate_constant_scalar_internal(
                                result_expr,
                                allow_aggregates,
                            );
                        }
                        Some(false) => {}
                        None => {}
                    }
                }
            }

            if let Some(else_branch) = else_expr {
                evaluate_constant_scalar_internal(else_branch, allow_aggregates)
            } else {
                Some(Literal::Null)
            }
        }
        ScalarExpr::Column(_) => None,
        ScalarExpr::Aggregate(call) => {
            if allow_aggregates {
                evaluate_constant_aggregate(call, allow_aggregates)
            } else {
                None
            }
        }
        ScalarExpr::GetField { .. } => None,
        ScalarExpr::Random => None, // RANDOM() is non-deterministic so not a constant
        ScalarExpr::ScalarSubquery(_) => None,
    }
}

fn evaluate_constant_aggregate(
    call: &AggregateCall<String>,
    allow_aggregates: bool,
) -> Option<Literal> {
    match call {
        AggregateCall::CountStar => Some(Literal::Int128(1)),
        AggregateCall::Count { expr, .. } => {
            let value = evaluate_constant_scalar_internal(expr, allow_aggregates)?;
            if matches!(value, Literal::Null) {
                Some(Literal::Int128(0))
            } else {
                Some(Literal::Int128(1))
            }
        }
        AggregateCall::Sum { expr, .. } => {
            let value = evaluate_constant_scalar_internal(expr, allow_aggregates)?;
            match value {
                Literal::Null => Some(Literal::Null),
                Literal::Int128(value) => Some(Literal::Int128(value)),
                Literal::Float64(value) => Some(Literal::Float64(value)),
                Literal::Boolean(flag) => Some(Literal::Int128(if flag { 1 } else { 0 })),
                _ => None,
            }
        }
        AggregateCall::Total { expr, .. } => {
            let value = evaluate_constant_scalar_internal(expr, allow_aggregates)?;
            match value {
                Literal::Null => Some(Literal::Int128(0)),
                Literal::Int128(value) => Some(Literal::Int128(value)),
                Literal::Float64(value) => Some(Literal::Float64(value)),
                Literal::Boolean(flag) => Some(Literal::Int128(if flag { 1 } else { 0 })),
                _ => None,
            }
        }
        AggregateCall::Avg { expr, .. } => {
            let value = evaluate_constant_scalar_internal(expr, allow_aggregates)?;
            match value {
                Literal::Null => Some(Literal::Null),
                other => {
                    let numeric = literal_to_f64(&other)?;
                    Some(Literal::Float64(numeric))
                }
            }
        }
        AggregateCall::Min(expr) => {
            let value = evaluate_constant_scalar_internal(expr, allow_aggregates)?;
            match value {
                Literal::Null => Some(Literal::Null),
                other => Some(other),
            }
        }
        AggregateCall::Max(expr) => {
            let value = evaluate_constant_scalar_internal(expr, allow_aggregates)?;
            match value {
                Literal::Null => Some(Literal::Null),
                other => Some(other),
            }
        }
        AggregateCall::CountNulls(expr) => {
            let value = evaluate_constant_scalar_internal(expr, allow_aggregates)?;
            let count = if matches!(value, Literal::Null) { 1 } else { 0 };
            Some(Literal::Int128(count))
        }
        AggregateCall::GroupConcat {
            expr, separator: _, ..
        } => {
            let value = evaluate_constant_scalar_internal(expr, allow_aggregates)?;
            match value {
                Literal::Null => Some(Literal::Null),
                Literal::String(s) => Some(Literal::String(s)),
                Literal::Int128(i) => Some(Literal::String(i.to_string())),
                Literal::Float64(f) => Some(Literal::String(f.to_string())),
                Literal::Boolean(b) => Some(Literal::String(if b { "1" } else { "0" }.to_string())),
                _ => None,
            }
        }
    }
}

fn evaluate_binary_literal(op: BinaryOp, left: &Literal, right: &Literal) -> Option<Literal> {
    match op {
        BinaryOp::And => evaluate_literal_logical_and(left, right),
        BinaryOp::Or => evaluate_literal_logical_or(left, right),
        BinaryOp::Add
        | BinaryOp::Subtract
        | BinaryOp::Multiply
        | BinaryOp::Divide
        | BinaryOp::Modulo => {
            if matches!(left, Literal::Null) || matches!(right, Literal::Null) {
                return Some(Literal::Null);
            }

            match op {
                BinaryOp::Add => add_literals(left, right),
                BinaryOp::Subtract => subtract_literals(left, right),
                BinaryOp::Multiply => multiply_literals(left, right),
                BinaryOp::Divide => divide_literals(left, right),
                BinaryOp::Modulo => modulo_literals(left, right),
                BinaryOp::And
                | BinaryOp::Or
                | BinaryOp::BitwiseShiftLeft
                | BinaryOp::BitwiseShiftRight => unreachable!(),
            }
        }
        BinaryOp::BitwiseShiftLeft | BinaryOp::BitwiseShiftRight => {
            if matches!(left, Literal::Null) || matches!(right, Literal::Null) {
                return Some(Literal::Null);
            }

            // Convert both operands to integers
            let lhs = literal_to_i128(left)?;
            let rhs = literal_to_i128(right)?;

            // SQLite uses wrapping arithmetic for shifts (operate on i64 then extend to i128)
            let result = match op {
                BinaryOp::BitwiseShiftLeft => (lhs as i64).wrapping_shl(rhs as u32) as i128,
                BinaryOp::BitwiseShiftRight => (lhs as i64).wrapping_shr(rhs as u32) as i128,
                _ => unreachable!(),
            };

            Some(Literal::Int128(result))
        }
    }
}

fn evaluate_literal_logical_and(left: &Literal, right: &Literal) -> Option<Literal> {
    let left_truth = literal_truthiness(left);
    if matches!(left_truth, Some(false)) {
        return Some(Literal::Int128(0));
    }

    let right_truth = literal_truthiness(right);
    if matches!(right_truth, Some(false)) {
        return Some(Literal::Int128(0));
    }

    match (left_truth, right_truth) {
        (Some(true), Some(true)) => Some(Literal::Int128(1)),
        (Some(true), None) | (None, Some(true)) | (None, None) => Some(Literal::Null),
        _ => Some(Literal::Null),
    }
}

fn evaluate_literal_logical_or(left: &Literal, right: &Literal) -> Option<Literal> {
    let left_truth = literal_truthiness(left);
    if matches!(left_truth, Some(true)) {
        return Some(Literal::Int128(1));
    }

    let right_truth = literal_truthiness(right);
    if matches!(right_truth, Some(true)) {
        return Some(Literal::Int128(1));
    }

    match (left_truth, right_truth) {
        (Some(false), Some(false)) => Some(Literal::Int128(0)),
        (Some(false), None) | (None, Some(false)) | (None, None) => Some(Literal::Null),
        _ => Some(Literal::Null),
    }
}

fn add_literals(left: &Literal, right: &Literal) -> Option<Literal> {
    match (left, right) {
        (Literal::Int128(lhs), Literal::Int128(rhs)) => {
            Some(Literal::Int128(lhs.saturating_add(*rhs)))
        }
        _ => {
            let lhs = literal_to_f64(left)?;
            let rhs = literal_to_f64(right)?;
            Some(Literal::Float64(lhs + rhs))
        }
    }
}

fn subtract_literals(left: &Literal, right: &Literal) -> Option<Literal> {
    match (left, right) {
        (Literal::Int128(lhs), Literal::Int128(rhs)) => {
            Some(Literal::Int128(lhs.saturating_sub(*rhs)))
        }
        _ => {
            let lhs = literal_to_f64(left)?;
            let rhs = literal_to_f64(right)?;
            Some(Literal::Float64(lhs - rhs))
        }
    }
}

fn multiply_literals(left: &Literal, right: &Literal) -> Option<Literal> {
    match (left, right) {
        (Literal::Int128(lhs), Literal::Int128(rhs)) => {
            Some(Literal::Int128(lhs.saturating_mul(*rhs)))
        }
        _ => {
            let lhs = literal_to_f64(left)?;
            let rhs = literal_to_f64(right)?;
            Some(Literal::Float64(lhs * rhs))
        }
    }
}

fn divide_literals(left: &Literal, right: &Literal) -> Option<Literal> {
    fn literal_to_i128_from_integer_like(literal: &Literal) -> Option<i128> {
        match literal {
            Literal::Int128(value) => Some(*value),
            Literal::Decimal128(value) => llkv_compute::scalar::decimal::rescale(*value, 0)
                .ok()
                .map(|integral| integral.raw_value()),
            Literal::Boolean(value) => Some(if *value { 1 } else { 0 }),
            Literal::Date32(value) => Some(*value as i128),
            _ => None,
        }
    }

    if let (Some(lhs), Some(rhs)) = (
        literal_to_i128_from_integer_like(left),
        literal_to_i128_from_integer_like(right),
    ) {
        if rhs == 0 {
            return Some(Literal::Null);
        }

        if lhs == i128::MIN && rhs == -1 {
            return Some(Literal::Float64((lhs as f64) / (rhs as f64)));
        }

        return Some(Literal::Int128(lhs / rhs));
    }

    let lhs = literal_to_f64(left)?;
    let rhs = literal_to_f64(right)?;
    if rhs == 0.0 {
        return Some(Literal::Null);
    }
    Some(Literal::Float64(lhs / rhs))
}

fn modulo_literals(left: &Literal, right: &Literal) -> Option<Literal> {
    let lhs = literal_to_i128(left)?;
    let rhs = literal_to_i128(right)?;
    if rhs == 0 {
        return Some(Literal::Null);
    }
    Some(Literal::Int128(lhs % rhs))
}

fn literal_to_f64(literal: &Literal) -> Option<f64> {
    match literal {
        Literal::Int128(value) => Some(*value as f64),
        Literal::Float64(value) => Some(*value),
        Literal::Decimal128(value) => Some(value.to_f64()),
        Literal::Boolean(value) => Some(if *value { 1.0 } else { 0.0 }),
        Literal::Date32(value) => Some(*value as f64),
        _ => None,
    }
}

fn literal_to_i128(literal: &Literal) -> Option<i128> {
    match literal {
        Literal::Int128(value) => Some(*value),
        Literal::Float64(value) => Some(*value as i128),
        Literal::Decimal128(value) => llkv_compute::scalar::decimal::rescale(*value, 0)
            .ok()
            .map(|integral| integral.raw_value()),
        Literal::Boolean(value) => Some(if *value { 1 } else { 0 }),
        Literal::Date32(value) => Some(*value as i128),
        _ => None,
    }
}

fn literal_truthiness(literal: &Literal) -> Option<bool> {
    match literal {
        Literal::Boolean(value) => Some(*value),
        Literal::Int128(value) => Some(*value != 0),
        Literal::Float64(value) => Some(*value != 0.0),
        Literal::Decimal128(value) => Some(decimal_truthy(*value)),
        Literal::Date32(value) => Some(*value != 0),
        Literal::Null => None,
        _ => None,
    }
}

fn plan_value_truthiness(value: &PlanValue) -> Option<bool> {
    match value {
        PlanValue::Integer(v) => Some(*v != 0),
        PlanValue::Float(v) => Some(*v != 0.0),
        PlanValue::Decimal(v) => Some(decimal_truthy(*v)),
        PlanValue::Date32(v) => Some(*v != 0),
        PlanValue::Null => None,
        _ => None,
    }
}

fn option_i64_truthiness(value: Option<i64>) -> Option<bool> {
    value.map(|v| v != 0)
}

fn evaluate_plan_value_logical_and(left: PlanValue, right: PlanValue) -> PlanValue {
    let left_truth = plan_value_truthiness(&left);
    if matches!(left_truth, Some(false)) {
        return PlanValue::Integer(0);
    }

    let right_truth = plan_value_truthiness(&right);
    if matches!(right_truth, Some(false)) {
        return PlanValue::Integer(0);
    }

    match (left_truth, right_truth) {
        (Some(true), Some(true)) => PlanValue::Integer(1),
        (Some(true), None) | (None, Some(true)) | (None, None) => PlanValue::Null,
        _ => PlanValue::Null,
    }
}

fn evaluate_plan_value_logical_or(left: PlanValue, right: PlanValue) -> PlanValue {
    let left_truth = plan_value_truthiness(&left);
    if matches!(left_truth, Some(true)) {
        return PlanValue::Integer(1);
    }

    let right_truth = plan_value_truthiness(&right);
    if matches!(right_truth, Some(true)) {
        return PlanValue::Integer(1);
    }

    match (left_truth, right_truth) {
        (Some(false), Some(false)) => PlanValue::Integer(0),
        (Some(false), None) | (None, Some(false)) | (None, None) => PlanValue::Null,
        _ => PlanValue::Null,
    }
}

fn evaluate_option_logical_and(left: Option<i64>, right: Option<i64>) -> Option<i64> {
    let left_truth = option_i64_truthiness(left);
    if matches!(left_truth, Some(false)) {
        return Some(0);
    }

    let right_truth = option_i64_truthiness(right);
    if matches!(right_truth, Some(false)) {
        return Some(0);
    }

    match (left_truth, right_truth) {
        (Some(true), Some(true)) => Some(1),
        (Some(true), None) | (None, Some(true)) | (None, None) => None,
        _ => None,
    }
}

fn evaluate_option_logical_or(left: Option<i64>, right: Option<i64>) -> Option<i64> {
    let left_truth = option_i64_truthiness(left);
    if matches!(left_truth, Some(true)) {
        return Some(1);
    }

    let right_truth = option_i64_truthiness(right);
    if matches!(right_truth, Some(true)) {
        return Some(1);
    }

    match (left_truth, right_truth) {
        (Some(false), Some(false)) => Some(0),
        (Some(false), None) | (None, Some(false)) | (None, None) => None,
        _ => None,
    }
}

fn cast_literal_to_type(literal: &Literal, data_type: &DataType) -> Option<Literal> {
    if matches!(literal, Literal::Null) {
        return Some(Literal::Null);
    }

    match data_type {
        DataType::Boolean => literal_truthiness(literal).map(Literal::Boolean),
        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
            let value = literal_to_f64(literal)?;
            Some(Literal::Float64(value))
        }
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => {
            let value = literal_to_i128(literal)?;
            Some(Literal::Int128(value))
        }
        DataType::Utf8 | DataType::LargeUtf8 => Some(Literal::String(match literal {
            Literal::String(text) => text.clone(),
            Literal::Int128(value) => value.to_string(),
            Literal::Float64(value) => value.to_string(),
            Literal::Decimal128(value) => value.to_string(),
            Literal::Boolean(value) => {
                if *value {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            }
            Literal::Date32(days) => format_date32_literal(*days).ok()?,
            Literal::Struct(_) | Literal::Null | Literal::Interval(_) => return None,
        })),
        DataType::Decimal128(precision, scale) => {
            literal_to_decimal_literal(literal, *precision, *scale)
        }
        DataType::Decimal256(precision, scale) => {
            literal_to_decimal_literal(literal, *precision, *scale)
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => match literal {
            Literal::Interval(interval) => Some(Literal::Interval(*interval)),
            Literal::Null => Some(Literal::Null),
            _ => None,
        },
        DataType::Date32 => match literal {
            Literal::Null => Some(Literal::Null),
            Literal::Date32(days) => Some(Literal::Date32(*days)),
            Literal::String(text) => parse_date32_literal(text).ok().map(Literal::Date32),
            _ => None,
        },
        _ => None,
    }
}

fn literal_to_decimal_literal(literal: &Literal, precision: u8, scale: i8) -> Option<Literal> {
    match literal {
        Literal::Decimal128(value) => align_decimal_to_scale(*value, precision, scale)
            .ok()
            .map(Literal::Decimal128),
        Literal::Int128(value) => {
            let int = i64::try_from(*value).ok()?;
            decimal_from_i64(int, precision, scale)
                .ok()
                .map(Literal::Decimal128)
        }
        Literal::Float64(value) => decimal_from_f64(*value, precision, scale)
            .ok()
            .map(Literal::Decimal128),
        Literal::Boolean(value) => {
            let int = if *value { 1 } else { 0 };
            decimal_from_i64(int, precision, scale)
                .ok()
                .map(Literal::Decimal128)
        }
        Literal::Null => Some(Literal::Null),
        _ => None,
    }
}

fn compare_literals(op: CompareOp, left: &Literal, right: &Literal) -> Option<bool> {
    compare_literals_with_mode(op, left, right, NullComparisonBehavior::ThreeValuedLogic)
}

fn bind_select_filter(
    filter: &llkv_plan::SelectFilter,
    bindings: &FxHashMap<String, Literal>,
) -> ExecutorResult<llkv_plan::SelectFilter> {
    let predicate = bind_predicate_expr(&filter.predicate, bindings)?;
    let subqueries = filter
        .subqueries
        .iter()
        .map(|subquery| bind_filter_subquery(subquery, bindings))
        .collect::<ExecutorResult<Vec<_>>>()?;

    Ok(llkv_plan::SelectFilter {
        predicate,
        subqueries,
    })
}

fn bind_filter_subquery(
    subquery: &llkv_plan::FilterSubquery,
    bindings: &FxHashMap<String, Literal>,
) -> ExecutorResult<llkv_plan::FilterSubquery> {
    let bound_plan = bind_select_plan(&subquery.plan, bindings)?;
    Ok(llkv_plan::FilterSubquery {
        id: subquery.id,
        plan: Box::new(bound_plan),
        correlated_columns: subquery.correlated_columns.clone(),
    })
}

fn bind_scalar_subquery(
    subquery: &llkv_plan::ScalarSubquery,
    bindings: &FxHashMap<String, Literal>,
) -> ExecutorResult<llkv_plan::ScalarSubquery> {
    let bound_plan = bind_select_plan(&subquery.plan, bindings)?;
    Ok(llkv_plan::ScalarSubquery {
        id: subquery.id,
        plan: Box::new(bound_plan),
        correlated_columns: subquery.correlated_columns.clone(),
    })
}

fn bind_projection(
    projection: &SelectProjection,
    bindings: &FxHashMap<String, Literal>,
) -> ExecutorResult<SelectProjection> {
    match projection {
        SelectProjection::AllColumns => Ok(projection.clone()),
        SelectProjection::AllColumnsExcept { exclude } => Ok(SelectProjection::AllColumnsExcept {
            exclude: exclude.clone(),
        }),
        SelectProjection::Column { name, alias } => {
            if let Some(literal) = bindings.get(name) {
                let expr = ScalarExpr::Literal(literal.clone());
                Ok(SelectProjection::Computed {
                    expr,
                    alias: alias.clone().unwrap_or_else(|| name.clone()),
                })
            } else {
                Ok(projection.clone())
            }
        }
        SelectProjection::Computed { expr, alias } => Ok(SelectProjection::Computed {
            expr: bind_scalar_expr(expr, bindings)?,
            alias: alias.clone(),
        }),
    }
}

fn bind_aggregate_expr(
    aggregate: &AggregateExpr,
    bindings: &FxHashMap<String, Literal>,
) -> ExecutorResult<AggregateExpr> {
    match aggregate {
        AggregateExpr::CountStar { .. } => Ok(aggregate.clone()),
        AggregateExpr::Column {
            column,
            alias,
            function,
            distinct,
        } => {
            if bindings.contains_key(column) {
                return Err(Error::InvalidArgumentError(
                    "correlated columns are not supported inside aggregate expressions".into(),
                ));
            }
            Ok(AggregateExpr::Column {
                column: column.clone(),
                alias: alias.clone(),
                function: function.clone(),
                distinct: *distinct,
            })
        }
    }
}

fn bind_scalar_expr(
    expr: &ScalarExpr<String>,
    bindings: &FxHashMap<String, Literal>,
) -> ExecutorResult<ScalarExpr<String>> {
    match expr {
        ScalarExpr::Column(name) => {
            if let Some(literal) = bindings.get(name) {
                Ok(ScalarExpr::Literal(literal.clone()))
            } else {
                Ok(ScalarExpr::Column(name.clone()))
            }
        }
        ScalarExpr::Literal(literal) => Ok(ScalarExpr::Literal(literal.clone())),
        ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
            left: Box::new(bind_scalar_expr(left, bindings)?),
            op: *op,
            right: Box::new(bind_scalar_expr(right, bindings)?),
        }),
        ScalarExpr::Compare { left, op, right } => Ok(ScalarExpr::Compare {
            left: Box::new(bind_scalar_expr(left, bindings)?),
            op: *op,
            right: Box::new(bind_scalar_expr(right, bindings)?),
        }),
        ScalarExpr::Aggregate(call) => Ok(ScalarExpr::Aggregate(call.clone())),
        ScalarExpr::GetField { base, field_name } => {
            let bound_base = bind_scalar_expr(base, bindings)?;
            match bound_base {
                ScalarExpr::Literal(literal) => {
                    let value = extract_struct_field(&literal, field_name).unwrap_or(Literal::Null);
                    Ok(ScalarExpr::Literal(value))
                }
                other => Ok(ScalarExpr::GetField {
                    base: Box::new(other),
                    field_name: field_name.clone(),
                }),
            }
        }
        ScalarExpr::Cast { expr, data_type } => Ok(ScalarExpr::Cast {
            expr: Box::new(bind_scalar_expr(expr, bindings)?),
            data_type: data_type.clone(),
        }),
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            let bound_operand = match operand {
                Some(inner) => Some(Box::new(bind_scalar_expr(inner, bindings)?)),
                None => None,
            };
            let mut bound_branches = Vec::with_capacity(branches.len());
            for (when_expr, then_expr) in branches {
                bound_branches.push((
                    bind_scalar_expr(when_expr, bindings)?,
                    bind_scalar_expr(then_expr, bindings)?,
                ));
            }
            let bound_else = match else_expr {
                Some(inner) => Some(Box::new(bind_scalar_expr(inner, bindings)?)),
                None => None,
            };
            Ok(ScalarExpr::Case {
                operand: bound_operand,
                branches: bound_branches,
                else_expr: bound_else,
            })
        }
        ScalarExpr::Coalesce(items) => {
            let mut bound_items = Vec::with_capacity(items.len());
            for item in items {
                bound_items.push(bind_scalar_expr(item, bindings)?);
            }
            Ok(ScalarExpr::Coalesce(bound_items))
        }
        ScalarExpr::Not(inner) => Ok(ScalarExpr::Not(Box::new(bind_scalar_expr(
            inner, bindings,
        )?))),
        ScalarExpr::IsNull { expr, negated } => Ok(ScalarExpr::IsNull {
            expr: Box::new(bind_scalar_expr(expr, bindings)?),
            negated: *negated,
        }),
        ScalarExpr::Random => Ok(ScalarExpr::Random),
        ScalarExpr::ScalarSubquery(sub) => Ok(ScalarExpr::ScalarSubquery(sub.clone())),
    }
}

fn bind_predicate_expr(
    expr: &LlkvExpr<'static, String>,
    bindings: &FxHashMap<String, Literal>,
) -> ExecutorResult<LlkvExpr<'static, String>> {
    match expr {
        LlkvExpr::And(children) => {
            let mut bound = Vec::with_capacity(children.len());
            for child in children {
                bound.push(bind_predicate_expr(child, bindings)?);
            }
            Ok(LlkvExpr::And(bound))
        }
        LlkvExpr::Or(children) => {
            let mut bound = Vec::with_capacity(children.len());
            for child in children {
                bound.push(bind_predicate_expr(child, bindings)?);
            }
            Ok(LlkvExpr::Or(bound))
        }
        LlkvExpr::Not(inner) => Ok(LlkvExpr::Not(Box::new(bind_predicate_expr(
            inner, bindings,
        )?))),
        LlkvExpr::Pred(filter) => bind_filter_predicate(filter, bindings),
        LlkvExpr::Compare { left, op, right } => Ok(LlkvExpr::Compare {
            left: bind_scalar_expr(left, bindings)?,
            op: *op,
            right: bind_scalar_expr(right, bindings)?,
        }),
        LlkvExpr::InList {
            expr,
            list,
            negated,
        } => {
            let target = bind_scalar_expr(expr, bindings)?;
            let mut bound_list = Vec::with_capacity(list.len());
            for item in list {
                bound_list.push(bind_scalar_expr(item, bindings)?);
            }
            Ok(LlkvExpr::InList {
                expr: target,
                list: bound_list,
                negated: *negated,
            })
        }
        LlkvExpr::IsNull { expr, negated } => Ok(LlkvExpr::IsNull {
            expr: bind_scalar_expr(expr, bindings)?,
            negated: *negated,
        }),
        LlkvExpr::Literal(value) => Ok(LlkvExpr::Literal(*value)),
        LlkvExpr::Exists(subquery) => Ok(LlkvExpr::Exists(subquery.clone())),
    }
}

fn bind_filter_predicate(
    filter: &Filter<'static, String>,
    bindings: &FxHashMap<String, Literal>,
) -> ExecutorResult<LlkvExpr<'static, String>> {
    if let Some(literal) = bindings.get(&filter.field_id) {
        let result = evaluate_filter_against_literal(literal, &filter.op)?;
        return Ok(LlkvExpr::Literal(result));
    }
    Ok(LlkvExpr::Pred(filter.clone()))
}

fn evaluate_filter_against_literal(value: &Literal, op: &Operator) -> ExecutorResult<bool> {
    use std::ops::Bound;

    match op {
        Operator::IsNull => Ok(matches!(value, Literal::Null)),
        Operator::IsNotNull => Ok(!matches!(value, Literal::Null)),
        Operator::Equals(rhs) => Ok(literal_equals(value, rhs).unwrap_or(false)),
        Operator::GreaterThan(rhs) => Ok(literal_compare(value, rhs)
            .map(|cmp| cmp == std::cmp::Ordering::Greater)
            .unwrap_or(false)),
        Operator::GreaterThanOrEquals(rhs) => Ok(literal_compare(value, rhs)
            .map(|cmp| matches!(cmp, std::cmp::Ordering::Greater | std::cmp::Ordering::Equal))
            .unwrap_or(false)),
        Operator::LessThan(rhs) => Ok(literal_compare(value, rhs)
            .map(|cmp| cmp == std::cmp::Ordering::Less)
            .unwrap_or(false)),
        Operator::LessThanOrEquals(rhs) => Ok(literal_compare(value, rhs)
            .map(|cmp| matches!(cmp, std::cmp::Ordering::Less | std::cmp::Ordering::Equal))
            .unwrap_or(false)),
        Operator::In(values) => Ok(values
            .iter()
            .any(|candidate| literal_equals(value, candidate).unwrap_or(false))),
        Operator::Range { lower, upper } => {
            let lower_ok = match lower {
                Bound::Unbounded => Some(true),
                Bound::Included(bound) => literal_compare(value, bound).map(|cmp| {
                    matches!(cmp, std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                }),
                Bound::Excluded(bound) => {
                    literal_compare(value, bound).map(|cmp| cmp == std::cmp::Ordering::Greater)
                }
            }
            .unwrap_or(false);

            let upper_ok = match upper {
                Bound::Unbounded => Some(true),
                Bound::Included(bound) => literal_compare(value, bound)
                    .map(|cmp| matches!(cmp, std::cmp::Ordering::Less | std::cmp::Ordering::Equal)),
                Bound::Excluded(bound) => {
                    literal_compare(value, bound).map(|cmp| cmp == std::cmp::Ordering::Less)
                }
            }
            .unwrap_or(false);

            Ok(lower_ok && upper_ok)
        }
        Operator::StartsWith {
            pattern,
            case_sensitive,
        } => {
            let target = if *case_sensitive {
                pattern.to_string()
            } else {
                pattern.to_ascii_lowercase()
            };
            Ok(literal_string(value, *case_sensitive)
                .map(|source| source.starts_with(&target))
                .unwrap_or(false))
        }
        Operator::EndsWith {
            pattern,
            case_sensitive,
        } => {
            let target = if *case_sensitive {
                pattern.to_string()
            } else {
                pattern.to_ascii_lowercase()
            };
            Ok(literal_string(value, *case_sensitive)
                .map(|source| source.ends_with(&target))
                .unwrap_or(false))
        }
        Operator::Contains {
            pattern,
            case_sensitive,
        } => {
            let target = if *case_sensitive {
                pattern.to_string()
            } else {
                pattern.to_ascii_lowercase()
            };
            Ok(literal_string(value, *case_sensitive)
                .map(|source| source.contains(&target))
                .unwrap_or(false))
        }
    }
}

fn literal_compare(lhs: &Literal, rhs: &Literal) -> Option<std::cmp::Ordering> {
    match (lhs, rhs) {
        (Literal::Int128(a), Literal::Int128(b)) => Some(a.cmp(b)),
        (Literal::Float64(a), Literal::Float64(b)) => a.partial_cmp(b),
        (Literal::Int128(a), Literal::Float64(b)) => (*a as f64).partial_cmp(b),
        (Literal::Float64(a), Literal::Int128(b)) => a.partial_cmp(&(*b as f64)),
        (Literal::Date32(a), Literal::Date32(b)) => Some(a.cmp(b)),
        (Literal::Date32(a), Literal::Int128(b)) => Some((*a as i128).cmp(b)),
        (Literal::Int128(a), Literal::Date32(b)) => Some(a.cmp(&(*b as i128))),
        (Literal::Date32(a), Literal::Float64(b)) => (*a as f64).partial_cmp(b),
        (Literal::Float64(a), Literal::Date32(b)) => a.partial_cmp(&(*b as f64)),
        (Literal::String(a), Literal::String(b)) => Some(a.cmp(b)),
        (Literal::Interval(a), Literal::Interval(b)) => Some(compare_interval_values(*a, *b)),
        _ => None,
    }
}

fn literal_equals(lhs: &Literal, rhs: &Literal) -> Option<bool> {
    match (lhs, rhs) {
        (Literal::Boolean(a), Literal::Boolean(b)) => Some(a == b),
        (Literal::String(a), Literal::String(b)) => Some(a == b),
        (Literal::Int128(_), Literal::Int128(_))
        | (Literal::Int128(_), Literal::Float64(_))
        | (Literal::Float64(_), Literal::Int128(_))
        | (Literal::Float64(_), Literal::Float64(_))
        | (Literal::Date32(_), Literal::Date32(_))
        | (Literal::Date32(_), Literal::Int128(_))
        | (Literal::Int128(_), Literal::Date32(_))
        | (Literal::Date32(_), Literal::Float64(_))
        | (Literal::Float64(_), Literal::Date32(_))
        | (Literal::Interval(_), Literal::Interval(_)) => {
            literal_compare(lhs, rhs).map(|cmp| cmp == std::cmp::Ordering::Equal)
        }
        _ => None,
    }
}

fn literal_string(literal: &Literal, case_sensitive: bool) -> Option<String> {
    match literal {
        Literal::String(value) => {
            if case_sensitive {
                Some(value.clone())
            } else {
                Some(value.to_ascii_lowercase())
            }
        }
        Literal::Date32(value) => {
            let formatted = format_date32_literal(*value).ok()?;
            if case_sensitive {
                Some(formatted)
            } else {
                Some(formatted.to_ascii_lowercase())
            }
        }
        _ => None,
    }
}

fn extract_struct_field(literal: &Literal, field_name: &str) -> Option<Literal> {
    if let Literal::Struct(fields) = literal {
        for (name, value) in fields {
            if name.eq_ignore_ascii_case(field_name) {
                return Some((**value).clone());
            }
        }
    }
    None
}

fn collect_scalar_subquery_ids(expr: &ScalarExpr<FieldId>, ids: &mut FxHashSet<SubqueryId>) {
    match expr {
        ScalarExpr::ScalarSubquery(subquery) => {
            ids.insert(subquery.id);
        }
        ScalarExpr::Binary { left, right, .. } => {
            collect_scalar_subquery_ids(left, ids);
            collect_scalar_subquery_ids(right, ids);
        }
        ScalarExpr::Compare { left, right, .. } => {
            collect_scalar_subquery_ids(left, ids);
            collect_scalar_subquery_ids(right, ids);
        }
        ScalarExpr::GetField { base, .. } => {
            collect_scalar_subquery_ids(base, ids);
        }
        ScalarExpr::Cast { expr, .. } => {
            collect_scalar_subquery_ids(expr, ids);
        }
        ScalarExpr::Not(expr) => {
            collect_scalar_subquery_ids(expr, ids);
        }
        ScalarExpr::IsNull { expr, .. } => {
            collect_scalar_subquery_ids(expr, ids);
        }
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            if let Some(op) = operand {
                collect_scalar_subquery_ids(op, ids);
            }
            for (when_expr, then_expr) in branches {
                collect_scalar_subquery_ids(when_expr, ids);
                collect_scalar_subquery_ids(then_expr, ids);
            }
            if let Some(else_expr) = else_expr {
                collect_scalar_subquery_ids(else_expr, ids);
            }
        }
        ScalarExpr::Coalesce(items) => {
            for item in items {
                collect_scalar_subquery_ids(item, ids);
            }
        }
        ScalarExpr::Aggregate(_)
        | ScalarExpr::Column(_)
        | ScalarExpr::Literal(_)
        | ScalarExpr::Random => {}
    }
}

fn collect_predicate_scalar_subquery_ids(
    expr: &LlkvExpr<'static, FieldId>,
    ids: &mut FxHashSet<SubqueryId>,
) {
    match expr {
        LlkvExpr::And(children) | LlkvExpr::Or(children) => {
            for child in children {
                collect_predicate_scalar_subquery_ids(child, ids);
            }
        }
        LlkvExpr::Not(inner) => collect_predicate_scalar_subquery_ids(inner, ids),
        LlkvExpr::Compare { left, right, .. } => {
            collect_scalar_subquery_ids(left, ids);
            collect_scalar_subquery_ids(right, ids);
        }
        LlkvExpr::InList { expr, list, .. } => {
            collect_scalar_subquery_ids(expr, ids);
            for item in list {
                collect_scalar_subquery_ids(item, ids);
            }
        }
        LlkvExpr::IsNull { expr, .. } => {
            collect_scalar_subquery_ids(expr, ids);
        }
        LlkvExpr::Exists(_) | LlkvExpr::Pred(_) | LlkvExpr::Literal(_) => {
            // EXISTS subqueries handled separately.
        }
    }
}

fn rewrite_scalar_expr_for_subqueries(
    expr: &ScalarExpr<FieldId>,
    mapping: &FxHashMap<SubqueryId, FieldId>,
) -> ScalarExpr<FieldId> {
    match expr {
        ScalarExpr::ScalarSubquery(subquery) => mapping
            .get(&subquery.id)
            .map(|field_id| ScalarExpr::Column(*field_id))
            .unwrap_or_else(|| ScalarExpr::ScalarSubquery(subquery.clone())),
        ScalarExpr::Binary { left, op, right } => ScalarExpr::Binary {
            left: Box::new(rewrite_scalar_expr_for_subqueries(left, mapping)),
            op: *op,
            right: Box::new(rewrite_scalar_expr_for_subqueries(right, mapping)),
        },
        ScalarExpr::Compare { left, op, right } => ScalarExpr::Compare {
            left: Box::new(rewrite_scalar_expr_for_subqueries(left, mapping)),
            op: *op,
            right: Box::new(rewrite_scalar_expr_for_subqueries(right, mapping)),
        },
        ScalarExpr::GetField { base, field_name } => ScalarExpr::GetField {
            base: Box::new(rewrite_scalar_expr_for_subqueries(base, mapping)),
            field_name: field_name.clone(),
        },
        ScalarExpr::Cast { expr, data_type } => ScalarExpr::Cast {
            expr: Box::new(rewrite_scalar_expr_for_subqueries(expr, mapping)),
            data_type: data_type.clone(),
        },
        ScalarExpr::Not(expr) => {
            ScalarExpr::Not(Box::new(rewrite_scalar_expr_for_subqueries(expr, mapping)))
        }
        ScalarExpr::IsNull { expr, negated } => ScalarExpr::IsNull {
            expr: Box::new(rewrite_scalar_expr_for_subqueries(expr, mapping)),
            negated: *negated,
        },
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => ScalarExpr::Case {
            operand: operand
                .as_ref()
                .map(|op| Box::new(rewrite_scalar_expr_for_subqueries(op, mapping))),
            branches: branches
                .iter()
                .map(|(when_expr, then_expr)| {
                    (
                        rewrite_scalar_expr_for_subqueries(when_expr, mapping),
                        rewrite_scalar_expr_for_subqueries(then_expr, mapping),
                    )
                })
                .collect(),
            else_expr: else_expr
                .as_ref()
                .map(|expr| Box::new(rewrite_scalar_expr_for_subqueries(expr, mapping))),
        },
        ScalarExpr::Coalesce(items) => ScalarExpr::Coalesce(
            items
                .iter()
                .map(|item| rewrite_scalar_expr_for_subqueries(item, mapping))
                .collect(),
        ),
        ScalarExpr::Aggregate(_)
        | ScalarExpr::Column(_)
        | ScalarExpr::Literal(_)
        | ScalarExpr::Random => expr.clone(),
    }
}

fn collect_correlated_bindings(
    context: &mut CrossProductExpressionContext,
    batch: &RecordBatch,
    row_idx: usize,
    columns: &[llkv_plan::CorrelatedColumn],
) -> ExecutorResult<FxHashMap<String, Literal>> {
    let mut out = FxHashMap::default();

    for correlated in columns {
        if !correlated.field_path.is_empty() {
            return Err(Error::InvalidArgumentError(
                "correlated field path resolution is not yet supported".into(),
            ));
        }

        let field_id = context
            .field_id_for_column(&correlated.column)
            .ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "correlated column '{}' not found in outer query output",
                    correlated.column
                ))
            })?;

        let accessor = context.column_accessor(field_id, batch)?;
        let literal = accessor.literal_at(row_idx)?;
        out.insert(correlated.placeholder.clone(), literal);
    }

    Ok(out)
}

/// Streaming execution handle for SELECT queries.
#[derive(Clone)]
pub struct SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table_name: String,
    schema: Arc<Schema>,
    stream: SelectStream<P>,
    limit: Option<usize>,
    offset: Option<usize>,
}

#[derive(Clone)]
enum SelectStream<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    Projection {
        table: Arc<ExecutorTable<P>>,
        projections: Vec<ScanProjection>,
        filter_expr: LlkvExpr<'static, FieldId>,
        options: ScanStreamOptions<P>,
        full_table_scan: bool,
        order_by: Vec<OrderByPlan>,
        distinct: bool,
    },
    Aggregation {
        batch: RecordBatch,
    },
}

impl<P> SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    #[allow(clippy::too_many_arguments)]
    fn new_projection(
        table_name: String,
        schema: Arc<Schema>,
        table: Arc<ExecutorTable<P>>,
        projections: Vec<ScanProjection>,
        filter_expr: LlkvExpr<'static, FieldId>,
        options: ScanStreamOptions<P>,
        full_table_scan: bool,
        order_by: Vec<OrderByPlan>,
        distinct: bool,
    ) -> Self {
        Self {
            table_name,
            schema,
            stream: SelectStream::Projection {
                table,
                projections,
                filter_expr,
                options,
                full_table_scan,
                order_by,
                distinct,
            },
            limit: None,
            offset: None,
        }
    }

    pub fn new_single_batch(table_name: String, schema: Arc<Schema>, batch: RecordBatch) -> Self {
        Self {
            table_name,
            schema,
            stream: SelectStream::Aggregation { batch },
            limit: None,
            offset: None,
        }
    }

    pub fn from_batch(table_name: String, schema: Arc<Schema>, batch: RecordBatch) -> Self {
        Self::new_single_batch(table_name, schema, batch)
    }

    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    pub fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    pub fn with_limit(mut self, limit: Option<usize>) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_offset(mut self, offset: Option<usize>) -> Self {
        self.offset = offset;
        self
    }

    pub fn stream(
        self,
        mut on_batch: impl FnMut(RecordBatch) -> ExecutorResult<()>,
    ) -> ExecutorResult<()> {
        let limit = self.limit;
        let mut offset = self.offset.unwrap_or(0);
        let mut rows_emitted = 0;

        let mut on_batch = |batch: RecordBatch| -> ExecutorResult<()> {
            let rows = batch.num_rows();
            let mut batch_to_emit = batch;

            // Handle offset
            if offset > 0 {
                if rows == 0 {
                    // Pass through empty batches to preserve schema
                } else if rows <= offset {
                    offset -= rows;
                    return Ok(());
                } else {
                    batch_to_emit = batch_to_emit.slice(offset, rows - offset);
                    offset = 0;
                }
            }

            // Handle limit
            if let Some(limit_val) = limit {
                if rows_emitted >= limit_val {
                    return Ok(());
                }
                let remaining = limit_val - rows_emitted;
                if batch_to_emit.num_rows() > remaining {
                    batch_to_emit = batch_to_emit.slice(0, remaining);
                }
                rows_emitted += batch_to_emit.num_rows();
            }

            on_batch(batch_to_emit)
        };

        let schema = Arc::clone(&self.schema);
        match self.stream {
            SelectStream::Projection {
                table,
                projections,
                filter_expr,
                options,
                full_table_scan,
                order_by,
                distinct,
            } => {
                // Early return for empty tables to avoid ColumnStore data_type() errors
                let total_rows = table.total_rows.load(Ordering::SeqCst);
                if total_rows == 0 {
                    // Empty table - return empty result with correct schema
                    return Ok(());
                }

                let mut error: Option<Error> = None;
                let mut produced = false;
                let mut produced_rows: u64 = 0;
                let capture_nulls_first = matches!(options.order, Some(spec) if spec.nulls_first);
                let needs_post_sort =
                    !order_by.is_empty() && (order_by.len() > 1 || options.order.is_none());
                let collect_batches = needs_post_sort || capture_nulls_first;
                let include_nulls = options.include_nulls;
                let has_row_id_filter = options.row_id_filter.is_some();
                let mut distinct_state = if distinct {
                    Some(DistinctState::default())
                } else {
                    None
                };
                let scan_options = options;
                let mut buffered_batches: Vec<RecordBatch> = Vec::new();
                table
                    .table
                    .scan_stream(projections, &filter_expr, scan_options, |batch| {
                        if error.is_some() {
                            return;
                        }
                        let mut batch = batch;
                        if let Some(state) = distinct_state.as_mut() {
                            match distinct_filter_batch(batch, state) {
                                Ok(Some(filtered)) => {
                                    batch = filtered;
                                }
                                Ok(None) => {
                                    return;
                                }
                                Err(err) => {
                                    error = Some(err);
                                    return;
                                }
                            }
                        }
                        produced = true;
                        produced_rows = produced_rows.saturating_add(batch.num_rows() as u64);
                        if collect_batches {
                            buffered_batches.push(batch);
                        } else if let Err(err) = on_batch(batch) {
                            error = Some(err);
                        }
                    })?;
                if let Some(err) = error {
                    return Err(err);
                }
                if !produced {
                    // Only synthesize null rows if this was a full table scan
                    // If there was a filter and it matched no rows, we should return empty results
                    if !distinct && full_table_scan && total_rows > 0 {
                        for batch in synthesize_null_scan(Arc::clone(&schema), total_rows)? {
                            on_batch(batch)?;
                        }
                    }
                    return Ok(());
                }
                let mut null_batches: Vec<RecordBatch> = Vec::new();
                // Only synthesize null rows if:
                // 1. include_nulls is true
                // 2. This is a full table scan
                // 3. We produced fewer rows than the total
                // 4. We DON'T have a row_id_filter (e.g., MVCC filter) that intentionally filtered rows
                if !distinct
                    && include_nulls
                    && full_table_scan
                    && produced_rows < total_rows
                    && !has_row_id_filter
                {
                    let missing = total_rows - produced_rows;
                    if missing > 0 {
                        null_batches = synthesize_null_scan(Arc::clone(&schema), missing)?;
                    }
                }

                if collect_batches {
                    if needs_post_sort {
                        if !null_batches.is_empty() {
                            buffered_batches.extend(null_batches);
                        }
                        if !buffered_batches.is_empty() {
                            let combined =
                                concat_batches(&schema, &buffered_batches).map_err(|err| {
                                    Error::InvalidArgumentError(format!(
                                        "failed to concatenate result batches for ORDER BY: {}",
                                        err
                                    ))
                                })?;
                            let sorted_batch =
                                sort_record_batch_with_order(&schema, &combined, &order_by)?;
                            on_batch(sorted_batch)?;
                        }
                    } else if capture_nulls_first {
                        for batch in null_batches {
                            on_batch(batch)?;
                        }
                        for batch in buffered_batches {
                            on_batch(batch)?;
                        }
                    }
                } else if !null_batches.is_empty() {
                    for batch in null_batches {
                        on_batch(batch)?;
                    }
                }
                Ok(())
            }
            SelectStream::Aggregation { batch } => on_batch(batch),
        }
    }

    pub fn collect(self) -> ExecutorResult<Vec<RecordBatch>> {
        let mut batches = Vec::new();
        self.stream(|batch| {
            batches.push(batch);
            Ok(())
        })?;
        Ok(batches)
    }

    pub fn collect_rows(self) -> ExecutorResult<ExecutorRowBatch> {
        let schema = self.schema();
        let mut rows: Vec<Vec<PlanValue>> = Vec::new();
        self.stream(|batch| {
            for row_idx in 0..batch.num_rows() {
                let mut row: Vec<PlanValue> = Vec::with_capacity(batch.num_columns());
                for col_idx in 0..batch.num_columns() {
                    let value = llkv_plan::plan_value_from_array(batch.column(col_idx), row_idx)?;
                    row.push(value);
                }
                rows.push(row);
            }
            Ok(())
        })?;
        let columns = schema
            .fields()
            .iter()
            .map(|field| field.name().to_string())
            .collect();
        Ok(ExecutorRowBatch { columns, rows })
    }

    pub fn into_rows(self) -> ExecutorResult<Vec<Vec<PlanValue>>> {
        Ok(self.collect_rows()?.rows)
    }
}

impl<P> fmt::Debug for SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SelectExecution")
            .field("table_name", &self.table_name)
            .field("schema", &self.schema)
            .finish()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Recursively infer the data type of a scalar expression from the schema.
fn infer_type_recursive(
    expr: &ScalarExpr<String>,
    base_schema: &Schema,
    column_lookup_map: &FxHashMap<String, usize>,
) -> Option<DataType> {
    use arrow::datatypes::IntervalUnit;
    use llkv_expr::literal::Literal;

    match expr {
        ScalarExpr::Column(name) => resolve_column_name_to_index(name, column_lookup_map)
            .map(|idx| base_schema.field(idx).data_type().clone()),
        ScalarExpr::Literal(lit) => match lit {
            Literal::Decimal128(v) => Some(DataType::Decimal128(v.precision(), v.scale())),
            Literal::Float64(_) => Some(DataType::Float64),
            Literal::Int128(_) => Some(DataType::Int64),
            Literal::Boolean(_) => Some(DataType::Boolean),
            Literal::String(_) => Some(DataType::Utf8),
            Literal::Date32(_) => Some(DataType::Date32),
            Literal::Null => Some(DataType::Null),
            Literal::Interval(_) => Some(DataType::Interval(IntervalUnit::MonthDayNano)),
            _ => None,
        },
        ScalarExpr::Binary { left, op: _, right } => {
            let l = infer_type_recursive(left, base_schema, column_lookup_map)?;
            let r = infer_type_recursive(right, base_schema, column_lookup_map)?;

            if matches!(l, DataType::Float64) || matches!(r, DataType::Float64) {
                return Some(DataType::Float64);
            }

            match (l, r) {
                (DataType::Decimal128(_, s1), DataType::Decimal128(_, s2)) => {
                    // Propagate decimal type with max scale
                    Some(DataType::Decimal128(38, s1.max(s2)))
                }
                (DataType::Decimal128(p, s), _) => Some(DataType::Decimal128(p, s)),
                (_, DataType::Decimal128(p, s)) => Some(DataType::Decimal128(p, s)),
                (l, _) => Some(l),
            }
        }
        ScalarExpr::Cast { data_type, .. } => Some(data_type.clone()),
        // For other types, return None to fall back to sample evaluation
        _ => None,
    }
}

fn expand_order_targets(
    order_items: &[OrderByPlan],
    projections: &[ScanProjection],
) -> ExecutorResult<Vec<OrderByPlan>> {
    let mut expanded = Vec::new();

    for item in order_items {
        match &item.target {
            OrderTarget::All => {
                if projections.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "ORDER BY ALL requires at least one projection".into(),
                    ));
                }

                for (idx, projection) in projections.iter().enumerate() {
                    if matches!(projection, ScanProjection::Computed { .. }) {
                        return Err(Error::InvalidArgumentError(
                            "ORDER BY ALL cannot reference computed projections".into(),
                        ));
                    }

                    let mut clone = item.clone();
                    clone.target = OrderTarget::Index(idx);
                    expanded.push(clone);
                }
            }
            _ => expanded.push(item.clone()),
        }
    }

    Ok(expanded)
}

fn resolve_scan_order<P>(
    table: &ExecutorTable<P>,
    projections: &[ScanProjection],
    order_plan: &OrderByPlan,
) -> ExecutorResult<ScanOrderSpec>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let (column, field_id) = match &order_plan.target {
        OrderTarget::Column(name) => {
            let column = table.schema.resolve(name).ok_or_else(|| {
                Error::InvalidArgumentError(format!("unknown column '{}' in ORDER BY", name))
            })?;
            (column, column.field_id)
        }
        OrderTarget::Index(position) => {
            let projection = projections.get(*position).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "ORDER BY position {} is out of range",
                    position + 1
                ))
            })?;
            match projection {
                ScanProjection::Column(store_projection) => {
                    let field_id = store_projection.logical_field_id.field_id();
                    let column = table.schema.column_by_field_id(field_id).ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "unknown column with field id {field_id} in ORDER BY"
                        ))
                    })?;
                    (column, field_id)
                }
                ScanProjection::Computed { .. } => {
                    return Err(Error::InvalidArgumentError(
                        "ORDER BY position referring to computed projection is not supported"
                            .into(),
                    ));
                }
            }
        }
        OrderTarget::All => {
            return Err(Error::InvalidArgumentError(
                "ORDER BY ALL should be expanded before execution".into(),
            ));
        }
    };

    let transform = match order_plan.sort_type {
        OrderSortType::Native => match column.data_type {
            DataType::Int64 => ScanOrderTransform::IdentityInt64,
            DataType::Int32 => ScanOrderTransform::IdentityInt32,
            DataType::Utf8 => ScanOrderTransform::IdentityUtf8,
            ref other => {
                return Err(Error::InvalidArgumentError(format!(
                    "ORDER BY on column type {:?} is not supported",
                    other
                )));
            }
        },
        OrderSortType::CastTextToInteger => {
            if column.data_type != DataType::Utf8 {
                return Err(Error::InvalidArgumentError(
                    "ORDER BY CAST expects a text column".into(),
                ));
            }
            ScanOrderTransform::CastUtf8ToInteger
        }
    };

    let direction = if order_plan.ascending {
        ScanOrderDirection::Ascending
    } else {
        ScanOrderDirection::Descending
    };

    Ok(ScanOrderSpec {
        field_id,
        direction,
        nulls_first: order_plan.nulls_first,
        transform,
    })
}

fn synthesize_null_scan(schema: Arc<Schema>, total_rows: u64) -> ExecutorResult<Vec<RecordBatch>> {
    let row_count = usize::try_from(total_rows).map_err(|_| {
        Error::InvalidArgumentError("table row count exceeds supported in-memory batch size".into())
    })?;

    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        match field.data_type() {
            DataType::Int64 => {
                let mut builder = Int64Builder::with_capacity(row_count);
                for _ in 0..row_count {
                    builder.append_null();
                }
                arrays.push(Arc::new(builder.finish()));
            }
            DataType::Float64 => {
                let mut builder = arrow::array::Float64Builder::with_capacity(row_count);
                for _ in 0..row_count {
                    builder.append_null();
                }
                arrays.push(Arc::new(builder.finish()));
            }
            DataType::Utf8 => {
                let mut builder = arrow::array::StringBuilder::with_capacity(row_count, 0);
                for _ in 0..row_count {
                    builder.append_null();
                }
                arrays.push(Arc::new(builder.finish()));
            }
            DataType::Date32 => {
                let mut builder = arrow::array::Date32Builder::with_capacity(row_count);
                for _ in 0..row_count {
                    builder.append_null();
                }
                arrays.push(Arc::new(builder.finish()));
            }
            other => {
                return Err(Error::InvalidArgumentError(format!(
                    "unsupported data type in null synthesis: {other:?}"
                )));
            }
        }
    }

    let batch = RecordBatch::try_new(schema, arrays)?;
    Ok(vec![batch])
}

struct TableCrossProductData {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
    column_counts: Vec<usize>,
    table_indices: Vec<usize>,
}

fn plan_value_to_literal(value: &PlanValue) -> ExecutorResult<Literal> {
    match value {
        PlanValue::String(s) => Ok(Literal::String(s.clone())),
        PlanValue::Integer(i) => Ok(Literal::Int128(*i as i128)),
        PlanValue::Float(f) => Ok(Literal::Float64(*f)),
        PlanValue::Null => Ok(Literal::Null),
        PlanValue::Date32(d) => Ok(Literal::Date32(*d)),
        PlanValue::Decimal(d) => Ok(Literal::Decimal128(*d)),
        _ => Err(Error::Internal(format!(
            "unsupported plan value for literal conversion: {:?}",
            value
        ))),
    }
}

fn collect_table_data<P>(
    table_index: usize,
    table_ref: &llkv_plan::TableRef,
    table: &ExecutorTable<P>,
    constraints: &[ColumnConstraint],
) -> ExecutorResult<TableCrossProductData>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    if table.schema.columns.is_empty() {
        return Err(Error::InvalidArgumentError(format!(
            "table '{}' has no columns; cross products require at least one column",
            table_ref.qualified_name()
        )));
    }

    let mut projections = Vec::with_capacity(table.schema.columns.len());
    let mut fields = Vec::with_capacity(table.schema.columns.len());

    for column in &table.schema.columns {
        let table_component = table_ref
            .alias
            .as_deref()
            .unwrap_or(table_ref.table.as_str());
        let qualified_name = format!("{}.{}.{}", table_ref.schema, table_component, column.name);
        projections.push(ScanProjection::from(StoreProjection::with_alias(
            LogicalFieldId::for_user(table.table.table_id(), column.field_id),
            qualified_name.clone(),
        )));
        fields.push(Field::new(
            qualified_name,
            column.data_type.clone(),
            column.nullable,
        ));
    }

    let schema = Arc::new(Schema::new(fields));

    let filter_field_id = table.schema.first_field_id().unwrap_or(ROW_ID_FIELD_ID);

    // Build filter expression from constraints to push down to scan
    let mut filter_exprs = Vec::new();
    for constraint in constraints {
        match constraint {
            ColumnConstraint::Equality(lit) => {
                let col_idx = lit.column.column;
                if col_idx < table.schema.columns.len() {
                    let field_id = table.schema.columns[col_idx].field_id;
                    if let Ok(literal) = plan_value_to_literal(&lit.value) {
                        filter_exprs.push(LlkvExpr::Compare {
                            left: ScalarExpr::Column(field_id),
                            op: CompareOp::Eq,
                            right: ScalarExpr::Literal(literal),
                        });
                    }
                }
            }
            ColumnConstraint::InList(in_list) => {
                let col_idx = in_list.column.column;
                if col_idx < table.schema.columns.len() {
                    let field_id = table.schema.columns[col_idx].field_id;
                    let literals: Vec<Literal> = in_list
                        .values
                        .iter()
                        .filter_map(|v| plan_value_to_literal(v).ok())
                        .collect();

                    if !literals.is_empty() {
                        filter_exprs.push(LlkvExpr::InList {
                            expr: ScalarExpr::Column(field_id),
                            list: literals.into_iter().map(ScalarExpr::Literal).collect(),
                            negated: false,
                        });
                    }
                }
            }
        }
    }

    let filter_expr = if filter_exprs.is_empty() {
        crate::translation::expression::full_table_scan_filter(filter_field_id)
    } else if filter_exprs.len() == 1 {
        filter_exprs.pop().unwrap()
    } else {
        LlkvExpr::And(filter_exprs)
    };

    let mut raw_batches = Vec::new();
    table.table.scan_stream(
        projections,
        &filter_expr,
        ScanStreamOptions {
            include_nulls: true,
            ..ScanStreamOptions::default()
        },
        |batch| {
            raw_batches.push(batch);
        },
    )?;

    let mut normalized_batches = Vec::with_capacity(raw_batches.len());
    for batch in raw_batches {
        let normalized = RecordBatch::try_new(Arc::clone(&schema), batch.columns().to_vec())
            .map_err(|err| {
                Error::Internal(format!(
                    "failed to align scan batch for table '{}': {}",
                    table_ref.qualified_name(),
                    err
                ))
            })?;
        normalized_batches.push(normalized);
    }

    if !constraints.is_empty() {
        normalized_batches = apply_column_constraints_to_batches(normalized_batches, constraints)?;
    }

    Ok(TableCrossProductData {
        schema,
        batches: normalized_batches,
        column_counts: vec![table.schema.columns.len()],
        table_indices: vec![table_index],
    })
}

fn apply_column_constraints_to_batches(
    batches: Vec<RecordBatch>,
    constraints: &[ColumnConstraint],
) -> ExecutorResult<Vec<RecordBatch>> {
    if batches.is_empty() {
        return Ok(batches);
    }

    let mut filtered = batches;
    for constraint in constraints {
        match constraint {
            ColumnConstraint::Equality(lit) => {
                filtered = filter_batches_by_literal(filtered, lit.column.column, &lit.value)?;
            }
            ColumnConstraint::InList(in_list) => {
                filtered =
                    filter_batches_by_in_list(filtered, in_list.column.column, &in_list.values)?;
            }
        }
        if filtered.is_empty() {
            break;
        }
    }

    Ok(filtered)
}

fn filter_batches_by_literal(
    batches: Vec<RecordBatch>,
    column_idx: usize,
    literal: &PlanValue,
) -> ExecutorResult<Vec<RecordBatch>> {
    let mut result = Vec::with_capacity(batches.len());

    for batch in batches {
        if column_idx >= batch.num_columns() {
            return Err(Error::Internal(
                "literal constraint referenced invalid column index".into(),
            ));
        }

        if batch.num_rows() == 0 {
            result.push(batch);
            continue;
        }

        let column = batch.column(column_idx);
        let mut keep_rows: Vec<u32> = Vec::with_capacity(batch.num_rows());

        for row_idx in 0..batch.num_rows() {
            if array_value_equals_plan_value(column.as_ref(), row_idx, literal)? {
                keep_rows.push(row_idx as u32);
            }
        }

        if keep_rows.len() == batch.num_rows() {
            result.push(batch);
            continue;
        }

        if keep_rows.is_empty() {
            // Constraint filtered out entire batch; skip it.
            continue;
        }

        let indices = UInt32Array::from(keep_rows);
        let mut filtered_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
        for col_idx in 0..batch.num_columns() {
            let filtered = take(batch.column(col_idx).as_ref(), &indices, None)
                .map_err(|err| Error::Internal(format!("failed to apply literal filter: {err}")))?;
            filtered_columns.push(filtered);
        }

        let filtered_batch =
            RecordBatch::try_new(batch.schema(), filtered_columns).map_err(|err| {
                Error::Internal(format!(
                    "failed to rebuild batch after literal filter: {err}"
                ))
            })?;
        result.push(filtered_batch);
    }

    Ok(result)
}

fn filter_batches_by_in_list(
    batches: Vec<RecordBatch>,
    column_idx: usize,
    values: &[PlanValue],
) -> ExecutorResult<Vec<RecordBatch>> {
    use arrow::array::*;
    use arrow::compute::or;

    if values.is_empty() {
        // Empty IN list matches nothing
        return Ok(Vec::new());
    }

    let mut result = Vec::with_capacity(batches.len());

    for batch in batches {
        if column_idx >= batch.num_columns() {
            return Err(Error::Internal(
                "IN list constraint referenced invalid column index".into(),
            ));
        }

        if batch.num_rows() == 0 {
            result.push(batch);
            continue;
        }

        let column = batch.column(column_idx);

        // Build a boolean mask: true if row matches ANY value in the IN list
        // Start with all false, then OR together comparisons for each value
        let mut mask = BooleanArray::from(vec![false; batch.num_rows()]);

        for value in values {
            let comparison_mask = build_comparison_mask(column.as_ref(), value)?;
            mask = or(&mask, &comparison_mask)
                .map_err(|err| Error::Internal(format!("failed to OR comparison masks: {err}")))?;
        }

        // Check if all rows match or no rows match for optimization
        let true_count = mask.true_count();
        if true_count == batch.num_rows() {
            result.push(batch);
            continue;
        }

        if true_count == 0 {
            // IN list filtered out entire batch; skip it.
            continue;
        }

        // Use Arrow's filter kernel for vectorized filtering
        let filtered_batch = arrow::compute::filter_record_batch(&batch, &mask)
            .map_err(|err| Error::Internal(format!("failed to apply IN list filter: {err}")))?;

        result.push(filtered_batch);
    }

    Ok(result)
}

/// Build a boolean mask for column == value comparison using vectorized operations.
fn build_comparison_mask(column: &dyn Array, value: &PlanValue) -> ExecutorResult<BooleanArray> {
    use arrow::array::*;
    use arrow::datatypes::DataType;

    match value {
        PlanValue::Null => {
            // For NULL, check if each element is null
            let mut builder = BooleanBuilder::with_capacity(column.len());
            for i in 0..column.len() {
                builder.append_value(column.is_null(i));
            }
            Ok(builder.finish())
        }
        PlanValue::Integer(val) => {
            let mut builder = BooleanBuilder::with_capacity(column.len());
            match column.data_type() {
                DataType::Int8 => {
                    let arr = column
                        .as_any()
                        .downcast_ref::<Int8Array>()
                        .ok_or_else(|| Error::Internal("failed to downcast to Int8Array".into()))?;
                    let target = *val as i8;
                    for i in 0..arr.len() {
                        builder.append_value(!arr.is_null(i) && arr.value(i) == target);
                    }
                }
                DataType::Int16 => {
                    let arr = column
                        .as_any()
                        .downcast_ref::<Int16Array>()
                        .ok_or_else(|| {
                            Error::Internal("failed to downcast to Int16Array".into())
                        })?;
                    let target = *val as i16;
                    for i in 0..arr.len() {
                        builder.append_value(!arr.is_null(i) && arr.value(i) == target);
                    }
                }
                DataType::Int32 => {
                    let arr = column
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .ok_or_else(|| {
                            Error::Internal("failed to downcast to Int32Array".into())
                        })?;
                    let target = *val as i32;
                    for i in 0..arr.len() {
                        builder.append_value(!arr.is_null(i) && arr.value(i) == target);
                    }
                }
                DataType::Int64 => {
                    let arr = column
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| {
                            Error::Internal("failed to downcast to Int64Array".into())
                        })?;
                    for i in 0..arr.len() {
                        builder.append_value(!arr.is_null(i) && arr.value(i) == *val);
                    }
                }
                DataType::UInt8 => {
                    let arr = column
                        .as_any()
                        .downcast_ref::<UInt8Array>()
                        .ok_or_else(|| {
                            Error::Internal("failed to downcast to UInt8Array".into())
                        })?;
                    let target = *val as u8;
                    for i in 0..arr.len() {
                        builder.append_value(!arr.is_null(i) && arr.value(i) == target);
                    }
                }
                DataType::UInt16 => {
                    let arr = column
                        .as_any()
                        .downcast_ref::<UInt16Array>()
                        .ok_or_else(|| {
                            Error::Internal("failed to downcast to UInt16Array".into())
                        })?;
                    let target = *val as u16;
                    for i in 0..arr.len() {
                        builder.append_value(!arr.is_null(i) && arr.value(i) == target);
                    }
                }
                DataType::UInt32 => {
                    let arr = column
                        .as_any()
                        .downcast_ref::<UInt32Array>()
                        .ok_or_else(|| {
                            Error::Internal("failed to downcast to UInt32Array".into())
                        })?;
                    let target = *val as u32;
                    for i in 0..arr.len() {
                        builder.append_value(!arr.is_null(i) && arr.value(i) == target);
                    }
                }
                DataType::UInt64 => {
                    let arr = column
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .ok_or_else(|| {
                            Error::Internal("failed to downcast to UInt64Array".into())
                        })?;
                    let target = *val as u64;
                    for i in 0..arr.len() {
                        builder.append_value(!arr.is_null(i) && arr.value(i) == target);
                    }
                }
                _ => {
                    return Err(Error::Internal(format!(
                        "unsupported integer type for IN list: {:?}",
                        column.data_type()
                    )));
                }
            }
            Ok(builder.finish())
        }
        PlanValue::Float(val) => {
            let mut builder = BooleanBuilder::with_capacity(column.len());
            match column.data_type() {
                DataType::Float32 => {
                    let arr = column
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| {
                            Error::Internal("failed to downcast to Float32Array".into())
                        })?;
                    let target = *val as f32;
                    for i in 0..arr.len() {
                        builder.append_value(!arr.is_null(i) && arr.value(i) == target);
                    }
                }
                DataType::Float64 => {
                    let arr = column
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| {
                            Error::Internal("failed to downcast to Float64Array".into())
                        })?;
                    for i in 0..arr.len() {
                        builder.append_value(!arr.is_null(i) && arr.value(i) == *val);
                    }
                }
                _ => {
                    return Err(Error::Internal(format!(
                        "unsupported float type for IN list: {:?}",
                        column.data_type()
                    )));
                }
            }
            Ok(builder.finish())
        }
        PlanValue::Decimal(expected) => match column.data_type() {
            DataType::Decimal128(precision, scale) => {
                let arr = column
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| {
                        Error::Internal("failed to downcast to Decimal128Array".into())
                    })?;
                let expected_aligned = align_decimal_to_scale(*expected, *precision, *scale)
                    .map_err(|err| {
                        Error::InvalidArgumentError(format!(
                            "decimal literal {expected} incompatible with DECIMAL({}, {}): {err}",
                            precision, scale
                        ))
                    })?;
                let mut builder = BooleanBuilder::with_capacity(arr.len());
                for i in 0..arr.len() {
                    if arr.is_null(i) {
                        builder.append_value(false);
                    } else {
                        let actual = DecimalValue::new(arr.value(i), *scale).map_err(|err| {
                            Error::InvalidArgumentError(format!(
                                "invalid decimal value stored in column: {err}"
                            ))
                        })?;
                        builder.append_value(actual.raw_value() == expected_aligned.raw_value());
                    }
                }
                Ok(builder.finish())
            }
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Boolean => {
                if let Some(int_value) = decimal_exact_i64(*expected) {
                    return build_comparison_mask(column, &PlanValue::Integer(int_value));
                }
                Ok(BooleanArray::from(vec![false; column.len()]))
            }
            DataType::Float32 | DataType::Float64 => {
                build_comparison_mask(column, &PlanValue::Float(expected.to_f64()))
            }
            _ => Err(Error::Internal(format!(
                "unsupported decimal type for IN list: {:?}",
                column.data_type()
            ))),
        },
        PlanValue::String(val) => {
            let mut builder = BooleanBuilder::with_capacity(column.len());
            let arr = column
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::Internal("failed to downcast to StringArray".into()))?;
            for i in 0..arr.len() {
                builder.append_value(!arr.is_null(i) && arr.value(i) == val.as_str());
            }
            Ok(builder.finish())
        }
        PlanValue::Date32(days) => {
            let mut builder = BooleanBuilder::with_capacity(column.len());
            match column.data_type() {
                DataType::Date32 => {
                    let arr = column
                        .as_any()
                        .downcast_ref::<Date32Array>()
                        .ok_or_else(|| {
                            Error::Internal("failed to downcast to Date32Array".into())
                        })?;
                    for i in 0..arr.len() {
                        builder.append_value(!arr.is_null(i) && arr.value(i) == *days);
                    }
                }
                _ => {
                    return Err(Error::Internal(format!(
                        "unsupported DATE type for IN list: {:?}",
                        column.data_type()
                    )));
                }
            }
            Ok(builder.finish())
        }
        PlanValue::Interval(interval) => {
            let mut builder = BooleanBuilder::with_capacity(column.len());
            match column.data_type() {
                DataType::Interval(IntervalUnit::MonthDayNano) => {
                    let arr = column
                        .as_any()
                        .downcast_ref::<IntervalMonthDayNanoArray>()
                        .ok_or_else(|| {
                            Error::Internal(
                                "failed to downcast to IntervalMonthDayNanoArray".into(),
                            )
                        })?;
                    let expected = *interval;
                    for i in 0..arr.len() {
                        if arr.is_null(i) {
                            builder.append_value(false);
                        } else {
                            let candidate = interval_value_from_arrow(arr.value(i));
                            let matches = compare_interval_values(expected, candidate)
                                == std::cmp::Ordering::Equal;
                            builder.append_value(matches);
                        }
                    }
                }
                _ => {
                    return Err(Error::Internal(format!(
                        "unsupported INTERVAL type for IN list: {:?}",
                        column.data_type()
                    )));
                }
            }
            Ok(builder.finish())
        }
        PlanValue::Struct(_) => Err(Error::Internal(
            "struct comparison in IN list not supported".into(),
        )),
    }
}

fn array_value_equals_plan_value(
    array: &dyn Array,
    row_idx: usize,
    literal: &PlanValue,
) -> ExecutorResult<bool> {
    use arrow::array::*;
    use arrow::datatypes::DataType;

    match literal {
        PlanValue::Null => Ok(array.is_null(row_idx)),
        PlanValue::Decimal(expected) => match array.data_type() {
            DataType::Decimal128(precision, scale) => {
                if array.is_null(row_idx) {
                    return Ok(false);
                }
                let arr = array
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| {
                        Error::Internal("failed to downcast to Decimal128Array".into())
                    })?;
                let actual = DecimalValue::new(arr.value(row_idx), *scale).map_err(|err| {
                    Error::InvalidArgumentError(format!(
                        "invalid decimal value retrieved from column: {err}"
                    ))
                })?;
                let expected_aligned = align_decimal_to_scale(*expected, *precision, *scale)
                    .map_err(|err| {
                        Error::InvalidArgumentError(format!(
                            "failed to align decimal literal for comparison: {err}"
                        ))
                    })?;
                Ok(actual.raw_value() == expected_aligned.raw_value())
            }
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => {
                if array.is_null(row_idx) {
                    return Ok(false);
                }
                if let Some(int_value) = decimal_exact_i64(*expected) {
                    array_value_equals_plan_value(array, row_idx, &PlanValue::Integer(int_value))
                } else {
                    Ok(false)
                }
            }
            DataType::Float32 | DataType::Float64 => {
                if array.is_null(row_idx) {
                    return Ok(false);
                }
                array_value_equals_plan_value(array, row_idx, &PlanValue::Float(expected.to_f64()))
            }
            DataType::Boolean => {
                if array.is_null(row_idx) {
                    return Ok(false);
                }
                if let Some(int_value) = decimal_exact_i64(*expected) {
                    array_value_equals_plan_value(array, row_idx, &PlanValue::Integer(int_value))
                } else {
                    Ok(false)
                }
            }
            _ => Err(Error::InvalidArgumentError(format!(
                "decimal literal comparison not supported for {:?}",
                array.data_type()
            ))),
        },
        PlanValue::Integer(expected) => match array.data_type() {
            DataType::Int8 => Ok(!array.is_null(row_idx)
                && array
                    .as_any()
                    .downcast_ref::<Int8Array>()
                    .expect("int8 array")
                    .value(row_idx) as i64
                    == *expected),
            DataType::Int16 => Ok(!array.is_null(row_idx)
                && array
                    .as_any()
                    .downcast_ref::<Int16Array>()
                    .expect("int16 array")
                    .value(row_idx) as i64
                    == *expected),
            DataType::Int32 => Ok(!array.is_null(row_idx)
                && array
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("int32 array")
                    .value(row_idx) as i64
                    == *expected),
            DataType::Int64 => Ok(!array.is_null(row_idx)
                && array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("int64 array")
                    .value(row_idx)
                    == *expected),
            DataType::UInt8 if *expected >= 0 => Ok(!array.is_null(row_idx)
                && array
                    .as_any()
                    .downcast_ref::<UInt8Array>()
                    .expect("uint8 array")
                    .value(row_idx) as i64
                    == *expected),
            DataType::UInt16 if *expected >= 0 => Ok(!array.is_null(row_idx)
                && array
                    .as_any()
                    .downcast_ref::<UInt16Array>()
                    .expect("uint16 array")
                    .value(row_idx) as i64
                    == *expected),
            DataType::UInt32 if *expected >= 0 => Ok(!array.is_null(row_idx)
                && array
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .expect("uint32 array")
                    .value(row_idx) as i64
                    == *expected),
            DataType::UInt64 if *expected >= 0 => Ok(!array.is_null(row_idx)
                && array
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .expect("uint64 array")
                    .value(row_idx)
                    == *expected as u64),
            DataType::Boolean => {
                if array.is_null(row_idx) {
                    Ok(false)
                } else if *expected == 0 || *expected == 1 {
                    let value = array
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .expect("bool array")
                        .value(row_idx);
                    Ok(value == (*expected == 1))
                } else {
                    Ok(false)
                }
            }
            _ => Err(Error::InvalidArgumentError(format!(
                "literal integer comparison not supported for {:?}",
                array.data_type()
            ))),
        },
        PlanValue::Float(expected) => match array.data_type() {
            DataType::Float32 => Ok(!array.is_null(row_idx)
                && (array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("float32 array")
                    .value(row_idx) as f64
                    - *expected)
                    .abs()
                    .eq(&0.0)),
            DataType::Float64 => Ok(!array.is_null(row_idx)
                && (array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("float64 array")
                    .value(row_idx)
                    - *expected)
                    .abs()
                    .eq(&0.0)),
            _ => Err(Error::InvalidArgumentError(format!(
                "literal float comparison not supported for {:?}",
                array.data_type()
            ))),
        },
        PlanValue::String(expected) => match array.data_type() {
            DataType::Utf8 => Ok(!array.is_null(row_idx)
                && array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("string array")
                    .value(row_idx)
                    == expected),
            DataType::LargeUtf8 => Ok(!array.is_null(row_idx)
                && array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("large string array")
                    .value(row_idx)
                    == expected),
            _ => Err(Error::InvalidArgumentError(format!(
                "literal string comparison not supported for {:?}",
                array.data_type()
            ))),
        },
        PlanValue::Date32(expected) => match array.data_type() {
            DataType::Date32 => Ok(!array.is_null(row_idx)
                && array
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .expect("date32 array")
                    .value(row_idx)
                    == *expected),
            _ => Err(Error::InvalidArgumentError(format!(
                "literal date comparison not supported for {:?}",
                array.data_type()
            ))),
        },
        PlanValue::Interval(expected) => {
            match array.data_type() {
                DataType::Interval(IntervalUnit::MonthDayNano) => {
                    if array.is_null(row_idx) {
                        Ok(false)
                    } else {
                        let value = array
                            .as_any()
                            .downcast_ref::<IntervalMonthDayNanoArray>()
                            .expect("interval array")
                            .value(row_idx);
                        let arrow_value = interval_value_from_arrow(value);
                        Ok(compare_interval_values(*expected, arrow_value)
                            == std::cmp::Ordering::Equal)
                    }
                }
                _ => Err(Error::InvalidArgumentError(format!(
                    "literal interval comparison not supported for {:?}",
                    array.data_type()
                ))),
            }
        }
        PlanValue::Struct(_) => Err(Error::InvalidArgumentError(
            "struct literals are not supported in join filters".into(),
        )),
    }
}

fn hash_join_table_batches(
    left: TableCrossProductData,
    right: TableCrossProductData,
    join_keys: &[(usize, usize)],
    join_type: llkv_join::JoinType,
) -> ExecutorResult<TableCrossProductData> {
    let TableCrossProductData {
        schema: left_schema,
        batches: left_batches,
        column_counts: left_counts,
        table_indices: left_tables,
    } = left;

    let TableCrossProductData {
        schema: right_schema,
        batches: right_batches,
        column_counts: right_counts,
        table_indices: right_tables,
    } = right;

    let combined_fields: Vec<Field> = left_schema
        .fields()
        .iter()
        .chain(right_schema.fields().iter())
        .map(|field| field.as_ref().clone())
        .collect();

    let combined_schema = Arc::new(Schema::new(combined_fields));

    let mut column_counts = Vec::with_capacity(left_counts.len() + right_counts.len());
    column_counts.extend(left_counts.iter());
    column_counts.extend(right_counts.iter());

    let mut table_indices = Vec::with_capacity(left_tables.len() + right_tables.len());
    table_indices.extend(left_tables.iter().copied());
    table_indices.extend(right_tables.iter().copied());

    // Handle empty inputs
    if left_batches.is_empty() {
        return Ok(TableCrossProductData {
            schema: combined_schema,
            batches: Vec::new(),
            column_counts,
            table_indices,
        });
    }

    if right_batches.is_empty() {
        // For LEFT JOIN with no right rows, return all left rows with NULL right columns
        if join_type == llkv_join::JoinType::Left {
            let total_left_rows: usize = left_batches.iter().map(|b| b.num_rows()).sum();
            let mut left_arrays = Vec::new();
            for field in left_schema.fields() {
                let column_idx = left_schema.index_of(field.name()).map_err(|e| {
                    Error::Internal(format!("failed to find field {}: {}", field.name(), e))
                })?;
                let arrays: Vec<ArrayRef> = left_batches
                    .iter()
                    .map(|batch| batch.column(column_idx).clone())
                    .collect();
                let concatenated =
                    arrow::compute::concat(&arrays.iter().map(|a| a.as_ref()).collect::<Vec<_>>())
                        .map_err(|e| {
                            Error::Internal(format!("failed to concat left arrays: {}", e))
                        })?;
                left_arrays.push(concatenated);
            }

            // Add NULL arrays for right side
            for field in right_schema.fields() {
                let null_array = arrow::array::new_null_array(field.data_type(), total_left_rows);
                left_arrays.push(null_array);
            }

            let joined_batch = RecordBatch::try_new(Arc::clone(&combined_schema), left_arrays)
                .map_err(|err| {
                    Error::Internal(format!(
                        "failed to create LEFT JOIN batch with NULL right: {err}"
                    ))
                })?;

            return Ok(TableCrossProductData {
                schema: combined_schema,
                batches: vec![joined_batch],
                column_counts,
                table_indices,
            });
        } else {
            // For INNER JOIN, no right rows means no results
            return Ok(TableCrossProductData {
                schema: combined_schema,
                batches: Vec::new(),
                column_counts,
                table_indices,
            });
        }
    }

    match join_type {
        llkv_join::JoinType::Inner => {
            let (left_matches, right_matches) =
                build_join_match_indices(&left_batches, &right_batches, join_keys)?;

            if left_matches.is_empty() {
                return Ok(TableCrossProductData {
                    schema: combined_schema,
                    batches: Vec::new(),
                    column_counts,
                    table_indices,
                });
            }

            let left_arrays = gather_indices_from_batches(&left_batches, &left_matches)?;
            let right_arrays = gather_indices_from_batches(&right_batches, &right_matches)?;

            let mut combined_columns = Vec::with_capacity(left_arrays.len() + right_arrays.len());
            combined_columns.extend(left_arrays);
            combined_columns.extend(right_arrays);

            let joined_batch = RecordBatch::try_new(Arc::clone(&combined_schema), combined_columns)
                .map_err(|err| {
                    Error::Internal(format!("failed to materialize INNER JOIN batch: {err}"))
                })?;

            Ok(TableCrossProductData {
                schema: combined_schema,
                batches: vec![joined_batch],
                column_counts,
                table_indices,
            })
        }
        llkv_join::JoinType::Left => {
            let (left_matches, right_optional_matches) =
                build_left_join_match_indices(&left_batches, &right_batches, join_keys)?;

            if left_matches.is_empty() {
                // This shouldn't happen for LEFT JOIN since all left rows should be included
                return Ok(TableCrossProductData {
                    schema: combined_schema,
                    batches: Vec::new(),
                    column_counts,
                    table_indices,
                });
            }

            let left_arrays = gather_indices_from_batches(&left_batches, &left_matches)?;
            // Use gather_optional_indices to handle None values (unmatched rows)
            let right_arrays = llkv_column_map::gather::gather_optional_indices_from_batches(
                &right_batches,
                &right_optional_matches,
            )?;

            let mut combined_columns = Vec::with_capacity(left_arrays.len() + right_arrays.len());
            combined_columns.extend(left_arrays);
            combined_columns.extend(right_arrays);

            let joined_batch = RecordBatch::try_new(Arc::clone(&combined_schema), combined_columns)
                .map_err(|err| {
                    Error::Internal(format!("failed to materialize LEFT JOIN batch: {err}"))
                })?;

            Ok(TableCrossProductData {
                schema: combined_schema,
                batches: vec![joined_batch],
                column_counts,
                table_indices,
            })
        }
        // Other join types not yet supported in this helper (delegate to llkv-join)
        _ => Err(Error::Internal(format!(
            "join type {:?} not supported in hash_join_table_batches; use llkv-join",
            join_type
        ))),
    }
}

/// Type alias for join match index pairs (batch_idx, row_idx)
type JoinMatchIndices = Vec<(usize, usize)>;
/// Type alias for hash table mapping join keys to row positions
type JoinHashTable = FxHashMap<Vec<u8>, Vec<(usize, usize)>>;
/// Type alias for complete match pairs for inner-style joins
type JoinMatchPairs = (JoinMatchIndices, JoinMatchIndices);
/// Type alias for optional matches produced by LEFT joins
type OptionalJoinMatches = Vec<Option<(usize, usize)>>;
/// Type alias for LEFT join match outputs
type LeftJoinMatchPairs = (JoinMatchIndices, OptionalJoinMatches);

fn normalize_join_column(array: &ArrayRef) -> ExecutorResult<ArrayRef> {
    match array.data_type() {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => cast(array, &DataType::Int64)
            .map_err(|e| Error::Internal(format!("failed to cast integer/boolean to Int64: {e}"))),
        DataType::Float32 => cast(array, &DataType::Float64)
            .map_err(|e| Error::Internal(format!("failed to cast Float32 to Float64: {e}"))),
        DataType::Utf8 | DataType::LargeUtf8 => cast(array, &DataType::LargeUtf8)
            .map_err(|e| Error::Internal(format!("failed to cast Utf8 to LargeUtf8: {e}"))),
        DataType::Dictionary(_, value_type) => {
            let unpacked = cast(array, value_type)
                .map_err(|e| Error::Internal(format!("failed to unpack dictionary: {e}")))?;
            normalize_join_column(&unpacked)
        }
        _ => Ok(array.clone()),
    }
}

/// Build hash join match indices using parallel hash table construction and probing.
///
/// Constructs a hash table from the right batches (build phase), then probes it with
/// rows from the left batches to find matches. Both phases are parallelized using Rayon.
///
/// # Parallelization Strategy
///
/// **Build Phase**: Each right batch is processed in parallel. Each thread builds a local
/// hash table for its batch(es), then all local tables are merged into a single shared
/// hash table. This eliminates lock contention during the build phase.
///
/// **Probe Phase**: Each left batch is probed against the shared hash table in parallel.
/// Each thread generates local match lists which are concatenated at the end.
///
/// # Arguments
///
/// * `left_batches` - Batches to probe against the hash table
/// * `right_batches` - Batches used to build the hash table
/// * `join_keys` - Column indices for join keys: (left_column_idx, right_column_idx)
///
/// # Returns
///
/// Tuple of `(left_matches, right_matches)` where each vector contains (batch_idx, row_idx)
/// pairs indicating which rows from left and right should be joined together.
///
/// # Performance
///
/// Scales with available CPU cores via `llkv_column_map::parallel::with_thread_pool()`.
/// Respects `LLKV_MAX_THREADS` environment variable for thread pool sizing.
fn build_join_match_indices(
    left_batches: &[RecordBatch],
    right_batches: &[RecordBatch],
    join_keys: &[(usize, usize)],
) -> ExecutorResult<JoinMatchPairs> {
    let right_key_indices: Vec<usize> = join_keys.iter().map(|(_, right)| *right).collect();

    // Parallelize hash table build phase across batches
    // Each thread builds a local hash table for its batch(es), then we merge them
    let hash_table: JoinHashTable = llkv_column_map::parallel::with_thread_pool(|| {
        let local_tables: Vec<ExecutorResult<JoinHashTable>> = right_batches
            .par_iter()
            .enumerate()
            .map(|(batch_idx, batch)| {
                let mut local_table: JoinHashTable = FxHashMap::default();

                let columns: Vec<ArrayRef> = right_key_indices
                    .iter()
                    .map(|&idx| normalize_join_column(batch.column(idx)))
                    .collect::<ExecutorResult<Vec<_>>>()?;

                let sort_fields: Vec<SortField> = columns
                    .iter()
                    .map(|c| SortField::new(c.data_type().clone()))
                    .collect();

                let converter = RowConverter::new(sort_fields)
                    .map_err(|e| Error::Internal(format!("failed to create RowConverter: {e}")))?;
                let rows = converter.convert_columns(&columns).map_err(|e| {
                    Error::Internal(format!("failed to convert columns to rows: {e}"))
                })?;

                for (row_idx, row) in rows.iter().enumerate() {
                    // Skip rows with NULLs in join keys (standard SQL behavior)
                    if columns.iter().any(|c| c.is_null(row_idx)) {
                        continue;
                    }

                    local_table
                        .entry(row.as_ref().to_vec())
                        .or_default()
                        .push((batch_idx, row_idx));
                }

                Ok(local_table)
            })
            .collect();

        // Merge all local hash tables into one
        let mut merged_table: JoinHashTable = FxHashMap::default();
        for local_table_res in local_tables {
            if let Ok(local_table) = local_table_res {
                for (key, mut positions) in local_table {
                    merged_table.entry(key).or_default().append(&mut positions);
                }
            } else {
                tracing::error!("failed to build hash table for batch");
            }
        }

        merged_table
    });

    if hash_table.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let left_key_indices: Vec<usize> = join_keys.iter().map(|(left, _)| *left).collect();

    // Parallelize probe phase across left batches
    // Each thread probes its batch(es) against the shared hash table
    let matches: Vec<ExecutorResult<JoinMatchPairs>> =
        llkv_column_map::parallel::with_thread_pool(|| {
            left_batches
                .par_iter()
                .enumerate()
                .map(|(batch_idx, batch)| {
                    let mut local_left_matches: JoinMatchIndices = Vec::new();
                    let mut local_right_matches: JoinMatchIndices = Vec::new();

                    let columns: Vec<ArrayRef> = left_key_indices
                        .iter()
                        .map(|&idx| normalize_join_column(batch.column(idx)))
                        .collect::<ExecutorResult<Vec<_>>>()?;

                    let sort_fields: Vec<SortField> = columns
                        .iter()
                        .map(|c| SortField::new(c.data_type().clone()))
                        .collect();

                    let converter = RowConverter::new(sort_fields).map_err(|e| {
                        Error::Internal(format!("failed to create RowConverter: {e}"))
                    })?;
                    let rows = converter.convert_columns(&columns).map_err(|e| {
                        Error::Internal(format!("failed to convert columns to rows: {e}"))
                    })?;

                    for (row_idx, row) in rows.iter().enumerate() {
                        if columns.iter().any(|c| c.is_null(row_idx)) {
                            continue;
                        }

                        if let Some(positions) = hash_table.get(row.as_ref()) {
                            for &(r_batch_idx, r_row_idx) in positions {
                                local_left_matches.push((batch_idx, row_idx));
                                local_right_matches.push((r_batch_idx, r_row_idx));
                            }
                        }
                    }

                    Ok((local_left_matches, local_right_matches))
                })
                .collect()
        });

    // Merge all match results
    let mut left_matches: JoinMatchIndices = Vec::new();
    let mut right_matches: JoinMatchIndices = Vec::new();
    for match_res in matches {
        let (mut left, mut right) = match_res?;
        left_matches.append(&mut left);
        right_matches.append(&mut right);
    }

    Ok((left_matches, right_matches))
}

/// Build match indices for LEFT JOIN, returning all left rows with optional right matches.
///
/// Unlike `build_join_match_indices` which only returns matching pairs, this function
/// returns every left row. For rows with no match, the right match is `None`.
///
/// # Returns
///
/// Tuple of `(left_matches, right_optional_matches)` where:
/// - `left_matches`: (batch_idx, row_idx) for every left row
/// - `right_optional_matches`: `Some((batch_idx, row_idx))` for matched rows, `None` for unmatched
fn build_left_join_match_indices(
    left_batches: &[RecordBatch],
    right_batches: &[RecordBatch],
    join_keys: &[(usize, usize)],
) -> ExecutorResult<LeftJoinMatchPairs> {
    let right_key_indices: Vec<usize> = join_keys.iter().map(|(_, right)| *right).collect();

    // Build hash table from right batches
    let hash_table: JoinHashTable = llkv_column_map::parallel::with_thread_pool(|| {
        let local_tables: Vec<JoinHashTable> = right_batches
            .par_iter()
            .enumerate()
            .map(|(batch_idx, batch)| {
                let mut local_table: JoinHashTable = FxHashMap::default();
                let mut key_buffer: Vec<u8> = Vec::new();

                for row_idx in 0..batch.num_rows() {
                    key_buffer.clear();
                    match build_join_key(batch, &right_key_indices, row_idx, &mut key_buffer) {
                        Ok(true) => {
                            local_table
                                .entry(key_buffer.clone())
                                .or_default()
                                .push((batch_idx, row_idx));
                        }
                        Ok(false) => continue,
                        Err(_) => continue,
                    }
                }

                local_table
            })
            .collect();

        let mut merged_table: JoinHashTable = FxHashMap::default();
        for local_table in local_tables {
            for (key, mut positions) in local_table {
                merged_table.entry(key).or_default().append(&mut positions);
            }
        }

        merged_table
    });

    let left_key_indices: Vec<usize> = join_keys.iter().map(|(left, _)| *left).collect();

    // Probe phase: process ALL left rows, recording matches or None
    let matches: Vec<LeftJoinMatchPairs> = llkv_column_map::parallel::with_thread_pool(|| {
        left_batches
            .par_iter()
            .enumerate()
            .map(|(batch_idx, batch)| {
                let mut local_left_matches: JoinMatchIndices = Vec::new();
                let mut local_right_optional: Vec<Option<(usize, usize)>> = Vec::new();
                let mut key_buffer: Vec<u8> = Vec::new();

                for row_idx in 0..batch.num_rows() {
                    key_buffer.clear();
                    match build_join_key(batch, &left_key_indices, row_idx, &mut key_buffer) {
                        Ok(true) => {
                            if let Some(entries) = hash_table.get(&key_buffer) {
                                // Has matches - emit one output row per match
                                for &(r_batch, r_row) in entries {
                                    local_left_matches.push((batch_idx, row_idx));
                                    local_right_optional.push(Some((r_batch, r_row)));
                                }
                            } else {
                                // No match - emit left row with NULL right
                                local_left_matches.push((batch_idx, row_idx));
                                local_right_optional.push(None);
                            }
                        }
                        Ok(false) => {
                            // NULL key on left side - no match, emit with NULL right
                            local_left_matches.push((batch_idx, row_idx));
                            local_right_optional.push(None);
                        }
                        Err(_) => {
                            // Error reading key - treat as no match
                            local_left_matches.push((batch_idx, row_idx));
                            local_right_optional.push(None);
                        }
                    }
                }

                (local_left_matches, local_right_optional)
            })
            .collect()
    });

    // Merge all match results
    let mut left_matches: JoinMatchIndices = Vec::new();
    let mut right_optional: Vec<Option<(usize, usize)>> = Vec::new();
    for (mut left, mut right) in matches {
        left_matches.append(&mut left);
        right_optional.append(&mut right);
    }

    Ok((left_matches, right_optional))
}

fn build_join_key(
    batch: &RecordBatch,
    column_indices: &[usize],
    row_idx: usize,
    buffer: &mut Vec<u8>,
) -> ExecutorResult<bool> {
    buffer.clear();

    for &col_idx in column_indices {
        let array = batch.column(col_idx);
        if array.is_null(row_idx) {
            return Ok(false);
        }
        append_array_value_to_key(array.as_ref(), row_idx, buffer)?;
    }

    Ok(true)
}

fn append_array_value_to_key(
    array: &dyn Array,
    row_idx: usize,
    buffer: &mut Vec<u8>,
) -> ExecutorResult<()> {
    use arrow::array::*;
    use arrow::datatypes::DataType;

    match array.data_type() {
        DataType::Int8 => buffer.extend_from_slice(
            &array
                .as_any()
                .downcast_ref::<Int8Array>()
                .expect("int8 array")
                .value(row_idx)
                .to_le_bytes(),
        ),
        DataType::Int16 => buffer.extend_from_slice(
            &array
                .as_any()
                .downcast_ref::<Int16Array>()
                .expect("int16 array")
                .value(row_idx)
                .to_le_bytes(),
        ),
        DataType::Int32 => buffer.extend_from_slice(
            &array
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("int32 array")
                .value(row_idx)
                .to_le_bytes(),
        ),
        DataType::Int64 => buffer.extend_from_slice(
            &array
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("int64 array")
                .value(row_idx)
                .to_le_bytes(),
        ),
        DataType::UInt8 => buffer.extend_from_slice(
            &array
                .as_any()
                .downcast_ref::<UInt8Array>()
                .expect("uint8 array")
                .value(row_idx)
                .to_le_bytes(),
        ),
        DataType::UInt16 => buffer.extend_from_slice(
            &array
                .as_any()
                .downcast_ref::<UInt16Array>()
                .expect("uint16 array")
                .value(row_idx)
                .to_le_bytes(),
        ),
        DataType::UInt32 => buffer.extend_from_slice(
            &array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .expect("uint32 array")
                .value(row_idx)
                .to_le_bytes(),
        ),
        DataType::UInt64 => buffer.extend_from_slice(
            &array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("uint64 array")
                .value(row_idx)
                .to_le_bytes(),
        ),
        DataType::Float32 => buffer.extend_from_slice(
            &array
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("float32 array")
                .value(row_idx)
                .to_le_bytes(),
        ),
        DataType::Float64 => buffer.extend_from_slice(
            &array
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("float64 array")
                .value(row_idx)
                .to_le_bytes(),
        ),
        DataType::Boolean => buffer.push(
            array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("bool array")
                .value(row_idx) as u8,
        ),
        DataType::Utf8 => {
            let value = array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("utf8 array")
                .value(row_idx);
            buffer.extend_from_slice(&(value.len() as u32).to_le_bytes());
            buffer.extend_from_slice(value.as_bytes());
        }
        DataType::LargeUtf8 => {
            let value = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("large utf8 array")
                .value(row_idx);
            buffer.extend_from_slice(&(value.len() as u32).to_le_bytes());
            buffer.extend_from_slice(value.as_bytes());
        }
        DataType::Binary => {
            let value = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .expect("binary array")
                .value(row_idx);
            buffer.extend_from_slice(&(value.len() as u32).to_le_bytes());
            buffer.extend_from_slice(value);
        }
        other => {
            return Err(Error::InvalidArgumentError(format!(
                "hash join does not support join key type {:?}",
                other
            )));
        }
    }

    Ok(())
}

fn table_has_join_with_used(
    candidate: usize,
    used_tables: &FxHashSet<usize>,
    equalities: &[ColumnEquality],
) -> bool {
    equalities.iter().any(|equality| {
        (equality.left.table == candidate && used_tables.contains(&equality.right.table))
            || (equality.right.table == candidate && used_tables.contains(&equality.left.table))
    })
}

fn gather_join_keys(
    left: &TableCrossProductData,
    right: &TableCrossProductData,
    used_tables: &FxHashSet<usize>,
    right_table_index: usize,
    equalities: &[ColumnEquality],
) -> ExecutorResult<Vec<(usize, usize)>> {
    let mut keys = Vec::new();

    for equality in equalities {
        if equality.left.table == right_table_index && used_tables.contains(&equality.right.table) {
            let left_idx = resolve_column_index(left, &equality.right).ok_or_else(|| {
                Error::Internal("failed to resolve column offset for hash join".into())
            })?;
            let right_idx = resolve_column_index(right, &equality.left).ok_or_else(|| {
                Error::Internal("failed to resolve column offset for hash join".into())
            })?;
            keys.push((left_idx, right_idx));
        } else if equality.right.table == right_table_index
            && used_tables.contains(&equality.left.table)
        {
            let left_idx = resolve_column_index(left, &equality.left).ok_or_else(|| {
                Error::Internal("failed to resolve column offset for hash join".into())
            })?;
            let right_idx = resolve_column_index(right, &equality.right).ok_or_else(|| {
                Error::Internal("failed to resolve column offset for hash join".into())
            })?;
            keys.push((left_idx, right_idx));
        }
    }

    Ok(keys)
}

fn resolve_column_index(data: &TableCrossProductData, column: &ColumnRef) -> Option<usize> {
    let mut offset = 0;
    for (table_idx, count) in data.table_indices.iter().zip(data.column_counts.iter()) {
        if *table_idx == column.table {
            if column.column < *count {
                return Some(offset + column.column);
            } else {
                return None;
            }
        }
        offset += count;
    }
    None
}

fn build_cross_product_column_lookup(
    schema: &Schema,
    tables: &[llkv_plan::TableRef],
    column_counts: &[usize],
    table_indices: &[usize],
) -> FxHashMap<String, usize> {
    debug_assert_eq!(tables.len(), column_counts.len());
    debug_assert_eq!(column_counts.len(), table_indices.len());

    let mut column_occurrences: FxHashMap<String, usize> = FxHashMap::default();
    let mut table_column_counts: FxHashMap<String, usize> = FxHashMap::default();
    for field in schema.fields() {
        let column_name = extract_column_name(field.name());
        *column_occurrences.entry(column_name).or_insert(0) += 1;
        if let Some(pair) = table_column_suffix(field.name()) {
            *table_column_counts.entry(pair).or_insert(0) += 1;
        }
    }

    let mut base_table_totals: FxHashMap<String, usize> = FxHashMap::default();
    let mut base_table_unaliased: FxHashMap<String, usize> = FxHashMap::default();
    for table_ref in tables {
        let key = base_table_key(table_ref);
        *base_table_totals.entry(key.clone()).or_insert(0) += 1;
        if table_ref.alias.is_none() {
            *base_table_unaliased.entry(key).or_insert(0) += 1;
        }
    }

    let mut lookup = FxHashMap::default();

    if table_indices.is_empty() || column_counts.is_empty() {
        for (idx, field) in schema.fields().iter().enumerate() {
            let field_name_lower = field.name().to_ascii_lowercase();
            lookup.entry(field_name_lower).or_insert(idx);

            let trimmed_lower = field.name().trim_start_matches('.').to_ascii_lowercase();
            lookup.entry(trimmed_lower).or_insert(idx);

            if let Some(pair) = table_column_suffix(field.name())
                && table_column_counts.get(&pair).copied().unwrap_or(0) == 1
            {
                lookup.entry(pair).or_insert(idx);
            }

            let column_name = extract_column_name(field.name());
            if column_occurrences.get(&column_name).copied().unwrap_or(0) == 1 {
                lookup.entry(column_name).or_insert(idx);
            }
        }
        return lookup;
    }

    let mut offset = 0usize;
    for (&table_idx, &count) in table_indices.iter().zip(column_counts.iter()) {
        if table_idx >= tables.len() {
            continue;
        }
        let table_ref = &tables[table_idx];
        let alias_lower = table_ref
            .alias
            .as_ref()
            .map(|alias| alias.to_ascii_lowercase());
        let table_lower = table_ref.table.to_ascii_lowercase();
        let schema_lower = table_ref.schema.to_ascii_lowercase();
        let base_key = base_table_key(table_ref);
        let total_refs = base_table_totals.get(&base_key).copied().unwrap_or(0);
        let unaliased_refs = base_table_unaliased.get(&base_key).copied().unwrap_or(0);

        let allow_base_mapping = if table_ref.alias.is_none() {
            unaliased_refs == 1
        } else {
            unaliased_refs == 0 && total_refs == 1
        };

        let mut table_keys: Vec<String> = Vec::new();

        if let Some(alias) = &alias_lower {
            table_keys.push(alias.clone());
            if !schema_lower.is_empty() {
                table_keys.push(format!("{}.{}", schema_lower, alias));
            }
        }

        if allow_base_mapping {
            table_keys.push(table_lower.clone());
            if !schema_lower.is_empty() {
                table_keys.push(format!("{}.{}", schema_lower, table_lower));
            }
        }

        for local_idx in 0..count {
            let field_index = offset + local_idx;
            let field = schema.field(field_index);
            let field_name_lower = field.name().to_ascii_lowercase();
            lookup.entry(field_name_lower).or_insert(field_index);

            let trimmed_lower = field.name().trim_start_matches('.').to_ascii_lowercase();
            lookup.entry(trimmed_lower).or_insert(field_index);

            let column_name = extract_column_name(field.name());
            for table_key in &table_keys {
                lookup
                    .entry(format!("{}.{}", table_key, column_name))
                    .or_insert(field_index);
            }

            // Use first-match semantics for bare column names (matches SQLite behavior)
            // This allows ambiguous column names to resolve to the first occurrence
            // in FROM clause order
            lookup.entry(column_name.clone()).or_insert(field_index);

            if table_keys.is_empty()
                && let Some(pair) = table_column_suffix(field.name())
                && table_column_counts.get(&pair).copied().unwrap_or(0) == 1
            {
                lookup.entry(pair).or_insert(field_index);
            }
        }

        offset = offset.saturating_add(count);
    }

    lookup
}

fn base_table_key(table_ref: &llkv_plan::TableRef) -> String {
    let schema_lower = table_ref.schema.to_ascii_lowercase();
    let table_lower = table_ref.table.to_ascii_lowercase();
    if schema_lower.is_empty() {
        table_lower
    } else {
        format!("{}.{}", schema_lower, table_lower)
    }
}

fn extract_column_name(name: &str) -> String {
    name.trim_start_matches('.')
        .rsplit('.')
        .next()
        .unwrap_or(name)
        .to_ascii_lowercase()
}

fn table_column_suffix(name: &str) -> Option<String> {
    let trimmed = name.trim_start_matches('.');
    let mut parts: Vec<&str> = trimmed.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    let column = parts.pop()?.to_ascii_lowercase();
    let table = parts.pop()?.to_ascii_lowercase();
    Some(format!("{}.{}", table, column))
}

/// Combine two table batch sets into a cartesian product using parallel processing.
///
/// For each pair of (left_batch, right_batch), generates the cross product using
/// [`llkv_join::cross_join_pair`]. The computation is parallelized across all batch
/// pairs since they are independent.
///
/// # Parallelization
///
/// Uses nested parallel iteration via Rayon:
/// - Outer loop: parallel iteration over left batches
/// - Inner loop: parallel iteration over right batches
/// - Each (left, right) pair is processed independently
///
/// This effectively distributes N×M batch pairs across available CPU cores, providing
/// significant speedup for multi-batch joins.
///
/// # Arguments
///
/// * `left` - Left side table data with batches
/// * `right` - Right side table data with batches
///
/// # Returns
///
/// Combined table data containing the cartesian product of all left and right rows.
fn cross_join_table_batches(
    left: TableCrossProductData,
    right: TableCrossProductData,
) -> ExecutorResult<TableCrossProductData> {
    let TableCrossProductData {
        schema: left_schema,
        batches: left_batches,
        column_counts: mut left_counts,
        table_indices: mut left_tables,
    } = left;
    let TableCrossProductData {
        schema: right_schema,
        batches: right_batches,
        column_counts: right_counts,
        table_indices: right_tables,
    } = right;

    let combined_fields: Vec<Field> = left_schema
        .fields()
        .iter()
        .chain(right_schema.fields().iter())
        .map(|field| field.as_ref().clone())
        .collect();

    let mut column_counts = Vec::with_capacity(left_counts.len() + right_counts.len());
    column_counts.append(&mut left_counts);
    column_counts.extend(right_counts);

    let mut table_indices = Vec::with_capacity(left_tables.len() + right_tables.len());
    table_indices.append(&mut left_tables);
    table_indices.extend(right_tables);

    let combined_schema = Arc::new(Schema::new(combined_fields));

    let left_has_rows = left_batches.iter().any(|batch| batch.num_rows() > 0);
    let right_has_rows = right_batches.iter().any(|batch| batch.num_rows() > 0);

    if !left_has_rows || !right_has_rows {
        return Ok(TableCrossProductData {
            schema: combined_schema,
            batches: Vec::new(),
            column_counts,
            table_indices,
        });
    }

    // Parallelize cross join batch generation using nested parallel iteration
    // This is safe because cross_join_pair is pure and each batch pair is independent
    let output_batches: Vec<RecordBatch> = llkv_column_map::parallel::with_thread_pool(|| {
        left_batches
            .par_iter()
            .filter(|left_batch| left_batch.num_rows() > 0)
            .flat_map(|left_batch| {
                right_batches
                    .par_iter()
                    .filter(|right_batch| right_batch.num_rows() > 0)
                    .filter_map(|right_batch| {
                        cross_join_pair(left_batch, right_batch, &combined_schema).ok()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    });

    Ok(TableCrossProductData {
        schema: combined_schema,
        batches: output_batches,
        column_counts,
        table_indices,
    })
}

fn cross_join_all(staged: Vec<TableCrossProductData>) -> ExecutorResult<TableCrossProductData> {
    let mut iter = staged.into_iter();
    let mut current = iter
        .next()
        .ok_or_else(|| Error::Internal("cross product preparation yielded no tables".into()))?;
    for next in iter {
        current = cross_join_table_batches(current, next)?;
    }
    Ok(current)
}

struct TableInfo<'a> {
    index: usize,
    table_ref: &'a llkv_plan::TableRef,
    column_map: FxHashMap<String, usize>,
}

#[derive(Clone, Copy)]
struct ColumnRef {
    table: usize,
    column: usize,
}

#[derive(Clone, Copy)]
struct ColumnEquality {
    left: ColumnRef,
    right: ColumnRef,
}

#[derive(Clone)]
struct ColumnLiteral {
    column: ColumnRef,
    value: PlanValue,
}

#[derive(Clone)]
struct ColumnInList {
    column: ColumnRef,
    values: Vec<PlanValue>,
}

#[derive(Clone)]
enum ColumnConstraint {
    Equality(ColumnLiteral),
    InList(ColumnInList),
}

// TODO: Move `llkv-plan`?
struct JoinConstraintPlan {
    equalities: Vec<ColumnEquality>,
    literals: Vec<ColumnConstraint>,
    unsatisfiable: bool,
    /// Total number of conjuncts in the original WHERE clause
    total_conjuncts: usize,
    /// Number of conjuncts successfully handled (as equalities or literals)
    handled_conjuncts: usize,
}

/// Extract literal pushdown filters from a WHERE clause, even in the presence of OR clauses.
///
/// Unlike `extract_join_constraints`, this function is more lenient and extracts column-to-literal
/// comparisons and IN-list predicates regardless of OR clauses. This allows selective table scans
/// even when hash join optimization cannot be applied.
///
/// # Strategy
///
/// Extracts top-level AND-connected predicates:
/// - Column-to-literal equalities (e.g., `c2 = 374`)
/// - IN-list predicates (e.g., `b4 IN (408, 261, 877, 33)`)
///
/// OR clauses and other complex predicates are left for post-join filtering.
///
/// # Returns
///
/// A vector indexed by table position, where each element contains the constraints
/// that can be pushed down to that table.
fn extract_literal_pushdown_filters<P>(
    expr: &LlkvExpr<'static, String>,
    tables_with_handles: &[(llkv_plan::TableRef, Arc<ExecutorTable<P>>)],
) -> Vec<Vec<ColumnConstraint>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut table_infos = Vec::with_capacity(tables_with_handles.len());
    for (index, (table_ref, executor_table)) in tables_with_handles.iter().enumerate() {
        let mut column_map = FxHashMap::default();
        for (column_idx, column) in executor_table.schema.columns.iter().enumerate() {
            let column_name = column.name.to_ascii_lowercase();
            column_map.entry(column_name).or_insert(column_idx);
        }
        table_infos.push(TableInfo {
            index,
            table_ref,
            column_map,
        });
    }

    let mut constraints: Vec<Vec<ColumnConstraint>> = vec![Vec::new(); tables_with_handles.len()];

    // Collect all conjuncts, but be lenient about OR clauses - we'll skip them
    let mut conjuncts = Vec::new();
    collect_conjuncts_lenient(expr, &mut conjuncts);

    for conjunct in conjuncts {
        // Handle Compare expressions: column = literal
        if let LlkvExpr::Compare {
            left,
            op: CompareOp::Eq,
            right,
        } = conjunct
        {
            match (
                resolve_column_reference(left, &table_infos),
                resolve_column_reference(right, &table_infos),
            ) {
                (Some(column), None) => {
                    if let Some(literal) = extract_literal(right)
                        && let Some(value) = PlanValue::from_literal_for_join(literal)
                        && column.table < constraints.len()
                    {
                        constraints[column.table]
                            .push(ColumnConstraint::Equality(ColumnLiteral { column, value }));
                    }
                }
                (None, Some(column)) => {
                    if let Some(literal) = extract_literal(left)
                        && let Some(value) = PlanValue::from_literal_for_join(literal)
                        && column.table < constraints.len()
                    {
                        constraints[column.table]
                            .push(ColumnConstraint::Equality(ColumnLiteral { column, value }));
                    }
                }
                _ => {}
            }
        }
        // Handle Pred(Filter) expressions: these are already in filter form
        // We extract simple equality predicates only
        else if let LlkvExpr::Pred(filter) = conjunct {
            if let Operator::Equals(ref literal_val) = filter.op {
                // field_id is the column name in string form
                let field_name = filter.field_id.trim().to_ascii_lowercase();

                // Try to find which table this column belongs to
                for info in &table_infos {
                    if let Some(&col_idx) = info.column_map.get(&field_name) {
                        if let Some(value) = PlanValue::from_operator_literal(literal_val) {
                            let column_ref = ColumnRef {
                                table: info.index,
                                column: col_idx,
                            };
                            if info.index < constraints.len() {
                                constraints[info.index].push(ColumnConstraint::Equality(
                                    ColumnLiteral {
                                        column: column_ref,
                                        value,
                                    },
                                ));
                            }
                        }
                        break; // Found the column, no need to check other tables
                    }
                }
            }
        }
        // Handle InList expressions: column IN (val1, val2, ...)
        else if let LlkvExpr::InList {
            expr: col_expr,
            list,
            negated: false,
        } = conjunct
        {
            if let Some(column) = resolve_column_reference(col_expr, &table_infos) {
                let mut values = Vec::new();
                for item in list {
                    if let Some(literal) = extract_literal(item)
                        && let Some(value) = PlanValue::from_literal_for_join(literal)
                    {
                        values.push(value);
                    }
                }
                if !values.is_empty() && column.table < constraints.len() {
                    constraints[column.table]
                        .push(ColumnConstraint::InList(ColumnInList { column, values }));
                }
            }
        }
        // Handle OR expressions: try to convert (col=v1 OR col=v2) into col IN (v1, v2)
        else if let LlkvExpr::Or(or_children) = conjunct
            && let Some((column, values)) = try_extract_or_as_in_list(or_children, &table_infos)
            && !values.is_empty()
            && column.table < constraints.len()
        {
            constraints[column.table]
                .push(ColumnConstraint::InList(ColumnInList { column, values }));
        }
    }

    constraints
}

/// Collect conjuncts from an expression, including OR clauses for potential conversion.
///
/// Unlike `collect_conjuncts`, this function doesn't bail out on OR - it includes OR clauses
/// in the output so they can be analyzed for conversion to IN lists: `(col=v1 OR col=v2)` → `col IN (v1, v2)`.
fn collect_conjuncts_lenient<'a>(
    expr: &'a LlkvExpr<'static, String>,
    out: &mut Vec<&'a LlkvExpr<'static, String>>,
) {
    match expr {
        LlkvExpr::And(children) => {
            for child in children {
                collect_conjuncts_lenient(child, out);
            }
        }
        other => {
            // Include all non-AND expressions (including OR) for analysis
            out.push(other);
        }
    }
}

/// Attempt to extract an OR clause as an IN list if it matches the pattern (col=v1 OR col=v2 OR ...).
///
/// Returns Some((column_ref, values)) if successful, None otherwise.
fn try_extract_or_as_in_list(
    or_children: &[LlkvExpr<'static, String>],
    table_infos: &[TableInfo<'_>],
) -> Option<(ColumnRef, Vec<PlanValue>)> {
    if or_children.is_empty() {
        return None;
    }

    let mut common_column: Option<ColumnRef> = None;
    let mut values = Vec::new();

    for child in or_children {
        // Try to extract column = literal pattern from Compare expressions
        if let LlkvExpr::Compare {
            left,
            op: CompareOp::Eq,
            right,
        } = child
        {
            // Try col = literal
            if let (Some(column), None) = (
                resolve_column_reference(left, table_infos),
                resolve_column_reference(right, table_infos),
            ) && let Some(literal) = extract_literal(right)
                && let Some(value) = PlanValue::from_literal_for_join(literal)
            {
                // Check if this is the same column as previous OR branches
                match common_column {
                    None => common_column = Some(column),
                    Some(ref prev)
                        if prev.table == column.table && prev.column == column.column =>
                    {
                        // Same column, continue
                    }
                    _ => {
                        // Different column - OR cannot be converted to IN list
                        return None;
                    }
                }
                values.push(value);
                continue;
            }

            // Try literal = col
            if let (None, Some(column)) = (
                resolve_column_reference(left, table_infos),
                resolve_column_reference(right, table_infos),
            ) && let Some(literal) = extract_literal(left)
                && let Some(value) = PlanValue::from_literal_for_join(literal)
            {
                match common_column {
                    None => common_column = Some(column),
                    Some(ref prev)
                        if prev.table == column.table && prev.column == column.column => {}
                    _ => return None,
                }
                values.push(value);
                continue;
            }
        }
        // Also handle Pred(Filter{...}) expressions with Equals operator
        else if let LlkvExpr::Pred(filter) = child
            && let Operator::Equals(ref literal) = filter.op
            && let Some(column) =
                resolve_column_reference(&ScalarExpr::Column(filter.field_id.clone()), table_infos)
            && let Some(value) = PlanValue::from_literal_for_join(literal)
        {
            match common_column {
                None => common_column = Some(column),
                Some(ref prev) if prev.table == column.table && prev.column == column.column => {}
                _ => return None,
            }
            values.push(value);
            continue;
        }

        // If any branch doesn't match the pattern, OR cannot be converted
        return None;
    }

    common_column.map(|col| (col, values))
}

/// Extract join constraints from a WHERE clause predicate for hash join optimization.
///
/// Analyzes the predicate to identify:
/// - **Equality constraints**: column-to-column equalities for hash join keys
/// - **Literal constraints**: column-to-literal comparisons that can be pushed down
/// - **Unsatisfiable conditions**: `WHERE false` that makes result set empty
///
/// Returns `None` if the predicate structure is too complex for optimization (e.g.,
/// contains OR, NOT, or other non-conjunctive patterns).
///
/// # Partial Handling
///
/// The optimizer tracks `handled_conjuncts` vs `total_conjuncts`. If some predicates
/// cannot be optimized (e.g., complex expressions, unsupported operators), they are
/// left for post-join filtering. This allows partial optimization rather than falling
/// back to full cartesian product.
///
/// # Arguments
///
/// * `expr` - WHERE clause expression to analyze
/// * `table_infos` - Metadata about tables in the query (for column resolution)
///
/// # Returns
///
/// * `Some(JoinConstraintPlan)` - Successfully extracted constraints
/// * `None` - Predicate cannot be optimized (fall back to cartesian product)
fn extract_join_constraints(
    expr: &LlkvExpr<'static, String>,
    table_infos: &[TableInfo<'_>],
) -> Option<JoinConstraintPlan> {
    let mut conjuncts = Vec::new();
    // Use lenient collection to include OR clauses for potential conversion to IN lists
    collect_conjuncts_lenient(expr, &mut conjuncts);

    let total_conjuncts = conjuncts.len();
    let mut equalities = Vec::new();
    let mut literals = Vec::new();
    let mut unsatisfiable = false;
    let mut handled_conjuncts = 0;

    for conjunct in conjuncts {
        match conjunct {
            LlkvExpr::Literal(true) => {
                handled_conjuncts += 1;
            }
            LlkvExpr::Literal(false) => {
                unsatisfiable = true;
                handled_conjuncts += 1;
                break;
            }
            LlkvExpr::Compare {
                left,
                op: CompareOp::Eq,
                right,
            } => {
                match (
                    resolve_column_reference(left, table_infos),
                    resolve_column_reference(right, table_infos),
                ) {
                    (Some(left_col), Some(right_col)) => {
                        equalities.push(ColumnEquality {
                            left: left_col,
                            right: right_col,
                        });
                        handled_conjuncts += 1;
                        continue;
                    }
                    (Some(column), None) => {
                        if let Some(literal) = extract_literal(right)
                            && let Some(value) = PlanValue::from_literal_for_join(literal)
                        {
                            literals
                                .push(ColumnConstraint::Equality(ColumnLiteral { column, value }));
                            handled_conjuncts += 1;
                            continue;
                        }
                    }
                    (None, Some(column)) => {
                        if let Some(literal) = extract_literal(left)
                            && let Some(value) = PlanValue::from_literal_for_join(literal)
                        {
                            literals
                                .push(ColumnConstraint::Equality(ColumnLiteral { column, value }));
                            handled_conjuncts += 1;
                            continue;
                        }
                    }
                    _ => {}
                }
                // Ignore this predicate - it will be handled by post-join filter
            }
            // Handle InList - these can be used for hash join build side filtering
            LlkvExpr::InList {
                expr: col_expr,
                list,
                negated: false,
            } => {
                if let Some(column) = resolve_column_reference(col_expr, table_infos) {
                    // Extract all values from IN list
                    let mut in_list_values = Vec::new();
                    for item in list {
                        if let Some(literal) = extract_literal(item)
                            && let Some(value) = PlanValue::from_literal_for_join(literal)
                        {
                            in_list_values.push(value);
                        }
                    }
                    if !in_list_values.is_empty() {
                        literals.push(ColumnConstraint::InList(ColumnInList {
                            column,
                            values: in_list_values,
                        }));
                        handled_conjuncts += 1;
                        continue;
                    }
                }
                // Ignore - will be handled by post-join filter
            }
            // Handle OR clauses that can be converted to IN lists
            LlkvExpr::Or(or_children) => {
                if let Some((column, values)) = try_extract_or_as_in_list(or_children, table_infos)
                {
                    // Treat as IN list
                    literals.push(ColumnConstraint::InList(ColumnInList { column, values }));
                    handled_conjuncts += 1;
                    continue;
                }
                // OR clause couldn't be converted - ignore, will be handled by post-join filter
            }
            // Handle Pred(Filter{...}) expressions - these are field-based predicates
            LlkvExpr::Pred(filter) => {
                // Try to extract equality constraints
                if let Operator::Equals(ref literal) = filter.op
                    && let Some(column) = resolve_column_reference(
                        &ScalarExpr::Column(filter.field_id.clone()),
                        table_infos,
                    )
                    && let Some(value) = PlanValue::from_literal_for_join(literal)
                {
                    literals.push(ColumnConstraint::Equality(ColumnLiteral { column, value }));
                    handled_conjuncts += 1;
                    continue;
                }
                // Ignore other Pred expressions - will be handled by post-join filter
            }
            _ => {
                // Ignore unsupported predicates - they will be handled by post-join filter
            }
        }
    }

    Some(JoinConstraintPlan {
        equalities,
        literals,
        unsatisfiable,
        total_conjuncts,
        handled_conjuncts,
    })
}

fn resolve_column_reference(
    expr: &ScalarExpr<String>,
    table_infos: &[TableInfo<'_>],
) -> Option<ColumnRef> {
    let name = match expr {
        ScalarExpr::Column(name) => name.trim(),
        _ => return None,
    };

    let mut parts: Vec<&str> = name
        .trim_start_matches('.')
        .split('.')
        .filter(|segment| !segment.is_empty())
        .collect();

    if parts.is_empty() {
        return None;
    }

    let column_part = parts.pop()?.to_ascii_lowercase();
    if parts.is_empty() {
        // Use first-match semantics for bare column names (matches SQLite behavior)
        // This allows ambiguous column names in WHERE clauses to resolve to the
        // first occurrence in FROM clause order
        for info in table_infos {
            if let Some(&col_idx) = info.column_map.get(&column_part) {
                return Some(ColumnRef {
                    table: info.index,
                    column: col_idx,
                });
            }
        }
        return None;
    }

    let table_ident = parts.join(".").to_ascii_lowercase();
    for info in table_infos {
        if matches_table_ident(info.table_ref, &table_ident) {
            if let Some(&col_idx) = info.column_map.get(&column_part) {
                return Some(ColumnRef {
                    table: info.index,
                    column: col_idx,
                });
            } else {
                return None;
            }
        }
    }
    None
}

fn matches_table_ident(table_ref: &llkv_plan::TableRef, ident: &str) -> bool {
    if ident.is_empty() {
        return false;
    }
    if let Some(alias) = &table_ref.alias
        && alias.to_ascii_lowercase() == ident
    {
        return true;
    }
    if table_ref.table.to_ascii_lowercase() == ident {
        return true;
    }
    if !table_ref.schema.is_empty() {
        let full = format!(
            "{}.{}",
            table_ref.schema.to_ascii_lowercase(),
            table_ref.table.to_ascii_lowercase()
        );
        if full == ident {
            return true;
        }
    }
    false
}

fn extract_literal(expr: &ScalarExpr<String>) -> Option<&Literal> {
    match expr {
        ScalarExpr::Literal(lit) => Some(lit),
        _ => None,
    }
}

#[derive(Default)]
struct DistinctState {
    seen: FxHashSet<CanonicalRow>,
}

impl DistinctState {
    fn insert(&mut self, row: CanonicalRow) -> bool {
        self.seen.insert(row)
    }
}

fn distinct_filter_batch(
    batch: RecordBatch,
    state: &mut DistinctState,
) -> ExecutorResult<Option<RecordBatch>> {
    if batch.num_rows() == 0 {
        return Ok(None);
    }

    let mut keep_flags = Vec::with_capacity(batch.num_rows());
    let mut keep_count = 0usize;

    for row_idx in 0..batch.num_rows() {
        let row = CanonicalRow::from_batch(&batch, row_idx)?;
        if state.insert(row) {
            keep_flags.push(true);
            keep_count += 1;
        } else {
            keep_flags.push(false);
        }
    }

    if keep_count == 0 {
        return Ok(None);
    }

    if keep_count == batch.num_rows() {
        return Ok(Some(batch));
    }

    let mut builder = BooleanBuilder::with_capacity(batch.num_rows());
    for flag in keep_flags {
        builder.append_value(flag);
    }
    let mask = Arc::new(builder.finish());

    let filtered = filter_record_batch(&batch, &mask).map_err(|err| {
        Error::InvalidArgumentError(format!("failed to apply DISTINCT filter: {err}"))
    })?;

    Ok(Some(filtered))
}

fn sort_record_batch_with_order(
    schema: &Arc<Schema>,
    batch: &RecordBatch,
    order_by: &[OrderByPlan],
) -> ExecutorResult<RecordBatch> {
    if order_by.is_empty() {
        return Ok(batch.clone());
    }

    let mut sort_columns: Vec<SortColumn> = Vec::with_capacity(order_by.len());

    for order in order_by {
        let column_index = match &order.target {
            OrderTarget::Column(name) => schema.index_of(name).map_err(|_| {
                Error::InvalidArgumentError(format!(
                    "ORDER BY references unknown column '{}'",
                    name
                ))
            })?,
            OrderTarget::Index(idx) => {
                if *idx >= batch.num_columns() {
                    return Err(Error::InvalidArgumentError(format!(
                        "ORDER BY position {} is out of bounds for {} columns",
                        idx + 1,
                        batch.num_columns()
                    )));
                }
                *idx
            }
            OrderTarget::All => {
                return Err(Error::InvalidArgumentError(
                    "ORDER BY ALL should be expanded before sorting".into(),
                ));
            }
        };

        let source_array = batch.column(column_index);

        let values: ArrayRef = match order.sort_type {
            OrderSortType::Native => Arc::clone(source_array),
            OrderSortType::CastTextToInteger => {
                let strings = source_array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "ORDER BY CAST expects the underlying column to be TEXT".into(),
                        )
                    })?;
                let mut builder = Int64Builder::with_capacity(strings.len());
                for i in 0..strings.len() {
                    if strings.is_null(i) {
                        builder.append_null();
                    } else {
                        match strings.value(i).parse::<i64>() {
                            Ok(value) => builder.append_value(value),
                            Err(_) => builder.append_null(),
                        }
                    }
                }
                Arc::new(builder.finish()) as ArrayRef
            }
        };

        let sort_options = SortOptions {
            descending: !order.ascending,
            nulls_first: order.nulls_first,
        };

        sort_columns.push(SortColumn {
            values,
            options: Some(sort_options),
        });
    }

    let indices = lexsort_to_indices(&sort_columns, None).map_err(|err| {
        Error::InvalidArgumentError(format!("failed to compute ORDER BY indices: {err}"))
    })?;

    let perm = indices
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| Error::Internal("ORDER BY sorting produced unexpected index type".into()))?;

    let mut reordered_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    for col_idx in 0..batch.num_columns() {
        let reordered = take(batch.column(col_idx), perm, None).map_err(|err| {
            Error::InvalidArgumentError(format!(
                "failed to apply ORDER BY permutation to column {col_idx}: {err}"
            ))
        })?;
        reordered_columns.push(reordered);
    }

    RecordBatch::try_new(Arc::clone(schema), reordered_columns)
        .map_err(|err| Error::Internal(format!("failed to build reordered ORDER BY batch: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, ArrayRef, Date32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use llkv_expr::expr::{BinaryOp, CompareOp};
    use llkv_expr::literal::Literal;
    use llkv_storage::pager::MemPager;
    use std::sync::Arc;

    #[test]
    fn cross_product_context_evaluates_expressions() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("main.tab2.a", DataType::Int64, false),
            Field::new("main.tab2.b", DataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
                Arc::new(Int64Array::from(vec![10, 20, 30])) as ArrayRef,
            ],
        )
        .expect("valid batch");

        let lookup = build_cross_product_column_lookup(schema.as_ref(), &[], &[], &[]);
        let mut ctx = CrossProductExpressionContext::new(schema.as_ref(), lookup)
            .expect("context builds from schema");

        let literal_expr: ScalarExpr<String> = ScalarExpr::literal(67);
        let literal = ctx
            .evaluate(&literal_expr, &batch)
            .expect("literal evaluation succeeds");
        let literal_array = literal
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 literal result");
        assert_eq!(literal_array.len(), 3);
        assert!(literal_array.iter().all(|value| value == Some(67)));

        let add_expr = ScalarExpr::binary(
            ScalarExpr::column("tab2.a".to_string()),
            BinaryOp::Add,
            ScalarExpr::literal(5),
        );
        let added = ctx
            .evaluate(&add_expr, &batch)
            .expect("column addition succeeds");
        let added_array = added
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 addition result");
        assert_eq!(added_array.values(), &[6, 7, 8]);
    }

    #[test]
    fn cross_product_filter_handles_date32_columns() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "orders.o_orderdate",
            DataType::Date32,
            false,
        )]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Date32Array::from(vec![0, 1, 3])) as ArrayRef],
        )
        .expect("valid batch");

        let lookup = build_cross_product_column_lookup(schema.as_ref(), &[], &[], &[]);
        let mut ctx = CrossProductExpressionContext::new(schema.as_ref(), lookup)
            .expect("context builds from schema");

        let field_id = ctx
            .schema()
            .columns
            .first()
            .expect("schema exposes date column")
            .field_id;

        let predicate = LlkvExpr::Compare {
            left: ScalarExpr::Column(field_id),
            op: CompareOp::GtEq,
            right: ScalarExpr::Literal(Literal::Date32(1)),
        };

        let truths = ctx
            .evaluate_predicate_truths(&predicate, &batch, &mut |_, _, _, _| Ok(None))
            .expect("date comparison evaluates");

        assert_eq!(truths, vec![Some(false), Some(true), Some(true)]);
    }

    #[test]
    fn group_by_handles_date32_columns() {
        let array: ArrayRef = Arc::new(Date32Array::from(vec![Some(3), None, Some(-7)]));

        let first = group_key_value(&array, 0).expect("extract first group key");
        assert_eq!(first, GroupKeyValue::Int(3));

        let second = group_key_value(&array, 1).expect("extract second group key");
        assert_eq!(second, GroupKeyValue::Null);

        let third = group_key_value(&array, 2).expect("extract third group key");
        assert_eq!(third, GroupKeyValue::Int(-7));
    }

    #[test]
    fn aggregate_expr_allows_numeric_casts() {
        let expr = ScalarExpr::Cast {
            expr: Box::new(ScalarExpr::literal(31)),
            data_type: DataType::Int32,
        };
        let aggregates = FxHashMap::default();

        let value = QueryExecutor::<MemPager>::evaluate_expr_with_aggregates(&expr, &aggregates)
            .expect("cast should succeed for in-range integral values");

        assert_eq!(value, Some(31));
    }

    #[test]
    fn aggregate_expr_cast_rejects_out_of_range_values() {
        let expr = ScalarExpr::Cast {
            expr: Box::new(ScalarExpr::literal(-1)),
            data_type: DataType::UInt8,
        };
        let aggregates = FxHashMap::default();

        let result = QueryExecutor::<MemPager>::evaluate_expr_with_aggregates(&expr, &aggregates);

        assert!(matches!(result, Err(Error::InvalidArgumentError(_))));
    }

    #[test]
    fn aggregate_expr_null_literal_remains_null() {
        let expr = ScalarExpr::binary(
            ScalarExpr::literal(0),
            BinaryOp::Subtract,
            ScalarExpr::cast(ScalarExpr::literal(Literal::Null), DataType::Int64),
        );
        let aggregates = FxHashMap::default();

        let value = QueryExecutor::<MemPager>::evaluate_expr_with_aggregates(&expr, &aggregates)
            .expect("expression should evaluate");

        assert_eq!(value, None);
    }

    #[test]
    fn aggregate_expr_divide_by_zero_returns_null() {
        let expr = ScalarExpr::binary(
            ScalarExpr::literal(10),
            BinaryOp::Divide,
            ScalarExpr::literal(0),
        );
        let aggregates = FxHashMap::default();

        let value = QueryExecutor::<MemPager>::evaluate_expr_with_aggregates(&expr, &aggregates)
            .expect("division should evaluate");

        assert_eq!(value, None);
    }

    #[test]
    fn aggregate_expr_modulo_by_zero_returns_null() {
        let expr = ScalarExpr::binary(
            ScalarExpr::literal(10),
            BinaryOp::Modulo,
            ScalarExpr::literal(0),
        );
        let aggregates = FxHashMap::default();

        let value = QueryExecutor::<MemPager>::evaluate_expr_with_aggregates(&expr, &aggregates)
            .expect("modulo should evaluate");

        assert_eq!(value, None);
    }

    #[test]
    fn constant_and_with_null_yields_null() {
        let expr = ScalarExpr::binary(
            ScalarExpr::literal(Literal::Null),
            BinaryOp::And,
            ScalarExpr::literal(1),
        );

        let value = evaluate_constant_scalar_with_aggregates(&expr)
            .expect("expression should fold as constant");

        assert!(matches!(value, Literal::Null));
    }

    #[test]
    fn cross_product_handles_more_than_two_tables() {
        let schema_a = Arc::new(Schema::new(vec![Field::new(
            "main.t1.a",
            DataType::Int64,
            false,
        )]));
        let schema_b = Arc::new(Schema::new(vec![Field::new(
            "main.t2.b",
            DataType::Int64,
            false,
        )]));
        let schema_c = Arc::new(Schema::new(vec![Field::new(
            "main.t3.c",
            DataType::Int64,
            false,
        )]));

        let batch_a = RecordBatch::try_new(
            Arc::clone(&schema_a),
            vec![Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef],
        )
        .expect("valid batch");
        let batch_b = RecordBatch::try_new(
            Arc::clone(&schema_b),
            vec![Arc::new(Int64Array::from(vec![10, 20, 30])) as ArrayRef],
        )
        .expect("valid batch");
        let batch_c = RecordBatch::try_new(
            Arc::clone(&schema_c),
            vec![Arc::new(Int64Array::from(vec![100])) as ArrayRef],
        )
        .expect("valid batch");

        let data_a = TableCrossProductData {
            schema: schema_a,
            batches: vec![batch_a],
            column_counts: vec![1],
            table_indices: vec![0],
        };
        let data_b = TableCrossProductData {
            schema: schema_b,
            batches: vec![batch_b],
            column_counts: vec![1],
            table_indices: vec![1],
        };
        let data_c = TableCrossProductData {
            schema: schema_c,
            batches: vec![batch_c],
            column_counts: vec![1],
            table_indices: vec![2],
        };

        let ab = cross_join_table_batches(data_a, data_b).expect("two-table product");
        assert_eq!(ab.schema.fields().len(), 2);
        assert_eq!(ab.batches.len(), 1);
        assert_eq!(ab.batches[0].num_rows(), 6);

        let abc = cross_join_table_batches(ab, data_c).expect("three-table product");
        assert_eq!(abc.schema.fields().len(), 3);
        assert_eq!(abc.batches.len(), 1);

        let final_batch = &abc.batches[0];
        assert_eq!(final_batch.num_rows(), 6);

        let col_a = final_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("left column values");
        assert_eq!(col_a.values(), &[1, 1, 1, 2, 2, 2]);

        let col_b = final_batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("middle column values");
        assert_eq!(col_b.values(), &[10, 20, 30, 10, 20, 30]);

        let col_c = final_batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("right column values");
        assert_eq!(col_c.values(), &[100, 100, 100, 100, 100, 100]);
    }
}
