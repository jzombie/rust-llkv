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
//! - [`utils`]: Utility functions (time)
//!
//! The [`QueryExecutor`] and [`SelectExecution`] implementations are defined inline
//! in this module for now, but should be extracted to a dedicated `query` module
//! in a future refactoring.

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, Int64Builder, LargeStringArray, RecordBatch, StringArray,
    StructArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array, new_null_array,
};
use arrow::compute::{
    SortColumn, SortOptions, cast, concat_batches, filter_record_batch, lexsort_to_indices, take,
};
use arrow::datatypes::{DataType, Field, Float64Type, Int64Type, Schema};
use llkv_aggregate::{AggregateAccumulator, AggregateKind, AggregateSpec, AggregateState};
use llkv_column_map::gather::gather_indices_from_batches;
use llkv_column_map::store::Projection as StoreProjection;
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::SubqueryId;
use llkv_expr::expr::{
    AggregateCall, BinaryOp, CompareOp, Expr as LlkvExpr, Filter, Operator, ScalarExpr,
};
use llkv_expr::literal::Literal;
use llkv_expr::typed_predicate::{
    build_bool_predicate, build_fixed_width_predicate, build_var_width_predicate,
};
use llkv_join::cross_join_pair;
use llkv_plan::{
    AggregateExpr, AggregateFunction, CanonicalRow, CompoundOperator, CompoundQuantifier,
    CompoundSelectComponent, CompoundSelectPlan, OrderByPlan, OrderSortType, OrderTarget,
    PlanValue, SelectPlan, SelectProjection,
};
use llkv_result::Error;
use llkv_storage::pager::Pager;
use llkv_table::table::{
    RowIdFilter, ScanOrderDirection, ScanOrderSpec, ScanOrderTransform, ScanProjection,
    ScanStreamOptions,
};
use llkv_table::types::FieldId;
use llkv_table::{NumericArray, NumericArrayMap, NumericKernels, ROW_ID_FIELD_ID};
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
pub mod utils;

// ============================================================================
// Type Aliases and Re-exports
// ============================================================================

/// Result type for executor operations.
pub type ExecutorResult<T> = Result<T, Error>;

pub use insert::{
    build_array_for_column, normalize_insert_value_for_column, resolve_insert_columns,
};
pub use translation::{
    build_projected_columns, build_wildcard_projections, full_table_scan_filter,
    resolve_field_id_from_schema, schema_for_projections, translate_predicate,
    translate_predicate_with, translate_scalar, translate_scalar_with,
};
pub use types::{
    ExecutorColumn, ExecutorMultiColumnUnique, ExecutorRowBatch, ExecutorSchema, ExecutorTable,
    ExecutorTableProvider,
};
pub use utils::current_time_micros;

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
    Int64(i64),
    Float64(f64),
}

impl AggregateValue {
    /// Convert to i64, truncating floats if necessary
    fn to_i64(&self) -> i64 {
        match self {
            AggregateValue::Int64(v) => *v,
            AggregateValue::Float64(v) => *v as i64,
        }
    }

    /// Convert to f64, promoting integers if necessary
    #[allow(dead_code)]
    fn to_f64(&self) -> f64 {
        match self {
            AggregateValue::Int64(v) => *v as f64,
            AggregateValue::Float64(v) => *v,
        }
    }
}

struct GroupState {
    batch: RecordBatch,
    row_idx: usize,
}

/// State for a group when computing aggregates
struct GroupAggregateState {
    batch: RecordBatch,
    representative_row: usize,
    row_indices: Vec<usize>, // Track all rows belonging to this group
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
/// (possibly wrapped in unary + or - operators), otherwise returns `None`
/// (indicating a complex expression that needs full evaluation).
///
/// This handles common cases like `col`, `+col`, `-col`, `++col`, etc.
fn try_extract_simple_column<F: AsRef<str>>(expr: &ScalarExpr<F>) -> Option<&str> {
    match expr {
        ScalarExpr::Column(name) => Some(name.as_ref()),
        // Unwrap unary operators to check if there's a column underneath
        ScalarExpr::Binary { left, op, right } => {
            // Check for unary-like patterns: left or right is a literal that acts as identity
            match op {
                BinaryOp::Add => {
                    // Check if one side is zero (identity for addition)
                    if matches!(left.as_ref(), ScalarExpr::Literal(Literal::Integer(0))) {
                        return try_extract_simple_column(right);
                    }
                    if matches!(right.as_ref(), ScalarExpr::Literal(Literal::Integer(0))) {
                        return try_extract_simple_column(left);
                    }
                }
                // Note: We do NOT handle Subtract here because 0 - col is NOT the same as col
                // It needs to be evaluated as a negation
                BinaryOp::Multiply => {
                    // -col is represented as Multiply(-1, col)
                    if matches!(left.as_ref(), ScalarExpr::Literal(Literal::Integer(-1))) {
                        return try_extract_simple_column(right);
                    }
                    if matches!(right.as_ref(), ScalarExpr::Literal(Literal::Integer(-1))) {
                        return try_extract_simple_column(left);
                    }
                    // +col might be Multiply(1, col)
                    if matches!(left.as_ref(), ScalarExpr::Literal(Literal::Integer(1))) {
                        return try_extract_simple_column(right);
                    }
                    if matches!(right.as_ref(), ScalarExpr::Literal(Literal::Integer(1))) {
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
    use arrow::array::{Float64Array, Int64Array, StringArray};

    // Infer type from first non-null value
    let mut value_type = None;
    for v in values {
        if !matches!(v, PlanValue::Null) {
            value_type = Some(v);
            break;
        }
    }

    match value_type {
        Some(PlanValue::Integer(_)) => {
            let int_values: Vec<Option<i64>> = values
                .iter()
                .map(|v| match v {
                    PlanValue::Integer(i) => Some(*i),
                    PlanValue::Null => None,
                    _ => Some(0), // Type mismatch, use default
                })
                .collect();
            Ok(Arc::new(Int64Array::from(int_values)) as ArrayRef)
        }
        Some(PlanValue::Float(_)) => {
            let float_values: Vec<Option<f64>> = values
                .iter()
                .map(|v| match v {
                    PlanValue::Float(f) => Some(*f),
                    PlanValue::Integer(i) => Some(*i as f64),
                    PlanValue::Null => None,
                    _ => Some(0.0), // Type mismatch, use default
                })
                .collect();
            Ok(Arc::new(Float64Array::from(float_values)) as ArrayRef)
        }
        Some(PlanValue::String(_)) => {
            let string_values: Vec<Option<&str>> = values
                .iter()
                .map(|v| match v {
                    PlanValue::String(s) => Some(s.as_str()),
                    PlanValue::Null => None,
                    _ => Some(""), // Type mismatch, use default
                })
                .collect();
            Ok(Arc::new(StringArray::from(string_values)) as ArrayRef)
        }
        _ => {
            // All nulls, create an Int64 array of nulls
            let null_values: Vec<Option<i64>> = vec![None; values.len()];
            Ok(Arc::new(Int64Array::from(null_values)) as ArrayRef)
        }
    }
}

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
        if plan.compound.is_some() {
            return self.execute_compound_select(plan, row_filter);
        }

        // Handle SELECT without FROM clause (e.g., SELECT 42, SELECT {'a': 1})
        if plan.tables.is_empty() {
            return self.execute_select_without_table(plan);
        }

        if !plan.group_by.is_empty() {
            if plan.tables.len() > 1 {
                return self.execute_cross_product(plan);
            }
            let table_ref = &plan.tables[0];
            let table = self.provider.get_table(&table_ref.qualified_name())?;
            let display_name = table_ref.qualified_name();
            return self.execute_group_by_single_table(table, display_name, plan, row_filter);
        }

        // Handle multi-table queries (cross products/joins)
        if plan.tables.len() > 1 {
            return self.execute_cross_product(plan);
        }

        // Single table query
        let table_ref = &plan.tables[0];
        let table = self.provider.get_table(&table_ref.qualified_name())?;
        let display_name = table_ref.qualified_name();

        if !plan.aggregates.is_empty() {
            self.execute_aggregates(table, display_name, plan, row_filter)
        } else if self.has_computed_aggregates(&plan) {
            // Handle computed projections that contain embedded aggregates
            self.execute_computed_aggregates(table, display_name, plan, row_filter)
        } else {
            self.execute_projection(table, display_name, plan, row_filter)
        }
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
            ScalarExpr::Column(_) | ScalarExpr::Literal(_) => false,
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
        let bound_plan = bind_select_plan(&subquery.plan, &bindings)?;
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
                result = Some(array_value_to_literal(&column, idx)?);
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
    ) -> ExecutorResult<NumericArray> {
        let mut values: Vec<Option<f64>> = Vec::with_capacity(batch.num_rows());
        let mut all_integer = true;

        for row_idx in 0..batch.num_rows() {
            let literal =
                self.evaluate_scalar_subquery_literal(context, subquery, batch, row_idx)?;
            match literal {
                Literal::Null => values.push(None),
                Literal::Integer(value) => {
                    let cast = i64::try_from(value).map_err(|_| {
                        Error::InvalidArgumentError(
                            "scalar subquery integer result exceeds supported range".into(),
                        )
                    })?;
                    values.push(Some(cast as f64));
                }
                Literal::Float(value) => {
                    all_integer = false;
                    values.push(Some(value));
                }
                Literal::Boolean(flag) => {
                    let numeric = if flag { 1.0 } else { 0.0 };
                    values.push(Some(numeric));
                }
                Literal::String(_) | Literal::Struct(_) => {
                    return Err(Error::InvalidArgumentError(
                        "scalar subquery produced non-numeric result in numeric context".into(),
                    ));
                }
            }
        }

        if all_integer {
            let iter = values.into_iter().map(|opt| opt.map(|v| v as i64));
            let array = Int64Array::from_iter(iter);
            NumericArray::try_from_arrow(&(Arc::new(array) as ArrayRef))
        } else {
            let array = Float64Array::from_iter(values);
            NumericArray::try_from_arrow(&(Arc::new(array) as ArrayRef))
        }
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
        for subquery_id in subquery_ids {
            let info = scalar_lookup
                .get(&subquery_id)
                .ok_or_else(|| Error::Internal("missing scalar subquery metadata".into()))?;
            let field_id = context.allocate_synthetic_field_id()?;
            let numeric = self.evaluate_scalar_subquery_numeric(context, info, batch)?;
            context.numeric_cache.insert(field_id, numeric);
            mapping.insert(subquery_id, field_id);
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
                    // Infer the data type from the expression
                    let (field_name, dtype, array) = match expr {
                        ScalarExpr::Literal(lit) => {
                            let (dtype, array) = Self::literal_to_array(lit)?;
                            (alias.clone(), dtype, array)
                        }
                        _ => {
                            return Err(Error::InvalidArgumentError(
                                "SELECT without FROM only supports literal expressions".into(),
                            ));
                        }
                    };

                    fields.push(Field::new(field_name, dtype, true));
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
            ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray, StructArray,
            new_null_array,
        };
        use arrow::datatypes::{DataType, Field};
        use llkv_expr::literal::Literal;

        match lit {
            Literal::Integer(v) => {
                let val = i64::try_from(*v).unwrap_or(0);
                Ok((
                    DataType::Int64,
                    Arc::new(Int64Array::from(vec![val])) as ArrayRef,
                ))
            }
            Literal::Float(v) => Ok((
                DataType::Float64,
                Arc::new(Float64Array::from(vec![*v])) as ArrayRef,
            )),
            Literal::Boolean(v) => Ok((
                DataType::Boolean,
                Arc::new(BooleanArray::from(vec![*v])) as ArrayRef,
            )),
            Literal::String(v) => Ok((
                DataType::Utf8,
                Arc::new(StringArray::from(vec![v.clone()])) as ArrayRef,
            )),
            Literal::Null => Ok((DataType::Null, new_null_array(&DataType::Null, 1))),
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
        let join_data = if plan.scalar_subqueries.is_empty() && remaining_filter.as_ref().is_some()
        {
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
                use llkv_join::{JoinKey, JoinOptions, TableJoinExt};

                let (left_ref, left_table) = &tables_with_handles[0];
                let (right_ref, right_table) = &tables_with_handles[1];

                // Determine join type from plan and convert to llkv_join::JoinType
                let join_type = plan
                    .joins
                    .get(0)
                    .map(|j| match j.join_type {
                        llkv_plan::JoinPlan::Inner => llkv_join::JoinType::Inner,
                        llkv_plan::JoinPlan::Left => llkv_join::JoinType::Left,
                        llkv_plan::JoinPlan::Right => llkv_join::JoinType::Right,
                        llkv_plan::JoinPlan::Full => llkv_join::JoinType::Full,
                    })
                    .unwrap_or(llkv_join::JoinType::Inner);

                tracing::debug!(
                    "Using llkv-join for {join_type:?} join between {} and {}",
                    left_ref.qualified_name(),
                    right_ref.qualified_name()
                );

                // Extract join keys from constraints if available
                // For now, use empty keys (cross product) and rely on filter
                // TODO: Parse ON conditions to extract proper join keys
                let join_keys: Vec<JoinKey> = Vec::new();

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

                // Build combined schema and convert to TableCrossProductData
                let mut combined_fields = Vec::new();
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

                let column_counts = vec![
                    left_table.schema.columns.len(),
                    right_table.schema.columns.len(),
                ];
                let table_indices = vec![0, 1];

                TableCrossProductData {
                    schema: combined_schema,
                    batches: result_batches,
                    column_counts,
                    table_indices,
                }
            } else {
                // Fall back to cartesian product for other cases
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
                for (idx, (table_ref, table)) in tables_with_handles.iter().enumerate() {
                    let constraints = constraint_map.get(idx).map(|v| v.as_slice()).unwrap_or(&[]);
                    staged.push(collect_table_data(
                        idx,
                        table_ref,
                        table.as_ref(),
                        constraints,
                    )?);
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

            let mut filtered_batches = Vec::with_capacity(combined_batches.len());
            for batch in combined_batches.into_iter() {
                filter_context.reset();
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

        let scalar_lookup: FxHashMap<SubqueryId, &llkv_plan::ScalarSubquery> = plan
            .scalar_subqueries
            .iter()
            .map(|subquery| (subquery.id, subquery))
            .collect();

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

                        for (idx, field) in combined_schema.fields().iter().enumerate() {
                            let field_name_lower = field.name().to_ascii_lowercase();
                            if !exclude_lower.contains(&field_name_lower) {
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
                AggregateExpr::CountStar { alias } => {
                    specs.push(AggregateSpec {
                        alias: alias.clone(),
                        kind: AggregateKind::Count {
                            field_id: None,
                            distinct: false,
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
                            if field.data_type() != &DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "SUM currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::Sum {
                                field_id: column_index as u32,
                                data_type: DataType::Int64,
                                distinct: *distinct,
                            }
                        }
                        AggregateFunction::MinInt64 => {
                            if field.data_type() != &DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "MIN currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::Min {
                                field_id: column_index as u32,
                                data_type: DataType::Int64,
                            }
                        }
                        AggregateFunction::MaxInt64 => {
                            if field.data_type() != &DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "MAX currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::Max {
                                field_id: column_index as u32,
                                data_type: DataType::Int64,
                            }
                        }
                        AggregateFunction::CountNulls => AggregateKind::CountNulls {
                            field_id: column_index as u32,
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
                let value = Self::evaluate_expr_with_aggregates(expr, &aggregate_values)?;
                fields.push(Arc::new(Field::new(alias, DataType::Int64, false)));
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
                | AggregateCall::Avg { expr, .. }
                | AggregateCall::Min(expr)
                | AggregateCall::Max(expr)
                | AggregateCall::CountNulls(expr) => {
                    // For now, we only support simple column references in aggregates at this level
                    // Complex expressions in aggregates need expression evaluation support
                    let column = try_extract_simple_column(expr).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "complex expressions in aggregates not yet supported in this context"
                                .into(),
                        )
                    })?;
                    let key_lower = column.to_ascii_lowercase();
                    let column_index = *column_lookup_map.get(&key_lower).ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "unknown column '{column}' in aggregate"
                        ))
                    })?;
                    let field = combined_schema.field(column_index);
                    let kind = match agg {
                        AggregateCall::Count { distinct, .. } => AggregateKind::Count {
                            field_id: Some(column_index as u32),
                            distinct: *distinct,
                        },
                        AggregateCall::Sum { distinct, .. } => {
                            if field.data_type() != &DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "SUM currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::Sum {
                                field_id: column_index as u32,
                                data_type: DataType::Int64,
                                distinct: *distinct,
                            }
                        }
                        AggregateCall::Avg { distinct, .. } => {
                            if field.data_type() != &DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "AVG currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::Avg {
                                field_id: column_index as u32,
                                data_type: DataType::Int64,
                                distinct: *distinct,
                            }
                        }
                        AggregateCall::Min(_) => {
                            if field.data_type() != &DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "MIN currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::Min {
                                field_id: column_index as u32,
                                data_type: DataType::Int64,
                            }
                        }
                        AggregateCall::Max(_) => {
                            if field.data_type() != &DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "MAX currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::Max {
                                field_id: column_index as u32,
                                data_type: DataType::Int64,
                            }
                        }
                        AggregateCall::CountNulls(_) => AggregateKind::CountNulls {
                            field_id: column_index as u32,
                        },
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

        for batch in batches {
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
                    AggregateValue::Int64(0)
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
                    AggregateValue::Float64(0.0)
                } else {
                    AggregateValue::Float64(float_array.value(0))
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
            Some(filter) if filter.subqueries.is_empty() => filter,
            _ => {
                tracing::debug!(
                    "join_opt[{query_label}]: skipping optimization – filter missing or uses subqueries"
                );
                return Ok(None);
            }
        };

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
        if plan
            .filter
            .as_ref()
            .is_some_and(|filter| !filter.subqueries.is_empty())
            || !plan.scalar_subqueries.is_empty()
        {
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

        let mut translated_filter: Option<llkv_expr::expr::Expr<'static, FieldId>> = None;
        let pushdown_filter = if let Some(filter_wrapper) = filter_wrapper_opt {
            let translated = crate::translation::expression::translate_predicate(
                filter_wrapper.predicate.clone(),
                table_ref.schema.as_ref(),
                |name| Error::InvalidArgumentError(format!("unknown column '{}'", name)),
            )?;
            if !filter_wrapper.subqueries.is_empty() {
                translated_filter = Some(translated.clone());
                strip_exists(&translated)
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

        let constant_having = plan
            .having
            .as_ref()
            .and_then(|expr| evaluate_constant_predicate(expr));

        if let Some(result) = constant_having {
            if !result.unwrap_or(false) {
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
                    for (index, field) in base_schema.fields().iter().enumerate() {
                        if !exclude_lower.contains(&field.name().to_ascii_lowercase()) {
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
                SelectProjection::Computed { expr: _, alias } => {
                    // For GROUP BY with aggregates, we don't evaluate the expression here
                    // because it may contain aggregate functions. We'll evaluate it later
                    // per group. For now, assume Float64 as a conservative type.
                    let field = Field::new(alias.clone(), DataType::Float64, true);
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
                    for (idx, field) in schema.fields().iter().enumerate() {
                        let column_name = field.name().to_ascii_lowercase();
                        if !exclude_lower.contains(&column_name) {
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
        for batch in &batches {
            for row_idx in 0..batch.num_rows() {
                let key = build_group_key(batch, row_idx, &key_indices)?;

                if let Some(&group_idx) = group_index.get(&key) {
                    // Add row to existing group
                    group_states[group_idx].row_indices.push(row_idx);
                } else {
                    // New group
                    let group_idx = group_states.len();
                    group_index.insert(key, group_idx);
                    group_states.push(GroupAggregateState {
                        batch: batch.clone(),
                        representative_row: row_idx,
                        row_indices: vec![row_idx],
                    });
                }
            }
        }

        // Second pass: compute aggregates for each group using proper AggregateState
        let mut group_aggregate_values: Vec<FxHashMap<String, PlanValue>> =
            Vec::with_capacity(group_states.len());

        for group_state in &group_states {
            // Create a mini-batch containing only rows from this group
            let group_batch = {
                let schema = group_state.batch.schema();
                let mut arrays: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());

                for col_idx in 0..schema.fields().len() {
                    let source_array = group_state.batch.column(col_idx);
                    // Use Arrow's take kernel to extract rows for this group
                    let indices = arrow::array::UInt64Array::from(
                        group_state
                            .row_indices
                            .iter()
                            .map(|&i| i as u64)
                            .collect::<Vec<_>>(),
                    );
                    let taken = arrow::compute::take(source_array.as_ref(), &indices, None)?;
                    arrays.push(taken);
                }

                RecordBatch::try_new(schema, arrays)?
            };

            // Create AggregateState for each aggregate and compute
            let mut aggregate_values: FxHashMap<String, PlanValue> = FxHashMap::default();

            // We might need to add computed columns to the batch for complex aggregate expressions
            let mut working_batch = group_batch.clone();
            let mut next_temp_col_idx = working_batch.num_columns();

            for (key, agg_call) in &aggregate_specs {
                // For aggregates on columns, find the column index in the batch
                let (projection_idx, _needs_cleanup) = match agg_call {
                    AggregateCall::CountStar => (None, false),
                    AggregateCall::Count { expr, .. }
                    | AggregateCall::Sum { expr, .. }
                    | AggregateCall::Avg { expr, .. }
                    | AggregateCall::Min(expr)
                    | AggregateCall::Max(expr)
                    | AggregateCall::CountNulls(expr) => {
                        // Try to extract a simple column name first
                        if let Some(col_name) = try_extract_simple_column(expr) {
                            let idx = resolve_column_name_to_index(col_name, &column_lookup_map);
                            (idx, false)
                        } else {
                            // Complex expression - evaluate it and add as temporary column
                            // Evaluate the expression for each row in the working batch
                            let mut computed_values = Vec::with_capacity(working_batch.num_rows());
                            for row_idx in 0..working_batch.num_rows() {
                                let value = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                                    expr,
                                    &FxHashMap::default(), // No aggregates during expression eval
                                    Some(&working_batch),
                                    Some(&column_lookup_map),
                                    row_idx,
                                )?;
                                computed_values.push(value);
                            }

                            // Convert PlanValues to Arrow array
                            let computed_array = plan_values_to_arrow_array(&computed_values)?;

                            // Add this as a new column to the working batch
                            let mut new_columns: Vec<ArrayRef> = working_batch.columns().to_vec();
                            new_columns.push(computed_array);

                            let temp_field = Arc::new(Field::new(
                                format!("__temp_agg_expr_{}", next_temp_col_idx),
                                arrow::datatypes::DataType::Int64, // TODO: infer from expr
                                true,
                            ));
                            let mut new_fields: Vec<Arc<Field>> =
                                working_batch.schema().fields().iter().cloned().collect();
                            new_fields.push(temp_field);
                            let new_schema = Arc::new(Schema::new(new_fields));

                            working_batch = RecordBatch::try_new(new_schema, new_columns)?;

                            let col_idx = next_temp_col_idx;
                            next_temp_col_idx += 1;
                            (Some(col_idx), true)
                        }
                    }
                };

                // Build the AggregateSpec - use dummy field_id since projection_idx will override it
                let spec = Self::build_aggregate_spec_for_cross_product(agg_call, key.clone())?;

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

            let mut row: Vec<PlanValue> = Vec::with_capacity(output_columns.len());
            for output in &output_columns {
                match output.source {
                    OutputSource::TableColumn { index } => {
                        // Use the representative row from this group
                        let value = llkv_plan::plan_value_from_array(
                            group_state.batch.column(index),
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
                            &aggregate_values,
                            Some(&group_state.batch),
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
                // Evaluate HAVING expression recursively
                let passes = Self::evaluate_having_expr(
                    having,
                    aggregate_values,
                    &group_state.batch,
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
                AggregateExpr::CountStar { alias } => {
                    specs.push(AggregateSpec {
                        alias,
                        kind: AggregateKind::Count {
                            field_id: None,
                            distinct: false,
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
                            if col.data_type != DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "SUM currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::Sum {
                                field_id: col.field_id,
                                data_type: DataType::Int64,
                                distinct,
                            }
                        }
                        AggregateFunction::MinInt64 => {
                            if col.data_type != DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "MIN currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::Min {
                                field_id: col.field_id,
                                data_type: DataType::Int64,
                            }
                        }
                        AggregateFunction::MaxInt64 => {
                            if col.data_type != DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "MAX currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::Max {
                                field_id: col.field_id,
                                data_type: DataType::Int64,
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
                crate::translation::expression::translate_predicate(
                    filter_wrapper.predicate.clone(),
                    table.schema.as_ref(),
                    |name| Error::InvalidArgumentError(format!("unknown column '{}'", name)),
                )?
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
                    // Evaluate the expression with aggregates substituted
                    let value = Self::evaluate_expr_with_aggregates(expr, &computed_aggregates)?;

                    fields.push(arrow::datatypes::Field::new(alias, DataType::Int64, false));

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

    /// Build an AggregateSpec from an AggregateCall
    fn build_aggregate_spec_from_call(
        agg_call: &llkv_expr::expr::AggregateCall<String>,
        alias: String,
        schema: &Schema,
        column_lookup: &FxHashMap<String, usize>,
    ) -> ExecutorResult<llkv_aggregate::AggregateSpec> {
        use llkv_expr::expr::AggregateCall;

        let kind = match agg_call {
            AggregateCall::CountStar => llkv_aggregate::AggregateKind::Count {
                field_id: None,
                distinct: false,
            },
            AggregateCall::Count {
                expr: col_expr,
                distinct,
            } => {
                let col_name = try_extract_simple_column(col_expr).ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "complex expressions in COUNT not yet fully supported".into(),
                    )
                })?;
                let col_idx = column_lookup
                    .get(&col_name.to_ascii_lowercase())
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!("column '{}' not found", col_name))
                    })?;
                let field = schema.field(*col_idx);
                let field_id = field
                    .metadata()
                    .get("field_id")
                    .and_then(|s| s.parse::<u32>().ok())
                    .ok_or_else(|| {
                        Error::Internal(format!("field_id not found for column '{}'", col_name))
                    })?;
                llkv_aggregate::AggregateKind::Count {
                    field_id: Some(field_id),
                    distinct: *distinct,
                }
            }
            AggregateCall::Sum {
                expr: col_expr,
                distinct,
            } => {
                let col_name = try_extract_simple_column(col_expr).ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "complex expressions in SUM not yet fully supported".into(),
                    )
                })?;
                let col_idx = column_lookup
                    .get(&col_name.to_ascii_lowercase())
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!("column '{}' not found", col_name))
                    })?;
                let field = schema.field(*col_idx);
                let field_id = field
                    .metadata()
                    .get("field_id")
                    .and_then(|s| s.parse::<u32>().ok())
                    .ok_or_else(|| {
                        Error::Internal(format!("field_id not found for column '{}'", col_name))
                    })?;
                llkv_aggregate::AggregateKind::Sum {
                    field_id,
                    data_type: DataType::Int64,
                    distinct: *distinct,
                }
            }
            AggregateCall::Avg {
                expr: col_expr,
                distinct,
            } => {
                let col_name = try_extract_simple_column(col_expr).ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "complex expressions in AVG not yet fully supported".into(),
                    )
                })?;
                let col_idx = column_lookup
                    .get(&col_name.to_ascii_lowercase())
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!("column '{}' not found", col_name))
                    })?;
                let field = schema.field(*col_idx);
                let field_id = field
                    .metadata()
                    .get("field_id")
                    .and_then(|s| s.parse::<u32>().ok())
                    .ok_or_else(|| {
                        Error::Internal(format!("field_id not found for column '{}'", col_name))
                    })?;
                llkv_aggregate::AggregateKind::Avg {
                    field_id,
                    data_type: DataType::Int64,
                    distinct: *distinct,
                }
            }
            AggregateCall::Min(col_expr) => {
                let col_name = try_extract_simple_column(col_expr).ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "complex expressions in MIN not yet fully supported".into(),
                    )
                })?;
                let col_idx = column_lookup
                    .get(&col_name.to_ascii_lowercase())
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!("column '{}' not found", col_name))
                    })?;
                let field = schema.field(*col_idx);
                let field_id = field
                    .metadata()
                    .get("field_id")
                    .and_then(|s| s.parse::<u32>().ok())
                    .ok_or_else(|| {
                        Error::Internal(format!("field_id not found for column '{}'", col_name))
                    })?;
                llkv_aggregate::AggregateKind::Min {
                    field_id,
                    data_type: DataType::Int64,
                }
            }
            AggregateCall::Max(col_expr) => {
                let col_name = try_extract_simple_column(col_expr).ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "complex expressions in MAX not yet fully supported".into(),
                    )
                })?;
                let col_idx = column_lookup
                    .get(&col_name.to_ascii_lowercase())
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!("column '{}' not found", col_name))
                    })?;
                let field = schema.field(*col_idx);
                let field_id = field
                    .metadata()
                    .get("field_id")
                    .and_then(|s| s.parse::<u32>().ok())
                    .ok_or_else(|| {
                        Error::Internal(format!("field_id not found for column '{}'", col_name))
                    })?;
                llkv_aggregate::AggregateKind::Max {
                    field_id,
                    data_type: DataType::Int64,
                }
            }
            AggregateCall::CountNulls(col_expr) => {
                let col_name = try_extract_simple_column(col_expr).ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "complex expressions in CountNulls not yet fully supported".into(),
                    )
                })?;
                let col_idx = column_lookup
                    .get(&col_name.to_ascii_lowercase())
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!("column '{}' not found", col_name))
                    })?;
                let field = schema.field(*col_idx);
                let field_id = field
                    .metadata()
                    .get("field_id")
                    .and_then(|s| s.parse::<u32>().ok())
                    .ok_or_else(|| {
                        Error::Internal(format!("field_id not found for column '{}'", col_name))
                    })?;
                llkv_aggregate::AggregateKind::CountNulls { field_id }
            }
        };

        Ok(llkv_aggregate::AggregateSpec { alias, kind })
    }

    /// Build an AggregateSpec for cross product GROUP BY (no field_id metadata required).
    /// Uses dummy field_id=0 since projection_index will override it in new_with_projection_index.
    fn build_aggregate_spec_for_cross_product(
        agg_call: &llkv_expr::expr::AggregateCall<String>,
        alias: String,
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
                data_type: DataType::Int64,
                distinct: *distinct,
            },
            AggregateCall::Avg { distinct, .. } => llkv_aggregate::AggregateKind::Avg {
                field_id: 0,
                data_type: DataType::Int64,
                distinct: *distinct,
            },
            AggregateCall::Min(_) => llkv_aggregate::AggregateKind::Min {
                field_id: 0,
                data_type: DataType::Int64,
            },
            AggregateCall::Max(_) => llkv_aggregate::AggregateKind::Max {
                field_id: 0,
                data_type: DataType::Int64,
            },
            AggregateCall::CountNulls(_) => {
                llkv_aggregate::AggregateKind::CountNulls { field_id: 0 }
            }
        };

        Ok(llkv_aggregate::AggregateSpec { alias, kind })
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
            ScalarExpr::Column(_) | ScalarExpr::Literal(_) => {}
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

        // Build aggregate specs for the aggregator
        let mut specs: Vec<AggregateSpec> = Vec::new();
        for (key, agg) in aggregate_specs {
            let kind = match agg {
                AggregateCall::CountStar => AggregateKind::Count {
                    field_id: None,
                    distinct: false,
                },
                AggregateCall::Count {
                    expr: col_expr,
                    distinct,
                } => {
                    let col_name = try_extract_simple_column(col_expr).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "complex expressions in COUNT not yet fully supported".into(),
                        )
                    })?;
                    let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", col_name))
                    })?;
                    AggregateKind::Count {
                        field_id: Some(col.field_id),
                        distinct: *distinct,
                    }
                }
                AggregateCall::Sum {
                    expr: col_expr,
                    distinct,
                } => {
                    let col_name = try_extract_simple_column(col_expr).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "complex expressions in SUM not yet fully supported".into(),
                        )
                    })?;
                    let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", col_name))
                    })?;
                    AggregateKind::Sum {
                        field_id: col.field_id,
                        data_type: DataType::Int64,
                        distinct: *distinct,
                    }
                }
                AggregateCall::Avg {
                    expr: col_expr,
                    distinct,
                } => {
                    let col_name = try_extract_simple_column(col_expr).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "complex expressions in AVG not yet fully supported".into(),
                        )
                    })?;
                    let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", col_name))
                    })?;
                    AggregateKind::Avg {
                        field_id: col.field_id,
                        data_type: DataType::Int64,
                        distinct: *distinct,
                    }
                }
                AggregateCall::Min(col_expr) => {
                    let col_name = try_extract_simple_column(col_expr).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "complex expressions in MIN not yet fully supported".into(),
                        )
                    })?;
                    let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", col_name))
                    })?;
                    AggregateKind::Min {
                        field_id: col.field_id,
                        data_type: DataType::Int64,
                    }
                }
                AggregateCall::Max(col_expr) => {
                    let col_name = try_extract_simple_column(col_expr).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "complex expressions in MAX not yet fully supported".into(),
                        )
                    })?;
                    let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", col_name))
                    })?;
                    AggregateKind::Max {
                        field_id: col.field_id,
                        data_type: DataType::Int64,
                    }
                }
                AggregateCall::CountNulls(col_expr) => {
                    let col_name = try_extract_simple_column(col_expr).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "complex expressions in CountNulls not yet fully supported".into(),
                        )
                    })?;
                    let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", col_name))
                    })?;
                    AggregateKind::CountNulls {
                        field_id: col.field_id,
                    }
                }
            };
            specs.push(AggregateSpec {
                alias: key.clone(),
                kind,
            });
        }

        // Prepare filter and projections
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

        let mut projections: Vec<ScanProjection> = Vec::new();
        let mut spec_to_projection: Vec<Option<usize>> = Vec::with_capacity(specs.len());
        let count_star_override: Option<i64> = None;

        for spec in &specs {
            if let Some(field_id) = spec.kind.field_id() {
                spec_to_projection.push(Some(projections.len()));
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

        // Extract the computed values
        for state in states {
            let alias = state.alias.clone();
            let (_field, array) = state.finalize()?;

            // Try Int64Array first
            if let Some(int64_array) = array.as_any().downcast_ref::<arrow::array::Int64Array>() {
                if int64_array.len() != 1 {
                    return Err(Error::Internal(format!(
                        "Expected single value from aggregate, got {}",
                        int64_array.len()
                    )));
                }
                let value = if int64_array.is_null(0) {
                    AggregateValue::Int64(0)
                } else {
                    AggregateValue::Int64(int64_array.value(0))
                };
                results.insert(alias, value);
            }
            // Try Float64Array for AVG
            else if let Some(float64_array) =
                array.as_any().downcast_ref::<arrow::array::Float64Array>()
            {
                if float64_array.len() != 1 {
                    return Err(Error::Internal(format!(
                        "Expected single value from aggregate, got {}",
                        float64_array.len()
                    )));
                }
                let value = if float64_array.is_null(0) {
                    AggregateValue::Float64(0.0)
                } else {
                    AggregateValue::Float64(float64_array.value(0))
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

    /// Evaluate expressions with PlanValue aggregates (for GROUP BY with aggregates)
    fn evaluate_expr_with_plan_value_aggregates(
        expr: &ScalarExpr<String>,
        aggregates: &FxHashMap<String, PlanValue>,
    ) -> ExecutorResult<PlanValue> {
        Self::evaluate_expr_with_plan_value_aggregates_and_row(expr, aggregates, None, None, 0)
    }

    fn evaluate_having_expr(
        expr: &llkv_expr::expr::Expr<String>,
        aggregates: &FxHashMap<String, PlanValue>,
        row_batch: &RecordBatch,
        column_lookup: &FxHashMap<String, usize>,
        row_idx: usize,
    ) -> ExecutorResult<Option<bool>> {
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
            ScalarExpr::Literal(Literal::Integer(v)) => Ok(PlanValue::Integer(*v as i64)),
            ScalarExpr::Literal(Literal::Float(v)) => Ok(PlanValue::Float(*v)),
            ScalarExpr::Literal(Literal::Boolean(v)) => {
                Ok(PlanValue::Integer(if *v { 1 } else { 0 }))
            }
            ScalarExpr::Literal(Literal::String(s)) => Ok(PlanValue::String(s.clone())),
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

                // Handle NULL comparisons (return NULL as 0 for integer context)
                if matches!(left_val, PlanValue::Null) || matches!(right_val, PlanValue::Null) {
                    return Ok(PlanValue::Integer(0));
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

                // Convert to numeric values for binary operations
                let left_num = match left_val {
                    PlanValue::Integer(i) => i as f64,
                    PlanValue::Float(f) => f,
                    PlanValue::Null => return Ok(PlanValue::Null),
                    _ => {
                        return Err(Error::InvalidArgumentError(
                            "Non-numeric value in binary operation".into(),
                        ));
                    }
                };
                let right_num = match right_val {
                    PlanValue::Integer(i) => i as f64,
                    PlanValue::Float(f) => f,
                    PlanValue::Null => return Ok(PlanValue::Null),
                    _ => {
                        return Err(Error::InvalidArgumentError(
                            "Non-numeric value in binary operation".into(),
                        ));
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
                };

                // Return as float if either operand was float, otherwise as integer
                if matches!(left_val, PlanValue::Float(_))
                    || matches!(right_val, PlanValue::Float(_))
                {
                    Ok(PlanValue::Float(result))
                } else {
                    Ok(PlanValue::Integer(result as i64))
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
                        _ => Err(Error::InvalidArgumentError(format!(
                            "Cannot cast {:?} to string",
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
                        // Simple CASE: compare operand with WHEN value
                        let when_val = Self::evaluate_expr_with_plan_value_aggregates_and_row(
                            when_expr,
                            aggregates,
                            row_batch,
                            column_lookup,
                            row_idx,
                        )?;
                        op_val == &when_val
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
            ScalarExpr::GetField { .. } => Err(Error::InvalidArgumentError(
                "GetField not supported in aggregate expressions".into(),
            )),
            ScalarExpr::ScalarSubquery(_) => Err(Error::InvalidArgumentError(
                "Scalar subqueries not supported in aggregate expressions".into(),
            )),
        }
    }

    fn evaluate_expr_with_aggregates(
        expr: &ScalarExpr<String>,
        aggregates: &FxHashMap<String, AggregateValue>,
    ) -> ExecutorResult<i64> {
        use llkv_expr::expr::BinaryOp;
        use llkv_expr::literal::Literal;

        match expr {
            ScalarExpr::Literal(Literal::Integer(v)) => Ok(*v as i64),
            ScalarExpr::Literal(Literal::Float(v)) => Ok(*v as i64),
            ScalarExpr::Literal(Literal::Boolean(v)) => Ok(if *v { 1 } else { 0 }),
            ScalarExpr::Literal(Literal::String(_)) => Err(Error::InvalidArgumentError(
                "String literals not supported in aggregate expressions".into(),
            )),
            ScalarExpr::Literal(Literal::Null) => Err(Error::InvalidArgumentError(
                "NULL literals not supported in aggregate expressions".into(),
            )),
            ScalarExpr::Literal(Literal::Struct(_)) => Err(Error::InvalidArgumentError(
                "Struct literals not supported in aggregate expressions".into(),
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
                // Convert to i64 for arithmetic (truncates floats)
                Ok(value.to_i64())
            }
            ScalarExpr::Not(inner) => {
                let value = Self::evaluate_expr_with_aggregates(inner, aggregates)?;
                Ok(if value != 0 { 0 } else { 1 })
            }
            ScalarExpr::Binary { left, op, right } => {
                let left_val = Self::evaluate_expr_with_aggregates(left, aggregates)?;
                let right_val = Self::evaluate_expr_with_aggregates(right, aggregates)?;

                let result = match op {
                    BinaryOp::Add => left_val.checked_add(right_val),
                    BinaryOp::Subtract => left_val.checked_sub(right_val),
                    BinaryOp::Multiply => left_val.checked_mul(right_val),
                    BinaryOp::Divide => {
                        if right_val == 0 {
                            return Err(Error::InvalidArgumentError("Division by zero".into()));
                        }
                        left_val.checked_div(right_val)
                    }
                    BinaryOp::Modulo => {
                        if right_val == 0 {
                            return Err(Error::InvalidArgumentError("Modulo by zero".into()));
                        }
                        left_val.checked_rem(right_val)
                    }
                };

                result.ok_or_else(|| {
                    Error::InvalidArgumentError("Arithmetic overflow in expression".into())
                })
            }
            ScalarExpr::Cast { .. } => Err(Error::InvalidArgumentError(
                "CAST is not supported in aggregate-only expressions".into(),
            )),
            ScalarExpr::GetField { .. } => Err(Error::InvalidArgumentError(
                "GetField not supported in aggregate-only expressions".into(),
            )),
            ScalarExpr::Case { .. } => Err(Error::InvalidArgumentError(
                "CASE not supported in aggregate-only expressions".into(),
            )),
            ScalarExpr::Coalesce(_) => Err(Error::InvalidArgumentError(
                "COALESCE not supported in aggregate-only expressions".into(),
            )),
            ScalarExpr::ScalarSubquery(_) => Err(Error::InvalidArgumentError(
                "Scalar subqueries not supported in aggregate-only expressions".into(),
            )),
        }
    }
}

struct CrossProductExpressionContext {
    schema: Arc<ExecutorSchema>,
    field_id_to_index: FxHashMap<FieldId, usize>,
    numeric_cache: FxHashMap<FieldId, NumericArray>,
    column_cache: FxHashMap<FieldId, ColumnAccessor>,
    next_field_id: FieldId,
}

#[derive(Clone)]
enum ColumnAccessor {
    Int64(Arc<Int64Array>),
    Float64(Arc<Float64Array>),
    Boolean(Arc<BooleanArray>),
    Utf8(Arc<StringArray>),
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
            DataType::Null => Ok(Self::Null(array.len())),
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported column type {:?} in cross product filter",
                other
            ))),
        }
    }

    fn len(&self) -> usize {
        match self {
            ColumnAccessor::Int64(array) => array.len(),
            ColumnAccessor::Float64(array) => array.len(),
            ColumnAccessor::Boolean(array) => array.len(),
            ColumnAccessor::Utf8(array) => array.len(),
            ColumnAccessor::Null(len) => *len,
        }
    }

    fn is_null(&self, idx: usize) -> bool {
        match self {
            ColumnAccessor::Int64(array) => array.is_null(idx),
            ColumnAccessor::Float64(array) => array.is_null(idx),
            ColumnAccessor::Boolean(array) => array.is_null(idx),
            ColumnAccessor::Utf8(array) => array.is_null(idx),
            ColumnAccessor::Null(_) => true,
        }
    }

    fn literal_at(&self, idx: usize) -> ExecutorResult<Literal> {
        if self.is_null(idx) {
            return Ok(Literal::Null);
        }
        match self {
            ColumnAccessor::Int64(array) => Ok(Literal::Integer(array.value(idx) as i128)),
            ColumnAccessor::Float64(array) => Ok(Literal::Float(array.value(idx))),
            ColumnAccessor::Boolean(array) => Ok(Literal::Boolean(array.value(idx))),
            ColumnAccessor::Utf8(array) => Ok(Literal::String(array.value(idx).to_string())),
            ColumnAccessor::Null(_) => Ok(Literal::Null),
        }
    }

    fn as_array_ref(&self) -> ArrayRef {
        match self {
            ColumnAccessor::Int64(array) => Arc::clone(array) as ArrayRef,
            ColumnAccessor::Float64(array) => Arc::clone(array) as ArrayRef,
            ColumnAccessor::Boolean(array) => Arc::clone(array) as ArrayRef,
            ColumnAccessor::Utf8(array) => Arc::clone(array) as ArrayRef,
            ColumnAccessor::Null(len) => new_null_array(&DataType::Null, *len),
        }
    }
}

#[derive(Clone)]
enum ValueArray {
    Numeric(NumericArray),
    Boolean(Arc<BooleanArray>),
    Utf8(Arc<StringArray>),
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
            DataType::Null => Ok(Self::Null(array.len())),
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64 => {
                let numeric = NumericArray::try_from_arrow(&array)?;
                Ok(Self::Numeric(numeric))
            }
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
            ValueArray::Null(len) => *len,
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

fn compare_bool(op: CompareOp, lhs: bool, rhs: bool) -> bool {
    let l = lhs as u8;
    let r = rhs as u8;
    match op {
        CompareOp::Eq => lhs == rhs,
        CompareOp::NotEq => lhs != rhs,
        CompareOp::Lt => l < r,
        CompareOp::LtEq => l <= r,
        CompareOp::Gt => l > r,
        CompareOp::GtEq => l >= r,
    }
}

fn compare_str(op: CompareOp, lhs: &str, rhs: &str) -> bool {
    match op {
        CompareOp::Eq => lhs == rhs,
        CompareOp::NotEq => lhs != rhs,
        CompareOp::Lt => lhs < rhs,
        CompareOp::LtEq => lhs <= rhs,
        CompareOp::Gt => lhs > rhs,
        CompareOp::GtEq => lhs >= rhs,
    }
}

fn finalize_in_list_result(has_match: bool, saw_null: bool, negated: bool) -> Option<bool> {
    if has_match {
        Some(!negated)
    } else if saw_null {
        None
    } else if negated {
        Some(true)
    } else {
        Some(false)
    }
}

fn literal_to_constant_array(literal: &Literal, len: usize) -> ExecutorResult<ArrayRef> {
    match literal {
        Literal::Integer(v) => {
            let value = i64::try_from(*v).unwrap_or(0);
            let values = vec![value; len];
            Ok(Arc::new(Int64Array::from(values)) as ArrayRef)
        }
        Literal::Float(v) => {
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
        Literal::Null => Ok(new_null_array(&DataType::Null, len)),
        Literal::Struct(_) => Err(Error::InvalidArgumentError(
            "struct literals are not supported in cross product filters".into(),
        )),
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
        match (&left_values, &right_values) {
            (ValueArray::Null(_), _) | (_, ValueArray::Null(_)) => Ok(vec![None; len]),
            (ValueArray::Numeric(lhs), ValueArray::Numeric(rhs)) => {
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    match (lhs.value(idx), rhs.value(idx)) {
                        (Some(lv), Some(rv)) => out.push(Some(NumericKernels::compare(op, lv, rv))),
                        _ => out.push(None),
                    }
                }
                Ok(out)
            }
            (ValueArray::Boolean(lhs), ValueArray::Boolean(rhs)) => {
                let lhs = lhs.as_ref();
                let rhs = rhs.as_ref();
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    if lhs.is_null(idx) || rhs.is_null(idx) {
                        out.push(None);
                    } else {
                        out.push(Some(compare_bool(op, lhs.value(idx), rhs.value(idx))));
                    }
                }
                Ok(out)
            }
            (ValueArray::Utf8(lhs), ValueArray::Utf8(rhs)) => {
                let lhs = lhs.as_ref();
                let rhs = rhs.as_ref();
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    if lhs.is_null(idx) || rhs.is_null(idx) {
                        out.push(None);
                    } else {
                        out.push(Some(compare_str(op, lhs.value(idx), rhs.value(idx))));
                    }
                }
                Ok(out)
            }
            _ => Err(Error::InvalidArgumentError(
                "unsupported comparison between mismatched types in cross product filter".into(),
            )),
        }
    }

    fn evaluate_is_null_truths(
        &mut self,
        expr: &ScalarExpr<FieldId>,
        negated: bool,
        batch: &RecordBatch,
    ) -> ExecutorResult<Vec<Option<bool>>> {
        let values = self.materialize_value_array(expr, batch)?;
        let len = values.len();

        match &values {
            ValueArray::Null(len) => {
                // All values are NULL
                let result = if negated {
                    Some(false) // IS NOT NULL on NULL column
                } else {
                    Some(true) // IS NULL on NULL column
                };
                Ok(vec![result; *len])
            }
            ValueArray::Numeric(arr) => {
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    let is_null = arr.value(idx).is_none();
                    let result = if negated {
                        !is_null // IS NOT NULL
                    } else {
                        is_null // IS NULL
                    };
                    out.push(Some(result));
                }
                Ok(out)
            }
            ValueArray::Boolean(arr) => {
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    let is_null = arr.is_null(idx);
                    let result = if negated { !is_null } else { is_null };
                    out.push(Some(result));
                }
                Ok(out)
            }
            ValueArray::Utf8(arr) => {
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    let is_null = arr.is_null(idx);
                    let result = if negated { !is_null } else { is_null };
                    out.push(Some(result));
                }
                Ok(out)
            }
        }
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

        match &target_values {
            ValueArray::Numeric(target_numeric) => {
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    let target_value = match target_numeric.value(idx) {
                        Some(value) => value,
                        None => {
                            out.push(None);
                            continue;
                        }
                    };
                    let mut has_match = false;
                    let mut saw_null = false;
                    for candidate in &list_values {
                        match candidate {
                            ValueArray::Numeric(array) => match array.value(idx) {
                                Some(value) => {
                                    if NumericKernels::compare(CompareOp::Eq, target_value, value) {
                                        has_match = true;
                                        break;
                                    }
                                }
                                None => saw_null = true,
                            },
                            ValueArray::Null(_) => saw_null = true,
                            _ => {
                                return Err(Error::InvalidArgumentError(
                                    "type mismatch in IN list evaluation".into(),
                                ));
                            }
                        }
                    }
                    out.push(finalize_in_list_result(has_match, saw_null, negated));
                }
                Ok(out)
            }
            ValueArray::Boolean(target_bool) => {
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    if target_bool.is_null(idx) {
                        out.push(None);
                        continue;
                    }
                    let target_value = target_bool.value(idx);
                    let mut has_match = false;
                    let mut saw_null = false;
                    for candidate in &list_values {
                        match candidate {
                            ValueArray::Boolean(array) => {
                                if array.is_null(idx) {
                                    saw_null = true;
                                } else if array.value(idx) == target_value {
                                    has_match = true;
                                    break;
                                }
                            }
                            ValueArray::Null(_) => saw_null = true,
                            _ => {
                                return Err(Error::InvalidArgumentError(
                                    "type mismatch in IN list evaluation".into(),
                                ));
                            }
                        }
                    }
                    out.push(finalize_in_list_result(has_match, saw_null, negated));
                }
                Ok(out)
            }
            ValueArray::Utf8(target_utf8) => {
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    if target_utf8.is_null(idx) {
                        out.push(None);
                        continue;
                    }
                    let target_value = target_utf8.value(idx);
                    let mut has_match = false;
                    let mut saw_null = false;
                    for candidate in &list_values {
                        match candidate {
                            ValueArray::Utf8(array) => {
                                if array.is_null(idx) {
                                    saw_null = true;
                                } else if array.value(idx) == target_value {
                                    has_match = true;
                                    break;
                                }
                            }
                            ValueArray::Null(_) => saw_null = true,
                            _ => {
                                return Err(Error::InvalidArgumentError(
                                    "type mismatch in IN list evaluation".into(),
                                ));
                            }
                        }
                    }
                    out.push(finalize_in_list_result(has_match, saw_null, negated));
                }
                Ok(out)
            }
            ValueArray::Null(len) => Ok(vec![None; *len]),
        }
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
    ) -> ExecutorResult<NumericArray> {
        if let Some(existing) = self.numeric_cache.get(&field_id) {
            return Ok(existing.clone());
        }

        let column_index = *self.field_id_to_index.get(&field_id).ok_or_else(|| {
            Error::Internal("field mapping missing during cross product evaluation".into())
        })?;

        let array_ref = batch.column(column_index).clone();
        let numeric = NumericArray::try_from_arrow(&array_ref)?;
        self.numeric_cache.insert(field_id, numeric.clone());
        Ok(numeric)
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
            ScalarExpr::ScalarSubquery(_) => Err(Error::InvalidArgumentError(
                "scalar subqueries are not supported in cross product filters".into(),
            )),
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
            | AggregateCall::Avg { expr, .. }
            | AggregateCall::Min(expr)
            | AggregateCall::Max(expr)
            | AggregateCall::CountNulls(expr) => {
                collect_field_ids(expr, out);
            }
        },
        ScalarExpr::GetField { base, .. } => collect_field_ids(base, out),
        ScalarExpr::Cast { expr, .. } => collect_field_ids(expr, out),
        ScalarExpr::Not(expr) => collect_field_ids(expr, out),
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
        ScalarExpr::Literal(_) => {}
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
        PlanValue::String(s) => {
            buf.push(3);
            let bytes = s.as_bytes();
            let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
            buf.extend_from_slice(&len.to_be_bytes());
            buf.extend_from_slice(bytes);
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
        _ => None,
    }
}

fn evaluate_constant_scalar(expr: &ScalarExpr<String>) -> Option<Literal> {
    match expr {
        ScalarExpr::Literal(lit) => Some(lit.clone()),
        _ => None,
    }
}

fn compare_literals(op: CompareOp, left: &Literal, right: &Literal) -> Option<bool> {
    use std::cmp::Ordering;

    match (left, right) {
        (Literal::Null, _) | (_, Literal::Null) => None,
        (Literal::Integer(lhs), Literal::Integer(rhs)) => {
            let ord = lhs.cmp(rhs);
            Some(match op {
                CompareOp::Eq => ord == Ordering::Equal,
                CompareOp::NotEq => ord != Ordering::Equal,
                CompareOp::Lt => ord == Ordering::Less,
                CompareOp::LtEq => ord != Ordering::Greater,
                CompareOp::Gt => ord == Ordering::Greater,
                CompareOp::GtEq => ord != Ordering::Less,
            })
        }
        (Literal::Float(lhs), Literal::Float(rhs)) => Some(match op {
            CompareOp::Eq => lhs == rhs,
            CompareOp::NotEq => lhs != rhs,
            CompareOp::Lt => lhs < rhs,
            CompareOp::LtEq => lhs <= rhs,
            CompareOp::Gt => lhs > rhs,
            CompareOp::GtEq => lhs >= rhs,
        }),
        (Literal::Integer(lhs), Literal::Float(_rhs)) => {
            compare_literals(op, &Literal::Float(*lhs as f64), right)
        }
        (Literal::Float(_lhs), Literal::Integer(rhs)) => {
            compare_literals(op, left, &Literal::Float(*rhs as f64))
        }
        (Literal::Boolean(lhs), Literal::Boolean(rhs)) => Some(match op {
            CompareOp::Eq => lhs == rhs,
            CompareOp::NotEq => lhs != rhs,
            CompareOp::Lt => (*lhs as u8) < (*rhs as u8),
            CompareOp::LtEq => (*lhs as u8) <= (*rhs as u8),
            CompareOp::Gt => (*lhs as u8) > (*rhs as u8),
            CompareOp::GtEq => (*lhs as u8) >= (*rhs as u8),
        }),
        (Literal::String(lhs), Literal::String(rhs)) => {
            let ord = lhs.cmp(rhs);
            Some(match op {
                CompareOp::Eq => ord == Ordering::Equal,
                CompareOp::NotEq => ord != Ordering::Equal,
                CompareOp::Lt => ord == Ordering::Less,
                CompareOp::LtEq => ord != Ordering::Greater,
                CompareOp::Gt => ord == Ordering::Greater,
                CompareOp::GtEq => ord != Ordering::Less,
            })
        }
        _ => None,
    }
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
        (Literal::Integer(a), Literal::Integer(b)) => Some(a.cmp(b)),
        (Literal::Float(a), Literal::Float(b)) => a.partial_cmp(b),
        (Literal::Integer(a), Literal::Float(b)) => (*a as f64).partial_cmp(b),
        (Literal::Float(a), Literal::Integer(b)) => a.partial_cmp(&(*b as f64)),
        (Literal::String(a), Literal::String(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

fn literal_equals(lhs: &Literal, rhs: &Literal) -> Option<bool> {
    match (lhs, rhs) {
        (Literal::Boolean(a), Literal::Boolean(b)) => Some(a == b),
        (Literal::String(a), Literal::String(b)) => Some(a == b),
        (Literal::Integer(_), Literal::Integer(_))
        | (Literal::Integer(_), Literal::Float(_))
        | (Literal::Float(_), Literal::Integer(_))
        | (Literal::Float(_), Literal::Float(_)) => {
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

fn array_value_to_literal(array: &ArrayRef, idx: usize) -> ExecutorResult<Literal> {
    if array.is_null(idx) {
        return Ok(Literal::Null);
    }

    match array.data_type() {
        DataType::Boolean => {
            let array = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| Error::Internal("failed to downcast boolean array".into()))?;
            Ok(Literal::Boolean(array.value(idx)))
        }
        DataType::Int8 => {
            let array = array
                .as_any()
                .downcast_ref::<Int8Array>()
                .ok_or_else(|| Error::Internal("failed to downcast int8 array".into()))?;
            Ok(Literal::Integer(array.value(idx) as i128))
        }
        DataType::Int16 => {
            let array = array
                .as_any()
                .downcast_ref::<Int16Array>()
                .ok_or_else(|| Error::Internal("failed to downcast int16 array".into()))?;
            Ok(Literal::Integer(array.value(idx) as i128))
        }
        DataType::Int32 => {
            let array = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| Error::Internal("failed to downcast int32 array".into()))?;
            Ok(Literal::Integer(array.value(idx) as i128))
        }
        DataType::Int64 => {
            let array = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| Error::Internal("failed to downcast int64 array".into()))?;
            Ok(Literal::Integer(array.value(idx) as i128))
        }
        DataType::UInt8 => {
            let array = array
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| Error::Internal("failed to downcast uint8 array".into()))?;
            Ok(Literal::Integer(array.value(idx) as i128))
        }
        DataType::UInt16 => {
            let array = array
                .as_any()
                .downcast_ref::<UInt16Array>()
                .ok_or_else(|| Error::Internal("failed to downcast uint16 array".into()))?;
            Ok(Literal::Integer(array.value(idx) as i128))
        }
        DataType::UInt32 => {
            let array = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| Error::Internal("failed to downcast uint32 array".into()))?;
            Ok(Literal::Integer(array.value(idx) as i128))
        }
        DataType::UInt64 => {
            let array = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("failed to downcast uint64 array".into()))?;
            Ok(Literal::Integer(array.value(idx) as i128))
        }
        DataType::Float32 => {
            let array = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| Error::Internal("failed to downcast float32 array".into()))?;
            Ok(Literal::Float(array.value(idx) as f64))
        }
        DataType::Float64 => {
            let array = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| Error::Internal("failed to downcast float64 array".into()))?;
            Ok(Literal::Float(array.value(idx)))
        }
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::Internal("failed to downcast utf8 array".into()))?;
            Ok(Literal::String(array.value(idx).to_string()))
        }
        DataType::LargeUtf8 => {
            let array = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| Error::Internal("failed to downcast large utf8 array".into()))?;
            Ok(Literal::String(array.value(idx).to_string()))
        }
        DataType::Struct(fields) => {
            let struct_array = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| Error::Internal("failed to downcast struct array".into()))?;
            let mut members = Vec::with_capacity(fields.len());
            for (field_idx, field) in fields.iter().enumerate() {
                let child = struct_array.column(field_idx);
                let literal = array_value_to_literal(child, idx)?;
                members.push((field.name().clone(), Box::new(literal)));
            }
            Ok(Literal::Struct(members))
        }
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported scalar subquery result type: {other:?}"
        ))),
    }
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
        ScalarExpr::Aggregate(_) | ScalarExpr::Column(_) | ScalarExpr::Literal(_) => {}
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
        ScalarExpr::Aggregate(_) | ScalarExpr::Column(_) | ScalarExpr::Literal(_) => expr.clone(),
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
        }
    }

    pub fn new_single_batch(table_name: String, schema: Arc<Schema>, batch: RecordBatch) -> Self {
        Self {
            table_name,
            schema,
            stream: SelectStream::Aggregation { batch },
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

    pub fn stream(
        self,
        mut on_batch: impl FnMut(RecordBatch) -> ExecutorResult<()>,
    ) -> ExecutorResult<()> {
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
            DataType::Int64 => ScanOrderTransform::IdentityInteger,
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
    let filter_expr = crate::translation::expression::full_table_scan_filter(filter_field_id);

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
) -> ExecutorResult<(JoinMatchIndices, JoinMatchIndices)> {
    let right_key_indices: Vec<usize> = join_keys.iter().map(|(_, right)| *right).collect();

    // Parallelize hash table build phase across batches
    // Each thread builds a local hash table for its batch(es), then we merge them
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
                        Err(_) => continue, // Skip rows with errors during parallel build
                    }
                }

                local_table
            })
            .collect();

        // Merge all local hash tables into one
        let mut merged_table: JoinHashTable = FxHashMap::default();
        for local_table in local_tables {
            for (key, mut positions) in local_table {
                merged_table.entry(key).or_default().append(&mut positions);
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
    let matches: Vec<(JoinMatchIndices, JoinMatchIndices)> =
        llkv_column_map::parallel::with_thread_pool(|| {
            left_batches
                .par_iter()
                .enumerate()
                .map(|(batch_idx, batch)| {
                    let mut local_left_matches: JoinMatchIndices = Vec::new();
                    let mut local_right_matches: JoinMatchIndices = Vec::new();
                    let mut key_buffer: Vec<u8> = Vec::new();

                    for row_idx in 0..batch.num_rows() {
                        key_buffer.clear();
                        match build_join_key(batch, &left_key_indices, row_idx, &mut key_buffer) {
                            Ok(true) => {
                                if let Some(entries) = hash_table.get(&key_buffer) {
                                    for &(r_batch, r_row) in entries {
                                        local_left_matches.push((batch_idx, row_idx));
                                        local_right_matches.push((r_batch, r_row));
                                    }
                                }
                            }
                            Ok(false) => continue,
                            Err(_) => continue, // Skip rows with errors during parallel probe
                        }
                    }

                    (local_left_matches, local_right_matches)
                })
                .collect()
        });

    // Merge all match results
    let mut left_matches: JoinMatchIndices = Vec::new();
    let mut right_matches: JoinMatchIndices = Vec::new();
    for (mut left, mut right) in matches {
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
) -> ExecutorResult<(JoinMatchIndices, Vec<Option<(usize, usize)>>)> {
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
    let matches: Vec<(JoinMatchIndices, Vec<Option<(usize, usize)>>)> =
        llkv_column_map::parallel::with_thread_pool(|| {
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
                        && let Some(value) = literal_to_plan_value_for_join(literal)
                        && column.table < constraints.len()
                    {
                        constraints[column.table]
                            .push(ColumnConstraint::Equality(ColumnLiteral { column, value }));
                    }
                }
                (None, Some(column)) => {
                    if let Some(literal) = extract_literal(left)
                        && let Some(value) = literal_to_plan_value_for_join(literal)
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
                        if let Some(value) = plan_value_from_operator_literal(literal_val) {
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
                        && let Some(value) = literal_to_plan_value_for_join(literal)
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
                && let Some(value) = literal_to_plan_value_for_join(literal)
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
                && let Some(value) = literal_to_plan_value_for_join(literal)
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
            && let Some(value) = literal_to_plan_value_for_join(literal)
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
                            && let Some(value) = literal_to_plan_value_for_join(literal)
                        {
                            literals
                                .push(ColumnConstraint::Equality(ColumnLiteral { column, value }));
                            handled_conjuncts += 1;
                            continue;
                        }
                    }
                    (None, Some(column)) => {
                        if let Some(literal) = extract_literal(left)
                            && let Some(value) = literal_to_plan_value_for_join(literal)
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
                            && let Some(value) = literal_to_plan_value_for_join(literal)
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
                    && let Some(value) = literal_to_plan_value_for_join(literal)
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

fn plan_value_from_operator_literal(op_value: &llkv_expr::literal::Literal) -> Option<PlanValue> {
    match op_value {
        llkv_expr::literal::Literal::Integer(v) => i64::try_from(*v).ok().map(PlanValue::Integer),
        llkv_expr::literal::Literal::Float(v) => Some(PlanValue::Float(*v)),
        llkv_expr::literal::Literal::Boolean(v) => Some(PlanValue::Integer(if *v { 1 } else { 0 })),
        llkv_expr::literal::Literal::String(v) => Some(PlanValue::String(v.clone())),
        _ => None,
    }
}

fn literal_to_plan_value_for_join(literal: &Literal) -> Option<PlanValue> {
    match literal {
        Literal::Integer(v) => i64::try_from(*v).ok().map(PlanValue::Integer),
        Literal::Float(v) => Some(PlanValue::Float(*v)),
        Literal::Boolean(v) => Some(PlanValue::Integer(if *v { 1 } else { 0 })),
        Literal::String(v) => Some(PlanValue::String(v.clone())),
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
    use arrow::array::{Array, ArrayRef, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use llkv_expr::expr::BinaryOp;
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
