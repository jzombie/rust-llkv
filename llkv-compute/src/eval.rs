use std::hash::Hash;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, StructArray, UInt32Array, new_null_array};
use arrow::compute::kernels::cast;
use arrow::compute::kernels::zip::zip;
use arrow::compute::{concat, is_not_null, take};
use arrow::datatypes::{DataType, Field, IntervalMonthDayNanoType};
use llkv_expr::literal::{Literal, LiteralExt};
use llkv_expr::{AggregateCall, BinaryOp, CompareOp, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};
use llkv_types::IntervalValue;
use rustc_hash::{FxHashMap, FxHashSet};
use sqlparser::ast::BinaryOperator;

use crate::date::{add_interval_to_date32, parse_date32_literal, subtract_interval_from_date32};
use crate::fast_numeric::NumericFastPath;
use crate::kernels::{compute_binary, compute_compare, get_common_type};

/// Mapping from field identifiers to the numeric Arrow array used for evaluation.
pub type NumericArrayMap<F> = FxHashMap<F, ArrayRef>;

/// Intermediate representation for vectorized evaluators.
enum VectorizedExpr {
    Array(ArrayRef),
    Scalar(ArrayRef),
}

impl VectorizedExpr {
    fn materialize(self, len: usize) -> ArrayRef {
        match self {
            VectorizedExpr::Array(array) => array,
            VectorizedExpr::Scalar(scalar_array) => {
                if scalar_array.is_empty() {
                    return new_null_array(scalar_array.data_type(), len);
                }
                if scalar_array.is_null(0) {
                    return new_null_array(scalar_array.data_type(), len);
                }

                // Expand scalar to array of length len
                let indices = UInt32Array::from(vec![0; len]);
                take(&scalar_array, &indices, None)
                    .unwrap_or_else(|_| new_null_array(scalar_array.data_type(), len))
            }
        }
    }
}

/// Extension methods for type inference on `ScalarExpr`.
pub trait ScalarExprTypeExt<F> {
    fn infer_result_type<R>(&self, resolve_type: &mut R) -> Option<DataType>
    where
        F: Hash + Eq + Copy,
        R: FnMut(F) -> Option<DataType>;

    fn infer_result_type_from_arrays(&self, arrays: &NumericArrayMap<F>) -> DataType
    where
        F: Hash + Eq + Copy;

    fn contains_interval(&self) -> bool;
}

impl<F: Hash + Eq + Copy> ScalarExprTypeExt<F> for ScalarExpr<F> {
    fn infer_result_type<R>(&self, resolve_type: &mut R) -> Option<DataType>
    where
        R: FnMut(F) -> Option<DataType>,
    {
        match self {
            ScalarExpr::Literal(lit) => Some(literal_type(lit)),
            ScalarExpr::Column(fid) => resolve_type(*fid),
            ScalarExpr::Binary { left, op, right } => {
                let left_type = left.infer_result_type(resolve_type)?;
                let right_type = right.infer_result_type(resolve_type)?;
                Some(binary_result_type(*op, left_type, right_type))
            }
            ScalarExpr::Compare { .. } => Some(DataType::Boolean),
            ScalarExpr::Not(_) => Some(DataType::Boolean),
            ScalarExpr::IsNull { .. } => Some(DataType::Boolean),
            ScalarExpr::Aggregate(call) => aggregate_result_type(call, resolve_type),
            ScalarExpr::GetField { base, field_name } => {
                let base_type = base.infer_result_type(resolve_type)?;
                match base_type {
                    DataType::Struct(fields) => fields
                        .iter()
                        .find(|f| f.name() == field_name)
                        .map(|f| f.data_type().clone()),
                    _ => None,
                }
            }
            ScalarExpr::Cast { data_type, .. } => Some(data_type.clone()),
            ScalarExpr::Case {
                branches,
                else_expr,
                ..
            } => {
                let mut types = Vec::new();
                for (_, then_expr) in branches {
                    if let Some(t) = then_expr.infer_result_type(resolve_type) {
                        types.push(t);
                    }
                }
                if let Some(else_expr) = else_expr {
                    if let Some(t) = else_expr.infer_result_type(resolve_type) {
                        types.push(t);
                    }
                } else {
                    // Implicit ELSE NULL
                    types.push(DataType::Null);
                }

                if types.is_empty() {
                    return None;
                }

                let mut common = types[0].clone();
                for t in &types[1..] {
                    common = get_common_type(&common, t);
                }

                Some(common)
            }
            ScalarExpr::Coalesce(items) => {
                let mut types = Vec::new();
                for item in items {
                    if let Some(t) = item.infer_result_type(resolve_type) {
                        types.push(t);
                    }
                }
                if types.is_empty() {
                    return None;
                }
                let mut common = types[0].clone();
                for t in &types[1..] {
                    common = get_common_type(&common, t);
                }
                Some(common)
            }
            ScalarExpr::Random => Some(DataType::Float64),
            ScalarExpr::ScalarSubquery(sub) => Some(sub.data_type.clone()),
        }
    }

    fn infer_result_type_from_arrays(&self, arrays: &NumericArrayMap<F>) -> DataType {
        let mut resolver = |fid| arrays.get(&fid).map(|a| a.data_type().clone());
        self.infer_result_type(&mut resolver)
            .unwrap_or(DataType::Float64)
    }

    fn contains_interval(&self) -> bool {
        match self {
            ScalarExpr::Literal(Literal::Interval(_)) => true,
            ScalarExpr::Binary { left, right, .. } => {
                left.contains_interval() || right.contains_interval()
            }
            _ => false,
        }
    }
}

fn literal_type(lit: &Literal) -> DataType {
    match lit {
        Literal::Null => DataType::Null,
        Literal::Boolean(_) => DataType::Boolean,
        Literal::Int128(_) => DataType::Int64, // Default to Int64 for literals
        Literal::Float64(_) => DataType::Float64,
        Literal::Decimal128(d) => DataType::Decimal128(std::cmp::max(d.precision(), d.scale() as u8), d.scale()),
        Literal::String(_) => DataType::Utf8,
        Literal::Date32(_) => DataType::Date32,
        Literal::Interval(_) => DataType::Interval(arrow::datatypes::IntervalUnit::MonthDayNano),
        Literal::Struct(fields) => {
            let arrow_fields = fields
                .iter()
                .map(|(name, lit)| Field::new(name, literal_type(lit), true))
                .collect();
            DataType::Struct(arrow_fields)
        }
    }
}

fn aggregate_result_type<F, R>(call: &AggregateCall<F>, resolve_type: &mut R) -> Option<DataType>
where
    F: Hash + Eq + Copy,
    R: FnMut(F) -> Option<DataType>,
{
    match call {
        AggregateCall::CountStar | AggregateCall::Count { .. } | AggregateCall::CountNulls(_) => {
            Some(DataType::Int64)
        }
        AggregateCall::Sum { expr, .. } => {
            let child = expr.infer_result_type(resolve_type)?;
            Some(match child {
                DataType::Decimal128(p, s) => DataType::Decimal128(p, s),
                DataType::Float32 | DataType::Float64 => DataType::Float64,
                DataType::UInt64 | DataType::Int64 => child,
                DataType::UInt32
                | DataType::UInt16
                | DataType::UInt8
                | DataType::Int32
                | DataType::Int16
                | DataType::Int8 => DataType::Int64,
                _ => DataType::Float64,
            })
        }
        AggregateCall::Total { expr, .. } | AggregateCall::Avg { expr, .. } => {
            let child = expr.infer_result_type(resolve_type)?;
            Some(match child {
                DataType::Decimal128(p, s) => DataType::Decimal128(p, s),
                _ => DataType::Float64,
            })
        }
        AggregateCall::Min(expr) | AggregateCall::Max(expr) => expr.infer_result_type(resolve_type),
        AggregateCall::GroupConcat { .. } => Some(DataType::Utf8),
    }
}

fn binary_result_type(op: BinaryOp, lhs: DataType, rhs: DataType) -> DataType {
    crate::kernels::common_type_for_op(&lhs, &rhs, op)
}

/// Represents an affine transformation `scale * field + offset`.
#[derive(Clone, Copy, Debug)]
pub struct AffineExpr<F> {
    pub field: F,
    pub scale: f64,
    pub offset: f64,
}

/// Internal accumulator representing a partially merged affine expression.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct AffineState<F> {
    field: Option<F>,
    scale: f64,
    offset: f64,
}

/// Centralizes the numeric kernels applied to scalar expressions so they can be
/// tuned without touching the surrounding table scan logic.
pub struct ScalarEvaluator;

impl ScalarEvaluator {
    /// Combine field identifiers while tracking whether multiple fields were encountered.
    #[allow(dead_code)]
    fn merge_field<F: Eq + Copy>(lhs: Option<F>, rhs: Option<F>) -> Option<Option<F>> {
        match (lhs, rhs) {
            (Some(a), Some(b)) => {
                if a == b {
                    Some(Some(a))
                } else {
                    None
                }
            }
            (Some(a), None) => Some(Some(a)),
            (None, Some(b)) => Some(Some(b)),
            (None, None) => Some(None),
        }
    }

    /// Collect every field referenced by the scalar expression into `acc`.
    pub fn collect_fields<F: Hash + Eq + Copy>(expr: &ScalarExpr<F>, acc: &mut FxHashSet<F>) {
        match expr {
            ScalarExpr::Column(fid) => {
                acc.insert(*fid);
            }
            ScalarExpr::Literal(_) => {}
            ScalarExpr::Binary { left, right, .. } => {
                Self::collect_fields(left, acc);
                Self::collect_fields(right, acc);
            }
            ScalarExpr::Compare { left, right, .. } => {
                Self::collect_fields(left, acc);
                Self::collect_fields(right, acc);
            }
            ScalarExpr::Not(inner) => {
                Self::collect_fields(inner, acc);
            }
            ScalarExpr::IsNull { expr, .. } => {
                Self::collect_fields(expr, acc);
            }
            ScalarExpr::Aggregate(agg) => {
                // Collect fields referenced by the aggregate expression
                match agg {
                    AggregateCall::CountStar => {}
                    AggregateCall::Count { expr, .. }
                    | AggregateCall::Sum { expr, .. }
                    | AggregateCall::Total { expr, .. }
                    | AggregateCall::Avg { expr, .. }
                    | AggregateCall::Min(expr)
                    | AggregateCall::Max(expr)
                    | AggregateCall::CountNulls(expr)
                    | AggregateCall::GroupConcat { expr, .. } => {
                        Self::collect_fields(expr, acc);
                    }
                }
            }
            ScalarExpr::GetField { base, .. } => {
                // Collect fields from the base expression
                Self::collect_fields(base, acc);
            }
            ScalarExpr::Cast { expr, .. } => {
                Self::collect_fields(expr, acc);
            }
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                if let Some(inner) = operand.as_deref() {
                    Self::collect_fields(inner, acc);
                }
                for (when_expr, then_expr) in branches {
                    Self::collect_fields(when_expr, acc);
                    Self::collect_fields(then_expr, acc);
                }
                if let Some(inner) = else_expr.as_deref() {
                    Self::collect_fields(inner, acc);
                }
            }
            ScalarExpr::Coalesce(items) => {
                for item in items {
                    Self::collect_fields(item, acc);
                }
            }
            ScalarExpr::Random => {
                // Random does not reference any fields
            }
            ScalarExpr::ScalarSubquery(_) => {
                // Scalar subqueries don't directly reference fields from the outer query
            }
        }
    }

    pub fn prepare_numeric_arrays<F: Hash + Eq + Copy>(
        arrays: &FxHashMap<F, ArrayRef>,
        _row_count: usize,
    ) -> NumericArrayMap<F> {
        arrays.clone()
    }

    /// Attempts to represent the expression as `scale * column + offset`.
    /// Returns `None` when the expression depends on multiple columns or is non-linear.
    pub fn extract_affine<F: Hash + Eq + Copy>(expr: &ScalarExpr<F>) -> Option<AffineExpr<F>> {
        let simplified = Self::simplify(expr);
        Self::extract_affine_simplified(&simplified)
    }

    /// Variant of `extract_affine` that assumes `expr` is already simplified.
    pub fn extract_affine_simplified<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
    ) -> Option<AffineExpr<F>> {
        let state = Self::affine_state(expr)?;
        let field = state.field?;
        Some(AffineExpr {
            field,
            scale: state.scale,
            offset: state.offset,
        })
    }

    fn affine_state<F: Hash + Eq + Copy>(expr: &ScalarExpr<F>) -> Option<AffineState<F>> {
        match expr {
            ScalarExpr::Column(fid) => Some(AffineState {
                field: Some(*fid),
                scale: 1.0,
                offset: 0.0,
            }),
            ScalarExpr::Literal(lit) => {
                let arr = Self::literal_to_array(lit);
                let val = cast::cast(&arr, &DataType::Float64).ok()?;
                let val = val.as_any().downcast_ref::<Float64Array>()?;
                if val.is_null(0) {
                    return None;
                }
                Some(AffineState {
                    field: None,
                    scale: 0.0,
                    offset: val.value(0),
                })
            }
            ScalarExpr::Aggregate(_) => None,
            ScalarExpr::GetField { .. } => None,
            ScalarExpr::Binary { left, op, right } => {
                let left_state = Self::affine_state(left)?;
                let right_state = Self::affine_state(right)?;
                match op {
                    BinaryOp::Add => Self::affine_add(left_state, right_state),
                    BinaryOp::Subtract => Self::affine_sub(left_state, right_state),
                    BinaryOp::Multiply => Self::affine_mul(left_state, right_state),
                    BinaryOp::Divide => Self::affine_div(left_state, right_state),
                    _ => None,
                }
            }
            ScalarExpr::Cast { expr, .. } => Self::affine_state(expr),
            _ => None,
        }
    }

    /// Evaluate a scalar expression for the row at `idx` using the provided numeric arrays.
    pub fn evaluate_value<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
        idx: usize,
        arrays: &NumericArrayMap<F>,
    ) -> LlkvResult<ArrayRef> {
        match expr {
            ScalarExpr::Column(fid) => {
                let array = arrays
                    .get(fid)
                    .ok_or_else(|| Error::Internal("missing column for field".into()))?;
                Ok(array.slice(idx, 1))
            }
            ScalarExpr::Literal(lit) => Ok(Self::literal_to_array(lit)),
            ScalarExpr::Binary { left, op, right } => {
                let l = Self::evaluate_value(left, idx, arrays)?;
                let r = Self::evaluate_value(right, idx, arrays)?;
                Self::evaluate_binary_scalar(&l, *op, &r)
            }
            ScalarExpr::Compare { left, op, right } => {
                let l = Self::evaluate_value(left, idx, arrays)?;
                let r = Self::evaluate_value(right, idx, arrays)?;
                crate::kernels::compute_compare(&l, *op, &r)
            }
            ScalarExpr::Not(expr) => {
                let val = Self::evaluate_value(expr, idx, arrays)?;
                let bool_arr = cast::cast(&val, &DataType::Boolean)
                    .map_err(|e| Error::Internal(e.to_string()))?;
                let bool_arr = bool_arr
                    .as_any()
                    .downcast_ref::<arrow::array::BooleanArray>()
                    .unwrap();
                let result = arrow::compute::kernels::boolean::not(bool_arr)
                    .map_err(|e| Error::Internal(e.to_string()))?;
                Ok(Arc::new(result))
            }
            ScalarExpr::IsNull { expr, negated } => {
                let val = Self::evaluate_value(expr, idx, arrays)?;
                let is_null = val.data_type() == &DataType::Null || val.is_null(0);
                let result = if *negated { !is_null } else { is_null };
                Ok(Arc::new(arrow::array::BooleanArray::from(vec![result])))
            }
            ScalarExpr::Cast { expr, data_type } => {
                let val = Self::evaluate_value(expr, idx, arrays)?;
                cast::cast(&val, data_type).map_err(|e| Error::Internal(e.to_string()))
            }
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                let operand_val = if let Some(op) = operand {
                    Some(Self::evaluate_value(op, idx, arrays)?)
                } else {
                    None
                };

                for (when_expr, then_expr) in branches {
                    let when_val = Self::evaluate_value(when_expr, idx, arrays)?;

                    let is_match = if let Some(op_val) = &operand_val {
                        // Simple CASE: operand = when_val
                        // If either is null, result is null (false for condition)
                        if op_val.data_type() == &DataType::Null
                            || when_val.data_type() == &DataType::Null
                            || op_val.is_null(0)
                            || when_val.is_null(0)
                        {
                            false
                        } else {
                            let eq =
                                crate::kernels::compute_compare(op_val, CompareOp::Eq, &when_val)?;
                            let bool_arr = eq
                                .as_any()
                                .downcast_ref::<arrow::array::BooleanArray>()
                                .unwrap();
                            bool_arr.is_valid(0) && bool_arr.value(0)
                        }
                    } else {
                        // Searched CASE: when_val is boolean condition
                        if when_val.data_type() == &DataType::Null || when_val.is_null(0) {
                            false
                        } else {
                            let bool_arr = cast::cast(&when_val, &DataType::Boolean)
                                .map_err(|e| Error::Internal(e.to_string()))?;
                            let bool_arr = bool_arr
                                .as_any()
                                .downcast_ref::<arrow::array::BooleanArray>()
                                .unwrap();
                            bool_arr.is_valid(0) && bool_arr.value(0)
                        }
                    };

                    if is_match {
                        return Self::evaluate_value(then_expr, idx, arrays);
                    }
                }
                if let Some(else_expr) = else_expr {
                    Self::evaluate_value(else_expr, idx, arrays)
                } else {
                    Ok(new_null_array(&DataType::Null, 1))
                }
            }
            ScalarExpr::Coalesce(items) => {
                for item in items {
                    let val = Self::evaluate_value(item, idx, arrays)?;
                    if !val.is_null(0) && val.data_type() != &DataType::Null {
                        return Ok(val);
                    }
                }
                Ok(new_null_array(&DataType::Null, 1))
            }
            ScalarExpr::Random => {
                let val = rand::random::<f64>();
                Ok(Arc::new(Float64Array::from(vec![val])))
            }
            ScalarExpr::Aggregate(call) => {
                let kind = match call {
                    AggregateCall::CountStar => "count_star",
                    AggregateCall::Count { .. } => "count",
                    AggregateCall::Sum { .. } => "sum",
                    AggregateCall::Total { .. } => "total",
                    AggregateCall::Avg { .. } => "avg",
                    AggregateCall::Min(_) => "min",
                    AggregateCall::Max(_) => "max",
                    AggregateCall::CountNulls(_) => "count_nulls",
                    AggregateCall::GroupConcat { .. } => "group_concat",
                };
                if std::env::var("LLKV_DEBUG_SUBQS").is_ok() {
                    eprintln!(
                        "[compute] evaluating aggregate in scalar context: {kind}"
                    );
                    eprintln!("{:?}", std::backtrace::Backtrace::capture());
                }
                Err(Error::Internal(format!(
                    "Unsupported aggregate scalar expression ({kind}); aggregate should be rewritten before evaluation"
                )))
            }
            ScalarExpr::GetField { .. } => Err(Error::Internal(
                "Unsupported get_field scalar expression".into(),
            )),
            ScalarExpr::ScalarSubquery(_) => Err(Error::Internal(
                "Unsupported scalar subquery expression".into(),
            )),
        }
    }

    fn literal_to_array(lit: &Literal) -> ArrayRef {
        match lit {
            Literal::Null => new_null_array(&DataType::Null, 1),
            Literal::Boolean(b) => Arc::new(arrow::array::BooleanArray::from(vec![*b])),
            Literal::Int128(i) => Arc::new(arrow::array::Int64Array::from(vec![*i as i64])),
            Literal::Float64(f) => Arc::new(Float64Array::from(vec![*f])),
            Literal::Decimal128(d) => {
                let precision = std::cmp::max(d.precision(), d.scale() as u8);
                let array = arrow::array::Decimal128Array::from(vec![Some(d.raw_value())])
                    .with_precision_and_scale(precision, d.scale())
                    .unwrap();
                Arc::new(array)
            }
            Literal::String(s) => Arc::new(arrow::array::StringArray::from(vec![s.clone()])),
            Literal::Date32(d) => Arc::new(arrow::array::Date32Array::from(vec![*d])),
            Literal::Interval(i) => {
                let val = IntervalMonthDayNanoType::make_value(i.months, i.days, i.nanos);
                Arc::new(arrow::array::IntervalMonthDayNanoArray::from(vec![val]))
            }
            Literal::Struct(fields) => {
                if fields.is_empty() {
                    let struct_array = StructArray::new_empty_fields(1, None);
                    return Arc::new(struct_array);
                }

                let mut arrow_fields = Vec::new();
                let mut child_arrays = Vec::new();

                for (name, val) in fields {
                    let arr = Self::literal_to_array(val);
                    let field = Field::new(name, arr.data_type().clone(), true);
                    arrow_fields.push(field);
                    child_arrays.push(arr);
                }

                let struct_array = StructArray::try_new(arrow_fields.into(), child_arrays, None)
                    .expect("failed to create struct literal array");
                Arc::new(struct_array)
            }
        }
    }

    fn evaluate_binary_scalar(
        lhs: &ArrayRef,
        op: BinaryOp,
        rhs: &ArrayRef,
    ) -> LlkvResult<ArrayRef> {
        compute_binary(lhs, rhs, op)
    }

    /// Evaluate a scalar expression for every row in the batch.
    #[allow(dead_code)]
    pub fn evaluate_batch<F: Hash + Eq + Copy + std::fmt::Debug>(
        expr: &ScalarExpr<F>,
        len: usize,
        arrays: &NumericArrayMap<F>,
    ) -> LlkvResult<ArrayRef> {
        let simplified = Self::simplify(expr);
        Self::evaluate_batch_simplified(&simplified, len, arrays)
    }

    /// Evaluate a scalar expression that has already been simplified.
    pub fn evaluate_batch_simplified<F: Hash + Eq + Copy + std::fmt::Debug>(
        expr: &ScalarExpr<F>,
        len: usize,
        arrays: &NumericArrayMap<F>,
    ) -> LlkvResult<ArrayRef> {
        let preferred = expr.infer_result_type_from_arrays(arrays);


        if len == 0 {
            return Ok(new_null_array(&preferred, 0));
        }

        if let Some(fast_path) = NumericFastPath::compile(expr, arrays, &preferred) {
            let fast_result = fast_path.execute(len, arrays)?;
            if fast_result.data_type() != &preferred {
                let casted = cast::cast(&fast_result, &preferred).map_err(|e| {
                    Error::Internal(format!("Failed to cast fast path result: {}", e))
                })?;
                return Ok(casted);
            }
            return Ok(fast_result);
        }

        if let Some(vectorized) = Self::try_evaluate_vectorized(expr, len, arrays)? {
            let result = vectorized.materialize(len);
            if result.data_type() != &preferred {
                let casted = cast::cast(&result, &preferred).map_err(|e| {
                    Error::Internal(format!("Failed to cast vectorized result: {}", e))
                })?;
                return Ok(casted);
            }
            return Ok(result);
        }

        let mut values = Vec::with_capacity(len);
        for idx in 0..len {
            let val = Self::evaluate_value(expr, idx, arrays)?;
            if val.data_type() != &preferred {
                let casted = cast::cast(&val, &preferred).map_err(|e| {
                    Error::Internal(format!(
                        "Failed to cast row {}: {} (Val type: {:?}, Preferred: {:?})",
                        idx,
                        e,
                        val.data_type(),
                        preferred
                    ))
                })?;
                values.push(casted);
            } else {
                values.push(val);
            }
        }
        let result = concat(&values.iter().map(|a| a.as_ref()).collect::<Vec<_>>())
            .map_err(|e| Error::Internal(e.to_string()))?;


        Ok(result)
    }

    fn try_evaluate_vectorized<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
        len: usize,
        arrays: &NumericArrayMap<F>,
    ) -> LlkvResult<Option<VectorizedExpr>> {
        if expr.contains_interval() {
            return Ok(None);
        }
        match expr {
            ScalarExpr::Column(fid) => {
                let array = arrays
                    .get(fid)
                    .ok_or_else(|| Error::Internal("missing column for field".into()))?;
                Ok(Some(VectorizedExpr::Array(array.clone())))
            }
            ScalarExpr::Literal(lit) => {
                let array = Self::literal_to_array(lit);
                Ok(Some(VectorizedExpr::Scalar(array)))
            }
            ScalarExpr::Binary { left, op, right } => {
                let left_vec = Self::try_evaluate_vectorized(left, len, arrays)?;
                let right_vec = Self::try_evaluate_vectorized(right, len, arrays)?;

                match (left_vec, right_vec) {
                    (Some(VectorizedExpr::Scalar(lhs)), Some(VectorizedExpr::Scalar(rhs))) => {
                        let result = compute_binary(&lhs, &rhs, *op)?;
                        Ok(Some(VectorizedExpr::Scalar(result)))
                    }
                    (Some(VectorizedExpr::Array(lhs)), Some(VectorizedExpr::Array(rhs))) => {
                        let array = compute_binary(&lhs, &rhs, *op)?;
                        Ok(Some(VectorizedExpr::Array(array)))
                    }
                    (Some(VectorizedExpr::Array(lhs)), Some(VectorizedExpr::Scalar(rhs))) => {
                        let rhs_expanded = VectorizedExpr::Scalar(rhs).materialize(lhs.len());
                        let array = compute_binary(&lhs, &rhs_expanded, *op)?;
                        Ok(Some(VectorizedExpr::Array(array)))
                    }
                    (Some(VectorizedExpr::Scalar(lhs)), Some(VectorizedExpr::Array(rhs))) => {
                        let lhs_expanded = VectorizedExpr::Scalar(lhs).materialize(rhs.len());
                        let array = compute_binary(&lhs_expanded, &rhs, *op)?;
                        Ok(Some(VectorizedExpr::Array(array)))
                    }
                    _ => Ok(None),
                }
            }
            ScalarExpr::Compare { left, op, right } => {
                let left_vec = Self::try_evaluate_vectorized(left, len, arrays)?;
                let right_vec = Self::try_evaluate_vectorized(right, len, arrays)?;

                match (left_vec, right_vec) {
                    (Some(VectorizedExpr::Scalar(lhs)), Some(VectorizedExpr::Scalar(rhs))) => {
                        let result = compute_compare(&lhs, *op, &rhs)?;
                        Ok(Some(VectorizedExpr::Scalar(result)))
                    }
                    (Some(VectorizedExpr::Array(lhs)), Some(VectorizedExpr::Array(rhs))) => {
                        let result = compute_compare(&lhs, *op, &rhs)?;
                        Ok(Some(VectorizedExpr::Array(result)))
                    }
                    (Some(VectorizedExpr::Array(lhs)), Some(VectorizedExpr::Scalar(rhs))) => {
                        let rhs_expanded = VectorizedExpr::Scalar(rhs).materialize(lhs.len());
                        let result = compute_compare(&lhs, *op, &rhs_expanded)?;
                        Ok(Some(VectorizedExpr::Array(result)))
                    }
                    (Some(VectorizedExpr::Scalar(lhs)), Some(VectorizedExpr::Array(rhs))) => {
                        let lhs_expanded = VectorizedExpr::Scalar(lhs).materialize(rhs.len());
                        let result = compute_compare(&lhs_expanded, *op, &rhs)?;
                        Ok(Some(VectorizedExpr::Array(result)))
                    }
                    _ => Ok(None),
                }
            }
            ScalarExpr::Cast { expr, data_type } => {
                let inner_vec = Self::try_evaluate_vectorized(expr, len, arrays)?;

                match inner_vec {
                    Some(VectorizedExpr::Scalar(array)) => {
                        let casted = cast::cast(&array, data_type)
                            .map_err(|e| Error::Internal(e.to_string()))?;
                        Ok(Some(VectorizedExpr::Scalar(casted)))
                    }
                    Some(VectorizedExpr::Array(array)) => {
                        let casted = cast::cast(&array, data_type)
                            .map_err(|e| Error::Internal(e.to_string()))?;
                        Ok(Some(VectorizedExpr::Array(casted)))
                    }
                    None => Ok(None),
                }
            }
            ScalarExpr::Coalesce(items) => {
                let mut evaluated_items = Vec::with_capacity(items.len());
                let mut types = Vec::with_capacity(items.len());

                for item in items {
                    let vec_expr = match Self::try_evaluate_vectorized(item, len, arrays)? {
                        Some(v) => v,
                        None => return Ok(None),
                    };

                    let array = vec_expr.materialize(len);
                    types.push(array.data_type().clone());
                    evaluated_items.push(array);
                }

                if evaluated_items.is_empty() {
                    return Ok(Some(VectorizedExpr::Array(new_null_array(
                        &DataType::Null,
                        len,
                    ))));
                }

                // Determine common type
                let mut common_type = types[0].clone();
                for t in &types[1..] {
                    common_type = get_common_type(&common_type, t);
                }

                // Cast all arrays to common type
                let mut casted_arrays = Vec::with_capacity(evaluated_items.len());
                for array in evaluated_items {
                    if array.data_type() != &common_type {
                        let casted = cast::cast(&array, &common_type)
                            .map_err(|e| Error::Internal(e.to_string()))?;
                        casted_arrays.push(casted);
                    } else {
                        casted_arrays.push(array);
                    }
                }

                let mut result = casted_arrays[0].clone();
                for next_array in &casted_arrays[1..] {
                    let mask = is_not_null(&result).map_err(|e| Error::Internal(e.to_string()))?;
                    // result = zip(mask, result, next_array)
                    // if mask is true (result is not null), keep result.
                    // if mask is false (result is null), take next_array.
                    result = zip(&mask, &result, next_array)
                        .map_err(|e| Error::Internal(e.to_string()))?;
                }
                Ok(Some(VectorizedExpr::Array(result)))
            }
            ScalarExpr::Random => {
                let values: Vec<f64> = (0..len).map(|_| rand::random::<f64>()).collect();
                let array = Float64Array::from(values);
                Ok(Some(VectorizedExpr::Array(Arc::new(array))))
            }
            _ => Ok(None),
        }
    }

    /// Returns the column referenced by an expression when it's a direct or additive identity passthrough.
    pub fn passthrough_column<F: Hash + Eq + Copy>(expr: &ScalarExpr<F>) -> Option<F> {
        match Self::simplify(expr) {
            ScalarExpr::Column(fid) => Some(fid),
            _ => None,
        }
    }

    fn is_null<F>(expr: &ScalarExpr<F>) -> bool {
        match expr {
            ScalarExpr::Literal(Literal::Null) => true,
            // We do NOT recurse into Cast here because we want to preserve the type information
            // of CAST(NULL AS T). If we identify it as null and simplify it to Literal::Null,
            // we lose the type T, which causes schema mismatches in RecordBatch creation.
            // ScalarExpr::Cast { expr, .. } => Self::is_null(expr),
            _ => false,
        }
    }

    fn contains_aggregate<F>(expr: &ScalarExpr<F>) -> bool {
        match expr {
            ScalarExpr::Aggregate(_) => true,
            ScalarExpr::Binary { left, right, .. } => {
                Self::contains_aggregate(left) || Self::contains_aggregate(right)
            }
            ScalarExpr::Compare { left, right, .. } => {
                Self::contains_aggregate(left) || Self::contains_aggregate(right)
            }
            ScalarExpr::Not(expr) => Self::contains_aggregate(expr),
            ScalarExpr::IsNull { expr, .. } => Self::contains_aggregate(expr),
            ScalarExpr::Cast { expr, .. } => Self::contains_aggregate(expr),
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                operand
                    .as_ref()
                    .map_or(false, |e| Self::contains_aggregate(e))
                    || branches.iter().any(|(w, t)| {
                        Self::contains_aggregate(w) || Self::contains_aggregate(t)
                    })
                    || else_expr
                        .as_ref()
                        .map_or(false, |e| Self::contains_aggregate(e))
            }
            ScalarExpr::Coalesce(exprs) => exprs.iter().any(Self::contains_aggregate),
            ScalarExpr::GetField { base, .. } => Self::contains_aggregate(base),
            _ => false,
        }
    }

    /// Simplify an expression by constant folding and identity removal.
    pub fn simplify<F: Hash + Eq + Clone>(expr: &ScalarExpr<F>) -> ScalarExpr<F> {
        let try_get_type = |e: &ScalarExpr<F>| -> Option<DataType> {
            match e {
                ScalarExpr::Cast { data_type, .. } => Some(data_type.clone()),
                ScalarExpr::Literal(lit) => match lit {
                    Literal::Int128(_) => Some(DataType::Int64),
                    Literal::Float64(_) => Some(DataType::Float64),
                    Literal::Boolean(_) => Some(DataType::Boolean),
                    Literal::String(_) => Some(DataType::Utf8),
                    Literal::Date32(_) => Some(DataType::Date32),
                    Literal::Decimal128(d) => Some(DataType::Decimal128(d.precision(), d.scale())),
                    _ => None,
                },
                _ => None,
            }
        };

        match expr {
            ScalarExpr::Binary { left, op, right } => {
                let l = Self::simplify(left);
                let r = Self::simplify(right);

                let is_logic_op = matches!(op, BinaryOp::And | BinaryOp::Or);
                
                let is_effectively_null = |e: &ScalarExpr<F>| {
                    Self::is_null(e) || matches!(e, ScalarExpr::Cast { expr, .. } if Self::is_null(expr))
                };

                if !is_logic_op && (is_effectively_null(&l) || is_effectively_null(&r)) {
                    // If one side is NULL, the result is NULL (for most ops).
                    // However, if the other side contains an aggregate, we MUST NOT simplify it away.
                    // This is because the presence of an aggregate changes the query semantics
                    // (e.g. scalar aggregation on empty input returns 1 row, whereas simple select returns 0 rows).
                    // If we simplify `COUNT(*) + NULL` to `NULL`, the planner might see no aggregates
                    // and treat it as a simple select, returning 0 rows instead of 1 row with NULL.
                    let discard_aggregate = (is_effectively_null(&l) && Self::contains_aggregate(&r))
                        || (is_effectively_null(&r) && Self::contains_aggregate(&l));

                    if !discard_aggregate {
                        let type_l = try_get_type(&l);
                        let type_r = try_get_type(&r);

                        let target_type = match (type_l, type_r) {
                            (Some(t1), Some(t2)) => Some(get_common_type(&t1, &t2)),
                            (Some(t), None) => Some(t),
                            (None, Some(t)) => Some(t),
                            (None, None) => None,
                        };

                        if let Some(dt) = target_type {
                            return ScalarExpr::Cast {
                                expr: Box::new(ScalarExpr::Literal(Literal::Null)),
                                data_type: dt,
                            };
                        }

                        // If both are untyped nulls, return Literal::Null
                        if Self::is_null(&l) && Self::is_null(&r) {
                            return ScalarExpr::Literal(Literal::Null);
                        }
                    }
                }

                if let (ScalarExpr::Literal(ll), ScalarExpr::Literal(rr)) = (&l, &r)
                    && let Some(folded) = fold_binary_literals(*op, ll, rr)
                {
                    return ScalarExpr::Literal(folded);
                }
                ScalarExpr::Binary {
                    left: Box::new(l),
                    op: *op,
                    right: Box::new(r),
                }
            }
            ScalarExpr::Compare { left, op, right } => {
                let l = Self::simplify(left);
                let r = Self::simplify(right);

                let is_effectively_null = |e: &ScalarExpr<F>| {
                    Self::is_null(e) || matches!(e, ScalarExpr::Cast { expr, .. } if Self::is_null(expr))
                };

                // Simplify to Literal::Null if EITHER is effectively null.
                if is_effectively_null(&l) || is_effectively_null(&r) {
                    return ScalarExpr::Literal(Literal::Null);
                }

                if let (ScalarExpr::Literal(ll), ScalarExpr::Literal(rr)) = (&l, &r) {
                    let l_arr = Self::literal_to_array(ll);
                    let r_arr = Self::literal_to_array(rr);
                    if let Ok(res) = crate::kernels::compute_compare(&l_arr, *op, &r_arr) {
                         if let Ok(lit) = Literal::from_array_ref(&res, 0) {
                             return ScalarExpr::Literal(lit);
                         }
                    }
                }

                ScalarExpr::Compare {
                    left: Box::new(l),
                    op: *op,
                    right: Box::new(r),
                }
            }
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                let operand = operand.as_ref().map(|o| Box::new(Self::simplify(o)));
                let s_else = else_expr.as_ref().map(|e| Box::new(Self::simplify(e)));

                // Helper to check if an expression carries type info (is not Literal::Null)
                let is_typed = |e: &ScalarExpr<F>| !Self::is_null(e);

                let mut simplified_branches = Vec::with_capacity(branches.len());
                for (when, then) in branches {
                    simplified_branches.push((Self::simplify(when), Self::simplify(then)));
                }

                // Helper to find a typed fallback if the result is untyped NULL
                let get_safe_result = |result: ScalarExpr<F>, branches: &[(ScalarExpr<F>, ScalarExpr<F>)]| -> ScalarExpr<F> {
                    if is_typed(&result) {
                        return result;
                    }

                    // Result is untyped NULL. Check if we are discarding any typed branches.
                    for (_, then) in branches {
                        if is_typed(then) {
                            if let Some(dt) = try_get_type(then) {
                                return ScalarExpr::Cast {
                                    expr: Box::new(ScalarExpr::Literal(Literal::Null)),
                                    data_type: dt,
                                };
                            }
                            // Found a typed branch. Return a dummy CASE to preserve type.
                            return ScalarExpr::Case {
                                operand: None,
                                branches: vec![(ScalarExpr::Literal(Literal::Boolean(false)), then.clone())],
                                else_expr: Some(Box::new(result)),
                            };
                        }
                    }
                    if let Some(e) = &s_else {
                        if is_typed(e) {
                             if let Some(dt) = try_get_type(e) {
                                return ScalarExpr::Cast {
                                    expr: Box::new(ScalarExpr::Literal(Literal::Null)),
                                    data_type: dt,
                                };
                            }
                             return ScalarExpr::Case {
                                operand: None,
                                branches: vec![(ScalarExpr::Literal(Literal::Boolean(false)), *e.clone())],
                                else_expr: Some(Box::new(result)),
                            };
                        }
                    }
                    result
                };

                // If operand is NULL, the whole CASE expression evaluates to ELSE
                if let Some(op) = &operand {
                    if Self::is_null(op) {
                        let result = s_else
                            .clone()
                            .map(|b| *b)
                            .unwrap_or(ScalarExpr::Literal(Literal::Null));
                        return get_safe_result(result, &simplified_branches);
                    }
                }

                let mut new_branches = Vec::new();

                for (s_when, s_then) in simplified_branches.iter().cloned() {
                    if Self::is_null(&s_when) {
                        continue;
                    }

                    // If we have a simple CASE (operand is Some)
                    if let Some(op) = &operand {
                        if let (ScalarExpr::Literal(op_lit), ScalarExpr::Literal(when_lit)) = (op.as_ref(), &s_when) {
                            if op_lit == when_lit {
                                return get_safe_result(s_then, &simplified_branches);
                            } else {
                                continue;
                            }
                        }
                    }

                    // If we have a searched CASE (operand is None)
                    if operand.is_none() {
                        if let ScalarExpr::Literal(lit) = &s_when {
                            // If condition is FALSE, skip this branch
                            if matches!(lit, Literal::Boolean(false)) {
                                continue;
                            }
                            // If condition is TRUE, return this branch and ignore the rest
                            if matches!(lit, Literal::Boolean(true)) {
                                return get_safe_result(s_then, &simplified_branches);
                            }
                        }
                    }
                    new_branches.push((s_when, s_then));
                }

                if new_branches.is_empty() {
                    let result = s_else
                        .clone()
                        .map(|b| *b)
                        .unwrap_or(ScalarExpr::Literal(Literal::Null));
                    return get_safe_result(result, &simplified_branches);
                }

                ScalarExpr::Case {
                    operand,
                    branches: new_branches,
                    else_expr: s_else,
                }
            }
            ScalarExpr::Coalesce(items) => {
                let mut simplified_items = Vec::with_capacity(items.len());
                let mut non_null_items = Vec::new();

                for item in items {
                    let s_item = Self::simplify(item);
                    simplified_items.push(s_item.clone());

                    // Check if item is effectively NULL (Literal::Null, Cast(Null), or Aggregate(Null))
                    let is_effectively_null = if Self::is_null(&s_item) {
                        true
                    } else if let ScalarExpr::Cast { expr, .. } = &s_item {
                        Self::is_null(expr)
                    } else if let ScalarExpr::Aggregate(agg) = &s_item {
                        match agg {
                            AggregateCall::Avg { expr, .. }
                            | AggregateCall::Sum { expr, .. }
                            | AggregateCall::Min(expr)
                            | AggregateCall::Max(expr) => {
                                let inner = Self::simplify(expr);
                                Self::is_null(&inner)
                                    || (matches!(inner, ScalarExpr::Cast { expr, .. } if Self::is_null(&expr)))
                            }
                            // Count(NULL) is 0, not NULL.
                            _ => false,
                        }
                    } else {
                        false
                    };

                    if is_effectively_null {
                        continue;
                    }

                    if let ScalarExpr::Literal(lit) = &s_item {
                        // Found non-null literal.
                        if non_null_items.is_empty() {
                            return ScalarExpr::Literal(lit.clone());
                        }
                        // We have preceding non-constant expressions.
                        // This literal terminates the coalesce chain.
                        non_null_items.push(s_item);
                        break;
                    }
                    non_null_items.push(s_item);
                }

                if non_null_items.is_empty() {
                    // All items were effectively NULL.
                    // To preserve types (e.g. AVG(NULL) is Double, Literal::Null is Untyped),
                    // we should return the original simplified items if they contain any typed expressions.
                    // If all were Literal::Null, we can return Literal::Null.
                    let all_literals = simplified_items
                        .iter()
                        .all(|i| matches!(i, ScalarExpr::Literal(Literal::Null)));
                    if all_literals {
                        return ScalarExpr::Literal(Literal::Null);
                    }
                    return ScalarExpr::Coalesce(simplified_items);
                }

                if non_null_items.len() == 1 {
                    return non_null_items[0].clone();
                }

                ScalarExpr::Coalesce(non_null_items)
            }
            ScalarExpr::Aggregate(agg) => {
                let simplified_agg = match agg {
                    AggregateCall::Avg { expr, distinct } => ScalarExpr::Aggregate(AggregateCall::Avg {
                        expr: Box::new(Self::simplify(expr)),
                        distinct: *distinct,
                    }),
                    AggregateCall::Sum { expr, distinct } => ScalarExpr::Aggregate(AggregateCall::Sum {
                        expr: Box::new(Self::simplify(expr)),
                        distinct: *distinct,
                    }),
                    AggregateCall::Total { expr, distinct } => ScalarExpr::Aggregate(AggregateCall::Total {
                        expr: Box::new(Self::simplify(expr)),
                        distinct: *distinct,
                    }),
                    AggregateCall::Min(expr) => {
                        ScalarExpr::Aggregate(AggregateCall::Min(Box::new(Self::simplify(expr))))
                    }
                    AggregateCall::Max(expr) => {
                        ScalarExpr::Aggregate(AggregateCall::Max(Box::new(Self::simplify(expr))))
                    }
                    AggregateCall::Count { expr, distinct } => {
                        ScalarExpr::Aggregate(AggregateCall::Count {
                            expr: Box::new(Self::simplify(expr)),
                            distinct: *distinct,
                        })
                    }
                    AggregateCall::CountNulls(expr) => {
                        ScalarExpr::Aggregate(AggregateCall::CountNulls(Box::new(Self::simplify(expr))))
                    }
                    AggregateCall::GroupConcat {
                        expr,
                        distinct,
                        separator,
                    } => ScalarExpr::Aggregate(AggregateCall::GroupConcat {
                        expr: Box::new(Self::simplify(expr)),
                        distinct: *distinct,
                        separator: separator.clone(),
                    }),
                    AggregateCall::CountStar => ScalarExpr::Aggregate(AggregateCall::CountStar),
                };

                // Check for effective nulls and return typed Nulls if possible
                match &simplified_agg {
                    ScalarExpr::Aggregate(agg_call) => {
                        match agg_call {
                            AggregateCall::Avg { expr, .. } => {
                                if Self::is_null(expr) {
                                    return ScalarExpr::Cast {
                                        expr: Box::new(ScalarExpr::Literal(Literal::Null)),
                                        data_type: DataType::Float64,
                                    };
                                }
                            }
                            AggregateCall::Min(expr) | AggregateCall::Max(expr) => {
                                if Self::is_null(expr) {
                                    if let Some(dt) = try_get_type(expr) {
                                         return ScalarExpr::Cast {
                                            expr: Box::new(ScalarExpr::Literal(Literal::Null)),
                                            data_type: dt,
                                        };
                                    }
                                }
                            }
                            AggregateCall::Sum { expr, .. } | AggregateCall::Total { expr, .. } => {
                                if Self::is_null(expr) {
                                    if let Some(dt) = try_get_type(expr) {
                                         return ScalarExpr::Cast {
                                            expr: Box::new(ScalarExpr::Literal(Literal::Null)),
                                            data_type: dt,
                                        };
                                    }
                                }
                            }
                            AggregateCall::Count { expr, .. } => {
                                if Self::is_null(expr) {
                                    return ScalarExpr::Literal(Literal::Int128(0));
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
                simplified_agg
            }
            ScalarExpr::IsNull { expr, negated } => {
                let inner = Self::simplify(expr);
                let is_effectively_null = if let ScalarExpr::Literal(Literal::Null) = &inner {
                    true
                } else if let ScalarExpr::Cast { expr, .. } = &inner {
                    matches!(expr.as_ref(), ScalarExpr::Literal(Literal::Null))
                } else {
                    false
                };

                if is_effectively_null {
                    return ScalarExpr::Literal(Literal::Boolean(!*negated));
                }

                if let ScalarExpr::Literal(_) = &inner {
                    return ScalarExpr::Literal(Literal::Boolean(*negated));
                }

                ScalarExpr::IsNull {
                    expr: Box::new(inner),
                    negated: *negated,
                }
            }
            ScalarExpr::Not(expr) => {
                let inner = Self::simplify(expr);
                if let ScalarExpr::Literal(Literal::Boolean(b)) = &inner {
                    return ScalarExpr::Literal(Literal::Boolean(!b));
                }
                ScalarExpr::Not(Box::new(inner))
            }
            ScalarExpr::Cast { expr, data_type } => {
                let inner = Self::simplify(expr);
                if let ScalarExpr::Literal(lit) = &inner
                    && let Some(folded) = fold_cast_literal(lit, data_type)
                {
                    return ScalarExpr::Literal(folded);
                }
                ScalarExpr::Cast {
                    expr: Box::new(inner),
                    data_type: data_type.clone(),
                }
            }
            _ => expr.clone(),
        }
    }

    pub fn evaluate_constant_literal_expr<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
    ) -> LlkvResult<Option<Literal>> {
        let simplified = Self::simplify(expr);

        if let Some(literal) = Self::evaluate_constant_literal_non_numeric(&simplified)? {
            return Ok(Some(literal));
        }

        if let ScalarExpr::Literal(lit) = &simplified {
            return Ok(Some(lit.clone()));
        }

        let arrays = NumericArrayMap::default();
        let array = Self::evaluate_value(&simplified, 0, &arrays)?;
        if array.is_null(0) {
            return Ok(None);
        }
        Ok(Some(Literal::from_array_ref(&array, 0)?))
    }

    pub fn evaluate_constant_literal_non_numeric<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
    ) -> LlkvResult<Option<Literal>> {
        match expr {
            ScalarExpr::Literal(lit) => Ok(Some(lit.clone())),
            ScalarExpr::Cast {
                expr,
                data_type: DataType::Date32,
            } => {
                let inner = Self::evaluate_constant_literal_non_numeric(expr)?;
                match inner {
                    Some(Literal::Null) => Ok(Some(Literal::Null)),
                    Some(Literal::String(text)) => {
                        let days = parse_date32_literal(&text)?;
                        Ok(Some(Literal::Date32(days)))
                    }
                    Some(Literal::Date32(days)) => Ok(Some(Literal::Date32(days))),
                    Some(other) => Err(Error::InvalidArgumentError(format!(
                        "cannot cast literal of type {} to DATE",
                        other.type_name()
                    ))),
                    None => Ok(None),
                }
            }
            ScalarExpr::Cast { .. } => Ok(None),
            ScalarExpr::Binary { left, op, right } => {
                let left_lit = match Self::evaluate_constant_literal_non_numeric(left)? {
                    Some(lit) => lit,
                    None => return Ok(None),
                };
                let right_lit = match Self::evaluate_constant_literal_non_numeric(right)? {
                    Some(lit) => lit,
                    None => return Ok(None),
                };

                if matches!(left_lit, Literal::Null) || matches!(right_lit, Literal::Null) {
                    return Ok(Some(Literal::Null));
                }

                match op {
                    BinaryOp::Add => match (&left_lit, &right_lit) {
                        (Literal::Date32(days), Literal::Interval(interval))
                        | (Literal::Interval(interval), Literal::Date32(days)) => {
                            let adjusted = add_interval_to_date32(*days, *interval)?;
                            Ok(Some(Literal::Date32(adjusted)))
                        }
                        (Literal::Interval(left), Literal::Interval(right)) => {
                            let sum = left.checked_add(*right).ok_or_else(|| {
                                Error::InvalidArgumentError(
                                    "interval addition overflow during constant folding".into(),
                                )
                            })?;
                            Ok(Some(Literal::Interval(sum)))
                        }
                        _ => Ok(None),
                    },
                    BinaryOp::Subtract => match (&left_lit, &right_lit) {
                        (Literal::Date32(days), Literal::Interval(interval)) => {
                            let adjusted = subtract_interval_from_date32(*days, *interval)?;
                            Ok(Some(Literal::Date32(adjusted)))
                        }
                        (Literal::Interval(left), Literal::Interval(right)) => {
                            let diff = left.checked_sub(*right).ok_or_else(|| {
                                Error::InvalidArgumentError(
                                    "interval subtraction overflow during constant folding".into(),
                                )
                            })?;
                            Ok(Some(Literal::Interval(diff)))
                        }
                        (Literal::Date32(lhs), Literal::Date32(rhs)) => {
                            let delta = i64::from(*lhs) - i64::from(*rhs);
                            if delta < i64::from(i32::MIN) || delta > i64::from(i32::MAX) {
                                return Err(Error::InvalidArgumentError(
                                    "DATE subtraction overflowed day precision".into(),
                                ));
                            }
                            Ok(Some(Literal::Interval(IntervalValue::new(
                                0,
                                delta as i32,
                                0,
                            ))))
                        }
                        _ => Ok(None),
                    },
                    _ => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }

    // TODO: Should Decimal types be included here?
    pub fn is_supported_numeric(dtype: &DataType) -> bool {
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

    #[allow(dead_code)]
    fn affine_add<F: Eq + Copy>(
        lhs: AffineState<F>,
        rhs: AffineState<F>,
    ) -> Option<AffineState<F>> {
        let merged_field = Self::merge_field(lhs.field, rhs.field)?;
        if merged_field.is_none() {
            // Both constant
            return Some(AffineState {
                field: None,
                scale: 0.0,
                offset: lhs.offset + rhs.offset,
            });
        }
        Some(AffineState {
            field: merged_field,
            scale: lhs.scale + rhs.scale,
            offset: lhs.offset + rhs.offset,
        })
    }

    #[allow(dead_code)]
    fn affine_sub<F: Eq + Copy>(
        lhs: AffineState<F>,
        rhs: AffineState<F>,
    ) -> Option<AffineState<F>> {
        let merged_field = Self::merge_field(lhs.field, rhs.field)?;
        if merged_field.is_none() {
            return Some(AffineState {
                field: None,
                scale: 0.0,
                offset: lhs.offset - rhs.offset,
            });
        }
        Some(AffineState {
            field: merged_field,
            scale: lhs.scale - rhs.scale,
            offset: lhs.offset - rhs.offset,
        })
    }

    #[allow(dead_code)]
    fn affine_mul<F: Eq + Copy>(
        lhs: AffineState<F>,
        rhs: AffineState<F>,
    ) -> Option<AffineState<F>> {
        if lhs.field.is_some() && rhs.field.is_some() {
            return None; // Non-linear
        }
        if lhs.field.is_none() {
            let factor = lhs.offset;
            return Some(AffineState {
                field: rhs.field,
                scale: rhs.scale * factor,
                offset: rhs.offset * factor,
            });
        }
        if rhs.field.is_none() {
            let factor = rhs.offset;
            return Some(AffineState {
                field: lhs.field,
                scale: lhs.scale * factor,
                offset: lhs.offset * factor,
            });
        }
        None
    }

    #[allow(dead_code)]
    fn affine_div<F: Eq + Copy>(
        lhs: AffineState<F>,
        rhs: AffineState<F>,
    ) -> Option<AffineState<F>> {
        if rhs.field.is_some() {
            return None;
        }
        let denom = rhs.offset;
        if denom == 0.0 {
            return None;
        }
        Some(AffineState {
            field: lhs.field,
            scale: lhs.scale / denom,
            offset: lhs.offset / denom,
        })
    }
}

fn fold_binary_literals(op: BinaryOp, left: &Literal, right: &Literal) -> Option<Literal> {
    match op {
        BinaryOp::BitwiseShiftLeft | BinaryOp::BitwiseShiftRight => {
            let pg_op = match op {
                BinaryOp::BitwiseShiftLeft => BinaryOperator::PGBitwiseShiftLeft,
                BinaryOp::BitwiseShiftRight => BinaryOperator::PGBitwiseShiftRight,
                _ => unreachable!(),
            };
            crate::literal::bitshift_literals(pg_op, left, right).ok()
        }
        _ => {
            let l_arr = ScalarEvaluator::literal_to_array(left);
            let r_arr = ScalarEvaluator::literal_to_array(right);
            let result = compute_binary(&l_arr, &r_arr, op).ok()?;
            if result.is_null(0) {
                Some(Literal::Null)
            } else {
                Literal::from_array_ref(&result, 0).ok()
            }
        }
    }
}

fn fold_cast_literal(lit: &Literal, data_type: &DataType) -> Option<Literal> {
    if matches!(lit, Literal::Null) {
        // Preserve explicit casts of NULL so the target type is kept.
        return None;
    }
    let arr = ScalarEvaluator::literal_to_array(lit);
    let casted = cast::cast(&arr, data_type).ok()?;
    if casted.is_null(0) {
        Some(Literal::Null)
    } else {
        Literal::from_array_ref(&casted, 0).ok()
    }
}
