//! Numeric scalar expression evaluation utilities for table scans.
//!
//! Planner and executor components leverage these helpers to coerce input columns
//! into a minimal numeric representation and apply lightweight kernels without
//! duplicating logic throughout the scan pipeline.

use std::{convert::TryFrom, sync::Arc};

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::{BinaryOp, CompareOp, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::types::FieldId;

/// Mapping from field identifiers to the numeric Arrow array used for evaluation.
pub type NumericArrayMap = FxHashMap<FieldId, NumericArray>;

/// Describes whether a numeric value is represented as an integer or a float.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericKind {
    Integer,
    Float,
}

/// Holds a numeric value while preserving whether it originated as an integer or float.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NumericValue {
    Integer(i64),
    Float(f64),
}

impl NumericValue {
    #[inline]
    pub fn as_f64(self) -> f64 {
        match self {
            NumericValue::Integer(v) => v as f64,
            NumericValue::Float(v) => v,
        }
    }

    #[inline]
    pub fn as_i64(self) -> Option<i64> {
        match self {
            NumericValue::Integer(v) => Some(v),
            NumericValue::Float(_) => None,
        }
    }

    #[inline]
    pub fn kind(self) -> NumericKind {
        match self {
            NumericValue::Integer(_) => NumericKind::Integer,
            NumericValue::Float(_) => NumericKind::Float,
        }
    }
}

impl From<i64> for NumericValue {
    fn from(value: i64) -> Self {
        NumericValue::Integer(value)
    }
}

impl From<f64> for NumericValue {
    fn from(value: f64) -> Self {
        NumericValue::Float(value)
    }
}

/// Wraps an Arrow array that stores numeric values alongside its numeric kind.
#[derive(Clone)]
pub struct NumericArray {
    kind: NumericKind,
    len: usize,
    int_data: Option<Arc<Int64Array>>,
    float_data: Option<Arc<Float64Array>>,
}

impl NumericArray {
    pub(crate) fn from_int(array: Arc<Int64Array>) -> Self {
        let len = array.len();
        Self {
            kind: NumericKind::Integer,
            len,
            int_data: Some(array),
            float_data: None,
        }
    }

    pub(crate) fn from_float(array: Arc<Float64Array>) -> Self {
        let len = array.len();
        Self {
            kind: NumericKind::Float,
            len,
            int_data: None,
            float_data: Some(array),
        }
    }

    /// Build a [`NumericArray`] from an Arrow array, casting when necessary.
    pub fn try_from_arrow(array: &ArrayRef) -> LlkvResult<Self> {
        match array.data_type() {
            DataType::Int64 => {
                let int_array = array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| Error::Internal("expected Int64 array".into()))?
                    .clone();
                Ok(NumericArray::from_int(Arc::new(int_array)))
            }
            DataType::Float64 => {
                let float_array = array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| Error::Internal("expected Float64 array".into()))?
                    .clone();
                Ok(NumericArray::from_float(Arc::new(float_array)))
            }
            DataType::Int8 | DataType::Int16 | DataType::Int32 => {
                let casted = cast(array.as_ref(), &DataType::Int64)
                    .map_err(|e| Error::Internal(format!("cast to Int64 failed: {e}")))?;
                let int_array = casted
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| Error::Internal("cast produced non-Int64 array".into()))?
                    .clone();
                Ok(NumericArray::from_int(Arc::new(int_array)))
            }
            DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32 => {
                let casted = cast(array.as_ref(), &DataType::Float64)
                    .map_err(|e| Error::Internal(format!("cast to Float64 failed: {e}")))?;
                let float_array = casted
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| Error::Internal("cast produced non-Float64 array".into()))?
                    .clone();
                Ok(NumericArray::from_float(Arc::new(float_array)))
            }
            DataType::Boolean => {
                let casted = cast(array.as_ref(), &DataType::Int64)
                    .map_err(|e| Error::Internal(format!("cast to Int64 failed: {e}")))?;
                let int_array = casted
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| Error::Internal("cast produced non-Int64 array".into()))?
                    .clone();
                Ok(NumericArray::from_int(Arc::new(int_array)))
            }
            DataType::Null => {
                let float_array = Float64Array::from(vec![None; array.len()]);
                Ok(NumericArray::from_float(Arc::new(float_array)))
            }
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported data type in numeric kernel: {other:?}"
            ))),
        }
    }

    #[inline]
    pub fn kind(&self) -> NumericKind {
        self.kind
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn value(&self, idx: usize) -> Option<NumericValue> {
        match self.kind {
            NumericKind::Integer => {
                let array = self
                    .int_data
                    .as_ref()
                    .expect("integer array missing backing data");
                if array.is_null(idx) {
                    None
                } else {
                    Some(NumericValue::Integer(array.value(idx)))
                }
            }
            NumericKind::Float => {
                let array = self
                    .float_data
                    .as_ref()
                    .expect("float array missing backing data");
                if array.is_null(idx) {
                    None
                } else {
                    Some(NumericValue::Float(array.value(idx)))
                }
            }
        }
    }

    fn to_array_ref(&self) -> ArrayRef {
        match self.kind {
            NumericKind::Integer => Arc::clone(
                self.int_data
                    .as_ref()
                    .expect("integer array missing backing data"),
            ) as ArrayRef,
            NumericKind::Float => Arc::clone(
                self.float_data
                    .as_ref()
                    .expect("float array missing backing data"),
            ) as ArrayRef,
        }
    }

    fn promote_to_float(&self) -> NumericArray {
        match self.kind {
            NumericKind::Float => self.clone(),
            NumericKind::Integer => {
                let array = self
                    .int_data
                    .as_ref()
                    .expect("integer array missing backing data");
                let iter = (0..self.len).map(|idx| {
                    if array.is_null(idx) {
                        None
                    } else {
                        Some(array.value(idx) as f64)
                    }
                });
                let float_array = Float64Array::from_iter(iter);
                NumericArray::from_float(Arc::new(float_array))
            }
        }
    }

    fn to_aligned_array_ref(&self, preferred: NumericKind) -> ArrayRef {
        match (preferred, self.kind) {
            (NumericKind::Float, NumericKind::Integer) => self.promote_to_float().to_array_ref(),
            _ => self.to_array_ref(),
        }
    }

    fn from_numeric_values(values: Vec<Option<NumericValue>>, preferred: NumericKind) -> Self {
        let contains_float = values
            .iter()
            .any(|opt| matches!(opt, Some(NumericValue::Float(_))));
        match (contains_float, preferred) {
            (true, _) => {
                let iter = values.into_iter().map(|opt| opt.map(|v| v.as_f64()));
                let array = Float64Array::from_iter(iter);
                NumericArray::from_float(Arc::new(array))
            }
            (false, NumericKind::Float) => {
                let iter = values.into_iter().map(|opt| opt.map(|v| v.as_f64()));
                let array = Float64Array::from_iter(iter);
                NumericArray::from_float(Arc::new(array))
            }
            (false, NumericKind::Integer) => {
                let iter = values
                    .into_iter()
                    .map(|opt| opt.map(|v| v.as_i64().expect("expected integer")));
                let array = Int64Array::from_iter(iter);
                NumericArray::from_int(Arc::new(array))
            }
        }
    }
}

/// Intermediate representation for vectorized evaluators.
enum VectorizedExpr {
    Array(NumericArray),
    Scalar(Option<NumericValue>),
}

impl VectorizedExpr {
    fn materialize(self, len: usize, kind: NumericKind) -> ArrayRef {
        match self {
            VectorizedExpr::Array(array) => array.to_aligned_array_ref(kind),
            VectorizedExpr::Scalar(Some(value)) => {
                let target_kind = match (value.kind(), kind) {
                    (NumericKind::Float, _) => NumericKind::Float,
                    (NumericKind::Integer, NumericKind::Float) => NumericKind::Float,
                    (NumericKind::Integer, NumericKind::Integer) => NumericKind::Integer,
                };
                let values = vec![Some(value); len];
                let array = NumericArray::from_numeric_values(values, target_kind);
                array.to_aligned_array_ref(kind)
            }
            VectorizedExpr::Scalar(None) => {
                let values = vec![None; len];
                let array = NumericArray::from_numeric_values(values, kind);
                array.to_aligned_array_ref(kind)
            }
        }
    }
}

/// Represents an affine transformation `scale * field + offset`.
#[derive(Clone, Copy, Debug)]
pub struct AffineExpr {
    pub field: FieldId,
    pub scale: f64,
    pub offset: f64,
}

/// Internal accumulator representing a partially merged affine expression.
#[derive(Clone, Copy, Debug)]
struct AffineState {
    field: Option<FieldId>,
    scale: f64,
    offset: f64,
}

// TODO: Place in impl?
/// Combine field identifiers while tracking whether multiple fields were encountered.
fn merge_field(lhs: Option<FieldId>, rhs: Option<FieldId>) -> Option<Option<FieldId>> {
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

/// Centralizes the numeric kernels applied to scalar expressions so they can be
/// tuned without touching the surrounding table scan logic.
pub struct NumericKernels;

impl NumericKernels {
    /// Collect every field referenced by the scalar expression into `acc`.
    pub fn collect_fields(expr: &ScalarExpr<FieldId>, acc: &mut FxHashSet<FieldId>) {
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
            ScalarExpr::Aggregate(agg) => {
                // Collect fields referenced by the aggregate
                match agg {
                    llkv_expr::expr::AggregateCall::CountStar => {}
                    llkv_expr::expr::AggregateCall::Count { column: fid, .. }
                    | llkv_expr::expr::AggregateCall::Sum { column: fid, .. }
                    | llkv_expr::expr::AggregateCall::Avg { column: fid, .. }
                    | llkv_expr::expr::AggregateCall::Min(fid)
                    | llkv_expr::expr::AggregateCall::Max(fid)
                    | llkv_expr::expr::AggregateCall::CountNulls(fid) => {
                        acc.insert(*fid);
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
            ScalarExpr::ScalarSubquery(_) => {
                // Scalar subqueries don't directly reference fields from the outer query
            }
        }
    }

    /// Ensure each referenced column is materialized as a `NumericArray`, casting as needed.
    pub fn prepare_numeric_arrays(
        lfids: &[LogicalFieldId],
        arrays: &[ArrayRef],
        needed_fields: &FxHashSet<FieldId>,
    ) -> LlkvResult<NumericArrayMap> {
        let mut out: NumericArrayMap = FxHashMap::default();
        if needed_fields.is_empty() {
            return Ok(out);
        }
        for (lfid, array) in lfids.iter().zip(arrays.iter()) {
            let fid = lfid.field_id();
            if !needed_fields.contains(&fid) {
                continue;
            }
            let numeric = Self::coerce_array(array)?;
            out.insert(fid, numeric);
        }
        Ok(out)
    }

    /// Evaluate a scalar expression for the row at `idx` using the provided numeric arrays.
    pub fn evaluate_value(
        expr: &ScalarExpr<FieldId>,
        idx: usize,
        arrays: &NumericArrayMap,
    ) -> LlkvResult<Option<NumericValue>> {
        match expr {
            ScalarExpr::Column(fid) => {
                let array = arrays
                    .get(fid)
                    .ok_or_else(|| Error::Internal(format!("missing column for field {fid}")))?;
                Ok(array.value(idx))
            }
            ScalarExpr::Literal(_) => Ok(Self::literal_numeric_value(expr)),
            ScalarExpr::Binary { left, op, right } => {
                let l = Self::evaluate_value(left, idx, arrays)?;
                let r = Self::evaluate_value(right, idx, arrays)?;
                Ok(Self::apply_binary(*op, l, r))
            }
            ScalarExpr::Compare { left, op, right } => {
                let l = Self::evaluate_value(left, idx, arrays)?;
                let r = Self::evaluate_value(right, idx, arrays)?;
                match (l, r) {
                    (Some(lhs), Some(rhs)) => {
                        let result = Self::compare(*op, lhs, rhs);
                        Ok(Some(NumericValue::Integer(result as i64)))
                    }
                    _ => Ok(None),
                }
            }
            ScalarExpr::Aggregate(_) => Err(Error::Internal(
                "Aggregate expressions should not appear in row-level evaluation".into(),
            )),
            ScalarExpr::GetField { .. } => Err(Error::Internal(
                "GetField expressions should not be evaluated in numeric kernels".into(),
            )),
            ScalarExpr::Cast { expr, data_type } => {
                let value = Self::evaluate_value(expr, idx, arrays)?;
                let target_kind = Self::kind_for_data_type(data_type).ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "unsupported cast target type {:?}",
                        data_type
                    ))
                })?;
                Self::cast_numeric_value_to_kind(value, target_kind)
            }
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                let operand_value = match operand.as_deref() {
                    Some(op) => Some(Self::evaluate_value(op, idx, arrays)?),
                    None => None,
                };

                for (when_expr, then_expr) in branches {
                    let matched = if let Some(op_val_opt) = &operand_value {
                        let when_val = Self::evaluate_value(when_expr, idx, arrays)?;
                        match (op_val_opt, &when_val) {
                            (Some(op_val), Some(branch_val)) => {
                                Self::numeric_equals(*op_val, *branch_val)
                            }
                            _ => false,
                        }
                    } else {
                        let cond_val = Self::evaluate_value(when_expr, idx, arrays)?;
                        cond_val.is_some_and(Self::truthy_numeric)
                    };

                    if matched {
                        return Self::evaluate_value(then_expr, idx, arrays);
                    }
                }

                if let Some(else_expr) = else_expr.as_deref() {
                    Self::evaluate_value(else_expr, idx, arrays)
                } else {
                    Ok(None)
                }
            }
            ScalarExpr::Coalesce(items) => {
                for item in items {
                    if let Some(value) = Self::evaluate_value(item, idx, arrays)? {
                        return Ok(Some(value));
                    }
                }
                Ok(None)
            }
            ScalarExpr::ScalarSubquery(_) => Err(Error::Internal(
                "Scalar subquery evaluation requires a separate execution context".into(),
            )),
        }
    }

    /// Evaluate a scalar expression for every row in the batch.
    #[allow(dead_code)]
    pub fn evaluate_batch(
        expr: &ScalarExpr<FieldId>,
        len: usize,
        arrays: &NumericArrayMap,
    ) -> LlkvResult<ArrayRef> {
        let simplified = Self::simplify(expr);
        Self::evaluate_batch_simplified(&simplified, len, arrays)
    }

    /// Evaluate a scalar expression that has already been simplified.
    pub fn evaluate_batch_simplified(
        expr: &ScalarExpr<FieldId>,
        len: usize,
        arrays: &NumericArrayMap,
    ) -> LlkvResult<ArrayRef> {
        let preferred = Self::infer_result_kind(expr, arrays);
        if let Some(vectorized) = Self::try_evaluate_vectorized(expr, len, arrays, preferred)? {
            return Ok(vectorized.materialize(len, preferred));
        }

        let mut values: Vec<Option<NumericValue>> = Vec::with_capacity(len);
        for idx in 0..len {
            values.push(Self::evaluate_value(expr, idx, arrays)?);
        }
        let array = NumericArray::from_numeric_values(values, preferred);
        Ok(array.to_aligned_array_ref(preferred))
    }

    fn try_evaluate_vectorized(
        expr: &ScalarExpr<FieldId>,
        len: usize,
        arrays: &NumericArrayMap,
        preferred: NumericKind,
    ) -> LlkvResult<Option<VectorizedExpr>> {
        match expr {
            ScalarExpr::Column(fid) => {
                let array = arrays
                    .get(fid)
                    .ok_or_else(|| Error::Internal(format!("missing column for field {fid}")))?;
                Ok(Some(VectorizedExpr::Array(array.clone())))
            }
            ScalarExpr::Literal(_) => Ok(Some(VectorizedExpr::Scalar(
                Self::literal_numeric_value(expr),
            ))),
            ScalarExpr::Binary { left, op, right } => {
                let left_kind = Self::infer_result_kind(left, arrays);
                let right_kind = Self::infer_result_kind(right, arrays);

                let left_vec = Self::try_evaluate_vectorized(left, len, arrays, left_kind)?;
                let right_vec = Self::try_evaluate_vectorized(right, len, arrays, right_kind)?;

                match (left_vec, right_vec) {
                    (Some(VectorizedExpr::Scalar(lhs)), Some(VectorizedExpr::Scalar(rhs))) => Ok(
                        Some(VectorizedExpr::Scalar(Self::apply_binary(*op, lhs, rhs))),
                    ),
                    (Some(VectorizedExpr::Array(lhs)), Some(VectorizedExpr::Array(rhs))) => {
                        let array =
                            Self::compute_binary_array_array(&lhs, &rhs, len, *op, preferred)?;
                        Ok(Some(VectorizedExpr::Array(array)))
                    }
                    (Some(VectorizedExpr::Array(lhs)), Some(VectorizedExpr::Scalar(rhs))) => {
                        let array = Self::compute_binary_array_scalar(
                            &lhs, rhs, len, *op, true, preferred,
                        )?;
                        Ok(Some(VectorizedExpr::Array(array)))
                    }
                    (Some(VectorizedExpr::Scalar(lhs)), Some(VectorizedExpr::Array(rhs))) => {
                        let array = Self::compute_binary_array_scalar(
                            &rhs, lhs, len, *op, false, preferred,
                        )?;
                        Ok(Some(VectorizedExpr::Array(array)))
                    }
                    _ => Ok(None),
                }
            }
            ScalarExpr::Compare { .. } => Ok(None),
            ScalarExpr::Aggregate(_) => Err(Error::Internal(
                "Aggregate expressions should not appear in row-level evaluation".into(),
            )),
            ScalarExpr::GetField { .. } => Err(Error::Internal(
                "GetField expressions should not be evaluated in numeric kernels".into(),
            )),
            ScalarExpr::Cast { expr, data_type } => {
                let inner_kind = Self::infer_result_kind(expr, arrays);
                let inner_vec = Self::try_evaluate_vectorized(expr, len, arrays, inner_kind)?;
                let target_kind = Self::kind_for_data_type(data_type).ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "unsupported cast target type {:?}",
                        data_type
                    ))
                })?;

                match inner_vec {
                    Some(VectorizedExpr::Scalar(value)) => Ok(Some(VectorizedExpr::Scalar(
                        Self::cast_numeric_value_to_kind(value, target_kind)?,
                    ))),
                    Some(VectorizedExpr::Array(array)) => Ok(Some(VectorizedExpr::Array(
                        Self::cast_numeric_array_to_kind(&array, target_kind)?,
                    ))),
                    None => Ok(None),
                }
            }
            ScalarExpr::Case { .. } => Ok(None),
            ScalarExpr::Coalesce(_) => Ok(None),
            ScalarExpr::ScalarSubquery(_) => Ok(None),
        }
    }

    fn compute_binary_array_array(
        left: &NumericArray,
        right: &NumericArray,
        len: usize,
        op: BinaryOp,
        preferred: NumericKind,
    ) -> LlkvResult<NumericArray> {
        if left.len() != len || right.len() != len {
            return Err(Error::Internal("scalar expression length mismatch".into()));
        }

        let iter = (0..len).map(|idx| {
            let lhs = left.value(idx);
            let rhs = right.value(idx);
            Self::apply_binary(op, lhs, rhs)
        });

        let values = iter.collect::<Vec<_>>();
        Ok(NumericArray::from_numeric_values(values, preferred))
    }

    fn compute_binary_array_scalar(
        array: &NumericArray,
        scalar: Option<NumericValue>,
        len: usize,
        op: BinaryOp,
        array_is_left: bool,
        preferred: NumericKind,
    ) -> LlkvResult<NumericArray> {
        if array.len() != len {
            return Err(Error::Internal("scalar expression length mismatch".into()));
        }

        if scalar.is_none() {
            return Ok(NumericArray::from_numeric_values(
                vec![None; len],
                preferred,
            ));
        }
        let scalar_value = scalar.expect("checked above");

        if array_is_left && matches!(op, BinaryOp::Divide | BinaryOp::Modulo) {
            let is_zero = matches!(
                scalar_value,
                NumericValue::Integer(0) | NumericValue::Float(0.0)
            );
            if is_zero {
                return Ok(NumericArray::from_numeric_values(
                    vec![None; len],
                    preferred,
                ));
            }
        }

        let iter = (0..len).map(|idx| {
            let array_val = array.value(idx);
            let (lhs, rhs) = if array_is_left {
                (array_val, Some(scalar_value))
            } else {
                (Some(scalar_value), array_val)
            };
            Self::apply_binary(op, lhs, rhs)
        });

        let values = iter.collect::<Vec<_>>();
        Ok(NumericArray::from_numeric_values(values, preferred))
    }

    /// Returns the column referenced by an expression when it's a direct or additive identity passthrough.
    pub fn passthrough_column(expr: &ScalarExpr<FieldId>) -> Option<FieldId> {
        match Self::simplify(expr) {
            ScalarExpr::Column(fid) => Some(fid),
            _ => None,
        }
    }

    fn literal_numeric_value(expr: &ScalarExpr<FieldId>) -> Option<NumericValue> {
        if let ScalarExpr::Literal(lit) = expr {
            match lit {
                llkv_expr::literal::Literal::Float(f) => Some(NumericValue::Float(*f)),
                llkv_expr::literal::Literal::Integer(i) => {
                    if let Ok(value) = i64::try_from(*i) {
                        Some(NumericValue::Integer(value))
                    } else {
                        Some(NumericValue::Float(*i as f64))
                    }
                }
                llkv_expr::literal::Literal::Boolean(b) => {
                    Some(NumericValue::Integer(if *b { 1 } else { 0 }))
                }
                llkv_expr::literal::Literal::String(_) => None,
                llkv_expr::literal::Literal::Struct(_) => None,
                llkv_expr::literal::Literal::Null => None,
            }
        } else {
            None
        }
    }

    fn literal_is_zero(expr: &ScalarExpr<FieldId>) -> bool {
        matches!(
            Self::literal_numeric_value(expr),
            Some(NumericValue::Integer(0)) | Some(NumericValue::Float(0.0))
        )
    }

    fn literal_is_one(expr: &ScalarExpr<FieldId>) -> bool {
        matches!(
            Self::literal_numeric_value(expr),
            Some(NumericValue::Integer(1)) | Some(NumericValue::Float(1.0))
        )
    }

    #[inline]
    fn numeric_equals(lhs: NumericValue, rhs: NumericValue) -> bool {
        match (lhs, rhs) {
            (NumericValue::Integer(a), NumericValue::Integer(b)) => a == b,
            _ => lhs.as_f64() == rhs.as_f64(),
        }
    }

    #[inline]
    fn truthy_numeric(value: NumericValue) -> bool {
        match value {
            NumericValue::Integer(v) => v != 0,
            NumericValue::Float(v) => v != 0.0,
        }
    }

    /// Recursively simplify the expression by folding literals and eliminating identity operations.
    pub fn simplify(expr: &ScalarExpr<FieldId>) -> ScalarExpr<FieldId> {
        match expr {
            ScalarExpr::Column(_)
            | ScalarExpr::Literal(_)
            | ScalarExpr::Aggregate(_)
            | ScalarExpr::GetField { .. } => expr.clone(),
            ScalarExpr::Binary { left, op, right } => {
                let left_s = Self::simplify(left);
                let right_s = Self::simplify(right);

                if let (Some(lv), Some(rv)) = (
                    Self::literal_numeric_value(&left_s),
                    Self::literal_numeric_value(&right_s),
                ) && let Some(lit) = Self::apply_binary_literal(*op, lv, rv)
                {
                    return lit;
                }

                match op {
                    BinaryOp::Add => {
                        if Self::literal_is_zero(&left_s) {
                            return right_s;
                        }
                        if Self::literal_is_zero(&right_s) {
                            return left_s;
                        }
                    }
                    BinaryOp::Subtract => {
                        if Self::literal_is_zero(&right_s) {
                            return left_s;
                        }
                    }
                    BinaryOp::Multiply => {
                        if Self::literal_is_one(&left_s) {
                            return right_s;
                        }
                        if Self::literal_is_one(&right_s) {
                            return left_s;
                        }
                    }
                    BinaryOp::Divide => {
                        if Self::literal_is_one(&right_s) {
                            return left_s;
                        }
                    }
                    BinaryOp::Modulo => {}
                }

                ScalarExpr::binary(left_s, *op, right_s)
            }
            ScalarExpr::Compare { left, op, right } => {
                let left_s = Self::simplify(left);
                let right_s = Self::simplify(right);
                ScalarExpr::compare(left_s, *op, right_s)
            }
            ScalarExpr::Cast { expr, data_type } => {
                let inner = Self::simplify(expr);
                ScalarExpr::cast(inner, data_type.clone())
            }
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                let operand_s = operand.as_ref().map(|inner| Self::simplify(inner));
                let mut branch_vec = Vec::with_capacity(branches.len());
                for (when_expr, then_expr) in branches {
                    branch_vec.push((Self::simplify(when_expr), Self::simplify(then_expr)));
                }
                let else_s = else_expr.as_ref().map(|inner| Self::simplify(inner));
                ScalarExpr::case(operand_s, branch_vec, else_s)
            }
            ScalarExpr::Coalesce(items) => {
                let simplified_items = items.iter().map(Self::simplify).collect();
                ScalarExpr::coalesce(simplified_items)
            }
            ScalarExpr::ScalarSubquery(_) => expr.clone(),
        }
    }

    /// Attempts to represent the expression as `scale * column + offset`.
    /// Returns `None` when the expression depends on multiple columns or is non-linear.
    #[allow(dead_code)]
    pub fn extract_affine(expr: &ScalarExpr<FieldId>) -> Option<AffineExpr> {
        let simplified = Self::simplify(expr);
        Self::extract_affine_simplified(&simplified)
    }

    /// Variant of \[`extract_affine`\] that assumes `expr` is already simplified.
    pub fn extract_affine_simplified(expr: &ScalarExpr<FieldId>) -> Option<AffineExpr> {
        let state = Self::affine_state(expr)?;
        let field = state.field?;
        Some(AffineExpr {
            field,
            scale: state.scale,
            offset: state.offset,
        })
    }

    fn affine_state(expr: &ScalarExpr<FieldId>) -> Option<AffineState> {
        match expr {
            ScalarExpr::Column(fid) => Some(AffineState {
                field: Some(*fid),
                scale: 1.0,
                offset: 0.0,
            }),
            ScalarExpr::Literal(_) => {
                let value = Self::literal_numeric_value(expr)?.as_f64();
                Some(AffineState {
                    field: None,
                    scale: 0.0,
                    offset: value,
                })
            }
            ScalarExpr::Aggregate(_) => None, // Aggregates not supported in affine transformations
            ScalarExpr::GetField { .. } => None, // GetField not supported in affine transformations
            ScalarExpr::Binary { left, op, right } => {
                let left_state = Self::affine_state(left)?;
                let right_state = Self::affine_state(right)?;
                match op {
                    BinaryOp::Add => Self::affine_add(left_state, right_state),
                    BinaryOp::Subtract => Self::affine_sub(left_state, right_state),
                    BinaryOp::Multiply => Self::affine_mul(left_state, right_state),
                    BinaryOp::Divide => Self::affine_div(left_state, right_state),
                    BinaryOp::Modulo => None,
                }
            }
            ScalarExpr::Compare { .. } => None,
            ScalarExpr::Cast { expr, .. } => Self::affine_state(expr),
            ScalarExpr::Case { .. } => None,
            ScalarExpr::Coalesce(_) => None,
            ScalarExpr::ScalarSubquery(_) => None,
        }
    }

    fn affine_add(lhs: AffineState, rhs: AffineState) -> Option<AffineState> {
        let field = merge_field(lhs.field, rhs.field)?;
        Some(AffineState {
            field,
            scale: lhs.scale + rhs.scale,
            offset: lhs.offset + rhs.offset,
        })
    }

    fn affine_sub(lhs: AffineState, rhs: AffineState) -> Option<AffineState> {
        let neg_rhs = AffineState {
            field: rhs.field,
            scale: -rhs.scale,
            offset: -rhs.offset,
        };
        Self::affine_add(lhs, neg_rhs)
    }

    fn affine_mul(lhs: AffineState, rhs: AffineState) -> Option<AffineState> {
        if rhs.field.is_none() {
            let factor = rhs.offset;
            return Some(AffineState {
                field: lhs.field,
                scale: lhs.scale * factor,
                offset: lhs.offset * factor,
            });
        }
        if lhs.field.is_none() {
            let factor = lhs.offset;
            return Some(AffineState {
                field: rhs.field,
                scale: rhs.scale * factor,
                offset: rhs.offset * factor,
            });
        }
        None
    }

    fn affine_div(lhs: AffineState, rhs: AffineState) -> Option<AffineState> {
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

    fn apply_binary_literal(
        op: BinaryOp,
        lhs: NumericValue,
        rhs: NumericValue,
    ) -> Option<ScalarExpr<FieldId>> {
        match op {
            BinaryOp::Add => Some(Self::literal_from_numeric(Self::add_values(lhs, rhs))),
            BinaryOp::Subtract => Some(Self::literal_from_numeric(Self::sub_values(lhs, rhs))),
            BinaryOp::Multiply => Some(Self::literal_from_numeric(Self::mul_values(lhs, rhs))),
            BinaryOp::Divide => Self::div_values(lhs, rhs).map(Self::literal_from_numeric),
            BinaryOp::Modulo => Self::mod_values(lhs, rhs).map(Self::literal_from_numeric),
        }
    }

    fn literal_from_numeric(value: NumericValue) -> ScalarExpr<FieldId> {
        match value {
            NumericValue::Integer(i) => ScalarExpr::literal(i),
            NumericValue::Float(f) => ScalarExpr::literal(f),
        }
    }

    /// Apply an arithmetic kernel. Returns `None` when the computation results in a null (e.g. divide by zero).
    pub fn apply_binary(
        op: BinaryOp,
        lhs: Option<NumericValue>,
        rhs: Option<NumericValue>,
    ) -> Option<NumericValue> {
        match (lhs, rhs) {
            (Some(lv), Some(rv)) => Self::apply_binary_values(op, lv, rv),
            _ => None,
        }
    }

    fn apply_binary_values(
        op: BinaryOp,
        lhs: NumericValue,
        rhs: NumericValue,
    ) -> Option<NumericValue> {
        match op {
            BinaryOp::Add => Some(Self::add_values(lhs, rhs)),
            BinaryOp::Subtract => Some(Self::sub_values(lhs, rhs)),
            BinaryOp::Multiply => Some(Self::mul_values(lhs, rhs)),
            BinaryOp::Divide => Self::div_values(lhs, rhs),
            BinaryOp::Modulo => Self::mod_values(lhs, rhs),
        }
    }

    fn add_values(lhs: NumericValue, rhs: NumericValue) -> NumericValue {
        match (lhs, rhs) {
            (NumericValue::Integer(li), NumericValue::Integer(ri)) => match li.checked_add(ri) {
                Some(sum) => NumericValue::Integer(sum),
                None => NumericValue::Float(li as f64 + ri as f64),
            },
            _ => NumericValue::Float(lhs.as_f64() + rhs.as_f64()),
        }
    }

    fn sub_values(lhs: NumericValue, rhs: NumericValue) -> NumericValue {
        match (lhs, rhs) {
            (NumericValue::Integer(li), NumericValue::Integer(ri)) => match li.checked_sub(ri) {
                Some(diff) => NumericValue::Integer(diff),
                None => NumericValue::Float(li as f64 - ri as f64),
            },
            _ => NumericValue::Float(lhs.as_f64() - rhs.as_f64()),
        }
    }

    fn mul_values(lhs: NumericValue, rhs: NumericValue) -> NumericValue {
        match (lhs, rhs) {
            (NumericValue::Integer(li), NumericValue::Integer(ri)) => match li.checked_mul(ri) {
                Some(prod) => NumericValue::Integer(prod),
                None => NumericValue::Float(li as f64 * ri as f64),
            },
            _ => NumericValue::Float(lhs.as_f64() * rhs.as_f64()),
        }
    }

    fn div_values(lhs: NumericValue, rhs: NumericValue) -> Option<NumericValue> {
        match rhs {
            NumericValue::Integer(0) | NumericValue::Float(0.0) => return None,
            _ => {}
        }

        match (lhs, rhs) {
            (NumericValue::Integer(li), NumericValue::Integer(ri)) => {
                Some(NumericValue::Integer(li / ri))
            }
            _ => Some(NumericValue::Float(lhs.as_f64() / rhs.as_f64())),
        }
    }

    fn mod_values(lhs: NumericValue, rhs: NumericValue) -> Option<NumericValue> {
        match rhs {
            NumericValue::Integer(0) | NumericValue::Float(0.0) => return None,
            _ => {}
        }

        match (lhs, rhs) {
            (NumericValue::Integer(li), NumericValue::Integer(ri)) => {
                Some(NumericValue::Integer(li % ri))
            }
            _ => Some(NumericValue::Float(lhs.as_f64() % rhs.as_f64())),
        }
    }

    fn cast_numeric_value_to_kind(
        value: Option<NumericValue>,
        target: NumericKind,
    ) -> LlkvResult<Option<NumericValue>> {
        match value {
            None => Ok(None),
            Some(NumericValue::Integer(v)) => Ok(Some(match target {
                NumericKind::Integer => NumericValue::Integer(v),
                NumericKind::Float => NumericValue::Float(v as f64),
            })),
            Some(NumericValue::Float(v)) => {
                if !v.is_finite() {
                    return Err(Error::InvalidArgumentError(
                        "cannot cast non-finite float value".into(),
                    ));
                }
                match target {
                    NumericKind::Float => Ok(Some(NumericValue::Float(v))),
                    NumericKind::Integer => {
                        let truncated = v.trunc();
                        if truncated < i64::MIN as f64 || truncated > i64::MAX as f64 {
                            return Err(Error::InvalidArgumentError(
                                "float out of range for INT64 cast".into(),
                            ));
                        }
                        Ok(Some(NumericValue::Integer(truncated as i64)))
                    }
                }
            }
        }
    }

    fn cast_numeric_array_to_kind(
        array: &NumericArray,
        target: NumericKind,
    ) -> LlkvResult<NumericArray> {
        match target {
            NumericKind::Float => Ok(array.promote_to_float()),
            NumericKind::Integer => {
                if array.kind() == NumericKind::Integer {
                    Ok(array.clone())
                } else {
                    let mut values = Vec::with_capacity(array.len());
                    for idx in 0..array.len() {
                        let value = array.value(idx);
                        values.push(Self::cast_numeric_value_to_kind(value, target)?);
                    }
                    Ok(NumericArray::from_numeric_values(
                        values,
                        NumericKind::Integer,
                    ))
                }
            }
        }
    }

    fn infer_result_kind(expr: &ScalarExpr<FieldId>, arrays: &NumericArrayMap) -> NumericKind {
        match expr {
            ScalarExpr::Literal(lit) => match lit {
                llkv_expr::literal::Literal::Float(_) => NumericKind::Float,
                llkv_expr::literal::Literal::Integer(_) => NumericKind::Integer,
                llkv_expr::literal::Literal::Boolean(_) => NumericKind::Integer,
                llkv_expr::literal::Literal::Null => NumericKind::Integer,
                llkv_expr::literal::Literal::String(_) => NumericKind::Float,
                llkv_expr::literal::Literal::Struct(_) => NumericKind::Float,
            },
            ScalarExpr::Column(fid) => arrays
                .get(fid)
                .map(|arr| arr.kind())
                .unwrap_or(NumericKind::Float),
            ScalarExpr::Binary { left, op, right } => {
                let left_kind = Self::infer_result_kind(left, arrays);
                let right_kind = Self::infer_result_kind(right, arrays);
                Self::binary_result_kind(*op, left_kind, right_kind)
            }
            ScalarExpr::Compare { .. } => NumericKind::Integer,
            ScalarExpr::Aggregate(_) => NumericKind::Float,
            ScalarExpr::GetField { .. } => NumericKind::Float,
            ScalarExpr::Cast { expr, data_type } => {
                let target_kind = Self::kind_for_data_type(data_type);
                target_kind.unwrap_or_else(|| Self::infer_result_kind(expr, arrays))
            }
            ScalarExpr::Case {
                branches,
                else_expr,
                ..
            } => {
                let mut result_kind = NumericKind::Integer;
                for (_, then_expr) in branches {
                    if matches!(
                        Self::infer_result_kind(then_expr, arrays),
                        NumericKind::Float
                    ) {
                        result_kind = NumericKind::Float;
                        break;
                    }
                }
                if result_kind != NumericKind::Float
                    && let Some(inner) = else_expr.as_deref()
                    && matches!(Self::infer_result_kind(inner, arrays), NumericKind::Float)
                {
                    result_kind = NumericKind::Float;
                }
                result_kind
            }
            ScalarExpr::Coalesce(items) => {
                let mut result_kind = NumericKind::Integer;
                for item in items {
                    if matches!(Self::infer_result_kind(item, arrays), NumericKind::Float) {
                        result_kind = NumericKind::Float;
                        break;
                    }
                }
                result_kind
            }
            ScalarExpr::ScalarSubquery(_) => NumericKind::Float,
        }
    }

    /// Infer the numeric kind of an expression using only the kinds of its referenced columns.
    pub fn infer_result_kind_from_types<F>(
        expr: &ScalarExpr<FieldId>,
        resolve_kind: &mut F,
    ) -> Option<NumericKind>
    where
        F: FnMut(FieldId) -> Option<NumericKind>,
    {
        match expr {
            ScalarExpr::Literal(_) => Self::literal_numeric_value(expr).map(|v| v.kind()),
            ScalarExpr::Column(fid) => resolve_kind(*fid),
            ScalarExpr::Binary { left, op, right } => {
                let left_kind = Self::infer_result_kind_from_types(left, resolve_kind)?;
                let right_kind = Self::infer_result_kind_from_types(right, resolve_kind)?;
                Some(Self::binary_result_kind(*op, left_kind, right_kind))
            }
            ScalarExpr::Compare { .. } => Some(NumericKind::Integer),
            ScalarExpr::Aggregate(_) => Some(NumericKind::Float),
            ScalarExpr::GetField { .. } => None,
            ScalarExpr::Cast { expr, data_type } => {
                let target_kind = Self::kind_for_data_type(data_type);
                target_kind.or_else(|| Self::infer_result_kind_from_types(expr, resolve_kind))
            }
            ScalarExpr::Case {
                branches,
                else_expr,
                ..
            } => {
                let mut result_kind = NumericKind::Integer;
                for (_, then_expr) in branches {
                    let kind = Self::infer_result_kind_from_types(then_expr, resolve_kind)?;
                    if matches!(kind, NumericKind::Float) {
                        result_kind = NumericKind::Float;
                        break;
                    }
                }
                if result_kind != NumericKind::Float
                    && let Some(inner) = else_expr.as_deref()
                    && let Some(kind) = Self::infer_result_kind_from_types(inner, resolve_kind)
                    && matches!(kind, NumericKind::Float)
                {
                    result_kind = NumericKind::Float;
                }
                Some(result_kind)
            }
            ScalarExpr::Coalesce(items) => {
                let mut result_kind = NumericKind::Integer;
                for item in items {
                    let kind = Self::infer_result_kind_from_types(item, resolve_kind)?;
                    if matches!(kind, NumericKind::Float) {
                        result_kind = NumericKind::Float;
                        break;
                    }
                }
                Some(result_kind)
            }
            ScalarExpr::ScalarSubquery(_) => Some(NumericKind::Float),
        }
    }

    /// Map an Arrow `DataType` to the corresponding numeric kind when supported.
    pub fn kind_for_data_type(dtype: &DataType) -> Option<NumericKind> {
        match dtype {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Boolean => Some(NumericKind::Integer),
            DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Null => Some(NumericKind::Float),
            _ => None,
        }
    }

    fn binary_result_kind(
        op: BinaryOp,
        lhs_kind: NumericKind,
        rhs_kind: NumericKind,
    ) -> NumericKind {
        let lhs_value = match lhs_kind {
            NumericKind::Integer => NumericValue::Integer(1),
            NumericKind::Float => NumericValue::Float(1.0),
        };
        let rhs_value = match rhs_kind {
            NumericKind::Integer => NumericValue::Integer(1),
            NumericKind::Float => NumericValue::Float(1.0),
        };

        Self::apply_binary_values(op, lhs_value, rhs_value)
            .unwrap_or(NumericValue::Float(0.0))
            .kind()
    }

    /// Compare two numeric values using the provided operator.
    pub fn compare(op: CompareOp, lhs: NumericValue, rhs: NumericValue) -> bool {
        match (lhs, rhs) {
            (NumericValue::Integer(li), NumericValue::Integer(ri)) => match op {
                CompareOp::Eq => li == ri,
                CompareOp::NotEq => li != ri,
                CompareOp::Lt => li < ri,
                CompareOp::LtEq => li <= ri,
                CompareOp::Gt => li > ri,
                CompareOp::GtEq => li >= ri,
            },
            (lv, rv) => {
                let lf = lv.as_f64();
                let rf = rv.as_f64();
                match op {
                    CompareOp::Eq => lf == rf,
                    CompareOp::NotEq => lf != rf,
                    CompareOp::Lt => lf < rf,
                    CompareOp::LtEq => lf <= rf,
                    CompareOp::Gt => lf > rf,
                    CompareOp::GtEq => lf >= rf,
                }
            }
        }
    }

    fn coerce_array(array: &ArrayRef) -> LlkvResult<NumericArray> {
        NumericArray::try_from_arrow(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};
    use llkv_expr::Literal;

    fn float_array(values: &[Option<f64>]) -> NumericArray {
        let array = Float64Array::from(values.to_vec());
        NumericArray::from_float(Arc::new(array))
    }

    fn int_array(values: &[Option<i64>]) -> NumericArray {
        let array = Int64Array::from(values.to_vec());
        NumericArray::from_int(Arc::new(array))
    }

    #[test]
    fn integer_addition_preserves_int_type() {
        const F1: FieldId = 30;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(F1, int_array(&[Some(1), Some(-5), None, Some(42)]));

        let expr = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Add,
            ScalarExpr::literal(3),
        );

        let result = NumericKernels::evaluate_batch(&expr, 4, &arrays).unwrap();
        let array = result
            .as_ref()
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("expected Int64Array");

        assert_eq!(array.len(), 4);
        assert_eq!(array.value(0), 4);
        assert_eq!(array.value(1), -2);
        assert!(array.is_null(2));
        assert_eq!(array.value(3), 45);
    }

    #[test]
    fn integer_division_matches_sqlite_semantics() {
        const F1: FieldId = 31;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(F1, int_array(&[Some(5), Some(-7), Some(0), None]));

        let expr = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Divide,
            ScalarExpr::literal(2),
        );

        let result = NumericKernels::evaluate_batch(&expr, 4, &arrays).unwrap();
        let array = result
            .as_ref()
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("expected Int64Array");

        assert_eq!(array.len(), 4);
        assert_eq!(array.value(0), 2);
        assert_eq!(array.value(1), -3);
        assert_eq!(array.value(2), 0);
        assert!(array.is_null(3));
    }

    #[test]
    fn integer_overflow_promotes_to_float_array() {
        const F1: FieldId = 32;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(F1, int_array(&[Some(i64::MAX), Some(10)]));

        let expr = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Add,
            ScalarExpr::literal(1),
        );

        let result = NumericKernels::evaluate_batch(&expr, 2, &arrays).unwrap();
        assert!(
            result
                .as_ref()
                .as_any()
                .downcast_ref::<Int64Array>()
                .is_none()
        );

        let array = result
            .as_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("expected Float64Array after overflow");

        assert_eq!(array.len(), 2);
        assert!(array.value(0).is_finite());
        assert_eq!(array.value(1), 11.0);
    }

    #[test]
    fn vectorized_add_columns() {
        const F1: FieldId = 1;
        const F2: FieldId = 2;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(F1, float_array(&[Some(1.0), Some(2.0), None, Some(-1.0)]));
        arrays.insert(
            F2,
            float_array(&[Some(5.0), Some(-1.0), Some(3.0), Some(4.0)]),
        );

        let expr = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Add,
            ScalarExpr::column(F2),
        );

        let result = NumericKernels::evaluate_batch(&expr, 4, &arrays).unwrap();
        let result = result
            .as_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(result.len(), 4);
        assert_eq!(result.value(0), 6.0);
        assert_eq!(result.value(1), 1.0);
        assert!(result.is_null(2));
        assert_eq!(result.value(3), 3.0);
    }

    #[test]
    fn coalesce_evaluation_in_comparison() {
        const A: FieldId = 401;
        const B: FieldId = 402;
        const C: FieldId = 403;
        const D: FieldId = 404;
        const E: FieldId = 405;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(A, int_array(&[Some(1), None, None, None, None, None]));
        arrays.insert(B, int_array(&[Some(2), Some(2), None, None, None, None]));
        arrays.insert(C, int_array(&[Some(3), None, Some(3), None, None, None]));
        arrays.insert(D, int_array(&[Some(4), None, None, Some(4), None, None]));
        arrays.insert(E, int_array(&[Some(5), None, None, None, Some(5), None]));

        let coalesce_expr = ScalarExpr::coalesce(vec![
            ScalarExpr::column(A),
            ScalarExpr::column(B),
            ScalarExpr::column(C),
            ScalarExpr::column(D),
            ScalarExpr::column(E),
        ]);

        let expected_values = [Some(1), Some(2), Some(3), Some(4), Some(5), None];
        for (idx, expected) in expected_values.iter().enumerate() {
            let value = NumericKernels::evaluate_value(&coalesce_expr, idx, &arrays).unwrap();
            let actual = value.map(|num| match num {
                NumericValue::Integer(v) => v,
                NumericValue::Float(v) => v as i64,
            });
            assert_eq!(actual, *expected, "row {idx} did not match");
        }

        let compare_expr =
            ScalarExpr::compare(coalesce_expr, CompareOp::NotEq, ScalarExpr::literal(0));

        let expected_flags = [Some(1), Some(1), Some(1), Some(1), Some(1), None];
        for (idx, expected) in expected_flags.iter().enumerate() {
            let value = NumericKernels::evaluate_value(&compare_expr, idx, &arrays).unwrap();
            let actual = value.map(|num| match num {
                NumericValue::Integer(v) => v,
                NumericValue::Float(v) => v as i64,
            });
            assert_eq!(actual, *expected, "comparison row {idx} mismatch");
        }
    }

    #[test]
    fn vectorized_multiply_literal() {
        const F1: FieldId = 10;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(F1, float_array(&[Some(1.0), Some(-2.5), Some(0.0), None]));

        let expr = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Multiply,
            ScalarExpr::literal(3),
        );

        let result = NumericKernels::evaluate_batch(&expr, 4, &arrays).unwrap();
        let result = result
            .as_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(result.len(), 4);
        assert_eq!(result.value(0), 3.0);
        assert_eq!(result.value(1), -7.5);
        assert_eq!(result.value(2), 0.0);
        assert!(result.is_null(3));
    }

    #[test]
    fn vectorized_add_column_scalar_literal() {
        const F1: FieldId = 11;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(F1, float_array(&[Some(2.0), None, Some(-5.5)]));

        let expr = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Add,
            ScalarExpr::literal(4),
        );

        let result = NumericKernels::evaluate_batch(&expr, 3, &arrays).unwrap();
        let result = result
            .as_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result.value(0), 6.0);
        assert!(result.is_null(1));
        assert!((result.value(2) - (-1.5)).abs() < f64::EPSILON);
    }

    #[test]
    fn vectorized_literal_minus_column() {
        const F1: FieldId = 12;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(F1, float_array(&[Some(3.0), Some(-2.0), None]));

        let expr = ScalarExpr::binary(
            ScalarExpr::literal(10),
            BinaryOp::Subtract,
            ScalarExpr::column(F1),
        );

        let result = NumericKernels::evaluate_batch(&expr, 3, &arrays).unwrap();
        let result = result
            .as_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result.value(0), 7.0);
        assert_eq!(result.value(1), 12.0);
        assert!(result.is_null(2));
    }

    #[test]
    fn vectorized_divide_by_zero_yields_null() {
        const NUM: FieldId = 20;
        const DEN: FieldId = 21;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(
            NUM,
            float_array(&[Some(4.0), Some(9.0), Some(5.0), Some(-6.0)]),
        );
        arrays.insert(DEN, float_array(&[Some(2.0), Some(0.0), None, Some(-3.0)]));

        let expr = ScalarExpr::binary(
            ScalarExpr::column(NUM),
            BinaryOp::Divide,
            ScalarExpr::column(DEN),
        );

        let result = NumericKernels::evaluate_batch(&expr, 4, &arrays).unwrap();
        let result = result
            .as_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(result.len(), 4);
        assert_eq!(result.value(0), 2.0);
        assert!(result.is_null(1));
        assert!(result.is_null(2));
        assert_eq!(result.value(3), 2.0);
    }

    #[test]
    fn vectorized_divide_by_zero_literal_rhs_yields_nulls() {
        const F1: FieldId = 22;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(F1, float_array(&[Some(1.0), Some(-4.0), None]));

        let expr = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Divide,
            ScalarExpr::literal(0),
        );

        let result = NumericKernels::evaluate_batch(&expr, 3, &arrays).unwrap();
        let result = result
            .as_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(result.len(), 3);
        assert!(result.is_null(0));
        assert!(result.is_null(1));
        assert!(result.is_null(2));
    }

    #[test]
    fn vectorized_modulo_literals() {
        let expr = ScalarExpr::binary(
            ScalarExpr::literal(13),
            BinaryOp::Modulo,
            ScalarExpr::literal(5),
        );

        let simplified = NumericKernels::simplify(&expr);
        let ScalarExpr::Literal(Literal::Integer(value)) = simplified else {
            panic!("expected literal result");
        };
        assert_eq!(value, 3);
    }

    #[test]
    fn vectorized_modulo_column_rhs_zero_yields_null() {
        const NUM: FieldId = 23;
        const DEN: FieldId = 24;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(NUM, float_array(&[Some(4.0), Some(7.0), None, Some(-6.0)]));
        arrays.insert(
            DEN,
            float_array(&[Some(2.0), Some(0.0), Some(3.0), Some(-4.0)]),
        );

        let expr = ScalarExpr::binary(
            ScalarExpr::column(NUM),
            BinaryOp::Modulo,
            ScalarExpr::column(DEN),
        );

        let result = NumericKernels::evaluate_batch(&expr, 4, &arrays).unwrap();
        let result = result
            .as_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(result.len(), 4);
        assert_eq!(result.value(0), 0.0);
        assert!(result.is_null(1));
        assert!(result.is_null(2));
        assert_eq!(result.value(3), -6.0 % -4.0);
    }

    #[test]
    fn evaluate_simple_case_expression() {
        const F1: FieldId = 200;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(F1, int_array(&[Some(1), Some(2), None]));

        let expr = ScalarExpr::case(
            Some(ScalarExpr::column(F1)),
            vec![(ScalarExpr::literal(1), ScalarExpr::literal(10))],
            Some(ScalarExpr::literal(20)),
        );

        let result = NumericKernels::evaluate_batch(&expr, 3, &arrays).unwrap();
        let array = result
            .as_ref()
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("expected Int64Array");

        assert_eq!(array.len(), 3);
        assert_eq!(array.value(0), 10);
        assert_eq!(array.value(1), 20);
        assert_eq!(array.value(2), 20);
    }

    #[test]
    fn evaluate_searched_case_expression() {
        const F1: FieldId = 201;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(F1, int_array(&[Some(2), Some(5), None]));

        let condition = ScalarExpr::compare(
            ScalarExpr::column(F1),
            CompareOp::Gt,
            ScalarExpr::literal(3),
        );
        let expr = ScalarExpr::case(
            None,
            vec![(condition, ScalarExpr::column(F1))],
            Some(ScalarExpr::literal(0)),
        );

        let result = NumericKernels::evaluate_batch(&expr, 3, &arrays).unwrap();
        let array = result
            .as_ref()
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("expected Int64Array");

        assert_eq!(array.len(), 3);
        assert_eq!(array.value(0), 0);
        assert_eq!(array.value(1), 5);
        assert_eq!(array.value(2), 0);
    }

    #[test]
    fn passthrough_detects_identity_ops() {
        const F1: FieldId = 99;

        let expr_add = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Add,
            ScalarExpr::literal(0),
        );
        assert_eq!(NumericKernels::passthrough_column(&expr_add), Some(F1));

        let expr_sub = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Subtract,
            ScalarExpr::literal(0),
        );
        assert_eq!(NumericKernels::passthrough_column(&expr_sub), Some(F1));

        let expr_mul = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Multiply,
            ScalarExpr::literal(1),
        );
        assert_eq!(NumericKernels::passthrough_column(&expr_mul), Some(F1));

        let expr_div = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Divide,
            ScalarExpr::literal(1),
        );
        assert_eq!(NumericKernels::passthrough_column(&expr_div), Some(F1));

        // Non-identity literal should not passthrough.
        let expr_add_two = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Add,
            ScalarExpr::literal(2),
        );
        assert_eq!(NumericKernels::passthrough_column(&expr_add_two), None);
    }
}
