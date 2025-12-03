//! Stack-based arithmetic evaluator that uses native Arrow kernels.

use std::{hash::Hash, sync::Arc};

use arrow::array::{ArrayRef, Float64Array, Int64Array, PrimitiveArray, new_null_array};
use arrow::compute::kernels::numeric;
use arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use llkv_expr::{BinaryOp, ScalarExpr, literal::Literal};
use llkv_result::{Error, Result as LlkvResult};
use rustc_hash::FxHashMap;

use crate::eval::{NumericArrayMap, ScalarEvaluator};
use crate::kernels::common_type_for_op;

/// Pre-compiled arithmetic path that evaluates directly using Arrow kernels.
pub struct NumericFastPath<F> {
    tokens: Vec<Token>,
    column_order: Vec<F>,
    target_type: DataType,
}

#[derive(Clone)]
enum Token {
    Column { slot: usize },
    Literal(NumericLiteral),
    Binary(BinaryOp),
}

#[derive(Clone)]
enum NumericLiteral {
    Int(i128),
    Float(f64),
    Null,
}

impl<F: Hash + Eq + Copy> NumericFastPath<F> {
    pub fn compile(
        expr: &ScalarExpr<F>,
        arrays: &NumericArrayMap<F>,
        target_type: &DataType,
    ) -> Option<Self> {
        if !ScalarEvaluator::is_supported_numeric(target_type) {
            return None;
        }

        let mut builder = PlanBuilder {
            tokens: Vec::new(),
            column_slots: FxHashMap::default(),
            column_order: Vec::new(),
            arrays,
        };
        let output_type = builder.visit(expr)?;

        // Only proceed if the inferred output type aligns with the preferred target type
        if &output_type != target_type {
            return None;
        }

        Some(Self {
            tokens: builder.tokens,
            column_order: builder.column_order,
            target_type: target_type.clone(),
        })
    }

    pub fn execute(&self, len: usize, arrays: &NumericArrayMap<F>) -> LlkvResult<ArrayRef> {
        let mut column_refs = Vec::with_capacity(self.column_order.len());

        for field in &self.column_order {
            let array = arrays
                .get(field)
                .ok_or_else(|| Error::Internal("missing numeric array for fast path".into()))?;
            if !ScalarEvaluator::is_supported_numeric(array.data_type()) {
                return Err(Error::Internal(
                    "unsupported array type for numeric fast path".into(),
                ));
            }
            let casted = if array.data_type() == &self.target_type {
                array.clone()
            } else {
                arrow::compute::cast(array, &self.target_type)
                    .map_err(|e| Error::Internal(format!("fast path cast failed: {}", e)))?
            };
            column_refs.push(casted);
        }

        let mut stack: Vec<ArrayRef> = Vec::with_capacity(8);

        for token in &self.tokens {
            match token {
                Token::Column { slot } => {
                    stack.push(column_refs[*slot].clone());
                }
                Token::Literal(lit) => {
                    let array = make_literal_array(lit, len, &self.target_type)?;
                    stack.push(array);
                }
                Token::Binary(op) => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| Error::Internal("fast path stack underflow".into()))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| Error::Internal("fast path stack underflow".into()))?;

                    let result = compute_binary_no_coerce(&lhs, &rhs, *op, &self.target_type)?;
                    stack.push(result);
                }
            }
        }

        let result = stack
            .pop()
            .ok_or_else(|| Error::Internal("fast path evaluation missing result".into()))?;

        Ok(result)
    }
}

fn make_literal_array(
    lit: &NumericLiteral,
    len: usize,
    target_type: &DataType,
) -> LlkvResult<ArrayRef> {
    match target_type {
        DataType::Int64 => match lit {
            NumericLiteral::Int(v) => {
                if *v < i64::MIN as i128 || *v > i64::MAX as i128 {
                    return Err(Error::Internal("fast path literal out of i64 range".into()));
                }
                Ok(Arc::new(Int64Array::from_value(*v as i64, len)))
            }
            NumericLiteral::Float(v) => Ok(Arc::new(Int64Array::from_value(*v as i64, len))),
            NumericLiteral::Null => Ok(new_null_array(target_type, len)),
        },
        DataType::UInt64 => match lit {
            NumericLiteral::Int(v) => {
                if *v < 0 || *v > u64::MAX as i128 {
                    return Err(Error::Internal("fast path literal out of u64 range".into()));
                }
                Ok(Arc::new(arrow::array::UInt64Array::from_value(
                    *v as u64, len,
                )))
            }
            NumericLiteral::Float(v) => Ok(Arc::new(arrow::array::UInt64Array::from_value(
                *v as u64, len,
            ))),
            NumericLiteral::Null => Ok(new_null_array(target_type, len)),
        },
        DataType::Float64 => match lit {
            NumericLiteral::Int(v) => Ok(Arc::new(Float64Array::from_value(*v as f64, len))),
            NumericLiteral::Float(v) => Ok(Arc::new(Float64Array::from_value(*v, len))),
            NumericLiteral::Null => Ok(new_null_array(target_type, len)),
        },
        DataType::Float32 => match lit {
            NumericLiteral::Int(v) => Ok(Arc::new(arrow::array::Float32Array::from_value(
                *v as f32, len,
            ))),
            NumericLiteral::Float(v) => Ok(Arc::new(arrow::array::Float32Array::from_value(
                *v as f32, len,
            ))),
            NumericLiteral::Null => Ok(new_null_array(target_type, len)),
        },
        DataType::Int32 => match lit {
            NumericLiteral::Int(v) => Ok(Arc::new(arrow::array::Int32Array::from_value(
                *v as i32, len,
            ))),
            NumericLiteral::Float(v) => Ok(Arc::new(arrow::array::Int32Array::from_value(
                *v as i32, len,
            ))),
            NumericLiteral::Null => Ok(new_null_array(target_type, len)),
        },
        DataType::UInt32 => match lit {
            NumericLiteral::Int(v) => {
                if *v < 0 || *v > u32::MAX as i128 {
                    return Err(Error::Internal("fast path literal out of u32 range".into()));
                }
                Ok(Arc::new(arrow::array::UInt32Array::from_value(
                    *v as u32, len,
                )))
            }
            NumericLiteral::Float(v) => Ok(Arc::new(arrow::array::UInt32Array::from_value(
                *v as u32, len,
            ))),
            NumericLiteral::Null => Ok(new_null_array(target_type, len)),
        },
        DataType::Int16 => match lit {
            NumericLiteral::Int(v) => Ok(Arc::new(arrow::array::Int16Array::from_value(
                *v as i16, len,
            ))),
            NumericLiteral::Float(v) => Ok(Arc::new(arrow::array::Int16Array::from_value(
                *v as i16, len,
            ))),
            NumericLiteral::Null => Ok(new_null_array(target_type, len)),
        },
        DataType::UInt16 => match lit {
            NumericLiteral::Int(v) => {
                if *v < 0 || *v > u16::MAX as i128 {
                    return Err(Error::Internal("fast path literal out of u16 range".into()));
                }
                Ok(Arc::new(arrow::array::UInt16Array::from_value(
                    *v as u16, len,
                )))
            }
            NumericLiteral::Float(v) => Ok(Arc::new(arrow::array::UInt16Array::from_value(
                *v as u16, len,
            ))),
            NumericLiteral::Null => Ok(new_null_array(target_type, len)),
        },
        DataType::Int8 => match lit {
            NumericLiteral::Int(v) => {
                Ok(Arc::new(arrow::array::Int8Array::from_value(*v as i8, len)))
            }
            NumericLiteral::Float(v) => {
                Ok(Arc::new(arrow::array::Int8Array::from_value(*v as i8, len)))
            }
            NumericLiteral::Null => Ok(new_null_array(target_type, len)),
        },
        DataType::UInt8 => match lit {
            NumericLiteral::Int(v) => {
                if *v < 0 || *v > u8::MAX as i128 {
                    return Err(Error::Internal("fast path literal out of u8 range".into()));
                }
                Ok(Arc::new(arrow::array::UInt8Array::from_value(
                    *v as u8, len,
                )))
            }
            NumericLiteral::Float(v) => Ok(Arc::new(arrow::array::UInt8Array::from_value(
                *v as u8, len,
            ))),
            NumericLiteral::Null => Ok(new_null_array(target_type, len)),
        },
        _ => Err(Error::Internal(
            "unsupported literal type for numeric fast path".into(),
        )),
    }
}

struct PlanBuilder<'a, F> {
    tokens: Vec<Token>,
    column_slots: FxHashMap<F, usize>,
    column_order: Vec<F>,
    arrays: &'a NumericArrayMap<F>,
}

impl<'a, F: Copy + Eq + Hash> PlanBuilder<'a, F> {
    fn visit(&mut self, expr: &ScalarExpr<F>) -> Option<DataType> {
        match expr {
            ScalarExpr::Column(fid) => {
                let array = self.arrays.get(fid)?;
                if !ScalarEvaluator::is_supported_numeric(array.data_type()) {
                    return None;
                }
                let slot = *self.column_slots.entry(*fid).or_insert_with(|| {
                    self.column_order.push(*fid);
                    self.column_order.len() - 1
                });
                self.tokens.push(Token::Column { slot });
                Some(array.data_type().clone())
            }
            ScalarExpr::Literal(lit) => {
                let literal = NumericLiteral::from_literal(lit)?;
                self.tokens.push(Token::Literal(literal.clone()));
                match literal {
                    NumericLiteral::Int(_) | NumericLiteral::Null => Some(DataType::Int64),
                    NumericLiteral::Float(_) => Some(DataType::Float64),
                }
            }
            ScalarExpr::Binary { left, op, right } => {
                if *op == BinaryOp::Divide {
                    return None; // Use standard path to preserve integer-division semantics.
                }
                if !matches!(
                    op,
                    BinaryOp::Add
                        | BinaryOp::Subtract
                        | BinaryOp::Multiply
                        | BinaryOp::Divide
                        | BinaryOp::Modulo
                ) {
                    return None;
                }
                let left_type = self.visit(left)?;
                let right_type = self.visit(right)?;
                let result_type = common_type_for_op(&left_type, &right_type, *op);
                if !ScalarEvaluator::is_supported_numeric(&result_type) {
                    return None;
                }
                self.tokens.push(Token::Binary(*op));
                Some(result_type)
            }
            _ => None,
        }
    }
}

impl NumericLiteral {
    fn from_literal(lit: &Literal) -> Option<Self> {
        match lit {
            Literal::Int128(v) => Some(NumericLiteral::Int(*v)),
            Literal::Float64(v) => Some(NumericLiteral::Float(*v)),
            Literal::Decimal128(d) => Some(NumericLiteral::Int(d.raw_value())),
            Literal::Null => Some(NumericLiteral::Null),
            _ => None,
        }
    }
}

fn compute_binary_no_coerce(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: BinaryOp,
    dtype: &DataType,
) -> LlkvResult<ArrayRef> {
    macro_rules! binary_prim {
        ($arrow_ty:ty) => {{
            let l = lhs
                .as_any()
                .downcast_ref::<PrimitiveArray<$arrow_ty>>()
                .ok_or_else(|| Error::Internal("fast path lhs type mismatch".into()))?;
            let r = rhs
                .as_any()
                .downcast_ref::<PrimitiveArray<$arrow_ty>>()
                .ok_or_else(|| Error::Internal("fast path rhs type mismatch".into()))?;
            let arr = match op {
                BinaryOp::Add => numeric::add(l, r),
                BinaryOp::Subtract => numeric::sub(l, r),
                BinaryOp::Multiply => numeric::mul(l, r),
                BinaryOp::Divide => numeric::div(l, r),
                BinaryOp::Modulo => numeric::rem(l, r),
                _ => return Err(Error::Internal("unsupported op for fast path".into())),
            }
            .map_err(|e| Error::Internal(e.to_string()))?;
            Ok(Arc::new(arr) as ArrayRef)
        }};
    }

    match dtype {
        DataType::Float64 => binary_prim!(Float64Type),
        DataType::Float32 => binary_prim!(Float32Type),
        DataType::Int64 => binary_prim!(Int64Type),
        DataType::Int32 => binary_prim!(Int32Type),
        DataType::Int16 => binary_prim!(Int16Type),
        DataType::Int8 => binary_prim!(Int8Type),
        DataType::UInt64 => binary_prim!(UInt64Type),
        DataType::UInt32 => binary_prim!(UInt32Type),
        DataType::UInt16 => binary_prim!(UInt16Type),
        DataType::UInt8 => binary_prim!(UInt8Type),
        _ => Err(Error::Internal(
            "unsupported data type for fast path execution".into(),
        )),
    }
}
