use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, Scalar, UInt64Array, make_array};
use arrow::compute::{cast, kernels::cmp, kernels::numeric, nullif};
use arrow::datatypes::DataType;
use llkv_expr::expr::BinaryOp;
use llkv_result::Error;

use crate::array::NumericArray;
use crate::{NumericKind, NumericValue};

pub fn compute_binary(
    lhs: &NumericArray,
    rhs: &NumericArray,
    op: BinaryOp,
) -> Result<NumericArray, Error> {
    // Determine result type
    let result_kind = match op {
        BinaryOp::Divide => match (lhs.kind(), rhs.kind()) {
            (NumericKind::Decimal, _) | (_, NumericKind::Decimal) => NumericKind::Decimal,
            _ => NumericKind::Float,
        },
        _ => match (lhs.kind(), rhs.kind()) {
            (NumericKind::Float, _) | (_, NumericKind::Float) => NumericKind::Float,
            (NumericKind::Decimal, _) | (_, NumericKind::Decimal) => NumericKind::Decimal,
            (NumericKind::Integer, NumericKind::Integer) => NumericKind::Integer,
            (NumericKind::UnsignedInteger, NumericKind::UnsignedInteger) => {
                NumericKind::UnsignedInteger
            }
            (NumericKind::Integer, NumericKind::UnsignedInteger)
            | (NumericKind::UnsignedInteger, NumericKind::Integer) => NumericKind::Float,
            (NumericKind::String, _) | (_, NumericKind::String) => {
                return Err(Error::Internal(
                    "Cannot perform binary arithmetic on string arrays".to_string(),
                ));
            }
        },
    };

    // Cast inputs to result type (or compatible type)
    let (lhs_arr, rhs_arr) = cast_to_common_type(lhs, rhs, result_kind)?;

    let result_arr: ArrayRef = match op {
        BinaryOp::Add => match numeric::add(&lhs_arr, &rhs_arr) {
            Ok(res) => res,
            Err(e) if e.to_string().contains("overflow") => {
                let lhs_f = cast(&lhs_arr, &DataType::Float64)
                    .map_err(|e| Error::Internal(e.to_string()))?;
                let rhs_f = cast(&rhs_arr, &DataType::Float64)
                    .map_err(|e| Error::Internal(e.to_string()))?;
                numeric::add(&lhs_f, &rhs_f).map_err(|e| Error::Internal(e.to_string()))?
            }
            Err(e) => return Err(Error::Internal(e.to_string())),
        },
        BinaryOp::Subtract => match numeric::sub(&lhs_arr, &rhs_arr) {
            Ok(res) => res,
            Err(e) if e.to_string().contains("overflow") => {
                let lhs_f = cast(&lhs_arr, &DataType::Float64)
                    .map_err(|e| Error::Internal(e.to_string()))?;
                let rhs_f = cast(&rhs_arr, &DataType::Float64)
                    .map_err(|e| Error::Internal(e.to_string()))?;
                numeric::sub(&lhs_f, &rhs_f).map_err(|e| Error::Internal(e.to_string()))?
            }
            Err(e) => return Err(Error::Internal(e.to_string())),
        },
        BinaryOp::Multiply => {
            let res = match numeric::mul(&lhs_arr, &rhs_arr) {
                Ok(res) => res,
                Err(e) if e.to_string().contains("overflow") => {
                    let lhs_f = cast(&lhs_arr, &DataType::Float64)
                        .map_err(|e| Error::Internal(e.to_string()))?;
                    let rhs_f = cast(&rhs_arr, &DataType::Float64)
                        .map_err(|e| Error::Internal(e.to_string()))?;
                    numeric::mul(&lhs_f, &rhs_f).map_err(|e| Error::Internal(e.to_string()))?
                }
                Err(e) => return Err(Error::Internal(e.to_string())),
            };

            if result_kind == NumericKind::Decimal {
                if let DataType::Decimal128(p, s) = res.data_type() {
                    // Result of mul has scale 's' (same as inputs).
                    // But value is i1*i2.
                    // Correct scale should be s+s.
                    // We update the data type to reflect this.
                    let new_scale = (s + s).min(38);
                    let new_type = DataType::Decimal128(*p, new_scale);
                    let data = res.to_data();
                    let new_data = data
                        .into_builder()
                        .data_type(new_type)
                        .build()
                        .map_err(|e| Error::Internal(e.to_string()))?;
                    make_array(new_data)
                } else {
                    res
                }
            } else {
                res
            }
        }
        BinaryOp::Divide | BinaryOp::Modulo => {
            // Handle divide by zero by nulling out zeros in rhs
            let is_zero = match result_kind {
                NumericKind::Integer => {
                    let zero_scalar = Scalar::new(Int64Array::from(vec![0]));
                    cmp::eq(&rhs_arr, &zero_scalar)?
                }
                NumericKind::UnsignedInteger => {
                    let zero_scalar = Scalar::new(UInt64Array::from(vec![0]));
                    cmp::eq(&rhs_arr, &zero_scalar)?
                }
                NumericKind::Float => {
                    let zero_scalar = Scalar::new(Float64Array::from(vec![0.0]));
                    cmp::eq(&rhs_arr, &zero_scalar)?
                }
                NumericKind::Decimal => {
                    // Cast to Float64 to check for zero.
                    // This is safe because 10^-38 (min decimal) is > 10^-308 (min float).
                    // So any non-zero decimal will be non-zero float.
                    let rhs_float_arr = cast(&rhs_arr, &DataType::Float64)?;
                    let zero_scalar = Scalar::new(Float64Array::from(vec![0.0]));
                    cmp::eq(&rhs_float_arr, &zero_scalar)?
                }
                _ => return Err(Error::Internal("Unsupported type for division".into())),
            };

            let safe_rhs = nullif(&rhs_arr, &is_zero)?;

            if matches!(op, BinaryOp::Divide) {
                numeric::div(&lhs_arr, &safe_rhs)?
            } else {
                numeric::rem(&lhs_arr, &safe_rhs)?
            }
        }
        _ => return Err(Error::Internal(format!("Unsupported binary op {:?}", op))),
    };

    NumericArray::try_from_arrow(&result_arr)
}

pub fn compute_binary_scalar(
    array: &NumericArray,
    scalar: Option<NumericValue>,
    op: BinaryOp,
    array_is_left: bool,
) -> Result<NumericArray, Error> {
    let len = array.len();
    let scalar_values = vec![scalar.clone(); len];

    // Infer preferred kind from scalar
    let scalar_kind = scalar
        .as_ref()
        .map(|v| v.kind())
        .unwrap_or(NumericKind::Integer);

    let scalar_array = NumericArray::from_numeric_values(scalar_values, scalar_kind);

    if array_is_left {
        compute_binary(array, &scalar_array, op)
    } else {
        compute_binary(&scalar_array, array, op)
    }
}

fn cast_to_common_type(
    lhs: &NumericArray,
    rhs: &NumericArray,
    target_kind: NumericKind,
) -> Result<(ArrayRef, ArrayRef), Error> {
    match target_kind {
        NumericKind::Integer => {
            // Both should be Integer
            Ok((lhs.to_array_ref(), rhs.to_array_ref()))
        }
        NumericKind::UnsignedInteger => {
            // Both should be UnsignedInteger
            Ok((lhs.to_array_ref(), rhs.to_array_ref()))
        }
        NumericKind::Float => {
            let lhs_cast = cast(&lhs.to_array_ref(), &DataType::Float64)?;
            let rhs_cast = cast(&rhs.to_array_ref(), &DataType::Float64)?;
            Ok((lhs_cast, rhs_cast))
        }
        NumericKind::Decimal => {
            // Determine common decimal type (max precision, max scale)
            let lhs_type = lhs.to_array_ref().data_type().clone();
            let rhs_type = rhs.to_array_ref().data_type().clone();

            let target_scale = match (lhs_type, rhs_type) {
                (DataType::Decimal128(_, s1), DataType::Decimal128(_, s2)) => s1.max(s2),
                (DataType::Decimal128(_, s), _) => s,
                (_, DataType::Decimal128(_, s)) => s,
                _ => 10, // Default fallback
            };

            // Use max precision to avoid overflow during intermediate ops
            let target_type = DataType::Decimal128(38, target_scale);

            let lhs_cast = cast(&lhs.to_array_ref(), &target_type)?;
            let rhs_cast = cast(&rhs.to_array_ref(), &target_type)?;
            Ok((lhs_cast, rhs_cast))
        }
        NumericKind::String => Err(Error::Internal(
            "Cannot cast to String for arithmetic".to_string(),
        )),
    }
}
