use arrow::array::{
    Array, ArrayRef, Date32Array, IntervalMonthDayNanoArray, Scalar, new_null_array,
};
use arrow::compute::kernels::cmp;
use arrow::compute::{cast, kernels::numeric, nullif};
use arrow::datatypes::{DataType, IntervalMonthDayNanoType};
use llkv_expr::expr::{BinaryOp, CompareOp};
use llkv_result::Error;
use llkv_types::IntervalValue;
use std::sync::Arc;

use crate::date::{add_interval_to_date32, subtract_interval_from_date32};

fn numeric_priority(dt: &DataType) -> Option<Numeric> {
    match dt {
        DataType::Int8 => Some(Numeric::Signed(8)),
        DataType::Int16 => Some(Numeric::Signed(16)),
        DataType::Int32 => Some(Numeric::Signed(32)),
        DataType::Int64 => Some(Numeric::Signed(64)),
        DataType::UInt8 => Some(Numeric::Unsigned(8)),
        DataType::UInt16 => Some(Numeric::Unsigned(16)),
        DataType::UInt32 => Some(Numeric::Unsigned(32)),
        DataType::UInt64 => Some(Numeric::Unsigned(64)),
        DataType::Float32 => Some(Numeric::F32),
        DataType::Float64 => Some(Numeric::F64),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Numeric {
    Signed(u8),
    Unsigned(u8),
    F32,
    F64,
}

fn coerce_decimals(lhs: (u8, i8), rhs: (u8, i8)) -> DataType {
    let scale = lhs.1.max(rhs.1);
    let lhs_int = i32::from(lhs.0) - i32::from(lhs.1);
    let rhs_int = i32::from(rhs.0) - i32::from(rhs.1);
    let int_digits = lhs_int.max(rhs_int);
    let precision = (int_digits + i32::from(scale)).clamp(1, 38) as u8;
    let precision = std::cmp::max(precision, scale as u8);
    DataType::Decimal128(precision, scale)
}

fn compute_date_interval_op(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: BinaryOp,
) -> Result<Option<ArrayRef>, Error> {
    let (date_arr, interval_arr, swap) = match (lhs.data_type(), rhs.data_type()) {
        (DataType::Date32, DataType::Interval(_)) => (lhs, rhs, false),
        (DataType::Interval(_), DataType::Date32) => (rhs, lhs, true),
        _ => return Ok(None),
    };

    let dates = date_arr.as_any().downcast_ref::<Date32Array>().unwrap();
    let intervals = interval_arr
        .as_any()
        .downcast_ref::<IntervalMonthDayNanoArray>()
        .unwrap();

    let len = dates.len();
    let mut result_builder = arrow::array::Date32Builder::with_capacity(len);

    for i in 0..len {
        if dates.is_null(i) || intervals.is_null(i) {
            result_builder.append_null();
            continue;
        }

        let date_val = dates.value(i);
        let interval_val = intervals.value(i);
        let (months, days, nanos) = IntervalMonthDayNanoType::to_parts(interval_val);
        let interval = IntervalValue::new(months, days, nanos);

        let res = match op {
            BinaryOp::Add => add_interval_to_date32(date_val, interval),
            BinaryOp::Subtract => {
                if swap {
                    return Err(Error::Internal("Cannot subtract Date from Interval".into()));
                } else {
                    subtract_interval_from_date32(date_val, interval)
                }
            }
            _ => return Ok(None),
        };

        match res {
            Ok(val) => result_builder.append_value(val),
            Err(e) => return Err(e),
        }
    }

    Ok(Some(Arc::new(result_builder.finish())))
}

pub fn compute_binary(lhs: &ArrayRef, rhs: &ArrayRef, op: BinaryOp) -> Result<ArrayRef, Error> {
    if let Some(result) = compute_date_interval_op(lhs, rhs, op)? {
        return Ok(result);
    }

    // Coerce inputs to common type
    let (lhs_arr, rhs_arr) = coerce_types(lhs, rhs, op)?;

    if lhs_arr.data_type() == &DataType::Null {
        return Ok(new_null_array(&DataType::Null, lhs_arr.len()));
    }

    let result_arr: ArrayRef = match op {
        BinaryOp::Add => {
            numeric::add(&lhs_arr, &rhs_arr).map_err(|e| Error::Internal(e.to_string()))?
        }
        BinaryOp::Subtract => {
            numeric::sub(&lhs_arr, &rhs_arr).map_err(|e| Error::Internal(e.to_string()))?
        }
        BinaryOp::Multiply => {
            numeric::mul(&lhs_arr, &rhs_arr).map_err(|e| Error::Internal(e.to_string()))?
        }
        BinaryOp::Divide => {
            // Handle division by zero by treating 0s as NULLs
            let zero = arrow::array::Int64Array::from(vec![0]);
            let zero = cast(&zero, rhs_arr.data_type())
                .map_err(|e| Error::Internal(format!("Failed to cast 0: {}", e)))?;
            let zero_scalar = Scalar::new(zero);

            let is_zero = cmp::eq(&rhs_arr, &zero_scalar)
                .map_err(|e| Error::Internal(format!("Failed to compare with 0: {}", e)))?;

            let safe_rhs = nullif(&rhs_arr, &is_zero)
                .map_err(|e| Error::Internal(format!("Failed to nullif zeros: {}", e)))?;

            numeric::div(&lhs_arr, &safe_rhs).map_err(|e| Error::Internal(e.to_string()))?
        }
        BinaryOp::Modulo => {
            numeric::rem(&lhs_arr, &rhs_arr).map_err(|e| Error::Internal(e.to_string()))?
        }
        BinaryOp::And => {
            let lhs_bool =
                cast(&lhs_arr, &DataType::Boolean).map_err(|e| Error::Internal(e.to_string()))?;
            let rhs_bool =
                cast(&rhs_arr, &DataType::Boolean).map_err(|e| Error::Internal(e.to_string()))?;
            let lhs_bool = lhs_bool
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .unwrap();
            let rhs_bool = rhs_bool
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .unwrap();
            let result = arrow::compute::kernels::boolean::and_kleene(lhs_bool, rhs_bool)
                .map_err(|e| Error::Internal(e.to_string()))?;
            Arc::new(result)
        }
        BinaryOp::Or => {
            let lhs_bool =
                cast(&lhs_arr, &DataType::Boolean).map_err(|e| Error::Internal(e.to_string()))?;
            let rhs_bool =
                cast(&rhs_arr, &DataType::Boolean).map_err(|e| Error::Internal(e.to_string()))?;
            let lhs_bool = lhs_bool
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .unwrap();
            let rhs_bool = rhs_bool
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .unwrap();
            let result = arrow::compute::kernels::boolean::or_kleene(lhs_bool, rhs_bool)
                .map_err(|e| Error::Internal(e.to_string()))?;
            Arc::new(result)
        }
        _ => return Err(Error::Internal(format!("Unsupported binary op: {:?}", op))),
    };

    Ok(result_arr)
}

pub fn get_common_type(lhs_type: &DataType, rhs_type: &DataType) -> DataType {
    if lhs_type == rhs_type {
        return lhs_type.clone();
    }

    match (lhs_type, rhs_type) {
        (DataType::Null, other) | (other, DataType::Null) => other.clone(),
        (DataType::Boolean, DataType::Boolean) => DataType::Boolean,
        (DataType::Decimal128(lp, ls), DataType::Decimal128(rp, rs)) => {
            coerce_decimals((*lp, *ls), (*rp, *rs))
        }
        (DataType::Decimal128(p, s), other) | (other, DataType::Decimal128(p, s)) => match other {
            DataType::Float64 | DataType::Float32 => DataType::Float64,
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => coerce_decimals((*p, *s), (38u8, 0)),
            _ => DataType::Float64,
        },
        (DataType::Date32, DataType::Interval(_)) | (DataType::Interval(_), DataType::Date32) => {
            DataType::Date32
        }
        _ => match (numeric_priority(lhs_type), numeric_priority(rhs_type)) {
            (Some(Numeric::F64), _) | (_, Some(Numeric::F64)) => DataType::Float64,
            (Some(Numeric::F32), _) | (_, Some(Numeric::F32)) => DataType::Float64,
            (Some(Numeric::Signed(lhs)), Some(Numeric::Unsigned(rhs)))
            | (Some(Numeric::Unsigned(lhs)), Some(Numeric::Signed(rhs))) => {
                let max = std::cmp::max(lhs, rhs);
                if max >= 64 {
                    DataType::Float64
                } else {
                    DataType::Int64
                }
            }
            (Some(Numeric::Signed(lhs)), Some(Numeric::Signed(rhs))) => {
                if lhs >= 64 || rhs >= 64 {
                    DataType::Int64
                } else if lhs >= 32 || rhs >= 32 {
                    DataType::Int32
                } else if lhs >= 16 || rhs >= 16 {
                    DataType::Int16
                } else {
                    DataType::Int8
                }
            }
            (Some(Numeric::Unsigned(lhs)), Some(Numeric::Unsigned(rhs))) => {
                if lhs >= 64 || rhs >= 64 {
                    DataType::UInt64
                } else if lhs >= 32 || rhs >= 32 {
                    DataType::UInt32
                } else if lhs >= 16 || rhs >= 16 {
                    DataType::UInt16
                } else {
                    DataType::UInt8
                }
            }
            _ => DataType::Float64,
        },
    }
}

/// Common type for a binary operator.
pub fn common_type_for_op(lhs_type: &DataType, rhs_type: &DataType, _op: BinaryOp) -> DataType {
    get_common_type(lhs_type, rhs_type)
}

pub fn coerce_types(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: BinaryOp,
) -> Result<(ArrayRef, ArrayRef), Error> {
    let lhs_type = lhs.data_type();
    let rhs_type = rhs.data_type();

    let target_type = common_type_for_op(lhs_type, rhs_type, op);

    if lhs_type == rhs_type && lhs_type == &target_type {
        return Ok((lhs.clone(), rhs.clone()));
    }

    let lhs_casted = cast(lhs, &target_type).map_err(|e| Error::Internal(e.to_string()))?;
    let rhs_casted = cast(rhs, &target_type).map_err(|e| Error::Internal(e.to_string()))?;

    Ok((lhs_casted, rhs_casted))
}

pub fn compute_compare(lhs: &ArrayRef, op: CompareOp, rhs: &ArrayRef) -> Result<ArrayRef, Error> {
    // Coerce inputs to common type for comparison
    // We can reuse coerce_types logic or similar.
    // For comparison, we usually want common type.
    // We can pass a dummy BinaryOp to coerce_types.
    let (lhs_arr, rhs_arr) = coerce_types(lhs, rhs, BinaryOp::Add)?;

    let result_arr: ArrayRef = match op {
        CompareOp::Eq => {
            Arc::new(cmp::eq(&lhs_arr, &rhs_arr).map_err(|e| Error::Internal(e.to_string()))?)
        }
        CompareOp::NotEq => {
            Arc::new(cmp::neq(&lhs_arr, &rhs_arr).map_err(|e| Error::Internal(e.to_string()))?)
        }
        CompareOp::Lt => {
            Arc::new(cmp::lt(&lhs_arr, &rhs_arr).map_err(|e| Error::Internal(e.to_string()))?)
        }
        CompareOp::LtEq => {
            Arc::new(cmp::lt_eq(&lhs_arr, &rhs_arr).map_err(|e| Error::Internal(e.to_string()))?)
        }
        CompareOp::Gt => {
            Arc::new(cmp::gt(&lhs_arr, &rhs_arr).map_err(|e| Error::Internal(e.to_string()))?)
        }
        CompareOp::GtEq => {
            Arc::new(cmp::gt_eq(&lhs_arr, &rhs_arr).map_err(|e| Error::Internal(e.to_string()))?)
        }
    };
    Ok(result_arr)
}
