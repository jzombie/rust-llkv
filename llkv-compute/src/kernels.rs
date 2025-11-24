use arrow::array::{Array, ArrayRef};
use arrow::compute::kernels::cmp;
use arrow::compute::{cast, kernels::numeric};
use arrow::datatypes::DataType;
use llkv_expr::expr::{BinaryOp, CompareOp};
use llkv_result::Error;
use std::sync::Arc;

pub fn compute_binary(lhs: &ArrayRef, rhs: &ArrayRef, op: BinaryOp) -> Result<ArrayRef, Error> {
    // Coerce inputs to common type
    let (lhs_arr, rhs_arr) = coerce_types(lhs, rhs, op)?;

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
            numeric::div(&lhs_arr, &rhs_arr).map_err(|e| Error::Internal(e.to_string()))?
        }
        BinaryOp::Modulo => {
            numeric::rem(&lhs_arr, &rhs_arr).map_err(|e| Error::Internal(e.to_string()))?
        }
        _ => return Err(Error::Internal(format!("Unsupported binary op: {:?}", op))),
    };

    Ok(result_arr)
}

fn coerce_types(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    _op: BinaryOp,
) -> Result<(ArrayRef, ArrayRef), Error> {
    let lhs_type = lhs.data_type();
    let rhs_type = rhs.data_type();

    if lhs_type == rhs_type {
        return Ok((lhs.clone(), rhs.clone()));
    }

    // Simple coercion rules
    // TODO: Implement full type coercion matrix (like DataFusion)
    let target_type = match (lhs_type, rhs_type) {
        (DataType::Float64, _) | (_, DataType::Float64) => DataType::Float64,
        (DataType::Float32, _) | (_, DataType::Float32) => DataType::Float64, // Promote to f64

        (DataType::Int64, DataType::Int64) => DataType::Int64,
        (DataType::Int32, DataType::Int32) => DataType::Int32,
        (DataType::Int16, DataType::Int16) => DataType::Int16,
        (DataType::Int8, DataType::Int8) => DataType::Int8,

        (DataType::Int32, DataType::Int64) | (DataType::Int64, DataType::Int32) => DataType::Int64,
        (DataType::Int16, DataType::Int64) | (DataType::Int64, DataType::Int16) => DataType::Int64,
        (DataType::Int8, DataType::Int64) | (DataType::Int64, DataType::Int8) => DataType::Int64,

        (DataType::Int16, DataType::Int32) | (DataType::Int32, DataType::Int16) => DataType::Int32,
        (DataType::Int8, DataType::Int32) | (DataType::Int32, DataType::Int8) => DataType::Int32,

        (DataType::Int8, DataType::Int16) | (DataType::Int16, DataType::Int8) => DataType::Int16,

        (DataType::Int64, DataType::UInt64) => DataType::Float64, // Avoid overflow
        (DataType::UInt64, DataType::Int64) => DataType::Float64,
        (DataType::UInt64, DataType::UInt64) => DataType::UInt64,
        // ... handle other types
        _ => DataType::Float64, // Fallback
    };

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
        _ => return Err(Error::Internal(format!("Unsupported compare op: {:?}", op))),
    };
    Ok(result_arr)
}
