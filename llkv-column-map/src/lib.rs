pub mod store;
pub mod types;

pub use llkv_result::{Error, Result};
pub use store::{
    ColumnStore, IndexKind,
    scan::{self, ScanBuilder},
    ROW_ID_COLUMN_NAME
};

pub mod debug {
    pub use super::store::debug::*;
}

/// Expands to the provided body with `$ty` bound to the concrete Arrow primitive type that
/// matches the supplied `DataType`. Integer and floating-point primitives are supported; any
/// other `DataType` triggers the `$unsupported` expression. This is used to avoid dynamic
/// dispatch in hot paths like scans and row gathers.
#[macro_export]
macro_rules! with_integer_arrow_type {
    ($dtype:expr, |$ty:ident| $body:expr, $unsupported:expr $(,)?) => {{
        match $dtype {
            arrow::datatypes::DataType::UInt64 => {
                type $ty = arrow::datatypes::UInt64Type;
                $body
            }
            arrow::datatypes::DataType::UInt32 => {
                type $ty = arrow::datatypes::UInt32Type;
                $body
            }
            arrow::datatypes::DataType::UInt16 => {
                type $ty = arrow::datatypes::UInt16Type;
                $body
            }
            arrow::datatypes::DataType::UInt8 => {
                type $ty = arrow::datatypes::UInt8Type;
                $body
            }
            arrow::datatypes::DataType::Int64 => {
                type $ty = arrow::datatypes::Int64Type;
                $body
            }
            arrow::datatypes::DataType::Int32 => {
                type $ty = arrow::datatypes::Int32Type;
                $body
            }
            arrow::datatypes::DataType::Int16 => {
                type $ty = arrow::datatypes::Int16Type;
                $body
            }
            arrow::datatypes::DataType::Int8 => {
                type $ty = arrow::datatypes::Int8Type;
                $body
            }
            arrow::datatypes::DataType::Float64 => {
                type $ty = arrow::datatypes::Float64Type;
                $body
            }
            arrow::datatypes::DataType::Float32 => {
                type $ty = arrow::datatypes::Float32Type;
                $body
            }
            _ => $unsupported,
        }
    }};
}
