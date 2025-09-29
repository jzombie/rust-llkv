pub mod store;
pub mod types;

pub use llkv_result::{Error, Result};
pub use store::{
    ColumnStore, IndexKind, ROW_ID_COLUMN_NAME,
    scan::{self, ScanBuilder},
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
        use std::borrow::Borrow;

        let dtype_value = $dtype;
        let dtype_ref: &arrow::datatypes::DataType = dtype_value.borrow();
        let mut result: Option<_> = None;

        macro_rules! __llkv_dispatch_integer_arrow_type {
            (
                    $base:ident,
                    $chunk_fn:ident,
                    $chunk_with_rids_fn:ident,
                    $run_fn:ident,
                    $run_with_rids_fn:ident,
                    $array_ty:ty,
                    $physical_ty:ty,
                    $dtype_expr:expr,
                    $native_ty:ty,
                    $cast_expr:expr
                ) => {
                if dtype_ref == &$dtype_expr {
                    type $ty = $physical_ty;
                    result = Some($body);
                }
            };
        }

        $crate::llkv_for_each_arrow_numeric!(__llkv_dispatch_integer_arrow_type);

        result.unwrap_or_else(|| $unsupported)
    }};
}

/// Invokes `$macro` with metadata for each supported Arrow numeric primitive.
///
/// The callback receives the following tokens per invocation:
///
/// ```text
/// $macro!(
///     $base,               // identifier prefix (e.g. u64)
///     $chunk_fn,           // chunk visitor method name (e.g. u64_chunk)
///     $chunk_rids_fn,      // chunk-with-row-ids method name
///     $run_fn,             // sorted run method name
///     $run_rids_fn,        // sorted run-with-row-ids method name
///     $array_ty,           // Arrow array type (e.g. arrow::array::UInt64Array)
///     $physical_ty,        // Arrow physical type (e.g. arrow::datatypes::UInt64Type)
///     $dtype_expr,         // arrow::datatypes::DataType variant expression
///     $native_ty,          // native Rust value type (e.g. u64)
///     $cast_expr           // expression that casts native values to f64
/// );
/// ```
#[macro_export]
macro_rules! llkv_for_each_arrow_numeric {
    ($macro:ident) => {
        $macro!(
            u64,
            u64_chunk,
            u64_chunk_with_rids,
            u64_run,
            u64_run_with_rids,
            arrow::array::UInt64Array,
            arrow::datatypes::UInt64Type,
            arrow::datatypes::DataType::UInt64,
            u64,
            |v: u64| v as f64
        );
        $macro!(
            u32,
            u32_chunk,
            u32_chunk_with_rids,
            u32_run,
            u32_run_with_rids,
            arrow::array::UInt32Array,
            arrow::datatypes::UInt32Type,
            arrow::datatypes::DataType::UInt32,
            u32,
            |v: u32| v as f64
        );
        $macro!(
            u16,
            u16_chunk,
            u16_chunk_with_rids,
            u16_run,
            u16_run_with_rids,
            arrow::array::UInt16Array,
            arrow::datatypes::UInt16Type,
            arrow::datatypes::DataType::UInt16,
            u16,
            |v: u16| v as f64
        );
        $macro!(
            u8,
            u8_chunk,
            u8_chunk_with_rids,
            u8_run,
            u8_run_with_rids,
            arrow::array::UInt8Array,
            arrow::datatypes::UInt8Type,
            arrow::datatypes::DataType::UInt8,
            u8,
            |v: u8| v as f64
        );
        $macro!(
            i64,
            i64_chunk,
            i64_chunk_with_rids,
            i64_run,
            i64_run_with_rids,
            arrow::array::Int64Array,
            arrow::datatypes::Int64Type,
            arrow::datatypes::DataType::Int64,
            i64,
            |v: i64| v as f64
        );
        $macro!(
            i32,
            i32_chunk,
            i32_chunk_with_rids,
            i32_run,
            i32_run_with_rids,
            arrow::array::Int32Array,
            arrow::datatypes::Int32Type,
            arrow::datatypes::DataType::Int32,
            i32,
            |v: i32| v as f64
        );
        $macro!(
            i16,
            i16_chunk,
            i16_chunk_with_rids,
            i16_run,
            i16_run_with_rids,
            arrow::array::Int16Array,
            arrow::datatypes::Int16Type,
            arrow::datatypes::DataType::Int16,
            i16,
            |v: i16| v as f64
        );
        $macro!(
            i8,
            i8_chunk,
            i8_chunk_with_rids,
            i8_run,
            i8_run_with_rids,
            arrow::array::Int8Array,
            arrow::datatypes::Int8Type,
            arrow::datatypes::DataType::Int8,
            i8,
            |v: i8| v as f64
        );
        $macro!(
            f64,
            f64_chunk,
            f64_chunk_with_rids,
            f64_run,
            f64_run_with_rids,
            arrow::array::Float64Array,
            arrow::datatypes::Float64Type,
            arrow::datatypes::DataType::Float64,
            f64,
            |v: f64| v
        );
        $macro!(
            f32,
            f32_chunk,
            f32_chunk_with_rids,
            f32_run,
            f32_run_with_rids,
            arrow::array::Float32Array,
            arrow::datatypes::Float32Type,
            arrow::datatypes::DataType::Float32,
            f32,
            |v: f32| v as f64
        );
    };
}
