// TODO: Remove; test commit

// NOTE: rustfmt appears to repeatedly re-indent portions of some macros in
// this file when running `cargo fmt` (likely a rustfmt bug). To avoid noisy
// diffs and churn, skip automatic formatting on the affected macro_rules!
// declarations. Keep the rest of the module formatted normally.

/// Expands to the provided body with `$ty` bound to the concrete Arrow primitive type that
/// matches the supplied `DataType`. Integer and floating-point primitives are supported; any
/// other `DataType` triggers the `$unsupported` expression. This is used to avoid dynamic
/// dispatch in hot paths like scans and row gathers.
#[macro_export]
#[rustfmt::skip]
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

        llkv_for_each_arrow_numeric!(__llkv_dispatch_integer_arrow_type);

        result.unwrap_or_else(|| $unsupported)
    }};
}

/// Invokes `$macro` with metadata for each supported Arrow numeric primitive.
#[macro_export]
#[rustfmt::skip]
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
        $macro!(
            date64,
            date64_chunk,
            date64_chunk_with_rids,
            date64_run,
            date64_run_with_rids,
            arrow::array::Date64Array,
            arrow::datatypes::Date64Type,
            arrow::datatypes::DataType::Date64,
            i64,
            |v: i64| v as f64
        );
        $macro!(
            date32,
            date32_chunk,
            date32_chunk_with_rids,
            date32_run,
            date32_run_with_rids,
            arrow::array::Date32Array,
            arrow::datatypes::Date32Type,
            arrow::datatypes::DataType::Date32,
            i32,
            |v: i32| v as f64
        );
    };
}

#[macro_export]
#[rustfmt::skip]
macro_rules! llkv_for_each_arrow_boolean {
    ($macro:ident) => {
        $macro!(
            bool,
            bool_chunk,
            bool_chunk_with_rids,
            bool_run,
            bool_run_with_rids,
            arrow::array::BooleanArray,
            arrow::datatypes::BooleanType,
            arrow::datatypes::DataType::Boolean,
            bool,
            |v: bool| if v { 1.0 } else { 0.0 }
        );
    };
}

pub fn is_supported_arrow_type(dtype: &arrow::datatypes::DataType) -> bool {
    use arrow::datatypes::DataType;

    if matches!(dtype, DataType::Utf8 | DataType::LargeUtf8) {
        return true;
    }

    let mut matched = false;

    macro_rules! __llkv_match_dtype {
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
            if dtype == &$dtype_expr {
                matched = true;
            }
        };
    }

    llkv_for_each_arrow_numeric!(__llkv_match_dtype);
    llkv_for_each_arrow_boolean!(__llkv_match_dtype);

    matched
}

pub fn supported_arrow_types() -> Vec<arrow::datatypes::DataType> {
    use arrow::datatypes::DataType;

    let mut types = vec![DataType::Utf8, DataType::LargeUtf8];

    macro_rules! __llkv_push_dtype {
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
            types.push($dtype_expr.clone());
        };
    }

    llkv_for_each_arrow_numeric!(__llkv_push_dtype);
    llkv_for_each_arrow_boolean!(__llkv_push_dtype);

    types
}

pub fn ensure_supported_arrow_type(dtype: &arrow::datatypes::DataType) -> Result<()> {
    if is_supported_arrow_type(dtype) {
        return Ok(());
    }

    let mut supported = supported_arrow_types()
        .into_iter()
        .map(|dtype| format!("{dtype:?}"))
        .collect::<Vec<_>>();
    supported.sort();
    supported.dedup();

    Err(Error::InvalidArgumentError(format!(
        "unsupported Arrow type {dtype:?}; supported types are {}",
        supported.join(", ")
    )))
}

pub mod parallel;
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
