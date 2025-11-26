//! Columnar storage engine for LLKV.
//!
//! This crate provides the low-level columnar layer that persists Apache Arrow
//! [`RecordBatch`]es to disk and supports efficient scans, filters, and updates.
//! It serves as the foundation for [`llkv-table`] and higher-level query
//! execution.
//!
//! # Role in the Story
//!
//! The column map is where LLKV’s Arrow-first design meets pager-backed
//! persistence. Every [`sqllogictest`](https://sqlite.org/sqllogictest/doc/trunk/about.wiki) shipped with SQLite—and an expanding set of
//! DuckDB suites—ultimately routes through these descriptors and chunk walkers.
//! The storage layer therefore carries the burden of matching SQLite semantics
//! while staying efficient enough for OLAP workloads. Gaps uncovered by the
//! logic tests are treated as defects in this crate, not harness exceptions.
//!
//! The engine is maintained in the open by a single developer. These docs aim
//! to give newcomers the same context captured in the README and DeepWiki pages
//! so the story remains accessible as the project grows.
//!
//! # Architecture
//!
//! The storage engine is organized into several key components:
//!
//! - **[`ColumnStore`]**: Primary interface for storing and retrieving columnar data.
//!   Manages column descriptors, metadata catalogs, and coordinates with the pager
//!   for persistent storage.
//!
//! - **[`LogicalFieldId`](types::LogicalFieldId)**: Namespaced identifier for columns.
//!   Combines a namespace (user data, row ID shadow, MVCC metadata), table ID, and
//!   field ID into a single 64-bit value to prevent collisions.
//!
//! - **[`ScanBuilder`]**: Builder pattern for constructing column scans with various
//!   options (filters, ordering, row ID inclusion).
//!
//! - **Visitor Pattern**: Scans emit data through visitor callbacks rather than
//!   materializing entire columns in memory, enabling streaming and aggregation.
//!
//! # Storage Model
//!
//! Data is stored in columnar chunks:
//! - Each column is identified by a `LogicalFieldId`
//! - Columns are broken into chunks for incremental writes
//! - Each chunk stores Arrow-serialized data plus metadata (row count, min/max values)
//! - Shadow columns track row IDs separately from user data
//! - MVCC columns (`created_by`, `deleted_by`) track transaction visibility
//!
//! # Namespaces
//!
//! Columns are organized into namespaces to prevent ID collisions:
//! - `UserData`: Regular table columns
//! - `RowIdShadow`: Internal row ID tracking for each column
//! - `TxnCreatedBy`: MVCC transaction that created each row
//! - `TxnDeletedBy`: MVCC transaction that deleted each row
//!
//! # Test Coverage
//!
//! - **SQLite suites**: The storage layer powers every SQLite [`sqllogictest`](https://sqlite.org/sqllogictest/doc/trunk/about.wiki)
//!   case that upstream publishes. Passing those suites provides a baseline for
//!   SQLite compatibility, but LLKV still diverges from SQLite behavior in
//!   places and should not be treated as a drop-in replacement yet.
//! - **DuckDB extensions**: DuckDB-focused suites exercise MVCC edge cases and
//!   typed transaction flows. Coverage is early and informs the roadmap rather
//!   than proving full DuckDB parity today. All suites run through the
//!   [`sqllogictest` crate](https://crates.io/crates/sqllogictest).
//!
//! # Thread Safety
//!
//! `ColumnStore` is thread-safe (`Send + Sync`) with internal locking for
//! catalog updates. Read operations can occur concurrently; writes are
//! serialized through the catalog lock.
//!
//! [`RecordBatch`]: arrow::record_batch::RecordBatch
//! [`llkv-table`]: https://docs.rs/llkv-table
//! [`ColumnStore`]: store::ColumnStore
//! [`ScanBuilder`]: scan::ScanBuilder
//!
//! # Macros and Type Dispatch
//!
//! This crate provides macros for efficient type-specific operations without runtime
//! dispatch overhead. See [`with_integer_arrow_type!`] for details.

// NOTE: rustfmt currently re-indents portions of macro_rules! blocks in this
// file (observed when running `cargo fmt`). This produces noisy diffs and
// churn because rustfmt will flip formatting between runs. The problematic
// locations in this module are the macro_rules! dispatch macros declared
// below. Until the underlying rustfmt bug is fixed, we intentionally opt out
// of automatic formatting for those specific macros using `#[rustfmt::skip]`,
// while keeping the rest of the module formatted normally.
//
// Reproduction / debugging tips for contributors:
// - Run `rustup run stable rustfmt -- --version` to confirm the rustfmt
//   version, then `cargo fmt` to reproduce the behavior.
// - Narrow the change by running rustfmt on this file only:
//     rustfmt llkv-column-map/src/store/scan/unsorted.rs
// - If you can produce a minimal self-contained example that triggers the
//   re-indent, open an issue with rustfmt (include rustfmt version and the
//   minimal example) and link it here.
//
// NOTE: Once a minimal reproducer for the rustfmt regression exists, link the
// upstream issue here and remove the `#[rustfmt::skip]` attributes so the file
// can return to standard formatting. Progress is tracked at
// https://github.com/rust-lang/rustfmt/issues/6629#issuecomment-3395446770.

/// Dispatches to type-specific code based on an Arrow `DataType`.
///
/// This macro eliminates runtime type checking by expanding to type-specific code
/// at compile time. It matches the provided `DataType` against supported numeric types
/// and binds the corresponding Arrow primitive type to the specified identifier.
///
/// # Parameters
///
/// - `$dtype` - Expression evaluating to `&arrow::datatypes::DataType`
/// - `$ty` - Identifier to bind the Arrow primitive type to (e.g., `UInt64Type`)
/// - `$body` - Code to execute with `$ty` bound to the matched type
/// - `$unsupported` - Fallback expression if the type is not supported
///
/// # Performance
///
/// This macro is used in hot paths to avoid runtime `match` statements and virtual
/// dispatch. The compiler generates specialized code for each type.
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

/// Invokes a macro for each supported Arrow numeric type.
///
/// This is a helper macro that generates repetitive type-specific code. It calls
/// the provided macro once for each numeric Arrow type with metadata about that type.
///
/// # Macro Arguments Provided to Callback
///
/// For each type, the callback macro receives:
/// 1. Base type name (e.g., `u64`, `i32`, `f64`)
/// 2. Chunk visitor method name (e.g., `u64_chunk`)
/// 3. Chunk with row IDs visitor method name (e.g., `u64_chunk_with_rids`)
/// 4. Run visitor method name (e.g., `u64_run`)
/// 5. Run with row IDs visitor method name (e.g., `u64_run_with_rids`)
/// 6. Arrow array type (e.g., `arrow::array::UInt64Array`)
/// 7. Arrow physical type (e.g., `arrow::datatypes::UInt64Type`)
/// 8. Arrow DataType enum variant (e.g., `arrow::datatypes::DataType::UInt64`)
/// 9. Native Rust type (e.g., `u64`)
/// 10. Cast expression for type conversion
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

#[macro_export]
#[rustfmt::skip]
macro_rules! llkv_for_each_arrow_string {
    ($macro:ident) => {
        $macro!(
            utf8,
            utf8_chunk,
            utf8_chunk_with_rids,
            utf8_run,
            utf8_run_with_rids,
            arrow::array::StringArray,
            arrow::datatypes::Utf8Type,
            arrow::datatypes::DataType::Utf8,
            &str,
            |_v: &str| 0.0
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

pub mod codecs;
pub mod gather;
pub mod parallel;
pub mod serialization;
pub mod store;

pub use llkv_result::{Error, Result};
pub use store::{
    ColumnStore, IndexKind, ROW_ID_COLUMN_NAME,
    scan::{self, ScanBuilder},
};

pub mod debug {
    pub use super::store::debug::*;
}
