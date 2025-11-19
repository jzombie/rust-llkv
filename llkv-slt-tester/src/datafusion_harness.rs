//! SQL-engine-based test harness for sqllogictest integration.
//!
//! This module wires the resurrected `llkv-sql` engine into sqllogictest's
//! [`AsyncDB`] trait. The SQL engine intercepts DDL such as `CREATE TABLE`
//! before delegating all other statements to DataFusion via `llkv-fusion`.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    Array as ArrowArray, BooleanArray, Float64Array, Int64Array, StringArray, StructArray,
    UInt64Array,
};
use arrow::datatypes::DataType;
use llkv_result::Error;
use llkv_sql::{SqlEngine, SqlStatementResult};
use llkv_storage::pager::{BoxedPager, MemPager};
use sqllogictest::{AsyncDB, DBOutput, DefaultColumnType};

use crate::slt_test_engine::{expectations, record_query, record_statement};

/// Harness backed by `SqlEngine` that implements sqllogictest's AsyncDB trait.
pub struct DataFusionHarness {
    engine: SqlEngine,
}

impl DataFusionHarness {
    /// Create a new harness with an in-memory pager.
    pub fn new() -> Result<Self, Error> {
        let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
        Self::with_pager(pager)
    }

    /// Create a harness with a custom pager.
    pub fn with_pager(pager: Arc<BoxedPager>) -> Result<Self, Error> {
        let engine = SqlEngine::new(pager)?;
        Ok(Self { engine })
    }

    /// Format Arrow array value as string for sqllogictest.
    fn format_value(
        array: &Arc<dyn ArrowArray>,
        row_idx: usize,
        expected_type: Option<DefaultColumnType>,
    ) -> String {
        if array.is_null(row_idx) {
            return "NULL".to_string();
        }

        match array.data_type() {
            DataType::Int64 => {
                let a = array.as_any().downcast_ref::<Int64Array>().unwrap();
                a.value(row_idx).to_string()
            }
            DataType::Int32 => {
                let a = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .unwrap();
                a.value(row_idx).to_string()
            }
            DataType::UInt64 => {
                let a = array.as_any().downcast_ref::<UInt64Array>().unwrap();
                a.value(row_idx).to_string()
            }
            DataType::Float64 => {
                let a = array.as_any().downcast_ref::<Float64Array>().unwrap();
                let value = a.value(row_idx);
                if matches!(expected_type, Some(DefaultColumnType::Integer)) {
                    let truncated = value.trunc();
                    if truncated.is_finite()
                        && truncated >= i64::MIN as f64
                        && truncated <= i64::MAX as f64
                    {
                        (truncated as i64).to_string()
                    } else {
                        truncated.to_string()
                    }
                } else {
                    value.to_string()
                }
            }
            DataType::Float32 => {
                let a = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float32Array>()
                    .unwrap();
                let value = a.value(row_idx);
                if matches!(expected_type, Some(DefaultColumnType::Integer)) {
                    let truncated = value.trunc();
                    if truncated.is_finite()
                        && truncated >= i64::MIN as f32
                        && truncated <= i64::MAX as f32
                    {
                        (truncated as i64).to_string()
                    } else {
                        truncated.to_string()
                    }
                } else {
                    value.to_string()
                }
            }
            DataType::Utf8 => {
                let a = array.as_any().downcast_ref::<StringArray>().unwrap();
                let text = a.value(row_idx);
                // SQLite-style coercion for expected INTEGER output
                if matches!(expected_type, Some(DefaultColumnType::Integer)) {
                    text.trim().parse::<i64>().unwrap_or(0).to_string()
                } else {
                    text.to_string()
                }
            }
            DataType::Boolean => {
                let a = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                if a.value(row_idx) {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            }
            DataType::Struct(_) => {
                let a = array.as_any().downcast_ref::<StructArray>().unwrap();
                Self::format_struct_value(a, row_idx)
            }
            DataType::Null => "NULL".to_string(),
            _ => "".to_string(),
        }
    }

    /// Format struct value for display.
    fn format_struct_value(struct_array: &StructArray, row_idx: usize) -> String {
        let mut parts = Vec::new();
        let field_names = struct_array.column_names();

        for (field_idx, column) in struct_array.columns().iter().enumerate() {
            let field_name = field_names[field_idx];
            let value_str = Self::format_value(column, row_idx, None);
            parts.push(format!("'{}': {}", field_name, value_str));
        }

        format!("{{{}}}", parts.join(", "))
    }

    /// Infer sqllogictest column type from Arrow DataType.
    fn infer_column_type(dtype: &DataType) -> DefaultColumnType {
        match dtype {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => DefaultColumnType::Integer,
            DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _) => DefaultColumnType::FloatingPoint,
            DataType::Utf8 | DataType::LargeUtf8 => DefaultColumnType::Text,
            _ => DefaultColumnType::Any,
        }
    }
}

impl Default for DataFusionHarness {
    fn default() -> Self {
        Self::new().expect("Failed to create SQL engine harness")
    }
}

#[async_trait::async_trait]
impl AsyncDB for DataFusionHarness {
    type Error = Error;
    type ColumnType = DefaultColumnType;

    async fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        tracing::debug!("[SQL_ENGINE] run() called, sql=\"{}\"", sql.trim());
        let start = Instant::now();

        let mut results = self.engine.execute(sql).await?;
        if results.is_empty() {
            let duration = start.elapsed();
            record_statement(sql, duration, "STATEMENT");
            return Ok(DBOutput::StatementComplete(0));
        }
        if results.len() > 1 {
            return Err(Error::Internal(format!(
                "multi-statement inputs are not supported ({} statements)",
                results.len()
            )));
        }

        let in_query_context = expectations::is_set();
        let mut expected_types = expectations::take();

        match results.pop().unwrap() {
            SqlStatementResult::Query { batches } => {
                let mut rows: Vec<Vec<String>> = Vec::new();
                for batch in &batches {
                    for row_idx in 0..batch.num_rows() {
                        let mut row: Vec<String> = Vec::new();
                        for col in 0..batch.num_columns() {
                            let array = batch.column(col);
                            let expected_type = expected_types
                                .as_ref()
                                .and_then(|types| types.get(col))
                                .cloned();
                            let val = Self::format_value(array, row_idx, expected_type);
                            row.push(val);
                        }
                        rows.push(row);
                    }
                }

                let types = if let Some(expected) = expected_types.take() {
                    expected
                } else if let Some(first) = batches.first() {
                    (0..first.num_columns())
                        .map(|col| Self::infer_column_type(first.column(col).data_type()))
                        .collect()
                } else {
                    vec![]
                };

                let duration = start.elapsed();
                record_query(sql, duration, "SELECT");
                Ok(DBOutput::Rows { types, rows })
            }
            SqlStatementResult::Statement { rows_affected } => {
                let duration = start.elapsed();
                record_statement(sql, duration, "STATEMENT");

                if in_query_context {
                    let types = expected_types
                        .take()
                        .unwrap_or_else(|| vec![DefaultColumnType::Integer]);
                    Ok(DBOutput::Rows {
                        types,
                        rows: vec![vec![rows_affected.to_string()]],
                    })
                } else {
                    Ok(DBOutput::StatementComplete(rows_affected as u64))
                }
            }
        }
    }

    async fn shutdown(&mut self) {
        // Nothing to do for SqlEngine
    }
}

/// Create a factory that produces SQL engine harnesses.
pub fn make_datafusion_factory() -> impl Fn() -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<DataFusionHarness, ()>> + Send + 'static>,
> + Clone {
    move || Box::pin(async move { DataFusionHarness::new().map_err(|_| ()) })
}
