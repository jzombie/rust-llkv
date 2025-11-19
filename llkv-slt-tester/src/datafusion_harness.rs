//! DataFusion-based test harness for sqllogictest integration.
//!
//! This module replaces the legacy `llkv-sql` engine with DataFusion, using
//! `llkv-fusion` to bridge LLKV's columnar storage into DataFusion's query engine.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    Array as ArrowArray, BooleanArray, Float64Array, Int64Array, StringArray, StructArray,
    UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::parser::Parser;
use llkv_result::Error;
use llkv_storage::pager::{BoxedPager, MemPager};
use llkv_table::catalog::TableCatalog;
use sqllogictest::{AsyncDB, DBOutput, DefaultColumnType};

use crate::slt_test_engine::{expectations, record_query, record_statement};

/// DataFusion-backed harness that implements sqllogictest's AsyncDB trait.
pub struct DataFusionHarness {
    ctx: SessionContext,
    catalog: Arc<TableCatalog<BoxedPager>>,
    next_table_id: u16,
}

impl DataFusionHarness {
    /// Create a new harness with an in-memory pager.
    pub fn new() -> Result<Self, Error> {
        let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
        Self::with_pager(pager)
    }

    /// Create a harness with a custom pager.
    pub fn with_pager(pager: Arc<BoxedPager>) -> Result<Self, Error> {
        let catalog = Arc::new(TableCatalog::open(pager)?);
        let ctx = SessionContext::new();
        Ok(Self {
            ctx,
            catalog,
            next_table_id: 1,
        })
    }

    /// Parse SQL to determine statement type.
    fn parse_statement_type(sql: &str) -> Option<String> {
        let dialect = GenericDialect {};
        Parser::parse_sql(&dialect, sql)
            .ok()
            .and_then(|stmts| stmts.first().map(|s| format!("{:?}", s).split('(').next().unwrap_or("").to_uppercase()))
    }

    /// Handle CREATE TABLE statements by parsing schema and registering with catalog.
    async fn handle_create_table(&mut self, sql: &str) -> Result<DBOutput<DefaultColumnType>, Error> {
        // Parse CREATE TABLE statement
        let dialect = GenericDialect {};
        let stmts = Parser::parse_sql(&dialect, sql)
            .map_err(|e| Error::Internal(format!("Failed to parse CREATE TABLE: {}", e)))?;
        
        let _stmt = stmts.first()
            .ok_or_else(|| Error::Internal("No statement found".to_string()))?;

        // Extract table name and columns (simplified parsing)
        let table_name = self.extract_table_name(sql)?;
        let schema = self.extract_schema_from_create_table(sql)?;

        // Register table in catalog
        let _table_id = self.next_table_id;
        self.next_table_id += 1;

        let builder = self.catalog.create_table(&table_name, Arc::new(schema))?;
        let provider = builder.finish()?;

        // Register with DataFusion
        self.ctx
            .register_table(&table_name, Arc::new(provider))
            .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

        Ok(DBOutput::StatementComplete(0))
    }

    /// Extract table name from CREATE TABLE statement (simple parsing).
    fn extract_table_name(&self, sql: &str) -> Result<String, Error> {
        let sql = sql.to_lowercase();
        let parts: Vec<&str> = sql.split_whitespace().collect();
        
        if let Some(pos) = parts.iter().position(|&p| p == "table") {
            if let Some(&name) = parts.get(pos + 1) {
                return Ok(name.trim_matches(|c| c == '(' || c == ')' || c == ';').to_string());
            }
        }
        
        Err(Error::Internal("Could not extract table name".to_string()))
    }

    /// Extract Arrow schema from CREATE TABLE statement (simplified).
    fn extract_schema_from_create_table(&self, sql: &str) -> Result<Schema, Error> {
        // Very basic parsing - you'll want to improve this
        let sql = sql.to_lowercase();
        
        // Extract column definitions between parentheses
        let start = sql.find('(').ok_or_else(|| Error::Internal("No column definitions found".to_string()))?;
        let end = sql.rfind(')').ok_or_else(|| Error::Internal("Unclosed column definitions".to_string()))?;
        let cols_str = &sql[start + 1..end];

        let mut fields = Vec::new();
        for col_def in cols_str.split(',') {
            let parts: Vec<&str> = col_def.trim().split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            let name = parts[0].to_string();
            let type_str = parts[1];

            let data_type = match type_str {
                "int" | "integer" | "bigint" => DataType::Int64,
                "float" | "real" | "double" => DataType::Float64,
                "text" | "varchar" | "string" => DataType::Utf8,
                "boolean" | "bool" => DataType::Boolean,
                _ => DataType::Utf8, // default fallback
            };

            fields.push(Field::new(name, data_type, true));
        }

        if fields.is_empty() {
            return Err(Error::Internal("No columns found in CREATE TABLE".to_string()));
        }

        Ok(Schema::new(fields))
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
                let a = array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
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
            DataType::Float16 | DataType::Float32 | DataType::Float64 | DataType::Decimal128(_, _) | DataType::Decimal256(_, _) => {
                DefaultColumnType::FloatingPoint
            }
            DataType::Utf8 | DataType::LargeUtf8 => DefaultColumnType::Text,
            _ => DefaultColumnType::Any,
        }
    }
}

impl Default for DataFusionHarness {
    fn default() -> Self {
        Self::new().expect("Failed to create DataFusionHarness")
    }
}

#[async_trait::async_trait]
impl AsyncDB for DataFusionHarness {
    type Error = Error;
    type ColumnType = DefaultColumnType;

    async fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        tracing::debug!("[DATAFUSION] run() called, sql=\"{}\"", sql.trim());
        let start = Instant::now();

        // Handle CREATE TABLE separately
        if sql.trim().to_lowercase().starts_with("create table") {
            return self.handle_create_table(sql).await;
        }

        // Execute via DataFusion
        match self.ctx.sql(sql).await {
            Ok(df) => {
                let batches = df
                    .collect()
                    .await
                    .map_err(|e| Error::Internal(format!("DataFusion execution failed: {}", e)))?;

                let stmt_type = Self::parse_statement_type(sql);
                let in_query_context = expectations::is_set();
                let mut expected_types = expectations::take();

                // Handle SELECT queries
                if !batches.is_empty() && batches[0].num_columns() > 0 {
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
                } else {
                    // DML statement (INSERT/UPDATE/DELETE)
                    let duration = start.elapsed();
                    let stmt_desc = stmt_type.as_deref().unwrap_or("STATEMENT");
                    record_statement(sql, duration, stmt_desc);

                    if in_query_context {
                        // Return row count as a result set
                        let types = expected_types.take().unwrap_or_else(|| vec![DefaultColumnType::Integer]);
                        Ok(DBOutput::Rows {
                            types,
                            rows: vec![vec!["0".to_string()]], // TODO: extract actual row count
                        })
                    } else {
                        Ok(DBOutput::StatementComplete(0))
                    }
                }
            }
            Err(e) => {
                let duration = start.elapsed();
                record_statement(sql, duration, "ERROR");
                Err(Error::Internal(format!("DataFusion error: {}", e)))
            }
        }
    }

    async fn shutdown(&mut self) {
        // Nothing to do for DataFusion
    }
}

/// Create a factory that produces DataFusion harnesses.
pub fn make_datafusion_factory() -> impl Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<DataFusionHarness, ()>> + Send + 'static>> + Clone {
    move || {
        Box::pin(async move {
            DataFusionHarness::new().map_err(|_| ())
        })
    }
}
