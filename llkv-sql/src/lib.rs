//! Minimal SQL front-end that routes statements through llkv-fusion.
//!
//! This engine parses SQL statements so it can intercept `CREATE TABLE`
//! definitions, materialize them inside LLKV's [`TableCatalog`], and register
//! the resulting [`LlkvTableProvider`]s with DataFusion. All other statements
//! are delegated to a `SessionContext` that uses [`LlkvQueryPlanner`] for scan
//! planning, keeping LLKV storage as the backing store.

use std::collections::HashMap;
// use std::sync::Arc; // Removed duplicate import

use arrow::record_batch::RecordBatch;
use datafusion::error::DataFusionError;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use llkv_column_map::store::ColumnStore;
use llkv_parquet_store::ParquetStore;
use llkv_plan::LlkvQueryPlanner;
use llkv_result::{Error, Result};
use llkv_storage::pager::BoxedPager;
use llkv_table::catalog::{LlkvSchemaProvider, TableCatalog};
use llkv_table::providers::column_map::ColumnStoreBackend;
use llkv_table::providers::parquet::ParquetStoreBackend;
use llkv_table::traits::CatalogBackend;

use datafusion::logical_expr::{DdlStatement, LogicalPlan};
use datafusion::sql::parser::{DFParser, Statement as DFStatement};
use llkv_plan::LlkvCreateTable;
use sqlparser::ast::{
    CreateTableOptions, Expr, SqlOption, Statement as SQLStatement, Value, ValueWithSpan,
};
use std::sync::Arc;

/// Result of executing a SQL statement.
#[derive(Debug)]
pub enum SqlStatementResult {
    /// A query that produced Arrow record batches.
    Query { batches: Vec<RecordBatch> },
    /// A statement that did not return rows (affects rows_affected if known).
    Statement { rows_affected: usize },
}

/// SQL engine that intercepts DDL before delegating to DataFusion.
pub struct SqlEngine {
    ctx: SessionContext,
    catalog: Arc<TableCatalog>,
}

impl SqlEngine {
    /// Create a new SQL engine backed by the provided pager.
    pub fn new(pager: Arc<BoxedPager>) -> Result<Self> {
        let store = Arc::new(ColumnStore::open(Arc::clone(&pager))?);
        let metadata_backend = Box::new(ColumnStoreBackend::new(Arc::clone(&store)));

        let parquet_store = Arc::new(ParquetStore::open(Arc::clone(&pager))?);
        let parquet_backend = Box::new(ParquetStoreBackend::new(parquet_store));

        let mut data_backends: HashMap<String, Box<dyn CatalogBackend>> = HashMap::new();
        data_backends.insert(
            "columnstore".to_string(),
            Box::new(ColumnStoreBackend::new(Arc::clone(&store))),
        );
        data_backends.insert("parquet".to_string(), parquet_backend);

        let catalog = TableCatalog::new(metadata_backend, data_backends, "parquet".to_string())?;

        let session_state = SessionStateBuilder::new()
            .with_default_features()
            .with_query_planner(Arc::new(LlkvQueryPlanner::new(Arc::clone(&catalog))))
            .build();

        let ctx = SessionContext::new_with_state(session_state);

        // Register catalog as "public" schema
        let schema_provider = Arc::new(LlkvSchemaProvider::new(Arc::clone(&catalog)));

        if let Some(default_catalog) = ctx.catalog("datafusion") {
            default_catalog
                .register_schema("public", schema_provider)
                .map_err(|e| Error::Internal(format!("failed to register public schema: {e}")))?;
        } else {
            return Err(Error::Internal(
                "default catalog 'datafusion' not found".to_string(),
            ));
        }

        let engine = Self { ctx, catalog };
        Ok(engine)
    }

    /// Get the underlying table catalog
    pub fn catalog(&self) -> &Arc<TableCatalog> {
        &self.catalog
    }

    /// Execute raw SQL text.
    pub async fn execute(&self, sql: &str) -> Result<Vec<SqlStatementResult>> {
        let statements = DFParser::parse_sql(sql).map_err(|e| {
            map_datafusion_error("Parser error", DataFusionError::Execution(e.to_string()))
        })?;

        let mut results = Vec::new();

        for mut statement in statements {
            let mut backend_option = None;

            if let DFStatement::Statement(inner_stmt) = &mut statement {
                if let SQLStatement::CreateTable(create_table) = &mut **inner_stmt {
                    let options_vec = match &mut create_table.table_options {
                        CreateTableOptions::With(opts) => Some(opts),
                        CreateTableOptions::Options(opts) => Some(opts),
                        CreateTableOptions::TableProperties(opts) => Some(opts),
                        CreateTableOptions::Plain(opts) => Some(opts),
                        CreateTableOptions::None => None,
                    };

                    if let Some(options) = options_vec {
                        if let Some(idx) = options.iter().position(|o| {
                            if let SqlOption::KeyValue { key, .. } = o {
                                key.value.eq_ignore_ascii_case("backend")
                            } else {
                                false
                            }
                        }) {
                            if let SqlOption::KeyValue { value, .. } = &options[idx] {
                                match value {
                                    Expr::Value(ValueWithSpan {
                                        value: Value::SingleQuotedString(s),
                                        ..
                                    })
                                    | Expr::Value(ValueWithSpan {
                                        value: Value::DoubleQuotedString(s),
                                        ..
                                    }) => {
                                        backend_option = Some(s.clone());
                                    }
                                    _ => {}
                                }
                            }
                            options.remove(idx);
                            if options.is_empty() {
                                create_table.table_options = CreateTableOptions::None;
                            }
                        }
                    }
                }
            }

            // Convert to LogicalPlan
            let plan = self
                .ctx
                .state()
                .statement_to_plan(statement)
                .await
                .map_err(|e| map_datafusion_error("Planning failed", e))?;

            let final_plan = if let Some(backend) = backend_option {
                if let LogicalPlan::Ddl(DdlStatement::CreateMemoryTable(cmd)) = plan {
                    // Wrap in LlkvCreateTable
                    let schema = cmd.input.schema().clone();
                    LogicalPlan::Extension(datafusion::logical_expr::Extension {
                        node: Arc::new(LlkvCreateTable {
                            name: cmd.name.table().to_string(),
                            schema,
                            input: Some((*cmd.input).clone()),
                            if_not_exists: cmd.if_not_exists,
                            backend: Some(backend),
                        }),
                    })
                } else {
                    plan
                }
            } else {
                plan
            };

            let df = self
                .ctx
                .execute_logical_plan(final_plan)
                .await
                .map_err(|e| map_datafusion_error("Execution failed", e))?;

            let schema_is_empty = df.schema().fields().is_empty();

            let batches = df
                .collect()
                .await
                .map_err(|e| map_datafusion_error("DataFusion execution failed", e))?;

            if schema_is_empty {
                results.push(SqlStatementResult::Statement { rows_affected: 0 });
            } else {
                results.push(SqlStatementResult::Query { batches });
            }
        }

        Ok(results)
    }
}

fn map_datafusion_error(context: &str, err: DataFusionError) -> Error {
    Error::Internal(format!("{context}: {err}"))
}
