//! Minimal SQL front-end that routes statements through llkv-fusion.
//!
//! This engine parses SQL statements so it can intercept `CREATE TABLE`
//! definitions, materialize them inside LLKV's [`TableCatalog`], and register
//! the resulting [`LlkvTableProvider`]s with DataFusion. All other statements
//! are delegated to a `SessionContext` that uses [`LlkvQueryPlanner`] for scan
//! planning, keeping LLKV storage as the backing store.

use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use datafusion::error::DataFusionError;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use llkv_column_map::store::ColumnStore;
use llkv_plan::LlkvQueryPlanner;
use llkv_result::{Error, Result};
use llkv_storage::pager::BoxedPager;
use llkv_table::catalog::{LlkvSchemaProvider, TableCatalog};
use llkv_table::providers::column_map::ColumnStoreBackend;

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
        let store = Arc::new(ColumnStore::open(pager)?);
        let backend = Box::new(ColumnStoreBackend::new(store));
        let catalog = TableCatalog::new(backend)?;

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
        let result = self.execute_via_datafusion(sql).await?;
        Ok(vec![result])
    }

    async fn execute_via_datafusion(&self, sql: &str) -> Result<SqlStatementResult> {
        let df = self
            .ctx
            .sql(sql)
            .await
            .map_err(|e| map_datafusion_error("DataFusion planning failed", e))?;
        let schema_is_empty = df.schema().fields().is_empty();

        let batches = df
            .collect()
            .await
            .map_err(|e| map_datafusion_error("DataFusion execution failed", e))?;

        if schema_is_empty {
            Ok(SqlStatementResult::Statement { rows_affected: 0 })
        } else {
            Ok(SqlStatementResult::Query { batches })
        }
    }
}

fn map_datafusion_error(context: &str, err: DataFusionError) -> Error {
    Error::Internal(format!("{context}: {err}"))
}
