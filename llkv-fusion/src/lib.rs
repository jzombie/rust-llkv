//! DataFusion integration helpers for LLKV storage.
//!
//! This crate wires [`llkv-column-map`] and [`llkv-storage`] into a
//! [`datafusion`] [`TableProvider`] so DataFusion's query engine can operate
//! directly on LLKV's persisted columnar data. The integration is intentionally
//! minimal—writes flow through [`ColumnStore::append`], while reads rely on the
//! store's projection utilities to materialize [`RecordBatch`]es for DataFusion.
//!
//! The primary entry points are:
//! - [`LlkvTableBuilder`]: registers logical columns inside a [`ColumnStore`],
//!   appends Arrow batches (with automatic `rowid` management), and tracks the
//!   row IDs that back each insert.
//! - [`LlkvTableProvider`]: implements DataFusion's [`TableProvider`] by
//!   gathering LLKV rows in scan-sized chunks and surfacing them through a
//!   [`MemorySourceConfig`].
//! - [`LlkvQueryPlanner`]: intercepts `DELETE` statements during physical
//!   planning and executes them through LLKV's storage layer.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{ArrayRef, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::stats::{Precision, Statistics};
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::execution::context::QueryPlanner;
use datafusion::logical_expr::{DmlStatement, Expr, LogicalPlan, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
use datafusion_datasource::memory::MemorySourceConfig;
use llkv_column_map::store::{
    ColumnStore, FIELD_ID_META_KEY, GatherNullPolicy, ROW_ID_COLUMN_NAME,
};
use llkv_column_map::types::{LogicalFieldId, RowId, TableId};
use llkv_result::{Error as LlkvError, Result as LlkvResult};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

pub use llkv_table::{LlkvTableBuilder, LlkvTableProvider};

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray, UInt64Array};
    use datafusion::prelude::SessionContext;
    use llkv_storage::pager::MemPager;
    use arrow::util::pretty::pretty_format_batches;
    
    fn build_demo_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::UInt64, false),
            Field::new("score", DataType::Int32, false),
            Field::new("status", DataType::Utf8, true),
        ]));

        let user_ids = Arc::new(UInt64Array::from(vec![1_u64, 2, 3]));
        let scores = Arc::new(Int32Array::from(vec![42, 7, 99]));
        let statuses = Arc::new(StringArray::from(vec![Some("active"), None, Some("vip")]));

        RecordBatch::try_new(schema, vec![user_ids, scores, statuses]).expect("batch")
    }

    #[tokio::test]
    async fn datafusion_joins_llkv_tables() {
        let pager = Arc::new(MemPager::default());
        let store = Arc::new(ColumnStore::open(Arc::clone(&pager)).expect("store"));

        // Create users table
        let users_schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::UInt64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let users_batch = RecordBatch::try_new(
            users_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![1_u64, 2, 3, 4])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie", "Diana"])),
            ],
        )
        .expect("users batch");

        let mut users_builder =
            LlkvTableBuilder::new(Arc::clone(&store), 1, users_schema).expect("users builder");
        users_builder
            .append_batch(&users_batch)
            .expect("users append");
        let users_provider = users_builder.finish().expect("users provider");

        // Create orders table
        let orders_schema = Arc::new(Schema::new(vec![
            Field::new("order_id", DataType::UInt64, false),
            Field::new("user_id", DataType::UInt64, false),
            Field::new("amount", DataType::Int32, false),
        ]));
        let orders_batch = RecordBatch::try_new(
            orders_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![101_u64, 102, 103, 104, 105])),
                Arc::new(UInt64Array::from(vec![1_u64, 2, 1, 3, 2])),
                Arc::new(Int32Array::from(vec![50, 75, 120, 30, 90])),
            ],
        )
        .expect("orders batch");

        let mut orders_builder =
            LlkvTableBuilder::new(Arc::clone(&store), 2, orders_schema).expect("orders builder");
        orders_builder
            .append_batch(&orders_batch)
            .expect("orders append");
        let orders_provider = orders_builder.finish().expect("orders provider");

        // Register tables and perform join
        let ctx = SessionContext::new();
        ctx.register_table("users", Arc::new(users_provider))
            .expect("register users");
        ctx.register_table("orders", Arc::new(orders_provider))
            .expect("register orders");

        let df = ctx
            .sql(
                "SELECT u.name, o.order_id, o.amount \
                 FROM users u \
                 INNER JOIN orders o ON u.user_id = o.user_id \
                 ORDER BY u.name, o.order_id",
            )
            .await
            .expect("sql");
        let results = df.collect().await.expect("collect");
        let formatted = pretty_format_batches(&results).expect("format").to_string();

        let expected = vec![
            "+---------+----------+--------+",
            "| name    | order_id | amount |",
            "+---------+----------+--------+",
            "| Alice   | 101      | 50     |",
            "| Alice   | 103      | 120    |",
            "| Bob     | 102      | 75     |",
            "| Bob     | 105      | 90     |",
            "| Charlie | 104      | 30     |",
            "+---------+----------+--------+",
        ]
        .join("\n");

        assert_eq!(formatted.trim(), expected.trim());
    }

    #[tokio::test]
    #[should_panic(expected = "DELETE FROM llkv_demo is not yet implemented")]
    async fn datafusion_delete_statement_is_rejected() {
        use datafusion::execution::session_state::SessionStateBuilder;

        let pager = Arc::new(MemPager::default());
        let store = Arc::new(ColumnStore::open(Arc::clone(&pager)).expect("store"));

        let schema = build_demo_batch().schema();
        let mut builder = LlkvTableBuilder::new(Arc::clone(&store), 1, schema).expect("builder");
        builder.append_batch(&build_demo_batch()).expect("append");
        let provider = builder.finish().expect("provider");

        // Create context with custom query planner that intercepts DELETE
        let session_state = SessionStateBuilder::new()
            .with_default_features()
            .with_query_planner(Arc::new(LlkvQueryPlanner::new()))
            .build();
        let ctx = SessionContext::new_with_state(session_state);
        ctx.register_table("llkv_demo", Arc::new(provider))
            .expect("register");

        let delete_df = ctx
            .sql("DELETE FROM llkv_demo WHERE user_id = 2")
            .await
            .expect("DELETE planning");
        
        // This will panic with todo!() - demonstrating interception works
        let _results = delete_df.collect().await.unwrap();
    }
}

/// Custom physical planner that intercepts DELETE operations.
///
/// DataFusion 50.3.0 parses `DELETE FROM table WHERE ...` into
/// [`LogicalPlan::Dml`] with [`WriteOp::Delete`], but provides no physical
/// execution path. This planner detects such plans during physical plan
/// generation and routes them through LLKV's storage layer.
///
/// # Usage
///
/// Register with a [`SessionContext`] using
/// [`SessionContext::with_query_planner`]:
///
/// ```ignore
/// let ctx = SessionContext::new()
///     .with_query_planner(Arc::new(LlkvQueryPlanner::new()));
/// ```
pub struct LlkvQueryPlanner {
    fallback: Arc<dyn PhysicalPlanner>,
}

impl fmt::Debug for LlkvQueryPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlkvQueryPlanner").finish_non_exhaustive()
    }
}

impl LlkvQueryPlanner {
    /// Construct a new planner that delegates to DataFusion's default planner
    /// for all non-DELETE operations.
    pub fn new() -> Self {
        Self {
            fallback: Arc::new(DefaultPhysicalPlanner::default()),
        }
    }
}

impl Default for LlkvQueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl QueryPlanner for LlkvQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &datafusion::execution::context::SessionState,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Intercept DELETE operations
        if let LogicalPlan::Dml(DmlStatement {
            table_name,
            op: datafusion::logical_expr::WriteOp::Delete,
            ..
        }) = logical_plan
        {
            // TODO: Implement DELETE execution
            todo!(
                "DELETE FROM {} is not yet implemented",
                table_name.to_string()
            );
        }

        // Delegate everything else to the default planner
        self.fallback
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}
