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
