//! DataFusion integration helpers for LLKV Column Map storage.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::physical_plan::ExecutionPlan;

use llkv_column_map::store::ColumnStore;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

/// Custom [`TableProvider`] that surfaces LLKV Column Map data to DataFusion.
pub struct ColumnMapTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    store: Arc<ColumnStore<P>>,
    schema: SchemaRef,
}

impl<P> fmt::Debug for ColumnMapTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ColumnMapTableProvider")
            .field("schema", &self.schema)
            .finish()
    }
}

impl<P> ColumnMapTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(store: Arc<ColumnStore<P>>, schema: SchemaRef) -> Self {
        Self { store, schema }
    }
}

#[async_trait]
impl<P> TableProvider for ColumnMapTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Err(DataFusionError::NotImplemented(
            "ColumnMap scan not implemented yet".to_string(),
        ))
    }

    async fn insert_into(
        &self,
        _state: &dyn Session,
        _input: Arc<dyn ExecutionPlan>,
        _insert_op: InsertOp,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Err(DataFusionError::NotImplemented(
            "ColumnMap insert not implemented yet".to_string(),
        ))
    }
}
