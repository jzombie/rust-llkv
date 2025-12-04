use std::any::Any;
use std::sync::Arc;
use std::fmt;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use crate::physical::PhysicalPlan;
use crate::physical::table::ExecutionTable;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use llkv_scan::{ScanProjection, ScanStreamOptions, RowIdFilter, ScanOrderSpec};
use llkv_column_map::store::scan::ranges::IntRanges;
use llkv_expr::Expr;
use llkv_types::LogicalFieldId;
use llkv_types::FieldId;

pub struct ScanExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub table_name: String,
    pub schema: SchemaRef,
    pub table: Arc<dyn ExecutionTable<P>>,
    pub ranges: Option<IntRanges>,
    pub driving_column: Option<LogicalFieldId>,
    pub filter: Option<Expr<'static, FieldId>>,
    pub projections: Vec<ScanProjection>,
    pub row_filter: Option<Arc<dyn RowIdFilter<P>>>,
    pub order: Option<ScanOrderSpec>,
}

impl<P> fmt::Debug for ScanExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanExec")
            .field("table_name", &self.table_name)
            .field("schema", &self.schema)
            .field("ranges", &self.ranges)
            .finish()
    }
}

impl<P> ScanExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn new(
        table_name: String,
        schema: SchemaRef,
        table: Arc<dyn ExecutionTable<P>>,
        ranges: Option<IntRanges>,
        driving_column: Option<LogicalFieldId>,
        filter: Option<Expr<'static, FieldId>>,
        projections: Vec<ScanProjection>,
        row_filter: Option<Arc<dyn RowIdFilter<P>>>,
        order: Option<ScanOrderSpec>,
    ) -> Self {
        Self {
            table_name,
            schema,
            table,
            ranges,
            driving_column,
            filter,
            projections,
            row_filter,
            order,
        }
    }
}

impl<P> PhysicalPlan for ScanExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn execute(&self) -> Result<Box<dyn Iterator<Item = Result<RecordBatch, String>> + Send>, String> {
        let mut batches = Vec::new();
        
        // Use the provided filter or default to true
        let filter_expr = self.filter.clone().unwrap_or(Expr::Literal(true));
        
        let options = ScanStreamOptions {
            ranges: self.ranges.clone(),
            driving_column: self.driving_column,
            row_id_filter: self.row_filter.clone(),
            order: self.order,
            include_nulls: true,
            ..ScanStreamOptions::default()
        };

        self.table.scan_stream(
            &self.projections,
            &filter_expr,
            options,
            &mut |batch| batches.push(Ok(batch))
        )?;

        Ok(Box::new(batches.into_iter()))
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalPlan>> {
        vec![]
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn PhysicalPlan>>,
    ) -> Result<Arc<dyn PhysicalPlan>, String> {
        if !children.is_empty() {
            return Err("ScanExec expects no children".to_string());
        }
        Ok(Arc::new(ScanExec {
            table_name: self.table_name.clone(),
            schema: self.schema.clone(),
            table: self.table.clone(),
            ranges: self.ranges.clone(),
            driving_column: self.driving_column,
            filter: self.filter.clone(),
            projections: self.projections.clone(),
            row_filter: self.row_filter.clone(),
            order: self.order,
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
