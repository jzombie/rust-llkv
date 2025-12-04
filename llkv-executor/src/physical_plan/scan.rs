use std::any::Any;
use std::sync::Arc;
use std::fmt;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use llkv_plan::physical::PhysicalPlan;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use crate::types::ExecutorTable;
use llkv_table::table::{ScanStreamOptions, ScanProjection};
use llkv_column_map::store::scan::ranges::IntRanges;
use llkv_expr::Expr;
use llkv_types::LogicalFieldId;
use llkv_types::FieldId;
use llkv_column_map::store::Projection;

pub struct ScanExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub table_name: String,
    pub schema: SchemaRef,
    pub table: Arc<ExecutorTable<P>>,
    pub ranges: Option<IntRanges>,
    pub driving_column: Option<LogicalFieldId>,
    pub filter: Option<Expr<'static, FieldId>>,
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
        table: Arc<ExecutorTable<P>>,
        ranges: Option<IntRanges>,
        driving_column: Option<LogicalFieldId>,
        filter: Option<Expr<'static, FieldId>>,
    ) -> Self {
        Self {
            table_name,
            schema,
            table,
            ranges,
            driving_column,
            filter,
        }
    }
}

impl<P> PhysicalPlan for ScanExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn execute(&self) -> Result<Box<dyn Iterator<Item = Result<RecordBatch, String>> + Send>, String> {
        let mut batches = Vec::new();
        // Default to scanning all columns in the schema
        let projections: Vec<ScanProjection> = self.schema.fields().iter()
            .map(|f| {
                let col = self.table.schema.resolve(f.name())
                    .ok_or_else(|| format!("Column not found in table schema: {}", f.name()))?;
                let logical_field_id = LogicalFieldId::for_user(self.table.table_id(), col.field_id);
                Ok(ScanProjection::Column(Projection::new(logical_field_id)))
            })
            .collect::<Result<Vec<_>, String>>()?;
        
        // Use the provided filter or default to true
        let filter_expr = self.filter.clone().unwrap_or(Expr::Literal(true));
        
        let options = ScanStreamOptions {
            ranges: self.ranges.clone(),
            driving_column: self.driving_column,
            ..ScanStreamOptions::default()
        };

        self.table.table.scan_stream(
            &projections,
            &filter_expr,
            options,
            |batch| batches.push(Ok(batch))
        ).map_err(|e| e.to_string())?;

        Ok(Box::new(batches.into_iter()))
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalPlan>> {
        vec![]
    }

    fn with_new_children(&self, children: Vec<Arc<dyn PhysicalPlan>>) -> Result<Arc<dyn PhysicalPlan>, String> {
        if !children.is_empty() {
            return Err("ScanExec expects no children".to_string());
        }
        Ok(Arc::new(ScanExec {
            table_name: self.table_name.clone(),
            schema: Arc::clone(&self.schema),
            table: Arc::clone(&self.table),
            ranges: self.ranges.clone(),
            driving_column: self.driving_column,
            filter: self.filter.clone(),
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
