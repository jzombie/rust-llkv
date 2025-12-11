use crate::physical_plan::{BatchIter, PhysicalPlan};
use arrow::compute::cast;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use llkv_expr::Expr;
use llkv_result::{Error, Result};
use llkv_scan::{RowIdFilter, RowStream, ScanProjection, ScanStorage, ScanStreamOptions};
use llkv_storage::pager::Pager;
use llkv_table::table::Table;
use llkv_types::FieldId;
use simd_r_drive_entry_handle::EntryHandle;
use std::any::Any;
use std::fmt;
use std::sync::Arc;

pub struct ScanExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub table: Arc<Table<P>>,
    pub schema: SchemaRef,
    pub projections: Arc<[ScanProjection]>,
    pub filter: Option<Expr<'static, FieldId>>,
    pub limit: Option<usize>,
    pub row_filter: Option<Arc<dyn RowIdFilter<P>>>,
}

impl<P> fmt::Debug for ScanExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanExec")
            .field("schema", &self.schema)
            .field("projections", &self.projections)
            .field("filter", &self.filter)
            .field("limit", &self.limit)
            .field("row_filter", &self.row_filter.is_some())
            .finish()
    }
}

impl<P> PhysicalPlan<P> for ScanExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + std::fmt::Debug,
{
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn execute(&self) -> Result<BatchIter> {
        let filter = self.filter.as_ref().unwrap_or(&Expr::Literal(true));
        let mut options = ScanStreamOptions::default();
        options.include_nulls = true;
        options.row_id_filter = self.row_filter.clone();

        let mut stream = llkv_scan::execute::prepare_scan_stream(
            self.table.clone(),
            self.table.table_id(),
            self.projections.as_ref(),
            filter,
            options,
        )?;

        let schema = self.schema.clone();
        let iter = std::iter::from_fn(move || {
            match stream.next_chunk() {
                Ok(Some(chunk)) => {
                    let batch = chunk.to_record_batch();
                    // Ensure batch matches ScanExec schema (handle storage/schema drift)
                    // We check field types because schema metadata might differ but types match
                    let types_match = batch.schema().fields().len() == schema.fields().len()
                        && batch
                            .schema()
                            .fields()
                            .iter()
                            .zip(schema.fields())
                            .all(|(a, b)| a.data_type() == b.data_type());

                    if !types_match {
                        let columns = batch
                            .columns()
                            .iter()
                            .zip(schema.fields())
                            .map(|(col, field)| {
                                if col.data_type() != field.data_type() {
                                    cast(col, field.data_type())
                                } else {
                                    Ok(col.clone())
                                }
                            })
                            .collect::<std::result::Result<Vec<_>, _>>();

                        match columns {
                            Ok(cols) => {
                                if cols.is_empty() {
                                    eprintln!(
                                        "DEBUG: ScanExec cols is empty! Schema fields: {}, Batch cols: {}",
                                        schema.fields().len(),
                                        batch.columns().len()
                                    );
                                }
                                Some(
                                    RecordBatch::try_new(schema.clone(), cols)
                                        .map_err(|e| Error::Internal(e.to_string())),
                                )
                            }
                            Err(e) => Some(Err(Error::Internal(format!(
                                "Failed to cast scan columns: {}",
                                e
                            )))),
                        }
                    } else {
                        Some(Ok(batch))
                    }
                }
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            }
        });

        Ok(Box::new(iter))
    }

    fn children(&self) -> Arc<[Arc<dyn PhysicalPlan<P>>]> {
        Arc::from([])
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Arc<[Arc<dyn PhysicalPlan<P>>]>,
    ) -> Result<Arc<dyn PhysicalPlan<P>>> {
        if !children.is_empty() {
            return Err(llkv_result::Error::Internal(
                "ScanExec expects no children".into(),
            ));
        }
        Ok(Arc::new(ScanExec {
            table: self.table.clone(),
            schema: self.schema.clone(),
            projections: self.projections.clone(),
            filter: self.filter.clone(),
            limit: self.limit,
            row_filter: self.row_filter.clone(),
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
