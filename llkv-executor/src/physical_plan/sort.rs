use crate::physical_plan::{BatchIter, PhysicalPlan};
use arrow::array::{Array, ArrayRef, Int64Builder, StringArray};
use arrow::compute::{SortColumn, SortOptions, lexsort_to_indices};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use llkv_plan::plans::{OrderByPlan, OrderSortType, OrderTarget};
use llkv_result::Result;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use std::any::Any;
use std::fmt;
use std::sync::Arc;

pub struct SortExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub input: Arc<dyn PhysicalPlan<P>>,
    // TODO: Back vector w/ Arc?
    pub order_by: Vec<OrderByPlan>,
}

impl<P> SortExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn new(input: Arc<dyn PhysicalPlan<P>>, order_by: Vec<OrderByPlan>) -> Self {
        Self { input, order_by }
    }
}

impl<P> fmt::Debug for SortExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SortExec")
            .field("order_by", &self.order_by)
            .finish()
    }
}

impl<P> PhysicalPlan<P> for SortExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn execute(&self) -> Result<BatchIter> {
        // SortExec needs to collect all batches to sort globally
        // This is a blocking operation
        let input_stream = self.input.execute()?;
        let mut batches = Vec::new();
        for batch_result in input_stream {
            batches.push(batch_result?);
        }

        if batches.is_empty() {
            return Ok(Box::new(std::iter::empty()));
        }

        let schema = self.schema();
        let combined_batch = arrow::compute::concat_batches(&schema, &batches)
            .map_err(|e| llkv_result::Error::Internal(e.to_string()))?;

        let mut sort_columns = Vec::new();
        for order in &self.order_by {
            let column_index = match &order.target {
                OrderTarget::Column(name) => {
                    // Try to find by name, case-insensitive
                    if let Ok(idx) = schema.index_of(name) {
                        idx
                    } else {
                        // Try case-insensitive search
                        schema
                            .fields()
                            .iter()
                            .position(|f| f.name().eq_ignore_ascii_case(name))
                            .ok_or_else(|| {
                                llkv_result::Error::Internal(format!(
                                    "Column {} not found in schema",
                                    name
                                ))
                            })?
                    }
                }
                OrderTarget::Index(idx) => *idx,
                OrderTarget::All => {
                    return Err(llkv_result::Error::Internal(
                        "OrderTarget::All not supported in SortExec".to_string(),
                    ));
                }
            };

            let source_array = combined_batch.column(column_index);

            let values: ArrayRef = match order.sort_type {
                OrderSortType::Native => source_array.clone(),
                OrderSortType::CastTextToInteger => {
                    let strings = source_array
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            llkv_result::Error::Internal(
                                "ORDER BY CAST expects the underlying column to be TEXT"
                                    .to_string(),
                            )
                        })?;
                    let mut builder = Int64Builder::with_capacity(strings.len());
                    for i in 0..strings.len() {
                        if strings.is_null(i) {
                            builder.append_null();
                        } else {
                            match strings.value(i).parse::<i64>() {
                                Ok(value) => builder.append_value(value),
                                Err(_) => builder.append_null(),
                            }
                        }
                    }
                    Arc::new(builder.finish()) as ArrayRef
                }
            };

            sort_columns.push(SortColumn {
                values,
                options: Some(SortOptions {
                    descending: !order.ascending,
                    nulls_first: order.nulls_first,
                }),
            });
        }

        let indices = lexsort_to_indices(&sort_columns, None)
            .map_err(|e| llkv_result::Error::Internal(e.to_string()))?;

        let columns = combined_batch
            .columns()
            .iter()
            .map(|c| {
                arrow::compute::take(c, &indices, None)
                    .map_err(|e| llkv_result::Error::Internal(e.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;

        let sorted_batch = RecordBatch::try_new(schema, columns)
            .map_err(|e| llkv_result::Error::Internal(e.to_string()))?;

        Ok(Box::new(std::iter::once(Ok(sorted_batch))))
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalPlan<P>>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalPlan<P>>>,
    ) -> Result<Arc<dyn PhysicalPlan<P>>> {
        if children.len() != 1 {
            return Err(llkv_result::Error::Internal(
                "SortExec expects exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(SortExec::new(
            children[0].clone(),
            self.order_by.clone(),
        )))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
