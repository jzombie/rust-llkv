//! Example demonstrating DELETE statement support via custom query planner.
//!
//! DataFusion 50.3.0 doesn't provide built-in DELETE execution. This example
//! shows how `LlkvQueryPlanner` intercepts DELETE statements during physical
//! planning, allowing custom implementation.

use arrow::array::{Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use llkv_column_map::store::ColumnStore;
use llkv_fusion::{LlkvQueryPlanner, LlkvTableBuilder};
use llkv_storage::pager::MemPager;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up LLKV storage
    let pager = Arc::new(MemPager::default());
    let store = Arc::new(ColumnStore::open(pager)?);

    // Create schema and insert data
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let mut builder = LlkvTableBuilder::new(Arc::clone(&store), 1, schema.clone())?;

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![10, 20, 30])),
        ],
    )?;

    builder.append_batch(&batch)?;
    let provider = builder.finish()?;

    // Create DataFusion context with custom query planner
    let session_state = SessionStateBuilder::new()
        .with_default_features()
        .with_query_planner(Arc::new(LlkvQueryPlanner::new()))
        .build();

    let ctx = SessionContext::new_with_state(session_state);
    ctx.register_table("demo", Arc::new(provider))?;

    // Query works normally
    println!("Before DELETE:");
    let df = ctx.sql("SELECT * FROM demo ORDER BY id").await?;
    df.show().await?;

    // DELETE statement is intercepted by LlkvQueryPlanner
    // Currently stubbed with todo!() - will panic
    println!("\nAttempting DELETE (will panic with todo!):");
    let result = ctx
        .sql("DELETE FROM demo WHERE id = 2")
        .await?
        .collect()
        .await;

    match result {
        Ok(_) => println!("DELETE succeeded"),
        Err(e) => println!("DELETE failed: {}", e),
    }

    Ok(())
}
