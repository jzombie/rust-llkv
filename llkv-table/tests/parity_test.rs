use arrow::array::{Int32Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use llkv_column_map::store::{ColumnStore, FIELD_ID_META_KEY, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::LogicalFieldId;
use llkv_parquet_store::{ParquetStore, TableId};
use llkv_storage::pager::MemPager;
use llkv_table::providers::column_map::ColumnMapTableBuilder;
use llkv_table::providers::parquet::LlkvTableProvider;
use std::sync::Arc;

#[tokio::test]
async fn test_provider_parity() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup
    let pager = Arc::new(MemPager::new());
    let col_store = Arc::new(ColumnStore::open(Arc::clone(&pager))?);
    let pq_store = Arc::new(ParquetStore::open(Arc::clone(&pager))?);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let table_id_u64 = 1;
    let table_id = TableId(table_id_u64);

    // 2. Create Providers
    // ColumnMap
    let cm_builder =
        ColumnMapTableBuilder::new(Arc::clone(&col_store), table_id_u64, schema.clone());
    let cm_provider = Arc::new(cm_builder.finish());

    // Parquet
    // ParquetStore needs table creation with FULL schema (including hidden columns)
    // because LlkvTableProvider expects them to be present in the store.
    let full_schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("created_by_txn", DataType::UInt64, false),
        Field::new("deleted_by_txn", DataType::UInt64, true),
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let _ = pq_store.create_table("test_table", full_schema)?;
    let pq_provider = Arc::new(LlkvTableProvider::new(
        Arc::clone(&pq_store),
        table_id,
        schema.clone(),
    )?);

    // 3. Ingest Data
    let row_ids = UInt64Array::from(vec![1, 2, 3]);
    let ids = Int32Array::from(vec![1, 2, 3]);
    let names = StringArray::from(vec![Some("Alice"), Some("Bob"), None]);

    // Ingest into ParquetStore
    // ParquetStore expects row_id and MVCC columns.
    let created_by = UInt64Array::from(vec![1, 1, 1]); // Created by txn 1
    let deleted_by = UInt64Array::from(vec![None, None, None]); // Not deleted

    let batch_pq = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("created_by_txn", DataType::UInt64, false),
            Field::new("deleted_by_txn", DataType::UInt64, true),
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(row_ids.clone()),
            Arc::new(created_by),
            Arc::new(deleted_by),
            Arc::new(ids.clone()),
            Arc::new(names.clone()),
        ],
    )?;
    pq_store.append_many(table_id, vec![batch_pq])?;

    // Ingest into ColumnMap
    // We need to construct a schema with metadata for user columns, AND row_id.
    let mut fields = vec![Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false)];

    for (i, f) in schema.fields().iter().enumerate() {
        let mut metadata = f.metadata().clone();
        let field_id =
            LogicalFieldId::for_user(table_id_u64.try_into().unwrap(), (i + 1) as u64 as u32);
        metadata.insert(
            FIELD_ID_META_KEY.to_string(),
            u64::from(field_id).to_string(),
        );
        fields.push(f.as_ref().clone().with_metadata(metadata));
    }

    let schema_cm = Arc::new(Schema::new(fields));
    let batch_cm = RecordBatch::try_new(
        schema_cm,
        vec![Arc::new(row_ids), Arc::new(ids), Arc::new(names)],
    )?;
    col_store.append(&batch_cm)?;

    // 4. Query and Compare
    let ctx = SessionContext::new();
    ctx.register_table("cm_table", cm_provider)?;
    ctx.register_table("pq_table", pq_provider)?;

    let df_cm = ctx.sql("SELECT * FROM cm_table ORDER BY id").await?;
    let df_pq = ctx.sql("SELECT * FROM pq_table ORDER BY id").await?;

    let results_cm = df_cm.collect().await?;
    let results_pq = df_pq.collect().await?;

    // Print results for debugging
    println!("ColumnMap Results:");
    for batch in &results_cm {
        arrow::util::pretty::print_batches(&[batch.clone()]).unwrap();
    }
    println!("Parquet Results:");
    for batch in &results_pq {
        arrow::util::pretty::print_batches(&[batch.clone()]).unwrap();
    }

    // Strip metadata from ColumnMap results for comparison
    let results_cm_clean: Vec<RecordBatch> = results_cm
        .iter()
        .map(|batch| {
            let fields: Vec<Field> = batch
                .schema()
                .fields()
                .iter()
                .map(|f| {
                    let mut f = f.as_ref().clone();
                    f.set_metadata(std::collections::HashMap::new());
                    f
                })
                .collect();
            let schema = Arc::new(Schema::new(fields));
            RecordBatch::try_new(schema, batch.columns().to_vec()).unwrap()
        })
        .collect();

    assert_eq!(results_cm_clean, results_pq);

    Ok(())
}
