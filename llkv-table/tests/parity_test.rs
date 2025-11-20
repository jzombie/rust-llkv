use arrow::array::{Int32Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use llkv_column_map::store::{
    CREATED_BY_COLUMN_NAME, ColumnStore, DELETED_BY_COLUMN_NAME, FIELD_ID_META_KEY,
    ROW_ID_COLUMN_NAME,
};
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

    // Full Schema (System + User columns)
    // Both stores use this schema structure. Note that ColumnMap additionally requires
    // `FIELD_ID_META_KEY` metadata on user columns to map them to their physical storage location.
    let mut full_fields = vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new(CREATED_BY_COLUMN_NAME, DataType::UInt64, false),
        Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, true),
    ];
    full_fields.extend(schema.fields().iter().map(|f| f.as_ref().clone()));
    let full_schema = Arc::new(Schema::new(full_fields));

    let table_id_u64 = 1;
    let table_id = TableId(table_id_u64);

    // 2. Create Providers
    // ColumnMap
    let cm_builder =
        ColumnMapTableBuilder::new(Arc::clone(&col_store), table_id_u64, schema.clone());
    let cm_provider = Arc::new(cm_builder.finish());

    // Parquet
    // ParquetStore operates as a "dumb" column store that persists exactly what it is given.
    // It does not automatically inject MVCC columns during append; the caller (or upper layer)
    // must provide them. Therefore, we manually construct batches with `created_by_txn` and
    // `deleted_by_txn` to satisfy the store's schema requirements.
    let _ = pq_store.create_table("test_table", full_schema.clone())?;
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
        full_schema.clone(),
        vec![
            Arc::new(row_ids.clone()),
            Arc::new(created_by.clone()),
            Arc::new(deleted_by.clone()),
            Arc::new(ids.clone()),
            Arc::new(names.clone()),
        ],
    )?;
    pq_store.append_many(table_id, vec![batch_pq])?;

    // Ingest into ColumnMap
    // We need to construct a schema with metadata for user columns.
    // We start from full_schema to ensure parity, adding metadata where needed.
    // Ingest into ColumnMap
    // ColumnMap uses a `LogicalFieldId` mapping to locate columns in the pager. This mapping
    // is passed via schema metadata. The test manually injects this metadata because we are
    // bypassing the catalog/table layer that would normally handle this.
    //
    // Note: ColumnMap currently requires Field IDs for all columns except row_id.
    // We skip MVCC columns here because we haven't assigned them Field IDs in this test setup,
    // effectively testing it in a non-MVCC mode.
    let mut cm_fields = Vec::with_capacity(full_schema.fields().len());
    for f in full_schema.fields() {
        let mut f = f.as_ref().clone();

        if f.name() == ROW_ID_COLUMN_NAME {
            cm_fields.push(f);
        } else if let Ok(idx) = schema.index_of(f.name()) {
            // If it's a user column (present in the user schema), add metadata
            let mut metadata = f.metadata().clone();
            let field_id =
                LogicalFieldId::for_user(table_id_u64.try_into().unwrap(), (idx + 1) as u64 as u32);
            metadata.insert(
                FIELD_ID_META_KEY.to_string(),
                u64::from(field_id).to_string(),
            );
            f.set_metadata(metadata);
            cm_fields.push(f);
        }
    }

    let schema_cm = Arc::new(Schema::new(cm_fields));
    let batch_cm = RecordBatch::try_new(
        schema_cm,
        vec![
            Arc::new(row_ids.clone()),
            Arc::new(ids.clone()),
            Arc::new(names.clone()),
        ],
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
