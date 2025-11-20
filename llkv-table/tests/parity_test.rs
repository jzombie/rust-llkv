use arrow::array::{Int32Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
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

/// Helper to create the stores and providers, then ingest data.
///
/// - `schema`: The user schema (no system columns).
/// - `cm_batches`: Batches to append to ColumnMap (simulating history/updates).
/// - `pq_batches`: Batches to append to ParquetStore (representing the expected final state).
async fn setup_parity_test_custom(
    schema: SchemaRef,
    cm_batches: Vec<RecordBatch>,
    pq_batches: Vec<RecordBatch>,
) -> Result<SessionContext, Box<dyn std::error::Error>> {
    // 1. Setup
    let pager = Arc::new(MemPager::new());
    let col_store = Arc::new(ColumnStore::open(Arc::clone(&pager))?);
    let pq_store = Arc::new(ParquetStore::open(Arc::clone(&pager))?);

    // Full Schema (System + User columns)
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
    let _ = pq_store.create_table("test_table", full_schema.clone())?;
    let pq_provider = Arc::new(LlkvTableProvider::new(
        Arc::clone(&pq_store),
        table_id,
        schema.clone(),
    )?);

    // 3. Ingest Data into ParquetStore (Expected State)
    if !pq_batches.is_empty() {
        pq_store.append_many(table_id, pq_batches)?;
    }

    // 4. Ingest Data into ColumnMap (History/Updates)
    // We need to attach metadata to the schema for ColumnMap ingestion
    let mut cm_fields = Vec::with_capacity(full_schema.fields().len());
    for f in full_schema.fields() {
        let mut f = f.as_ref().clone();
        if f.name() == ROW_ID_COLUMN_NAME {
            cm_fields.push(f);
        } else if let Ok(idx) = schema.index_of(f.name()) {
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

    for batch in cm_batches {
        // Cast batch to the metadata-rich schema
        // Note: This assumes the input batch has the same physical layout (columns) as schema_cm
        // which should be true if they were created with the same structure.
        // We just replace the schema pointer to include the metadata.
        let batch_with_meta = RecordBatch::try_new(schema_cm.clone(), batch.columns().to_vec())?;
        col_store.append(&batch_with_meta)?;
    }

    // 5. Register Tables
    let ctx = SessionContext::new();
    ctx.register_table("cm_table", cm_provider)?;
    ctx.register_table("pq_table", pq_provider)?;

    Ok(ctx)
}

async fn setup_parity_test() -> Result<SessionContext, Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let row_ids = UInt64Array::from(vec![1, 2, 3, 4, 5]);
    let ids = Int32Array::from(vec![1, 2, 3, 4, 5]);
    let names = StringArray::from(vec![
        Some("Alice"),
        Some("Bob"),
        None,
        Some("David"),
        Some("Eve"),
    ]);

    // Parquet Data (Full Schema with MVCC)
    let created_by = UInt64Array::from(vec![1, 1, 1, 1, 1]);
    let deleted_by = UInt64Array::from(vec![None, None, None, None, None]);

    // Reconstruct full schema for batch creation
    let mut full_fields = vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new(CREATED_BY_COLUMN_NAME, DataType::UInt64, false),
        Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, true),
    ];
    full_fields.extend(schema.fields().iter().map(|f| f.as_ref().clone()));
    let full_schema = Arc::new(Schema::new(full_fields));

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

    // ColumnMap Data (User Schema + RowId)
    // Note: setup_parity_test_custom expects batches that match the "User Schema + RowId" structure
    // but without the MVCC columns, because ColumnMap ingestion doesn't strictly require them
    // (or rather, the test helper adds metadata to a schema that matches this structure).
    // Wait, the previous test code constructed `schema_cm` which had RowId + User Columns.
    // Let's match that.
    let cm_fields = vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ];
    let schema_cm_input = Arc::new(Schema::new(cm_fields));

    let batch_cm = RecordBatch::try_new(
        schema_cm_input,
        vec![
            Arc::new(row_ids.clone()),
            Arc::new(ids.clone()),
            Arc::new(names.clone()),
        ],
    )?;

    setup_parity_test_custom(schema, vec![batch_cm], vec![batch_pq]).await
}

async fn assert_parity(
    ctx: &SessionContext,
    query_suffix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let sql_cm = format!("SELECT * FROM cm_table {}", query_suffix);
    let sql_pq = format!("SELECT * FROM pq_table {}", query_suffix);

    let df_cm = ctx.sql(&sql_cm).await?;
    let df_pq = ctx.sql(&sql_pq).await?;

    let results_cm = df_cm.collect().await?;
    let results_pq = df_pq.collect().await?;

    // Print results for debugging
    println!("Query: {}", query_suffix);
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

    assert_eq!(
        results_cm_clean, results_pq,
        "Mismatch for query: {}",
        query_suffix
    );

    Ok(())
}

#[tokio::test]
async fn test_provider_parity_basic() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = setup_parity_test().await?;
    assert_parity(&ctx, "ORDER BY id").await?;
    Ok(())
}

#[tokio::test]
async fn test_provider_parity_filter_eq() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = setup_parity_test().await?;
    assert_parity(&ctx, "WHERE id = 2 ORDER BY id").await?;
    Ok(())
}

#[tokio::test]
async fn test_provider_parity_filter_gt() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = setup_parity_test().await?;
    assert_parity(&ctx, "WHERE id > 2 ORDER BY id").await?;
    Ok(())
}

#[tokio::test]
async fn test_provider_parity_filter_lt() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = setup_parity_test().await?;
    assert_parity(&ctx, "WHERE id < 4 ORDER BY id").await?;
    Ok(())
}

#[tokio::test]
async fn test_provider_parity_limit() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = setup_parity_test().await?;
    assert_parity(&ctx, "ORDER BY id LIMIT 2").await?;
    Ok(())
}

#[tokio::test]
async fn test_provider_parity_filter_and_limit() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = setup_parity_test().await?;
    assert_parity(&ctx, "WHERE id > 1 ORDER BY id LIMIT 2").await?;
    Ok(())
}

#[tokio::test]
async fn test_provider_parity_lww() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int64, true),
    ]));

    // Batch 1: Insert row 1
    let row_ids1 = UInt64Array::from(vec![1]);
    let ids1 = Int32Array::from(vec![1]);
    let vals1 = arrow::array::Int64Array::from(vec![Some(100)]);

    let batch_cm1 = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("id", DataType::Int32, false),
            Field::new("val", DataType::Int64, true),
        ])),
        vec![Arc::new(row_ids1), Arc::new(ids1), Arc::new(vals1)],
    )?;

    // Batch 2: Update row 1
    let row_ids2 = UInt64Array::from(vec![1]);
    let ids2 = Int32Array::from(vec![1]);
    let vals2 = arrow::array::Int64Array::from(vec![Some(200)]);

    let batch_cm2 = RecordBatch::try_new(
        batch_cm1.schema(),
        vec![Arc::new(row_ids2), Arc::new(ids2), Arc::new(vals2)],
    )?;

    // Expected State (Parquet): Only the final state of row 1
    let row_ids_pq = UInt64Array::from(vec![1]);
    let ids_pq = Int32Array::from(vec![1]);
    let vals_pq = arrow::array::Int64Array::from(vec![Some(200)]);
    let created_by = UInt64Array::from(vec![1]);
    let deleted_by = UInt64Array::from(vec![None]);

    let mut full_fields = vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new(CREATED_BY_COLUMN_NAME, DataType::UInt64, false),
        Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, true),
    ];
    full_fields.extend(schema.fields().iter().map(|f| f.as_ref().clone()));
    let full_schema = Arc::new(Schema::new(full_fields));

    let batch_pq = RecordBatch::try_new(
        full_schema,
        vec![
            Arc::new(row_ids_pq),
            Arc::new(created_by),
            Arc::new(deleted_by),
            Arc::new(ids_pq),
            Arc::new(vals_pq),
        ],
    )?;

    let ctx = setup_parity_test_custom(schema, vec![batch_cm1, batch_cm2], vec![batch_pq]).await?;

    assert_parity(&ctx, "ORDER BY id").await?;
    Ok(())
}

#[tokio::test]
async fn test_provider_parity_null_updates() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int64, true),
    ]));

    let schema_cm = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int64, true),
    ]));

    // Batch 1: row 1 = 10, row 2 = NULL
    let batch_cm1 = RecordBatch::try_new(
        schema_cm.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![1, 2])),
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(arrow::array::Int64Array::from(vec![Some(10), None])),
        ],
    )?;

    // Batch 2: row 1 = NULL, row 2 = 20
    let batch_cm2 = RecordBatch::try_new(
        schema_cm.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![1, 2])),
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(arrow::array::Int64Array::from(vec![None, Some(20)])),
        ],
    )?;

    // Expected State (Parquet)
    let row_ids_pq = UInt64Array::from(vec![1, 2]);
    let ids_pq = Int32Array::from(vec![1, 2]);
    let vals_pq = arrow::array::Int64Array::from(vec![None, Some(20)]);
    let created_by = UInt64Array::from(vec![1, 1]);
    let deleted_by = UInt64Array::from(vec![None, None]);

    let mut full_fields = vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new(CREATED_BY_COLUMN_NAME, DataType::UInt64, false),
        Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, true),
    ];
    full_fields.extend(schema.fields().iter().map(|f| f.as_ref().clone()));
    let full_schema = Arc::new(Schema::new(full_fields));

    let batch_pq = RecordBatch::try_new(
        full_schema,
        vec![
            Arc::new(row_ids_pq),
            Arc::new(created_by),
            Arc::new(deleted_by),
            Arc::new(ids_pq),
            Arc::new(vals_pq),
        ],
    )?;

    let ctx = setup_parity_test_custom(schema, vec![batch_cm1, batch_cm2], vec![batch_pq]).await?;

    assert_parity(&ctx, "ORDER BY id").await?;
    Ok(())
}

#[tokio::test]
async fn test_provider_parity_large_batch() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int64, true),
    ]));

    let num_rows = 2000;
    let row_ids: Vec<u64> = (1..=num_rows as u64).collect();
    let ids: Vec<i32> = (1..=num_rows as i32).collect();
    let vals: Vec<Option<i64>> = (1..=num_rows)
        .map(|i| {
            if i % 10 == 0 {
                None
            } else {
                Some(i as i64 * 10)
            }
        })
        .collect();

    let row_ids_arr = UInt64Array::from(row_ids);
    let ids_arr = Int32Array::from(ids);
    let vals_arr = arrow::array::Int64Array::from(vals);

    // ColumnMap Batch
    let schema_cm = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int64, true),
    ]));

    let batch_cm = RecordBatch::try_new(
        schema_cm,
        vec![
            Arc::new(row_ids_arr.clone()),
            Arc::new(ids_arr.clone()),
            Arc::new(vals_arr.clone()),
        ],
    )?;

    // Parquet Batch
    let created_by = UInt64Array::from(vec![1; num_rows]);
    let deleted_by = UInt64Array::from(vec![None; num_rows]);

    let mut full_fields = vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new(CREATED_BY_COLUMN_NAME, DataType::UInt64, false),
        Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, true),
    ];
    full_fields.extend(schema.fields().iter().map(|f| f.as_ref().clone()));
    let full_schema = Arc::new(Schema::new(full_fields));

    let batch_pq = RecordBatch::try_new(
        full_schema,
        vec![
            Arc::new(row_ids_arr),
            Arc::new(created_by),
            Arc::new(deleted_by),
            Arc::new(ids_arr),
            Arc::new(vals_arr),
        ],
    )?;

    let ctx = setup_parity_test_custom(schema, vec![batch_cm], vec![batch_pq]).await?;

    assert_parity(&ctx, "ORDER BY id").await?;

    // Test filter on large batch
    assert_parity(&ctx, "WHERE id > 1900 ORDER BY id").await?;

    // Test limit on large batch
    assert_parity(&ctx, "ORDER BY id LIMIT 10").await?;

    Ok(())
}
