use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::compute;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::ROW_ID_COLUMN_NAME;
use llkv_column_map::store::ColumnStore;
use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanOptions,
};
use llkv_column_map::types::LogicalFieldId;
use llkv_storage::pager::MemPager;

fn schema_with_row_id(field: Field) -> Arc<Schema> {
    let rid = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    Arc::new(Schema::new(vec![rid, field]))
}

struct ChunkAnalyzer {
    chunks: Vec<usize>,
    total_sum: u128,
}

impl ChunkAnalyzer {
    fn new() -> Self {
        Self {
            chunks: Vec::new(),
            total_sum: 0,
        }
    }

    fn report(&self) {
        println!("=== Chunk Analysis ===");
        println!("Total chunks: {}", self.chunks.len());
        println!("Total rows: {}", self.chunks.iter().sum::<usize>());
        println!("Chunk sizes: {:?}", self.chunks);
        if !self.chunks.is_empty() {
            let avg = self.chunks.iter().sum::<usize>() as f64 / self.chunks.len() as f64;
            let min = *self.chunks.iter().min().unwrap();
            let max = *self.chunks.iter().max().unwrap();
            println!("Avg chunk size: {:.1}", avg);
            println!("Min chunk size: {}", min);
            println!("Max chunk size: {}", max);
        }
        println!("Total sum: {}", self.total_sum);
        println!("=======================");
    }
}

impl PrimitiveVisitor for ChunkAnalyzer {
    fn u64_chunk(&mut self, a: &UInt64Array) {
        self.chunks.push(a.len());
        if let Some(s) = compute::sum(a) {
            self.total_sum += s as u128;
        }
    }
}

impl PrimitiveSortedVisitor for ChunkAnalyzer {}
impl PrimitiveWithRowIdsVisitor for ChunkAnalyzer {}
impl PrimitiveSortedWithRowIdsVisitor for ChunkAnalyzer {}

fn main() {
    println!("Analyzing chunk fragmentation...\n");

    // Test 1: Single large batch (1M rows)
    println!("=== Test 1: Single batch (1M rows) ===");
    {
        let pager = Arc::new(MemPager::new());
        let store = ColumnStore::open(pager).unwrap();
        let field_id = LogicalFieldId::for_user_table_0(7777);

        let mut md = HashMap::new();
        md.insert("field_id".to_string(), u64::from(field_id).to_string());
        let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);
        let schema = schema_with_row_id(data_f);

        // Single batch with 1M rows
        let rid: Vec<u64> = (0..1_000_000u64).collect();
        let vals: Vec<u64> = (0..1_000_000u64).collect();

        let rid_arr = Arc::new(UInt64Array::from(rid));
        let val_arr = Arc::new(UInt64Array::from(vals));
        let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();

        store.append(&batch).unwrap();

        let mut analyzer = ChunkAnalyzer::new();
        store
            .scan(
                field_id,
                ScanOptions {
                    sorted: false,
                    reverse: false,
                    with_row_ids: false,
                    limit: None,
                    offset: 0,
                    include_nulls: false,
                    nulls_first: false,
                    anchor_row_id_field: None,
                },
                &mut analyzer,
            )
            .unwrap();

        analyzer.report();
    }

    println!();

    // Test 2: Many small batches (1000 batches of 1000 rows each)
    println!("=== Test 2: Fragmented (1000 batches of 1000 rows) ===");
    {
        let pager = Arc::new(MemPager::new());
        let store = ColumnStore::open(pager).unwrap();
        let field_id = LogicalFieldId::for_user_table_0(8888);

        let mut md = HashMap::new();
        md.insert("field_id".to_string(), u64::from(field_id).to_string());
        let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);
        let schema = schema_with_row_id(data_f);

        // 1000 separate batches of 1000 rows each
        for chunk_id in 0..1000u64 {
            let start = chunk_id * 1000;
            let end = start + 1000;

            let rid: Vec<u64> = (start..end).collect();
            let vals: Vec<u64> = (start..end).collect();

            let rid_arr = Arc::new(UInt64Array::from(rid));
            let val_arr = Arc::new(UInt64Array::from(vals));
            let batch = RecordBatch::try_new(schema.clone(), vec![rid_arr, val_arr]).unwrap();

            store.append(&batch).unwrap();
        }

        let mut analyzer = ChunkAnalyzer::new();
        store
            .scan(
                field_id,
                ScanOptions {
                    sorted: false,
                    reverse: false,
                    with_row_ids: false,
                    limit: None,
                    offset: 0,
                    include_nulls: false,
                    nulls_first: false,
                    anchor_row_id_field: None,
                },
                &mut analyzer,
            )
            .unwrap();

        analyzer.report();
    }
}
