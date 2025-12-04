use arrow::array::*;
use llkv_column_map::store::pruning::{IntRanges, RangeKey, compute_chunk_stats};
use std::ops::Bound;
use std::sync::Arc;
use std::collections::HashMap;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanBuilder, ScanOptions,
};
use llkv_column_map::store::{ColumnStore, IndexKind, ROW_ID_COLUMN_NAME};
use llkv_storage::pager::{MemPager, InstrumentedPager};
use llkv_types::LogicalFieldId;

#[test]
fn test_compute_chunk_stats_integers() {
    let array = Int32Array::from(vec![Some(1), Some(5), Some(10), None]);
    let stats = compute_chunk_stats(&(Arc::new(array) as ArrayRef)).unwrap();

    // 1 -> 0x80000001
    // 10 -> 0x8000000A
    assert_eq!(stats.min_u64, (1i32 as u32 ^ 0x8000_0000) as u64);
    assert_eq!(stats.max_u64, (10i32 as u32 ^ 0x8000_0000) as u64);
    assert_eq!(stats.null_count, 1);
    assert_eq!(stats.distinct_count, 3);
}

#[test]
fn test_compute_chunk_stats_floats() {
    let array = Float64Array::from(vec![Some(1.0), Some(-5.0), Some(10.5), None]);
    let stats = compute_chunk_stats(&(Arc::new(array) as ArrayRef)).unwrap();

    // Check that min corresponds to -5.0 and max to 10.5
    // We can't easily check the exact u64 bits without the helper, but we can check ordering.
    assert!(stats.min_u64 < stats.max_u64);
    assert_eq!(stats.null_count, 1);
    assert_eq!(stats.distinct_count, 3);
}

#[test]
fn test_compute_chunk_stats_strings() {
    let array = StringArray::from(vec![Some("apple"), Some("banana"), Some("cherry"), None]);
    let stats = compute_chunk_stats(&(Arc::new(array) as ArrayRef)).unwrap();

    assert!(stats.min_u64 < stats.max_u64);
    assert_eq!(stats.null_count, 1);
    assert_eq!(stats.distinct_count, 3);
}

#[test]
fn test_compute_chunk_stats_all_null() {
    let array = Int32Array::from(vec![None, None]);
    let stats = compute_chunk_stats(&(Arc::new(array) as ArrayRef)).unwrap();

    assert_eq!(stats.min_u64, 0);
    assert_eq!(stats.max_u64, 0);
    assert_eq!(stats.null_count, 2);
    assert_eq!(stats.distinct_count, 0);
}

#[test]
fn test_pruning_ranges_overlap() {
    let mut ranges = IntRanges::default();
    // Range: [5, 10]
    RangeKey::store(&mut ranges, Bound::Included(5i32), Bound::Included(10i32));

    // Chunk: [1, 4] -> No overlap
    let chunk_min = (1i32 as u32 ^ 0x8000_0000) as u64;
    let chunk_max = (4i32 as u32 ^ 0x8000_0000) as u64;
    assert!(!ranges.matches(chunk_min, chunk_max));

    // Chunk: [1, 6] -> Overlap
    let chunk_max = (6i32 as u32 ^ 0x8000_0000) as u64;
    assert!(ranges.matches(chunk_min, chunk_max));

    // Chunk: [11, 20] -> No overlap
    let chunk_min = (11i32 as u32 ^ 0x8000_0000) as u64;
    let chunk_max = (20i32 as u32 ^ 0x8000_0000) as u64;
    assert!(!ranges.matches(chunk_min, chunk_max));
}

#[test]
fn test_pruning_ranges_floats() {
    let mut ranges = IntRanges::default();
    // Range: [-1.0, 1.0]
    RangeKey::store(&mut ranges, Bound::Included(-1.0f64), Bound::Included(1.0f64));

    // Helper to convert f64 to sortable u64
    fn f64_to_u64(val: f64) -> u64 {
        let bits = val.to_bits();
        if bits & 0x8000_0000_0000_0000 != 0 {
            !bits
        } else {
            bits | 0x8000_0000_0000_0000
        }
    }

    // Chunk: [-2.0, -1.5] -> No overlap
    let chunk_min = f64_to_u64(-2.0);
    let chunk_max = f64_to_u64(-1.5);
    assert!(!ranges.matches(chunk_min, chunk_max));

    // Chunk: [-0.5, 0.5] -> Overlap
    let chunk_min = f64_to_u64(-0.5);
    let chunk_max = f64_to_u64(0.5);
    assert!(ranges.matches(chunk_min, chunk_max));
}

#[test]
fn test_pruning_skips_irrelevant_chunks() {
    let (pager, stats) = InstrumentedPager::new(MemPager::new());
    let pager = Arc::new(pager);
    let store = ColumnStore::open(pager.clone()).unwrap();
    let field_id = LogicalFieldId::for_user_table_0(100);

    let mut md = HashMap::new();
    md.insert(
        llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
        u64::from(field_id).to_string(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md),
    ]));

    // Ingest 3 distinct chunks of data
    // Chunk 1: 0..1000
    // Chunk 2: 2000..3000
    // Chunk 3: 4000..5000
    
    let ranges = vec![
        (0..1000),
        (2000..3000),
        (4000..5000),
    ];

    for range in ranges {
        let _n = range.end - range.start;
        let rid: Vec<u64> = range.clone().collect(); // Use value as row id for simplicity
        let vals: Vec<u64> = range.clone().collect();
        let rid_arr = Arc::new(UInt64Array::from(rid));
        let val_arr = Arc::new(UInt64Array::from(vals));
        let batch = RecordBatch::try_new(schema.clone(), vec![rid_arr, val_arr]).unwrap();
        store.append(&batch).unwrap();
    }

    // Force index creation. This should create sorted runs.
    store.register_index(field_id, IndexKind::Sort).unwrap();

    // Clear access history from ingestion/indexing
    stats.reset();

    // Query for [2500, 2600]. Should only hit the middle chunk.
    struct Collect {
        out: Vec<u64>,
    }
    impl PrimitiveVisitor for Collect {}
    impl PrimitiveWithRowIdsVisitor for Collect {}
    impl PrimitiveSortedWithRowIdsVisitor for Collect {}
    impl PrimitiveSortedVisitor for Collect {
        fn u64_run(&mut self, a: &UInt64Array, s: usize, l: usize) {
            let e = s + l;
            for i in s..e {
                self.out.push(a.value(i));
            }
        }
    }

    let mut coll = Collect { out: Vec::new() };
    ScanBuilder::new(&store, field_id)
        .options(ScanOptions {
            sorted: true,
            ..Default::default()
        })
        .with_range::<u64, _>(2500..=2600)
        .run(&mut coll)
        .unwrap();

    assert!(!coll.out.is_empty());
    assert!(coll.out.iter().all(|&x| x >= 2500 && x <= 2600));

    let access_count_pruned = stats.snapshot().physical_gets;
    println!("Access count with pruning target: {}", access_count_pruned);

    // Now query for everything [0, 5000]. Should hit all chunks.
    stats.reset();
    let mut coll_all = Collect { out: Vec::new() };
    ScanBuilder::new(&store, field_id)
        .options(ScanOptions {
            sorted: true,
            ..Default::default()
        })
        .with_range::<u64, _>(0..=5000)
        .run(&mut coll_all)
        .unwrap();
    
    let access_count_all = stats.snapshot().physical_gets;
    println!("Access count all: {}", access_count_all);

    // If pruning works, access_count_pruned should be significantly less than access_count_all.
    assert!(access_count_pruned < access_count_all, "Pruning did not reduce IO! Pruned: {}, All: {}", access_count_pruned, access_count_all);
}
