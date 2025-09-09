//! Smoke test: build an Arrow RecordBatch, ingest via append_batch,
//! then read a few keys back and check the bytes.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StringArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use llkv_arrowhead::{ColumnMap, KeySpec, append_batch};
use llkv_column_map::ColumnStore;
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId};

#[test]
fn arrow_recordbatch_roundtrip() {
    // ---------- build a small RecordBatch ----------
    // cols: key:u64, a:i32, b:Utf8, c:FixedSizeList<f32; 4>
    let schema = Schema::new(vec![
        Field::new("key", DataType::UInt64, false),
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Utf8, false),
        Field::new(
            "c",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 4),
            false,
        ),
    ]);

    let keys = UInt64Array::from(vec![1u64, 2, 3]);
    let a = Int32Array::from(vec![10, 20, 30]);
    let b = StringArray::from(vec!["aa", "bbb", "cccc"]);

    // c: three rows of 4 float32 elements each.
    let c_vals = Float32Array::from(vec![
        0.0f32, 1.0, 2.0, 3.0, //
        4.0, 5.0, 6.0, 7.0, //
        8.0, 9.0, 10.0, 11.0,
    ]);
    let item_field = Arc::new(Field::new("item", DataType::Float32, false));
    let c = FixedSizeListArray::new(item_field, 4, Arc::new(c_vals), None);

    let cols: Vec<ArrayRef> = vec![Arc::new(keys), Arc::new(a), Arc::new(b), Arc::new(c)];
    let batch = RecordBatch::try_new(Arc::new(schema), cols).expect("batch");

    // ---------- map columns to field IDs ----------
    const FID_A: LogicalFieldId = 10_001;
    const FID_B: LogicalFieldId = 10_002;
    const FID_C: LogicalFieldId = 10_003;

    let map = ColumnMap {
        cols: vec![(1, FID_A), (2, FID_B), (3, FID_C)],
    };
    let keys = KeySpec::U64Be { col_idx: 0 };

    // ---------- create store and append ----------
    let p = MemPager::default();
    let store = ColumnStore::init_empty(&p);
    let mut opts = AppendOptions::default();
    // keep segments comfortably large for the test
    opts.segment_max_entries = 1_000;
    append_batch(&store, &batch, &keys, &map, opts);

    // ---------- read back a couple keys ----------
    // get_many takes (field_id, Vec<LogicalKeyBytes>).
    // Keys are u64::to_be_bytes().
    let k1 = 1u64.to_be_bytes().to_vec();
    let k3 = 3u64.to_be_bytes().to_vec();

    // Check column A (i32 values).
    let got_a = store.get_many(vec![(FID_A, vec![k1.clone(), k3.clone()])]);
    assert_eq!(got_a[0][0].as_deref().unwrap(), 10i32.to_le_bytes());
    assert_eq!(got_a[0][1].as_deref().unwrap(), 30i32.to_le_bytes());

    // Check column B (Utf8 -> bytes).
    let got_b = store.get_many(vec![(FID_B, vec![k1.clone(), k3.clone()])]);
    assert_eq!(got_b[0][0].as_deref().unwrap(), b"aa");
    assert_eq!(got_b[0][1].as_deref().unwrap(), b"cccc");

    // Check column C (f32[4] packed as fixed 16 bytes).
    let got_c = store.get_many(vec![(FID_C, vec![k1.clone(), k3.clone()])]);
    let row1 = got_c[0][0].as_deref().unwrap();
    let row3 = got_c[0][1].as_deref().unwrap();
    assert_eq!(row1.len(), 16);
    assert_eq!(row3.len(), 16);
}
