//! After upserting, fetch a RecordBatch with get_many and validate.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StringArray,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use llkv_arrowhead::{ColumnMap, KeySpec, ReadCodec, ReadSpec, append_batch, read_record_batch};
use llkv_column_map::ColumnStore;
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId};

#[test]
fn upsert_then_fetch_recordbatch() {
    // ---------- seed a batch (same as your prior test) ----------
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

    const FID_A: LogicalFieldId = 10_001;
    const FID_B: LogicalFieldId = 10_002;
    const FID_C: LogicalFieldId = 10_003;

    let map = ColumnMap {
        cols: vec![(1, FID_A), (2, FID_B), (3, FID_C)],
    };
    let keyspec = KeySpec::U64Be { col_idx: 0 };

    let p = Arc::new(MemPager::default());
    let store = ColumnStore::open(p);

    let opts = AppendOptions {
        segment_max_entries: 1_000,
        ..Default::default()
    };

    // Writer encodes values as LE (as updated).
    append_batch(&store, &batch, &keyspec, &map, opts);

    // ---------- fetch a subset of keys as a RecordBatch ----------
    // Request rows for keys [1, 3] in that order.
    let k1 = 1u64.to_be_bytes().to_vec();
    let k3 = 3u64.to_be_bytes().to_vec();
    let want_keys = vec![k1.clone(), k3.clone()];

    // Describe columns to read and how to decode them.
    let specs = vec![
        ReadSpec {
            field_id: FID_A,
            field: Field::new("a", DataType::Int32, true),
            codec: ReadCodec::I32Le,
        },
        ReadSpec {
            field_id: FID_B,
            field: Field::new("b", DataType::Utf8, true),
            codec: ReadCodec::Utf8,
        },
        ReadSpec {
            field_id: FID_C,
            field: Field::new(
                "c",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 4),
                true,
            ),
            codec: ReadCodec::F32Fixed { len: 4 },
        },
    ];

    let rb = read_record_batch(&store, &want_keys, &specs).unwrap();
    assert_eq!(rb.num_rows(), 2);
    assert_eq!(rb.num_columns(), 3);

    // a: Int32 LE -> [10, 30].
    let a = rb.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(a.len(), 2);
    assert_eq!(a.value(0), 10);
    assert_eq!(a.value(1), 30);

    // b: Utf8 -> ["aa", "cccc"].
    let b = rb.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(b.value(0), "aa");
    assert_eq!(b.value(1), "cccc");

    // c: FixedSizeList<f32;4> -> rows [0..3], [8..11].
    let c = rb
        .column(2)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    assert_eq!(c.value_length(), 4);
    let ch = c.values();
    let ch = ch.as_any().downcast_ref::<Float32Array>().unwrap();
    // First row slice
    assert!((ch.value(0) - 0.0).abs() < 1e-6);
    assert!((ch.value(1) - 1.0).abs() < 1e-6);
    assert!((ch.value(2) - 2.0).abs() < 1e-6);
    assert!((ch.value(3) - 3.0).abs() < 1e-6);
    // Second row slice
    assert!((ch.value(4) - 8.0).abs() < 1e-6);
    assert!((ch.value(5) - 9.0).abs() < 1e-6);
    assert!((ch.value(6) - 10.0).abs() < 1e-6);
    assert!((ch.value(7) - 11.0).abs() < 1e-6);
}
