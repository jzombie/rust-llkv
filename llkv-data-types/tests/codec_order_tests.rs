use std::borrow::Cow;
use std::ops::Bound;
use std::sync::Arc;

use llkv_column_map::ColumnStore;
use llkv_column_map::column_store::read_scan::{Direction, OrderBy, ValueScanOpts};
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode};

use llkv_data_types::{DataType, DecodedValue, decode_value, encode_value};

/* --------------------------- Shared helpers ---------------------------- */

/// Set up a fresh in-memory ColumnStore and return (store, new fid).
#[inline]
fn setup_store() -> (ColumnStore<MemPager>, LogicalFieldId) {
    let pager = Arc::new(MemPager::default());
    let store = ColumnStore::open(pager);
    let fid: LogicalFieldId = 42;
    (store, fid)
}

/// Append a single Put batch to the given field id.
#[inline]
#[allow(clippy::type_complexity)] // TODO: Alias type
fn append_one_put(
    store: &ColumnStore<MemPager>,
    fid: LogicalFieldId,
    items: Vec<(Cow<'static, [u8]>, Cow<'static, [u8]>)>,
) {
    let puts = vec![Put {
        field_id: fid,
        items,
    }];
    let opts = AppendOptions {
        mode: ValueMode::Auto,
        segment_max_entries: 100_000,
        segment_max_bytes: 32 << 20,
        last_write_wins_in_batch: true,
        value_order: None,
    };
    store.append_many(puts, opts);
}

/// Run a value-ordered scan (ascending) over `fid` and return the raw
/// encoded value frames as owned Vec<u8> (avoid lifetime issues).
#[inline]
fn scan_value_frames(store: &ColumnStore<MemPager>, fid: LogicalFieldId) -> Vec<Vec<u8>> {
    let it = store
        .scan_values_lww(
            fid,
            ValueScanOpts {
                order_by: OrderBy::Value,
                dir: Direction::Forward,
                lo: Bound::Unbounded,
                hi: Bound::Unbounded,
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 16,
                frame_predicate: None,
            },
        )
        .expect("scan init");

    let mut out = Vec::new();
    for item in it {
        let a = item.value.start() as usize;
        let b = item.value.end() as usize;
        let arc = item.value.data();
        out.push(arc.as_ref()[a..b].to_vec());
    }
    out
}

/* --------------- Public-API generic item builder (no traits) ----------- */

/// Build encoded items (key=row-id BE, val=encoded via public API).
/// `dtype` must match the variants contained in `vals`; we `expect` on mismatch
/// because tests should fail loudly if the input is inconsistent.
#[inline]
#[allow(clippy::type_complexity)]
fn build_items_from_decoded(
    dtype: DataType,
    vals: &[DecodedValue<'static>],
) -> Vec<(Cow<'static, [u8]>, Cow<'static, [u8]>)> {
    let mut items = Vec::with_capacity(vals.len());
    for (i, dv) in vals.iter().enumerate() {
        let key = (i as u64 + 1).to_be_bytes().to_vec(); // lex==numeric
        let mut val = Vec::new();
        encode_value(*dv, &dtype, &mut val).expect("encode");
        items.push((Cow::Owned(key), Cow::Owned(val)));
    }
    items
}

/* ------------------------------ Tests ---------------------------------- */

/// `Utf8` should order case-insensitively (with tie-breaks on original bytes)
/// and round-trip back to the original strings using the public API only.
#[test]
fn test_utf8_order_roundtrip() {
    let (store, fid) = setup_store();

    let words: [&'static str; 9] = [
        "zeta",
        "Apple",
        "alpha",
        "Banana",
        "banana",
        "Large Words",
        "lower words",
        "LARGE words",
        "aardvark",
    ];

    // Build DecodedValue<'static> list (string literals are 'static).
    let vals: Vec<DecodedValue<'static>> = words.iter().map(|&s| DecodedValue::Str(s)).collect();

    let items = build_items_from_decoded(DataType::Utf8, &vals);
    append_one_put(&store, fid, items);

    // Value-ordered scan (ascending)
    let frames = scan_value_frames(&store, fid);

    // Decode via public API.
    let mut got: Vec<String> = Vec::new();
    for bytes in frames {
        let dv = decode_value(bytes.as_slice(), &DataType::Utf8).expect("decode utf8");
        if let DecodedValue::Str(s) = dv {
            got.push(s.to_string());
        } else {
            panic!("expected DecodedValue::Str");
        }
    }

    let expect = vec![
        "aardvark",
        "alpha",
        "Apple",
        "Banana",
        "banana",
        "LARGE words",
        "Large Words",
        "lower words",
        "zeta",
    ];
    assert_eq!(got, expect);
}

/// `Bytes` should order by raw bytewise lex of the payload and round-trip
/// losslessly using the public API only.
#[test]
fn test_bytes_order_roundtrip() {
    let (store, fid) = setup_store();

    let blobs: [&'static [u8]; 7] = [
        b"zed",
        b"abd",
        b"\x00",
        b"abc",
        b"\x00\xff",
        b"ab",
        b"\x00\x01",
    ];

    // Build DecodedValue<'static> list (byte string literals are 'static).
    let vals: Vec<DecodedValue<'static>> = blobs.iter().map(|&b| DecodedValue::Bytes(b)).collect();

    let items = build_items_from_decoded(DataType::Bytes, &vals);
    append_one_put(&store, fid, items);

    // Value-ordered scan (ascending)
    let frames = scan_value_frames(&store, fid);

    // Decode via public API.
    let mut got: Vec<Vec<u8>> = Vec::new();
    for bytes in frames {
        let dv = decode_value(bytes.as_slice(), &DataType::Bytes).expect("decode bytes");
        if let DecodedValue::Bytes(b) = dv {
            got.push(b.to_vec());
        } else {
            panic!("expected DecodedValue::Bytes");
        }
    }

    let expect: Vec<Vec<u8>> = vec![
        b"\x00".to_vec(),
        b"\x00\x01".to_vec(),
        b"\x00\xff".to_vec(),
        b"ab".to_vec(),
        b"abc".to_vec(),
        b"abd".to_vec(),
        b"zed".to_vec(),
    ];
    assert_eq!(got, expect);
}

/// `I64` should order by numeric value (including negatives) and round-trip
/// using the public API only. Encoded bytes must compare lexicographically
/// in the same order as the numeric order.
/// NOTE: The seed order below is intentionally *not* the expected order.
#[test]
fn test_i64_order_roundtrip() {
    let (store, fid) = setup_store();

    // Deliberately scrambled: negatives, zero, positives, extremes.
    let nums: [i64; 9] = [
        42,
        -1,
        i64::MAX,
        -10,
        0,
        i64::MIN + 1, // avoid MIN overflow in abs
        123_456_789,
        -1_000_000,
        1,
    ];

    // Build DecodedValue<'static> list.
    let vals: Vec<DecodedValue<'static>> = nums.iter().copied().map(DecodedValue::I64).collect();

    // Sanity: explicit expected numeric ascending order; assert seed != expect.
    let expect: Vec<i64> = vec![
        i64::MIN + 1,
        -1_000_000,
        -10,
        -1,
        0,
        1,
        42,
        123_456_789,
        i64::MAX,
    ];
    assert_ne!(nums.to_vec(), expect, "seed order must differ");

    let items = build_items_from_decoded(DataType::I64, &vals);
    append_one_put(&store, fid, items);

    // Value-ordered scan (ascending)
    let frames = scan_value_frames(&store, fid);

    // Decode via public API.
    let mut got: Vec<i64> = Vec::new();
    for bytes in frames {
        let dv = decode_value(bytes.as_slice(), &DataType::I64).expect("decode i64");
        if let DecodedValue::I64(x) = dv {
            got.push(x);
        } else {
            panic!("expected DecodedValue::I64");
        }
    }

    assert_eq!(got, expect);
}

/// `CategoryId` should order by numeric value (u32) and round-trip
/// using the public API only. Encoded bytes must compare lexicographically
/// in the same order as numeric order.
/// NOTE: The seed order below is intentionally *not* the expected order.
#[test]
fn test_categoryid_order_roundtrip() {
    let (store, fid) = setup_store();

    // Deliberately scrambled
    let cats: [u32; 7] = [42, 0, u32::MAX, 7, 1, 999_999, 100];

    // Build DecodedValue<'static> list.
    let vals: Vec<DecodedValue<'static>> =
        cats.iter().copied().map(DecodedValue::CategoryId).collect();

    // Sanity: explicit expected numeric ascending order; assert seed != expect.
    let expect: Vec<u32> = vec![0, 1, 7, 42, 100, 999_999, u32::MAX];
    assert_ne!(cats.to_vec(), expect, "seed order must differ");

    let items = build_items_from_decoded(DataType::CategoryId, &vals);
    append_one_put(&store, fid, items);

    // Value-ordered scan (ascending)
    let frames = scan_value_frames(&store, fid);

    // Decode via public API.
    let mut got: Vec<u32> = Vec::new();
    for bytes in frames {
        let dv = decode_value(bytes.as_slice(), &DataType::CategoryId).expect("decode categoryid");
        if let DecodedValue::CategoryId(x) = dv {
            got.push(x);
        } else {
            panic!("expected DecodedValue::CategoryId");
        }
    }

    assert_eq!(got, expect);
}
