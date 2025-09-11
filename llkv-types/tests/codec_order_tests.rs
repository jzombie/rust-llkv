use std::borrow::Cow;
use std::ops::Bound;

use llkv_column_map::ColumnStore;
use llkv_column_map::column_store::read_scan::{Direction, OrderBy, ValueScanOpts};
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode};

use llkv_types::internal::Codec;
use llkv_types::internal::{Bytes, Utf8CaseFold};

/* --------------------------- Shared helpers ---------------------------- */

/// Set up a fresh in-memory ColumnStore and return (store, new fid).
#[inline]
fn setup_store() -> (ColumnStore<'static, MemPager>, LogicalFieldId) {
    // Leak the pager for 'static lifetime in tests. Simpler than threading
    // lifetimes through helpers and perfectly fine for unit tests.
    let pager: &'static MemPager = Box::leak(Box::new(MemPager::default()));
    let store = ColumnStore::init_empty(pager);
    let fid: LogicalFieldId = 42;
    (store, fid)
}

/// Append a single Put batch to the given field id.
#[inline]
fn append_one_put(
    store: &ColumnStore<'static, MemPager>,
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
    };
    store.append_many(puts, opts);
}

/// Run a value-ordered scan (ascending) over `fid` and return the raw
/// encoded value frames as owned Vec<u8> (avoid lifetime issues).
#[inline]
fn scan_value_frames(store: &ColumnStore<'static, MemPager>, fid: LogicalFieldId) -> Vec<Vec<u8>> {
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

/// Build encoded items (key=row-id BE, val=encoded via `C`) from a slice of
/// codec-borrowed values (e.g., `&[&str]` for `Utf8CaseFold`,
/// `&[&[u8]]` for `Bytes`).
#[inline]
fn build_items<'a, C>(vals: &[C::Borrowed<'a>]) -> Vec<(Cow<'static, [u8]>, Cow<'static, [u8]>)>
where
    C: Codec,
    C::Borrowed<'a>: Clone, // <-- require Clone, fixes the move
{
    let mut items = Vec::with_capacity(vals.len());
    for (i, v) in vals.iter().enumerate() {
        let key = (i as u64 + 1).to_be_bytes().to_vec(); // lex==numeric
        let mut val = Vec::new();
        // Pass a clone instead of moving
        C::encode_into(&mut val, v.clone()).expect("encoding must succeed");
        items.push((Cow::Owned(key), Cow::Owned(val)));
    }
    items
}

/* ------------------------------ Tests ---------------------------------- */

/// `Utf8CaseFold` should order case-insensitively (with tie-breaks on
/// original bytes) and round-trip back to the original strings.
#[test]
fn test_utf8_order_roundtrip() {
    // Set up a fresh in-memory ColumnStore.
    let (store, fid) = setup_store();

    // Any field id is fine for this test.
    let words = [
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

    // Prepare a single Put batch: (key=row-id BE, val=encoded).
    #[allow(clippy::type_complexity)] // TODO: Alias complex type
    let items = build_items::<Utf8CaseFold>(&words);
    append_one_put(&store, fid, items);

    // Value-ordered scan (ascending)
    let frames = scan_value_frames(&store, fid);

    // Pull back originals (not the folded key)
    let mut got: Vec<String> = Vec::new();
    for bytes in frames {
        got.push(Utf8CaseFold::decode(bytes.as_slice()).expect("decoding must succeed"));
    }

    // Expected order (case-insensitive collation), tie-broken by original
    // bytes:
    // alpha < Apple (because fold(alpha)<fold(apple)), Banana < banana
    // (original bytes tie-break) LARGE words < Large Words < lower words
    // (all grouped under "l...")
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

/// `Bytes` should order by raw bytewise lex of the leading payload and
/// round-trip losslessly.
#[test]
fn test_bytes_order_roundtrip() {
    // Set up a fresh in-memory ColumnStore.
    let (store, fid) = setup_store();

    // Same set of blobs, but deliberately shuffled so they aren't already
    // in sorted order.
    let blobs: [&[u8]; 7] = [
        b"zed",
        b"abd",
        b"\x00",
        b"abc",
        b"\x00\xff",
        b"ab",
        b"\x00\x01",
    ];

    // Prepare a single Put batch: (key=row-id BE, val=encoded).
    #[allow(clippy::type_complexity)] // TODO: Alias complex type
    let items = build_items::<Bytes>(&blobs);
    append_one_put(&store, fid, items);

    // Value-ordered scan (ascending)
    let frames = scan_value_frames(&store, fid);

    // Decode and collect owned copies to compare.
    let mut got: Vec<Vec<u8>> = Vec::new();
    for bytes in frames {
        let b = Bytes::decode(bytes.as_slice()).expect("decoding must succeed");
        got.push(b);
    }

    // Explicit expected order: bytewise lex of the payloads.
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
