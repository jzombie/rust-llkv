use std::borrow::Cow;
use std::ops::Bound;

use llkv_column_map::ColumnStore;
use llkv_column_map::column_store::read_scan::{Direction, OrderBy, ValueScanOpts};
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode};

use llkv_types::Utf8CaseFold;

#[test]
fn utf8_casefold_orders_case_insensitively_and_preserves_original() {
    // Set up a fresh in-memory ColumnStore.
    let pager = Box::leak(Box::new(MemPager::default()));
    let store = ColumnStore::init_empty(pager);

    // Any field id is fine for this test.
    let fid: LogicalFieldId = 42;

    // Intentionally mixed case and spacing; ensure “Large Words” and “lower words” sort together.
    let words = vec![
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
    let mut items: Vec<(Cow<[u8]>, Cow<[u8]>)> = Vec::new();
    for (i, s) in words.iter().enumerate() {
        let key = (i as u64 + 1).to_be_bytes().to_vec(); // lex==numeric
        let mut val = Vec::new();
        Utf8CaseFold::encode_into(&mut val, s);
        items.push((Cow::Owned(key), Cow::Owned(val)));
    }

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

    // Value-ordered scan (ascending)
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

    // Pull back originals (not the folded key)
    let mut got: Vec<String> = Vec::new();
    for item in it {
        let a = item.value.start() as usize;
        let b = item.value.end() as usize;
        let arc = item.value.data();
        let bytes = &arc.as_ref()[a..b];
        got.push(Utf8CaseFold::decode(bytes));
    }

    // Expected order (case-insensitive collation), tie-broken by original bytes:
    // alpha < Apple (because fold(alpha)<fold(apple)), Banana < banana (original bytes tie-break)
    // LARGE words < Large Words < lower words (all grouped under "l...")
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
