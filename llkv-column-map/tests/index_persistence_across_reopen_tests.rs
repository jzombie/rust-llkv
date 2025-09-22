use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::storage::pager::{BatchGet, GetResult, MemPager, Pager};
use llkv_column_map::store::ColumnStore;
use llkv_column_map::store::catalog::ColumnCatalog;
use llkv_column_map::store::descriptor::{ColumnDescriptor, DescriptorIterator};
use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanOptions,
};
use llkv_column_map::types::{LogicalFieldId, Namespace};

fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

// Minimal visitor that records that callbacks occurred (to ensure scan executed).
struct TouchVisitor {
    pub saw: std::cell::Cell<bool>,
}
impl PrimitiveVisitor for TouchVisitor {}
impl PrimitiveWithRowIdsVisitor for TouchVisitor {}
impl PrimitiveSortedVisitor for TouchVisitor {}
impl PrimitiveSortedWithRowIdsVisitor for TouchVisitor {
    fn u64_run_with_rids(&mut self, _v: &UInt64Array, _r: &UInt64Array, _s: usize, _l: usize) {
        self.saw.set(true);
    }
    fn null_run(&mut self, _r: &UInt64Array, _s: usize, _l: usize) {
        self.saw.set(true);
    }
}

#[test]
fn indices_persist_after_drop_and_reopen() {
    let pager = Arc::new(MemPager::new());

    let anchor_fid = fid(41);
    let target_fid = fid(42);

    // Scope 1: create store, seed data, create indices.
    {
        let store = ColumnStore::open(pager.clone()).unwrap();

        // Anchor column: values for all row_ids 0..100
        let mut md_a = HashMap::new();
        md_a.insert("field_id".to_string(), u64::from(anchor_fid).to_string());
        let schema_a = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("data", DataType::UInt64, false).with_metadata(md_a),
        ]));
        let r: Vec<u64> = (0..100).collect();
        let b_anchor = RecordBatch::try_new(
            schema_a,
            vec![
                Arc::new(UInt64Array::from(r.clone())),
                Arc::new(UInt64Array::from(r.clone())),
            ],
        )
        .unwrap();
        store.append(&b_anchor).unwrap();

        // Target column: values only on multiples of 3
        let mut md_t = HashMap::new();
        md_t.insert("field_id".to_string(), u64::from(target_fid).to_string());
        let schema_t = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("data", DataType::UInt64, false).with_metadata(md_t),
        ]));
        let t_r: Vec<u64> = (0..100).filter(|x| x % 3 == 0).collect();
        let t_v: Vec<u64> = t_r.iter().map(|x| x * 10).collect();
        let b_target = RecordBatch::try_new(
            schema_t,
            vec![
                Arc::new(UInt64Array::from(t_r)),
                Arc::new(UInt64Array::from(t_v)),
            ],
        )
        .unwrap();
        store.append(&b_target).unwrap();

        // Build sort index for target column.
        store.create_sort_index(target_fid).unwrap();

        // Presence index should be enabled by append: verify some membership.
        assert!(store.has_row_id(target_fid, 0).unwrap()); // present
        assert!(!store.has_row_id(target_fid, 1).unwrap()); // absent

        // Sorted scan should succeed now that sort perms exist.
        let mut v = TouchVisitor {
            saw: std::cell::Cell::new(false),
        };
        store
            .scan(
                target_fid,
                ScanOptions {
                    sorted: true,
                    reverse: false,
                    with_row_ids: true,
                    limit: Some(5),
                    offset: 0,
                    include_nulls: true,
                    nulls_first: true,
                    anchor_row_id_field: Some(anchor_fid.with_namespace(Namespace::RowIdShadow)),
                },
                &mut v,
            )
            .unwrap();
        assert!(v.saw.get());
    }

    // Scope 2: reopen the store with the same pager; indices should persist.
    {
        let store = ColumnStore::open(pager.clone()).unwrap();

        // Verify persisted indexes using the high-level API (no low-level digging).
        let idx = store.list_persisted_indexes(target_fid).unwrap();
        assert!(idx.iter().any(|n| n == "presence"));
        assert!(idx.iter().any(|n| n == "sort"));

        // TODO: Evaluate a column that has had different indexes applied

        // TODO: Evaluate a column that has not had any indexes applied

        // TODO: Evaluate a column that does not exist
    }
}

#[test]
fn index_can_be_removed_and_persists() {
    let pager = Arc::new(MemPager::new());
    let target_fid = fid(50);

    // Scope 1: Create store, seed data, create an index, then remove it.
    {
        let store = ColumnStore::open(pager.clone()).unwrap();

        // Seed some data for the target column
        let mut md_t = HashMap::new();
        md_t.insert("field_id".to_string(), u64::from(target_fid).to_string());
        let schema_t = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("data", DataType::UInt64, false).with_metadata(md_t),
        ]));
        let t_r: Vec<u64> = (0..50).collect();
        let b_target = RecordBatch::try_new(
            schema_t,
            vec![
                Arc::new(UInt64Array::from(t_r.clone())),
                Arc::new(UInt64Array::from(t_r)),
            ],
        )
        .unwrap();
        store.append(&b_target).unwrap();

        // 1. Create the sort index and verify it's there.
        store.create_sort_index(target_fid).unwrap();
        let idx_before = store.list_persisted_indexes(target_fid).unwrap();
        assert!(idx_before.contains(&"sort".to_string()));
        assert!(idx_before.contains(&"presence".to_string()));

        // 2. Unregister the sort index.
        store.unregister_index(target_fid, "sort").unwrap();

        // 3. Verify it's gone within the same session.
        let idx_after = store.list_persisted_indexes(target_fid).unwrap();
        assert!(!idx_after.contains(&"sort".to_string()));
        assert!(idx_after.contains(&"presence".to_string())); // Presence should remain
    }

    // Scope 2: Reopen the store and verify the removal persisted.
    {
        let store = ColumnStore::open(pager.clone()).unwrap();

        // 4. Verify the index is still gone after reopening.
        let idx_reopened = store.list_persisted_indexes(target_fid).unwrap();
        assert!(!idx_reopened.contains(&"sort".to_string()));
        assert!(idx_reopened.contains(&"presence".to_string()));
    }
}
