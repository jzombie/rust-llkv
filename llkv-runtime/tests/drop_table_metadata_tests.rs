use std::sync::Arc;

use arrow::datatypes::DataType;
use llkv_column_map::store::ColumnStore;
use llkv_result::Error;
use llkv_runtime::{ColumnSpec, RuntimeContext};
use llkv_storage::pager::MemPager;
use llkv_table::MetadataManager;
use llkv_table::constraints::{
    ConstraintKind, ConstraintRecord, ConstraintState, ForeignKeyAction, ForeignKeyConstraint,
};

#[test]
fn drop_table_removes_persisted_metadata() {
    let pager = Arc::new(MemPager::default());
    let context = Arc::new(RuntimeContext::new(Arc::clone(&pager)));

    context
        .create_table_builder("drop_test")
        .with_column_spec(ColumnSpec::new("id", DataType::Int64, false).with_primary_key(true))
        .with_column_spec(ColumnSpec::new("name", DataType::Utf8, true))
        .finish()
        .expect("create table");

    let table_id = context
        .table_catalog()
        .table_id("drop_test")
        .expect("table id registered");

    let store = Arc::new(ColumnStore::open(Arc::clone(&pager)).expect("open column store"));
    let metadata_view = MetadataManager::new(Arc::clone(&store));
    assert!(
        metadata_view
            .table_meta(table_id)
            .expect("load table meta")
            .is_some()
    );
    assert!(
        !metadata_view
            .constraint_records(table_id)
            .expect("constraint records before drop")
            .is_empty()
    );

    context
        .drop_table_immediate("drop_test", false)
        .expect("drop table succeeds");

    let metadata_after = MetadataManager::new(Arc::clone(&store));
    assert!(
        metadata_after
            .table_meta(table_id)
            .expect("table meta after drop")
            .is_none()
    );

    let constraint_records = metadata_after
        .constraint_records(table_id)
        .expect("constraint records after drop");
    assert!(constraint_records.is_empty());

    let column_metas = metadata_after
        .column_metas(table_id, &[1, 2])
        .expect("column metas after drop");
    assert!(column_metas.into_iter().all(|meta| meta.is_none()));
}

#[test]
fn drop_table_respects_foreign_keys_after_restart() {
    let pager = Arc::new(MemPager::default());

    {
        let context = Arc::new(RuntimeContext::new(Arc::clone(&pager)));

        context
            .create_table_builder("parents")
            .with_column_spec(ColumnSpec::new("id", DataType::Int64, false).with_primary_key(true))
            .finish()
            .expect("create parents");

        context
            .create_table_builder("children")
            .with_column_spec(ColumnSpec::new("id", DataType::Int64, false).with_primary_key(true))
            .with_column_spec(ColumnSpec::new("parent_id", DataType::Int64, true))
            .finish()
            .expect("create children");

        let catalog = context.table_catalog();
        let parent_table_id = catalog
            .table_id("parents")
            .expect("parent table id registered");
        let child_table_id = catalog
            .table_id("children")
            .expect("child table id registered");

        let store = Arc::new(ColumnStore::open(Arc::clone(&pager)).expect("open column store"));
        let metadata = MetadataManager::new(Arc::clone(&store));
        let record = ConstraintRecord {
            constraint_id: 1,
            kind: ConstraintKind::ForeignKey(ForeignKeyConstraint {
                referencing_field_ids: vec![2],
                referenced_table: parent_table_id,
                referenced_field_ids: vec![1],
                on_delete: ForeignKeyAction::Restrict,
                on_update: ForeignKeyAction::Restrict,
            }),
            state: ConstraintState::Active,
            revision: 1,
            last_modified_micros: 0,
        };
        metadata
            .put_constraint_records(child_table_id, &[record])
            .expect("persist foreign key record");
        metadata
            .flush_table(child_table_id)
            .expect("flush child metadata");
    }

    let context = RuntimeContext::new(Arc::clone(&pager));
    context
        .lookup_table("parents")
        .expect("load parent table before drop");
    let err = context
        .drop_table_immediate("parents", false)
        .expect_err("dropping referenced parent should fail");

    match err {
        Error::ConstraintError(message) | Error::CatalogError(message) => {
            assert!(
                message.to_ascii_lowercase().contains("children"),
                "error message should mention child table, got '{}'",
                message
            );
        }
        other => panic!("expected catalog or constraint error, got {:?}", other),
    }
}
