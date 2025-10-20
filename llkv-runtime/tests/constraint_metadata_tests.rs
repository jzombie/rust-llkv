use std::sync::Arc;

use arrow::datatypes::DataType;
use llkv_column_map::store::ColumnStore;
use llkv_runtime::{
    ColumnSpec, CreateTablePlan, ForeignKeyAction, ForeignKeySpec, RuntimeContext,
    RuntimeStatementResult,
};
use llkv_storage::pager::MemPager;
use llkv_table::{ConstraintKind, MetadataManager, PrimaryKeyConstraint, Table, UniqueConstraint};

#[test]
fn primary_key_and_unique_constraints_reload_from_metadata() {
    let pager = Arc::new(MemPager::default());

    {
        let context = Arc::new(RuntimeContext::new(Arc::clone(&pager)));

        let result = context
            .create_table_builder("accounts")
            .with_column_spec(ColumnSpec::new("id", DataType::Int64, false).with_primary_key(true))
            .with_column_spec(ColumnSpec::new("email", DataType::Utf8, false).with_unique(true))
            .finish()
            .expect("create table");
        assert!(matches!(result, RuntimeStatementResult::CreateTable { .. }));

        let specs = context.table_column_specs("accounts").expect("table specs");
        assert_eq!(specs.len(), 2);
        let id_spec = specs.iter().find(|spec| spec.name == "id").unwrap();
        assert!(id_spec.primary_key);
        assert!(id_spec.unique);

        let email_spec = specs.iter().find(|spec| spec.name == "email").unwrap();
        assert!(!email_spec.primary_key);
        assert!(email_spec.unique);
        assert_accounts_constraints(&pager);
    }

    let context = Arc::new(RuntimeContext::new(Arc::clone(&pager)));
    let names = context.table_names();
    assert!(
        names.iter().any(|name| name == "accounts"),
        "expected accounts table, got {:?}",
        names
    );
    let specs_result = context.table_column_specs("accounts");
    assert_accounts_constraints(&pager);
    let specs = specs_result.unwrap_or_else(|err| panic!("table specs after restart: {:?}", err));
    assert_eq!(specs.len(), 2);
    let id_spec = specs.iter().find(|spec| spec.name == "id").unwrap();
    assert!(id_spec.primary_key);
    assert!(id_spec.unique);
    let email_spec = specs.iter().find(|spec| spec.name == "email").unwrap();
    assert!(!email_spec.primary_key);
    assert!(email_spec.unique);
}

#[test]
fn foreign_key_views_reload_from_metadata() {
    let pager = Arc::new(MemPager::default());

    {
        let context = Arc::new(RuntimeContext::new(Arc::clone(&pager)));

        context
            .create_table_builder("parents")
            .with_column_spec(ColumnSpec::new("id", DataType::Int64, false).with_primary_key(true))
            .finish()
            .expect("create parents table");

        let mut plan = CreateTablePlan::new("children");
        plan.columns
            .push(ColumnSpec::new("id", DataType::Int64, false).with_primary_key(true));
        plan.columns
            .push(ColumnSpec::new("parent_id", DataType::Int64, true));
        plan.foreign_keys.push(ForeignKeySpec {
            name: Some("fk_children_parent".into()),
            columns: vec!["parent_id".into()],
            referenced_table: "parents".into(),
            referenced_columns: vec!["id".into()],
            on_delete: ForeignKeyAction::Restrict,
            on_update: ForeignKeyAction::Restrict,
        });

        match context
            .create_table_plan(plan)
            .expect("create children table with foreign key")
        {
            RuntimeStatementResult::CreateTable { .. } => {}
            other => panic!("expected CreateTable result, got {:?}", other),
        }
    }

    let context = Arc::new(RuntimeContext::new(Arc::clone(&pager)));
    let views = context
        .foreign_key_views("children")
        .expect("load foreign key views");

    assert_eq!(views.len(), 1);
    let view = &views[0];
    assert_eq!(view.referencing_table_display, "children");
    assert_eq!(view.referenced_table_display, "parents");
    assert_eq!(view.referencing_column_names, vec!["parent_id"]);
    assert_eq!(view.referenced_column_names, vec!["id"]);
}

fn assert_accounts_constraints(pager: &Arc<MemPager>) {
    let store = Arc::new(ColumnStore::open(Arc::clone(pager)).expect("open column store"));
    let metadata = MetadataManager::new(Arc::clone(&store));
    let tables = metadata.all_table_metas().expect("all table metas");
    let (table_id, _) = tables
        .into_iter()
        .find(|(_, meta)| meta.name.as_deref() == Some("accounts"))
        .expect("accounts table meta");
    let table =
        Table::from_id_and_store(table_id, Arc::clone(&store)).expect("open table succeeds");
    table.schema().expect("load schema succeeds");
    let records = metadata
        .constraint_records(table_id)
        .expect("constraint records");
    assert_eq!(records.len(), 2);

    assert!(records.iter().any(|record| {
        matches!(
            &record.kind,
            ConstraintKind::PrimaryKey(PrimaryKeyConstraint { field_ids })
            if field_ids == &[1]
        )
    }));
    assert!(records.iter().any(|record| {
        matches!(
            &record.kind,
            ConstraintKind::Unique(UniqueConstraint { field_ids })
            if field_ids == &[2]
        )
    }));
}
