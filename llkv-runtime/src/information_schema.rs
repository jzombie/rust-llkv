use std::cmp::Ordering;
use std::sync::{Arc, RwLock};

use arrow::array::{ArrayRef, BooleanArray, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_plan::{CreateTablePlan, CreateTableSource, DropTablePlan};
use llkv_result::{Error, Result};
use llkv_storage::pager::BoxedPager;
use llkv_table::resolvers::QualifiedTableName;
use llkv_table::{
    CatalogDdl, ConstraintId, ConstraintKind, FieldId, ForeignKeyAction as CatalogForeignKeyAction,
    TableConstraintSummaryView, TableId,
};
use rustc_hash::FxHashMap;

use crate::{
    RuntimeContext, RuntimeNamespaceId, RuntimeStorageNamespaceRegistry, canonical_table_name,
};

#[derive(Clone)]
pub(crate) struct InformationSchemaTableData {
    pub name: &'static str,
    pub schema: Arc<Schema>,
    pub batches: Vec<RecordBatch>,
}

impl InformationSchemaTableData {
    fn new(name: &'static str, schema: Arc<Schema>, batch: RecordBatch) -> Self {
        Self {
            name,
            schema,
            batches: vec![batch],
        }
    }
}

pub(crate) fn build_information_schema_tables(
    context: &RuntimeContext<BoxedPager>,
) -> Result<Vec<InformationSchemaTableData>> {
    let tables = build_tables_table(context)?;
    let columns = build_columns_table(context)?;
    let cache = collect_information_schema_cache(context)?;
    let table_constraints = build_table_constraints_table(&cache)?;
    let key_column_usage = build_key_column_usage_table(&cache)?;
    let constraint_column_usage = build_constraint_column_usage_table(&cache)?;
    let referential_constraints = build_referential_constraints_table(&cache)?;

    Ok(vec![
        tables,
        columns,
        table_constraints,
        key_column_usage,
        constraint_column_usage,
        referential_constraints,
    ])
}

/// Rebuilds every `information_schema.*` table from catalog metadata alone.
///
/// This helper performs three high-level steps:
/// 1. Snapshot table/constraint metadata directly from the catalog (no user data scan).
/// 2. Materialize Arrow batches entirely in memory to represent each system table.
/// 3. Drop and recreate the `information_schema.*` tables inside the dedicated
///    `information_schema` runtime namespace using `CreateTableSource::Batches`, which
///    keeps all writes on a MemPager-backed heap.
///
/// The recreated tables stay queryable through regular SQL because the namespace
/// registers its objects with the shared catalog, but their pages and metadata live in a
/// transient pager arena that never touches persistent storage. The refresh is safe to
/// call from tooling like `llkv-tpch install`—it cannot dirty user tables, advance WAL
/// state, or trigger full scans on the primary pager.
pub(crate) fn refresh_information_schema(
    source_context: &Arc<RuntimeContext<BoxedPager>>,
    target_context: &Arc<RuntimeContext<BoxedPager>>,
    registry: &Arc<RwLock<RuntimeStorageNamespaceRegistry>>,
    namespace_id: &RuntimeNamespaceId,
) -> Result<()> {
    let tables = build_information_schema_tables(source_context.as_ref())?;
    recreate_information_schema_tables(
        source_context.as_ref(),
        target_context.as_ref(),
        registry,
        namespace_id,
        tables,
    )?;

    // Run a second pass for the metadata tables once the rest of the
    // information_schema objects exist so they list themselves and stay discoverable.
    let metadata_tables = vec![
        build_tables_table(source_context.as_ref())?,
        build_columns_table(source_context.as_ref())?,
    ];
    recreate_information_schema_tables(
        source_context.as_ref(),
        target_context.as_ref(),
        registry,
        namespace_id,
        metadata_tables,
    )
}

fn recreate_information_schema_tables(
    source_context: &RuntimeContext<BoxedPager>,
    target_context: &RuntimeContext<BoxedPager>,
    registry: &Arc<RwLock<RuntimeStorageNamespaceRegistry>>,
    namespace_id: &RuntimeNamespaceId,
    tables: Vec<InformationSchemaTableData>,
) -> Result<()> {
    for table in tables {
        let (display_name, canonical_name) = canonical_table_name(table.name)?;
        if !std::ptr::eq(source_context, target_context) {
            drop_table_if_registered(
                source_context,
                display_name.as_str(),
                canonical_name.as_str(),
                true,
            )?;
        }
        drop_table_if_registered(
            target_context,
            display_name.as_str(),
            canonical_name.as_str(),
            false,
        )?;
        {
            let mut guard = registry.write().expect("namespace registry poisoned");
            guard.unregister_table(&canonical_name);
        }

        let mut plan = CreateTablePlan::new(display_name.clone());
        plan.or_replace = true;
        plan.source = Some(CreateTableSource::Batches {
            schema: Arc::clone(&table.schema),
            batches: table.batches.clone(),
        });
        plan.columns.clear();
        CatalogDdl::create_table(target_context, plan)?;
        {
            let mut guard = registry.write().expect("namespace registry poisoned");
            guard.register_table(namespace_id, canonical_name.clone());
        }
    }

    Ok(())
}

fn drop_table_if_registered(
    context: &RuntimeContext<BoxedPager>,
    display_name: &str,
    canonical_name: &str,
    require_lookup: bool,
) -> Result<()> {
    if context.catalog().table_id(canonical_name).is_none() {
        return Ok(());
    }

    if require_lookup && context.lookup_table(canonical_name).is_err() {
        return Ok(());
    }

    let plan = DropTablePlan::new(display_name.to_string()).if_exists(true);
    CatalogDdl::drop_table(context, plan)
}

fn build_tables_table(context: &RuntimeContext<BoxedPager>) -> Result<InformationSchemaTableData> {
    let catalog = context.catalog();
    let mut table_names = catalog.table_names();
    table_names.sort_by_key(|name| name.to_ascii_lowercase());

    let mut schema_values = Vec::with_capacity(table_names.len());
    let mut table_values = Vec::with_capacity(table_names.len());
    let mut type_values = Vec::with_capacity(table_names.len());

    for name in table_names {
        let qualified = QualifiedTableName::from(name.as_str());
        schema_values.push(qualified.schema().map(|s| s.to_string()));
        table_values.push(Some(qualified.table().to_string()));
        type_values.push(Some("BASE TABLE".to_string()));
    }

    let fields = vec![
        Field::new("table_schema", DataType::Utf8, true),
        Field::new("table_name", DataType::Utf8, false),
        Field::new("table_type", DataType::Utf8, false),
    ];
    let schema = Arc::new(Schema::new(fields));
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(schema_values)) as ArrayRef,
        Arc::new(StringArray::from(table_values)) as ArrayRef,
        Arc::new(StringArray::from(type_values)) as ArrayRef,
    ];
    let batch = record_batch(schema.clone(), arrays)?;
    Ok(InformationSchemaTableData::new(
        "information_schema.tables",
        schema,
        batch,
    ))
}

fn build_columns_table(context: &RuntimeContext<BoxedPager>) -> Result<InformationSchemaTableData> {
    let catalog = context.catalog();
    let mut table_names = catalog.table_names();
    table_names.sort_by_key(|name| name.to_ascii_lowercase());

    let mut table_schema_values = Vec::new();
    let mut table_name_values = Vec::new();
    let mut column_name_values = Vec::new();
    let mut ordinal_values = Vec::new();
    let mut data_type_values = Vec::new();
    let mut nullable_values = Vec::new();
    let mut primary_key_values = Vec::new();
    let mut unique_values = Vec::new();
    let mut check_expression_values = Vec::new();

    for name in table_names {
        let qualified = QualifiedTableName::from(name.as_str());
        let schema = qualified.schema().map(|s| s.to_string());
        let table = qualified.table().to_string();
        let (_, canonical) = canonical_table_name(name.as_str())?;
        let columns = catalog.table_column_specs(&canonical)?;
        for (idx, col) in columns.iter().enumerate() {
            table_schema_values.push(schema.clone());
            table_name_values.push(Some(table.clone()));
            column_name_values.push(Some(col.name.clone()));
            ordinal_values.push(Some((idx + 1) as i32));
            data_type_values.push(Some(col.data_type.to_string()));
            nullable_values.push(Some(col.nullable));
            primary_key_values.push(Some(col.primary_key));
            unique_values.push(Some(col.unique));
            check_expression_values.push(col.check_expr.clone());
        }
    }

    let fields = vec![
        Field::new("table_schema", DataType::Utf8, true),
        Field::new("table_name", DataType::Utf8, false),
        Field::new("column_name", DataType::Utf8, false),
        Field::new("ordinal_position", DataType::Int32, false),
        Field::new("data_type", DataType::Utf8, false),
        Field::new("is_nullable", DataType::Boolean, false),
        Field::new("is_primary_key", DataType::Boolean, false),
        Field::new("is_unique", DataType::Boolean, false),
        Field::new("check_expression", DataType::Utf8, true),
    ];
    let schema = Arc::new(Schema::new(fields));
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(table_schema_values)) as ArrayRef,
        Arc::new(StringArray::from(table_name_values)) as ArrayRef,
        Arc::new(StringArray::from(column_name_values)) as ArrayRef,
        Arc::new(Int32Array::from(ordinal_values)) as ArrayRef,
        Arc::new(StringArray::from(data_type_values)) as ArrayRef,
        Arc::new(BooleanArray::from(nullable_values)) as ArrayRef,
        Arc::new(BooleanArray::from(primary_key_values)) as ArrayRef,
        Arc::new(BooleanArray::from(unique_values)) as ArrayRef,
        Arc::new(StringArray::from(check_expression_values)) as ArrayRef,
    ];
    let batch = record_batch(schema.clone(), arrays)?;
    Ok(InformationSchemaTableData::new(
        "information_schema.columns",
        schema,
        batch,
    ))
}

fn build_table_constraints_table(
    cache: &InformationSchemaCache,
) -> Result<InformationSchemaTableData> {
    let mut constraint_catalog: Vec<Option<String>> = Vec::new();
    let mut constraint_schema: Vec<Option<String>> = Vec::new();
    let mut constraint_name: Vec<Option<String>> = Vec::new();
    let mut table_schema: Vec<Option<String>> = Vec::new();
    let mut table_name: Vec<Option<String>> = Vec::new();
    let mut constraint_type: Vec<Option<String>> = Vec::new();
    let mut is_deferrable: Vec<Option<String>> = Vec::new();
    let mut initially_deferred: Vec<Option<String>> = Vec::new();
    let mut enforced: Vec<Option<String>> = Vec::new();

    for snapshot in &cache.tables {
        for constraint in &snapshot.constraints {
            constraint_catalog.push(None);
            constraint_schema.push(snapshot.schema.clone());
            constraint_name.push(Some(constraint.constraint_name.clone()));
            table_schema.push(snapshot.schema.clone());
            table_name.push(Some(snapshot.table_name.clone()));
            constraint_type.push(Some(constraint.constraint_type.label().into()));
            is_deferrable.push(Some("NO".to_string()));
            initially_deferred.push(Some("NO".to_string()));
            enforced.push(Some("YES".to_string()));
        }
    }

    let fields = vec![
        Field::new("constraint_catalog", DataType::Utf8, true),
        Field::new("constraint_schema", DataType::Utf8, true),
        Field::new("constraint_name", DataType::Utf8, false),
        Field::new("table_schema", DataType::Utf8, true),
        Field::new("table_name", DataType::Utf8, false),
        Field::new("constraint_type", DataType::Utf8, false),
        Field::new("is_deferrable", DataType::Utf8, false),
        Field::new("initially_deferred", DataType::Utf8, false),
        Field::new("enforced", DataType::Utf8, false),
    ];
    let schema = Arc::new(Schema::new(fields));
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(constraint_catalog)) as ArrayRef,
        Arc::new(StringArray::from(constraint_schema)) as ArrayRef,
        Arc::new(StringArray::from(constraint_name)) as ArrayRef,
        Arc::new(StringArray::from(table_schema)) as ArrayRef,
        Arc::new(StringArray::from(table_name)) as ArrayRef,
        Arc::new(StringArray::from(constraint_type)) as ArrayRef,
        Arc::new(StringArray::from(is_deferrable)) as ArrayRef,
        Arc::new(StringArray::from(initially_deferred)) as ArrayRef,
        Arc::new(StringArray::from(enforced)) as ArrayRef,
    ];
    let batch = record_batch(schema.clone(), arrays)?;
    Ok(InformationSchemaTableData::new(
        "information_schema.table_constraints",
        schema,
        batch,
    ))
}

fn build_key_column_usage_table(
    cache: &InformationSchemaCache,
) -> Result<InformationSchemaTableData> {
    let mut constraint_catalog: Vec<Option<String>> = Vec::new();
    let mut constraint_schema: Vec<Option<String>> = Vec::new();
    let mut constraint_name: Vec<Option<String>> = Vec::new();
    let mut table_schema: Vec<Option<String>> = Vec::new();
    let mut table_name: Vec<Option<String>> = Vec::new();
    let mut column_name: Vec<Option<String>> = Vec::new();
    let mut ordinal_position: Vec<Option<i32>> = Vec::new();
    let mut position_in_unique: Vec<Option<i32>> = Vec::new();

    for snapshot in &cache.tables {
        for constraint in &snapshot.constraints {
            if !constraint.constraint_type.is_key_usage_member() {
                continue;
            }
            for (idx, field_id) in constraint.column_ids.iter().enumerate() {
                let column = snapshot.column_name(*field_id);
                constraint_catalog.push(None);
                constraint_schema.push(snapshot.schema.clone());
                constraint_name.push(Some(constraint.constraint_name.clone()));
                table_schema.push(snapshot.schema.clone());
                table_name.push(Some(snapshot.table_name.clone()));
                column_name.push(Some(column));
                ordinal_position.push(Some((idx + 1) as i32));
                if constraint.constraint_type == InformationSchemaConstraintType::ForeignKey {
                    position_in_unique.push(Some((idx + 1) as i32));
                } else {
                    position_in_unique.push(None);
                }
            }
        }
    }

    let fields = vec![
        Field::new("constraint_catalog", DataType::Utf8, true),
        Field::new("constraint_schema", DataType::Utf8, true),
        Field::new("constraint_name", DataType::Utf8, false),
        Field::new("table_schema", DataType::Utf8, true),
        Field::new("table_name", DataType::Utf8, false),
        Field::new("column_name", DataType::Utf8, false),
        Field::new("ordinal_position", DataType::Int32, false),
        Field::new("position_in_unique_constraint", DataType::Int32, true),
    ];
    let schema = Arc::new(Schema::new(fields));
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(constraint_catalog)) as ArrayRef,
        Arc::new(StringArray::from(constraint_schema)) as ArrayRef,
        Arc::new(StringArray::from(constraint_name)) as ArrayRef,
        Arc::new(StringArray::from(table_schema)) as ArrayRef,
        Arc::new(StringArray::from(table_name)) as ArrayRef,
        Arc::new(StringArray::from(column_name)) as ArrayRef,
        Arc::new(Int32Array::from(ordinal_position)) as ArrayRef,
        Arc::new(Int32Array::from(position_in_unique)) as ArrayRef,
    ];
    let batch = record_batch(schema.clone(), arrays)?;
    Ok(InformationSchemaTableData::new(
        "information_schema.key_column_usage",
        schema,
        batch,
    ))
}

fn build_constraint_column_usage_table(
    cache: &InformationSchemaCache,
) -> Result<InformationSchemaTableData> {
    let mut constraint_catalog: Vec<Option<String>> = Vec::new();
    let mut constraint_schema: Vec<Option<String>> = Vec::new();
    let mut constraint_name: Vec<Option<String>> = Vec::new();
    let mut table_schema: Vec<Option<String>> = Vec::new();
    let mut table_name: Vec<Option<String>> = Vec::new();
    let mut column_name: Vec<Option<String>> = Vec::new();

    for snapshot in &cache.tables {
        for constraint in &snapshot.constraints {
            if !constraint.constraint_type.is_unique_like() {
                continue;
            }
            for field_id in &constraint.column_ids {
                let column = snapshot.column_name(*field_id);
                constraint_catalog.push(None);
                constraint_schema.push(snapshot.schema.clone());
                constraint_name.push(Some(constraint.constraint_name.clone()));
                table_schema.push(snapshot.schema.clone());
                table_name.push(Some(snapshot.table_name.clone()));
                column_name.push(Some(column));
            }
        }
    }

    let fields = vec![
        Field::new("constraint_catalog", DataType::Utf8, true),
        Field::new("constraint_schema", DataType::Utf8, true),
        Field::new("constraint_name", DataType::Utf8, false),
        Field::new("table_schema", DataType::Utf8, true),
        Field::new("table_name", DataType::Utf8, false),
        Field::new("column_name", DataType::Utf8, false),
    ];
    let schema = Arc::new(Schema::new(fields));
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(constraint_catalog)) as ArrayRef,
        Arc::new(StringArray::from(constraint_schema)) as ArrayRef,
        Arc::new(StringArray::from(constraint_name)) as ArrayRef,
        Arc::new(StringArray::from(table_schema)) as ArrayRef,
        Arc::new(StringArray::from(table_name)) as ArrayRef,
        Arc::new(StringArray::from(column_name)) as ArrayRef,
    ];
    let batch = record_batch(schema.clone(), arrays)?;
    Ok(InformationSchemaTableData::new(
        "information_schema.constraint_column_usage",
        schema,
        batch,
    ))
}

fn build_referential_constraints_table(
    cache: &InformationSchemaCache,
) -> Result<InformationSchemaTableData> {
    let mut constraint_catalog: Vec<Option<String>> = Vec::new();
    let mut constraint_schema: Vec<Option<String>> = Vec::new();
    let mut constraint_name: Vec<Option<String>> = Vec::new();
    let mut unique_constraint_catalog: Vec<Option<String>> = Vec::new();
    let mut unique_constraint_schema: Vec<Option<String>> = Vec::new();
    let mut unique_constraint_name: Vec<Option<String>> = Vec::new();
    let mut match_option: Vec<Option<String>> = Vec::new();
    let mut update_rule: Vec<Option<String>> = Vec::new();
    let mut delete_rule: Vec<Option<String>> = Vec::new();
    let mut is_deferrable: Vec<Option<String>> = Vec::new();
    let mut initially_deferred: Vec<Option<String>> = Vec::new();

    for snapshot in &cache.tables {
        for constraint in &snapshot.constraints {
            if constraint.constraint_type != InformationSchemaConstraintType::ForeignKey {
                continue;
            }
            let Some(referenced_table_id) = constraint.referenced_table_id else {
                continue;
            };
            let Some(unique_ref) =
                cache.lookup_unique(referenced_table_id, &constraint.referenced_column_ids)
            else {
                continue;
            };
            constraint_catalog.push(None);
            constraint_schema.push(snapshot.schema.clone());
            constraint_name.push(Some(constraint.constraint_name.clone()));
            unique_constraint_catalog.push(None);
            unique_constraint_schema.push(unique_ref.schema.clone());
            unique_constraint_name.push(Some(unique_ref.constraint_name.clone()));
            match_option.push(Some("SIMPLE".to_string()));
            let update_text = constraint
                .on_update
                .map(format_fk_action)
                .unwrap_or("NO ACTION")
                .to_string();
            update_rule.push(Some(update_text));
            let delete_text = constraint
                .on_delete
                .map(format_fk_action)
                .unwrap_or("NO ACTION")
                .to_string();
            delete_rule.push(Some(delete_text));
            is_deferrable.push(Some("NO".to_string()));
            initially_deferred.push(Some("NO".to_string()));
        }
    }

    let fields = vec![
        Field::new("constraint_catalog", DataType::Utf8, true),
        Field::new("constraint_schema", DataType::Utf8, true),
        Field::new("constraint_name", DataType::Utf8, false),
        Field::new("unique_constraint_catalog", DataType::Utf8, true),
        Field::new("unique_constraint_schema", DataType::Utf8, true),
        Field::new("unique_constraint_name", DataType::Utf8, false),
        Field::new("match_option", DataType::Utf8, false),
        Field::new("update_rule", DataType::Utf8, false),
        Field::new("delete_rule", DataType::Utf8, false),
        Field::new("is_deferrable", DataType::Utf8, false),
        Field::new("initially_deferred", DataType::Utf8, false),
    ];
    let schema = Arc::new(Schema::new(fields));
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(constraint_catalog)) as ArrayRef,
        Arc::new(StringArray::from(constraint_schema)) as ArrayRef,
        Arc::new(StringArray::from(constraint_name)) as ArrayRef,
        Arc::new(StringArray::from(unique_constraint_catalog)) as ArrayRef,
        Arc::new(StringArray::from(unique_constraint_schema)) as ArrayRef,
        Arc::new(StringArray::from(unique_constraint_name)) as ArrayRef,
        Arc::new(StringArray::from(match_option)) as ArrayRef,
        Arc::new(StringArray::from(update_rule)) as ArrayRef,
        Arc::new(StringArray::from(delete_rule)) as ArrayRef,
        Arc::new(StringArray::from(is_deferrable)) as ArrayRef,
        Arc::new(StringArray::from(initially_deferred)) as ArrayRef,
    ];
    let batch = record_batch(schema.clone(), arrays)?;
    Ok(InformationSchemaTableData::new(
        "information_schema.referential_constraints",
        schema,
        batch,
    ))
}

fn record_batch(schema: Arc<Schema>, arrays: Vec<ArrayRef>) -> Result<RecordBatch> {
    RecordBatch::try_new(schema, arrays)
        .map_err(|err| Error::Internal(format!("failed to build information_schema batch: {err}")))
}

fn collect_information_schema_cache(
    context: &RuntimeContext<BoxedPager>,
) -> Result<InformationSchemaCache> {
    let catalog = context.catalog();
    let mut table_names = catalog.table_names();
    table_names.sort_by_key(|name| name.to_ascii_lowercase());

    let mut tables = Vec::new();
    for name in table_names {
        let qualified = QualifiedTableName::from(name.as_str());
        let schema = qualified.schema().map(|s| s.to_string());
        let table_name = qualified.table().to_string();
        let (_, canonical) = canonical_table_name(name.as_str())?;
        let Some(table_id) = catalog.table_id(&canonical) else {
            continue;
        };

        let TableConstraintSummaryView {
            table_meta: _,
            column_metas,
            constraint_records,
            multi_column_uniques,
            constraint_names,
        } = catalog.table_constraint_summary(&canonical)?;

        let mut column_names = FxHashMap::default();
        for meta in column_metas.into_iter().flatten() {
            let name = meta
                .name
                .clone()
                .unwrap_or_else(|| default_column_name(meta.col_id));
            column_names.insert(meta.col_id, name);
        }

        let mut constraints = Vec::new();
        for record in constraint_records
            .into_iter()
            .filter(|record| record.is_active())
        {
            let constraint_id = Some(record.constraint_id);
            match record.kind {
                ConstraintKind::PrimaryKey(payload) => {
                    let name = constraint_name_or_fallback(
                        constraint_names.get(&record.constraint_id),
                        &table_name,
                        InformationSchemaConstraintType::PrimaryKey,
                        constraint_id,
                    );
                    constraints.push(InformationSchemaConstraint {
                        constraint_id,
                        constraint_name: name,
                        constraint_type: InformationSchemaConstraintType::PrimaryKey,
                        column_ids: payload.field_ids,
                        referenced_table_id: None,
                        referenced_column_ids: Vec::new(),
                        on_delete: None,
                        on_update: None,
                    });
                }
                ConstraintKind::Unique(payload) => {
                    let name = constraint_name_or_fallback(
                        constraint_names.get(&record.constraint_id),
                        &table_name,
                        InformationSchemaConstraintType::Unique,
                        constraint_id,
                    );
                    constraints.push(InformationSchemaConstraint {
                        constraint_id,
                        constraint_name: name,
                        constraint_type: InformationSchemaConstraintType::Unique,
                        column_ids: payload.field_ids,
                        referenced_table_id: None,
                        referenced_column_ids: Vec::new(),
                        on_delete: None,
                        on_update: None,
                    });
                }
                ConstraintKind::ForeignKey(payload) => {
                    let name = constraint_name_or_fallback(
                        constraint_names.get(&record.constraint_id),
                        &table_name,
                        InformationSchemaConstraintType::ForeignKey,
                        constraint_id,
                    );
                    constraints.push(InformationSchemaConstraint {
                        constraint_id,
                        constraint_name: name,
                        constraint_type: InformationSchemaConstraintType::ForeignKey,
                        column_ids: payload.referencing_field_ids,
                        referenced_table_id: Some(payload.referenced_table),
                        referenced_column_ids: payload.referenced_field_ids,
                        on_delete: Some(payload.on_delete),
                        on_update: Some(payload.on_update),
                    });
                }
                ConstraintKind::Check(payload) => {
                    let name = constraint_name_or_fallback(
                        constraint_names.get(&record.constraint_id),
                        &table_name,
                        InformationSchemaConstraintType::Check,
                        constraint_id,
                    );
                    constraints.push(InformationSchemaConstraint {
                        constraint_id,
                        constraint_name: name,
                        constraint_type: InformationSchemaConstraintType::Check,
                        column_ids: payload.field_ids,
                        referenced_table_id: None,
                        referenced_column_ids: Vec::new(),
                        on_delete: None,
                        on_update: None,
                    });
                }
            }
        }

        for unique in multi_column_uniques
            .into_iter()
            .filter(|entry| entry.unique)
        {
            let name = unique
                .index_name
                .clone()
                .unwrap_or_else(|| unique.canonical_name.clone());
            constraints.push(InformationSchemaConstraint {
                constraint_id: None,
                constraint_name: name,
                constraint_type: InformationSchemaConstraintType::Unique,
                column_ids: unique.column_ids,
                referenced_table_id: None,
                referenced_column_ids: Vec::new(),
                on_delete: None,
                on_update: None,
            });
        }

        sort_information_schema_constraints(&mut constraints);

        tables.push(InformationSchemaTableSnapshot {
            schema,
            table_name,
            table_id,
            column_names,
            constraints,
        });
    }

    let mut unique_lookup = FxHashMap::default();
    for snapshot in &tables {
        for constraint in snapshot
            .constraints
            .iter()
            .filter(|c| c.constraint_type.is_unique_like())
        {
            if constraint.column_ids.is_empty() {
                continue;
            }
            let key =
                InformationSchemaCache::constraint_key(snapshot.table_id, &constraint.column_ids);
            unique_lookup
                .entry(key)
                .or_insert_with(|| InformationSchemaUniqueRef {
                    constraint_name: constraint.constraint_name.clone(),
                    schema: snapshot.schema.clone(),
                });
        }
    }

    Ok(InformationSchemaCache {
        tables,
        unique_lookup,
    })
}

#[derive(Clone, Debug)]
struct InformationSchemaTableSnapshot {
    schema: Option<String>,
    table_name: String,
    table_id: TableId,
    column_names: FxHashMap<FieldId, String>,
    constraints: Vec<InformationSchemaConstraint>,
}

impl InformationSchemaTableSnapshot {
    fn column_name(&self, field_id: FieldId) -> String {
        self.column_names
            .get(&field_id)
            .cloned()
            .unwrap_or_else(|| default_column_name(field_id))
    }
}

#[derive(Clone, Debug)]
struct InformationSchemaConstraint {
    constraint_id: Option<ConstraintId>,
    constraint_name: String,
    constraint_type: InformationSchemaConstraintType,
    column_ids: Vec<FieldId>,
    referenced_table_id: Option<TableId>,
    referenced_column_ids: Vec<FieldId>,
    on_delete: Option<CatalogForeignKeyAction>,
    on_update: Option<CatalogForeignKeyAction>,
}

#[derive(Clone, Debug)]
struct InformationSchemaUniqueRef {
    constraint_name: String,
    schema: Option<String>,
}

struct InformationSchemaCache {
    tables: Vec<InformationSchemaTableSnapshot>,
    unique_lookup: FxHashMap<(TableId, Vec<FieldId>), InformationSchemaUniqueRef>,
}

impl InformationSchemaCache {
    fn constraint_key(table_id: TableId, columns: &[FieldId]) -> (TableId, Vec<FieldId>) {
        normalized_constraint_key(table_id, columns)
    }

    fn lookup_unique(
        &self,
        table_id: TableId,
        columns: &[FieldId],
    ) -> Option<&InformationSchemaUniqueRef> {
        let key = Self::constraint_key(table_id, columns);
        self.unique_lookup.get(&key)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InformationSchemaConstraintType {
    PrimaryKey,
    Unique,
    ForeignKey,
    Check,
}

impl InformationSchemaConstraintType {
    fn label(self) -> &'static str {
        match self {
            InformationSchemaConstraintType::PrimaryKey => "PRIMARY KEY",
            InformationSchemaConstraintType::Unique => "UNIQUE",
            InformationSchemaConstraintType::ForeignKey => "FOREIGN KEY",
            InformationSchemaConstraintType::Check => "CHECK",
        }
    }

    fn slug(self) -> &'static str {
        match self {
            InformationSchemaConstraintType::PrimaryKey => "primary_key",
            InformationSchemaConstraintType::Unique => "unique",
            InformationSchemaConstraintType::ForeignKey => "foreign_key",
            InformationSchemaConstraintType::Check => "check",
        }
    }

    fn is_unique_like(self) -> bool {
        matches!(
            self,
            InformationSchemaConstraintType::PrimaryKey | InformationSchemaConstraintType::Unique
        )
    }

    fn is_key_usage_member(self) -> bool {
        matches!(
            self,
            InformationSchemaConstraintType::PrimaryKey
                | InformationSchemaConstraintType::Unique
                | InformationSchemaConstraintType::ForeignKey
        )
    }
}

fn constraint_name_or_fallback(
    stored: Option<&String>,
    table_name: &str,
    constraint_type: InformationSchemaConstraintType,
    constraint_id: Option<ConstraintId>,
) -> String {
    if let Some(name) = stored {
        if !name.trim().is_empty() {
            return name.clone();
        }
    }

    match constraint_id {
        Some(id) => format!("{}_{}_{}", table_name, constraint_type.slug(), id),
        None => format!("{}_{}_auto", table_name, constraint_type.slug()),
    }
}

fn sort_information_schema_constraints(constraints: &mut [InformationSchemaConstraint]) {
    constraints.sort_by(|a, b| {
        let name_cmp = a
            .constraint_name
            .to_ascii_lowercase()
            .cmp(&b.constraint_name.to_ascii_lowercase());
        if name_cmp != Ordering::Equal {
            return name_cmp;
        }
        match (a.constraint_id, b.constraint_id) {
            (Some(left), Some(right)) => left.cmp(&right),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        }
    });
}

fn default_column_name(field_id: FieldId) -> String {
    format!("col_{}", field_id)
}

fn format_fk_action(action: CatalogForeignKeyAction) -> &'static str {
    match action {
        CatalogForeignKeyAction::NoAction => "NO ACTION",
        CatalogForeignKeyAction::Restrict => "RESTRICT",
    }
}

fn normalized_constraint_key(table_id: TableId, columns: &[FieldId]) -> (TableId, Vec<FieldId>) {
    let mut normalized = columns.to_vec();
    normalized.sort_unstable();
    (table_id, normalized)
}
