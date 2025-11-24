// TODO: Move portions of this to llkv-table?

use sqlparser::ast::{CreateTable, ObjectName, ObjectNamePart, TableConstraint};
use std::collections::HashSet;

/// Return the final identifier component of an [`ObjectName`] in uppercase form.
pub fn canonical_table_ident(name: &ObjectName) -> Option<String> {
    name.0.last().and_then(|part| match part {
        ObjectNamePart::Identifier(ident) => Some(ident.value.to_ascii_uppercase()),
        ObjectNamePart::Function(_) => None,
    })
}

/// Normalize `TableConstraint` instances so downstream consumers do not need to
/// worry about dialect-specific adornments (e.g., unnamed index metadata).
pub fn normalize_table_constraint(constraint: TableConstraint) -> TableConstraint {
    match constraint {
        TableConstraint::ForeignKey {
            mut name,
            mut index_name,
            columns,
            foreign_table,
            referred_columns,
            on_delete,
            on_update,
            characteristics,
        } => {
            if name.is_none() {
                name = index_name.clone();
            }
            index_name = None;
            TableConstraint::ForeignKey {
                name,
                index_name,
                columns,
                foreign_table,
                referred_columns,
                on_delete,
                on_update,
                characteristics,
            }
        }
        TableConstraint::PrimaryKey {
            name,
            index_name: _,
            index_type,
            columns,
            index_options,
            characteristics,
        } => TableConstraint::PrimaryKey {
            name,
            index_name: None,
            index_type,
            columns,
            index_options,
            characteristics,
        },
        TableConstraint::Unique {
            name,
            index_name: _,
            index_type,
            columns,
            index_options,
            nulls_distinct,
            characteristics,
            index_type_display,
        } => TableConstraint::Unique {
            name,
            index_name: None,
            index_type,
            columns,
            index_options,
            nulls_distinct,
            characteristics,
            index_type_display,
        },
        other => other,
    }
}

/// Order `CREATE TABLE` statements such that referenced tables appear before the
/// tables that depend on them. If the graph contains cycles, the remaining
/// statements are appended at the end preserving their original order.
pub fn order_create_tables_by_foreign_keys(tables: Vec<CreateTable>) -> Vec<CreateTable> {
    let mut remaining = tables;
    let mut ordered = Vec::with_capacity(remaining.len());
    if remaining.is_empty() {
        return ordered;
    }

    let table_names: HashSet<String> = remaining
        .iter()
        .filter_map(|table| canonical_table_ident(&table.name))
        .collect();

    let mut resolved: HashSet<String> = HashSet::new();

    while !remaining.is_empty() {
        let mut progress = false;
        let current_round = std::mem::take(&mut remaining);
        let mut next_round = Vec::new();

        for table in current_round.into_iter() {
            let table_name = canonical_table_ident(&table.name).unwrap_or_default();
            let deps = collect_foreign_key_dependencies(&table);
            let deps_satisfied = deps.into_iter().all(|dep| {
                dep.is_empty()
                    || dep == table_name
                    || !table_names.contains(&dep)
                    || resolved.contains(&dep)
            });

            if deps_satisfied {
                resolved.insert(table_name);
                ordered.push(table);
                progress = true;
            } else {
                next_round.push(table);
            }
        }

        if !progress {
            ordered.extend(next_round.into_iter());
            break;
        }

        remaining = next_round;
    }

    ordered
}

fn collect_foreign_key_dependencies(table: &CreateTable) -> Vec<String> {
    table
        .constraints
        .iter()
        .filter_map(|constraint| match constraint {
            TableConstraint::ForeignKey { foreign_table, .. } => {
                canonical_table_ident(foreign_table)
            }
            _ => None,
        })
        .collect()
}
