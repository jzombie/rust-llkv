//! Shared catalog DDL trait for contexts that can execute plan-based schema operations.
//!
//! This trait lives in `llkv-table` so that the runtime, transaction, and session layers
//! can agree on a single contract without introducing circular dependencies. It captures
//! the common plan-driven schema mutations (create/drop table, rename table, create/drop
//! index) while letting each layer choose its own return types via associated outputs.

use llkv_plan::{
    AlterTablePlan, CreateIndexPlan, CreateTablePlan, CreateViewPlan, DropIndexPlan, DropTablePlan,
    DropViewPlan, RenameTablePlan,
};
use llkv_result::Result;
use sqlparser::ast::{CreateTable, ObjectName, ObjectNamePart, TableConstraint};
use std::collections::HashSet;

/// Common DDL operations that operate on catalog-backed tables and indexes.
///
/// Implementors provide plan-based entry points for schema mutations so higher layers
/// (runtime session, transaction staging, etc.) can share the same method signatures.
/// Associated output types keep the contract generic while still conveying the concrete
/// results used by each layer (e.g., `RuntimeStatementResult`, `TransactionResult`, or
/// unit for operations that only signal success).
pub trait CatalogDdl {
    /// Output of executing a CREATE TABLE plan.
    type CreateTableOutput;
    /// Output of executing a DROP TABLE plan.
    type DropTableOutput;
    /// Output of executing a table rename.
    type RenameTableOutput;
    /// Output of executing an ALTER TABLE plan.
    type AlterTableOutput;
    /// Output of executing a CREATE INDEX plan.
    type CreateIndexOutput;
    /// Output of executing a DROP INDEX plan.
    type DropIndexOutput;

    /// Creates a table described by the given plan.
    fn create_table(&self, plan: CreateTablePlan) -> Result<Self::CreateTableOutput>;

    /// Drops a table identified by the given plan.
    fn drop_table(&self, plan: DropTablePlan) -> Result<Self::DropTableOutput>;

    /// Creates a view described by the given plan.
    fn create_view(&self, plan: CreateViewPlan) -> Result<()>;

    /// Drops a view identified by the given plan.
    fn drop_view(&self, plan: DropViewPlan) -> Result<()>;

    /// Renames a table using the provided plan.
    fn rename_table(&self, plan: RenameTablePlan) -> Result<Self::RenameTableOutput>;

    /// Alters a table using the provided plan.
    fn alter_table(&self, plan: AlterTablePlan) -> Result<Self::AlterTableOutput>;

    /// Creates an index described by the given plan.
    fn create_index(&self, plan: CreateIndexPlan) -> Result<Self::CreateIndexOutput>;

    /// Drops a single-column index described by the given plan, returning metadata about the
    /// index when it existed.
    fn drop_index(&self, plan: DropIndexPlan) -> Result<Self::DropIndexOutput>;
}

/// Extension methods for working with `ObjectName`.
pub trait ObjectNameExt {
    /// Return the final identifier component in uppercase form.
    fn canonical_ident(&self) -> Option<String>;
}

impl ObjectNameExt for ObjectName {
    fn canonical_ident(&self) -> Option<String> {
        self.0.last().and_then(|part| match part {
            ObjectNamePart::Identifier(ident) => Some(ident.value.to_ascii_uppercase()),
            ObjectNamePart::Function(_) => None,
        })
    }
}

/// Extension methods for normalizing `TableConstraint` instances.
pub trait TableConstraintExt {
    fn normalize(self) -> TableConstraint;
}

impl TableConstraintExt for TableConstraint {
    fn normalize(self) -> TableConstraint {
        match self {
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
}

/// Order CREATE TABLE statements such that referenced tables appear before dependents.
pub trait OrderCreateTablesExt {
    fn order_by_foreign_keys(self) -> Vec<CreateTable>;
}

impl OrderCreateTablesExt for Vec<CreateTable> {
    fn order_by_foreign_keys(mut self) -> Vec<CreateTable> {
        let mut ordered = Vec::with_capacity(self.len());
        if self.is_empty() {
            return ordered;
        }

        let table_names: HashSet<String> = self
            .iter()
            .filter_map(|table| table.name.canonical_ident())
            .collect();

        let mut resolved: HashSet<String> = HashSet::new();

        while !self.is_empty() {
            let mut progress = false;
            let current_round = std::mem::take(&mut self);
            let mut next_round = Vec::new();

            for table in current_round.into_iter() {
                let table_name = table.name.canonical_ident().unwrap_or_default();
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
                ordered.append(&mut next_round);
                break;
            }

            self = next_round;
        }

        ordered
    }
}

fn collect_foreign_key_dependencies(table: &CreateTable) -> Vec<String> {
    let mut deps = Vec::new();
    for constraint in &table.constraints {
        if let TableConstraint::ForeignKey { foreign_table, .. } = constraint
            && let Some(name) = foreign_table.canonical_ident()
        {
            deps.push(name);
        }
    }
    deps
}
