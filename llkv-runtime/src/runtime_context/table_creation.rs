//! Table creation operations for RuntimeContext.
//!
//! This module contains logic for creating new tables:
//! - Public API methods (create_table, create_table_if_not_exists, create_table_builder)
//! - CREATE TABLE from column specifications
//! - CREATE TABLE AS SELECT (from batches)

use crate::{RuntimeCreateTableBuilder, RuntimeStatementResult, RuntimeTableHandle, TXN_ID_NONE, canonical_table_name};
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use llkv_executor::{current_time_micros, ExecutorColumn, ExecutorSchema, ExecutorTable};
use llkv_plan::{CreateTablePlan, ForeignKeySpec, IntoPlanColumnSpec, MultiColumnUniqueSpec, PlanColumnSpec};
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use llkv_table::{CatalogDdl, CreateTableResult, FieldId, ForeignKeyColumn, ForeignKeyTableInfo, Table};
use llkv_transaction::TransactionMvccBuilder;
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use super::RuntimeContext;

impl<P> RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Create a table from column specifications.
    pub(super) fn create_table_from_columns(
        &self,
        display_name: String,
        canonical_name: String,
        columns: Vec<PlanColumnSpec>,
        foreign_keys: Vec<ForeignKeySpec>,
        multi_column_uniques: Vec<MultiColumnUniqueSpec>,
        if_not_exists: bool,
    ) -> Result<RuntimeStatementResult<P>> {
        tracing::trace!(
            "\n=== CREATE_TABLE_FROM_COLUMNS: table='{}' columns={} ===",
            display_name,
            columns.len()
        );
        for (idx, col) in columns.iter().enumerate() {
            tracing::trace!(
                "  input column[{}]: name='{}' primary_key={}",
                idx,
                col.name,
                col.primary_key
            );
        }
        if columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE requires at least one column".into(),
            ));
        }

        // Avoid repeating catalog work if the table already exists.
        {
            let tables = self.tables.read().unwrap();
            if tables.contains_key(&canonical_name) {
                if if_not_exists {
                    return Ok(RuntimeStatementResult::CreateTable {
                        table_name: display_name,
                    });
                }

                return Err(Error::CatalogError(format!(
                    "Catalog Error: Table '{}' already exists",
                    display_name
                )));
            }
        }

        let CreateTableResult {
            table_id,
            table,
            table_columns,
            column_lookup,
        } = Table::create_from_columns(
            &display_name,
            &canonical_name,
            &columns,
            self.metadata.clone(),
            self.catalog.clone(),
            self.store.clone(),
        )?;

        tracing::trace!(
            "=== TABLE '{}' CREATED WITH table_id={} pager={:p} ===",
            display_name,
            table_id,
            &*self.pager
        );

        let mut column_defs: Vec<ExecutorColumn> = Vec::with_capacity(table_columns.len());
        for (idx, column) in table_columns.iter().enumerate() {
            tracing::trace!(
                "DEBUG create_table_from_columns[{}]: name='{}' data_type={:?} nullable={} primary_key={} unique={}",
                idx,
                column.name,
                column.data_type,
                column.nullable,
                column.primary_key,
                column.unique
            );
            column_defs.push(ExecutorColumn {
                name: column.name.clone(),
                data_type: column.data_type.clone(),
                nullable: column.nullable,
                primary_key: column.primary_key,
                unique: column.unique,
                field_id: column.field_id,
                check_expr: column.check_expr.clone(),
            });
        }

        let schema = Arc::new(ExecutorSchema {
            columns: column_defs.clone(),
            lookup: column_lookup,
        });
        let table_entry = Arc::new(ExecutorTable {
            table: Arc::clone(&table),
            schema,
            next_row_id: AtomicU64::new(0),
            total_rows: AtomicU64::new(0),
            multi_column_uniques: RwLock::new(Vec::new()),
        });

        let mut tables = self.tables.write().unwrap();
        if tables.contains_key(&canonical_name) {
            drop(tables);
            let field_ids: Vec<FieldId> =
                table_columns.iter().map(|column| column.field_id).collect();
            let _ = self
                .catalog_service
                .drop_table(&canonical_name, table_id, &field_ids);
            if if_not_exists {
                return Ok(RuntimeStatementResult::CreateTable {
                    table_name: display_name,
                });
            }
            return Err(Error::CatalogError(format!(
                "Catalog Error: Table '{}' already exists",
                display_name
            )));
        }
        tables.insert(canonical_name.clone(), Arc::clone(&table_entry));
        drop(tables);

        if !foreign_keys.is_empty() {
            let fk_result = self.catalog_service.register_foreign_keys_for_new_table(
                table_id,
                &display_name,
                &canonical_name,
                &table_columns,
                &foreign_keys,
                |table_name| {
                    let (display, canonical) = canonical_table_name(table_name)?;
                    let referenced_table = self.lookup_table(&canonical).map_err(|_| {
                        Error::CatalogError(format!(
                            "Catalog Error: referenced table '{}' does not exist",
                            table_name
                        ))
                    })?;

                    let columns = referenced_table
                        .schema
                        .columns
                        .iter()
                        .map(|column| ForeignKeyColumn {
                            name: column.name.clone(),
                            data_type: column.data_type.clone(),
                            nullable: column.nullable,
                            primary_key: column.primary_key,
                            unique: column.unique,
                            field_id: column.field_id,
                        })
                        .collect();

                    // Get multi-column unique constraints for this table
                    let multi_column_uniques = self
                        .catalog_service
                        .table_view(&canonical)?
                        .multi_column_uniques;

                    Ok(ForeignKeyTableInfo {
                        display_name: display,
                        canonical_name: canonical,
                        table_id: referenced_table.table.table_id(),
                        columns,
                        multi_column_uniques,
                    })
                },
                current_time_micros(),
            );

            if let Err(err) = fk_result {
                let field_ids: Vec<FieldId> =
                    table_columns.iter().map(|column| column.field_id).collect();
                let _ = self
                    .catalog_service
                    .drop_table(&canonical_name, table_id, &field_ids);
                self.remove_table_entry(&canonical_name);
                return Err(err);
            }
        }

        // Register multi-column unique constraints
        if !multi_column_uniques.is_empty() {
            // Build column name to field_id lookup
            let column_lookup: FxHashMap<String, FieldId> = table_columns
                .iter()
                .map(|col| (col.name.to_ascii_lowercase(), col.field_id))
                .collect();

            for unique_spec in &multi_column_uniques {
                // Map column names to field IDs
                let field_ids: Result<Vec<FieldId>> = unique_spec
                    .columns
                    .iter()
                    .map(|col_name| {
                        let normalized = col_name.to_ascii_lowercase();
                        column_lookup.get(&normalized).copied().ok_or_else(|| {
                            Error::InvalidArgumentError(format!(
                                "unknown column '{}' in UNIQUE constraint",
                                col_name
                            ))
                        })
                    })
                    .collect();

                let field_ids = field_ids?;

                let registration_result = self.catalog_service.register_multi_column_unique_index(
                    table_id,
                    &field_ids,
                    unique_spec.name.clone(),
                );

                if let Err(err) = registration_result {
                    let field_ids: Vec<FieldId> =
                        table_columns.iter().map(|column| column.field_id).collect();
                    let _ = self
                        .catalog_service
                        .drop_table(&canonical_name, table_id, &field_ids);
                    self.remove_table_entry(&canonical_name);
                    return Err(err);
                }
            }
        }

        Ok(RuntimeStatementResult::CreateTable {
            table_name: display_name,
        })
    }

    /// Create a table from batches (CREATE TABLE AS SELECT).
    pub(super) fn create_table_from_batches(
        &self,
        display_name: String,
        canonical_name: String,
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
        if_not_exists: bool,
    ) -> Result<RuntimeStatementResult<P>> {
        {
            let tables = self.tables.read().unwrap();
            if tables.contains_key(&canonical_name) {
                if if_not_exists {
                    return Ok(RuntimeStatementResult::CreateTable {
                        table_name: display_name,
                    });
                }
                return Err(Error::CatalogError(format!(
                    "Catalog Error: Table '{}' already exists",
                    display_name
                )));
            }
        }

        let CreateTableResult {
            table_id,
            table,
            table_columns,
            column_lookup,
        } = self.catalog_service.create_table_from_schema(
            &display_name,
            &canonical_name,
            &schema,
        )?;

        tracing::trace!(
            "=== CTAS table '{}' created with table_id={} pager={:p} ===",
            display_name,
            table_id,
            &*self.pager
        );

        let mut column_defs: Vec<ExecutorColumn> = Vec::with_capacity(table_columns.len());
        for column in &table_columns {
            column_defs.push(ExecutorColumn {
                name: column.name.clone(),
                data_type: column.data_type.clone(),
                nullable: column.nullable,
                primary_key: column.primary_key,
                unique: column.unique,
                field_id: column.field_id,
                check_expr: column.check_expr.clone(),
            });
        }

        let schema_arc = Arc::new(ExecutorSchema {
            columns: column_defs.clone(),
            lookup: column_lookup,
        });
        let table_entry = Arc::new(ExecutorTable {
            table: Arc::clone(&table),
            schema: schema_arc,
            next_row_id: AtomicU64::new(0),
            total_rows: AtomicU64::new(0),
            multi_column_uniques: RwLock::new(Vec::new()),
        });

        let creator_snapshot = self.txn_manager.begin_transaction();
        let creator_txn_id = creator_snapshot.txn_id;
        let mvcc_builder = TransactionMvccBuilder;
        let (next_row_id, total_rows) = match self.catalog_service.append_batches_with_mvcc(
            table.as_ref(),
            &table_columns,
            &batches,
            creator_txn_id,
            TXN_ID_NONE,
            0,
            &mvcc_builder,
        ) {
            Ok(result) => {
                self.txn_manager.mark_committed(creator_txn_id);
                result
            }
            Err(err) => {
                self.txn_manager.mark_aborted(creator_txn_id);
                let field_ids: Vec<FieldId> =
                    table_columns.iter().map(|column| column.field_id).collect();
                let _ = self
                    .catalog_service
                    .drop_table(&canonical_name, table_id, &field_ids);
                return Err(err);
            }
        };

        table_entry.next_row_id.store(next_row_id, Ordering::SeqCst);
        table_entry.total_rows.store(total_rows, Ordering::SeqCst);

        let mut tables = self.tables.write().unwrap();
        if tables.contains_key(&canonical_name) {
            if if_not_exists {
                return Ok(RuntimeStatementResult::CreateTable {
                    table_name: display_name,
                });
            }
            return Err(Error::CatalogError(format!(
                "Catalog Error: Table '{}' already exists",
                display_name
            )));
        }
        tables.insert(canonical_name.clone(), Arc::clone(&table_entry));
        drop(tables); // Release write lock before catalog operations

        Ok(RuntimeStatementResult::CreateTable {
            table_name: display_name,
        })
    }

    /// Create a new table with the given columns.
    ///
    /// Public API method for table creation.
    pub fn create_table<C, I>(
        self: &Arc<Self>,
        name: &str,
        columns: I,
    ) -> Result<RuntimeTableHandle<P>>
    where
        C: IntoPlanColumnSpec,
        I: IntoIterator<Item = C>,
    {
        self.create_table_with_options(name, columns, false)
    }

    /// Create a new table if it doesn't already exist.
    ///
    /// Public API method for conditional table creation.
    pub fn create_table_if_not_exists<C, I>(
        self: &Arc<Self>,
        name: &str,
        columns: I,
    ) -> Result<RuntimeTableHandle<P>>
    where
        C: IntoPlanColumnSpec,
        I: IntoIterator<Item = C>,
    {
        self.create_table_with_options(name, columns, true)
    }

    /// Creates a fluent builder for defining and creating a new table with columns and constraints.
    ///
    /// Public API method for advanced table creation with constraints.
    pub fn create_table_builder(&self, name: &str) -> RuntimeCreateTableBuilder<'_, P> {
        RuntimeCreateTableBuilder::new(self, name)
    }

    /// Internal helper for create_table and create_table_if_not_exists.
    fn create_table_with_options<C, I>(
        self: &Arc<Self>,
        name: &str,
        columns: I,
        if_not_exists: bool,
    ) -> Result<RuntimeTableHandle<P>>
    where
        C: IntoPlanColumnSpec,
        I: IntoIterator<Item = C>,
    {
        let mut plan = CreateTablePlan::new(name);
        plan.if_not_exists = if_not_exists;
        plan.columns = columns
            .into_iter()
            .map(|column| column.into_plan_column_spec())
            .collect();
        let result = CatalogDdl::create_table(self.as_ref(), plan)?;
        match result {
            RuntimeStatementResult::CreateTable { .. } => {
                RuntimeTableHandle::new(Arc::clone(self), name)
            }
            other => Err(Error::InvalidArgumentError(format!(
                "unexpected statement result {other:?} when creating table"
            ))),
        }
    }

    /// Execute a CREATE TABLE plan - internal storage API.
    ///
    /// Use RuntimeSession::execute_statement() instead for external use.
    pub(crate) fn execute_create_table(
        &self,
        plan: CreateTablePlan,
    ) -> Result<RuntimeStatementResult<P>> {
        CatalogDdl::create_table(self, plan)
    }
}
