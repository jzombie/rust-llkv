//! Runtime context submodules
//!
//! This module contains the RuntimeContext implementation split into logical submodules:
//! - `mvcc_helpers`: MVCC transaction visibility filtering
//! - `query_translation`: String-based expression to field-ID-based expression translation
//! - `types`: Helper types (PreparedAssignmentValue, TableConstraintContext)
//! - `provider`: ContextProvider for TableProvider trait

use crate::{
    RuntimeSession, RuntimeStatementResult, RuntimeTableHandle, RuntimeTransactionContext,
    TXN_ID_AUTO_COMMIT, canonical_table_name, is_table_missing_error,
};
use llkv_column_map::store::ColumnStore;
use llkv_executor::{ExecutorMultiColumnUnique, ExecutorTable};
use llkv_plan::{
    AlterTablePlan, CreateIndexPlan, CreateTablePlan, CreateTableSource, DropIndexPlan,
    DropTablePlan, RenameTablePlan,
};
use llkv_result::{Error, Result};
use llkv_storage::pager::{MemPager, Pager};
use llkv_table::catalog::TableCatalog;
use llkv_table::{
    CatalogDdl, CatalogManager, ConstraintService, MetadataManager, MultiColumnUniqueRegistration,
    SingleColumnIndexDescriptor, SingleColumnIndexRegistration, SysCatalog, TableId,
    ensure_multi_column_unique, ensure_single_column_unique, validate_alter_table_operation,
};
use llkv_transaction::{TransactionManager, TransactionSnapshot, TxnId, TxnIdManager};
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{Arc, RwLock};

mod alter;
mod constraints;
mod delete;
mod insert;
mod provider;
mod query;
mod table_access;
mod table_creation;
mod types;
mod update;
mod utils;

pub(crate) use types::{PreparedAssignmentValue, TableConstraintContext};

/// In-memory execution context shared by plan-based queries.
///
/// Important: "lazy loading" here refers to *table metadata only* (schema,
/// executor-side column descriptors, and a small next-row-id counter). We do
/// NOT eagerly load or materialize the table's row data into memory. All
/// row/column data remains on the ColumnStore and is streamed in chunks during
/// query execution. This keeps the memory footprint low even for very large
/// tables.
///
/// Typical resource usage:
/// - Metadata per table: ~100s of bytes to a few KB (schema + field ids)
/// - ExecutorTable struct: small (handles + counters)
/// - Actual table rows: streamed from disk in chunks (never fully resident)
pub struct RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub(crate) pager: Arc<P>,
    tables: RwLock<FxHashMap<String, Arc<ExecutorTable<P>>>>,
    pub(crate) dropped_tables: RwLock<FxHashSet<String>>,
    metadata: Arc<MetadataManager<P>>,
    constraint_service: ConstraintService<P>,
    pub(crate) catalog_service: CatalogManager<P>,
    // Centralized catalog for table/field name resolution
    pub(crate) catalog: Arc<TableCatalog>,
    // Shared column store for all tables in this context
    // This ensures catalog state is synchronized across all tables
    store: Arc<ColumnStore<P>>,
    // Transaction manager for session-based transactions
    transaction_manager:
        TransactionManager<RuntimeTransactionContext<P>, RuntimeTransactionContext<MemPager>>,
    txn_manager: Arc<TxnIdManager>,
    txn_tables_with_new_rows: RwLock<FxHashMap<TxnId, FxHashSet<String>>>,
}

impl<P> RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(pager: Arc<P>) -> Self {
        Self::new_with_catalog_inner(pager, None)
    }

    pub fn new_with_catalog(pager: Arc<P>, catalog: Arc<TableCatalog>) -> Self {
        Self::new_with_catalog_inner(pager, Some(catalog))
    }

    fn new_with_catalog_inner(pager: Arc<P>, shared_catalog: Option<Arc<TableCatalog>>) -> Self {
        tracing::trace!("RuntimeContext::new called, pager={:p}", &*pager);

        let store = ColumnStore::open(Arc::clone(&pager)).expect("failed to open ColumnStore");
        let catalog = SysCatalog::new(&store);

        let next_txn_id = match catalog.get_next_txn_id() {
            Ok(Some(id)) => {
                tracing::debug!("[CONTEXT] Loaded next_txn_id={} from catalog", id);
                id
            }
            Ok(None) => {
                tracing::debug!("[CONTEXT] No persisted next_txn_id found, starting from default");
                TXN_ID_AUTO_COMMIT + 1
            }
            Err(e) => {
                tracing::warn!("[CONTEXT] Failed to load next_txn_id: {}, using default", e);
                TXN_ID_AUTO_COMMIT + 1
            }
        };

        let last_committed = match catalog.get_last_committed_txn_id() {
            Ok(Some(id)) => {
                tracing::debug!("[CONTEXT] Loaded last_committed={} from catalog", id);
                id
            }
            Ok(None) => {
                tracing::debug!(
                    "[CONTEXT] No persisted last_committed found, starting from default"
                );
                TXN_ID_AUTO_COMMIT
            }
            Err(e) => {
                tracing::warn!(
                    "[CONTEXT] Failed to load last_committed: {}, using default",
                    e
                );
                TXN_ID_AUTO_COMMIT
            }
        };

        let store_arc = Arc::new(store);
        let metadata = Arc::new(MetadataManager::new(Arc::clone(&store_arc)));

        let loaded_tables = match metadata.all_table_metas() {
            Ok(metas) => {
                tracing::debug!("[CONTEXT] Loaded {} table(s) from catalog", metas.len());
                metas
            }
            Err(e) => {
                tracing::warn!(
                    "[CONTEXT] Failed to load table metas: {}, starting with empty registry",
                    e
                );
                Vec::new()
            }
        };

        let transaction_manager =
            TransactionManager::new_with_initial_state(next_txn_id, last_committed);
        let txn_manager = transaction_manager.txn_manager();

        // LAZY LOADING: Only load table metadata at first access. We intentionally
        // avoid loading any row/column data into memory here. The executor
        // performs streaming reads from the ColumnStore when a query runs, so
        // large tables are never fully materialized.
        //
        // Benefits of this approach:
        // - Instant database open (no upfront I/O for table data)
        // - Lower memory footprint (only metadata cached)
        // - Natural parallelism: if multiple threads request different tables
        //   concurrently, those tables will be loaded concurrently by the
        //   caller threads (no global preload required).
        //
        // Future Optimizations (if profiling shows need):
        // 1. Eager parallel preload of a short "hot" list of tables (rayon)
        // 2. Background preload of catalog entries after startup
        // 3. LRU-based eviction for extremely large deployments
        // 4. Cache compact representations of schemas to reduce per-table RAM
        //
        // Note: `loaded_tables` holds catalog metadata that helped us discover
        // which tables exist; we discard it here because metadata will be
        // fetched on-demand during lazy loads.
        tracing::debug!(
            "[CONTEXT] Initialized with lazy loading for {} table(s)",
            loaded_tables.len()
        );

        // Initialize catalog and populate with existing tables
        let (catalog, is_shared_catalog) = match shared_catalog {
            Some(existing) => (existing, true),
            None => (Arc::new(TableCatalog::new()), false),
        };
        for (table_id, table_meta) in &loaded_tables {
            if let Some(ref table_name) = table_meta.name
                && let Err(e) = catalog.register_table(table_name.as_str(), *table_id)
            {
                match e {
                    Error::CatalogError(ref msg)
                        if is_shared_catalog && msg.contains("already exists") =>
                    {
                        tracing::debug!(
                            "[CONTEXT] Shared catalog already contains table '{}' with id={}",
                            table_name,
                            table_id
                        );
                    }
                    other => {
                        tracing::warn!(
                            "[CONTEXT] Failed to register table '{}' (id={}) in catalog: {}",
                            table_name,
                            table_id,
                            other
                        );
                    }
                }
            }
        }
        tracing::debug!(
            "[CONTEXT] Catalog initialized with {} table(s)",
            catalog.table_count()
        );

        let constraint_service =
            ConstraintService::new(Arc::clone(&metadata), Arc::clone(&catalog));
        let catalog_service = CatalogManager::new(
            Arc::clone(&metadata),
            Arc::clone(&catalog),
            Arc::clone(&store_arc),
        );

        // Load custom types from SysCatalog into catalog_service
        if let Err(e) = catalog_service.load_types_from_catalog() {
            tracing::warn!("[CONTEXT] Failed to load custom types: {}", e);
        }

        Self {
            pager,
            tables: RwLock::new(FxHashMap::default()), // Start with empty table cache
            dropped_tables: RwLock::new(FxHashSet::default()),
            metadata,
            constraint_service,
            catalog_service,
            catalog,
            store: store_arc,
            transaction_manager,
            txn_manager,
            txn_tables_with_new_rows: RwLock::new(FxHashMap::default()),
        }
    }

    /// Return the transaction ID manager shared with sessions.
    pub fn txn_manager(&self) -> Arc<TxnIdManager> {
        Arc::clone(&self.txn_manager)
    }

    /// Return the column store for catalog operations.
    pub fn store(&self) -> &Arc<ColumnStore<P>> {
        &self.store
    }

    /// Register a custom type alias (CREATE TYPE/DOMAIN).
    pub fn register_type(&self, name: String, data_type: sqlparser::ast::DataType) {
        self.catalog_service.register_type(name, data_type);
    }

    /// Drop a custom type alias (DROP TYPE/DOMAIN).
    pub fn drop_type(&self, name: &str) -> Result<()> {
        self.catalog_service.drop_type(name)?;
        Ok(())
    }

    /// Create a view by storing its SQL definition in the catalog.
    /// The view will be registered as a table with a view_definition.
    pub fn create_view(&self, display_name: &str, view_definition: String) -> Result<TableId> {
        self.catalog_service
            .create_view(display_name, view_definition)
    }

    /// Check if a table is actually a view by looking at its metadata.
    /// Returns true if the table exists and has a view_definition.
    pub fn is_view(&self, table_id: TableId) -> Result<bool> {
        self.catalog_service.is_view(table_id)
    }

    /// Resolve a type name to its base DataType, recursively following aliases.
    pub fn resolve_type(&self, data_type: &sqlparser::ast::DataType) -> sqlparser::ast::DataType {
        self.catalog_service.resolve_type(data_type)
    }

    /// Persist the next_txn_id to the catalog.
    pub fn persist_next_txn_id(&self, next_txn_id: TxnId) -> Result<()> {
        let catalog = SysCatalog::new(&self.store);
        catalog.put_next_txn_id(next_txn_id)?;
        let last_committed = self.txn_manager.last_committed();
        catalog.put_last_committed_txn_id(last_committed)?;
        tracing::debug!(
            "[CONTEXT] Persisted next_txn_id={}, last_committed={}",
            next_txn_id,
            last_committed
        );
        Ok(())
    }

    /// Construct the default snapshot for auto-commit operations.
    pub fn default_snapshot(&self) -> TransactionSnapshot {
        TransactionSnapshot {
            txn_id: TXN_ID_AUTO_COMMIT,
            snapshot_id: self.txn_manager.last_committed(),
        }
    }

    /// Get the table catalog for schema and table name management.
    pub fn table_catalog(&self) -> Arc<TableCatalog> {
        Arc::clone(&self.catalog)
    }

    /// Access the catalog manager for type registry, view management, and metadata operations.
    pub fn catalog(&self) -> &CatalogManager<P> {
        &self.catalog_service
    }

    /// Create a new session for transaction management.
    /// Each session can have its own independent transaction.
    pub fn create_session(self: &Arc<Self>) -> RuntimeSession<P> {
        tracing::debug!("[SESSION] RuntimeContext::create_session called");
        let namespaces = Arc::new(crate::runtime_session::SessionNamespaces::new(Arc::clone(
            self,
        )));
        let wrapper = RuntimeTransactionContext::new(Arc::clone(self));
        let inner = self.transaction_manager.create_session(Arc::new(wrapper));
        tracing::debug!(
            "[SESSION] Created TransactionSession with session_id (will be logged by transaction manager)"
        );
        RuntimeSession::from_parts(inner, namespaces)
    }

    /// Get a handle to an existing table by name.
    pub fn table(self: &Arc<Self>, name: &str) -> Result<RuntimeTableHandle<P>> {
        RuntimeTableHandle::new(Arc::clone(self), name)
    }

    /// Create a table with explicit column definitions - Programmatic API.
    ///
    /// This is a **convenience method** for programmatically creating tables with explicit
    /// column definitions. Use this when you're writing Rust code that needs to create tables
    /// directly, rather than executing SQL.
    ///
    /// # When to use `create_table` vs [`CatalogDdl::create_table`]
    ///
    /// **Use `create_table`:**
    /// - You're writing Rust code (not parsing SQL)
    /// - You have explicit column definitions
    /// - You want a simple, ergonomic API: `ctx.create_table("users", vec![...])`
    /// - You want a `RuntimeTableHandle` to work with immediately
    /// - **Does NOT support**: CREATE TABLE AS SELECT, foreign keys, namespaces
    ///
    /// **Use [`CatalogDdl::create_table`]:**
    /// - You're implementing SQL execution (already have a parsed `CreateTablePlan`)
    /// - You need CREATE TABLE AS SELECT support
    /// - You need foreign key constraints
    /// - You need namespace support (temporary tables)
    /// - You need IF NOT EXISTS / OR REPLACE semantics
    /// - You're working within the transaction system
    ///
    /// # Usage Comparison
    ///
    /// **Programmatic API** (this method):
    /// - `ctx.create_table("users", vec![("id", DataType::Int64, false), ...])?`
    /// - Returns `RuntimeTableHandle` for immediate use
    /// - Simple, ergonomic, no plan construction needed
    ///
    /// **SQL execution API** ([`CatalogDdl::create_table`]):
    /// - Construct a `CreateTablePlan` with all SQL features
    /// - Delegates to the [`CatalogDdl`] trait for catalog-aware creation
    /// - Support for CTAS, foreign keys, namespaces, transactions
    /// - Returns `RuntimeStatementResult` for consistency with other SQL operations
    ///
    /// # Returns
    /// Returns a [`RuntimeTableHandle`] that provides immediate access to the table.
    /// Use this for further programmatic operations on the table.
    /// Returns all table names currently registered in the catalog.
    pub fn table_names(self: &Arc<Self>) -> Vec<String> {
        // Use catalog for table names (single source of truth)
        self.catalog.table_names()
    }
}

impl<P> CatalogDdl for RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    type CreateTableOutput = RuntimeStatementResult<P>;
    type DropTableOutput = ();
    type RenameTableOutput = ();
    type AlterTableOutput = RuntimeStatementResult<P>;
    type CreateIndexOutput = RuntimeStatementResult<P>;
    type DropIndexOutput = Option<SingleColumnIndexDescriptor>;

    fn create_table(&self, plan: CreateTablePlan) -> Result<Self::CreateTableOutput> {
        if plan.columns.is_empty() && plan.source.is_none() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE requires explicit columns or a source".into(),
            ));
        }

        let (display_name, canonical_name) = canonical_table_name(&plan.name)?;
        let CreateTablePlan {
            name: _,
            if_not_exists,
            or_replace,
            columns,
            source,
            namespace: _,
            foreign_keys,
            multi_column_uniques,
        } = plan;

        tracing::trace!(
            "DEBUG create_table (plan): table='{}' if_not_exists={} columns={}",
            display_name,
            if_not_exists,
            columns.len()
        );
        for (idx, col) in columns.iter().enumerate() {
            tracing::trace!(
                "  plan column[{}]: name='{}' primary_key={}",
                idx,
                col.name,
                col.primary_key
            );
        }
        let (exists, is_dropped) = {
            let tables = self.tables.read().unwrap();
            let in_cache = tables.contains_key(&canonical_name);
            let is_dropped = self
                .dropped_tables
                .read()
                .unwrap()
                .contains(&canonical_name);
            // Table exists if it's in cache and NOT marked as dropped
            (in_cache && !is_dropped, is_dropped)
        };
        tracing::trace!(
            "DEBUG create_table (plan): exists={}, is_dropped={}",
            exists,
            is_dropped
        );

        // If table was dropped, remove it from cache before creating new one
        if is_dropped {
            self.remove_table_entry(&canonical_name);
            self.dropped_tables.write().unwrap().remove(&canonical_name);
        }

        if exists {
            if or_replace {
                tracing::trace!(
                    "DEBUG create_table (plan): table '{}' exists and or_replace=true, removing existing table before recreation",
                    display_name
                );
                self.remove_table_entry(&canonical_name);
            } else if if_not_exists {
                tracing::trace!(
                    "DEBUG create_table (plan): table '{}' exists and if_not_exists=true, returning early WITHOUT creating",
                    display_name
                );
                return Ok(RuntimeStatementResult::CreateTable {
                    table_name: display_name,
                });
            } else {
                return Err(Error::CatalogError(format!(
                    "Catalog Error: Table '{}' already exists",
                    display_name
                )));
            }
        }

        match source {
            Some(CreateTableSource::Batches { schema, batches }) => self.create_table_from_batches(
                display_name,
                canonical_name,
                schema,
                batches,
                if_not_exists,
            ),
            Some(CreateTableSource::Select { .. }) => Err(Error::Internal(
                "CreateTableSource::Select should be materialized before reaching RuntimeContext::create_table"
                    .into(),
            )),
            None => self.create_table_from_columns(
                display_name,
                canonical_name,
                columns,
                foreign_keys,
                multi_column_uniques,
                if_not_exists,
            ),
        }
    }

    fn drop_table(&self, plan: DropTablePlan) -> Result<Self::DropTableOutput> {
        let DropTablePlan { name, if_exists } = plan;
        let (display_name, canonical_name) = canonical_table_name(&name)?;

        tracing::debug!("drop_table: attempting to drop table '{}'", canonical_name);

        let cached_entry = {
            let tables = self.tables.read().unwrap();
            tracing::debug!("drop_table: cache contains {} tables", tables.len());
            tables.get(&canonical_name).cloned()
        };

        let table_entry = match cached_entry {
            Some(entry) => entry,
            None => {
                tracing::debug!(
                    "drop_table: table '{}' not cached; attempting reload",
                    canonical_name
                );

                if self.catalog.table_id(&canonical_name).is_none() {
                    tracing::debug!(
                        "drop_table: no catalog entry for '{}'; if_exists={}",
                        canonical_name,
                        if_exists
                    );
                    if if_exists {
                        return Ok(());
                    }
                    return Err(Error::CatalogError(format!(
                        "Catalog Error: Table '{}' does not exist",
                        display_name
                    )));
                }

                match self.lookup_table(&canonical_name) {
                    Ok(entry) => entry,
                    Err(err) => {
                        tracing::warn!(
                            "drop_table: failed to reload table '{}': {:?}",
                            canonical_name,
                            err
                        );
                        if if_exists {
                            return Ok(());
                        }
                        return Err(err);
                    }
                }
            }
        };

        let column_field_ids = table_entry
            .schema
            .columns
            .iter()
            .map(|col| col.field_id)
            .collect::<Vec<_>>();
        let table_id = table_entry.table.table_id();

        let referencing = self.constraint_service.referencing_foreign_keys(table_id)?;

        for detail in referencing {
            if detail.referencing_table_canonical == canonical_name {
                continue;
            }

            if self.is_table_marked_dropped(&detail.referencing_table_canonical) {
                continue;
            }

            return Err(Error::CatalogError(format!(
                "Catalog Error: Could not drop the table because this table is main key table of the table \"{}\".",
                detail.referencing_table_display
            )));
        }

        self.catalog_service
            .drop_table(&canonical_name, table_id, &column_field_ids)?;
        tracing::debug!(
            "[CATALOG] Unregistered table '{}' (table_id={}) from catalog",
            canonical_name,
            table_id
        );

        self.dropped_tables
            .write()
            .unwrap()
            .insert(canonical_name.clone());
        Ok(())
    }

    fn rename_table(&self, plan: RenameTablePlan) -> Result<Self::RenameTableOutput> {
        let RenameTablePlan {
            current_name,
            new_name,
            if_exists,
        } = plan;

        let (current_display, current_canonical) = canonical_table_name(&current_name)?;
        let (new_display, new_canonical) = canonical_table_name(&new_name)?;

        if current_canonical == new_canonical && current_display == new_display {
            return Ok(());
        }

        if self.is_table_marked_dropped(&current_canonical) {
            if if_exists {
                return Ok(());
            }
            return Err(Error::CatalogError(format!(
                "Catalog Error: Table '{}' does not exist",
                current_display
            )));
        }

        let table_id = match self
            .catalog
            .table_id(&current_canonical)
            .or_else(|| self.catalog.table_id(&current_display))
        {
            Some(id) => id,
            None => {
                if if_exists {
                    return Ok(());
                }
                return Err(Error::CatalogError(format!(
                    "Catalog Error: Table '{}' does not exist",
                    current_display
                )));
            }
        };

        if !current_display.eq_ignore_ascii_case(&new_display)
            && (self.catalog.table_id(&new_canonical).is_some()
                || self.catalog.table_id(&new_display).is_some())
        {
            return Err(Error::CatalogError(format!(
                "Catalog Error: Table '{}' already exists",
                new_display
            )));
        }

        let referencing = self.constraint_service.referencing_foreign_keys(table_id)?;
        if !referencing.is_empty() {
            return Err(Error::CatalogError(format!(
                "Dependency Error: Cannot alter entry \"{}\" because there are entries that depend on it.",
                current_display
            )));
        }

        self.catalog_service
            .rename_table(table_id, &current_display, &new_display)?;

        let mut tables = self.tables.write().unwrap();
        if let Some(table) = tables.remove(&current_canonical) {
            tables.insert(new_canonical.clone(), table);
        }

        let mut dropped = self.dropped_tables.write().unwrap();
        dropped.remove(&current_canonical);
        dropped.remove(&new_canonical);

        Ok(())
    }

    fn alter_table(&self, plan: AlterTablePlan) -> Result<Self::AlterTableOutput> {
        let (_, canonical_table) = canonical_table_name(&plan.table_name)?;

        let view = match self.catalog_service.table_view(&canonical_table) {
            Ok(view) => view,
            Err(err) if plan.if_exists && is_table_missing_error(&err) => {
                return Ok(RuntimeStatementResult::NoOp);
            }
            Err(err) => return Err(err),
        };

        let table_meta = match view.table_meta.as_ref() {
            Some(meta) => meta,
            None => {
                if plan.if_exists {
                    return Ok(RuntimeStatementResult::NoOp);
                }
                return Err(Error::Internal("table metadata missing".into()));
            }
        };

        let table_id = table_meta.table_id;

        validate_alter_table_operation(&plan.operation, &view, table_id, &self.catalog_service)?;

        match &plan.operation {
            llkv_plan::AlterTableOperation::RenameColumn {
                old_column_name,
                new_column_name,
            } => {
                self.rename_column(&plan.table_name, old_column_name, new_column_name)?;
            }
            llkv_plan::AlterTableOperation::SetColumnDataType {
                column_name,
                new_data_type,
            } => {
                self.alter_column_type(&plan.table_name, column_name, new_data_type)?;
            }
            llkv_plan::AlterTableOperation::DropColumn { column_name, .. } => {
                self.drop_column(&plan.table_name, column_name)?;
            }
        }

        Ok(RuntimeStatementResult::NoOp)
    }

    fn create_index(&self, plan: CreateIndexPlan) -> Result<Self::CreateIndexOutput> {
        if plan.columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE INDEX requires at least one column".into(),
            ));
        }

        for column_plan in &plan.columns {
            if !column_plan.ascending || column_plan.nulls_first {
                return Err(Error::InvalidArgumentError(
                    "only ASC indexes with NULLS LAST are supported".into(),
                ));
            }
        }

        let mut index_name = plan.name.clone();
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = self.lookup_table(&canonical_name)?;

        let mut column_indices = Vec::with_capacity(plan.columns.len());
        let mut field_ids = Vec::with_capacity(plan.columns.len());
        let mut column_names = Vec::with_capacity(plan.columns.len());
        let mut seen_column_indices = FxHashSet::default();

        for column_plan in &plan.columns {
            let normalized = column_plan.name.to_ascii_lowercase();
            let col_idx = table
                .schema
                .lookup
                .get(&normalized)
                .copied()
                .ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "column '{}' does not exist in table '{}'",
                        column_plan.name, display_name
                    ))
                })?;
            if !seen_column_indices.insert(col_idx) {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column '{}' in CREATE INDEX",
                    column_plan.name
                )));
            }

            let column = &table.schema.columns[col_idx];
            column_indices.push(col_idx);
            field_ids.push(column.field_id);
            column_names.push(column.name.clone());
        }

        if plan.columns.len() == 1 {
            let field_id = field_ids[0];
            let column_name = column_names[0].clone();

            if plan.unique {
                let snapshot = self.default_snapshot();
                let existing_values =
                    self.scan_column_values(table.as_ref(), field_id, snapshot)?;
                ensure_single_column_unique(&existing_values, &[], &column_name)?;
            }

            let registration = self.catalog_service.register_single_column_index(
                &display_name,
                &canonical_name,
                &table.table,
                field_id,
                &column_name,
                plan.name.clone(),
                plan.unique,
                plan.if_not_exists,
            )?;

            let created_name = match registration {
                SingleColumnIndexRegistration::Created { index_name } => index_name,
                SingleColumnIndexRegistration::AlreadyExists { index_name } => {
                    drop(table);
                    return Ok(RuntimeStatementResult::CreateIndex {
                        table_name: display_name,
                        index_name: Some(index_name),
                    });
                }
            };

            index_name = Some(created_name.clone());

            if let Some(updated_table) =
                Self::rebuild_executor_table_with_unique(table.as_ref(), field_id)
            {
                self.tables
                    .write()
                    .unwrap()
                    .insert(canonical_name.clone(), Arc::clone(&updated_table));
            } else {
                self.remove_table_entry(&canonical_name);
            }

            drop(table);

            return Ok(RuntimeStatementResult::CreateIndex {
                table_name: display_name,
                index_name,
            });
        }

        if !plan.unique {
            return Err(Error::InvalidArgumentError(
                "multi-column CREATE INDEX currently supports UNIQUE indexes only".into(),
            ));
        }

        let table_id = table.table.table_id();

        let snapshot = self.default_snapshot();
        let existing_rows = self.scan_multi_column_values(table.as_ref(), &field_ids, snapshot)?;
        ensure_multi_column_unique(&existing_rows, &[], &column_names)?;

        let executor_entry = ExecutorMultiColumnUnique {
            index_name: index_name.clone(),
            column_indices: column_indices.clone(),
        };

        let registration = self.catalog_service.register_multi_column_unique_index(
            table_id,
            &field_ids,
            index_name.clone(),
        )?;

        match registration {
            MultiColumnUniqueRegistration::Created => {
                table.add_multi_column_unique(executor_entry);
            }
            MultiColumnUniqueRegistration::AlreadyExists {
                index_name: existing,
            } => {
                if plan.if_not_exists {
                    drop(table);
                    return Ok(RuntimeStatementResult::CreateIndex {
                        table_name: display_name,
                        index_name: existing,
                    });
                }
                return Err(Error::CatalogError(format!(
                    "Index already exists on columns '{}'",
                    column_names.join(", ")
                )));
            }
        }

        Ok(RuntimeStatementResult::CreateIndex {
            table_name: display_name,
            index_name,
        })
    }

    fn drop_index(&self, plan: DropIndexPlan) -> Result<Self::DropIndexOutput> {
        let descriptor = self.catalog_service.drop_single_column_index(plan)?;

        if let Some(descriptor) = &descriptor {
            self.remove_table_entry(&descriptor.canonical_table_name);
        }

        Ok(descriptor)
    }
}
