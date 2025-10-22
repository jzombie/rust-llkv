// RuntimeContext - extracted from lib.rs
// Complete implementation of the RuntimeContext struct and all its methods

use std::sync::{Arc, atomic::{AtomicU64, Ordering}, RwLock};
use std::mem;
use std::time::{SystemTime, UNIX_EPOCH};
use std::marker::PhantomData;
use std::ops::Bound;
use rustc_hash::{FxHashMap, FxHashSet};
use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Date32Builder, Float64Array, Float64Builder,
    Int64Array, Int64Builder, RecordBatch, StringBuilder, UInt64Array, UInt64Builder,
};
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use sqlparser::ast::{
    Expr as SqlExpr, GroupByExpr, ObjectName, ObjectNamePart, Select, SelectItem,
    SelectItemQualifiedWildcardKind, TableAlias, TableFactor, UnaryOperator, Value,
    ValueWithSpan, FunctionArg, FunctionArgExpr,
};
use time::{Date, Month};
use llkv_storage::pager::Pager;
use llkv_result::{Error, Result};
use llkv_column_map::store::ColumnStore;
use llkv_column_map::types::LogicalFieldId;
use llkv_column_map::store::GatherNullPolicy;
use llkv_table::{
    Table, MetadataManager, SysCatalog, TableId,
    CatalogManager, ConstraintService, ConstraintKind,
    CreateTableResult, FieldId, RowId, ROW_ID_FIELD_ID,
    ConstraintColumnInfo, InsertColumnConstraint, InsertUniqueColumn, InsertMultiColumnUnique,
    ForeignKeyTableInfo, ForeignKeyColumn, ForeignKeyView, TableView,
    SingleColumnIndexDescriptor, SingleColumnIndexRegistration, MultiColumnUniqueRegistration,
    TableConstraintSummaryView, MultiColumnUniqueEntryMeta,
    ensure_single_column_unique, ensure_multi_column_unique, build_composite_unique_key, UniqueKey,
    CatalogDdl,
};
use llkv_table::catalog::{TableCatalog, MvccColumnBuilder};
use llkv_table::resolvers::{FieldDefinition, FieldConstraints};
use llkv_table::table::{RowIdFilter, ScanProjection, ScanStreamOptions};
use llkv_transaction::{TransactionManager, TxnIdManager, TransactionSnapshot, TxnId};
use llkv_transaction::mvcc::{self, RowVersion};
use llkv_executor::{
    ExecutorTable, ExecutorColumn, ExecutorSchema, ExecutorMultiColumnUnique,
    QueryExecutor, TableProvider, RowBatch,
};
use llkv_expr::{Expr as LlkvExpr, Filter, Operator, ScalarExpr};
use llkv_plan::{
    CreateTablePlan, CreateTableSource, InsertPlan, InsertSource, UpdatePlan, DeletePlan,
    SelectPlan, DropTablePlan, RenameTablePlan, AlterTablePlan, CreateIndexPlan, DropIndexPlan,
    ColumnAssignment, AssignmentValue, PlanValue, ColumnSpec, ForeignKeySpec,
    MultiColumnUniqueSpec, IntoColumnSpec,
};
use crate::{
    RuntimeStatementResult, RuntimeTableHandle, RuntimeSession, SelectExecution,
    RuntimeCreateTableBuilder, canonical_table_name,
    ROW_ID_COLUMN_NAME,
    TXN_ID_AUTO_COMMIT, TXN_ID_NONE,
    RuntimeTransactionContext,
    validate_alter_table_operation, is_table_missing_error,
    sql_type_to_arrow, EntryHandle,
};
use llkv_storage::pager::MemPager;


/// Represents how a column assignment should be materialized during UPDATE/INSERT.
enum PreparedAssignmentValue {
    Literal(PlanValue),
    Expression { expr_index: usize },
}

struct TransactionMvccBuilder;

impl MvccColumnBuilder for TransactionMvccBuilder {
    fn build_insert_columns(
        &self,
        row_count: usize,
        start_row_id: RowId,
        creator_txn_id: u64,
        deleted_marker: u64,
    ) -> (ArrayRef, ArrayRef, ArrayRef) {
        mvcc::build_insert_mvcc_columns(row_count, start_row_id, creator_txn_id, deleted_marker)
    }

    fn mvcc_fields(&self) -> Vec<Field> {
        mvcc::build_mvcc_fields()
    }

    fn field_with_metadata(
        &self,
        name: &str,
        data_type: DataType,
        nullable: bool,
        field_id: FieldId,
    ) -> Field {
        mvcc::build_field_with_metadata(name, data_type, nullable, field_id)
    }
}

#[derive(Debug, Clone)]
struct TableConstraintContext {
    schema_field_ids: Vec<FieldId>,
    column_constraints: Vec<InsertColumnConstraint>,
    unique_columns: Vec<InsertUniqueColumn>,
    multi_column_uniques: Vec<InsertMultiColumnUnique>,
    primary_key: Option<InsertMultiColumnUnique>,
}

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
    // Type registry for custom type aliases (CREATE TYPE/DOMAIN)
    type_registry: RwLock<FxHashMap<String, sqlparser::ast::DataType>>,
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

        // Load custom types from catalog
        let sys_catalog = SysCatalog::new(&store_arc);
        let mut type_registry_map = FxHashMap::default();

        match sys_catalog.all_custom_type_metas() {
            Ok(type_metas) => {
                tracing::debug!(
                    "[CONTEXT] Loaded {} custom type(s) from catalog",
                    type_metas.len()
                );
                for type_meta in type_metas {
                    // Parse the base_type_sql back to a DataType
                    if let Ok(parsed_type) =
                        Self::parse_data_type_from_sql(&type_meta.base_type_sql)
                    {
                        type_registry_map.insert(type_meta.name.to_lowercase(), parsed_type);
                    } else {
                        tracing::warn!(
                            "[CONTEXT] Failed to parse base type SQL for type '{}': {}",
                            type_meta.name,
                            type_meta.base_type_sql
                        );
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    "[CONTEXT] Failed to load custom types: {}, starting with empty type registry",
                    e
                );
            }
        }
        tracing::debug!(
            "[CONTEXT] Type registry initialized with {} type(s)",
            type_registry_map.len()
        );

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
            type_registry: RwLock::new(type_registry_map),
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
        let mut registry = self.type_registry.write().unwrap();
        registry.insert(name.to_lowercase(), data_type);
    }

    /// Drop a custom type alias (DROP TYPE/DOMAIN).
    pub fn drop_type(&self, name: &str) -> Result<()> {
        let mut registry = self.type_registry.write().unwrap();
        if registry.remove(&name.to_lowercase()).is_none() {
            return Err(Error::InvalidArgumentError(format!(
                "Type '{}' does not exist",
                name
            )));
        }
        Ok(())
    }

    /// Create a view by storing its SQL definition in the catalog.
    /// The view will be registered as a table with a view_definition.
    pub fn create_view(&self, display_name: &str, view_definition: String) -> Result<TableId> {
        use llkv_table::TableMeta;
        use std::time::{SystemTime, UNIX_EPOCH};

        // Reserve a new table ID for the view
        let table_id = self.metadata.reserve_table_id()?;

        let created_at_micros = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        // Create the table metadata with view_definition set
        let table_meta = TableMeta {
            table_id,
            name: Some(display_name.to_string()),
            created_at_micros,
            flags: 0,
            epoch: 0,
            view_definition: Some(view_definition),
        };

        // Store the metadata and flush to disk
        self.metadata.set_table_meta(table_id, table_meta)?;
        self.metadata.flush_table(table_id)?;

        // Register the view in the catalog so it can be looked up by name
        self.catalog.register_table(display_name, table_id)?;

        tracing::debug!("Created view '{}' with table_id={}", display_name, table_id);
        Ok(table_id)
    }

    /// Check if a table is actually a view by looking at its metadata.
    /// Returns true if the table exists and has a view_definition.
    pub fn is_view(&self, table_id: TableId) -> Result<bool> {
        match self.metadata.table_meta(table_id)? {
            Some(meta) => Ok(meta.view_definition.is_some()),
            None => Ok(false),
        }
    }

    /// Resolve a type name to its base DataType, recursively following aliases.
    pub fn resolve_type(&self, data_type: &sqlparser::ast::DataType) -> sqlparser::ast::DataType {
        use sqlparser::ast::DataType;

        match data_type {
            DataType::Custom(obj_name, _) => {
                let name = obj_name.to_string().to_lowercase();
                let registry = self.type_registry.read().unwrap();
                if let Some(base_type) = registry.get(&name) {
                    // Recursively resolve in case the base type is also an alias
                    self.resolve_type(base_type)
                } else {
                    // Not a custom type, return as-is
                    data_type.clone()
                }
            }
            // For non-custom types, return as-is
            _ => data_type.clone(),
        }
    }

    /// Parse a SQL type string (e.g., "INTEGER") back into a DataType.
    fn parse_data_type_from_sql(sql: &str) -> Result<sqlparser::ast::DataType> {
        use sqlparser::dialect::GenericDialect;
        use sqlparser::parser::Parser;

        // Try to parse as a simple CREATE DOMAIN statement
        let create_sql = format!("CREATE DOMAIN dummy AS {}", sql);
        let dialect = GenericDialect {};

        match Parser::parse_sql(&dialect, &create_sql) {
            Ok(stmts) if !stmts.is_empty() => {
                if let sqlparser::ast::Statement::CreateDomain(create_domain) = &stmts[0] {
                    Ok(create_domain.data_type.clone())
                } else {
                    Err(Error::InvalidArgumentError(format!(
                        "Failed to parse type from SQL: {}",
                        sql
                    )))
                }
            }
            _ => Err(Error::InvalidArgumentError(format!(
                "Failed to parse type from SQL: {}",
                sql
            ))),
        }
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

    fn build_executor_multi_column_uniques(
        table: &ExecutorTable<P>,
        stored: &[MultiColumnUniqueEntryMeta],
    ) -> Vec<ExecutorMultiColumnUnique> {
        let mut results = Vec::with_capacity(stored.len());

        'outer: for entry in stored {
            if entry.column_ids.is_empty() {
                continue;
            }

            let mut column_indices = Vec::with_capacity(entry.column_ids.len());
            for field_id in &entry.column_ids {
                if let Some((idx, _)) = table
                    .schema
                    .columns
                    .iter()
                    .enumerate()
                    .find(|(_, col)| &col.field_id == field_id)
                {
                    column_indices.push(idx);
                } else {
                    tracing::warn!(
                        "[CATALOG] Skipping persisted multi-column UNIQUE {:?} for table_id={} missing field_id {}",
                        entry.index_name,
                        table.table.table_id(),
                        field_id
                    );
                    continue 'outer;
                }
            }

            results.push(ExecutorMultiColumnUnique {
                index_name: entry.index_name.clone(),
                column_indices,
            });
        }

        results
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

    /// Create a new session for transaction management.
    /// Each session can have its own independent transaction.
    pub fn create_session(self: &Arc<Self>) -> RuntimeSession<P> {
        tracing::debug!("[SESSION] RuntimeContext::create_session called");
        let namespaces = Arc::new(crate::runtime_session::SessionNamespaces::new(Arc::clone(self)));
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

    /// Check if there's an active transaction (checks if ANY session has a transaction).
    #[deprecated(note = "Use session-based transactions instead")]
    pub fn has_active_transaction(&self) -> bool {
        self.transaction_manager.has_active_transaction()
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
    pub fn create_table<C, I>(
        self: &Arc<Self>,
        name: &str,
        columns: I,
    ) -> Result<RuntimeTableHandle<P>>
    where
        C: IntoColumnSpec,
        I: IntoIterator<Item = C>,
    {
        self.create_table_with_options(name, columns, false)
    }

    pub fn create_table_if_not_exists<C, I>(
        self: &Arc<Self>,
        name: &str,
        columns: I,
    ) -> Result<RuntimeTableHandle<P>>
    where
        C: IntoColumnSpec,
        I: IntoIterator<Item = C>,
    {
        self.create_table_with_options(name, columns, true)
    }

    /// Returns all table names currently registered in the catalog.
    pub fn table_names(self: &Arc<Self>) -> Vec<String> {
        // Use catalog for table names (single source of truth)
        self.catalog.table_names()
    }

    /// Returns metadata snapshot for a table including columns, types, and constraints.
    pub fn table_view(&self, canonical_name: &str) -> Result<TableView> {
        self.catalog_service.table_view(canonical_name)
    }

    fn filter_visible_row_ids(
        &self,
        table: &ExecutorTable<P>,
        row_ids: Vec<RowId>,
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<RowId>> {
        filter_row_ids_for_snapshot(table.table.as_ref(), row_ids, &self.txn_manager, snapshot)
    }

    /// Creates a fluent builder for defining and creating a new table with columns and constraints.
    pub fn create_table_builder(&self, name: &str) -> RuntimeCreateTableBuilder<'_, P> {
        RuntimeCreateTableBuilder::new(self, name)
    }

    /// Returns column specifications for a table including names, types, and constraint flags.
    pub fn table_column_specs(self: &Arc<Self>, name: &str) -> Result<Vec<ColumnSpec>> {
        let (_, canonical_name) = canonical_table_name(name)?;
        self.catalog_service.table_column_specs(&canonical_name)
    }

    /// Renames a column in a table by delegating to the catalog layer.
    pub fn rename_column(
        &self,
        table_name: &str,
        old_column_name: &str,
        new_column_name: &str,
    ) -> Result<()> {
        let (_, canonical_table) = canonical_table_name(table_name)?;

        // Get table ID
        let table_id = self
            .catalog
            .table_id(&canonical_table)
            .ok_or_else(|| Error::CatalogError(format!("table '{}' not found", table_name)))?;

        // Delegate to catalog layer
        self.catalog_service
            .rename_column(table_id, old_column_name, new_column_name)?;

        // Invalidate cached executor table so subsequent operations see updated schema.
        self.remove_table_entry(&canonical_table);

        Ok(())
    }

    /// Alters the data type of a column by delegating to the catalog layer.
    pub fn alter_column_type(
        &self,
        table_name: &str,
        column_name: &str,
        new_data_type_sql: &str,
    ) -> Result<()> {
        let (_, canonical_table) = canonical_table_name(table_name)?;

        // Get table ID
        let table_id = self
            .catalog
            .table_id(&canonical_table)
            .ok_or_else(|| Error::CatalogError(format!("table '{}' not found", table_name)))?;

        // Parse SQL type string to Arrow DataType
        let arrow_type = sql_type_to_arrow(new_data_type_sql)?;

        // Delegate to catalog layer
        self.catalog_service
            .alter_column_type(table_id, column_name, &arrow_type)?;

        // Invalidate cached executor table so subsequent operations see updated schema.
        self.remove_table_entry(&canonical_table);

        Ok(())
    }

    /// Drops a column from a table by delegating to the catalog layer.
    pub fn drop_column(&self, table_name: &str, column_name: &str) -> Result<()> {
        let (_, canonical_table) = canonical_table_name(table_name)?;

        // Get table ID
        let table_id = self
            .catalog
            .table_id(&canonical_table)
            .ok_or_else(|| Error::CatalogError(format!("table '{}' not found", table_name)))?;

        // Delegate to catalog layer
        self.catalog_service.drop_column(table_id, column_name)?;

        // Invalidate cached executor table so subsequent operations see updated schema.
        self.remove_table_entry(&canonical_table);

        Ok(())
    }
    /// Returns foreign key relationships for a table, both referencing and referenced constraints.
    pub fn foreign_key_views(self: &Arc<Self>, name: &str) -> Result<Vec<ForeignKeyView>> {
        let (_, canonical_name) = canonical_table_name(name)?;
        self.catalog_service.foreign_key_views(&canonical_name)
    }

    /// Exports all rows from a table as a `RowBatch`, useful for data migration or inspection.
    pub fn export_table_rows(self: &Arc<Self>, name: &str) -> Result<RowBatch> {
        let handle = RuntimeTableHandle::new(Arc::clone(self), name)?;
        handle.lazy()?.collect_rows()
    }

    pub(crate) fn execute_create_table(&self, plan: CreateTablePlan) -> Result<RuntimeStatementResult<P>> {
        CatalogDdl::create_table(self, plan)
    }

    fn create_table_with_options<C, I>(
        self: &Arc<Self>,
        name: &str,
        columns: I,
        if_not_exists: bool,
    ) -> Result<RuntimeTableHandle<P>>
    where
        C: IntoColumnSpec,
        I: IntoIterator<Item = C>,
    {
        let mut plan = CreateTablePlan::new(name);
        plan.if_not_exists = if_not_exists;
        plan.columns = columns
            .into_iter()
            .map(|column| column.into_column_spec())
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

    pub fn insert(&self, plan: InsertPlan) -> Result<RuntimeStatementResult<P>> {
        // For non-transactional inserts, use TXN_ID_AUTO_COMMIT directly
        // instead of creating a temporary transaction
        let snapshot = TransactionSnapshot {
            txn_id: TXN_ID_AUTO_COMMIT,
            snapshot_id: self.txn_manager.last_committed(),
        };
        self.insert_with_snapshot(plan, snapshot)
    }

    pub fn insert_with_snapshot(
        &self,
        plan: InsertPlan,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = self.lookup_table(&canonical_name)?;

        // Targeted debug for 'keys' table only
        if display_name == "keys" {
            tracing::trace!(
                "\n[KEYS] INSERT starting - table_id={}, context_pager={:p}",
                table.table.table_id(),
                &*self.pager
            );
            tracing::trace!(
                "[KEYS] Table has {} columns, primary_key columns: {:?}",
                table.schema.columns.len(),
                table
                    .schema
                    .columns
                    .iter()
                    .filter(|c| c.primary_key)
                    .map(|c| &c.name)
                    .collect::<Vec<_>>()
            );
        }

        let result = match plan.source {
            InsertSource::Rows(rows) => self.insert_rows(
                table.as_ref(),
                display_name.clone(),
                canonical_name.clone(),
                rows,
                plan.columns,
                snapshot,
            ),
            InsertSource::Batches(batches) => self.insert_batches(
                table.as_ref(),
                display_name.clone(),
                canonical_name.clone(),
                batches,
                plan.columns,
                snapshot,
            ),
            InsertSource::Select { .. } => Err(Error::Internal(
                "InsertSource::Select should be materialized before reaching RuntimeContext::insert"
                    .into(),
            )),
        };

        if display_name == "keys" {
            tracing::trace!(
                "[KEYS] INSERT completed: {:?}",
                result
                    .as_ref()
                    .map(|_| "OK")
                    .map_err(|e| format!("{:?}", e))
            );
        }

        if matches!(result, Err(Error::NotFound)) {
            panic!(
                "BUG: insert yielded Error::NotFound for table '{}'. \
                 This should never happen: insert should never return NotFound after successful table lookup. \
                 This indicates a logic error in the runtime.",
                display_name
            );
        }

        result
    }

    /// Get raw batches from a table including row_ids, optionally filtered.
    /// This is used for transaction seeding where we need to preserve existing row_ids.
    pub fn get_batches_with_row_ids(
        &self,
        table_name: &str,
        filter: Option<LlkvExpr<'static, String>>,
    ) -> Result<Vec<RecordBatch>> {
        self.get_batches_with_row_ids_with_snapshot(table_name, filter, self.default_snapshot())
    }

    pub fn get_batches_with_row_ids_with_snapshot(
        &self,
        table_name: &str,
        filter: Option<LlkvExpr<'static, String>>,
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<RecordBatch>> {
        let (_, canonical_name) = canonical_table_name(table_name)?;
        let table = self.lookup_table(&canonical_name)?;

        let filter_expr = match filter {
            Some(expr) => translate_predicate(expr, table.schema.as_ref())?,
            None => {
                let field_id = table.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform wildcard scan".into(),
                    )
                })?;
                full_table_scan_filter(field_id)
            }
        };

        // First, get the row_ids that match the filter
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        if row_ids.is_empty() {
            return Ok(Vec::new());
        }

        let visible_row_ids = self.filter_visible_row_ids(table.as_ref(), row_ids, snapshot)?;
        if visible_row_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Scan to get the column data without materializing full columns
        let table_id = table.table.table_id();

        let mut fields: Vec<Field> = Vec::with_capacity(table.schema.columns.len() + 1);
        let mut logical_fields: Vec<LogicalFieldId> =
            Vec::with_capacity(table.schema.columns.len());

        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

        for column in &table.schema.columns {
            let logical_field_id = LogicalFieldId::for_user(table_id, column.field_id);
            logical_fields.push(logical_field_id);
            let field = mvcc::build_field_with_metadata(
                &column.name,
                column.data_type.clone(),
                column.nullable,
                column.field_id,
            );
            fields.push(field);
        }

        let schema = Arc::new(Schema::new(fields));

        if logical_fields.is_empty() {
            // Tables without user columns should still return row_id batches.
            let mut row_id_builder = UInt64Builder::with_capacity(visible_row_ids.len());
            for row_id in &visible_row_ids {
                row_id_builder.append_value(*row_id);
            }
            let arrays: Vec<ArrayRef> = vec![Arc::new(row_id_builder.finish()) as ArrayRef];
            let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
            return Ok(vec![batch]);
        }

        let mut stream = table.table.stream_columns(
            Arc::from(logical_fields),
            visible_row_ids,
            GatherNullPolicy::IncludeNulls,
        )?;

        let mut batches = Vec::new();
        while let Some(chunk) = stream.next_batch()? {
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(chunk.batch().num_columns() + 1);

            let mut row_id_builder = UInt64Builder::with_capacity(chunk.len());
            for row_id in chunk.row_ids() {
                row_id_builder.append_value(*row_id);
            }
            arrays.push(Arc::new(row_id_builder.finish()) as ArrayRef);

            let chunk_batch = chunk.into_batch();
            for column_array in chunk_batch.columns() {
                arrays.push(column_array.clone());
            }

            let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
            batches.push(batch);
        }

        Ok(batches)
    }

    /// Append batches directly to a table, preserving row_ids from the batches.
    /// This is used for transaction seeding where we need to preserve existing row_ids.
    pub fn append_batches_with_row_ids(
        &self,
        table_name: &str,
        batches: Vec<RecordBatch>,
    ) -> Result<usize> {
        let (_, canonical_name) = canonical_table_name(table_name)?;
        let table = self.lookup_table(&canonical_name)?;

        let mut total_rows = 0;
        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }

            // Verify the batch has a row_id column
            let _row_id_idx = batch.schema().index_of(ROW_ID_COLUMN_NAME).map_err(|_| {
                Error::InvalidArgumentError(
                    "batch must contain row_id column for direct append".into(),
                )
            })?;

            // Append the batch directly to the underlying table
            table.table.append(&batch)?;
            total_rows += batch.num_rows();
        }

        Ok(total_rows)
    }

    pub fn update(&self, plan: UpdatePlan) -> Result<RuntimeStatementResult<P>> {
        let snapshot = self.txn_manager.begin_transaction();
        match self.update_with_snapshot(plan, snapshot) {
            Ok(result) => {
                self.txn_manager.mark_committed(snapshot.txn_id);
                Ok(result)
            }
            Err(err) => {
                self.txn_manager.mark_aborted(snapshot.txn_id);
                Err(err)
            }
        }
    }

    pub fn update_with_snapshot(
        &self,
        plan: UpdatePlan,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        let UpdatePlan {
            table,
            assignments,
            filter,
        } = plan;
        let (display_name, canonical_name) = canonical_table_name(&table)?;
        let table = self.lookup_table(&canonical_name)?;
        if let Some(filter) = filter {
            self.update_filtered_rows(
                table.as_ref(),
                display_name,
                canonical_name,
                assignments,
                filter,
                snapshot,
            )
        } else {
            self.update_all_rows(
                table.as_ref(),
                display_name,
                canonical_name,
                assignments,
                snapshot,
            )
        }
    }

    pub fn delete(&self, plan: DeletePlan) -> Result<RuntimeStatementResult<P>> {
        let snapshot = self.txn_manager.begin_transaction();
        match self.delete_with_snapshot(plan, snapshot) {
            Ok(result) => {
                self.txn_manager.mark_committed(snapshot.txn_id);
                Ok(result)
            }
            Err(err) => {
                self.txn_manager.mark_aborted(snapshot.txn_id);
                Err(err)
            }
        }
    }

    pub fn delete_with_snapshot(
        &self,
        plan: DeletePlan,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        // Get table - will be checked against snapshot during actual deletion
        let table = match self.tables.read().unwrap().get(&canonical_name) {
            Some(t) => Arc::clone(t),
            None => return Err(Error::NotFound),
        };
        match plan.filter {
            Some(filter) => self.delete_filtered_rows(
                table.as_ref(),
                display_name,
                canonical_name.clone(),
                filter,
                snapshot,
            ),
            None => self.delete_all_rows(table.as_ref(), display_name, canonical_name, snapshot),
        }
    }

    pub fn table_handle(self: &Arc<Self>, name: &str) -> Result<RuntimeTableHandle<P>> {
        RuntimeTableHandle::new(Arc::clone(self), name)
    }

    pub fn execute_select(self: &Arc<Self>, plan: SelectPlan) -> Result<SelectExecution<P>> {
        let snapshot = self.default_snapshot();
        self.execute_select_with_snapshot(plan, snapshot)
    }

    pub fn execute_select_with_snapshot(
        self: &Arc<Self>,
        plan: SelectPlan,
        snapshot: TransactionSnapshot,
    ) -> Result<SelectExecution<P>> {
        // Handle SELECT without FROM clause (e.g., SELECT 42, SELECT {'a': 1})
        if plan.tables.is_empty() {
            let provider: Arc<dyn TableProvider<P>> = Arc::new(ContextProvider {
                context: Arc::clone(self),
            });
            let executor = QueryExecutor::new(provider);
            // No row filter needed since there's no table
            return executor.execute_select_with_filter(plan, None);
        }

        // Resolve canonical names for all tables (don't verify existence - executor will handle it with snapshot)
        let mut canonical_tables = Vec::new();
        for table_ref in &plan.tables {
            let qualified = table_ref.qualified_name();
            let (_display, canonical) = canonical_table_name(&qualified)?;
            // Parse canonical back into schema.table
            let parts: Vec<&str> = canonical.split('.').collect();
            let canon_ref = if parts.len() >= 2 {
                llkv_plan::TableRef::new(parts[0], parts[1])
            } else {
                llkv_plan::TableRef::new("", &canonical)
            };
            canonical_tables.push(canon_ref);
        }

        let mut canonical_plan = plan.clone();
        canonical_plan.tables = canonical_tables;

        let provider: Arc<dyn TableProvider<P>> = Arc::new(ContextProvider {
            context: Arc::clone(self),
        });
        let executor = QueryExecutor::new(provider);

        // For single-table queries, apply MVCC filtering
        let row_filter: Option<Arc<dyn RowIdFilter<P>>> = if canonical_plan.tables.len() == 1 {
            Some(Arc::new(MvccRowIdFilter::new(
                Arc::clone(&self.txn_manager),
                snapshot,
            )))
        } else {
            // TODO: Extend MVCC filtering once joins propagate per-table snapshots.
            // Multi-table plans rely on executor-level visibility checks
            None
        };

        executor.execute_select_with_filter(canonical_plan, row_filter)
    }

    fn create_table_from_columns(
        &self,
        display_name: String,
        canonical_name: String,
        columns: Vec<ColumnSpec>,
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

    fn create_table_from_batches(
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

    fn rebuild_executor_table_with_unique(
        table: &ExecutorTable<P>,
        field_id: FieldId,
    ) -> Option<Arc<ExecutorTable<P>>> {
        let mut columns = table.schema.columns.clone();
        let mut found = false;
        for column in &mut columns {
            if column.field_id == field_id {
                column.unique = true;
                found = true;
                break;
            }
        }
        if !found {
            return None;
        }

        let schema = Arc::new(ExecutorSchema {
            columns,
            lookup: table.schema.lookup.clone(),
        });

        let next_row_id = table.next_row_id.load(Ordering::SeqCst);
        let total_rows = table.total_rows.load(Ordering::SeqCst);
        let uniques = table.multi_column_uniques();

        Some(Arc::new(ExecutorTable {
            table: Arc::clone(&table.table),
            schema,
            next_row_id: AtomicU64::new(next_row_id),
            total_rows: AtomicU64::new(total_rows),
            multi_column_uniques: RwLock::new(uniques),
        }))
    }

    fn record_table_with_new_rows(&self, txn_id: TxnId, canonical_name: String) {
        if txn_id == TXN_ID_AUTO_COMMIT {
            return;
        }

        let mut guard = self.txn_tables_with_new_rows.write().unwrap();
        guard.entry(txn_id).or_default().insert(canonical_name);
    }

    fn collect_rows_created_by_txn(
        &self,
        table: &ExecutorTable<P>,
        txn_id: TxnId,
    ) -> Result<Vec<Vec<PlanValue>>> {
        if txn_id == TXN_ID_AUTO_COMMIT {
            return Ok(Vec::new());
        }

        if table.schema.columns.is_empty() {
            return Ok(Vec::new());
        }

        let Some(first_field_id) = table.schema.first_field_id() else {
            return Ok(Vec::new());
        };
        let filter_expr = full_table_scan_filter(first_field_id);

        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        if row_ids.is_empty() {
            return Ok(Vec::new());
        }

        let table_id = table.table.table_id();
        let mut logical_fields: Vec<LogicalFieldId> =
            Vec::with_capacity(table.schema.columns.len() + 2);
        logical_fields.push(LogicalFieldId::for_mvcc_created_by(table_id));
        logical_fields.push(LogicalFieldId::for_mvcc_deleted_by(table_id));
        for column in &table.schema.columns {
            logical_fields.push(LogicalFieldId::for_user(table_id, column.field_id));
        }

        let logical_fields: Arc<[LogicalFieldId]> = logical_fields.into();
        let mut stream = table.table.stream_columns(
            Arc::clone(&logical_fields),
            row_ids,
            GatherNullPolicy::IncludeNulls,
        )?;

        let mut rows = Vec::new();
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            if batch.num_columns() < table.schema.columns.len() + 2 {
                continue;
            }

            let created_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("missing created_by column in MVCC data".into()))?;
            let deleted_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("missing deleted_by column in MVCC data".into()))?;

            for row_idx in 0..batch.num_rows() {
                let created_by = if created_col.is_null(row_idx) {
                    TXN_ID_AUTO_COMMIT
                } else {
                    created_col.value(row_idx)
                };
                if created_by != txn_id {
                    continue;
                }

                let deleted_by = if deleted_col.is_null(row_idx) {
                    TXN_ID_NONE
                } else {
                    deleted_col.value(row_idx)
                };
                if deleted_by != TXN_ID_NONE {
                    continue;
                }

                let mut row_values = Vec::with_capacity(table.schema.columns.len());
                for col_idx in 0..table.schema.columns.len() {
                    let array = batch.column(col_idx + 2);
                    let value = llkv_plan::plan_value_from_array(array, row_idx)?;
                    row_values.push(value);
                }
                rows.push(row_values);
            }
        }

        Ok(rows)
    }

    pub(crate) fn validate_primary_keys_for_commit(&self, txn_id: TxnId) -> Result<()> {
        if txn_id == TXN_ID_AUTO_COMMIT {
            return Ok(());
        }

        let tables = {
            let mut guard = self.txn_tables_with_new_rows.write().unwrap();
            guard.remove(&txn_id)
        };

        let Some(tables) = tables else {
            return Ok(());
        };

        if tables.is_empty() {
            return Ok(());
        }

        let snapshot = TransactionSnapshot {
            txn_id: TXN_ID_AUTO_COMMIT,
            snapshot_id: self.txn_manager.last_committed(),
        };

        for table_name in tables {
            let table = self.lookup_table(&table_name)?;
            let constraint_ctx = self.build_table_constraint_context(table.as_ref())?;
            let Some(primary_key) = constraint_ctx.primary_key.as_ref() else {
                continue;
            };

            let new_rows = self.collect_rows_created_by_txn(table.as_ref(), txn_id)?;
            if new_rows.is_empty() {
                continue;
            }

            let column_order: Vec<usize> = (0..table.schema.columns.len()).collect();
            self.constraint_service.validate_primary_key_rows(
                &constraint_ctx.schema_field_ids,
                primary_key,
                &column_order,
                &new_rows,
                |field_ids| self.scan_multi_column_values(table.as_ref(), field_ids, snapshot),
            )?;
        }

        Ok(())
    }

    pub(crate) fn clear_transaction_state(&self, txn_id: TxnId) {
        if txn_id == TXN_ID_AUTO_COMMIT {
            return;
        }

        let mut guard = self.txn_tables_with_new_rows.write().unwrap();
        guard.remove(&txn_id);
    }

    fn coerce_plan_value_for_column(
        &self,
        value: PlanValue,
        column: &ExecutorColumn,
    ) -> Result<PlanValue> {
        match value {
            PlanValue::Null => Ok(PlanValue::Null),
            PlanValue::Integer(v) => match &column.data_type {
                DataType::Int64 => Ok(PlanValue::Integer(v)),
                DataType::Float64 => Ok(PlanValue::Float(v as f64)),
                DataType::Boolean => Ok(PlanValue::Integer(if v != 0 { 1 } else { 0 })),
                DataType::Utf8 => Ok(PlanValue::String(v.to_string())),
                DataType::Date32 => {
                    let casted = i32::try_from(v).map_err(|_| {
                        Error::InvalidArgumentError(format!(
                            "integer literal out of range for DATE column '{}'",
                            column.name
                        ))
                    })?;
                    Ok(PlanValue::Integer(casted as i64))
                }
                DataType::Struct(_) => Err(Error::InvalidArgumentError(format!(
                    "cannot assign integer to STRUCT column '{}'",
                    column.name
                ))),
                _ => Ok(PlanValue::Integer(v)),
            },
            PlanValue::Float(v) => match &column.data_type {
                DataType::Int64 => Ok(PlanValue::Integer(v as i64)),
                DataType::Float64 => Ok(PlanValue::Float(v)),
                DataType::Boolean => Ok(PlanValue::Integer(if v != 0.0 { 1 } else { 0 })),
                DataType::Utf8 => Ok(PlanValue::String(v.to_string())),
                DataType::Date32 => Err(Error::InvalidArgumentError(format!(
                    "cannot assign floating-point value to DATE column '{}'",
                    column.name
                ))),
                DataType::Struct(_) => Err(Error::InvalidArgumentError(format!(
                    "cannot assign floating-point value to STRUCT column '{}'",
                    column.name
                ))),
                _ => Ok(PlanValue::Float(v)),
            },
            PlanValue::String(s) => match &column.data_type {
                DataType::Boolean => {
                    let normalized = s.trim().to_ascii_lowercase();
                    match normalized.as_str() {
                        "true" | "t" | "1" => Ok(PlanValue::Integer(1)),
                        "false" | "f" | "0" => Ok(PlanValue::Integer(0)),
                        _ => Err(Error::InvalidArgumentError(format!(
                            "cannot assign string '{}' to BOOLEAN column '{}'",
                            s, column.name
                        ))),
                    }
                }
                DataType::Utf8 => Ok(PlanValue::String(s)),
                DataType::Date32 => {
                    let days = parse_date32_literal(&s)?;
                    Ok(PlanValue::Integer(days as i64))
                }
                DataType::Int64 | DataType::Float64 => Err(Error::InvalidArgumentError(format!(
                    "cannot assign string '{}' to numeric column '{}'",
                    s, column.name
                ))),
                DataType::Struct(_) => Err(Error::InvalidArgumentError(format!(
                    "cannot assign string to STRUCT column '{}'",
                    column.name
                ))),
                _ => Ok(PlanValue::String(s)),
            },
            PlanValue::Struct(map) => match &column.data_type {
                DataType::Struct(_) => Ok(PlanValue::Struct(map)),
                _ => Err(Error::InvalidArgumentError(format!(
                    "cannot assign struct value to column '{}'",
                    column.name
                ))),
            },
        }
    }

    /// Scan a single column and materialize values into memory.
    ///
    /// NOTE: Current implementation buffers the entire result set; convert to a
    /// streaming iterator once executor-side consumers support incremental consumption.
    fn scan_column_values(
        &self,
        table: &ExecutorTable<P>,
        field_id: FieldId,
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<PlanValue>> {
        let table_id = table.table.table_id();
        use llkv_expr::{Expr, Filter, Operator};
        use std::ops::Bound;

        // Create a filter that matches all rows (unbounded range)
        let match_all_filter = Filter {
            field_id,
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
        };
        let filter_expr = Expr::Pred(match_all_filter);

        // Get all matching row_ids first
        let row_ids = match table.table.filter_row_ids(&filter_expr) {
            Ok(ids) => ids,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        // Apply MVCC filtering manually using filter_row_ids_for_snapshot
        let row_ids = filter_row_ids_for_snapshot(
            table.table.as_ref(),
            row_ids,
            &self.txn_manager,
            snapshot,
        )?;

        if row_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Gather the column values for visible rows
        let logical_field_id = LogicalFieldId::for_user(table_id, field_id);
        let row_count = row_ids.len();
        let mut stream = match table.table.stream_columns(
            vec![logical_field_id],
            row_ids,
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(stream) => stream,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        // TODO: Don't buffer all values; make this streamable
        // NOTE: Values are accumulated eagerly; revisit when `llkv-plan` supports
        // incremental parameter binding.
        let mut values = Vec::with_capacity(row_count);
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            if batch.num_columns() == 0 {
                continue;
            }
            let array = batch.column(0);
            for row_idx in 0..batch.num_rows() {
                if let Ok(value) = llkv_plan::plan_value_from_array(array, row_idx) {
                    values.push(value);
                }
            }
        }

        Ok(values)
    }

    // TODO: Make streamable; don't buffer all values in memory at once
    /// Scan a set of columns and materialize rows into memory.
    ///
    /// NOTE: Similar to [`Self::scan_column_values`], this buffers eagerly pending
    /// enhancements to the executor pipeline.
    fn scan_multi_column_values(
        &self,
        table: &ExecutorTable<P>,
        field_ids: &[FieldId],
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<Vec<PlanValue>>> {
        if field_ids.is_empty() {
            return Ok(Vec::new());
        }

        let table_id = table.table.table_id();
        use llkv_expr::{Expr, Filter, Operator};
        use std::ops::Bound;

        let match_all_filter = Filter {
            field_id: field_ids[0],
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
        };
        let filter_expr = Expr::Pred(match_all_filter);

        let row_ids = match table.table.filter_row_ids(&filter_expr) {
            Ok(ids) => ids,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let row_ids = filter_row_ids_for_snapshot(
            table.table.as_ref(),
            row_ids,
            &self.txn_manager,
            snapshot,
        )?;

        if row_ids.is_empty() {
            return Ok(Vec::new());
        }

        let logical_field_ids: Vec<_> = field_ids
            .iter()
            .map(|&fid| LogicalFieldId::for_user(table_id, fid))
            .collect();

        let total_rows = row_ids.len();
        let mut stream = match table.table.stream_columns(
            logical_field_ids,
            row_ids,
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(stream) => stream,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let mut rows = vec![Vec::with_capacity(field_ids.len()); total_rows];
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            if batch.num_columns() == 0 {
                continue;
            }

            let base = chunk.row_offset();
            let local_len = batch.num_rows();
            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                for local_idx in 0..local_len {
                    let target_index = base + local_idx;
                    debug_assert!(
                        target_index < rows.len(),
                        "stream chunk produced out-of-bounds row index"
                    );
                    if let Some(row) = rows.get_mut(target_index) {
                        match llkv_plan::plan_value_from_array(array, local_idx) {
                            Ok(value) => row.push(value),
                            Err(_) => row.push(PlanValue::Null),
                        }
                    }
                }
            }
        }

        Ok(rows)
    }

    fn collect_row_values_for_ids(
        &self,
        table: &ExecutorTable<P>,
        row_ids: &[RowId],
        field_ids: &[FieldId],
    ) -> Result<Vec<Vec<PlanValue>>> {
        if row_ids.is_empty() || field_ids.is_empty() {
            return Ok(Vec::new());
        }

        let table_id = table.table.table_id();
        let logical_field_ids: Vec<LogicalFieldId> = field_ids
            .iter()
            .map(|&fid| LogicalFieldId::for_user(table_id, fid))
            .collect();

        let mut stream = match table.table.stream_columns(
            logical_field_ids.clone(),
            row_ids.to_vec(),
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(stream) => stream,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let mut rows = vec![Vec::with_capacity(field_ids.len()); row_ids.len()];
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            let base = chunk.row_offset();
            let local_len = batch.num_rows();
            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                for local_idx in 0..local_len {
                    let target_index = base + local_idx;
                    if let Some(row) = rows.get_mut(target_index) {
                        let value = llkv_plan::plan_value_from_array(array, local_idx)?;
                        row.push(value);
                    }
                }
            }
        }

        Ok(rows)
    }

    fn collect_visible_child_rows(
        &self,
        table: &ExecutorTable<P>,
        field_ids: &[FieldId],
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<(RowId, Vec<PlanValue>)>> {
        if field_ids.is_empty() {
            return Ok(Vec::new());
        }

        let anchor_field = field_ids[0];
        let filter_expr = full_table_scan_filter(anchor_field);
        let raw_row_ids = match table.table.filter_row_ids(&filter_expr) {
            Ok(ids) => ids,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let visible_row_ids = filter_row_ids_for_snapshot(
            table.table.as_ref(),
            raw_row_ids,
            &self.txn_manager,
            snapshot,
        )?;

        if visible_row_ids.is_empty() {
            return Ok(Vec::new());
        }

        let table_id = table.table.table_id();
        let logical_field_ids: Vec<LogicalFieldId> = field_ids
            .iter()
            .map(|&fid| LogicalFieldId::for_user(table_id, fid))
            .collect();

        let mut stream = match table.table.stream_columns(
            logical_field_ids.clone(),
            visible_row_ids.clone(),
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(stream) => stream,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let mut rows = vec![Vec::with_capacity(field_ids.len()); visible_row_ids.len()];
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            let base = chunk.row_offset();
            let local_len = batch.num_rows();
            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                for local_idx in 0..local_len {
                    let target_index = base + local_idx;
                    if let Some(row) = rows.get_mut(target_index) {
                        let value = llkv_plan::plan_value_from_array(array, local_idx)?;
                        row.push(value);
                    }
                }
            }
        }

        Ok(visible_row_ids.into_iter().zip(rows).collect())
    }

    fn build_table_constraint_context(
        &self,
        table: &ExecutorTable<P>,
    ) -> Result<TableConstraintContext> {
        let schema_field_ids: Vec<FieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| column.field_id)
            .collect();

        let column_constraints: Vec<InsertColumnConstraint> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .map(|(idx, column)| InsertColumnConstraint {
                schema_index: idx,
                column: ConstraintColumnInfo {
                    name: column.name.clone(),
                    field_id: column.field_id,
                    data_type: column.data_type.clone(),
                    nullable: column.nullable,
                    check_expr: column.check_expr.clone(),
                },
            })
            .collect();

        let unique_columns: Vec<InsertUniqueColumn> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .filter(|(_, column)| column.unique && !column.primary_key)
            .map(|(idx, column)| InsertUniqueColumn {
                schema_index: idx,
                field_id: column.field_id,
                name: column.name.clone(),
            })
            .collect();

        let mut multi_column_uniques: Vec<InsertMultiColumnUnique> = Vec::new();
        for constraint in table.multi_column_uniques() {
            if constraint.column_indices.is_empty() {
                continue;
            }

            let mut schema_indices = Vec::with_capacity(constraint.column_indices.len());
            let mut field_ids = Vec::with_capacity(constraint.column_indices.len());
            let mut column_names = Vec::with_capacity(constraint.column_indices.len());
            for &col_idx in &constraint.column_indices {
                let column = table.schema.columns.get(col_idx).ok_or_else(|| {
                    Error::Internal(format!(
                        "multi-column UNIQUE constraint references invalid column index {}",
                        col_idx
                    ))
                })?;
                schema_indices.push(col_idx);
                field_ids.push(column.field_id);
                column_names.push(column.name.clone());
            }

            multi_column_uniques.push(InsertMultiColumnUnique {
                schema_indices,
                field_ids,
                column_names,
            });
        }

        let primary_indices: Vec<usize> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .filter(|(_, column)| column.primary_key)
            .map(|(idx, _)| idx)
            .collect();

        let primary_key = if primary_indices.is_empty() {
            None
        } else {
            let mut field_ids = Vec::with_capacity(primary_indices.len());
            let mut column_names = Vec::with_capacity(primary_indices.len());
            for &idx in &primary_indices {
                let column = table.schema.columns.get(idx).ok_or_else(|| {
                    Error::Internal(format!(
                        "primary key references invalid column index {}",
                        idx
                    ))
                })?;
                field_ids.push(column.field_id);
                column_names.push(column.name.clone());
            }

            Some(InsertMultiColumnUnique {
                schema_indices: primary_indices.clone(),
                field_ids,
                column_names,
            })
        };

        Ok(TableConstraintContext {
            schema_field_ids,
            column_constraints,
            unique_columns,
            multi_column_uniques,
            primary_key,
        })
    }

    fn insert_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        mut rows: Vec<Vec<PlanValue>>,
        columns: Vec<String>,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        if rows.is_empty() {
            return Err(Error::InvalidArgumentError(
                "INSERT requires at least one row".into(),
            ));
        }

        let column_order = resolve_insert_columns(&columns, table.schema.as_ref())?;
        let expected_len = column_order.len();
        for row in &rows {
            if row.len() != expected_len {
                return Err(Error::InvalidArgumentError(format!(
                    "expected {} values in INSERT row, found {}",
                    expected_len,
                    row.len()
                )));
            }
        }

        for row in rows.iter_mut() {
            for (position, value) in row.iter_mut().enumerate() {
                let schema_index = column_order
                    .get(position)
                    .copied()
                    .ok_or_else(|| Error::Internal("invalid INSERT column index mapping".into()))?;
                let column = table.schema.columns.get(schema_index).ok_or_else(|| {
                    Error::Internal(format!(
                        "INSERT column index {} out of bounds for table '{}'",
                        schema_index, display_name
                    ))
                })?;
                let normalized = normalize_insert_value_for_column(column, value.clone())?;
                *value = normalized;
            }
        }

        let constraint_ctx = self.build_table_constraint_context(table)?;
        let primary_key_spec = constraint_ctx.primary_key.as_ref();

        if display_name == "keys" {
            tracing::trace!(
                "[KEYS] Validating constraints for {} row(s) before insert",
                rows.len()
            );
            for (i, row) in rows.iter().enumerate() {
                tracing::trace!("[KEYS]   row[{}]: {:?}", i, row);
            }
        }

        self.constraint_service.validate_insert_constraints(
            &constraint_ctx.schema_field_ids,
            &constraint_ctx.column_constraints,
            &constraint_ctx.unique_columns,
            &constraint_ctx.multi_column_uniques,
            primary_key_spec,
            &column_order,
            &rows,
            |field_id| self.scan_column_values(table, field_id, snapshot),
            |field_ids| self.scan_multi_column_values(table, field_ids, snapshot),
        )?;

        self.check_foreign_keys_on_insert(table, &display_name, &rows, &column_order, snapshot)?;

        let row_count = rows.len();
        let mut column_values: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(row_count); table.schema.columns.len()];
        for row in rows {
            for (idx, value) in row.into_iter().enumerate() {
                let dest_index = column_order[idx];
                column_values[dest_index].push(value);
            }
        }

        let start_row = table.next_row_id.load(Ordering::SeqCst);

        // Build MVCC columns using helper
        let (row_id_array, created_by_array, deleted_by_array) =
            mvcc::build_insert_mvcc_columns(row_count, start_row, snapshot.txn_id, TXN_ID_NONE);

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(column_values.len() + 3);
        arrays.push(row_id_array);
        arrays.push(created_by_array);
        arrays.push(deleted_by_array);

        let mut fields: Vec<Field> = Vec::with_capacity(column_values.len() + 3);
        fields.extend(mvcc::build_mvcc_fields());

        for (column, values) in table.schema.columns.iter().zip(column_values.into_iter()) {
            let array = build_array_for_column(&column.data_type, &values)?;
            let field = mvcc::build_field_with_metadata(
                &column.name,
                column.data_type.clone(),
                column.nullable,
                column.field_id,
            );
            arrays.push(array);
            fields.push(field);
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        tracing::trace!(
            table_name = %display_name,
            store_ptr = ?std::ptr::addr_of!(*table.table.store()),
            "About to call table.append"
        );
        table.table.append(&batch)?;
        table
            .next_row_id
            .store(start_row + row_count as u64, Ordering::SeqCst);
        table
            .total_rows
            .fetch_add(row_count as u64, Ordering::SeqCst);

        self.record_table_with_new_rows(snapshot.txn_id, canonical_name);

        Ok(RuntimeStatementResult::Insert {
            table_name: display_name,
            rows_inserted: row_count,
        })
    }

    fn insert_batches(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        batches: Vec<RecordBatch>,
        columns: Vec<String>,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        if batches.is_empty() {
            return Ok(RuntimeStatementResult::Insert {
                table_name: display_name,
                rows_inserted: 0,
            });
        }

        let expected_len = if columns.is_empty() {
            table.schema.columns.len()
        } else {
            columns.len()
        };
        let mut total_rows_inserted = 0usize;

        for batch in batches {
            if batch.num_columns() != expected_len {
                return Err(Error::InvalidArgumentError(format!(
                    "expected {} columns in INSERT batch, found {}",
                    expected_len,
                    batch.num_columns()
                )));
            }
            let row_count = batch.num_rows();
            if row_count == 0 {
                continue;
            }
            let mut rows: Vec<Vec<PlanValue>> = Vec::with_capacity(row_count);
            for row_idx in 0..row_count {
                let mut row: Vec<PlanValue> = Vec::with_capacity(expected_len);
                for col_idx in 0..expected_len {
                    let array = batch.column(col_idx);
                    row.push(llkv_plan::plan_value_from_array(array, row_idx)?);
                }
                rows.push(row);
            }

            match self.insert_rows(
                table,
                display_name.clone(),
                canonical_name.clone(),
                rows,
                columns.clone(),
                snapshot,
            )? {
                RuntimeStatementResult::Insert { rows_inserted, .. } => {
                    total_rows_inserted += rows_inserted;
                }
                _ => unreachable!("insert_rows must return Insert result"),
            }
        }

        Ok(RuntimeStatementResult::Insert {
            table_name: display_name,
            rows_inserted: total_rows_inserted,
        })
    }

    fn update_filtered_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        assignments: Vec<ColumnAssignment>,
        filter: LlkvExpr<'static, String>,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        if assignments.is_empty() {
            return Err(Error::InvalidArgumentError(
                "UPDATE requires at least one assignment".into(),
            ));
        }

        let schema = table.schema.as_ref();
        let filter_expr = translate_predicate(filter, schema)?;

        let mut seen_columns: FxHashSet<String> =
            FxHashSet::with_capacity_and_hasher(assignments.len(), Default::default());
        let mut prepared: Vec<(ExecutorColumn, PreparedAssignmentValue)> =
            Vec::with_capacity(assignments.len());
        let mut scalar_exprs: Vec<ScalarExpr<FieldId>> = Vec::new();

        for assignment in assignments {
            let normalized = assignment.column.to_ascii_lowercase();
            if !seen_columns.insert(normalized.clone()) {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column '{}' in UPDATE assignments",
                    assignment.column
                )));
            }
            let column = table.schema.resolve(&assignment.column).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "unknown column '{}' in UPDATE",
                    assignment.column
                ))
            })?;

            match assignment.value {
                AssignmentValue::Literal(value) => {
                    prepared.push((column.clone(), PreparedAssignmentValue::Literal(value)));
                }
                AssignmentValue::Expression(expr) => {
                    let translated = translate_scalar(&expr, schema)?;
                    let expr_index = scalar_exprs.len();
                    scalar_exprs.push(translated);
                    prepared.push((
                        column.clone(),
                        PreparedAssignmentValue::Expression { expr_index },
                    ));
                }
            }
        }

        let (row_ids, mut expr_values) =
            self.collect_update_rows(table, &filter_expr, &scalar_exprs, snapshot)?;

        if row_ids.is_empty() {
            return Ok(RuntimeStatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let row_count = row_ids.len();
        let table_id = table.table.table_id();
        let logical_fields: Vec<LogicalFieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| LogicalFieldId::for_user(table_id, column.field_id))
            .collect();

        let mut stream = table.table.stream_columns(
            logical_fields.clone(),
            row_ids.clone(),
            GatherNullPolicy::IncludeNulls,
        )?;

        let mut new_rows: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(table.schema.columns.len()); row_count];
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            let base = chunk.row_offset();
            let local_len = batch.num_rows();
            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                for local_idx in 0..local_len {
                    let target_index = base + local_idx;
                    debug_assert!(
                        target_index < new_rows.len(),
                        "column stream produced out-of-range row index"
                    );
                    if let Some(row) = new_rows.get_mut(target_index) {
                        let value = llkv_plan::plan_value_from_array(array, local_idx)?;
                        row.push(value);
                    }
                }
            }
        }
        debug_assert!(
            new_rows
                .iter()
                .all(|row| row.len() == table.schema.columns.len())
        );

        tracing::trace!(
            table = %display_name,
            row_count,
            rows = ?new_rows,
            "update_filtered_rows captured source rows"
        );

        let constraint_ctx = self.build_table_constraint_context(table)?;
        let primary_key_spec = constraint_ctx.primary_key.as_ref();
        let mut original_primary_key_keys: Vec<Option<UniqueKey>> = Vec::new();
        if let Some(pk) = primary_key_spec {
            original_primary_key_keys.reserve(row_count);
            for row in &new_rows {
                let mut values = Vec::with_capacity(pk.schema_indices.len());
                for &idx in &pk.schema_indices {
                    let value = row.get(idx).cloned().unwrap_or(PlanValue::Null);
                    values.push(value);
                }
                let key = build_composite_unique_key(&values, &pk.column_names)?;
                original_primary_key_keys.push(key);
            }
        }

        let column_positions: FxHashMap<FieldId, usize> = FxHashMap::from_iter(
            table
                .schema
                .columns
                .iter()
                .enumerate()
                .map(|(idx, column)| (column.field_id, idx)),
        );

        // Extract field IDs being updated for FK validation later
        let updated_field_ids: Vec<FieldId> =
            prepared.iter().map(|(column, _)| column.field_id).collect();

        for (column, value) in prepared {
            let column_index =
                column_positions
                    .get(&column.field_id)
                    .copied()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "column '{}' missing in table schema during UPDATE",
                            column.name
                        ))
                    })?;

            let values = match value {
                PreparedAssignmentValue::Literal(lit) => vec![lit; row_count],
                PreparedAssignmentValue::Expression { expr_index } => {
                    let column_values = expr_values.get_mut(expr_index).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "expression assignment value missing during UPDATE".into(),
                        )
                    })?;
                    if column_values.len() != row_count {
                        return Err(Error::InvalidArgumentError(
                            "expression result count did not match targeted row count".into(),
                        ));
                    }
                    mem::take(column_values)
                }
            };

            for (row_idx, new_value) in values.into_iter().enumerate() {
                if let Some(row) = new_rows.get_mut(row_idx) {
                    let coerced = self.coerce_plan_value_for_column(new_value, &column)?;
                    row[column_index] = coerced;
                }
            }
        }

        let column_names: Vec<String> = table
            .schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let column_order = resolve_insert_columns(&column_names, table.schema.as_ref())?;
        self.constraint_service.validate_row_level_constraints(
            &constraint_ctx.schema_field_ids,
            &constraint_ctx.column_constraints,
            &column_order,
            &new_rows,
        )?;

        if let Some(pk) = primary_key_spec {
            self.constraint_service.validate_update_primary_keys(
                &constraint_ctx.schema_field_ids,
                pk,
                &column_order,
                &new_rows,
                &original_primary_key_keys,
                |field_ids| self.scan_multi_column_values(table, field_ids, snapshot),
            )?;
        }

        // Check foreign key constraints before updating (update_filtered_rows)
        self.check_foreign_keys_on_update(
            table,
            &display_name,
            &canonical_name,
            &row_ids,
            &updated_field_ids,
            snapshot,
        )?;

        let _ = self.apply_delete(
            table,
            display_name.clone(),
            canonical_name.clone(),
            row_ids.clone(),
            snapshot,
            false,
        )?;

        let _ = self.insert_rows(
            table,
            display_name.clone(),
            canonical_name,
            new_rows,
            column_names,
            snapshot,
        )?;

        Ok(RuntimeStatementResult::Update {
            table_name: display_name,
            rows_updated: row_count,
        })
    }

    fn update_all_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        assignments: Vec<ColumnAssignment>,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        if assignments.is_empty() {
            return Err(Error::InvalidArgumentError(
                "UPDATE requires at least one assignment".into(),
            ));
        }

        let total_rows = table.total_rows.load(Ordering::SeqCst);
        let total_rows_usize = usize::try_from(total_rows).map_err(|_| {
            Error::InvalidArgumentError("table row count exceeds supported range".into())
        })?;
        if total_rows_usize == 0 {
            return Ok(RuntimeStatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let schema = table.schema.as_ref();

        let mut seen_columns: FxHashSet<String> =
            FxHashSet::with_capacity_and_hasher(assignments.len(), Default::default());
        let mut prepared: Vec<(ExecutorColumn, PreparedAssignmentValue)> =
            Vec::with_capacity(assignments.len());
        let mut scalar_exprs: Vec<ScalarExpr<FieldId>> = Vec::new();
        let mut first_field_id: Option<FieldId> = None;

        for assignment in assignments {
            let normalized = assignment.column.to_ascii_lowercase();
            if !seen_columns.insert(normalized.clone()) {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column '{}' in UPDATE assignments",
                    assignment.column
                )));
            }
            let column = table.schema.resolve(&assignment.column).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "unknown column '{}' in UPDATE",
                    assignment.column
                ))
            })?;
            if first_field_id.is_none() {
                first_field_id = Some(column.field_id);
            }

            match assignment.value {
                AssignmentValue::Literal(value) => {
                    prepared.push((column.clone(), PreparedAssignmentValue::Literal(value)));
                }
                AssignmentValue::Expression(expr) => {
                    let translated = translate_scalar(&expr, schema)?;
                    let expr_index = scalar_exprs.len();
                    scalar_exprs.push(translated);
                    prepared.push((
                        column.clone(),
                        PreparedAssignmentValue::Expression { expr_index },
                    ));
                }
            }
        }

        let anchor_field = first_field_id.ok_or_else(|| {
            Error::InvalidArgumentError("UPDATE requires at least one target column".into())
        })?;

        let filter_expr = full_table_scan_filter(anchor_field);
        let (row_ids, mut expr_values) =
            self.collect_update_rows(table, &filter_expr, &scalar_exprs, snapshot)?;

        if row_ids.is_empty() {
            return Ok(RuntimeStatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let row_count = row_ids.len();
        let table_id = table.table.table_id();
        let logical_fields: Vec<LogicalFieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| LogicalFieldId::for_user(table_id, column.field_id))
            .collect();

        let mut stream = table.table.stream_columns(
            logical_fields.clone(),
            row_ids.clone(),
            GatherNullPolicy::IncludeNulls,
        )?;

        let mut new_rows: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(table.schema.columns.len()); row_count];
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            let base = chunk.row_offset();
            let local_len = batch.num_rows();
            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                for local_idx in 0..local_len {
                    let target_index = base + local_idx;
                    debug_assert!(
                        target_index < new_rows.len(),
                        "column stream produced out-of-range row index"
                    );
                    if let Some(row) = new_rows.get_mut(target_index) {
                        let value = llkv_plan::plan_value_from_array(array, local_idx)?;
                        row.push(value);
                    }
                }
            }
        }
        debug_assert!(
            new_rows
                .iter()
                .all(|row| row.len() == table.schema.columns.len())
        );

        let constraint_ctx = self.build_table_constraint_context(table)?;
        let primary_key_spec = constraint_ctx.primary_key.as_ref();
        let mut original_primary_key_keys: Vec<Option<UniqueKey>> = Vec::new();
        if let Some(pk) = primary_key_spec {
            original_primary_key_keys.reserve(row_count);
            for row in &new_rows {
                let mut values = Vec::with_capacity(pk.schema_indices.len());
                for &idx in &pk.schema_indices {
                    let value = row.get(idx).cloned().unwrap_or(PlanValue::Null);
                    values.push(value);
                }
                let key = build_composite_unique_key(&values, &pk.column_names)?;
                original_primary_key_keys.push(key);
            }
        }

        let column_positions: FxHashMap<FieldId, usize> = FxHashMap::from_iter(
            table
                .schema
                .columns
                .iter()
                .enumerate()
                .map(|(idx, column)| (column.field_id, idx)),
        );

        // Extract field IDs being updated for FK validation later (update_all_rows)
        let updated_field_ids: Vec<FieldId> =
            prepared.iter().map(|(column, _)| column.field_id).collect();

        for (column, value) in prepared {
            let column_index =
                column_positions
                    .get(&column.field_id)
                    .copied()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "column '{}' missing in table schema during UPDATE",
                            column.name
                        ))
                    })?;

            let values = match value {
                PreparedAssignmentValue::Literal(lit) => vec![lit; row_count],
                PreparedAssignmentValue::Expression { expr_index } => {
                    let column_values = expr_values.get_mut(expr_index).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "expression assignment value missing during UPDATE".into(),
                        )
                    })?;
                    if column_values.len() != row_count {
                        return Err(Error::InvalidArgumentError(
                            "expression result count did not match targeted row count".into(),
                        ));
                    }
                    mem::take(column_values)
                }
            };

            for (row_idx, new_value) in values.into_iter().enumerate() {
                if let Some(row) = new_rows.get_mut(row_idx) {
                    let coerced = self.coerce_plan_value_for_column(new_value, &column)?;
                    row[column_index] = coerced;
                }
            }
        }

        let column_names: Vec<String> = table
            .schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let column_order = resolve_insert_columns(&column_names, table.schema.as_ref())?;
        self.constraint_service.validate_row_level_constraints(
            &constraint_ctx.schema_field_ids,
            &constraint_ctx.column_constraints,
            &column_order,
            &new_rows,
        )?;

        if let Some(pk) = primary_key_spec {
            self.constraint_service.validate_update_primary_keys(
                &constraint_ctx.schema_field_ids,
                pk,
                &column_order,
                &new_rows,
                &original_primary_key_keys,
                |field_ids| self.scan_multi_column_values(table, field_ids, snapshot),
            )?;
        }

        // Check foreign key constraints before updating (update_all_rows)
        self.check_foreign_keys_on_update(
            table,
            &display_name,
            &canonical_name,
            &row_ids,
            &updated_field_ids,
            snapshot,
        )?;

        let _ = self.apply_delete(
            table,
            display_name.clone(),
            canonical_name.clone(),
            row_ids.clone(),
            snapshot,
            false,
        )?;

        let _ = self.insert_rows(
            table,
            display_name.clone(),
            canonical_name,
            new_rows,
            column_names,
            snapshot,
        )?;

        Ok(RuntimeStatementResult::Update {
            table_name: display_name,
            rows_updated: row_count,
        })
    }

    fn delete_filtered_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        filter: LlkvExpr<'static, String>,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        let schema = table.schema.as_ref();
        let filter_expr = translate_predicate(filter, schema)?;
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        let row_ids = self.filter_visible_row_ids(table, row_ids, snapshot)?;
        tracing::trace!(
            table = %display_name,
            rows = row_ids.len(),
            "delete_filtered_rows collected row ids"
        );
        self.apply_delete(table, display_name, canonical_name, row_ids, snapshot, true)
    }

    fn delete_all_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        let total_rows = table.total_rows.load(Ordering::SeqCst);
        if total_rows == 0 {
            return Ok(RuntimeStatementResult::Delete {
                table_name: display_name,
                rows_deleted: 0,
            });
        }

        let anchor_field = table.schema.first_field_id().ok_or_else(|| {
            Error::InvalidArgumentError("DELETE requires a table with at least one column".into())
        })?;
        let filter_expr = full_table_scan_filter(anchor_field);
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        let row_ids = self.filter_visible_row_ids(table, row_ids, snapshot)?;
        self.apply_delete(table, display_name, canonical_name, row_ids, snapshot, true)
    }

    fn apply_delete(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        row_ids: Vec<RowId>,
        snapshot: TransactionSnapshot,
        enforce_foreign_keys: bool,
    ) -> Result<RuntimeStatementResult<P>> {
        if row_ids.is_empty() {
            return Ok(RuntimeStatementResult::Delete {
                table_name: display_name,
                rows_deleted: 0,
            });
        }

        if enforce_foreign_keys {
            self.check_foreign_keys_on_delete(
                table,
                &display_name,
                &canonical_name,
                &row_ids,
                snapshot,
            )?;
        }

        self.detect_delete_conflicts(table, &display_name, &row_ids, snapshot)?;

        let removed = row_ids.len();

        // Build DELETE batch using helper
        let batch = mvcc::build_delete_batch(row_ids.clone(), snapshot.txn_id)?;
        table.table.append(&batch)?;

        let removed_u64 = u64::try_from(removed)
            .map_err(|_| Error::InvalidArgumentError("row count exceeds supported range".into()))?;
        table.total_rows.fetch_sub(removed_u64, Ordering::SeqCst);

        Ok(RuntimeStatementResult::Delete {
            table_name: display_name,
            rows_deleted: removed,
        })
    }

    fn check_foreign_keys_on_delete(
        &self,
        table: &ExecutorTable<P>,
        _display_name: &str,
        _canonical_name: &str,
        row_ids: &[RowId],
        snapshot: TransactionSnapshot,
    ) -> Result<()> {
        if row_ids.is_empty() {
            return Ok(());
        }

        self.constraint_service.validate_delete_foreign_keys(
            table.table.table_id(),
            row_ids,
            |request| {
                self.collect_row_values_for_ids(
                    table,
                    request.referenced_row_ids,
                    request.referenced_field_ids,
                )
            },
            |request| {
                let child_table = self.lookup_table(request.referencing_table_canonical)?;
                self.collect_visible_child_rows(
                    child_table.as_ref(),
                    request.referencing_field_ids,
                    snapshot,
                )
            },
        )
    }

    fn check_foreign_keys_on_update(
        &self,
        table: &ExecutorTable<P>,
        _display_name: &str,
        _canonical_name: &str,
        row_ids: &[RowId],
        updated_field_ids: &[FieldId],
        snapshot: TransactionSnapshot,
    ) -> Result<()> {
        if row_ids.is_empty() || updated_field_ids.is_empty() {
            return Ok(());
        }

        self.constraint_service.validate_update_foreign_keys(
            table.table.table_id(),
            row_ids,
            updated_field_ids,
            |request| {
                self.collect_row_values_for_ids(
                    table,
                    request.referenced_row_ids,
                    request.referenced_field_ids,
                )
            },
            |request| {
                let child_table = self.lookup_table(request.referencing_table_canonical)?;
                self.collect_visible_child_rows(
                    child_table.as_ref(),
                    request.referencing_field_ids,
                    snapshot,
                )
            },
        )
    }

    fn check_foreign_keys_on_insert(
        &self,
        table: &ExecutorTable<P>,
        _display_name: &str,
        rows: &[Vec<PlanValue>],
        column_order: &[usize],
        snapshot: TransactionSnapshot,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }

        let schema_field_ids: Vec<FieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| column.field_id)
            .collect();

        self.constraint_service.validate_insert_foreign_keys(
            table.table.table_id(),
            &schema_field_ids,
            column_order,
            rows,
            |request| {
                let parent_table = self.lookup_table(request.referenced_table_canonical)?;
                self.scan_multi_column_values(
                    parent_table.as_ref(),
                    request.referenced_field_ids,
                    snapshot,
                )
            },
        )
    }

    fn detect_delete_conflicts(
        &self,
        table: &ExecutorTable<P>,
        display_name: &str,
        row_ids: &[RowId],
        snapshot: TransactionSnapshot,
    ) -> Result<()> {
        if row_ids.is_empty() {
            return Ok(());
        }

        let table_id = table.table.table_id();
        let deleted_lfid = LogicalFieldId::for_mvcc_deleted_by(table_id);
        let logical_fields: Arc<[LogicalFieldId]> = Arc::from([deleted_lfid]);

        if let Err(err) = table
            .table
            .store()
            .prepare_gather_context(logical_fields.as_ref())
        {
            match err {
                Error::NotFound => return Ok(()),
                other => return Err(other),
            }
        }

        let mut stream = table.table.stream_columns(
            Arc::clone(&logical_fields),
            row_ids.to_vec(),
            GatherNullPolicy::IncludeNulls,
        )?;

        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            let window = chunk.row_ids();
            let deleted_column = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    Error::Internal(
                        "failed to read MVCC deleted_by column for conflict detection".into(),
                    )
                })?;

            for (idx, row_id) in window.iter().enumerate() {
                let deleted_by = if deleted_column.is_null(idx) {
                    TXN_ID_NONE
                } else {
                    deleted_column.value(idx)
                };

                if deleted_by == TXN_ID_NONE || deleted_by == snapshot.txn_id {
                    continue;
                }

                let status = self.txn_manager.status(deleted_by);
                if !status.is_active() {
                    continue;
                }

                tracing::debug!(
                    "[MVCC] delete conflict: table='{}' row_id={} deleted_by={} status={:?} current_txn={}",
                    display_name,
                    row_id,
                    deleted_by,
                    status,
                    snapshot.txn_id
                );

                return Err(Error::TransactionContextError(format!(
                    "transaction conflict on table '{}' for row {}: row locked by transaction {} ({:?})",
                    display_name, row_id, deleted_by, status
                )));
            }
        }

        Ok(())
    }

    fn collect_update_rows(
        &self,
        table: &ExecutorTable<P>,
        filter_expr: &LlkvExpr<'static, FieldId>,
        expressions: &[ScalarExpr<FieldId>],
        snapshot: TransactionSnapshot,
    ) -> Result<(Vec<RowId>, Vec<Vec<PlanValue>>)> {
        let row_ids = table.table.filter_row_ids(filter_expr)?;
        let row_ids = self.filter_visible_row_ids(table, row_ids, snapshot)?;
        if row_ids.is_empty() {
            return Ok((row_ids, vec![Vec::new(); expressions.len()]));
        }

        if expressions.is_empty() {
            return Ok((row_ids, Vec::new()));
        }

        let mut projections: Vec<ScanProjection> = Vec::with_capacity(expressions.len());
        for (idx, expr) in expressions.iter().enumerate() {
            let alias = format!("__expr_{idx}");
            projections.push(ScanProjection::computed(expr.clone(), alias));
        }

        let mut expr_values: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(row_ids.len()); expressions.len()];
        let mut error: Option<Error> = None;
        let row_filter: Arc<dyn RowIdFilter<P>> = Arc::new(MvccRowIdFilter::new(
            Arc::clone(&self.txn_manager),
            snapshot,
        ));
        let options = ScanStreamOptions {
            include_nulls: true,
            order: None,
            row_id_filter: Some(row_filter),
        };

        table
            .table
            .scan_stream_with_exprs(&projections, filter_expr, options, |batch| {
                if error.is_some() {
                    return;
                }
                if let Err(err) = Self::collect_expression_values(&mut expr_values, batch) {
                    error = Some(err);
                }
            })?;

        if let Some(err) = error {
            return Err(err);
        }

        for values in &expr_values {
            if values.len() != row_ids.len() {
                return Err(Error::InvalidArgumentError(
                    "expression result count did not match targeted row count".into(),
                ));
            }
        }

        Ok((row_ids, expr_values))
    }

    fn collect_expression_values(
        expr_values: &mut [Vec<PlanValue>],
        batch: RecordBatch,
    ) -> Result<()> {
        for row_idx in 0..batch.num_rows() {
            for (expr_index, values) in expr_values.iter_mut().enumerate() {
                let value = llkv_plan::plan_value_from_array(batch.column(expr_index), row_idx)?;
                values.push(value);
            }
        }

        Ok(())
    }

    /// Looks up a table in the executor cache, lazily loading it from metadata if not already cached.
    /// This is the primary method for obtaining table references for query execution.
    pub fn lookup_table(&self, canonical_name: &str) -> Result<Arc<ExecutorTable<P>>> {
        // Fast path: check if table is already loaded
        {
            let tables = self.tables.read().unwrap();
            if let Some(table) = tables.get(canonical_name) {
                // Check if table has been dropped
                if self.dropped_tables.read().unwrap().contains(canonical_name) {
                    // Table was dropped - treat as not found
                    return Err(Error::NotFound);
                }
                tracing::trace!(
                    "=== LOOKUP_TABLE '{}' (cached) table_id={} columns={} context_pager={:p} ===",
                    canonical_name,
                    table.table.table_id(),
                    table.schema.columns.len(),
                    &*self.pager
                );
                return Ok(Arc::clone(table));
            }
        } // Release read lock

        // Slow path: load table from catalog (happens once per table)
        tracing::debug!(
            "[LAZY_LOAD] Loading table '{}' from catalog",
            canonical_name
        );

        // Check catalog first for table existence
        let catalog_table_id = self.catalog.table_id(canonical_name).ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown table '{}'", canonical_name))
        })?;

        let table_id = catalog_table_id;
        let table = Table::from_id_and_store(table_id, Arc::clone(&self.store))?;
        let store = table.store();
        let mut logical_fields = store.user_field_ids_for_table(table_id);
        logical_fields.sort_by_key(|lfid| lfid.field_id());
        let field_ids: Vec<FieldId> = logical_fields.iter().map(|lfid| lfid.field_id()).collect();
        let summary = self
            .catalog_service
            .table_constraint_summary(canonical_name)?;
        let TableConstraintSummaryView {
            table_meta,
            column_metas,
            constraint_records,
            multi_column_uniques,
        } = summary;
        let _table_meta = table_meta.ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown table '{}'", canonical_name))
        })?;
        let catalog_field_resolver = self.catalog.field_resolver(catalog_table_id);
        let mut metadata_primary_keys: FxHashSet<FieldId> = FxHashSet::default();
        let mut metadata_unique_fields: FxHashSet<FieldId> = FxHashSet::default();
        let mut has_primary_key_records = false;
        let mut has_single_unique_records = false;

        for record in constraint_records
            .iter()
            .filter(|record| record.is_active())
        {
            match &record.kind {
                ConstraintKind::PrimaryKey(pk) => {
                    has_primary_key_records = true;
                    for field_id in &pk.field_ids {
                        metadata_primary_keys.insert(*field_id);
                        metadata_unique_fields.insert(*field_id);
                    }
                }
                ConstraintKind::Unique(unique) => {
                    if unique.field_ids.len() == 1 {
                        has_single_unique_records = true;
                        metadata_unique_fields.insert(unique.field_ids[0]);
                    }
                }
                _ => {}
            }
        }

        // Build ExecutorSchema from metadata manager snapshots
        let mut executor_columns = Vec::new();
        let mut lookup = FxHashMap::with_capacity_and_hasher(field_ids.len(), Default::default());

        for (idx, lfid) in logical_fields.iter().enumerate() {
            let field_id = lfid.field_id();
            let normalized_index = executor_columns.len();

            let column_name = column_metas
                .get(idx)
                .and_then(|meta| meta.as_ref())
                .and_then(|meta| meta.name.clone())
                .unwrap_or_else(|| format!("col_{}", field_id));

            let normalized = column_name.to_ascii_lowercase();
            lookup.insert(normalized, normalized_index);

            let fallback_constraints: FieldConstraints = catalog_field_resolver
                .as_ref()
                .and_then(|resolver| resolver.field_constraints_by_name(&column_name))
                .unwrap_or_default();

            let metadata_primary = metadata_primary_keys.contains(&field_id);
            let primary_key = if has_primary_key_records {
                metadata_primary
            } else {
                fallback_constraints.primary_key
            };

            let metadata_unique = metadata_primary || metadata_unique_fields.contains(&field_id);
            let unique = if has_primary_key_records || has_single_unique_records {
                metadata_unique
            } else {
                fallback_constraints.primary_key || fallback_constraints.unique
            };

            let data_type = store.data_type(*lfid)?;
            let nullable = !primary_key;

            executor_columns.push(ExecutorColumn {
                name: column_name,
                data_type,
                nullable,
                primary_key,
                unique,
                field_id,
                check_expr: fallback_constraints.check_expr.clone(),
            });
        }

        let exec_schema = Arc::new(ExecutorSchema {
            columns: executor_columns,
            lookup,
        });

        // Find the maximum row_id in the table to set next_row_id correctly
        let max_row_id = {
            use arrow::array::UInt64Array;
            use llkv_column_map::store::rowid_fid;
            use llkv_column_map::store::scan::{
                PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
                PrimitiveWithRowIdsVisitor, ScanBuilder, ScanOptions,
            };

            struct MaxRowIdVisitor {
                max: RowId,
            }

            impl PrimitiveVisitor for MaxRowIdVisitor {
                fn u64_chunk(&mut self, values: &UInt64Array) {
                    for i in 0..values.len() {
                        let val = values.value(i);
                        if val > self.max {
                            self.max = val;
                        }
                    }
                }
            }

            impl PrimitiveWithRowIdsVisitor for MaxRowIdVisitor {}
            impl PrimitiveSortedVisitor for MaxRowIdVisitor {}
            impl PrimitiveSortedWithRowIdsVisitor for MaxRowIdVisitor {}

            // Scan the row_id column for any user field in this table
            let row_id_field = rowid_fid(LogicalFieldId::for_user(table_id, 1));
            let mut visitor = MaxRowIdVisitor { max: 0 };

            match ScanBuilder::new(table.store(), row_id_field)
                .options(ScanOptions::default())
                .run(&mut visitor)
            {
                Ok(_) => visitor.max,
                Err(llkv_result::Error::NotFound) => 0,
                Err(e) => {
                    tracing::warn!(
                        "[LAZY_LOAD] Failed to scan max row_id for table '{}': {}",
                        canonical_name,
                        e
                    );
                    0
                }
            }
        };

        let next_row_id = if max_row_id > 0 {
            max_row_id.saturating_add(1)
        } else {
            0
        };

        // Get the actual persisted row count from table metadata
        // This is an O(1) catalog lookup that reads ColumnDescriptor.total_row_count
        // Fallback to 0 for truly empty tables
        let total_rows = table.total_rows().unwrap_or(0);

        let executor_table = Arc::new(ExecutorTable {
            table: Arc::new(table),
            schema: exec_schema,
            next_row_id: AtomicU64::new(next_row_id),
            total_rows: AtomicU64::new(total_rows),
            multi_column_uniques: RwLock::new(Vec::new()),
        });

        if !multi_column_uniques.is_empty() {
            let executor_uniques =
                Self::build_executor_multi_column_uniques(&executor_table, &multi_column_uniques);
            executor_table.set_multi_column_uniques(executor_uniques);
        }

        // Cache the loaded table
        {
            let mut tables = self.tables.write().unwrap();
            tables.insert(canonical_name.to_string(), Arc::clone(&executor_table));
        }

        // Register fields in catalog (may already be registered from RuntimeContext::new())
        if let Some(field_resolver) = self.catalog.field_resolver(catalog_table_id) {
            for col in &executor_table.schema.columns {
                let definition = FieldDefinition::new(&col.name)
                    .with_primary_key(col.primary_key)
                    .with_unique(col.unique)
                    .with_check_expr(col.check_expr.clone());
                let _ = field_resolver.register_field(definition); // Ignore "already exists" errors
            }
            tracing::debug!(
                "[CATALOG] Registered {} field(s) for lazy-loaded table '{}'",
                executor_table.schema.columns.len(),
                canonical_name
            );
        }

        tracing::debug!(
            "[LAZY_LOAD] Loaded table '{}' (id={}) with {} columns, next_row_id={}",
            canonical_name,
            table_id,
            field_ids.len(),
            next_row_id
        );

        Ok(executor_table)
    }

    fn remove_table_entry(&self, canonical_name: &str) {
        let mut tables = self.tables.write().unwrap();
        if tables.remove(canonical_name).is_some() {
            tracing::trace!(
                "remove_table_entry: removed table '{}' from context cache",
                canonical_name
            );
        }
    }

    pub fn is_table_marked_dropped(&self, canonical_name: &str) -> bool {
        self.dropped_tables.read().unwrap().contains(canonical_name)
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
                return Ok(RuntimeStatementResult::NoOp)
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

        validate_alter_table_operation(
            &plan.operation,
            &view,
            table_id,
            &self.catalog_service,
        )?;

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

pub fn filter_row_ids_for_snapshot<P>(
    table: &Table<P>,
    row_ids: Vec<RowId>,
    txn_manager: &TxnIdManager,
    snapshot: TransactionSnapshot,
) -> Result<Vec<RowId>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    tracing::debug!(
        "[FILTER_ROWS] Filtering {} row IDs for snapshot txn_id={}, snapshot_id={}",
        row_ids.len(),
        snapshot.txn_id,
        snapshot.snapshot_id
    );

    if row_ids.is_empty() {
        return Ok(row_ids);
    }

    let table_id = table.table_id();
    let created_lfid = LogicalFieldId::for_mvcc_created_by(table_id);
    let deleted_lfid = LogicalFieldId::for_mvcc_deleted_by(table_id);
    let logical_fields: Arc<[LogicalFieldId]> = Arc::from([created_lfid, deleted_lfid]);

    if let Err(err) = table
        .store()
        .prepare_gather_context(logical_fields.as_ref())
    {
        match err {
            Error::NotFound => {
                tracing::trace!(
                    "[FILTER_ROWS] MVCC columns not found for table_id={}, treating all rows as visible",
                    table_id
                );
                return Ok(row_ids);
            }
            other => {
                tracing::error!(
                    "[FILTER_ROWS] Failed to prepare gather context: {:?}",
                    other
                );
                return Err(other);
            }
        }
    }

    let total_rows = row_ids.len();
    let mut stream = match table.stream_columns(
        Arc::clone(&logical_fields),
        row_ids,
        GatherNullPolicy::IncludeNulls,
    ) {
        Ok(stream) => stream,
        Err(err) => {
            tracing::error!("[FILTER_ROWS] stream_columns error: {:?}", err);
            return Err(err);
        }
    };

    let mut visible = Vec::with_capacity(total_rows);

    while let Some(chunk) = stream.next_batch()? {
        let batch = chunk.batch();
        let window = chunk.row_ids();

        if batch.num_columns() < 2 {
            tracing::debug!(
                "[FILTER_ROWS] version_batch has < 2 columns for table_id={}, returning window rows unfiltered",
                table_id
            );
            visible.extend_from_slice(window);
            continue;
        }

        let created_column = batch.column(0).as_any().downcast_ref::<UInt64Array>();
        let deleted_column = batch.column(1).as_any().downcast_ref::<UInt64Array>();

        if created_column.is_none() || deleted_column.is_none() {
            tracing::debug!(
                "[FILTER_ROWS] Failed to downcast MVCC columns for table_id={}, returning window rows unfiltered",
                table_id
            );
            visible.extend_from_slice(window);
            continue;
        }

        let created_column = created_column.unwrap();
        let deleted_column = deleted_column.unwrap();

        for (idx, row_id) in window.iter().enumerate() {
            let created_by = if created_column.is_null(idx) {
                TXN_ID_AUTO_COMMIT
            } else {
                created_column.value(idx)
            };
            let deleted_by = if deleted_column.is_null(idx) {
                TXN_ID_NONE
            } else {
                deleted_column.value(idx)
            };

            let version = RowVersion {
                created_by,
                deleted_by,
            };
            let is_visible = version.is_visible_for(txn_manager, snapshot);
            tracing::trace!(
                "[FILTER_ROWS] row_id={}: created_by={}, deleted_by={}, is_visible={}",
                row_id,
                created_by,
                deleted_by,
                is_visible
            );
            if is_visible {
                visible.push(*row_id);
            }
        }
    }

    tracing::debug!(
        "[FILTER_ROWS] Filtered from {} to {} visible rows",
        total_rows,
        visible.len()
    );
    Ok(visible)
}

struct MvccRowIdFilter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    txn_manager: Arc<TxnIdManager>,
    snapshot: TransactionSnapshot,
    _marker: PhantomData<fn(P)>,
}

impl<P> MvccRowIdFilter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn new(txn_manager: Arc<TxnIdManager>, snapshot: TransactionSnapshot) -> Self {
        Self {
            txn_manager,
            snapshot,
            _marker: PhantomData,
        }
    }
}

impl<P> RowIdFilter<P> for MvccRowIdFilter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn filter(&self, table: &Table<P>, row_ids: Vec<RowId>) -> Result<Vec<RowId>> {
        tracing::trace!(
            "[MVCC_FILTER] filter() called with row_ids {:?}, snapshot txn={}, snapshot_id={}",
            row_ids,
            self.snapshot.txn_id,
            self.snapshot.snapshot_id
        );
        let result = filter_row_ids_for_snapshot(table, row_ids, &self.txn_manager, self.snapshot);
        if let Ok(ref visible) = result {
            tracing::trace!(
                "[MVCC_FILTER] filter() returning visible row_ids: {:?}",
                visible
            );
        }
        result
    }
}

// TODO: Rename to `[what-kind-of]ContextProvider`
// Wrapper to implement TableProvider for Context
struct ContextProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<RuntimeContext<P>>,
}

impl<P> TableProvider<P> for ContextProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn get_table(&self, canonical_name: &str) -> Result<Arc<ExecutorTable<P>>> {
        self.context.lookup_table(canonical_name)
    }
}

fn current_time_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

pub fn resolve_insert_columns(columns: &[String], schema: &ExecutorSchema) -> Result<Vec<usize>> {
    if columns.is_empty() {
        return Ok((0..schema.columns.len()).collect());
    }
    let mut resolved = Vec::with_capacity(columns.len());
    for column in columns {
        let normalized = column.to_ascii_lowercase();
        let index = schema.lookup.get(&normalized).ok_or_else(|| {
            Error::InvalidArgumentError(format!(
                "Binder Error: does not have a column named '{}'",
                column
            ))
        })?;
        resolved.push(*index);
    }
    Ok(resolved)
}

fn normalize_insert_value_for_column(
    column: &ExecutorColumn,
    value: PlanValue,
) -> Result<PlanValue> {
    match (&column.data_type, value) {
        (_, PlanValue::Null) => Ok(PlanValue::Null),
        (DataType::Int64, PlanValue::Integer(v)) => Ok(PlanValue::Integer(v)),
        (DataType::Int64, PlanValue::Float(v)) => Ok(PlanValue::Integer(v as i64)),
        (DataType::Int64, other) => Err(Error::InvalidArgumentError(format!(
            "cannot insert {other:?} into INT column '{}'",
            column.name
        ))),
        (DataType::Boolean, PlanValue::Integer(v)) => {
            Ok(PlanValue::Integer(if v != 0 { 1 } else { 0 }))
        }
        (DataType::Boolean, PlanValue::Float(v)) => {
            Ok(PlanValue::Integer(if v != 0.0 { 1 } else { 0 }))
        }
        (DataType::Boolean, PlanValue::String(s)) => {
            let normalized = s.trim().to_ascii_lowercase();
            let value = match normalized.as_str() {
                "true" | "t" | "1" => 1,
                "false" | "f" | "0" => 0,
                _ => {
                    return Err(Error::InvalidArgumentError(format!(
                        "cannot insert string '{}' into BOOLEAN column '{}'",
                        s, column.name
                    )));
                }
            };
            Ok(PlanValue::Integer(value))
        }
        (DataType::Boolean, PlanValue::Struct(_)) => Err(Error::InvalidArgumentError(format!(
            "cannot insert struct into BOOLEAN column '{}'",
            column.name
        ))),
        (DataType::Float64, PlanValue::Integer(v)) => Ok(PlanValue::Float(v as f64)),
        (DataType::Float64, PlanValue::Float(v)) => Ok(PlanValue::Float(v)),
        (DataType::Float64, other) => Err(Error::InvalidArgumentError(format!(
            "cannot insert {other:?} into DOUBLE column '{}'",
            column.name
        ))),
        (DataType::Utf8, PlanValue::Integer(v)) => Ok(PlanValue::String(v.to_string())),
        (DataType::Utf8, PlanValue::Float(v)) => Ok(PlanValue::String(v.to_string())),
        (DataType::Utf8, PlanValue::String(s)) => Ok(PlanValue::String(s)),
        (DataType::Utf8, PlanValue::Struct(_)) => Err(Error::InvalidArgumentError(format!(
            "cannot insert struct into STRING column '{}'",
            column.name
        ))),
        (DataType::Date32, PlanValue::Integer(days)) => {
            let casted = i32::try_from(days).map_err(|_| {
                Error::InvalidArgumentError(format!(
                    "integer literal out of range for DATE column '{}'",
                    column.name
                ))
            })?;
            Ok(PlanValue::Integer(casted as i64))
        }
        (DataType::Date32, PlanValue::String(text)) => {
            let days = parse_date32_literal(&text)?;
            Ok(PlanValue::Integer(days as i64))
        }
        (DataType::Date32, other) => Err(Error::InvalidArgumentError(format!(
            "cannot insert {other:?} into DATE column '{}'",
            column.name
        ))),
        (DataType::Struct(_), PlanValue::Struct(map)) => Ok(PlanValue::Struct(map)),
        (DataType::Struct(_), other) => Err(Error::InvalidArgumentError(format!(
            "expected struct value for struct column '{}', got {other:?}",
            column.name
        ))),
        (other_type, other_value) => Err(Error::InvalidArgumentError(format!(
            "unsupported Arrow data type {:?} for INSERT value {:?} in column '{}'",
            other_type, other_value, column.name
        ))),
    }
}

pub fn build_array_for_column(dtype: &DataType, values: &[PlanValue]) -> Result<ArrayRef> {
    match dtype {
        DataType::Int64 => {
            let mut builder = Int64Builder::with_capacity(values.len());
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(v) => builder.append_value(*v),
                    PlanValue::Float(v) => builder.append_value(*v as i64),
                    PlanValue::String(_) | PlanValue::Struct(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert non-integer into INT column".into(),
                        ));
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Boolean => {
            let mut builder = BooleanBuilder::with_capacity(values.len());
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(v) => builder.append_value(*v != 0),
                    PlanValue::Float(v) => builder.append_value(*v != 0.0),
                    PlanValue::String(s) => {
                        let normalized = s.trim().to_ascii_lowercase();
                        match normalized.as_str() {
                            "true" | "t" | "1" => builder.append_value(true),
                            "false" | "f" | "0" => builder.append_value(false),
                            _ => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "cannot insert string '{}' into BOOLEAN column",
                                    s
                                )));
                            }
                        }
                    }
                    PlanValue::Struct(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert struct into BOOLEAN column".into(),
                        ));
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Float64 => {
            let mut builder = Float64Builder::with_capacity(values.len());
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(v) => builder.append_value(*v as f64),
                    PlanValue::Float(v) => builder.append_value(*v),
                    PlanValue::String(_) | PlanValue::Struct(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert non-numeric into DOUBLE column".into(),
                        ));
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Utf8 => {
            let mut builder = StringBuilder::with_capacity(values.len(), values.len() * 8);
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(v) => builder.append_value(v.to_string()),
                    PlanValue::Float(v) => builder.append_value(v.to_string()),
                    PlanValue::String(s) => builder.append_value(s),
                    PlanValue::Struct(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert struct into STRING column".into(),
                        ));
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Date32 => {
            let mut builder = Date32Builder::with_capacity(values.len());
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(days) => {
                        let casted = i32::try_from(*days).map_err(|_| {
                            Error::InvalidArgumentError(
                                "integer literal out of range for DATE column".into(),
                            )
                        })?;
                        builder.append_value(casted);
                    }
                    PlanValue::Float(_) | PlanValue::Struct(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert non-date value into DATE column".into(),
                        ));
                    }
                    PlanValue::String(text) => {
                        let days = parse_date32_literal(text)?;
                        builder.append_value(days);
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Struct(fields) => {
            use arrow::array::StructArray;
            let mut field_arrays: Vec<(FieldRef, ArrayRef)> = Vec::with_capacity(fields.len());

            for field in fields.iter() {
                let field_name = field.name();
                let field_type = field.data_type();
                let mut field_values = Vec::with_capacity(values.len());

                for value in values {
                    match value {
                        PlanValue::Null => field_values.push(PlanValue::Null),
                        PlanValue::Struct(map) => {
                            let field_value =
                                map.get(field_name).cloned().unwrap_or(PlanValue::Null);
                            field_values.push(field_value);
                        }
                        _ => {
                            return Err(Error::InvalidArgumentError(format!(
                                "expected struct value for struct column, got {:?}",
                                value
                            )));
                        }
                    }
                }

                let field_array = build_array_for_column(field_type, &field_values)?;
                field_arrays.push((Arc::clone(field), field_array));
            }

            Ok(Arc::new(StructArray::from(field_arrays)))
        }
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported Arrow data type for INSERT: {other:?}"
        ))),
    }
}

fn parse_date32_literal(text: &str) -> Result<i32> {
    let mut parts = text.split('-');
    let year_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    let month_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    let day_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    if parts.next().is_some() {
        return Err(Error::InvalidArgumentError(format!(
            "invalid DATE literal '{text}'"
        )));
    }

    let year = year_str.parse::<i32>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid year in DATE literal '{text}'"))
    })?;
    let month_num = month_str.parse::<u8>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid month in DATE literal '{text}'"))
    })?;
    let day = day_str.parse::<u8>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid day in DATE literal '{text}'"))
    })?;

    let month = Month::try_from(month_num).map_err(|_| {
        Error::InvalidArgumentError(format!("invalid month in DATE literal '{text}'"))
    })?;

    let date = Date::from_calendar_date(year, month, day).map_err(|err| {
        Error::InvalidArgumentError(format!("invalid DATE literal '{text}': {err}"))
    })?;
    let days = date.to_julian_day() - epoch_julian_day();
    Ok(days)
}

fn epoch_julian_day() -> i32 {
    Date::from_calendar_date(1970, Month::January, 1)
        .expect("1970-01-01 is a valid date")
        .to_julian_day()
}

// Array -> PlanValue conversion is provided by llkv-plan::plan_value_from_array

fn full_table_scan_filter(field_id: FieldId) -> LlkvExpr<'static, FieldId> {
    LlkvExpr::Pred(Filter {
        field_id,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    })
}

fn resolve_field_id_from_schema(schema: &ExecutorSchema, name: &str) -> Result<FieldId> {
    if name.eq_ignore_ascii_case(ROW_ID_COLUMN_NAME) {
        return Ok(ROW_ID_FIELD_ID);
    }

    schema
        .resolve(name)
        .map(|column| column.field_id)
        .ok_or_else(|| {
            Error::InvalidArgumentError(format!(
                "Binder Error: does not have a column named '{name}'"
            ))
        })
}

fn translate_predicate(
    expr: LlkvExpr<'static, String>,
    schema: &ExecutorSchema,
) -> Result<LlkvExpr<'static, FieldId>> {
    match expr {
        LlkvExpr::And(list) => {
            let mut converted = Vec::with_capacity(list.len());
            for item in list {
                converted.push(translate_predicate(item, schema)?);
            }
            Ok(LlkvExpr::And(converted))
        }
        LlkvExpr::Or(list) => {
            let mut converted = Vec::with_capacity(list.len());
            for item in list {
                converted.push(translate_predicate(item, schema)?);
            }
            Ok(LlkvExpr::Or(converted))
        }
        LlkvExpr::Not(inner) => Ok(LlkvExpr::Not(Box::new(translate_predicate(
            *inner, schema,
        )?))),
        LlkvExpr::Pred(Filter { field_id, op }) => {
            let resolved = resolve_field_id_from_schema(schema, &field_id)?;
            Ok(LlkvExpr::Pred(Filter {
                field_id: resolved,
                op,
            }))
        }
        LlkvExpr::Compare { left, op, right } => {
            let left = translate_scalar(&left, schema)?;
            let right = translate_scalar(&right, schema)?;
            Ok(LlkvExpr::Compare { left, op, right })
        }
    }
}

fn translate_scalar(
    expr: &ScalarExpr<String>,
    schema: &ExecutorSchema,
) -> Result<ScalarExpr<FieldId>> {
    match expr {
        ScalarExpr::Column(name) => {
            let field_id = resolve_field_id_from_schema(schema, name)?;
            Ok(ScalarExpr::column(field_id))
        }
        ScalarExpr::Literal(lit) => Ok(ScalarExpr::Literal(lit.clone())),
        ScalarExpr::Binary { left, op, right } => {
            let left_expr = translate_scalar(left, schema)?;
            let right_expr = translate_scalar(right, schema)?;
            Ok(ScalarExpr::Binary {
                left: Box::new(left_expr),
                op: *op,
                right: Box::new(right_expr),
            })
        }
        ScalarExpr::Aggregate(agg) => {
            // Translate column names in aggregate calls to field IDs
            use llkv_expr::expr::AggregateCall;
            let translated_agg = match agg {
                AggregateCall::CountStar => AggregateCall::CountStar,
                AggregateCall::Count(name) => {
                    let field_id = resolve_field_id_from_schema(schema, name).map_err(|_| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Count(field_id)
                }
                AggregateCall::Sum(name) => {
                    let field_id = resolve_field_id_from_schema(schema, name).map_err(|_| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Sum(field_id)
                }
                AggregateCall::Min(name) => {
                    let field_id = resolve_field_id_from_schema(schema, name).map_err(|_| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Min(field_id)
                }
                AggregateCall::Max(name) => {
                    let field_id = resolve_field_id_from_schema(schema, name).map_err(|_| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Max(field_id)
                }
                AggregateCall::CountNulls(name) => {
                    let field_id = resolve_field_id_from_schema(schema, name).map_err(|_| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::CountNulls(field_id)
                }
            };
            Ok(ScalarExpr::Aggregate(translated_agg))
        }
        ScalarExpr::GetField { base, field_name } => {
            let base_expr = translate_scalar(base, schema)?;
            Ok(ScalarExpr::GetField {
                base: Box::new(base_expr),
                field_name: field_name.clone(),
            })
        }
    }
}
