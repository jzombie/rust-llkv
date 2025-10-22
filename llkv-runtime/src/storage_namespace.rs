//! Runtime storage namespace abstraction.
//!
//! Namespaces wrap pager-backed contexts so sessions can route operations based
//! on table ownership (persistent vs. temporary).

use std::any::Any;
use std::sync::{Arc, RwLock};

use llkv_executor::ExecutorTable;
use llkv_plan::{
    AlterTablePlan, CreateIndexPlan, CreateTablePlan, DropIndexPlan, DropTablePlan, PlanOperation,
    RenameTablePlan,
};
use llkv_result::Error;
use llkv_storage::pager::Pager;
use llkv_transaction::TransactionResult;
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;

use crate::{RuntimeContext, RuntimeStatementResult};
use llkv_table::{CatalogDdl, SingleColumnIndexDescriptor};

// TODO: Rename to `RuntimeStorageNamespaceId`?
pub type NamespaceId = String;

// TODO: Extract to constants module
// NOTE: Keep string identifiers here until the runtime constants module is formalized.
pub const PERSISTENT_NAMESPACE_ID: &str = "main";
pub const TEMPORARY_NAMESPACE_ID: &str = "temp";

// TODO: Extend `CatalogDdl`
// TODO: Rename to `RuntimeStorageNamespace`?
// NOTE: Trait name mirrors the broader storage namespace concept used across crates.
/// Trait implemented by all runtime storage namespaces.
///
/// Each namespace encapsulates a single storage backend (pager + catalog + table cache).
/// Namespaces expose a minimal API so higher layers (RuntimeSession, SQL engine) can route
/// operations without hard-coding per-namespace logic.
pub trait StorageNamespace: Send + Sync + 'static {
    type Pager: Pager<Blob = EntryHandle> + Send + Sync + 'static;

    /// Identifier used when resolving schemas (e.g. "main", "temp").
    fn namespace_id(&self) -> &NamespaceId;

    /// Returns the runtime context bound to this namespace.
    fn context(&self) -> Arc<RuntimeContext<Self::Pager>>;

    /// Create a table inside this namespace.
    fn create_table(
        &self,
        plan: CreateTablePlan,
    ) -> crate::Result<RuntimeStatementResult<Self::Pager>>;

    /// Drop a table from this namespace by forwarding the planned request to the context.
    fn drop_table(&self, plan: DropTablePlan) -> crate::Result<()>;

    /// Rename a table within this namespace.
    fn rename_table(&self, plan: RenameTablePlan) -> crate::Result<()>;

    /// Alter a table within this namespace.
    fn alter_table(
        &self,
        plan: AlterTablePlan,
    ) -> crate::Result<RuntimeStatementResult<Self::Pager>>;

    /// Create an index within this namespace.
    fn create_index(
        &self,
        plan: CreateIndexPlan,
    ) -> crate::Result<RuntimeStatementResult<Self::Pager>>;

    /// Drop an index from this namespace by forwarding the planned request to the context.
    fn drop_index(&self, plan: DropIndexPlan)
    -> crate::Result<Option<SingleColumnIndexDescriptor>>;

    /// Execute a generic plan operation. Namespaces that do not yet support this entry point
    /// should override this method and provide an implementation.
    fn execute_operation(
        &self,
        operation: PlanOperation,
    ) -> crate::Result<TransactionResult<Self::Pager>> {
        let _ = operation;
        Err(Error::Internal(format!(
            "namespace '{}' does not yet support execute_operation",
            self.namespace_id()
        )))
    }

    /// Lookup a table by canonical name.
    fn lookup_table(&self, canonical: &str) -> crate::Result<Arc<ExecutorTable<Self::Pager>>>;

    /// List tables visible to this namespace.
    fn list_tables(&self) -> Vec<String>;

}

/// Helper trait for storing namespaces behind trait objects with basic inspection support.
pub trait StorageNamespaceOps: Send + Sync {
    fn namespace_id(&self) -> &NamespaceId;
    fn list_tables(&self) -> Vec<String>;
    fn as_any(&self) -> &dyn Any;
}

impl<T> StorageNamespaceOps for T
where
    T: StorageNamespace,
{
    fn namespace_id(&self) -> &NamespaceId {
        StorageNamespace::namespace_id(self)
    }

    fn list_tables(&self) -> Vec<String> {
        StorageNamespace::list_tables(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Clone)]
pub struct PersistentNamespace<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    id: NamespaceId,
    context: Arc<RuntimeContext<P>>,
}

impl<P> PersistentNamespace<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(id: NamespaceId, context: Arc<RuntimeContext<P>>) -> Self {
        Self { id, context }
    }
}

impl<P> StorageNamespace for PersistentNamespace<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    type Pager = P;

    fn namespace_id(&self) -> &NamespaceId {
        &self.id
    }

    fn context(&self) -> Arc<RuntimeContext<Self::Pager>> {
        Arc::clone(&self.context)
    }

    fn create_table(
        &self,
        plan: CreateTablePlan,
    ) -> crate::Result<RuntimeStatementResult<Self::Pager>> {
        CatalogDdl::create_table(self.context.as_ref(), plan)
    }

    fn drop_table(&self, plan: DropTablePlan) -> crate::Result<()> {
        CatalogDdl::drop_table(self.context.as_ref(), plan)
    }

    fn rename_table(&self, plan: RenameTablePlan) -> crate::Result<()> {
        CatalogDdl::rename_table(self.context.as_ref(), plan)
    }

    fn alter_table(
        &self,
        plan: AlterTablePlan,
    ) -> crate::Result<RuntimeStatementResult<Self::Pager>> {
        CatalogDdl::alter_table(self.context.as_ref(), plan)
    }

    fn create_index(
        &self,
        plan: CreateIndexPlan,
    ) -> crate::Result<RuntimeStatementResult<Self::Pager>> {
        CatalogDdl::create_index(self.context.as_ref(), plan)
    }

    fn drop_index(
        &self,
        plan: DropIndexPlan,
    ) -> crate::Result<Option<SingleColumnIndexDescriptor>> {
        CatalogDdl::drop_index(self.context.as_ref(), plan)
    }

    fn lookup_table(&self, canonical: &str) -> crate::Result<Arc<ExecutorTable<Self::Pager>>> {
        self.context.lookup_table(canonical)
    }

    fn list_tables(&self) -> Vec<String> {
        self.context.table_names()
    }
}

#[derive(Clone)]
pub struct TemporaryNamespace<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    id: NamespaceId,
    context: Arc<RwLock<Arc<RuntimeContext<P>>>>,
}

impl<P> TemporaryNamespace<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(id: NamespaceId, context: Arc<RuntimeContext<P>>) -> Self {
        Self {
            id,
            context: Arc::new(RwLock::new(context)),
        }
    }

    pub fn replace_context(&self, context: Arc<RuntimeContext<P>>) {
        let mut guard = self
            .context
            .write()
            .expect("temporary context lock poisoned");
        *guard = context;
    }

    pub fn clear_tables<I>(&self, canonical_names: I)
    where
        I: IntoIterator<Item = String>,
    {
        let canonical_names: Vec<String> = canonical_names.into_iter().collect();
        if canonical_names.is_empty() {
            return;
        }

        let catalog = self.context().table_catalog();
        for canonical in canonical_names {
            if let Some(table_id) = catalog.table_id(&canonical)
                && catalog.unregister_table(table_id)
            {
                tracing::debug!(
                    "[TEMP] Unregistered temporary table '{}' from shared catalog",
                    canonical
                );
            }
        }
    }

    pub fn context(&self) -> Arc<RuntimeContext<P>> {
        Arc::clone(
            &self
                .context
                .read()
                .expect("temporary context lock poisoned"),
        )
    }
}

impl<P> StorageNamespace for TemporaryNamespace<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    type Pager = P;

    fn namespace_id(&self) -> &NamespaceId {
        &self.id
    }

    fn context(&self) -> Arc<RuntimeContext<Self::Pager>> {
        self.context()
    }

    fn create_table(
        &self,
        plan: CreateTablePlan,
    ) -> crate::Result<RuntimeStatementResult<Self::Pager>> {
        let context = self.context();
        let result = CatalogDdl::create_table(context.as_ref(), plan)?;
        Ok(result)
    }

    fn drop_table(&self, plan: DropTablePlan) -> crate::Result<()> {
        let context = self.context();
        CatalogDdl::drop_table(context.as_ref(), plan)?;
        Ok(())
    }

    fn rename_table(&self, plan: RenameTablePlan) -> crate::Result<()> {
        let context = self.context();
        CatalogDdl::rename_table(context.as_ref(), plan)?;
        Ok(())
    }

    fn alter_table(
        &self,
        plan: AlterTablePlan,
    ) -> crate::Result<RuntimeStatementResult<Self::Pager>> {
        let context = self.context();
        CatalogDdl::alter_table(context.as_ref(), plan)
    }

    fn create_index(
        &self,
        plan: CreateIndexPlan,
    ) -> crate::Result<RuntimeStatementResult<Self::Pager>> {
        let context = self.context();
        CatalogDdl::create_index(context.as_ref(), plan)
    }

    fn drop_index(
        &self,
        plan: DropIndexPlan,
    ) -> crate::Result<Option<SingleColumnIndexDescriptor>> {
        let context = self.context();
        CatalogDdl::drop_index(context.as_ref(), plan)
    }

    fn lookup_table(&self, canonical: &str) -> crate::Result<Arc<ExecutorTable<Self::Pager>>> {
        self.context().lookup_table(canonical)
    }

    fn list_tables(&self) -> Vec<String> {
        self.context().table_names()
    }
}

#[derive(Clone)]
struct NamespaceEntry {
    ops: Arc<dyn StorageNamespaceOps>,
    erased: Arc<dyn Any + Send + Sync>,
}

impl NamespaceEntry {
    fn new<N>(namespace: Arc<N>) -> Self
    where
        N: StorageNamespace + 'static,
    {
        let ops: Arc<dyn StorageNamespaceOps> = namespace.clone();
        let erased: Arc<dyn Any + Send + Sync> = namespace;
        Self { ops, erased }
    }

    fn as_typed<N>(&self) -> Option<Arc<N>>
    where
        N: StorageNamespace + 'static,
    {
        self.erased.clone().downcast::<N>().ok()
    }
}

pub struct StorageNamespaceRegistry {
    persistent_id: NamespaceId,
    namespaces: FxHashMap<NamespaceId, NamespaceEntry>,
    schema_map: FxHashMap<String, NamespaceId>,
    priority: Vec<NamespaceId>,
    table_map: FxHashMap<String, NamespaceId>,
    namespace_tables: FxHashMap<NamespaceId, FxHashSet<String>>,
}

impl StorageNamespaceRegistry {
    pub fn new(persistent_id: NamespaceId) -> Self {
        Self {
            persistent_id: persistent_id.clone(),
            namespaces: FxHashMap::default(),
            schema_map: FxHashMap::default(),
            priority: vec![persistent_id],
            table_map: FxHashMap::default(),
            namespace_tables: FxHashMap::default(),
        }
    }

    pub fn persistent_id(&self) -> &str {
        &self.persistent_id
    }

    pub fn register_namespace<N, I>(
        &mut self,
        namespace: Arc<N>,
        schemas: I,
        prefer_unqualified: bool,
    ) where
        N: StorageNamespace + 'static,
        I: IntoIterator<Item = String>,
    {
        let id = namespace.namespace_id().to_string();
        let entry = NamespaceEntry::new(namespace);
        for schema in schemas {
            self.schema_map.insert(schema, id.clone());
        }
        if prefer_unqualified {
            if !self.priority.iter().any(|existing| existing == &id) {
                self.priority.insert(0, id.clone());
            }
        } else if !self.priority.iter().any(|existing| existing == &id) {
            self.priority.push(id.clone());
        }
        self
            .namespace_tables
            .entry(id.clone())
            .or_insert_with(FxHashSet::default);
        self.namespaces.insert(id, entry);
    }

    pub fn register_schema(&mut self, schema: impl Into<String>, namespace: NamespaceId) {
        self.schema_map.insert(schema.into(), namespace);
    }

    pub fn set_unqualified_priority(&mut self, order: Vec<NamespaceId>) {
        self.priority = order;
    }

    pub fn register_table(&mut self, namespace: &NamespaceId, canonical: String) {
        let namespace_id = namespace.clone();
        self.table_map
            .insert(canonical.clone(), namespace_id.clone());
        self
            .namespace_tables
            .entry(namespace_id)
            .or_insert_with(FxHashSet::default)
            .insert(canonical);
    }

    pub fn unregister_table(&mut self, canonical: &str) -> Option<NamespaceId> {
        if let Some(namespace_id) = self.table_map.remove(canonical) {
            if let Some(tables) = self.namespace_tables.get_mut(&namespace_id) {
                tables.remove(canonical);
            }
            Some(namespace_id)
        } else {
            None
        }
    }

    pub fn drain_namespace_tables(&mut self, namespace: &NamespaceId) -> Vec<String> {
        let tables = self
            .namespace_tables
            .entry(namespace.clone())
            .or_insert_with(FxHashSet::default);
        let drained: Vec<String> = tables.drain().collect();
        for canonical in &drained {
            self.table_map.remove(canonical);
        }
        drained
    }

    pub fn namespace_for_schema(&self, schema: &str) -> Option<&NamespaceId> {
        self.schema_map.get(schema)
    }

    pub fn namespace_ops(&self, id: &str) -> Option<Arc<dyn StorageNamespaceOps>> {
        self.namespaces.get(id).map(|entry| Arc::clone(&entry.ops))
    }

    pub fn namespace<N>(&self, id: &str) -> Option<Arc<N>>
    where
        N: StorageNamespace + 'static,
    {
        self.namespaces
            .get(id)
            .and_then(|entry| entry.as_typed::<N>())
    }

    pub fn unqualified_order(&self) -> Box<dyn Iterator<Item = &NamespaceId> + '_> {
        if self.priority.is_empty() {
            Box::new(std::iter::once(&self.persistent_id))
        } else {
            Box::new(self.priority.iter())
        }
    }

    pub fn is_temporary_table(&self, canonical: &str) -> bool {
        matches!(
            self.table_map.get(canonical),
            Some(namespace_id) if namespace_id == TEMPORARY_NAMESPACE_ID
        )
    }

    pub fn namespace_for_table(&self, canonical: &str) -> NamespaceId {
        self.table_map
            .get(canonical)
            .cloned()
            .unwrap_or_else(|| self.persistent_id.clone())
    }
}
