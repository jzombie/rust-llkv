use std::any::Any;
use std::sync::{Arc, RwLock};

use llkv_executor::ExecutorTable;
use llkv_plan::{CreateIndexPlan, CreateTablePlan, PlanOperation};
use llkv_result::Error;
use llkv_storage::pager::Pager;
use llkv_transaction::TransactionResult;
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;

use crate::{RuntimeContext, RuntimeStatementResult};
use llkv_table::canonical_table_name;

pub type NamespaceId = String;

// TODO: Extract to constants module
pub const PERSISTENT_NAMESPACE_ID: &str = "main";
pub const TEMPORARY_NAMESPACE_ID: &str = "temp";

// TODO: Rename to `RuntimeStorageNamespace`?
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

    /// Drop a table from this namespace.
    fn drop_table(&self, name: &str, if_exists: bool) -> crate::Result<()>;

    /// Create an index inside this namespace.
    fn create_index(
        &self,
        plan: CreateIndexPlan,
    ) -> crate::Result<RuntimeStatementResult<Self::Pager>>;

    /// Execute a generic plan operation. Namespaces that do not yet support this entry point
    /// should rely on the default error implementation.
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

    /// Returns true if the namespace owns (and should answer for) the canonical table name.
    fn owns_table(&self, canonical: &str) -> bool {
        let _ = canonical;
        false
    }
}

/// Helper trait for storing namespaces behind trait objects with basic inspection support.
pub trait StorageNamespaceOps: Send + Sync {
    fn namespace_id(&self) -> &NamespaceId;
    fn list_tables(&self) -> Vec<String>;
    fn owns_table(&self, canonical: &str) -> bool;
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

    fn owns_table(&self, canonical: &str) -> bool {
        StorageNamespace::owns_table(self, canonical)
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
        self.context.create_table_plan(plan)
    }

    fn drop_table(&self, name: &str, if_exists: bool) -> crate::Result<()> {
        self.context.drop_table_immediate(name, if_exists)
    }

    fn create_index(
        &self,
        plan: CreateIndexPlan,
    ) -> crate::Result<RuntimeStatementResult<Self::Pager>> {
        self.context.create_index(plan)
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
    tables: Arc<RwLock<FxHashSet<String>>>,
}

impl<P> TemporaryNamespace<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(id: NamespaceId, context: Arc<RuntimeContext<P>>) -> Self {
        Self {
            id,
            context: Arc::new(RwLock::new(context)),
            tables: Arc::new(RwLock::new(FxHashSet::default())),
        }
    }

    pub fn replace_context(&self, context: Arc<RuntimeContext<P>>) {
        let mut guard = self
            .context
            .write()
            .expect("temporary context lock poisoned");
        *guard = context;
    }

    pub fn register_table(&self, canonical: String) {
        self.tables
            .write()
            .expect("temporary table registry poisoned")
            .insert(canonical);
    }

    pub fn unregister_table(&self, canonical: &str) {
        self.tables
            .write()
            .expect("temporary table registry poisoned")
            .remove(canonical);
    }

    pub fn clear_tables(&self) {
        let canonical_names: Vec<String> = {
            let mut guard = self
                .tables
                .write()
                .expect("temporary table registry poisoned");
            guard.drain().collect()
        };

        if canonical_names.is_empty() {
            return;
        }

        let catalog = self.context().table_catalog();
        for canonical in canonical_names {
            if let Some(table_id) = catalog.table_id(&canonical) {
                if catalog.unregister_table(table_id) {
                    tracing::debug!(
                        "[TEMP] Unregistered temporary table '{}' from shared catalog",
                        canonical
                    );
                }
            }
        }
    }

    pub fn has_table(&self, canonical: &str) -> bool {
        self.tables
            .read()
            .expect("temporary table registry poisoned")
            .contains(canonical)
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
        let (_, canonical) = canonical_table_name(&plan.name)?;
        let context = self.context();
        let result = context.create_table_plan(plan)?;
        if !self.has_table(&canonical) {
            self.register_table(canonical);
        }
        Ok(result)
    }

    fn drop_table(&self, name: &str, if_exists: bool) -> crate::Result<()> {
        let (_, canonical) = canonical_table_name(name)?;
        let context = self.context();
        context.drop_table_immediate(name, if_exists)?;
        self.unregister_table(&canonical);
        Ok(())
    }

    fn create_index(
        &self,
        plan: CreateIndexPlan,
    ) -> crate::Result<RuntimeStatementResult<Self::Pager>> {
        self.context().create_index(plan)
    }

    fn lookup_table(&self, canonical: &str) -> crate::Result<Arc<ExecutorTable<Self::Pager>>> {
        self.context().lookup_table(canonical)
    }

    fn list_tables(&self) -> Vec<String> {
        let mut names = self.context().table_names();
        let tracked = self
            .tables
            .read()
            .expect("temporary table registry poisoned")
            .clone();
        for canonical in tracked {
            if !names.iter().any(|existing| existing == &canonical) {
                names.push(canonical);
            }
        }
        names
    }

    fn owns_table(&self, canonical: &str) -> bool {
        self.has_table(canonical)
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
}

impl StorageNamespaceRegistry {
    pub fn new(persistent_id: NamespaceId) -> Self {
        Self {
            persistent_id: persistent_id.clone(),
            namespaces: FxHashMap::default(),
            schema_map: FxHashMap::default(),
            priority: vec![persistent_id],
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
        self.namespaces.insert(id, entry);
    }

    pub fn register_schema(&mut self, schema: impl Into<String>, namespace: NamespaceId) {
        self.schema_map.insert(schema.into(), namespace);
    }

    pub fn set_unqualified_priority(&mut self, order: Vec<NamespaceId>) {
        self.priority = order;
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
        self.namespaces
            .values()
            .any(|entry| entry.ops.owns_table(canonical))
    }

    pub fn namespace_for_table(&self, canonical: &str) -> NamespaceId {
        if let Some(namespace_id) = self
            .priority
            .iter()
            .filter_map(|namespace_id| self.namespaces.get(namespace_id))
            .find(|entry| entry.ops.owns_table(canonical))
            .map(|entry| entry.ops.namespace_id().clone())
        {
            return namespace_id;
        }

        self.persistent_id.clone()
    }
}
