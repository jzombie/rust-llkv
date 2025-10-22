//! Runtime-facing namespace adapters.
//!
//! These adapters wrap `RuntimeContext` instances so sessions can route logical
//! operations (persistent vs. temporary) without exposing the underlying storage
//! engine to higher layers. They are **not** the same as namespaces inside the
//! storage engine or column-map crates; they strictly coordinate runtime-level
//! routing.

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
use simd_r_drive_entry_handle::EntryHandle;

use crate::{RuntimeContext, RuntimeStatementResult};
use llkv_table::{CatalogDdl, SingleColumnIndexDescriptor};

/// Identifier type used by runtime storage namespaces.
pub type RuntimeNamespaceId = String;

/// Canonical identifier for the persistent runtime namespace.
pub const PERSISTENT_NAMESPACE_ID: &str = "main";
/// Canonical identifier for the temporary runtime namespace.
pub const TEMPORARY_NAMESPACE_ID: &str = "temp";

/// Runtime-level namespace abstraction.
///
/// Implementors wrap a `RuntimeContext` for a specific routing domain (e.g.
/// persistent tables, temporary tables). The namespace knows how to forward
/// DDL/DML plan execution into the underlying context, but it does **not**
/// perform storage allocation or manage physical pagers on its own.
pub trait RuntimeStorageNamespace: Send + Sync + 'static {
    type Pager: Pager<Blob = EntryHandle> + Send + Sync + 'static;

    /// Identifier used when resolving schemas (e.g. "main", "temp").
    fn namespace_id(&self) -> &RuntimeNamespaceId;

    /// Returns the runtime context bound to this namespace.
    fn context(&self) -> Arc<RuntimeContext<Self::Pager>>;

    /// Create a table inside this namespace.
    fn create_table(
        &self,
        plan: CreateTablePlan,
    ) -> crate::Result<RuntimeStatementResult<Self::Pager>>;

    /// Drop a table from this namespace by forwarding the planned request.
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

    /// Drop an index from this namespace by forwarding the request.
    fn drop_index(
        &self,
        plan: DropIndexPlan,
    ) -> crate::Result<Option<SingleColumnIndexDescriptor>>;

    /// Execute a generic plan operation. Namespaces that do not yet support
    /// this entry point should override this method.
    fn execute_operation(
        &self,
        operation: PlanOperation,
    ) -> crate::Result<TransactionResult<Self::Pager>> {
        let _ = operation;
        Err(Error::Internal(format!(
            "runtime namespace '{}' does not yet support execute_operation",
            self.namespace_id()
        )))
    }

    /// Lookup a table by canonical name.
    fn lookup_table(&self, canonical: &str) -> crate::Result<Arc<ExecutorTable<Self::Pager>>>;

    /// List tables visible to this namespace.
    fn list_tables(&self) -> Vec<String>;
}

/// Helper trait for storing runtime namespaces behind trait objects with basic
/// inspection support.
pub trait RuntimeStorageNamespaceOps: Send + Sync {
    fn namespace_id(&self) -> &RuntimeNamespaceId;
    fn list_tables(&self) -> Vec<String>;
    fn as_any(&self) -> &dyn Any;
}

impl<T> RuntimeStorageNamespaceOps for T
where
    T: RuntimeStorageNamespace,
{
    fn namespace_id(&self) -> &RuntimeNamespaceId {
        RuntimeStorageNamespace::namespace_id(self)
    }

    fn list_tables(&self) -> Vec<String> {
        RuntimeStorageNamespace::list_tables(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Persistent runtime namespace wrapper.
#[derive(Clone)]
pub struct PersistentRuntimeNamespace<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    id: RuntimeNamespaceId,
    context: Arc<RuntimeContext<P>>,
}

impl<P> PersistentRuntimeNamespace<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(id: RuntimeNamespaceId, context: Arc<RuntimeContext<P>>) -> Self {
        Self { id, context }
    }
}

impl<P> RuntimeStorageNamespace for PersistentRuntimeNamespace<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    type Pager = P;

    fn namespace_id(&self) -> &RuntimeNamespaceId {
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

/// Temporary runtime namespace wrapper.
#[derive(Clone)]
pub struct TemporaryRuntimeNamespace<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    id: RuntimeNamespaceId,
    context: Arc<RwLock<Arc<RuntimeContext<P>>>>,
}

impl<P> TemporaryRuntimeNamespace<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(id: RuntimeNamespaceId, context: Arc<RuntimeContext<P>>) -> Self {
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

impl<P> RuntimeStorageNamespace for TemporaryRuntimeNamespace<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    type Pager = P;

    fn namespace_id(&self) -> &RuntimeNamespaceId {
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
