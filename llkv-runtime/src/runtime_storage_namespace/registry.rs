//! Registry for coordinating runtime storage namespaces.
//!
//! The registry keeps track of logical namespace instances (persistent,
//! temporary, etc.) and provides lookup utilities for runtime sessions. It does
//! **not** allocate storage or manage pager lifecycles; it simply records which
//! runtime namespace currently owns a canonical table name or schema prefix.

use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use super::namespace::{
    RuntimeNamespaceId, RuntimeStorageNamespace, RuntimeStorageNamespaceOps, TEMPORARY_NAMESPACE_ID,
};

struct NamespaceEntry {
    ops: Arc<dyn RuntimeStorageNamespaceOps>,
    erased: Arc<dyn std::any::Any + Send + Sync>,
}

impl NamespaceEntry {
    fn new<N>(namespace: Arc<N>) -> Self
    where
        N: RuntimeStorageNamespace + 'static,
    {
        let ops: Arc<dyn RuntimeStorageNamespaceOps> = namespace.clone();
        let erased: Arc<dyn std::any::Any + Send + Sync> = namespace;
        Self { ops, erased }
    }

    fn ops(&self) -> Arc<dyn RuntimeStorageNamespaceOps> {
        Arc::clone(&self.ops)
    }

    fn as_typed<N>(&self) -> Option<Arc<N>>
    where
        N: RuntimeStorageNamespace + 'static,
    {
        self.erased.clone().downcast::<N>().ok()
    }
}

/// Tracks runtime storage namespaces and their table ownership.
pub struct RuntimeStorageNamespaceRegistry {
    persistent_id: RuntimeNamespaceId,
    namespaces: FxHashMap<RuntimeNamespaceId, NamespaceEntry>,
    schema_map: FxHashMap<String, RuntimeNamespaceId>,
    priority: Vec<RuntimeNamespaceId>,
    table_map: FxHashMap<String, RuntimeNamespaceId>,
    namespace_tables: FxHashMap<RuntimeNamespaceId, FxHashSet<String>>,
}

impl RuntimeStorageNamespaceRegistry {
    pub fn new(persistent_id: RuntimeNamespaceId) -> Self {
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
        N: RuntimeStorageNamespace + 'static,
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
        self.namespace_tables.entry(id.clone()).or_default();
        self.namespaces.insert(id, entry);
    }

    pub fn register_schema(&mut self, schema: impl Into<String>, namespace: RuntimeNamespaceId) {
        self.schema_map.insert(schema.into(), namespace);
    }

    pub fn set_unqualified_priority(&mut self, order: Vec<RuntimeNamespaceId>) {
        self.priority = order;
    }

    pub fn register_table(&mut self, namespace: &RuntimeNamespaceId, canonical: String) {
        let namespace_id = namespace.clone();
        self.table_map
            .insert(canonical.clone(), namespace_id.clone());
        self.namespace_tables
            .entry(namespace_id)
            .or_default()
            .insert(canonical);
    }

    pub fn unregister_table(&mut self, canonical: &str) -> Option<RuntimeNamespaceId> {
        if let Some(namespace_id) = self.table_map.remove(canonical) {
            if let Some(tables) = self.namespace_tables.get_mut(&namespace_id) {
                tables.remove(canonical);
            }
            Some(namespace_id)
        } else {
            None
        }
    }

    pub fn drain_namespace_tables(&mut self, namespace: &RuntimeNamespaceId) -> Vec<String> {
        let tables = self.namespace_tables.entry(namespace.clone()).or_default();
        let drained: Vec<String> = tables.drain().collect();
        for canonical in &drained {
            self.table_map.remove(canonical);
        }
        drained
    }

    pub fn namespace_for_schema(&self, schema: &str) -> Option<&RuntimeNamespaceId> {
        self.schema_map.get(schema)
    }

    pub fn namespace_ops(&self, id: &str) -> Option<Arc<dyn RuntimeStorageNamespaceOps>> {
        self.namespaces.get(id).map(NamespaceEntry::ops)
    }

    pub fn namespace<N>(&self, id: &str) -> Option<Arc<N>>
    where
        N: RuntimeStorageNamespace + 'static,
    {
        self.namespaces
            .get(id)
            .and_then(|entry| entry.as_typed::<N>())
    }

    pub fn unqualified_order(&self) -> Box<dyn Iterator<Item = &RuntimeNamespaceId> + '_> {
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

    pub fn namespace_for_table(&self, canonical: &str) -> RuntimeNamespaceId {
        self.table_map
            .get(canonical)
            .cloned()
            .unwrap_or_else(|| self.persistent_id.clone())
    }
}
