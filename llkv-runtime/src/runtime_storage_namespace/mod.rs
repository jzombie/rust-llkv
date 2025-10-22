//! Runtime storage namespace abstractions and registries.
//!
//! This module holds the runtime-facing facade for namespace resolution. The
//! types defined here coordinate how logical namespaces are exposed to the rest
//! of the runtime without leaking the storage engine's internal layout.

pub mod namespace;
pub mod registry;

pub use namespace::{
    RuntimeNamespaceId, RuntimeStorageNamespace, RuntimeStorageNamespaceOps,
    PersistentRuntimeNamespace, TemporaryRuntimeNamespace, PERSISTENT_NAMESPACE_ID,
    TEMPORARY_NAMESPACE_ID,
};
pub use registry::RuntimeStorageNamespaceRegistry;
