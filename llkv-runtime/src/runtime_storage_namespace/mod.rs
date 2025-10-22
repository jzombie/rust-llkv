//! Runtime storage namespace abstractions and registries.
//!
//! This module holds the runtime-facing facade for namespace resolution. The
//! types defined here coordinate how logical namespaces are exposed to the rest
//! of the runtime without leaking the storage engine's internal layout.

pub mod namespace;
pub mod registry;

pub use namespace::{
    PERSISTENT_NAMESPACE_ID, PersistentRuntimeNamespace, RuntimeNamespaceId,
    RuntimeStorageNamespace, RuntimeStorageNamespaceOps, TEMPORARY_NAMESPACE_ID,
    TemporaryRuntimeNamespace,
};
pub use registry::RuntimeStorageNamespaceRegistry;
