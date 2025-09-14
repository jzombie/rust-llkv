use crate::types::PhysicalKey;

/// Bootstrap pointer to the current Manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bootstrap {
    pub manifest_physical_key: PhysicalKey,
}
