use crate::types::PhysicalKey;
use bitcode::{Decode, Encode};

/// Bootstrap record that tells a reader where to find the current `Manifest`.
///
/// In deployments, this is typically stored at a well-known **physical key**
/// (e.g., `0`) so a fresh process can discover the rest of the metadata with a
/// single read.
///
/// Encoding: serialized as a typed blob via `bitcode`.
#[derive(Debug, Clone, Encode, Decode)]
pub struct Bootstrap {
    pub manifest_physical_key: PhysicalKey,
}
