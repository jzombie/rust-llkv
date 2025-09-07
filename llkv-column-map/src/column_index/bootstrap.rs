use crate::types::PhysicalKey;
use bitcode::{Decode, Encode};

#[derive(Debug, Clone, Encode, Decode)]
pub struct Bootstrap {
    pub manifest_physical_key: PhysicalKey,
}
