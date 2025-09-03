use crate::types::RootId;

#[inline]
pub fn decode_root_id(bytes: &[u8]) -> RootId {
    u64::from_be_bytes(bytes.try_into().unwrap())
}
