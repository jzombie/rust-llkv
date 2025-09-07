use crate::types::ByteWidth;

#[derive(Clone, Debug)]
pub struct IndexLayoutInfo {
    pub kind: &'static str,             // "fixed" or "variable" (value layout)
    pub fixed_width: Option<ByteWidth>, // when value layout is fixed
    // TODO: Rename to indicate *logical* and *len*?
    pub key_bytes: usize,        // logical_key_bytes.len()
    pub key_offs_bytes: usize, // if KeyLayout::Variable: key_offsets.len() * sizeof(ByteOffset); else 0
    pub value_meta_bytes: usize, // Variable: value_offsets.len()*sizeof(ByteOffset), Fixed: 4 (ByteWidth)
}
