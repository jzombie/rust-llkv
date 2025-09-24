//! Common types for the table core.

#![forbid(unsafe_code)]

/// Field identifier type for addressing columns within a table.
pub type FieldId = u32;

/// Row identifier type.
pub type RowId = u64;

pub type RootId = u64;
pub type RootIdBytes = [u8; 8];
