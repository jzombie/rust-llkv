use super::column_entry::ColumnEntry;
use bitcode::{Decode, Encode};

// A manifest maps columns → their current ColumnIndex blob (newest version).
#[derive(Debug, Clone, Encode, Decode)]
pub struct Manifest {
    pub columns: Vec<ColumnEntry>,
}
