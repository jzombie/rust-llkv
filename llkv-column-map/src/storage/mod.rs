pub mod pager;
pub use pager::*;

use crate::layout::IndexLayoutInfo;
use crate::types::{IndexEntryCount, LogicalFieldId, PhysicalKey};

#[derive(Clone, Debug)]
pub struct StorageNode {
    pub pk: PhysicalKey,
    pub stored_len: usize,
    pub kind: StorageKind,
}

#[derive(Clone, Debug)]
pub enum StorageKind {
    Bootstrap,
    Manifest {
        column_count: usize,
    },
    ColumnIndex {
        field_id: LogicalFieldId,
        n_segments: usize,
    },
    IndexSegment {
        field_id: LogicalFieldId,
        n_entries: IndexEntryCount,
        layout: IndexLayoutInfo,
        data_pkey: PhysicalKey,
        owner_colindex_pk: PhysicalKey,
    },
    DataBlob {
        owner_index_pk: PhysicalKey,
    },
}
