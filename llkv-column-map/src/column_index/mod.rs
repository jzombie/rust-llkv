mod bootstrap;
pub use bootstrap::*;

mod column_entry;
pub use column_entry::*;

mod index_segment;
pub use index_segment::*;

mod manifest;
pub use manifest::*;

use crate::types::LogicalFieldId;

/// Column index header for one logical field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnIndex {
    pub field_id: LogicalFieldId,
    pub segments: Vec<IndexSegmentRef>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{KeyLayout, ValueLayout};
    use crate::types::{LogicalFieldId, PhysicalKey};

    #[test]
    fn sanity() {
        let idx = ColumnIndex {
            field_id: 1 as LogicalFieldId,
            segments: vec![IndexSegmentRef {
                index_physical_key: 2 as PhysicalKey,
                data_physical_key: 3 as PhysicalKey,
                logical_key_min: vec![0, 1],
                logical_key_max: vec![0xff, 0xfe],
                n_entries: 0,
            }],
        };
        assert_eq!(idx.field_id, 1);
    }
}
