use super::ColumnStore;
use crate::storage::Pager;
use crate::types::LogicalFieldId;

pub struct ValueScanOpts<'a> {
    pub lo: core::ops::Bound<&'a [u8]>,
    pub hi: core::ops::Bound<&'a [u8]>,
    pub prefix: Option<&'a [u8]>, // optional value prefix
    pub head_tag_len: usize,      // 8 or 16 bytes for PQ tiebreaks // TODO: Document, what is `PQ`?
    pub window_prefix_len: usize, // 0..=2 for windowed seeding
}

pub struct ValueScanItem<B> {
    pub key: &'static [u8], // or a borrowed slice from key bytes
    pub value: crate::views::ValueSlice<B>,
}

// TODO: Implement
// impl<'p, P: Pager> ColumnStore<'p, P> {
//     pub fn scan_values_lww<'a>(
//         &self,
//         field_id: LogicalFieldId,
//         opts: ValueScanOpts<'a>,
//     ) -> impl Iterator<Item = ValueScanItem<P::Blob>> + 'a {
//         // skeleton to be filled next step: prune segments by value_min/max,
//         // seed cursors via l1/l2 + lower_bound_by_value, then stream with
//         // the 2-tier radix PQ. Emits newest-only rows when LWW table is
//         // available; otherwise falls back to newest-first filtering.
//         unimplemented!()
//     }
// }
