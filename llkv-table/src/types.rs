//! Common types for the zero-alloc table core.

#![forbid(unsafe_code)]

use core::cmp::Ordering;
use llkv_btree::{
    bplus_tree::SharedBPlusTree,
    codecs::{BigEndianIdCodec, BigEndianKeyCodec},
    pager::Pager as BTreePager,
    views::value_view::ValueRef,
};

/// Field identifier type for addressing columns.
///
/// FieldId is a concrete integer here for ergonomics. If you later need
/// a different width or representation, you can swap this alias.
pub type FieldId = u32;

pub type RowId = u64;

/// Comparator for opaque row-ids represented as byte slices.
///
/// The comparator must implement a total ordering that matches the
/// order produced by all Sources. Callers provide the function to avoid
/// baking any row-id semantics into this layer.
pub type RowIdCmp = fn(&[u8], &[u8]) -> Ordering;

// pub type ValSink<'a> = dyn FnMut(&'a [u8]) + 'a;
// // consumes one value
// pub type RowEmit<'a> = dyn FnMut(&mut ValSink<'a>) + 'a;
// // calls sink for each value
// Small, descriptive aliases
// type PageT = <dyn Pager as BTreePager>::Page;
// type VRef = ValueRef<PageT>;
// type VIter<'a> = dyn Iterator<Item = VRef> + 'a;

// /// Callback: (row_id, iterator over projected value refs)
pub type PageOf<P> = <P as BTreePager>::Page;
pub type VRefOf<P> = ValueRef<PageOf<P>>;
pub type VIterOf<'a, P> = dyn Iterator<Item = VRefOf<P>> + 'a;
pub type OnRowOf<'a, P> = dyn FnMut(RowId, &mut VIterOf<'a, P>) + 'a;

// TODO: Don't hardcode u64
pub type ColumnTree<P> = SharedBPlusTree<P, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>>;
pub type PrimaryIndexTree<P> = SharedBPlusTree<P, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>>;
pub type RowIdSetTree<P> = SharedBPlusTree<P, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_id_cmp_lexicographic() {
        fn cmp(a: &[u8], b: &[u8]) -> Ordering {
            a.cmp(b)
        }
        assert_eq!(cmp(b"a", b"a"), Ordering::Equal);
        assert_eq!(cmp(b"a", b"b"), Ordering::Less);
        assert_eq!(cmp(b"b", b"a"), Ordering::Greater);
    }
}
