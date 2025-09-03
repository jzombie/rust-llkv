//! Common types for the table core.

#![forbid(unsafe_code)]

use core::cmp::Ordering;
use llkv_btree::{pager::Pager as BTreePager, views::value_view::ValueRef};
use std::borrow::Cow;
use std::collections::HashMap;

/// Field identifier type for addressing columns.
///
/// FieldId is a concrete integer here for ergonomics. If you later need
/// a different width or representation, you can swap this alias.
pub type FieldId = u32;
pub type RowId = u64;
pub type IndexKey = u64;

pub type RootId = u64;
pub type RootIdBytes = [u8; 8];

/// Comparator for opaque row-ids represented as byte slices.
///
/// The comparator must implement a total ordering that matches the
/// order produced by all Sources. Callers provide the function to avoid
/// baking any row-id semantics into this layer.
pub type RowIdCmp = fn(&[u8], &[u8]) -> Ordering;

pub type ColumnValue<'a> = &'a [u8];

/// A column value provided as input for an insert or update operation.
///
/// This uses a Cow (Copy-on-Write) to allow for either borrowed (zero-copy)
/// or owned data to be passed to the table.
pub type ColumnInput<'a> = Cow<'a, [u8]>;

// /// Callback: (row_id, iterator over projected value refs)
pub type PageOf<P> = <P as BTreePager>::Page;
pub type VRefOf<P> = ValueRef<PageOf<P>>;
pub type VIterOf<'a, P> = dyn Iterator<Item = VRefOf<P>> + 'a;
pub type OnRowOf<'a, P> = dyn FnMut(RowId, &mut VIterOf<'a, P>) + 'a;

// Codec aliases
pub type RowIdKeyCodec = llkv_btree::codecs::BigEndianKeyCodec<RowId>;
pub type IndexKeyCodec = llkv_btree::codecs::BigEndianKeyCodec<IndexKey>;
pub type RootIdIdCodec = llkv_btree::codecs::BigEndianIdCodec<RootId>;

// Tree aliases
pub type ColumnTree<P> = llkv_btree::bplus_tree::SharedBPlusTree<P, RowIdKeyCodec, RootIdIdCodec>;
pub type PrimaryIndexTree<P> =
    llkv_btree::bplus_tree::SharedBPlusTree<P, IndexKeyCodec, RootIdIdCodec>;
pub type RowIdSetTree<P> = llkv_btree::bplus_tree::SharedBPlusTree<P, RowIdKeyCodec, RootIdIdCodec>;

pub type RowPatch<'a> = (RowId, HashMap<FieldId, (IndexKey, ColumnInput<'a>)>);

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
