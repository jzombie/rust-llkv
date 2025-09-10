//! Common types for the table core.

#![forbid(unsafe_code)]

use core::cmp::Ordering;
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
