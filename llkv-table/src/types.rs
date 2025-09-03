// llkv-table/src/types.rs

//! Common types for the zero-alloc table core.

#![forbid(unsafe_code)]

use core::cmp::Ordering;

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
// pub type OnRow<'a> = dyn FnMut(&'a [u8], &mut RowEmit<'a>) + 'a;
// // (row_id, emit)

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
