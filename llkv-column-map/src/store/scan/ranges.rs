//! Range helpers used by scanner predicates.
//!
//! The `IntRanges` structure captures inclusive/exclusive bounds for all
//! primitive Arrow types so filter dispatchers can reason about range pruning
//! without monomorphizing the storage layout.

pub use crate::store::pruning::{IntRanges, RangeKey};

#[inline]
pub(crate) fn lower_idx_by<T: Ord>(
    mut lo: usize,
    mut hi: usize,
    pred: &T,
    get: &dyn Fn(usize) -> T,
) -> usize {
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if get(mid) < *pred {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}
#[inline]
pub(crate) fn upper_idx_by<T: Ord>(
    mut lo: usize,
    mut hi: usize,
    pred: &T,
    get: &dyn Fn(usize) -> T,
) -> usize {
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if get(mid) <= *pred {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}
