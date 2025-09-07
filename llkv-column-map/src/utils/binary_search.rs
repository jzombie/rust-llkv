use super::slice::slice_key_by_layout;
use crate::layout::KeyLayout;

pub fn binary_search_key_with_layout(
    bytes: &[u8],
    layout: &KeyLayout,
    n_entries: usize,
    target: &[u8],
) -> Option<usize> {
    let mut lo = 0usize;
    let mut hi = n_entries; // exclusive
    while lo < hi {
        let mid = (lo + hi) / 2;
        let k = slice_key_by_layout(bytes, layout, mid);
        match k.cmp(target) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return Some(mid),
        }
    }
    None
}
