use std::cmp::Ordering;

use arrow::array::{Array, PrimitiveArray};
use arrow::datatypes::ArrowPrimitiveType;
use croaring::Treemap;
use llkv_types::RowId;

/// Compare optional values with configurable ordering and null placement.
pub fn compare_option_values<T: Ord>(
    left: Option<T>,
    right: Option<T>,
    ascending: bool,
    nulls_first: bool,
) -> Ordering {
    match (left, right) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => {
            if nulls_first {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (Some(_), None) => {
            if nulls_first {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        (Some(a), Some(b)) => {
            if ascending {
                a.cmp(&b)
            } else {
                b.cmp(&a)
            }
        }
    }
}

/// Sort row ids by the values stored in `chunks`, preserving stable ordering for ties.
pub fn sort_row_ids_by_primitive<T>(
    row_ids: &Treemap,
    chunks: &[PrimitiveArray<T>],
    positions: &[(usize, usize)],
    ascending: bool,
    nulls_first: bool,
) -> Vec<RowId>
where
    T: ArrowPrimitiveType,
    T::Native: Ord,
{
    let mut indices: Vec<(usize, RowId)> = row_ids.iter().enumerate().collect();
    indices.sort_by(|(ai, arid), (bi, brid)| {
        let (chunk_a, offset_a) = positions[*ai];
        let (chunk_b, offset_b) = positions[*bi];
        let array_a = &chunks[chunk_a];
        let array_b = &chunks[chunk_b];
        let left = if array_a.is_null(offset_a) {
            None
        } else {
            Some(array_a.value(offset_a))
        };
        let right = if array_b.is_null(offset_b) {
            None
        } else {
            Some(array_b.value(offset_b))
        };
        let ord = compare_option_values(left, right, ascending, nulls_first);
        if ord == Ordering::Equal {
            arid.cmp(brid)
        } else {
            ord
        }
    });
    indices.into_iter().map(|(_, rid)| rid).collect()
}
