// TODO: Rename this file.

use std::cmp::Ordering;
use std::ops::Bound;

use arrow::array::{Array, PrimitiveArray};
use arrow::datatypes::ArrowPrimitiveType;
use croaring::Treemap;
use llkv_column_map::types::RowId;
use llkv_expr::Operator;
use llkv_expr::literal::{FromLiteral, Literal};
use llkv_result::{Error, Result as LlkvResult};

pub fn literal_to_row_id(lit: &Literal) -> LlkvResult<RowId> {
    u64::from_literal(lit)
        .map_err(|err| Error::InvalidArgumentError(format!("rowid literal cast failed: {err}")))
}

// TODO: Is this more generic than just RowId filtering? If so, it should be renamed and made more generic.
pub trait RowIdFilter {
    fn filter_by_operator(&self, op: &Operator<'_>) -> LlkvResult<Treemap>;
}

impl RowIdFilter for Treemap {
    fn filter_by_operator(&self, op: &Operator<'_>) -> LlkvResult<Treemap> {
        use Operator::*;

        match op {
            Equals(lit) => {
                let value = literal_to_row_id(lit)?;
                if self.contains(value) {
                    Ok(Treemap::from_iter([value]))
                } else {
                    Ok(Treemap::new())
                }
            }
            GreaterThan(lit) => {
                let value = literal_to_row_id(lit)?;
                let mut result = self.clone();
                // Remove <= value. i.e. 0..=value
                result.remove_range(0..=value);
                Ok(result)
            }
            GreaterThanOrEquals(lit) => {
                let value = literal_to_row_id(lit)?;
                let mut result = self.clone();
                // Remove < value. i.e. 0..value
                result.remove_range(0..value);
                Ok(result)
            }
            LessThan(lit) => {
                let value = literal_to_row_id(lit)?;
                let mut result = self.clone();
                // Remove >= value. i.e. value..=MAX
                result.remove_range(value..=u64::MAX);
                Ok(result)
            }
            LessThanOrEquals(lit) => {
                let value = literal_to_row_id(lit)?;
                let mut result = self.clone();
                // Remove > value. i.e. (value+1)..=MAX
                if value < u64::MAX {
                    result.remove_range((value + 1)..=u64::MAX);
                }
                Ok(result)
            }
            Range { lower, upper } => {
                let mut result = self.clone();
                match lower {
                    Bound::Included(lit) => {
                        let val = literal_to_row_id(lit)?;
                        result.remove_range(0..val);
                    }
                    Bound::Excluded(lit) => {
                        let val = literal_to_row_id(lit)?;
                        result.remove_range(0..=val);
                    }
                    Bound::Unbounded => {}
                }
                match upper {
                    Bound::Included(lit) => {
                        let val = literal_to_row_id(lit)?;
                        if val < u64::MAX {
                            result.remove_range((val + 1)..=u64::MAX);
                        }
                    }
                    Bound::Excluded(lit) => {
                        let val = literal_to_row_id(lit)?;
                        result.remove_range(val..=u64::MAX);
                    }
                    Bound::Unbounded => {}
                }
                Ok(result)
            }
            In(literals) => {
                if literals.is_empty() {
                    return Ok(Treemap::new());
                }
                let mut targets = Treemap::new();
                for lit in *literals {
                    targets.add(literal_to_row_id(lit)?);
                }
                Ok(self & &targets)
            }
            StartsWith { .. } | EndsWith { .. } | Contains { .. } => {
                Err(Error::InvalidArgumentError(
                    "rowid predicates do not support string pattern matching".into(),
                ))
            }
            IsNull | IsNotNull => Err(Error::InvalidArgumentError(
                "rowid predicates do not support null checks".into(),
            )),
        }
    }
}

// TODO: Move to new `compare` module (within llkv-compute)
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

// TODO: Rename to `sort_row_ids_by_primitive` (or equivalent)
/// Sorts row ids by the values stored in `chunks`, preserving stable ordering for ties.
pub fn sort_by_primitive<T>(
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
