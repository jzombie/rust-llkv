// TODO: Rename this file.

use std::ops::Bound;

use llkv_column_map::types::RowId;
use llkv_expr::Operator;
use llkv_expr::literal::{FromLiteral, Literal};
use llkv_result::{Error, Result as LlkvResult};
use rustc_hash::FxHashSet;

/// Sort and deduplicate a set of values (typically row ids).
pub fn normalize_row_ids<T: Ord + Copy>(mut values: Vec<T>) -> Vec<T> {
    values.sort_unstable();
    values.dedup();
    values
}

pub fn literal_to_row_id(lit: &Literal) -> LlkvResult<RowId> {
    u64::from_literal(lit)
        .map_err(|err| Error::InvalidArgumentError(format!("rowid literal cast failed: {err}")))
}

pub fn lower_bound_index(row_ids: &[RowId], bound: &Bound<Literal>) -> LlkvResult<usize> {
    Ok(match bound {
        Bound::Unbounded => 0,
        Bound::Included(lit) => {
            let value = literal_to_row_id(lit)?;
            row_ids.partition_point(|&rid| rid < value)
        }
        Bound::Excluded(lit) => {
            let value = literal_to_row_id(lit)?;
            row_ids.partition_point(|&rid| rid <= value)
        }
    })
}

pub fn upper_bound_index(row_ids: &[RowId], bound: &Bound<Literal>) -> LlkvResult<usize> {
    Ok(match bound {
        Bound::Unbounded => row_ids.len(),
        Bound::Included(lit) => {
            let value = literal_to_row_id(lit)?;
            row_ids.partition_point(|&rid| rid <= value)
        }
        Bound::Excluded(lit) => {
            let value = literal_to_row_id(lit)?;
            row_ids.partition_point(|&rid| rid < value)
        }
    })
}

// TODO: Is this more generic than just RowId filtering? If so, it should be renamed and made more generic.
pub trait RowIdSliceExt {
    fn filter_by_operator(&self, op: &Operator<'_>) -> LlkvResult<Vec<RowId>>;
}

impl RowIdSliceExt for [RowId] {
    fn filter_by_operator(&self, op: &Operator<'_>) -> LlkvResult<Vec<RowId>> {
        use Operator::*;

        match op {
            Equals(lit) => {
                let value = literal_to_row_id(lit)?;
                Ok(self
                    .binary_search(&value)
                    .map_or(Vec::new(), |idx| vec![self[idx]]))
            }
            GreaterThan(lit) => {
                let value = literal_to_row_id(lit)?;
                let idx = self.partition_point(|&rid| rid <= value);
                Ok(self[idx..].to_vec())
            }
            GreaterThanOrEquals(lit) => {
                let value = literal_to_row_id(lit)?;
                let idx = self.partition_point(|&rid| rid < value);
                Ok(self[idx..].to_vec())
            }
            LessThan(lit) => {
                let value = literal_to_row_id(lit)?;
                let idx = self.partition_point(|&rid| rid < value);
                Ok(self[..idx].to_vec())
            }
            LessThanOrEquals(lit) => {
                let value = literal_to_row_id(lit)?;
                let idx = self.partition_point(|&rid| rid <= value);
                Ok(self[..idx].to_vec())
            }
            Range { lower, upper } => {
                let start = lower_bound_index(self, lower)?;
                let end = upper_bound_index(self, upper)?;
                if start >= end {
                    Ok(Vec::new())
                } else {
                    Ok(self[start..end].to_vec())
                }
            }
            In(literals) => {
                if literals.is_empty() {
                    return Ok(Vec::new());
                }
                let mut targets = FxHashSet::default();
                for lit in *literals {
                    targets.insert(literal_to_row_id(lit)?);
                }
                if targets.is_empty() {
                    return Ok(Vec::new());
                }
                let mut matches = Vec::with_capacity(targets.len());
                for &rid in self {
                    if targets.remove(&rid) {
                        matches.push(rid);
                        if targets.is_empty() {
                            break;
                        }
                    }
                }
                Ok(matches)
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

pub trait SortedSliceOps<T: Ord + Copy> {
    fn intersect_sorted(&self, other: &[T]) -> Vec<T>;
    fn union_sorted(&self, other: &[T]) -> Vec<T>;
    fn difference_sorted(&self, subtract: &[T]) -> Vec<T>;
}

impl<T: Ord + Copy> SortedSliceOps<T> for [T] {
    // TODO: Can this be vectored with SIMD?
    fn intersect_sorted(&self, other: &[T]) -> Vec<T> {
        let mut result = Vec::with_capacity(self.len().min(other.len()));
        let mut i = 0;
        let mut j = 0;
        while i < self.len() && j < other.len() {
            let lv = self[i];
            let rv = other[j];
            if lv == rv {
                result.push(lv);
                i += 1;
                j += 1;
            } else if lv < rv {
                i += 1;
            } else {
                j += 1;
            }
        }
        result
    }

    // TODO: Can this be vectored with SIMD?
    fn union_sorted(&self, other: &[T]) -> Vec<T> {
        if self.is_empty() {
            return other.to_vec();
        }
        if other.is_empty() {
            return self.to_vec();
        }

        let mut result = Vec::with_capacity(self.len() + other.len());
        let mut i = 0;
        let mut j = 0;
        while i < self.len() && j < other.len() {
            let lv = self[i];
            let rv = other[j];
            if lv == rv {
                result.push(lv);
                i += 1;
                j += 1;
            } else if lv < rv {
                result.push(lv);
                i += 1;
            } else {
                result.push(rv);
                j += 1;
            }
        }

        while i < self.len() {
            result.push(self[i]);
            i += 1;
        }
        while j < other.len() {
            result.push(other[j]);
            j += 1;
        }

        result.dedup();
        result
    }

    // TODO: Can this be vectored with SIMD?
    fn difference_sorted(&self, subtract: &[T]) -> Vec<T> {
        if self.is_empty() {
            return Vec::new();
        }
        if subtract.is_empty() {
            return self.to_vec();
        }

        let mut result = Vec::with_capacity(self.len());
        let mut i = 0;
        let mut j = 0;
        while i < self.len() && j < subtract.len() {
            let bv = self[i];
            let sv = subtract[j];
            if bv == sv {
                i += 1;
                j += 1;
            } else if bv < sv {
                result.push(bv);
                i += 1;
            } else {
                j += 1;
            }
        }
        while i < self.len() {
            result.push(self[i]);
            i += 1;
        }
        result
    }
}
