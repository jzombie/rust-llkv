// TODO: Most of these methods should be more generic than just row ids, and the file should be renamed.

use std::ops::Bound;

use llkv_expr::Operator;
use llkv_expr::literal::{FromLiteral, Literal};
use llkv_result::{Error, Result as LlkvResult};
use rustc_hash::FxHashSet;

/// Sort and deduplicate a set of row ids.
pub fn normalize_row_ids(mut row_ids: Vec<u64>) -> Vec<u64> {
    row_ids.sort_unstable();
    row_ids.dedup();
    row_ids
}

// TODO: Move to Literal impl
pub fn literal_to_u64(lit: &Literal) -> LlkvResult<u64> {
    u64::from_literal(lit)
        .map_err(|err| Error::InvalidArgumentError(format!("rowid literal cast failed: {err}")))
}

pub fn lower_bound_index(row_ids: &[u64], bound: &Bound<Literal>) -> LlkvResult<usize> {
    Ok(match bound {
        Bound::Unbounded => 0,
        Bound::Included(lit) => {
            let value = literal_to_u64(lit)?;
            row_ids.partition_point(|&rid| rid < value)
        }
        Bound::Excluded(lit) => {
            let value = literal_to_u64(lit)?;
            row_ids.partition_point(|&rid| rid <= value)
        }
    })
}

pub fn upper_bound_index(row_ids: &[u64], bound: &Bound<Literal>) -> LlkvResult<usize> {
    Ok(match bound {
        Bound::Unbounded => row_ids.len(),
        Bound::Included(lit) => {
            let value = literal_to_u64(lit)?;
            row_ids.partition_point(|&rid| rid <= value)
        }
        Bound::Excluded(lit) => {
            let value = literal_to_u64(lit)?;
            row_ids.partition_point(|&rid| rid < value)
        }
    })
}

// TODO: Don't hardcode u64 type; use RowId type parameter
pub fn filter_row_ids_by_operator(row_ids: &[u64], op: &Operator<'_>) -> LlkvResult<Vec<u64>> {
    use Operator::*;

    match op {
        Equals(lit) => {
            let value = literal_to_u64(lit)?;
            match row_ids.binary_search(&value) {
                Ok(idx) => Ok(vec![row_ids[idx]]),
                Err(_) => Ok(Vec::new()),
            }
        }
        GreaterThan(lit) => {
            let value = literal_to_u64(lit)?;
            let idx = row_ids.partition_point(|&rid| rid <= value);
            Ok(row_ids[idx..].to_vec())
        }
        GreaterThanOrEquals(lit) => {
            let value = literal_to_u64(lit)?;
            let idx = row_ids.partition_point(|&rid| rid < value);
            Ok(row_ids[idx..].to_vec())
        }
        LessThan(lit) => {
            let value = literal_to_u64(lit)?;
            let idx = row_ids.partition_point(|&rid| rid < value);
            Ok(row_ids[..idx].to_vec())
        }
        LessThanOrEquals(lit) => {
            let value = literal_to_u64(lit)?;
            let idx = row_ids.partition_point(|&rid| rid <= value);
            Ok(row_ids[..idx].to_vec())
        }
        Range { lower, upper } => {
            let start = lower_bound_index(row_ids, lower)?;
            let end = upper_bound_index(row_ids, upper)?;
            if start >= end {
                Ok(Vec::new())
            } else {
                Ok(row_ids[start..end].to_vec())
            }
        }
        In(literals) => {
            if literals.is_empty() {
                return Ok(Vec::new());
            }
            let mut targets = FxHashSet::default();
            for lit in *literals {
                targets.insert(literal_to_u64(lit)?);
            }
            if targets.is_empty() {
                return Ok(Vec::new());
            }
            let mut matches = Vec::with_capacity(targets.len());
            for &rid in row_ids {
                if targets.remove(&rid) {
                    matches.push(rid);
                    if targets.is_empty() {
                        break;
                    }
                }
            }
            Ok(matches)
        }
        StartsWith { .. } | EndsWith { .. } | Contains { .. } => Err(Error::InvalidArgumentError(
            "rowid predicates do not support string pattern matching".into(),
        )),
        IsNull | IsNotNull => Err(Error::InvalidArgumentError(
            "rowid predicates do not support null checks".into(),
        )),
    }
}

// TODO: Move to another file
// TODO: Make generic over T: Ord
// TODO: Can this be made more efficient by avoiding allocations and/or vectorizing?
pub fn intersect_sorted(left: Vec<u64>, right: Vec<u64>) -> Vec<u64> {
    let mut result = Vec::with_capacity(left.len().min(right.len()));
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        let lv = left[i];
        let rv = right[j];
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

// TODO: Move to another file
// TODO: Make generic over T: Ord
// TODO: Can this be made more efficient by avoiding allocations and/or vectorizing?
pub fn union_sorted(left: Vec<u64>, right: Vec<u64>) -> Vec<u64> {
    if left.is_empty() {
        return right;
    }
    if right.is_empty() {
        return left;
    }

    let mut result = Vec::with_capacity(left.len() + right.len());
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        let lv = left[i];
        let rv = right[j];
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

    while i < left.len() {
        result.push(left[i]);
        i += 1;
    }
    while j < right.len() {
        result.push(right[j]);
        j += 1;
    }

    result.dedup();
    result
}

// TODO: Move to another file
// TODO: Make generic over T: Ord
// TODO: Can this be made more efficient by avoiding allocations and/or vectorizing?
pub fn difference_sorted(base: Vec<u64>, subtract: Vec<u64>) -> Vec<u64> {
    if base.is_empty() || subtract.is_empty() {
        return base;
    }

    let mut result = Vec::with_capacity(base.len());
    let mut i = 0;
    let mut j = 0;
    while i < base.len() && j < subtract.len() {
        let bv = base[i];
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
    while i < base.len() {
        result.push(base[i]);
        i += 1;
    }
    result
}

// TODO: Make generic over T: Ord
// TODO: Can this be made more efficient by avoiding allocations and/or vectorizing?
pub fn difference_sorted_slice(base: &[u64], subtract: &[u64]) -> Vec<u64> {
    if base.is_empty() {
        return Vec::new();
    }
    if subtract.is_empty() {
        return base.to_vec();
    }

    let mut result = Vec::with_capacity(base.len());
    let mut i = 0;
    let mut j = 0;
    while i < base.len() && j < subtract.len() {
        let bv = base[i];
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
    while i < base.len() {
        result.push(base[i]);
        i += 1;
    }
    result
}
