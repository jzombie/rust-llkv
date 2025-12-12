use std::ops::Bound;

use croaring::Treemap;
use llkv_expr::Operator;
use llkv_expr::literal::{FromLiteral, Literal};
use llkv_result::{Error, Result as LlkvResult};
use llkv_types::RowId;

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
                for lit in literals.iter() {
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
