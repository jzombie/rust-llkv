//! Correlated subquery bookkeeping shared across SQL planning and execution.
//!
//! This module centralizes placeholder generation and correlated column tracking so that
//! higher layers can request synthetic bindings without keeping duplicate hash maps or
//! placeholder naming rules in sync. Planner code records correlated accesses while
//! translating SQL expressions and surfaces the collected metadata alongside the final
//! [`SelectPlan`](crate::plans::SelectPlan).

use crate::plans::CorrelatedColumn;
use std::collections::HashMap;
use std::collections::hash_map::Entry;

/// Prefix applied to synthetic field placeholders injected for correlated columns.
pub const CORRELATED_PLACEHOLDER_PREFIX: &str = "__llkv_outer$";

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct OuterColumnKey {
    column: String,
    field_path: Vec<String>,
}

impl OuterColumnKey {
    fn new(column: &str, field_path: &[String]) -> Self {
        Self {
            column: column.to_ascii_lowercase(),
            field_path: field_path
                .iter()
                .map(|segment| segment.to_ascii_lowercase())
                .collect(),
        }
    }
}
/// Tracks correlated columns discovered while translating scalar or EXISTS subqueries.
///
/// The tracker exposes stable placeholder names for a given `(column, field_path)` pair and
/// records the corresponding [`CorrelatedColumn`] metadata in encounter order.
#[derive(Default)]
pub struct CorrelatedColumnTracker {
    placeholders: HashMap<OuterColumnKey, usize>,
    columns: Vec<CorrelatedColumn>,
}

impl CorrelatedColumnTracker {
    /// Create a new tracker instance.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the placeholder name for the given column/field path pair, creating it when absent.
    pub fn placeholder_for_column_path(&mut self, column: &str, field_path: &[String]) -> String {
        let key = OuterColumnKey::new(column, field_path);
        match self.placeholders.entry(key) {
            Entry::Occupied(existing) => correlated_placeholder(*existing.get()),
            Entry::Vacant(slot) => {
                let index = self.columns.len();
                let placeholder = correlated_placeholder(index);
                self.columns.push(CorrelatedColumn {
                    placeholder: placeholder.clone(),
                    column: column.to_string(),
                    field_path: field_path.to_vec(),
                });
                slot.insert(index);
                placeholder
            }
        }
    }

    /// Consume the tracker, yielding correlated column metadata in discovery order.
    #[inline]
    pub fn into_columns(self) -> Vec<CorrelatedColumn> {
        self.columns
    }
}

/// Generate the synthetic placeholder name for a correlated column binding.
#[inline]
pub fn correlated_placeholder(index: usize) -> String {
    format!("{CORRELATED_PLACEHOLDER_PREFIX}{index}")
}

/// Optional wrapper used when correlated tracking is conditional.
pub enum CorrelatedTracker<'a> {
    Active(&'a mut CorrelatedColumnTracker),
    Inactive,
}

impl<'a> CorrelatedTracker<'a> {
    /// Create a tracker wrapper from an optional mutable reference.
    #[inline]
    pub fn from_option(tracker: Option<&'a mut CorrelatedColumnTracker>) -> Self {
        match tracker {
            Some(inner) => CorrelatedTracker::Active(inner),
            None => CorrelatedTracker::Inactive,
        }
    }

    /// Return a placeholder for the provided column path when tracking is active.
    #[inline]
    pub fn placeholder_for_column_path(
        &mut self,
        column: &str,
        field_path: &[String],
    ) -> Option<String> {
        match self {
            CorrelatedTracker::Active(tracker) => {
                Some(tracker.placeholder_for_column_path(column, field_path))
            }
            CorrelatedTracker::Inactive => None,
        }
    }

    /// Reborrow the tracker, preserving the active/inactive state.
    #[inline]
    pub fn reborrow(&mut self) -> CorrelatedTracker<'_> {
        match self {
            CorrelatedTracker::Active(tracker) => CorrelatedTracker::Active(tracker),
            CorrelatedTracker::Inactive => CorrelatedTracker::Inactive,
        }
    }

    /// True when correlation tracking is enabled.
    #[inline]
    pub fn is_active(&self) -> bool {
        matches!(self, CorrelatedTracker::Active(_))
    }
}
