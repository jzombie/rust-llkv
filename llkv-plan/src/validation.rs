//! Shared validation utilities for plan construction.
//!
//! The helpers here return [`llkv_result::Error::InvalidArgumentError`] with caller-provided
//! messages so the planner can surface precise diagnostics while keeping the validation logic
//! reusable across plan builders.

#![forbid(unsafe_code)]

use llkv_result::{Error, Result};
use std::collections::HashSet;

/// Ensure that a slice is not empty.
pub fn ensure_non_empty<T, F>(items: &[T], make_error: F) -> Result<()>
where
    F: Fn() -> String,
{
    if items.is_empty() {
        return Err(Error::InvalidArgumentError(make_error()));
    }
    Ok(())
}

/// Ensure that all names are unique when compared case-insensitively.
pub fn ensure_unique_case_insensitive<'a, I, F>(names: I, make_error: F) -> Result<()>
where
    I: IntoIterator<Item = &'a str>,
    F: Fn(&str) -> String,
{
    let mut seen: HashSet<String> = HashSet::new();
    for name in names {
        let normalized = name.to_ascii_lowercase();
        if !seen.insert(normalized) {
            return Err(Error::InvalidArgumentError(make_error(name)));
        }
    }
    Ok(())
}

/// Ensure that every column in `columns` exists in `known_lower` (case-insensitive lookup).
pub fn ensure_known_columns_case_insensitive<'a, I, F>(
    columns: I,
    known_lower: &HashSet<String>,
    make_error: F,
) -> Result<()>
where
    I: IntoIterator<Item = &'a str>,
    F: Fn(&str) -> String,
{
    for column in columns {
        let normalized = column.to_ascii_lowercase();
        if !known_lower.contains(&normalized) {
            return Err(Error::InvalidArgumentError(make_error(column)));
        }
    }
    Ok(())
}
