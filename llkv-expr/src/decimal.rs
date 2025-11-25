//! Decimal utilities shared across LLKV crates.
//!
//! The runtime stores decimal values using Arrow's `Decimal128` semantics.
//! This module provides a lightweight helper type for manipulating those
//! values without pulling in heavier dependencies.
//!
//! Note: The implementation has been moved to `llkv-types` to avoid circular dependencies.

pub use llkv_types::decimal::{
    DecimalError, DecimalValue, MAX_DECIMAL_PRECISION, scale_within_bounds,
};
