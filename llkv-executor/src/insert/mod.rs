//! INSERT operation utilities.
//!
//! This module provides utilities for INSERT operations, including value coercion,
//! column resolution, and array building for various data types.

pub mod value_coercion;

pub use value_coercion::{
    build_array_for_column, normalize_insert_value_for_column, 
    resolve_insert_columns,
};
