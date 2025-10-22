//! SQL type string parsing to Arrow DataType.
//!
//! This module provides utilities for converting SQL type strings (e.g., "VARCHAR", "INTEGER")
//! into Arrow DataType equivalents. This is used during table creation and schema inference.

use arrow::datatypes::DataType;
use llkv_result::{Error, Result};

/// Parse a SQL data type string to an Arrow DataType.
///
/// Supports common SQL types like TEXT, VARCHAR, INTEGER, etc.
/// For types with precision/scale (e.g., VARCHAR(100)), the precision is ignored
/// and the base type is used.
///
/// # Examples
///
/// ```ignore
/// use llkv_executor::translation::sql_type_to_arrow;
/// use arrow::datatypes::DataType;
///
/// let dtype = sql_type_to_arrow("VARCHAR(255)").unwrap();
/// assert_eq!(dtype, DataType::Utf8);
///
/// let dtype = sql_type_to_arrow("INTEGER").unwrap();
/// assert_eq!(dtype, DataType::Int64);
/// ```
///
/// # Supported Types
///
/// - Text types: TEXT, VARCHAR, CHAR, STRING → Utf8
/// - Integer types: INTEGER, INT, BIGINT, SMALLINT, TINYINT → Int64
/// - Float types: FLOAT, REAL, DOUBLE, DOUBLE PRECISION → Float64
/// - Decimal types: DECIMAL, NUMERIC → Float64
/// - Boolean types: BOOLEAN, BOOL → Boolean
/// - Date types: DATE → Date32
pub fn sql_type_to_arrow(type_str: &str) -> Result<DataType> {
    let normalized = type_str.trim().to_uppercase();

    // Remove precision/scale for simplicity (VARCHAR(100) -> VARCHAR)
    let base_type = if let Some(paren_pos) = normalized.find('(') {
        &normalized[..paren_pos]
    } else {
        &normalized
    };

    match base_type {
        "TEXT" | "VARCHAR" | "CHAR" | "STRING" => Ok(DataType::Utf8),
        "INTEGER" | "INT" | "BIGINT" | "SMALLINT" | "TINYINT" => Ok(DataType::Int64),
        "FLOAT" | "REAL" => Ok(DataType::Float64),
        "DOUBLE" | "DOUBLE PRECISION" => Ok(DataType::Float64),
        "DECIMAL" | "NUMERIC" => Ok(DataType::Float64),
        "BOOLEAN" | "BOOL" => Ok(DataType::Boolean),
        "DATE" => Ok(DataType::Date32),
        _ => Err(Error::InvalidArgumentError(format!(
            "unsupported SQL data type: '{}'",
            type_str
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_types() {
        assert_eq!(sql_type_to_arrow("TEXT").unwrap(), DataType::Utf8);
        assert_eq!(sql_type_to_arrow("VARCHAR").unwrap(), DataType::Utf8);
        assert_eq!(sql_type_to_arrow("VARCHAR(255)").unwrap(), DataType::Utf8);
        assert_eq!(sql_type_to_arrow("CHAR(10)").unwrap(), DataType::Utf8);
        assert_eq!(sql_type_to_arrow("STRING").unwrap(), DataType::Utf8);
    }

    #[test]
    fn test_integer_types() {
        assert_eq!(sql_type_to_arrow("INTEGER").unwrap(), DataType::Int64);
        assert_eq!(sql_type_to_arrow("INT").unwrap(), DataType::Int64);
        assert_eq!(sql_type_to_arrow("BIGINT").unwrap(), DataType::Int64);
        assert_eq!(sql_type_to_arrow("SMALLINT").unwrap(), DataType::Int64);
        assert_eq!(sql_type_to_arrow("TINYINT").unwrap(), DataType::Int64);
    }

    #[test]
    fn test_float_types() {
        assert_eq!(sql_type_to_arrow("FLOAT").unwrap(), DataType::Float64);
        assert_eq!(sql_type_to_arrow("REAL").unwrap(), DataType::Float64);
        assert_eq!(sql_type_to_arrow("DOUBLE").unwrap(), DataType::Float64);
        assert_eq!(sql_type_to_arrow("DECIMAL").unwrap(), DataType::Float64);
        assert_eq!(sql_type_to_arrow("NUMERIC").unwrap(), DataType::Float64);
    }

    #[test]
    fn test_boolean_type() {
        assert_eq!(sql_type_to_arrow("BOOLEAN").unwrap(), DataType::Boolean);
        assert_eq!(sql_type_to_arrow("BOOL").unwrap(), DataType::Boolean);
    }

    #[test]
    fn test_date_type() {
        assert_eq!(sql_type_to_arrow("DATE").unwrap(), DataType::Date32);
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(sql_type_to_arrow("varchar").unwrap(), DataType::Utf8);
        assert_eq!(sql_type_to_arrow("VarChar").unwrap(), DataType::Utf8);
        assert_eq!(sql_type_to_arrow("INTEGER").unwrap(), DataType::Int64);
        assert_eq!(sql_type_to_arrow("integer").unwrap(), DataType::Int64);
    }

    #[test]
    fn test_whitespace_handling() {
        assert_eq!(sql_type_to_arrow("  VARCHAR  ").unwrap(), DataType::Utf8);
        assert_eq!(sql_type_to_arrow("\tINTEGER\n").unwrap(), DataType::Int64);
    }

    #[test]
    fn test_unsupported_type() {
        assert!(sql_type_to_arrow("BLOB").is_err());
        assert!(sql_type_to_arrow("GEOMETRY").is_err());
    }
}
