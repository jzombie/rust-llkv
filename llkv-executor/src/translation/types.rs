//! SQL type string parsing to Arrow DataType.
//!
//! This module provides utilities for converting SQL type strings (e.g., "VARCHAR", "INTEGER")
//! into Arrow DataType equivalents. This is used during table creation and schema inference.

use arrow::datatypes::DataType;
use llkv_result::{Error, Result};

/// Parse a SQL data type string to an Arrow DataType.
///
/// Supports common SQL types like TEXT, VARCHAR, INTEGER, etc.
/// For types with precision/scale (e.g., DECIMAL(10,2)), the precision and scale are preserved.
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
///
/// let dtype = sql_type_to_arrow("DECIMAL(10,2)").unwrap();
/// assert_eq!(dtype, DataType::Decimal128(10, 2));
/// ```
///
/// # Supported Types
///
/// - Text types: TEXT, VARCHAR, CHAR, STRING → Utf8
/// - Integer types: INTEGER, INT, BIGINT, SMALLINT, TINYINT → Int64
/// - Float types: FLOAT, REAL, DOUBLE, DOUBLE PRECISION → Float64
/// - Decimal types: DECIMAL(p,s), NUMERIC(p,s) → Decimal128(p, s)
/// - Boolean types: BOOLEAN, BOOL → Boolean
/// - Date types: DATE → Date32
pub fn sql_type_to_arrow(type_str: &str) -> Result<DataType> {
    let normalized = type_str.trim().to_uppercase();

    // Extract precision/scale for DECIMAL/NUMERIC types
    let (base_type, params) = if let Some(paren_pos) = normalized.find('(') {
        let base = &normalized[..paren_pos];
        let end_paren = normalized.find(')').ok_or_else(|| {
            Error::InvalidArgumentError(format!(
                "missing closing parenthesis in type: '{}'",
                type_str
            ))
        })?;
        let params = &normalized[paren_pos + 1..end_paren];
        (base, Some(params))
    } else {
        (&normalized[..], None)
    };

    match base_type {
        "TEXT" | "VARCHAR" | "CHAR" | "STRING" => Ok(DataType::Utf8),
        "INTEGER" | "INT" | "BIGINT" | "SMALLINT" | "TINYINT" => Ok(DataType::Int64),
        "FLOAT" | "REAL" => Ok(DataType::Float64),
        "DOUBLE" | "DOUBLE PRECISION" => Ok(DataType::Float64),
        "DECIMAL" | "NUMERIC" => {
            if let Some(params_str) = params {
                parse_decimal_params(params_str)
            } else {
                // Default DECIMAL without parameters: DECIMAL(38, 10)
                Ok(DataType::Decimal128(38, 10))
            }
        }
        "BOOLEAN" | "BOOL" => Ok(DataType::Boolean),
        "DATE" => Ok(DataType::Date32),
        _ => Err(Error::InvalidArgumentError(format!(
            "unsupported SQL data type: '{}'",
            type_str
        ))),
    }
}

/// Parse DECIMAL(precision, scale) parameters.
fn parse_decimal_params(params: &str) -> Result<DataType> {
    let parts: Vec<&str> = params.split(',').map(|s| s.trim()).collect();

    match parts.len() {
        1 => {
            // DECIMAL(p) means scale = 0
            let precision = parts[0].parse::<u8>().map_err(|_| {
                Error::InvalidArgumentError(format!("invalid DECIMAL precision: '{}'", parts[0]))
            })?;

            if precision == 0 || precision > 38 {
                return Err(Error::InvalidArgumentError(format!(
                    "DECIMAL precision must be between 1 and 38, got: {}",
                    precision
                )));
            }

            Ok(DataType::Decimal128(precision, 0))
        }
        2 => {
            // DECIMAL(p, s)
            let precision = parts[0].parse::<u8>().map_err(|_| {
                Error::InvalidArgumentError(format!("invalid DECIMAL precision: '{}'", parts[0]))
            })?;
            let scale = parts[1].parse::<i8>().map_err(|_| {
                Error::InvalidArgumentError(format!("invalid DECIMAL scale: '{}'", parts[1]))
            })?;

            if precision == 0 || precision > 38 {
                return Err(Error::InvalidArgumentError(format!(
                    "DECIMAL precision must be between 1 and 38, got: {}",
                    precision
                )));
            }

            if scale < 0 || scale as u8 > precision {
                return Err(Error::InvalidArgumentError(format!(
                    "DECIMAL scale must be between 0 and precision ({}), got: {}",
                    precision, scale
                )));
            }

            Ok(DataType::Decimal128(precision, scale))
        }
        _ => Err(Error::InvalidArgumentError(format!(
            "invalid DECIMAL parameters: expected (precision) or (precision, scale), got: '{}'",
            params
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
    }

    #[test]
    fn test_decimal_types() {
        // Default DECIMAL without parameters
        assert_eq!(
            sql_type_to_arrow("DECIMAL").unwrap(),
            DataType::Decimal128(38, 10)
        );
        assert_eq!(
            sql_type_to_arrow("NUMERIC").unwrap(),
            DataType::Decimal128(38, 10)
        );

        // DECIMAL with precision only
        assert_eq!(
            sql_type_to_arrow("DECIMAL(10)").unwrap(),
            DataType::Decimal128(10, 0)
        );
        assert_eq!(
            sql_type_to_arrow("NUMERIC(15)").unwrap(),
            DataType::Decimal128(15, 0)
        );

        // DECIMAL with precision and scale
        assert_eq!(
            sql_type_to_arrow("DECIMAL(10,2)").unwrap(),
            DataType::Decimal128(10, 2)
        );
        assert_eq!(
            sql_type_to_arrow("NUMERIC(18,4)").unwrap(),
            DataType::Decimal128(18, 4)
        );
        assert_eq!(
            sql_type_to_arrow("DECIMAL(38,10)").unwrap(),
            DataType::Decimal128(38, 10)
        );

        // With whitespace
        assert_eq!(
            sql_type_to_arrow("DECIMAL(10, 2)").unwrap(),
            DataType::Decimal128(10, 2)
        );
        assert_eq!(
            sql_type_to_arrow("DECIMAL( 10 , 2 )").unwrap(),
            DataType::Decimal128(10, 2)
        );

        // Invalid cases
        assert!(sql_type_to_arrow("DECIMAL(0)").is_err()); // precision must be >= 1
        assert!(sql_type_to_arrow("DECIMAL(39)").is_err()); // precision must be <= 38
        assert!(sql_type_to_arrow("DECIMAL(10, 15)").is_err()); // scale > precision
        assert!(sql_type_to_arrow("DECIMAL(10, -1)").is_err()); // negative scale
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
