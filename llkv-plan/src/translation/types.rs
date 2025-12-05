use arrow::datatypes::DataType;
use llkv_result::{Error, Result};

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
        (base.trim(), Some(params))
    } else {
        (normalized.as_str(), None)
    };

    match base_type {
        "TEXT" | "VARCHAR" | "CHAR" | "STRING" | "CLOB" => Ok(DataType::Utf8),
        "INTEGER" | "INT" | "BIGINT" | "SMALLINT" | "TINYINT" => Ok(DataType::Int64),
        "FLOAT" | "REAL" | "DOUBLE" | "DOUBLE PRECISION" => Ok(DataType::Float64),
        "BOOLEAN" | "BOOL" => Ok(DataType::Boolean),
        "DATE" => Ok(DataType::Date32),
        "DECIMAL" | "NUMERIC" => {
            if let Some(params) = params {
                let parts: Vec<&str> = params.split(',').collect();
                if parts.len() == 2 {
                    let precision = parts[0].trim().parse::<u8>().map_err(|_| {
                        Error::InvalidArgumentError(format!(
                            "invalid precision in type: '{}'",
                            type_str
                        ))
                    })?;
                    let scale = parts[1].trim().parse::<i8>().map_err(|_| {
                        Error::InvalidArgumentError(format!(
                            "invalid scale in type: '{}'",
                            type_str
                        ))
                    })?;
                    Ok(DataType::Decimal128(precision, scale))
                } else if parts.len() == 1 {
                    let precision = parts[0].trim().parse::<u8>().map_err(|_| {
                        Error::InvalidArgumentError(format!(
                            "invalid precision in type: '{}'",
                            type_str
                        ))
                    })?;
                    Ok(DataType::Decimal128(precision, 0))
                } else {
                    Err(Error::InvalidArgumentError(format!(
                        "invalid parameters for DECIMAL type: '{}'",
                        type_str
                    )))
                }
            } else {
                // Default precision/scale if not specified?
                // Or error? Postgres defaults to (arbitrary precision), but Arrow needs fixed.
                // Let's default to (38, 10) for now or error.
                Ok(DataType::Decimal128(38, 10))
            }
        }
        _ => Err(Error::InvalidArgumentError(format!(
            "unsupported SQL type: '{}'",
            type_str
        ))),
    }
}
