//! SQL data type classification helpers shared across SQL-facing crates.
//!
//! This module centralizes the mapping from `sqlparser` data types to the
//! compact families our runtime, loader tooling, and planner adapters care about.
//! Keeping the mapping in one place avoids duplicating large `match` statements
//! every time a new SQL type lands, and it makes downstream crates independent
//! of the full `sqlparser` enum surface.

use llkv_result::{Error, Result};
use sqlparser::ast::{DataType, ExactNumberInfo};

/// Reduced set of SQL type families that matter for literal conversion and
/// row-shaping utilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlTypeFamily {
    /// Any textual or JSON-like value that should be treated as a string literal.
    String,
    /// Exact numeric values that fit into the integer pipeline (signed or unsigned).
    Integer,
    /// DECIMAL/NUMERIC values along with their declared scale.
    Decimal { scale: i8 },
    /// Logical DATE / DATE32 columns.
    Date32,
    /// Opaque binary payloads that must not be re-encoded as UTF-8.
    Binary,
}

/// Classify a `sqlparser` [`DataType`] into one of the planner-friendly families.
///
/// The helper intentionally favors broad groups instead of mirroring every
/// dialect-specific variant. Callers that require more nuance (for example,
/// distinguishing `Timestamp` from `Date`) should extend this helper so the
/// mapping continues to live in one place.
pub fn classify_sql_data_type(data_type: &DataType) -> Result<SqlTypeFamily> {
    use DataType::*;

    let family = match data_type {
        Character(_)
        | Char(_)
        | CharacterVarying(_)
        | CharVarying(_)
        | Varchar(_)
        | Nvarchar(_)
        | CharacterLargeObject(_)
        | CharLargeObject(_)
        | Clob(_)
        | Text
        | String(_)
        | Uuid
        | JSON
        | JSONB => SqlTypeFamily::String,
        Binary(_) | Varbinary(_) | Blob(_) | TinyBlob | MediumBlob | LongBlob | Bytes(_) => {
            SqlTypeFamily::Binary
        }
        Date | Date32 => SqlTypeFamily::Date32,
        Decimal(info)
        | DecimalUnsigned(info)
        | Numeric(info)
        | Dec(info)
        | DecUnsigned(info)
        | BigDecimal(info)
        | BigNumeric(info) => SqlTypeFamily::Decimal {
            scale: decimal_scale(info)?,
        },
        TinyInt(_) | TinyIntUnsigned(_) | UTinyInt | Int2(_) | Int2Unsigned(_) | SmallInt(_)
        | SmallIntUnsigned(_) | USmallInt | MediumInt(_) | MediumIntUnsigned(_) | Int(_)
        | Int4(_) | Int8(_) | Int16 | Int32 | Int64 | Int128 | Int256 | Integer(_)
        | IntUnsigned(_) | Int4Unsigned(_) | IntegerUnsigned(_) | HugeInt | UHugeInt | UInt8
        | UInt16 | UInt32 | UInt64 | UInt128 | UInt256 | BigInt(_) | BigIntUnsigned(_)
        | UBigInt | Int8Unsigned(_) | Signed | SignedInteger | Unsigned | UnsignedInteger => {
            SqlTypeFamily::Integer
        }
        other => {
            return Err(Error::InvalidArgumentError(format!(
                "unsupported SQL data type '{other}' for classification"
            )));
        }
    };

    Ok(family)
}

fn decimal_scale(info: &ExactNumberInfo) -> Result<i8> {
    let raw_scale = match info {
        ExactNumberInfo::None | ExactNumberInfo::Precision(_) => 0,
        ExactNumberInfo::PrecisionAndScale(_, scale) => *scale,
    };

    if raw_scale < i64::from(i8::MIN) || raw_scale > i64::from(i8::MAX) {
        return Err(Error::InvalidArgumentError(format!(
            "decimal scale {raw_scale} exceeds i8 range"
        )));
    }

    Ok(raw_scale as i8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_scale_defaults_to_zero() {
        let family = classify_sql_data_type(&DataType::Decimal(ExactNumberInfo::None)).unwrap();
        assert_eq!(family, SqlTypeFamily::Decimal { scale: 0 });
    }

    #[test]
    fn catches_unsupported_types() {
        let err = classify_sql_data_type(&DataType::Boolean).unwrap_err();
        assert!(matches!(err, Error::InvalidArgumentError(_)));
    }
}
