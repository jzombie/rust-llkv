//! Untyped literal values.

use crate::decimal::DecimalValue;
use crate::interval::IntervalValue;

/// A literal value that has not yet been coerced into a specific native
/// type. This allows for type inference to be deferred until the column
/// type is known.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Null,
    Integer(i128),
    Float(f64),
    /// Decimal literal stored as scaled integer with fixed precision.
    Decimal(DecimalValue),
    String(String),
    Boolean(bool),
    /// Date literal stored as days since the Unix epoch (1970-01-01).
    Date32(i32),
    /// Struct literal with field names and nested literals
    Struct(Vec<(String, Box<Literal>)>),
    /// Interval literal with mixed calendar and sub-day precision.
    Interval(IntervalValue),
    // Other types like Bytes, etc. can be added here.
}

macro_rules! impl_from_for_literal {
    ($variant:ident, $($t:ty),*) => {
        $(
            impl From<$t> for Literal {
                fn from(v: $t) -> Self {
                    Literal::$variant(v.into())
                }
            }
        )*
    };
}

impl_from_for_literal!(Integer, i8, i16, i32, i64, i128, u8, u16, u32, u64);
impl_from_for_literal!(Float, f32, f64);
impl_from_for_literal!(String, String);
impl_from_for_literal!(Boolean, bool);
impl_from_for_literal!(Decimal, DecimalValue);
impl_from_for_literal!(Interval, IntervalValue);

impl From<&str> for Literal {
    fn from(v: &str) -> Self {
        Literal::String(v.to_string())
    }
}

impl From<Vec<(String, Literal)>> for Literal {
    fn from(fields: Vec<(String, Literal)>) -> Self {
        let boxed_fields = fields
            .into_iter()
            .map(|(name, lit)| (name, Box::new(lit)))
            .collect();
        Literal::Struct(boxed_fields)
    }
}
