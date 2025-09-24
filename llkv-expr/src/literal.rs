/// A literal value that has not yet been coerced into a specific native
/// type. This allows for type inference to be deferred until the column
/// type is known.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i128),
    Float(f64),
    String(String),
    // Other types like Bool, Bytes can be added here.
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

impl From<&str> for Literal {
    fn from(v: &str) -> Self {
        Literal::String(v.to_string())
    }
}

/// Error converting a `Literal` into a concrete native type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiteralCastError {
    /// Tried to coerce a non-integer literal into an integer native type.
    TypeMismatch {
        expected: &'static str,
        got: &'static str,
    },
    /// Integer value does not fit in the destination type.
    OutOfRange { target: &'static str, value: i128 },
}

impl std::fmt::Display for LiteralCastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiteralCastError::TypeMismatch { expected, got } => {
                write!(f, "expected {}, got {}", expected, got)
            }
            LiteralCastError::OutOfRange { target, value } => {
                write!(f, "value {} out of range for {}", value, target)
            }
        }
    }
}

impl std::error::Error for LiteralCastError {}

/// Convert a `Literal` into a concrete integer-like native type `T`.
/// This crate does not depend on Arrow; it only requires `TryFrom<i128>`.
///
/// Note: the current implementation only permits integer-like targets.
/// Calling this with a float/string `Literal` returns `TypeMismatch`.
pub fn literal_to_native<T>(lit: &Literal) -> Result<T, LiteralCastError>
where
    T: TryFrom<i128> + Copy + 'static,
{
    match lit {
        Literal::Integer(i) => T::try_from(*i).map_err(|_| LiteralCastError::OutOfRange {
            target: std::any::type_name::<T>(),
            value: *i,
        }),
        Literal::Float(_) => Err(LiteralCastError::TypeMismatch {
            expected: "integer",
            got: "float",
        }),
        Literal::String(_) => Err(LiteralCastError::TypeMismatch {
            expected: "integer",
            got: "string",
        }),
    }
}
