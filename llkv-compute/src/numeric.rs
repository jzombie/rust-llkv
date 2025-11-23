use llkv_expr::DecimalValue;
use llkv_result::Error;

/// Describes whether a numeric value is represented as an integer, float, or decimal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericKind {
    Integer,
    Float,
    Decimal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumericValue {
    Int(i64),
    Float(f64),
    Decimal(DecimalValue),
}

impl NumericValue {
    pub fn kind(&self) -> NumericKind {
        match self {
            NumericValue::Int(_) => NumericKind::Integer,
            NumericValue::Float(_) => NumericKind::Float,
            NumericValue::Decimal(_) => NumericKind::Decimal,
        }
    }

    pub fn as_f64(&self) -> f64 {
        self.to_f64()
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            NumericValue::Int(v) => *v as f64,
            NumericValue::Float(v) => *v,
            NumericValue::Decimal(v) => v.to_f64(),
        }
    }

    pub fn add(&self, other: &NumericValue) -> Result<NumericValue, Error> {
        match (self, other) {
            (NumericValue::Int(a), NumericValue::Int(b)) => {
                // Check for overflow?
                a.checked_add(*b)
                    .map(NumericValue::Int)
                    .ok_or_else(|| Error::Internal("Integer overflow in sum".to_string()))
            }
            (NumericValue::Float(a), NumericValue::Float(b)) => Ok(NumericValue::Float(a + b)),
            (NumericValue::Decimal(a), NumericValue::Decimal(b)) => a
                .checked_add(*b)
                .map(NumericValue::Decimal)
                .map_err(|e| Error::Internal(format!("Decimal error: {}", e))),
            // Mixed types - promote to Decimal if one is Decimal and other is Int
            (NumericValue::Int(a), NumericValue::Decimal(b)) => {
                let a_dec = DecimalValue::from_i64(*a);
                a_dec
                    .checked_add(*b)
                    .map(NumericValue::Decimal)
                    .map_err(|e| Error::Internal(format!("Decimal error: {}", e)))
            }
            (NumericValue::Decimal(a), NumericValue::Int(b)) => {
                let b_dec = DecimalValue::from_i64(*b);
                a.checked_add(b_dec)
                    .map(NumericValue::Decimal)
                    .map_err(|e| Error::Internal(format!("Decimal error: {}", e)))
            }
            // Float dominates
            (a, b) => Ok(NumericValue::Float(a.to_f64() + b.to_f64())),
        }
    }

    pub fn sub(&self, other: &NumericValue) -> Result<NumericValue, Error> {
        match (self, other) {
            (NumericValue::Int(a), NumericValue::Int(b)) => a
                .checked_sub(*b)
                .map(NumericValue::Int)
                .ok_or_else(|| Error::Internal("Integer overflow in sub".to_string())),
            (NumericValue::Float(a), NumericValue::Float(b)) => Ok(NumericValue::Float(a - b)),
            (NumericValue::Decimal(a), NumericValue::Decimal(b)) => a
                .checked_sub(*b)
                .map(NumericValue::Decimal)
                .map_err(|e| Error::Internal(format!("Decimal error: {}", e))),
            (NumericValue::Int(a), NumericValue::Decimal(b)) => {
                let a_dec = DecimalValue::from_i64(*a);
                a_dec
                    .checked_sub(*b)
                    .map(NumericValue::Decimal)
                    .map_err(|e| Error::Internal(format!("Decimal error: {}", e)))
            }
            (NumericValue::Decimal(a), NumericValue::Int(b)) => {
                let b_dec = DecimalValue::from_i64(*b);
                a.checked_sub(b_dec)
                    .map(NumericValue::Decimal)
                    .map_err(|e| Error::Internal(format!("Decimal error: {}", e)))
            }
            (a, b) => Ok(NumericValue::Float(a.to_f64() - b.to_f64())),
        }
    }

    pub fn mul(&self, other: &NumericValue) -> Result<NumericValue, Error> {
        match (self, other) {
            (NumericValue::Int(a), NumericValue::Int(b)) => a
                .checked_mul(*b)
                .map(NumericValue::Int)
                .ok_or_else(|| Error::Internal("Integer overflow in mul".to_string())),
            (NumericValue::Float(a), NumericValue::Float(b)) => Ok(NumericValue::Float(a * b)),
            (NumericValue::Decimal(a), NumericValue::Decimal(b)) => a
                .checked_mul(*b)
                .map(NumericValue::Decimal)
                .map_err(|e| Error::Internal(format!("Decimal error: {}", e))),
            (NumericValue::Int(a), NumericValue::Decimal(b)) => {
                let a_dec = DecimalValue::from_i64(*a);
                a_dec
                    .checked_mul(*b)
                    .map(NumericValue::Decimal)
                    .map_err(|e| Error::Internal(format!("Decimal error: {}", e)))
            }
            (NumericValue::Decimal(a), NumericValue::Int(b)) => {
                let b_dec = DecimalValue::from_i64(*b);
                a.checked_mul(b_dec)
                    .map(NumericValue::Decimal)
                    .map_err(|e| Error::Internal(format!("Decimal error: {}", e)))
            }
            (a, b) => Ok(NumericValue::Float(a.to_f64() * b.to_f64())),
        }
    }

    pub fn div(&self, other: &NumericValue) -> Result<NumericValue, Error> {
        match (self, other) {
            (NumericValue::Int(a), NumericValue::Int(b)) => {
                if *b == 0 {
                    return Err(Error::Internal("Division by zero".to_string()));
                }
                Ok(NumericValue::Int(a / b))
            }
            (NumericValue::Float(a), NumericValue::Float(b)) => Ok(NumericValue::Float(a / b)),
            // For Decimal division, we need a target scale.
            // Since we don't have schema info here, we'll default to a reasonable scale
            // or fall back to float if we can't decide.
            // However, for TPC-H Q1, we need Decimal precision.
            // Let's try to use a default scale of 4 (common in TPC-H) or max of inputs + 4.
            (NumericValue::Decimal(a), NumericValue::Decimal(b)) => {
                // Heuristic: target scale = max(s1, s2) + 4, capped at 38?
                // Or just use a fixed high scale?
                // Let's use max(a.scale, b.scale) + 4 for now.
                let target_scale = (a.scale().max(b.scale()) + 4).min(38); // Assuming 38 is max
                a.checked_div(*b, target_scale)
                    .map(NumericValue::Decimal)
                    .map_err(|e| Error::Internal(format!("Decimal error: {}", e)))
            }
            (NumericValue::Int(a), NumericValue::Decimal(b)) => {
                let a_dec = DecimalValue::from_i64(*a);
                let target_scale = (b.scale() + 4).min(38);
                a_dec
                    .checked_div(*b, target_scale)
                    .map(NumericValue::Decimal)
                    .map_err(|e| Error::Internal(format!("Decimal error: {}", e)))
            }
            (NumericValue::Decimal(a), NumericValue::Int(b)) => {
                let b_dec = DecimalValue::from_i64(*b);
                let target_scale = (a.scale() + 4).min(38);
                a.checked_div(b_dec, target_scale)
                    .map(NumericValue::Decimal)
                    .map_err(|e| Error::Internal(format!("Decimal error: {}", e)))
            }
            (a, b) => Ok(NumericValue::Float(a.to_f64() / b.to_f64())),
        }
    }

    pub fn rem(&self, other: &NumericValue) -> Result<NumericValue, Error> {
        match (self, other) {
            (NumericValue::Int(a), NumericValue::Int(b)) => {
                if *b == 0 {
                    return Err(Error::Internal("Division by zero".to_string()));
                }
                Ok(NumericValue::Int(a % b))
            }
            (NumericValue::Float(a), NumericValue::Float(b)) => Ok(NumericValue::Float(a % b)),
            // Decimal rem not explicitly in snippet, but maybe available?
            // If not, fallback to float.
            (a, b) => Ok(NumericValue::Float(a.to_f64() % b.to_f64())),
        }
    }
}

impl From<i64> for NumericValue {
    fn from(v: i64) -> Self {
        NumericValue::Int(v)
    }
}

impl From<f64> for NumericValue {
    fn from(v: f64) -> Self {
        NumericValue::Float(v)
    }
}

impl From<DecimalValue> for NumericValue {
    fn from(v: DecimalValue) -> Self {
        NumericValue::Decimal(v)
    }
}
