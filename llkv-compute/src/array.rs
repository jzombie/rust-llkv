use arrow::array::{Array, ArrayRef, Decimal128Array, Float64Array, Int64Array, StringArray};
use arrow::datatypes::DataType;
use llkv_expr::DecimalValue;
use llkv_result::Error;
use std::sync::Arc;

use crate::{NumericKind, NumericValue};

/// Wraps an Arrow array that stores numeric values alongside its numeric kind.
#[derive(Clone, Debug)]
pub struct NumericArray {
    kind: NumericKind,
    len: usize,
    int_data: Option<Arc<Int64Array>>,
    float_data: Option<Arc<Float64Array>>,
    decimal_data: Option<Arc<Decimal128Array>>,
    string_data: Option<Arc<StringArray>>,
}

impl NumericArray {
    pub fn new_int(array: Arc<Int64Array>) -> Self {
        let len = array.len();
        Self {
            kind: NumericKind::Integer,
            len,
            int_data: Some(array),
            float_data: None,
            decimal_data: None,
            string_data: None,
        }
    }

    pub fn new_float(array: Arc<Float64Array>) -> Self {
        let len = array.len();
        Self {
            kind: NumericKind::Float,
            len,
            int_data: None,
            float_data: Some(array),
            decimal_data: None,
            string_data: None,
        }
    }

    pub fn new_decimal(array: Arc<Decimal128Array>) -> Self {
        let len = array.len();
        Self {
            kind: NumericKind::Decimal,
            len,
            int_data: None,
            float_data: None,
            decimal_data: Some(array),
            string_data: None,
        }
    }

    pub fn new_string(array: Arc<StringArray>) -> Self {
        let len = array.len();
        Self {
            kind: NumericKind::String,
            len,
            int_data: None,
            float_data: None,
            decimal_data: None,
            string_data: Some(array),
        }
    }

    pub fn try_from_arrow(array: &ArrayRef) -> Result<Self, Error> {
        match array.data_type() {
            DataType::Utf8 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| Error::Internal("expected String array".into()))?
                    .clone();
                Ok(Self::new_string(Arc::new(typed)))
            }
            DataType::Int64 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| Error::Internal("expected Int64 array".into()))?
                    .clone();
                Ok(Self::new_int(Arc::new(typed)))
            }
            DataType::Float64 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| Error::Internal("expected Float64 array".into()))?
                    .clone();
                Ok(Self::new_float(Arc::new(typed)))
            }
            DataType::Decimal128(_, _) => {
                let typed = array
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| Error::Internal("expected Decimal128 array".into()))?
                    .clone();
                Ok(Self::new_decimal(Arc::new(typed)))
            }
            // Coercions
            DataType::Int32 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .ok_or_else(|| Error::Internal("expected Int32 array".into()))?;
                let casted = arrow::compute::cast(typed, &DataType::Int64)?;
                let int_array = casted
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| Error::Internal("cast failed".into()))?
                    .clone();
                Ok(Self::new_int(Arc::new(int_array)))
            }
            DataType::Float32 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float32Array>()
                    .ok_or_else(|| Error::Internal("expected Float32 array".into()))?;
                let casted = arrow::compute::cast(typed, &DataType::Float64)?;
                let float_array = casted
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| Error::Internal("cast failed".into()))?
                    .clone();
                Ok(Self::new_float(Arc::new(float_array)))
            }
            DataType::Date32 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<arrow::array::Date32Array>()
                    .ok_or_else(|| Error::Internal("expected Date32 array".into()))?;
                // Date32 is essentially i32 days since epoch. Treat as Int64 for numeric ops.
                let casted = arrow::compute::cast(typed, &DataType::Int64)?;
                let int_array = casted
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| Error::Internal("cast failed".into()))?
                    .clone();
                Ok(Self::new_int(Arc::new(int_array)))
            }
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported numeric array type {:?}",
                other
            ))),
        }
    }

    pub fn to_array_ref(&self) -> ArrayRef {
        match self.kind {
            NumericKind::Integer => Arc::new(self.int_data.as_ref().unwrap().as_ref().clone()),
            NumericKind::Float => Arc::new(self.float_data.as_ref().unwrap().as_ref().clone()),
            NumericKind::Decimal => Arc::new(self.decimal_data.as_ref().unwrap().as_ref().clone()),
            NumericKind::String => Arc::new(self.string_data.as_ref().unwrap().as_ref().clone()),
        }
    }

    pub fn kind(&self) -> NumericKind {
        self.kind
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn int_data(&self) -> Option<&Arc<Int64Array>> {
        self.int_data.as_ref()
    }

    pub fn float_data(&self) -> Option<&Arc<Float64Array>> {
        self.float_data.as_ref()
    }

    pub fn decimal_data(&self) -> Option<&Arc<Decimal128Array>> {
        self.decimal_data.as_ref()
    }

    pub fn string_data(&self) -> Option<&Arc<StringArray>> {
        self.string_data.as_ref()
    }

    pub fn value(&self, idx: usize) -> Option<NumericValue> {
        match self.kind {
            NumericKind::Integer => {
                let array = self
                    .int_data
                    .as_ref()
                    .expect("integer array missing backing data");
                if array.is_null(idx) {
                    None
                } else {
                    Some(NumericValue::Int(array.value(idx)))
                }
            }
            NumericKind::Float => {
                let array = self
                    .float_data
                    .as_ref()
                    .expect("float array missing backing data");
                if array.is_null(idx) {
                    None
                } else {
                    Some(NumericValue::Float(array.value(idx)))
                }
            }
            NumericKind::Decimal => {
                let array = self
                    .decimal_data
                    .as_ref()
                    .expect("decimal array missing backing data");
                if array.is_null(idx) {
                    None
                } else {
                    Some(NumericValue::Decimal(
                        DecimalValue::new(array.value(idx), array.scale())
                            .expect("invalid decimal in array"),
                    ))
                }
            }
            NumericKind::String => {
                let array = self
                    .string_data
                    .as_ref()
                    .expect("string array missing backing data");
                if array.is_null(idx) {
                    None
                } else {
                    Some(NumericValue::String(array.value(idx).to_string()))
                }
            }
        }
    }

    pub fn from_numeric_values(values: Vec<Option<NumericValue>>, preferred: NumericKind) -> Self {
        let contains_string = values
            .iter()
            .any(|opt| matches!(opt, Some(NumericValue::String(_))));

        if contains_string || preferred == NumericKind::String {
            let iter = values.into_iter().map(|opt| {
                opt.map(|v| match v {
                    NumericValue::String(s) => s,
                    NumericValue::Int(i) => i.to_string(),
                    NumericValue::Float(f) => f.to_string(),
                    NumericValue::Decimal(d) => d.to_string(),
                })
            });
            let array = StringArray::from_iter(iter);
            return NumericArray::new_string(Arc::new(array));
        }

        let contains_float = values
            .iter()
            .any(|opt| matches!(opt, Some(NumericValue::Float(_))));
        let contains_decimal = values
            .iter()
            .any(|opt| matches!(opt, Some(NumericValue::Decimal(_))));

        match (contains_float, contains_decimal, preferred) {
            // If any float, convert all to float
            (true, _, _) => {
                let iter = values.into_iter().map(|opt| opt.map(|v| v.to_f64()));
                let array = Float64Array::from_iter(iter);
                NumericArray::new_float(Arc::new(array))
            }
            // If decimals but no floats
            (false, true, NumericKind::Float) => {
                let iter = values.into_iter().map(|opt| opt.map(|v| v.to_f64()));
                let array = Float64Array::from_iter(iter);
                NumericArray::new_float(Arc::new(array))
            }
            (false, true, _) => {
                // Convert to Decimal128Array
                let mut max_scale = 0;
                let mut max_precision = 0;

                for val in &values {
                    if let Some(NumericValue::Decimal(d)) = val {
                        max_scale = max_scale.max(d.scale());
                        max_precision = max_precision.max(d.precision());
                    }
                }

                // Default to something reasonable if empty or only nulls
                if max_precision == 0 {
                    max_precision = 38;
                    max_scale = 10;
                }

                let mut builder = arrow::array::Decimal128Builder::with_capacity(values.len());

                for val in values {
                    match val {
                        Some(NumericValue::Decimal(d)) => {
                            if let Ok(rescaled) = crate::scalar::decimal::rescale(d, max_scale) {
                                builder.append_value(rescaled.raw_value());
                            } else {
                                builder.append_null();
                            }
                        }
                        Some(NumericValue::Int(i)) => {
                            let d = DecimalValue::from_i64(i);
                            if let Ok(rescaled) = crate::scalar::decimal::rescale(d, max_scale) {
                                builder.append_value(rescaled.raw_value());
                            } else {
                                builder.append_null();
                            }
                        }
                        None => builder.append_null(),
                        _ => builder.append_null(),
                    }
                }

                let array = builder
                    .finish()
                    .with_precision_and_scale(max_precision, max_scale)
                    .unwrap_or_else(|_| {
                        // Fallback if precision/scale invalid
                        arrow::array::Decimal128Array::from(vec![None::<i128>; 0])
                    });
                NumericArray::new_decimal(Arc::new(array))
            }
            // Pure integers
            (false, false, NumericKind::Integer) => {
                let iter = values.into_iter().map(|opt| {
                    opt.map(|v| match v {
                        NumericValue::Int(i) => i,
                        _ => panic!("expected integer"),
                    })
                });
                let array = Int64Array::from_iter(iter);
                NumericArray::new_int(Arc::new(array))
            }
            // Pure integers but preferred float
            (false, false, NumericKind::Float) => {
                let iter = values.into_iter().map(|opt| opt.map(|v| v.to_f64()));
                let array = Float64Array::from_iter(iter);
                NumericArray::new_float(Arc::new(array))
            }
            // Pure integers but preferred decimal
            (false, false, NumericKind::Decimal) => {
                let max_scale = 0;
                let max_precision = 19; // i64 max digits
                let mut builder = arrow::array::Decimal128Builder::with_capacity(values.len());
                for val in values {
                    match val {
                        Some(NumericValue::Int(i)) => builder.append_value(i as i128),
                        None => builder.append_null(),
                        _ => builder.append_null(),
                    }
                }
                let array = builder
                    .finish()
                    .with_precision_and_scale(max_precision, max_scale)
                    .unwrap();
                NumericArray::new_decimal(Arc::new(array))
            }
            // Fallback for unexpected cases (shouldn't happen with current logic)
            _ => {
                // Default to float if we can't decide
                let iter = values.into_iter().map(|opt| opt.map(|v| v.to_f64()));
                let array = Float64Array::from_iter(iter);
                NumericArray::new_float(Arc::new(array))
            }
        }
    }

    pub fn promote_to_float(&self) -> NumericArray {
        match self.kind {
            NumericKind::Float => self.clone(),
            NumericKind::Integer => {
                let array = self
                    .int_data
                    .as_ref()
                    .expect("integer array missing backing data");
                let iter = (0..self.len).map(|idx| {
                    if array.is_null(idx) {
                        None
                    } else {
                        Some(array.value(idx) as f64)
                    }
                });
                let float_array = Float64Array::from_iter(iter);
                NumericArray::new_float(Arc::new(float_array))
            }
            NumericKind::Decimal => {
                let array = self
                    .decimal_data
                    .as_ref()
                    .expect("decimal array missing backing data");
                let iter = (0..self.len).map(|idx| {
                    if array.is_null(idx) {
                        None
                    } else {
                        let value_i128 = array.value(idx);
                        let scale = array.scale();
                        let decimal = DecimalValue::new(value_i128, scale)
                            .expect("valid decimal from Decimal128Array");
                        Some(decimal.to_f64())
                    }
                });
                let float_array = Float64Array::from_iter(iter);
                NumericArray::new_float(Arc::new(float_array))
            }
            NumericKind::String => {
                let array = self
                    .string_data
                    .as_ref()
                    .expect("string array missing backing data");
                let iter = (0..self.len).map(|idx| {
                    if array.is_null(idx) {
                        None
                    } else {
                        array.value(idx).parse::<f64>().ok()
                    }
                });
                let float_array = Float64Array::from_iter(iter);
                NumericArray::new_float(Arc::new(float_array))
            }
        }
    }

    pub fn to_aligned_array_ref(&self, preferred: NumericKind) -> ArrayRef {
        match (preferred, self.kind) {
            (NumericKind::Float, NumericKind::Integer) => self.promote_to_float().to_array_ref(),
            (NumericKind::Float, NumericKind::Decimal) => self.promote_to_float().to_array_ref(),
            _ => self.to_array_ref(),
        }
    }
}
