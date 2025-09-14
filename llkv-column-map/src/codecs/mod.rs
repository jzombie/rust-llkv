use crate::types::{TypedKind, TypedValue};
use std::io;
pub mod big_endian;

pub fn encode_typed(v: &TypedValue) -> Vec<u8> {
    match v {
        TypedValue::Bootstrap(x) => bitcode::encode(x),
        TypedValue::Manifest(x) => bitcode::encode(x),
        TypedValue::ColumnIndex(x) => bitcode::encode(x),
        TypedValue::IndexSegment(x) => bitcode::encode(x),
        TypedValue::ColumnarRegistry(x) => bitcode::encode(x),
        TypedValue::ColumnarDescriptor(x) => bitcode::encode(x),
    }
}

pub fn decode_typed(kind: TypedKind, bytes: &[u8]) -> io::Result<TypedValue> {
    use std::io::{Error, ErrorKind};

    match kind {
        TypedKind::Bootstrap => {
            let v: crate::column_index::Bootstrap =
                bitcode::decode(bytes).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            Ok(TypedValue::Bootstrap(v))
        }
        TypedKind::Manifest => {
            let v: crate::column_index::Manifest =
                bitcode::decode(bytes).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            Ok(TypedValue::Manifest(v))
        }
        TypedKind::ColumnIndex => {
            let v: crate::column_index::ColumnIndex =
                bitcode::decode(bytes).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            Ok(TypedValue::ColumnIndex(v))
        }
        TypedKind::IndexSegment => {
            let v: crate::column_index::IndexSegment =
                bitcode::decode(bytes).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            Ok(TypedValue::IndexSegment(v))
        }
        TypedKind::ColumnarRegistry => {
            let v: crate::column_store::ColumnarRegistry =
                bitcode::decode(bytes).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            Ok(TypedValue::ColumnarRegistry(v))
        }
        TypedKind::ColumnarDescriptor => {
            let v: crate::column_store::ColumnarDescriptor =
                bitcode::decode(bytes).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            Ok(TypedValue::ColumnarDescriptor(v))
        }
    }
}
