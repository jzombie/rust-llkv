//! Minimal-allocation Arrow IPC (single array) helpers.
//!
//! Preserves the public API and avoids repeated small Vec growth by
//! reserving approximately the needed space up-front.

use crate::error::{Error, Result};
use arrow::array::{Array, ArrayRef, make_array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use std::io::Cursor;
use std::sync::Arc;

/// Rough upper bound for an array's IPC size, in bytes.
/// This is conservative and avoids repeated reallocations.
#[inline]
fn estimate_ipc_size(array: &dyn Array) -> usize {
    let len = array.len();
    let validity = (len + 7) / 8;

    match array.data_type() {
        // Fixed-width primitives
        DataType::UInt8 | DataType::Int8 => validity + len * 1 + 128,
        DataType::UInt16 | DataType::Int16 => validity + len * 2 + 128,
        DataType::UInt32 | DataType::Int32 | DataType::Float32 => validity + len * 4 + 128,
        DataType::UInt64 | DataType::Int64 | DataType::Float64 => validity + len * 8 + 128,

        // Binary/Utf8: offsets (len+1)*4 + values ~ avg 8 bytes
        DataType::Binary | DataType::Utf8 => validity + (len + 1) * 4 + len * 8 + 256,

        // Large variants
        DataType::LargeBinary | DataType::LargeUtf8 => validity + (len + 1) * 8 + len * 8 + 256,

        // Fallback: reserve something sane
        _ => validity + len * 8 + 256,
    }
}

/// Serializes a single Arrow array into a byte vector.
/// Keeps the same signature as before.
pub fn serialize_array(array: &dyn Array) -> Result<Vec<u8>> {
    let schema = Schema::new(vec![Field::new(
        "item",
        array.data_type().clone(),
        array.is_nullable(),
    )]);

    // Pre-size the sink to reduce growth churn.
    let cap = estimate_ipc_size(array);
    let sink = Vec::with_capacity(cap);

    let array_ref = make_array(array.to_data());
    let batch = RecordBatch::try_new(Arc::new(schema), vec![array_ref])?;

    let mut writer = StreamWriter::try_new(sink, &batch.schema())?;
    writer.write(&batch)?;
    writer.finish()?;

    // Return the owned Vec produced by the writer.
    Ok(writer.into_inner()?)
}

/// Deserializes a byte slice back into an Arrow array.
/// Keeps the same signature as before.
pub fn deserialize_array(bytes: &[u8]) -> Result<ArrayRef> {
    // StreamReader wants a Read; Cursor<&[u8]> is a zero-copy wrapper.
    let mut reader = StreamReader::try_new(Cursor::new(bytes), None)?;
    let batch = reader
        .next()
        .transpose()?
        .ok_or_else(|| Error::Internal("empty IPC stream".into()))?;

    if batch.num_columns() != 1 {
        return Err(Error::Internal("expected single-column batch".to_string()));
    }
    Ok(batch.column(0).clone())
}
