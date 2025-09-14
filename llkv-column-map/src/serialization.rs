// File: src/serialization.rs
use crate::error::{Error, Result};
use arrow::array::{Array, ArrayRef, make_array};
use arrow::datatypes::Schema;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

/// Serializes a single Arrow array into a byte vector.
pub fn serialize_array(array: &dyn Array) -> Result<Vec<u8>> {
    let schema = Schema::new(vec![arrow::datatypes::Field::new(
        "item",
        array.data_type().clone(),
        array.is_nullable(),
    )]);
    let array_ref = make_array(array.to_data());
    let batch = RecordBatch::try_new(Arc::new(schema), vec![array_ref])?;
    let mut writer = StreamWriter::try_new(Vec::new(), &batch.schema())?;
    writer.write(&batch)?;
    writer.finish()?;
    Ok(writer.into_inner()?)
}

/// Deserializes a byte slice back into an Arrow array.
pub fn deserialize_array(bytes: &[u8]) -> Result<ArrayRef> {
    let mut reader = StreamReader::try_new(bytes, None)?;
    let batch = reader.next().ok_or(Error::Internal(
        "Serialized Arrow stream is empty".to_string(),
    ))??;
    if batch.num_columns() != 1 {
        return Err(Error::Internal(
            "Expected serialized batch to have exactly one column".to_string(),
        ));
    }
    Ok(batch.column(0).clone())
}
