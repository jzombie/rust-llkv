use arrow::array::ArrayRef;
use arrow::datatypes::{Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use llkv_result::{Error, Result};
use std::io::Cursor;
use std::sync::Arc;

pub fn serialize_array(array: &dyn arrow::array::Array) -> Result<Vec<u8>> {
    let array_ref = arrow::array::make_array(array.to_data());
    let schema = Arc::new(Schema::new(vec![Field::new(
        "col",
        array.data_type().clone(),
        true,
    )]));
    let batch = RecordBatch::try_new(schema, vec![array_ref])?;

    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &batch.schema())?;
        writer.write(&batch)?;
        writer.finish()?;
    }
    Ok(buffer)
}

pub fn deserialize_array(bytes: impl AsRef<[u8]>) -> Result<ArrayRef> {
    let cursor = Cursor::new(bytes.as_ref());
    let mut reader = StreamReader::try_new(cursor, None)?;

    if let Some(batch) = reader.next() {
        let batch = batch?;
        if batch.num_columns() > 0 {
            Ok(batch.column(0).clone())
        } else {
            Err(Error::Arrow(
                arrow::error::ArrowError::InvalidArgumentError("Empty batch".to_string()),
            ))
        }
    } else {
        Err(Error::Arrow(
            arrow::error::ArrowError::InvalidArgumentError("No batch found".to_string()),
        ))
    }
}
