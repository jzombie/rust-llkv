use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::ROW_ID_COLUMN_NAME;
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use llkv_table::{Table, types::FieldId};

use crate::{CsvReadOptions, open_csv_reader};

fn convert_row_id(array: &ArrayRef) -> LlkvResult<ArrayRef> {
    match array.data_type() {
        DataType::UInt64 => Ok(Arc::clone(array)),
        DataType::Int64 => {
            let int_array = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| Error::InvalidArgumentError("row_id column is not Int64".into()))?;

            if int_array.null_count() > 0 {
                return Err(Error::InvalidArgumentError(
                    "row_id column cannot contain nulls".into(),
                ));
            }

            let mut builder = UInt64Builder::with_capacity(int_array.len());
            for i in 0..int_array.len() {
                let value = int_array.value(i);
                if value < 0 {
                    return Err(Error::InvalidArgumentError(
                        "row_id column must contain non-negative values".into(),
                    ));
                }
                builder.append_value(value as u64);
            }

            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        other => Err(Error::InvalidArgumentError(format!(
            "row_id column must be Int64 or UInt64, got {other:?}"
        ))),
    }
}

fn ensure_supported_type(data_type: &DataType, column: &str) -> LlkvResult<()> {
    llkv_column_map::ensure_supported_arrow_type(data_type).map_err(|err| match err {
        Error::InvalidArgumentError(msg) => Error::InvalidArgumentError(format!(
            "column '{column}': {msg}"
        )),
        other => other,
    })
}

fn build_schema_with_metadata(
    schema: &Schema,
    field_mapping: &HashMap<String, FieldId>,
) -> LlkvResult<(Arc<Schema>, usize)> {
    let row_id_index = schema
        .fields()
        .iter()
        .position(|f| f.name() == ROW_ID_COLUMN_NAME)
        .ok_or_else(|| {
            Error::InvalidArgumentError(format!(
                "CSV schema must include a '{ROW_ID_COLUMN_NAME}' column"
            ))
        })?;

    let mut fields_with_metadata = Vec::with_capacity(schema.fields().len());
    for (idx, field) in schema.fields().iter().enumerate() {
        if idx == row_id_index {
            fields_with_metadata.push(Field::new(
                ROW_ID_COLUMN_NAME,
                DataType::UInt64,
                field.is_nullable(),
            ));
            continue;
        }

        ensure_supported_type(field.data_type(), field.name())?;

        let field_id = field_mapping.get(field.name()).ok_or_else(|| {
            Error::InvalidArgumentError(format!(
                "no field_id mapping provided for column '{}'",
                field.name()
            ))
        })?;

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("field_id".to_string(), field_id.to_string());

        fields_with_metadata.push(
            Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                .with_metadata(metadata),
        );
    }

    Ok((Arc::new(Schema::new(fields_with_metadata)), row_id_index))
}

pub fn append_csv_into_table<P, C>(
    table: &Table<P>,
    csv_path: C,
    field_mapping: &HashMap<String, FieldId>,
    csv_options: &CsvReadOptions,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    C: AsRef<Path>,
{
    let csv_path_ref = csv_path.as_ref();
    let (schema, mut reader) = open_csv_reader(csv_path_ref, csv_options)
        .map_err(|err| Error::Internal(format!("failed to open CSV: {err}")))?;

    let (schema_with_metadata, row_id_index) = build_schema_with_metadata(&schema, field_mapping)?;

    while let Some(batch_result) = reader.next() {
        let batch = batch_result
            .map_err(|err| Error::Internal(format!("failed to read CSV batch: {err}")))?;

        if batch.num_rows() == 0 {
            continue;
        }

        let mut columns: Vec<ArrayRef> = batch.columns().iter().cloned().collect();
        let row_id_array = convert_row_id(&columns[row_id_index])?;
        columns[row_id_index] = row_id_array;

        let new_batch = RecordBatch::try_new(Arc::clone(&schema_with_metadata), columns)?;
        table.append(&new_batch)?;
    }

    Ok(())
}
