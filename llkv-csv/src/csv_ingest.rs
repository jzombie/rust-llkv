use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::ROW_ID_COLUMN_NAME;
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use llkv_table::{ColMeta, Table, types::FieldId};

use crate::{CsvReadOptions, open_csv_reader};

// TODO: Migrate to common type utils
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
        Error::InvalidArgumentError(msg) => {
            Error::InvalidArgumentError(format!("column '{column}': {msg}"))
        }
        other => other,
    })
}

fn existing_column_mapping<P>(table: &Table<P>) -> HashMap<String, FieldId>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let logical_fields = table.store().user_field_ids_for_table(table.table_id());
    if logical_fields.is_empty() {
        return HashMap::new();
    }

    let mut field_ids: Vec<FieldId> = Vec::new();
    for lfid in logical_fields {
        let fid = lfid.field_id();
        if fid != 0 {
            field_ids.push(fid);
        }
    }

    if field_ids.is_empty() {
        return HashMap::new();
    }

    let metas = table.catalog().get_cols_meta(table.table_id(), &field_ids);
    let mut mapping = HashMap::with_capacity(metas.len());
    for (fid, meta_opt) in field_ids.into_iter().zip(metas.into_iter()) {
        if let Some(meta) = meta_opt
            && let Some(name) = meta.name
        {
            mapping.insert(name, fid);
        }
    }
    mapping
}

fn infer_field_mapping<'a, P>(
    table: &Table<P>,
    schema: &'a Schema,
    provided: Option<&'a HashMap<String, FieldId>>,
) -> LlkvResult<HashMap<String, FieldId>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut mapping = HashMap::new();
    let mut existing = existing_column_mapping(table);
    let mut used_ids: HashSet<FieldId> = existing.values().copied().collect();
    let mut next_field_id: FieldId = existing.values().copied().max().unwrap_or(0);

    for field in schema.fields() {
        if field.name() == ROW_ID_COLUMN_NAME {
            continue;
        }

        ensure_supported_type(field.data_type(), field.name())?;

        let mut chosen: Option<FieldId> = None;
        let mut should_register_meta = false;

        if let Some(manual) = provided
            && let Some(&fid) = manual.get(field.name())
        {
            if let Some(&existing_fid) = existing.get(field.name()) {
                if existing_fid != fid {
                    return Err(Error::InvalidArgumentError(format!(
                        "column '{}' mapped to field_id {} but existing schema expects {}",
                        field.name(),
                        fid,
                        existing_fid
                    )));
                }
            } else {
                should_register_meta = true;
            }
            chosen = Some(fid);
        }

        if chosen.is_none()
            && let Some(&fid) = existing.get(field.name())
        {
            chosen = Some(fid);
        }

        if chosen.is_none() {
            next_field_id = next_field_id
                .checked_add(1)
                .ok_or_else(|| Error::Internal("field_id overflow when inferring schema".into()))?;
            let fid = next_field_id;
            should_register_meta = true;
            chosen = Some(fid);
        }

        let fid = chosen.unwrap();
        if should_register_meta {
            let meta = ColMeta {
                col_id: fid,
                name: Some(field.name().to_string()),
                flags: 0,
                default: None,
            };
            table.put_col_meta(&meta);
            existing.insert(field.name().to_string(), fid);
        }
        if fid == 0 {
            return Err(Error::InvalidArgumentError(format!(
                "column '{}' cannot map to reserved field_id 0",
                field.name()
            )));
        }
        if !used_ids.insert(fid) {
            return Err(Error::InvalidArgumentError(format!(
                "field_id {} assigned to multiple columns during schema inference",
                fid
            )));
        }

        mapping.insert(field.name().to_string(), fid);
    }

    Ok(mapping)
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

fn append_csv_into_table_internal<P, C>(
    table: &Table<P>,
    csv_path: C,
    csv_options: &CsvReadOptions,
    field_mapping_override: Option<&HashMap<String, FieldId>>,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    C: AsRef<Path>,
{
    let csv_path_ref = csv_path.as_ref();
    let (schema, reader) = open_csv_reader(csv_path_ref, csv_options)
        .map_err(|err| Error::Internal(format!("failed to open CSV: {err}")))?;

    let inferred_mapping = infer_field_mapping(table, schema.as_ref(), field_mapping_override)?;
    let (schema_with_metadata, row_id_index) =
        build_schema_with_metadata(&schema, &inferred_mapping)?;

    for batch_result in reader {
        let batch = batch_result
            .map_err(|err| Error::Internal(format!("failed to read CSV batch: {err}")))?;

        if batch.num_rows() == 0 {
            continue;
        }

        let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
        let row_id_array = convert_row_id(&columns[row_id_index])?;
        columns[row_id_index] = row_id_array;

        let new_batch = RecordBatch::try_new(Arc::clone(&schema_with_metadata), columns)?;
        table.append(&new_batch)?;
    }

    // Defensive: ensure the catalog contains ColMeta.name for each column we
    // just inferred. In some code paths the field id metadata can be present
    // on appended batches without a corresponding ColMeta entry in the
    // catalog; make sure we persist the CSV header names so `Table::schema()`
    // returns friendly column names.
    for (col_name, fid) in inferred_mapping.iter() {
        let metas = table.get_cols_meta(&[*fid]);
        let need_put = match metas.get(0) {
            Some(Some(meta)) => meta.name.is_none(),
            _ => true,
        };
        if need_put {
            let meta = ColMeta {
                col_id: *fid,
                name: Some(col_name.clone()),
                flags: 0,
                default: None,
            };
            table.put_col_meta(&meta);
        }
    }

    Ok(())
}

pub fn append_csv_into_table<P, C>(
    table: &Table<P>,
    csv_path: C,
    csv_options: &CsvReadOptions,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    C: AsRef<Path>,
{
    append_csv_into_table_internal(table, csv_path, csv_options, None)
}

pub fn append_csv_into_table_with_mapping<P, C>(
    table: &Table<P>,
    csv_path: C,
    field_mapping: &HashMap<String, FieldId>,
    csv_options: &CsvReadOptions,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    C: AsRef<Path>,
{
    append_csv_into_table_internal(table, csv_path, csv_options, Some(field_mapping))
}
