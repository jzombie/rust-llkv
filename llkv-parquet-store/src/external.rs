//! External storage for large blob columns.
//!
//! This module provides a mechanism to store large columnar data (like embeddings)
//! directly in the pager as raw buffers, while keeping only pager key references
//! in the Parquet files. This bypasses Parquet's decode overhead for large blobs.
//!
//! # Design
//!
//! - **Schema annotation**: Fields marked with metadata `"llkv:storage" = "external"`
//!   are stored externally
//! - **Write path**: Extract array buffers → store in pager → replace with FixedSizeBinary(8)
//!   column containing PhysicalKey values
//! - **Read path**: Read key column → batch-fetch buffers from pager → reconstruct arrays
//!
//! # Supported Types
//!
//! Currently supports:
//! - `FixedSizeList`: Common for embeddings/vectors
//! - `Binary`/`LargeBinary`: Variable-length blobs
//!
//! # Example
//!
//! ```rust,no_run
//! use arrow::datatypes::{DataType, Field};
//! use std::collections::HashMap;
//! use std::sync::Arc;
//!
//! // Mark embedding field as external
//! let mut metadata = HashMap::new();
//! metadata.insert("llkv:storage".to_string(), "external".to_string());
//!
//! let embedding_field = Field::new(
//!     "embedding",
//!     DataType::FixedSizeList(
//!         Arc::new(Field::new("item", DataType::Float32, false)),
//!         1024,
//!     ),
//!     false,
//! ).with_metadata(metadata);
//! ```

use arrow::array::{Array, ArrayRef, FixedSizeBinaryArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use llkv_result::{Error, Result};
use llkv_storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use llkv_storage::types::PhysicalKey;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

/// Metadata key used to mark fields for external storage.
pub const EXTERNAL_STORAGE_KEY: &str = "llkv:storage";

/// Metadata value indicating external storage.
pub const EXTERNAL_STORAGE_VALUE: &str = "external";

/// Check if a field is marked for external storage.
pub fn is_external_field(field: &Field) -> bool {
    field
        .metadata()
        .get(EXTERNAL_STORAGE_KEY)
        .map(|v| v == EXTERNAL_STORAGE_VALUE)
        .unwrap_or(false)
}

/// Transform a schema by replacing external fields with FixedSizeBinary(8) key columns.
///
/// The transformed schema is what gets written to Parquet files.
pub fn transform_schema_for_storage(schema: &Schema) -> SchemaRef {
    let fields: Vec<Arc<Field>> = schema
        .fields()
        .iter()
        .map(|field| {
            if is_external_field(field) {
                // Replace with FixedSizeBinary(8) to hold PhysicalKey (u64)
                let mut metadata = field.metadata().clone();
                metadata.insert(
                    "llkv:original_type".to_string(),
                    format!("{:?}", field.data_type()),
                );

                Arc::new(
                    Field::new(
                        field.name(),
                        DataType::FixedSizeBinary(8),
                        field.is_nullable(),
                    )
                    .with_metadata(metadata),
                )
            } else {
                field.clone()
            }
        })
        .collect();

    Arc::new(Schema::new(fields))
}

/// Restore the original schema from a storage schema.
///
/// Converts FixedSizeBinary(8) key columns back to their original types.
pub fn restore_schema_from_storage(storage_schema: &Schema) -> Result<SchemaRef> {
    let fields: Result<Vec<Arc<Field>>> = storage_schema
        .fields()
        .iter()
        .map(|field| {
            if let Some(original_type_str) = field.metadata().get("llkv:original_type") {
                // Parse the original type - this is a simplified approach
                // In production, you'd want more robust type serialization
                // For now, we'll reconstruct based on the debug string
                let data_type = parse_data_type_from_debug(original_type_str)?;

                let mut metadata = field.metadata().clone();
                metadata.remove("llkv:original_type");

                Ok(Arc::new(
                    Field::new(field.name(), data_type, field.is_nullable())
                        .with_metadata(metadata),
                ))
            } else {
                Ok(field.clone())
            }
        })
        .collect();

    Ok(Arc::new(Schema::new(fields?)))
}

/// Parse a DataType from its Debug representation.
///
/// This is a temporary solution - ideally we'd serialize types properly.
/// For now, handles the common cases we care about (FixedSizeList, Binary).
fn parse_data_type_from_debug(debug_str: &str) -> Result<DataType> {
    // Simple pattern matching for common types
    // TODO: Replace with proper serialization/deserialization
    if debug_str.contains("FixedSizeList") {
        // Extract dimension and inner type
        // Format: "FixedSizeList(Field { name: \"item\", data_type: Float32, ... }, 1024)"
        if let Some(size_start) = debug_str.rfind(", ") {
            if let Some(size_end) = debug_str[size_start + 2..].find(')') {
                let size_str = &debug_str[size_start + 2..size_start + 2 + size_end];
                let size: i32 = size_str
                    .parse()
                    .map_err(|e| Error::Internal(format!("failed to parse size: {}", e)))?;

                // For now, assume Float32 inner type
                // TODO: Parse inner type properly
                return Ok(DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    size,
                ));
            }
        }
    } else if debug_str.contains("Binary") {
        return Ok(DataType::Binary);
    } else if debug_str.contains("LargeBinary") {
        return Ok(DataType::LargeBinary);
    }

    Err(Error::Internal(format!(
        "unsupported external storage type: {}",
        debug_str
    )))
}

/// Extract buffers from external columns and store them in the pager.
///
/// Returns a transformed RecordBatch where external columns are replaced with
/// FixedSizeBinary(8) columns containing pager keys.
pub fn externalize_columns<P>(batch: &RecordBatch, pager: &P) -> Result<RecordBatch>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let schema = batch.schema();
    let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());

    for (col_idx, field) in schema.fields().iter().enumerate() {
        let array = batch.column(col_idx);

        if is_external_field(field) {
            // Extract buffer and store in pager
            let key_array = store_array_externally(array, pager)?;
            new_columns.push(key_array);
        } else {
            new_columns.push(array.clone());
        }
    }

    let storage_schema = transform_schema_for_storage(schema.as_ref());
    RecordBatch::try_new(storage_schema, new_columns)
        .map_err(|e| Error::Internal(format!("failed to create externalized batch: {}", e)))
}

/// Store an array's data externally and return a FixedSizeBinary(8) array of keys.
fn store_array_externally<P>(array: &ArrayRef, pager: &P) -> Result<ArrayRef>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let num_rows = array.len();

    // Get the raw buffer data for each row
    // For FixedSizeList, we need to extract each element's buffer
    let buffers = extract_row_buffers(array)?;

    // Allocate pager keys
    let keys = pager.alloc_many(num_rows)?;

    // Store each buffer
    let puts: Vec<BatchPut> = buffers
        .into_iter()
        .zip(keys.iter())
        .map(|(buffer, key)| BatchPut::Raw {
            key: *key,
            bytes: buffer.to_vec(),
        })
        .collect();

    pager.batch_put(&puts)?;

    // Create FixedSizeBinary array containing the keys using builder
    use arrow::array::FixedSizeBinaryBuilder;
    let mut builder = FixedSizeBinaryBuilder::with_capacity(num_rows, 8);

    for key in keys.iter() {
        builder.append_value(&key.to_le_bytes())?;
    }

    let key_array = builder.finish();

    Ok(Arc::new(key_array))
}

/// Extract individual row buffers from an array.
fn extract_row_buffers(array: &ArrayRef) -> Result<Vec<bytes::Bytes>> {
    use arrow::array::{BinaryArray, FixedSizeListArray, LargeBinaryArray};

    match array.data_type() {
        DataType::FixedSizeList(_, size) => {
            let list_array = array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    Error::Internal("failed to downcast to FixedSizeListArray".into())
                })?;

            let values = list_array.values();

            // For FixedSizeList, extract fixed-size chunks
            let buffers: Result<Vec<_>> = (0..array.len())
                .map(|i| {
                    let start = i * (*size as usize);
                    let slice = values.slice(start, *size as usize);

                    // Get the raw buffer data
                    // For primitive arrays, we can access the data buffer directly
                    serialize_array_slice(&slice)
                })
                .collect();

            buffers
        }
        DataType::Binary => {
            let binary_array = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| Error::Internal("failed to downcast to BinaryArray".into()))?;

            Ok((0..array.len())
                .map(|i| bytes::Bytes::copy_from_slice(binary_array.value(i)))
                .collect())
        }
        DataType::LargeBinary => {
            let binary_array = array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| Error::Internal("failed to downcast to LargeBinaryArray".into()))?;

            Ok((0..array.len())
                .map(|i| bytes::Bytes::copy_from_slice(binary_array.value(i)))
                .collect())
        }
        _ => Err(Error::Internal(format!(
            "unsupported external storage type: {:?}",
            array.data_type()
        ))),
    }
}

/// Serialize an array slice to bytes.
///
/// For primitive arrays (Float32, Int64, etc.), this extracts the raw buffer.
fn serialize_array_slice(array: &ArrayRef) -> Result<bytes::Bytes> {
    // For now, handle primitive arrays (Float32, Int64, etc.)
    // TODO: Expand to handle more types
    // Skip dictionary arrays for now
    // if let Some(data) = array.as_any_dictionary_opt() {
    //     return Err(Error::Internal(
    //         "dictionary arrays not supported for external storage".into(),
    //     ));
    // }

    // Use Arrow's buffer directly for primitive types
    let array_data = array.to_data();
    let buffers = array_data.buffers();

    if buffers.is_empty() {
        return Err(Error::Internal("array has no buffers".into()));
    }

    // Get the values buffer
    let buffer = &buffers[0];
    Ok(bytes::Bytes::copy_from_slice(buffer.as_slice()))
}

/// Reconstruct external columns from pager keys.
///
/// Reads the FixedSizeBinary(8) key columns and fetches the actual data from the pager.
pub fn internalize_columns<P>(
    batch: &RecordBatch,
    original_schema: &SchemaRef,
    pager: &P,
) -> Result<RecordBatch>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());

    for (col_idx, field) in original_schema.fields().iter().enumerate() {
        if is_external_field(field) {
            // Read keys and fetch data
            let key_array = batch.column(col_idx);
            let reconstructed = fetch_and_reconstruct_array(key_array, field, pager)?;
            new_columns.push(reconstructed);
        } else {
            new_columns.push(batch.column(col_idx).clone());
        }
    }

    RecordBatch::try_new(original_schema.clone(), new_columns)
        .map_err(|e| Error::Internal(format!("failed to create internalized batch: {}", e)))
}

/// Fetch buffers from pager and reconstruct the original array.
fn fetch_and_reconstruct_array<P>(
    key_array: &ArrayRef,
    original_field: &Field,
    pager: &P,
) -> Result<ArrayRef>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    // Extract keys from FixedSizeBinary array
    let key_array = key_array
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or_else(|| {
            Error::Internal("expected FixedSizeBinary array for external column".into())
        })?;

    let keys: Vec<PhysicalKey> = (0..key_array.len())
        .map(|i| {
            let key_bytes = key_array.value(i);
            let key = u64::from_le_bytes(
                key_bytes
                    .try_into()
                    .map_err(|_| Error::Internal("invalid key size".into()))?,
            );
            Ok(key)
        })
        .collect::<Result<Vec<_>>>()?;

    // Batch fetch all buffers
    let gets: Vec<BatchGet> = keys.iter().map(|&key| BatchGet::Raw { key }).collect();
    let results = pager.batch_get(&gets)?;

    let buffers: Result<Vec<bytes::Bytes>> = results
        .into_iter()
        .map(|result| match result {
            GetResult::Raw { bytes, .. } => Ok(bytes.into_bytes()),
            _ => Err(Error::Internal(
                "expected raw blob for external column".into(),
            )),
        })
        .collect();

    let buffers = buffers?;

    // Reconstruct the array based on the original type
    reconstruct_array_from_buffers(&buffers, original_field)
}

/// Reconstruct an Arrow array from raw buffers.
fn reconstruct_array_from_buffers(buffers: &[bytes::Bytes], field: &Field) -> Result<ArrayRef> {
    use arrow::array::{FixedSizeListArray, Float32Array};

    match field.data_type() {
        DataType::FixedSizeList(inner_field, size) => {
            // Reconstruct FixedSizeListArray
            match inner_field.data_type() {
                DataType::Float32 => {
                    let total_values = buffers.len() * (*size as usize);
                    let mut all_values = Vec::with_capacity(total_values);

                    for buffer in buffers {
                        let floats: &[f32] = bytemuck::cast_slice(buffer.as_ref());
                        all_values.extend_from_slice(floats);
                    }

                    let values_array = Float32Array::from(all_values);
                    let list_array = FixedSizeListArray::new(
                        inner_field.clone(),
                        *size,
                        Arc::new(values_array),
                        None,
                    );

                    Ok(Arc::new(list_array))
                }
                _ => Err(Error::Internal(format!(
                    "unsupported inner type for FixedSizeList: {:?}",
                    inner_field.data_type()
                ))),
            }
        }
        _ => Err(Error::Internal(format!(
            "unsupported external storage reconstruction for type: {:?}",
            field.data_type()
        ))),
    }
}
