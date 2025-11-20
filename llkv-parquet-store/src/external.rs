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

use arrow::array::{
    Array, ArrayRef, BinaryArray, FixedSizeBinaryArray, LargeBinaryArray, RecordBatch,
};
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

/// Transform a schema by replacing external fields with FixedSizeBinary(12) key columns.
///
/// The 12 bytes encode:
/// - Bytes 0-7: Physical key pointing to the column blob
/// - Bytes 8-11: Row count (u32) - number of rows in the blob
///
/// The transformed schema is what gets written to Parquet files.
pub fn transform_schema_for_storage(schema: &Schema) -> SchemaRef {
    let fields: Vec<Arc<Field>> = schema
        .fields()
        .iter()
        .map(|field| {
            if is_external_field(field) {
                // Replace with FixedSizeBinary(12) to hold PhysicalKey (u64) + row count (u32)
                let mut metadata = field.metadata().clone();
                metadata.insert(
                    "llkv:original_type".to_string(),
                    format!("{:?}", field.data_type()),
                );

                Arc::new(
                    Field::new(
                        field.name(),
                        DataType::FixedSizeBinary(12),
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

/// Store an array's data externally and return a FixedSizeBinary(12) array of encoded keys.
///
/// For efficiency, stores the entire column as a single contiguous blob rather than
/// one blob per row. This dramatically reduces pager overhead and munmap storms.
///
/// Each row gets a 12-byte encoded key containing:
/// - Bytes 0-7: Physical key pointing to the column blob
/// - Bytes 8-11: Row count (u32) - number of rows in the blob
fn store_array_externally<P>(array: &ArrayRef, pager: &P) -> Result<ArrayRef>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let num_rows = array.len();

    // Serialize the entire column into a single contiguous buffer
    let column_buffer = serialize_entire_column(array)?;

    // Allocate a single key for the entire column
    let keys = pager.alloc_many(1)?;
    let column_key = keys[0];

    // Store as a single blob
    pager.batch_put(&[BatchPut::Raw {
        key: column_key,
        bytes: column_buffer,
    }])?;

    // Create FixedSizeBinary(12) array with encoded key + row count
    use arrow::array::FixedSizeBinaryBuilder;
    let mut builder = FixedSizeBinaryBuilder::with_capacity(num_rows, 12);

    // Encode: [physical_key (8 bytes), row_count (4 bytes)]
    let mut encoded = Vec::with_capacity(12);
    encoded.extend_from_slice(&column_key.to_le_bytes());
    encoded.extend_from_slice(&(num_rows as u32).to_le_bytes());

    for _ in 0..num_rows {
        builder.append_value(&encoded)?;
    }

    let key_array = builder.finish();

    Ok(Arc::new(key_array))
}

/// Serialize an entire column into a single contiguous buffer.
fn serialize_entire_column(array: &ArrayRef) -> Result<Vec<u8>> {
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
            let array_data = values.to_data();
            let buffers = array_data.buffers();

            if buffers.is_empty() {
                return Err(Error::Internal("array has no buffers".into()));
            }

            // The entire column's data is in the first buffer
            Ok(buffers[0].as_slice().to_vec())
        }
        DataType::Binary => {
            let binary_array = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| Error::Internal("failed to downcast to BinaryArray".into()))?;

            // Concatenate all binary values
            let mut buffer = Vec::new();
            for i in 0..array.len() {
                buffer.extend_from_slice(binary_array.value(i));
            }
            Ok(buffer)
        }
        DataType::LargeBinary => {
            let binary_array = array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| Error::Internal("failed to downcast to LargeBinaryArray".into()))?;

            // Concatenate all binary values
            let mut buffer = Vec::new();
            for i in 0..array.len() {
                buffer.extend_from_slice(binary_array.value(i));
            }
            Ok(buffer)
        }
        _ => Err(Error::Internal(format!(
            "unsupported external storage type: {:?}",
            array.data_type()
        ))),
    }
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
    // Extract keys from FixedSizeBinary(12) array
    let key_array = key_array
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or_else(|| {
            Error::Internal("expected FixedSizeBinary array for external column".into())
        })?;

    if key_array.is_empty() {
        return Err(Error::Internal("empty key array".into()));
    }

    // Decode key + row count from first entry (all rows have same metadata)
    let first_key_bytes = key_array.value(0);
    let physical_key = u64::from_le_bytes(first_key_bytes[0..8].try_into().unwrap());
    let blob_row_count = u32::from_le_bytes(first_key_bytes[8..12].try_into().unwrap()) as usize;

    fetch_column_blob_and_reconstruct(
        physical_key,
        blob_row_count,
        key_array.len(),
        original_field,
        pager,
    )
}

/// Fetch a single column blob and reconstruct the array.
fn fetch_column_blob_and_reconstruct<P>(
    column_key: PhysicalKey,
    blob_row_count: usize,
    num_rows_needed: usize,
    original_field: &Field,
    pager: &P,
) -> Result<ArrayRef>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    // Fetch the single column blob
    let results = pager.batch_get(&[BatchGet::Raw { key: column_key }])?;

    let column_bytes = match results.into_iter().next() {
        Some(GetResult::Raw { bytes, .. }) => bytes.into_bytes(),
        _ => {
            return Err(Error::Internal(
                "expected raw blob for external column".into(),
            ))
        }
    };

    // Reconstruct the array from the single blob, using blob_row_count and num_rows_needed
    reconstruct_array_from_column_blob(
        &column_bytes,
        blob_row_count,
        num_rows_needed,
        original_field,
    )
}

/// Reconstruct an Arrow array from a single column blob.
fn reconstruct_array_from_column_blob(
    column_bytes: &bytes::Bytes,
    blob_row_count: usize,
    num_rows_needed: usize,
    field: &Field,
) -> Result<ArrayRef> {
    use arrow::array::{FixedSizeListArray, Float32Array};

    match field.data_type() {
        DataType::FixedSizeList(inner_field, size) => {
            // Reconstruct FixedSizeListArray from contiguous data
            match inner_field.data_type() {
                DataType::Float32 => {
                    // Zero-copy construction: wrap the bytes directly in an Arrow buffer
                    let total_bytes_needed =
                        num_rows_needed * (*size as usize) * std::mem::size_of::<f32>();
                    let slice = column_bytes.slice(..total_bytes_needed);

                    // Create Arrow buffer from bytes without copying
                    let buffer = arrow::buffer::Buffer::from(slice);
                    let values_array = Float32Array::new(
                        arrow::buffer::ScalarBuffer::new(
                            buffer,
                            0,
                            num_rows_needed * (*size as usize),
                        ),
                        None,
                    );

                    let field = Arc::new(Field::new("item", DataType::Float32, false));
                    let list_array =
                        FixedSizeListArray::new(field, *size, Arc::new(values_array), None);
                    Ok(Arc::new(list_array))
                }
                _ => Err(Error::Internal(format!(
                    "unsupported FixedSizeList inner type: {:?}",
                    inner_field.data_type()
                ))),
            }
        }
        DataType::Binary | DataType::LargeBinary => {
            // For variable-length types, we need length information
            // This would require storing lengths separately or using a different format
            Err(Error::Internal(
                "Binary/LargeBinary not yet supported for column-level storage".into(),
            ))
        }
        _ => Err(Error::Internal(format!(
            "unsupported external storage type: {:?}",
            field.data_type()
        ))),
    }
}

/// Fetch individual row blobs and reconstruct the array (legacy path).
fn fetch_row_blobs_and_reconstruct<P>(
    key_array: &FixedSizeBinaryArray,
    original_field: &Field,
    pager: &P,
) -> Result<ArrayRef>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
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

    // Batch fetch all buffers for this column
    let gets: Vec<BatchGet> = keys.iter().map(|&key| BatchGet::Raw { key }).collect();
    let results = pager.batch_get(&gets)?;

    // Convert EntryHandles to bytes immediately
    let buffers: Vec<bytes::Bytes> = results
        .into_iter()
        .map(|result| match result {
            GetResult::Raw { bytes, .. } => Ok(bytes.into_bytes()),
            _ => Err(Error::Internal(
                "expected raw blob for external column".into(),
            )),
        })
        .collect::<Result<Vec<_>>>()?;

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
        DataType::Binary => {
            let values: Vec<Vec<u8>> = buffers.iter().map(|b| b.to_vec()).collect();
            Ok(Arc::new(BinaryArray::from_vec(
                values.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
            )))
        }
        DataType::LargeBinary => {
            use arrow::array::LargeBinaryArray;
            let values: Vec<Vec<u8>> = buffers.iter().map(|b| b.to_vec()).collect();
            Ok(Arc::new(LargeBinaryArray::from_vec(
                values.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
            )))
        }
        _ => Err(Error::Internal(format!(
            "unsupported external storage reconstruction for type: {:?}",
            field.data_type()
        ))),
    }
}
