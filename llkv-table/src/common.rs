//! Shared constants and types for LLKV table providers.

pub use llkv_storage::constants::{
    CREATED_BY_COLUMN_NAME, DELETED_BY_COLUMN_NAME, FIELD_ID_META_KEY, ROW_ID_COLUMN_NAME,
};

use arrow::datatypes::Schema;
use arrow::ipc::{
    convert::try_schema_from_flatbuffer_bytes,
    writer::{DictionaryTracker, IpcDataGenerator, IpcWriteOptions},
};
use llkv_result::Error as LlkvError;
use llkv_result::Result;

/// Trait for receiving notifications about table events.
pub trait TableEventListener: Send + Sync {
    /// Called when new rows are appended to a table.
    fn on_rows_appended(&self, row_ids: &[u64]) -> Result<()>;
}

/// Default batch size for scanning.
pub const DEFAULT_SCAN_BATCH_SIZE: usize = 1024;

/// Serialize an Arrow schema to IPC bytes.
pub fn schema_to_bytes(schema: &Schema) -> Result<Vec<u8>> {
    let data_gen = IpcDataGenerator::default();
    let mut tracker = DictionaryTracker::new(true);
    let encoded = data_gen.schema_to_bytes_with_dictionary_tracker(
        schema,
        &mut tracker,
        &IpcWriteOptions::default(),
    );
    Ok(encoded.ipc_message)
}

/// Deserialize an Arrow schema from IPC bytes.
pub fn bytes_to_schema(bytes: &[u8]) -> Result<Schema> {
    try_schema_from_flatbuffer_bytes(bytes)
        .map_err(|e| LlkvError::Internal(format!("failed to deserialize schema metadata: {e}")))
}
