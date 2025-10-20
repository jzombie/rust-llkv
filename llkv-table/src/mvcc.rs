use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, UInt64Array, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::store::{CREATED_BY_COLUMN_NAME, DELETED_BY_COLUMN_NAME, ROW_ID_COLUMN_NAME};
use llkv_result::Error;

use crate::types::{FieldId, RowId};

/// Build MVCC columns (row_id, created_by, deleted_by) for INSERT/CTAS operations.
pub fn build_insert_mvcc_columns(
    row_count: usize,
    start_row_id: RowId,
    creator_txn_id: u64,
    deleted_marker: u64,
) -> (ArrayRef, ArrayRef, ArrayRef) {
    let mut row_builder = UInt64Builder::with_capacity(row_count);
    for offset in 0..row_count {
        row_builder.append_value(start_row_id + offset as u64);
    }

    let mut created_builder = UInt64Builder::with_capacity(row_count);
    let mut deleted_builder = UInt64Builder::with_capacity(row_count);
    for _ in 0..row_count {
        created_builder.append_value(creator_txn_id);
        deleted_builder.append_value(deleted_marker);
    }

    (
        Arc::new(row_builder.finish()) as ArrayRef,
        Arc::new(created_builder.finish()) as ArrayRef,
        Arc::new(deleted_builder.finish()) as ArrayRef,
    )
}

/// Build MVCC field definitions (row_id, created_by, deleted_by).
pub fn build_mvcc_fields() -> Vec<Field> {
    vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new(CREATED_BY_COLUMN_NAME, DataType::UInt64, false),
        Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, false),
    ]
}

/// Build field with field_id metadata for a user column.
pub fn build_field_with_metadata(
    name: &str,
    data_type: DataType,
    nullable: bool,
    field_id: FieldId,
) -> Field {
    let mut metadata = HashMap::with_capacity(1);
    metadata.insert(
        crate::constants::FIELD_ID_META_KEY.to_string(),
        field_id.to_string(),
    );
    Field::new(name, data_type, nullable).with_metadata(metadata)
}

/// Build DELETE batch with row_id and deleted_by columns.
pub fn build_delete_batch(
    row_ids: Vec<RowId>,
    deleted_by_txn_id: u64,
) -> llkv_result::Result<RecordBatch> {
    let row_count = row_ids.len();

    let fields = vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, false),
    ];

    let arrays: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(row_ids)),
        Arc::new(UInt64Array::from(vec![deleted_by_txn_id; row_count])),
    ];

    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).map_err(Error::Arrow)
}
