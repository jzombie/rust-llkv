use std::convert::TryFrom;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Builder, StringArray};
use arrow::datatypes::DataType;
use arrow_array::Array;
use croaring::Treemap;
use llkv_column_map::store::GatherNullPolicy;
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use llkv_types::LogicalFieldId;
use simd_r_drive_entry_handle::EntryHandle;

use crate::{ScanOrderDirection, ScanOrderSpec, ScanOrderTransform, ScanStorage};

/// Sort a bitmap of row IDs according to the provided ORDER BY spec.
pub fn sort_row_ids_with_order<P, S>(
    storage: &S,
    row_ids: &Treemap,
    order_spec: ScanOrderSpec,
) -> LlkvResult<Vec<u64>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    let lfid = LogicalFieldId::for_user(storage.table_id(), order_spec.field_id);

    storage.field_data_type(lfid)?;
    if row_ids.is_empty() {
        return Ok(vec![]);
    }

    if matches!(
        order_spec.transform,
        ScanOrderTransform::IdentityInt64
            | ScanOrderTransform::IdentityInt32
            | ScanOrderTransform::IdentityUtf8
    ) && let Some(sorted) = try_full_table_sorted_scan(storage, row_ids, order_spec)?
    {
        return Ok(sorted);
    }

    arrow_sort_row_ids(storage, row_ids, order_spec, lfid)
}

fn arrow_sort_row_ids<P, S>(
    storage: &S,
    row_ids: &Treemap,
    order_spec: ScanOrderSpec,
    lfid: LogicalFieldId,
) -> LlkvResult<Vec<u64>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    let row_ids_vec: Vec<u64> = row_ids.iter().collect();

    let mut ctx = storage.prepare_gather_context(&[lfid])?;
    let batch = storage.gather_row_window_with_context(
        &[lfid],
        &row_ids_vec,
        GatherNullPolicy::IncludeNulls,
        Some(&mut ctx),
    )?;

    let array = batch.column(0);

    let order_values: ArrayRef = match order_spec.transform {
        ScanOrderTransform::IdentityInt64 => {
            if array.data_type() != &DataType::Int64 {
                return Err(Error::InvalidArgumentError(
                    "ORDER BY expected INT64 column for IdentityInt64 transform".into(),
                ));
            }
            Arc::clone(array)
        }
        ScanOrderTransform::IdentityInt32 => {
            if array.data_type() != &DataType::Int32 {
                return Err(Error::InvalidArgumentError(
                    "ORDER BY expected INT32 column for IdentityInt32 transform".into(),
                ));
            }
            Arc::clone(array)
        }
        ScanOrderTransform::IdentityUtf8 => {
            if array.data_type() != &DataType::Utf8 {
                return Err(Error::InvalidArgumentError(
                    "ORDER BY expected UTF8 column for IdentityUtf8 transform".into(),
                ));
            }
            Arc::clone(array)
        }
        ScanOrderTransform::CastUtf8ToInteger => {
            if array.data_type() != &DataType::Utf8 {
                return Err(Error::InvalidArgumentError(
                    "ORDER BY CAST expects a UTF8 column".into(),
                ));
            }
            let strings = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "ORDER BY CAST failed to downcast UTF8 column".into(),
                    )
                })?;
            let mut builder = Int64Builder::with_capacity(strings.len());
            for idx in 0..strings.len() {
                if strings.is_null(idx) {
                    builder.append_null();
                } else {
                    match strings.value(idx).parse::<i64>() {
                        Ok(value) => builder.append_value(value),
                        Err(_) => builder.append_null(),
                    }
                }
            }
            Arc::new(builder.finish()) as ArrayRef
        }
    };

    let sorted_indices = arrow::compute::sort_to_indices(
        &order_values,
        Some(arrow::compute::SortOptions {
            descending: matches!(order_spec.direction, ScanOrderDirection::Descending),
            nulls_first: order_spec.nulls_first,
        }),
        None,
    )?;

    let sorted_row_ids: Vec<u64> = sorted_indices
        .values()
        .iter()
        .map(|&idx| row_ids_vec[idx as usize])
        .collect();

    Ok(sorted_row_ids)
}

fn try_full_table_sorted_scan<P, S>(
    storage: &S,
    row_ids: &Treemap,
    order_spec: ScanOrderSpec,
) -> LlkvResult<Option<Vec<u64>>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    let total_rows = storage.total_rows()?;
    if row_ids.cardinality() != total_rows {
        return Ok(None);
    }

    let all_rows = storage.all_row_ids()?;
    if row_ids.iter().ne(all_rows.iter()) {
        return Ok(None);
    }

    match storage.sorted_row_ids_full_table(order_spec)? {
        Some(sorted) => {
            let len = u64::try_from(sorted.len()).unwrap_or(u64::MAX);
            if len == total_rows {
                Ok(Some(sorted))
            } else {
                Ok(None)
            }
        }
        None => Ok(None),
    }
}
