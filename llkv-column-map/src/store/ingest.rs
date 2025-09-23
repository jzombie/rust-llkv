use super::*;
use crate::error::Result;
use arrow::array::{ArrayRef, BooleanArray};
use arrow::compute;

/// Per-chunk logical edit description used by delete and LWW upsert.
/// - keep: optional boolean mask to filter the current rows.
/// - inject_*: optional tail to append after filtering (for upserts).
pub(crate) struct ChunkEdit {
    pub(crate) keep: Option<BooleanArray>,
    pub(crate) inject_data: Option<ArrayRef>,
    pub(crate) inject_rids: Option<ArrayRef>,
}

impl ChunkEdit {
    #[inline]
    pub(crate) fn none() -> Self {
        Self {
            keep: None,
            inject_data: None,
            inject_rids: None,
        }
    }

    /// Apply a ChunkEdit to in-memory arrays and return new arrays. No IO here.
    /// The caller is responsible for writing results and refreshing perms.
    #[inline]
    pub(crate) fn apply_edit_to_arrays(
        data_arr: &ArrayRef,
        rid_arr: Option<&ArrayRef>,
        edit: &ChunkEdit,
    ) -> Result<(ArrayRef, Option<ArrayRef>)> {
        let mut out_data = if let Some(ref k) = edit.keep {
            compute::filter(data_arr, k)?
        } else {
            data_arr.clone()
        };

        let mut out_rids = if let Some(r) = rid_arr {
            if let Some(ref k) = edit.keep {
                Some(compute::filter(r, k)?)
            } else {
                Some(r.clone())
            }
        } else {
            None
        };

        if let Some(ref inj) = edit.inject_data {
            out_data = concat_many(vec![&out_data, inj])?;
        }
        if let (Some(inj_r), Some(r)) = (edit.inject_rids.as_ref(), out_rids.as_ref()) {
            out_rids = Some(concat_many(vec![r, inj_r])?);
        }

        Ok((out_data, out_rids))
    }
}
