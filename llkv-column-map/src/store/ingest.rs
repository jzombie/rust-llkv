use crate::error::Result;
use arrow::array::{ArrayRef, BooleanArray, UInt32Array, UInt64Array};
use std::hash::BuildHasher;

/// Per-chunk logical edit description used by delete and LWW upsert.
/// - keep: optional boolean mask to filter the current rows.
/// - inject_*: optional tail to append after filtering (for upserts).
pub(crate) struct ChunkEdit {
    pub(crate) keep: Option<BooleanArray>,
    pub(crate) inject_data: Option<ArrayRef>,
    pub(crate) inject_rids: Option<ArrayRef>,
}

impl ChunkEdit {
    /// Build a delete-only edit from a set of row indices to drop.
    #[inline]
    pub(crate) fn from_delete_indices(
        rows: usize,
        del_local: &rustc_hash::FxHashSet<usize>,
    ) -> Self {
        let keep = BooleanArray::from_iter((0..rows).map(|j| Some(!del_local.contains(&j))));
        let mut e = ChunkEdit::none();
        e.keep = Some(keep);
        e
    }

    /// Build an LWW upsert edit.
    ///
    /// Accepts any HashMap hasher (e.g., FxHashMap via FxBuildHasher).
    #[inline]
    pub(crate) fn from_lww_upsert<S>(
        old_rid_arr: &UInt64Array,
        up_vec: &[u64],
        del_vec: &[u64],
        incoming_data: &ArrayRef,
        incoming_row_ids: &ArrayRef,
        incoming_ids_map: &std::collections::HashMap<u64, usize, S>,
    ) -> Result<Self>
    where
        S: BuildHasher,
    {
        use rustc_hash::FxHashSet;

        // Build keep mask: drop rows that are upserted or deleted.
        let drop_set: FxHashSet<u64> = up_vec
            .iter()
            .copied()
            .chain(del_vec.iter().copied())
            .collect();

        let keep = BooleanArray::from_iter((0..old_rid_arr.len()).map(|j| {
            let rid = old_rid_arr.value(j);
            Some(!drop_set.contains(&rid))
        }));

        // Injected tails for upsert rows (empty when no upserts).
        let update_indices_arr = if up_vec.is_empty() {
            UInt32Array::from(Vec::<u32>::new())
        } else {
            UInt32Array::from_iter_values(up_vec.iter().map(|rid| {
                let idx = *incoming_ids_map.get(rid).expect("incoming id present");
                u32::try_from(idx).expect("index exceeds u32::MAX")
            }))
        };

        let data_inj = if up_vec.is_empty() {
            None
        } else {
            Some(arrow::compute::take(
                incoming_data,
                &update_indices_arr,
                None,
            )?)
        };

        let rids_inj = if up_vec.is_empty() {
            None
        } else {
            Some(arrow::compute::take(
                incoming_row_ids,
                &update_indices_arr,
                None,
            )?)
        };

        let mut e = ChunkEdit::none();
        e.keep = Some(keep);
        e.inject_data = data_inj;
        e.inject_rids = rids_inj;
        Ok(e)
    }

    #[inline]
    pub(crate) fn none() -> Self {
        Self {
            keep: None,
            inject_data: None,
            inject_rids: None,
        }
    }

    /// Apply a ChunkEdit to in-memory arrays and return new arrays.
    /// No IO here. Caller writes results and refreshes perms.
    #[inline]
    pub(crate) fn apply_edit_to_arrays(
        data_arr: &ArrayRef,
        rid_arr_any: Option<&ArrayRef>,
        edit: &ChunkEdit,
    ) -> Result<(ArrayRef, Option<ArrayRef>)> {
        use crate::store::slicing::{concat_many, zero_offset};

        let mut out_data = data_arr.clone();
        let mut out_rids = rid_arr_any.cloned();

        if let Some(ref keep) = edit.keep {
            out_data = arrow::compute::filter(out_data.as_ref(), keep)?;
            if let Some(r) = out_rids.take() {
                out_rids = Some(arrow::compute::filter(r.as_ref(), keep)?);
            }
        }

        out_data = zero_offset(&out_data);
        if let Some(ref r) = out_rids {
            out_rids = Some(zero_offset(r));
        }

        if let Some(ref inj) = edit.inject_data {
            out_data = concat_many(vec![&out_data, inj])?;
        }
        if let (Some(inj_r), Some(r)) = (edit.inject_rids.as_ref(), out_rids.as_ref()) {
            out_rids = Some(concat_many(vec![r, inj_r])?);
        }

        Ok((out_data, out_rids))
    }
}
