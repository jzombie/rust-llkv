use super::*;
use crate::types::Namespace;
use arrow::datatypes::DataType;
use rustc_hash::FxHashMap;

#[derive(Debug)]
pub struct SortedChunkBuffers {
    values: Vec<EntryHandle>,
    perms: Vec<EntryHandle>,
    value_index: FxHashMap<PhysicalKey, usize>,
    perm_index: FxHashMap<PhysicalKey, usize>,
}

impl SortedChunkBuffers {
    pub(crate) fn from_parts(
        metas: &[ChunkMetadata],
        values: Vec<EntryHandle>,
        perms: Vec<EntryHandle>,
    ) -> Result<Self> {
        if metas.len() != values.len() || metas.len() != perms.len() {
            return Err(Error::Internal("sorted buffers length mismatch".into()));
        }
        let mut value_index = FxHashMap::with_capacity_and_hasher(metas.len(), Default::default());
        let mut perm_index = FxHashMap::with_capacity_and_hasher(metas.len(), Default::default());
        for (idx, meta) in metas.iter().enumerate() {
            value_index.insert(meta.chunk_pk, idx);
            perm_index.insert(meta.value_order_perm_pk, idx);
        }
        Ok(Self {
            values,
            perms,
            value_index,
            perm_index,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    #[inline]
    pub fn value_handle(&self, idx: usize) -> &EntryHandle {
        &self.values[idx]
    }

    #[inline]
    pub fn perm_handle(&self, idx: usize) -> &EntryHandle {
        &self.perms[idx]
    }

    #[inline]
    pub fn value_by_pk(&self, pk: PhysicalKey) -> Option<&EntryHandle> {
        self.value_index.get(&pk).map(|&idx| &self.values[idx])
    }

    #[inline]
    pub fn perm_by_pk(&self, pk: PhysicalKey) -> Option<&EntryHandle> {
        self.perm_index.get(&pk).map(|&idx| &self.perms[idx])
    }

    #[inline]
    pub fn values(&self) -> &[EntryHandle] {
        &self.values
    }

    #[inline]
    pub fn perms(&self) -> &[EntryHandle] {
        &self.perms
    }
}

#[derive(Debug)]
pub struct SortedChunkBuffersWithRids {
    base: SortedChunkBuffers,
    rids: Vec<EntryHandle>,
    rid_index: FxHashMap<PhysicalKey, usize>,
}

impl SortedChunkBuffersWithRids {
    pub(crate) fn from_parts(
        metas_val: &[ChunkMetadata],
        metas_rid: &[ChunkMetadata],
        values: Vec<EntryHandle>,
        perms: Vec<EntryHandle>,
        rids: Vec<EntryHandle>,
    ) -> Result<Self> {
        if metas_val.len() != metas_rid.len()
            || metas_val.len() != values.len()
            || metas_val.len() != perms.len()
            || metas_rid.len() != rids.len()
        {
            return Err(Error::Internal(
                "sorted with rids buffers length mismatch".into(),
            ));
        }
        let base = SortedChunkBuffers::from_parts(metas_val, values, perms)?;
        let mut rid_index =
            FxHashMap::with_capacity_and_hasher(metas_rid.len(), Default::default());
        for (idx, meta) in metas_rid.iter().enumerate() {
            rid_index.insert(meta.chunk_pk, idx);
        }
        Ok(Self {
            base,
            rids,
            rid_index,
        })
    }

    #[inline]
    pub fn base(&self) -> &SortedChunkBuffers {
        &self.base
    }

    #[inline]
    pub fn rid_handle(&self, idx: usize) -> &EntryHandle {
        &self.rids[idx]
    }

    #[inline]
    pub fn rid_by_pk(&self, pk: PhysicalKey) -> Option<&EntryHandle> {
        self.rid_index.get(&pk).map(|&idx| &self.rids[idx])
    }

    #[inline]
    pub fn rids(&self) -> &[EntryHandle] {
        &self.rids
    }
}

pub(crate) fn load_sorted_buffers<P: Pager<Blob = EntryHandle>>(
    pager: &P,
    metas: &[ChunkMetadata],
) -> Result<SortedChunkBuffers> {
    let mut gets: Vec<BatchGet> = Vec::with_capacity(metas.len() * 2);
    for m in metas {
        gets.push(BatchGet::Raw { key: m.chunk_pk });
        gets.push(BatchGet::Raw {
            key: m.value_order_perm_pk,
        });
    }
    let results = pager.batch_get(&gets)?;
    let mut values: Vec<EntryHandle> = Vec::with_capacity(metas.len());
    let mut perms: Vec<EntryHandle> = Vec::with_capacity(metas.len());
    let mut iter = results.into_iter();
    for m in metas {
        match iter.next() {
            Some(GetResult::Raw { key, bytes }) if key == m.chunk_pk => values.push(bytes),
            _ => return Err(Error::Internal("sorted: unexpected value blob".into())),
        }
        match iter.next() {
            Some(GetResult::Raw { key, bytes }) if key == m.value_order_perm_pk => {
                perms.push(bytes)
            }
            _ => return Err(Error::Internal("sorted: unexpected perm blob".into())),
        }
    }
    if iter.next().is_some() {
        return Err(Error::Internal("sorted: extra blobs returned".into()));
    }
    SortedChunkBuffers::from_parts(metas, values, perms)
}

pub(crate) fn load_sorted_buffers_with_rids<P: Pager<Blob = EntryHandle>>(
    pager: &P,
    metas_val: &[ChunkMetadata],
    metas_rid: &[ChunkMetadata],
) -> Result<SortedChunkBuffersWithRids> {
    if metas_val.len() != metas_rid.len() {
        return Err(Error::Internal(
            "sorted_with_row_ids: chunk count mismatch".into(),
        ));
    }
    let mut gets: Vec<BatchGet> = Vec::with_capacity(metas_val.len() * 3);
    for (mv, mr) in metas_val.iter().zip(metas_rid.iter()) {
        gets.push(BatchGet::Raw { key: mv.chunk_pk });
        gets.push(BatchGet::Raw {
            key: mv.value_order_perm_pk,
        });
        gets.push(BatchGet::Raw { key: mr.chunk_pk });
    }
    let results = pager.batch_get(&gets)?;
    let mut values: Vec<EntryHandle> = Vec::with_capacity(metas_val.len());
    let mut perms: Vec<EntryHandle> = Vec::with_capacity(metas_val.len());
    let mut rids: Vec<EntryHandle> = Vec::with_capacity(metas_val.len());
    let mut iter = results.into_iter();
    for (mv, mr) in metas_val.iter().zip(metas_rid.iter()) {
        match iter.next() {
            Some(GetResult::Raw { key, bytes }) if key == mv.chunk_pk => values.push(bytes),
            _ => return Err(Error::Internal("sorted: unexpected value blob".into())),
        }
        match iter.next() {
            Some(GetResult::Raw { key, bytes }) if key == mv.value_order_perm_pk => {
                perms.push(bytes)
            }
            _ => return Err(Error::Internal("sorted: unexpected perm blob".into())),
        }
        match iter.next() {
            Some(GetResult::Raw { key, bytes }) if key == mr.chunk_pk => rids.push(bytes),
            _ => return Err(Error::Internal("sorted: unexpected rid blob".into())),
        }
    }
    if iter.next().is_some() {
        return Err(Error::Internal("sorted: extra blobs returned".into()));
    }
    SortedChunkBuffersWithRids::from_parts(metas_val, metas_rid, values, perms, rids)
}

use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
struct FloatOrd64(f64);

impl FloatOrd64 {
    #[inline]
    fn new(v: f64) -> Self {
        Self(v)
    }
}

impl PartialEq for FloatOrd64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for FloatOrd64 {}

impl PartialOrd for FloatOrd64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloatOrd64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[derive(Clone, Copy, Debug)]
struct FloatOrd32(f32);

impl FloatOrd32 {
    #[inline]
    fn new(v: f32) -> Self {
        Self(v)
    }
}

impl PartialEq for FloatOrd32 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for FloatOrd32 {}

impl PartialOrd for FloatOrd32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloatOrd32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

macro_rules! sorted_visit_impl {
    ($name:ident, $name_rev:ident, $ArrTy:ty, $visit:ident) => {
        pub(crate) fn $name<P: Pager<Blob = EntryHandle>>(
            _pager: &P,
            metas: &[ChunkMetadata],
            buffers: &SortedChunkBuffers,
            visitor: &mut dyn PrimitiveSortedVisitor,
        ) -> Result<()> {
            let mut arrays: Vec<$ArrTy> = Vec::with_capacity(metas.len());
            for idx in 0..metas.len() {
                let data_any = deserialize_array(buffers.value_handle(idx).clone())?;
                let perm_any = deserialize_array(buffers.perm_handle(idx).clone())?;
                let perm = perm_any
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let taken_any = compute::take(&data_any, perm, None)?;
                let arr = taken_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("sorted downcast".into()))?
                    .clone();
                arrays.push(arr);
            }

            if arrays.is_empty() {
                return Ok(());
            }
            kmerge_coalesced::<_, _>(
                &arrays,
                &mut |a: &$ArrTy| a.len(),
                &mut |a: &$ArrTy, i: usize| a.value(i),
                &mut |c, s, l| visitor.$visit(&arrays[c], s, l),
            );
            Ok(())
        }
        pub(crate) fn $name_rev<P: Pager<Blob = EntryHandle>>(
            _pager: &P,
            metas: &[ChunkMetadata],
            buffers: &SortedChunkBuffers,
            visitor: &mut dyn PrimitiveSortedVisitor,
        ) -> Result<()> {
            let mut arrays: Vec<$ArrTy> = Vec::with_capacity(metas.len());
            for idx in 0..metas.len() {
                let data_any = deserialize_array(buffers.value_handle(idx).clone())?;
                let perm_any = deserialize_array(buffers.perm_handle(idx).clone())?;
                let perm = perm_any
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let taken_any = compute::take(&data_any, perm, None)?;
                let arr = taken_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("sorted downcast".into()))?
                    .clone();
                arrays.push(arr);
            }
            if arrays.is_empty() {
                return Ok(());
            }
            kmerge_coalesced_rev::<_, _>(
                &arrays,
                &mut |a: &$ArrTy| a.len(),
                &mut |a: &$ArrTy, i: usize| a.value(i),
                &mut |c, s, l| visitor.$visit(&arrays[c], s, l),
            );
            Ok(())
        }
    };
}

sorted_visit_impl!(sorted_visit_u64, sorted_visit_u64_rev, UInt64Array, u64_run);
sorted_visit_impl!(sorted_visit_u32, sorted_visit_u32_rev, UInt32Array, u32_run);
sorted_visit_impl!(sorted_visit_u16, sorted_visit_u16_rev, UInt16Array, u16_run);
sorted_visit_impl!(sorted_visit_u8, sorted_visit_u8_rev, UInt8Array, u8_run);
sorted_visit_impl!(sorted_visit_i64, sorted_visit_i64_rev, Int64Array, i64_run);
sorted_visit_impl!(sorted_visit_i32, sorted_visit_i32_rev, Int32Array, i32_run);
sorted_visit_impl!(sorted_visit_i16, sorted_visit_i16_rev, Int16Array, i16_run);
sorted_visit_impl!(sorted_visit_i8, sorted_visit_i8_rev, Int8Array, i8_run);

macro_rules! sorted_visit_float_impl {
    ($name:ident, $name_rev:ident, $ArrTy:ty, $visit:ident, $key:ty) => {
        pub(crate) fn $name<P: Pager<Blob = EntryHandle>>(
            _pager: &P,
            metas: &[ChunkMetadata],
            buffers: &SortedChunkBuffers,
            visitor: &mut dyn PrimitiveSortedVisitor,
        ) -> Result<()> {
            let mut arrays: Vec<$ArrTy> = Vec::with_capacity(metas.len());
            for idx in 0..metas.len() {
                let data_any = deserialize_array(buffers.value_handle(idx).clone())?;
                let perm_any = deserialize_array(buffers.perm_handle(idx).clone())?;
                let perm = perm_any
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let taken_any = compute::take(&data_any, perm, None)?;
                let arr = taken_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("sorted downcast".into()))?
                    .clone();
                arrays.push(arr);
            }
            if arrays.is_empty() {
                return Ok(());
            }
            kmerge_coalesced::<$key, _>(
                &arrays,
                &mut |a: &$ArrTy| a.len(),
                &mut |a: &$ArrTy, i: usize| <$key>::new(a.value(i)),
                &mut |c, s, l| visitor.$visit(&arrays[c], s, l),
            );
            Ok(())
        }
        pub(crate) fn $name_rev<P: Pager<Blob = EntryHandle>>(
            _pager: &P,
            metas: &[ChunkMetadata],
            buffers: &SortedChunkBuffers,
            visitor: &mut dyn PrimitiveSortedVisitor,
        ) -> Result<()> {
            let mut arrays: Vec<$ArrTy> = Vec::with_capacity(metas.len());
            for idx in 0..metas.len() {
                let data_any = deserialize_array(buffers.value_handle(idx).clone())?;
                let perm_any = deserialize_array(buffers.perm_handle(idx).clone())?;
                let perm = perm_any
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let taken_any = compute::take(&data_any, perm, None)?;
                let arr = taken_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("sorted downcast".into()))?
                    .clone();
                arrays.push(arr);
            }
            if arrays.is_empty() {
                return Ok(());
            }
            kmerge_coalesced_rev::<$key, _>(
                &arrays,
                &mut |a: &$ArrTy| a.len(),
                &mut |a: &$ArrTy, i: usize| <$key>::new(a.value(i)),
                &mut |c, s, l| visitor.$visit(&arrays[c], s, l),
            );
            Ok(())
        }
    };
}

sorted_visit_float_impl!(
    sorted_visit_f64,
    sorted_visit_f64_rev,
    Float64Array,
    f64_run,
    FloatOrd64
);
sorted_visit_float_impl!(
    sorted_visit_f32,
    sorted_visit_f32_rev,
    Float32Array,
    f32_run,
    FloatOrd32
);
sorted_visit_impl!(
    sorted_visit_bool,
    sorted_visit_bool_rev,
    BooleanArray,
    bool_run
);
sorted_visit_impl!(
    sorted_visit_date64,
    sorted_visit_date64_rev,
    Date64Array,
    date64_run
);
sorted_visit_impl!(
    sorted_visit_date32,
    sorted_visit_date32_rev,
    Date32Array,
    date32_run
);

// Note: A sorted-with-row-ids variant can be added similarly if needed.

macro_rules! sorted_with_rids_impl {
    ($name:ident, $name_rev:ident, $ArrTy:ty, $visit:ident) => {
        pub(crate) fn $name<P: Pager<Blob = EntryHandle>>(
            _pager: &P,
            metas_val: &[ChunkMetadata],
            metas_rid: &[ChunkMetadata],
            buffers: &SortedChunkBuffersWithRids,
            visitor: &mut dyn PrimitiveSortedWithRowIdsVisitor,
        ) -> Result<()> {
            if metas_val.len() != metas_rid.len() {
                return Err(Error::Internal(
                    "sorted_with_rids: chunk count mismatch".into(),
                ));
            }
            let mut vals: Vec<$ArrTy> = Vec::with_capacity(metas_val.len());
            let mut rids: Vec<UInt64Array> = Vec::with_capacity(metas_val.len());
            for idx in 0..metas_val.len() {
                let data_any = deserialize_array(buffers.base().value_handle(idx).clone())?;
                let perm_any = deserialize_array(buffers.base().perm_handle(idx).clone())?;
                let perm = perm_any
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let taken_any = compute::take(&data_any, perm, None)?;
                let arr = taken_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("sorted downcast".into()))?
                    .clone();
                vals.push(arr);
                let rid_any = deserialize_array(buffers.rid_handle(idx).clone())?;
                let taken_rid_any = compute::take(&rid_any, perm, None)?;
                let rid = taken_rid_any
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("row_id downcast".into()))?
                    .clone();
                rids.push(rid);
            }
            if vals.is_empty() {
                return Ok(());
            }
            kmerge_coalesced::<_, _>(
                &vals,
                &mut |a: &$ArrTy| a.len(),
                &mut |a: &$ArrTy, i: usize| a.value(i),
                &mut |c, s, l| visitor.$visit(&vals[c], &rids[c], s, l),
            );
            Ok(())
        }
        pub(crate) fn $name_rev<P: Pager<Blob = EntryHandle>>(
            _pager: &P,
            metas_val: &[ChunkMetadata],
            metas_rid: &[ChunkMetadata],
            buffers: &SortedChunkBuffersWithRids,
            visitor: &mut dyn PrimitiveSortedWithRowIdsVisitor,
        ) -> Result<()> {
            if metas_val.len() != metas_rid.len() {
                return Err(Error::Internal(
                    "sorted_with_rids: chunk count mismatch".into(),
                ));
            }
            let mut vals: Vec<$ArrTy> = Vec::with_capacity(metas_val.len());
            let mut rids: Vec<UInt64Array> = Vec::with_capacity(metas_val.len());
            for idx in 0..metas_val.len() {
                let data_any = deserialize_array(buffers.base().value_handle(idx).clone())?;
                let perm_any = deserialize_array(buffers.base().perm_handle(idx).clone())?;
                let perm = perm_any
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let taken_any = compute::take(&data_any, perm, None)?;
                let arr = taken_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("sorted downcast".into()))?
                    .clone();
                vals.push(arr);
                let rid_any = deserialize_array(buffers.rid_handle(idx).clone())?;
                let taken_rid_any = compute::take(&rid_any, perm, None)?;
                let rid = taken_rid_any
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("row_id downcast".into()))?
                    .clone();
                rids.push(rid);
            }
            if vals.is_empty() {
                return Ok(());
            }
            kmerge_coalesced_rev::<_, _>(
                &vals,
                &mut |a: &$ArrTy| a.len(),
                &mut |a: &$ArrTy, i: usize| a.value(i),
                &mut |c, s, l| visitor.$visit(&vals[c], &rids[c], s, l),
            );
            Ok(())
        }
    };
}

sorted_with_rids_impl!(
    sorted_visit_with_rids_u64,
    sorted_visit_with_rids_u64_rev,
    UInt64Array,
    u64_run_with_rids
);
sorted_with_rids_impl!(
    sorted_visit_with_rids_u32,
    sorted_visit_with_rids_u32_rev,
    UInt32Array,
    u32_run_with_rids
);
sorted_with_rids_impl!(
    sorted_visit_with_rids_u16,
    sorted_visit_with_rids_u16_rev,
    UInt16Array,
    u16_run_with_rids
);
sorted_with_rids_impl!(
    sorted_visit_with_rids_u8,
    sorted_visit_with_rids_u8_rev,
    UInt8Array,
    u8_run_with_rids
);
sorted_with_rids_impl!(
    sorted_visit_with_rids_i64,
    sorted_visit_with_rids_i64_rev,
    Int64Array,
    i64_run_with_rids
);
sorted_with_rids_impl!(
    sorted_visit_with_rids_i32,
    sorted_visit_with_rids_i32_rev,
    Int32Array,
    i32_run_with_rids
);
sorted_with_rids_impl!(
    sorted_visit_with_rids_i16,
    sorted_visit_with_rids_i16_rev,
    Int16Array,
    i16_run_with_rids
);
sorted_with_rids_impl!(
    sorted_visit_with_rids_i8,
    sorted_visit_with_rids_i8_rev,
    Int8Array,
    i8_run_with_rids
);
sorted_with_rids_impl!(
    sorted_visit_with_rids_bool,
    sorted_visit_with_rids_bool_rev,
    BooleanArray,
    bool_run_with_rids
);
sorted_with_rids_impl!(
    sorted_visit_with_rids_date64,
    sorted_visit_with_rids_date64_rev,
    Date64Array,
    date64_run_with_rids
);
sorted_with_rids_impl!(
    sorted_visit_with_rids_date32,
    sorted_visit_with_rids_date32_rev,
    Date32Array,
    date32_run_with_rids
);

macro_rules! sorted_with_rids_float_impl {
    ($name:ident, $name_rev:ident, $ArrTy:ty, $visit:ident, $key:ty) => {
        pub(crate) fn $name<P: Pager<Blob = EntryHandle>>(
            _pager: &P,
            metas_val: &[ChunkMetadata],
            metas_rid: &[ChunkMetadata],
            buffers: &SortedChunkBuffersWithRids,
            visitor: &mut dyn PrimitiveSortedWithRowIdsVisitor,
        ) -> Result<()> {
            if metas_val.len() != metas_rid.len() {
                return Err(Error::Internal(
                    "sorted_with_rids: chunk count mismatch".into(),
                ));
            }
            let mut vals: Vec<$ArrTy> = Vec::with_capacity(metas_val.len());
            let mut rids: Vec<UInt64Array> = Vec::with_capacity(metas_val.len());
            for idx in 0..metas_val.len() {
                let data_any = deserialize_array(buffers.base().value_handle(idx).clone())?;
                let perm_any = deserialize_array(buffers.base().perm_handle(idx).clone())?;
                let perm = perm_any
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let taken_any = compute::take(&data_any, perm, None)?;
                let arr = taken_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("sorted downcast".into()))?
                    .clone();
                vals.push(arr);
                let rid_any = deserialize_array(buffers.rid_handle(idx).clone())?;
                let taken_rid_any = compute::take(&rid_any, perm, None)?;
                let rid = taken_rid_any
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("row_id downcast".into()))?
                    .clone();
                rids.push(rid);
            }
            if vals.is_empty() {
                return Ok(());
            }
            kmerge_coalesced::<$key, _>(
                &vals,
                &mut |a: &$ArrTy| a.len(),
                &mut |a: &$ArrTy, i: usize| <$key>::new(a.value(i)),
                &mut |c, s, l| visitor.$visit(&vals[c], &rids[c], s, l),
            );
            Ok(())
        }
        pub(crate) fn $name_rev<P: Pager<Blob = EntryHandle>>(
            _pager: &P,
            metas_val: &[ChunkMetadata],
            metas_rid: &[ChunkMetadata],
            buffers: &SortedChunkBuffersWithRids,
            visitor: &mut dyn PrimitiveSortedWithRowIdsVisitor,
        ) -> Result<()> {
            if metas_val.len() != metas_rid.len() {
                return Err(Error::Internal(
                    "sorted_with_rids: chunk count mismatch".into(),
                ));
            }
            let mut vals: Vec<$ArrTy> = Vec::with_capacity(metas_val.len());
            let mut rids: Vec<UInt64Array> = Vec::with_capacity(metas_val.len());
            for idx in 0..metas_val.len() {
                let data_any = deserialize_array(buffers.base().value_handle(idx).clone())?;
                let perm_any = deserialize_array(buffers.base().perm_handle(idx).clone())?;
                let perm = perm_any
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let taken_any = compute::take(&data_any, perm, None)?;
                let arr = taken_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("sorted downcast".into()))?
                    .clone();
                vals.push(arr);
                let rid_any = deserialize_array(buffers.rid_handle(idx).clone())?;
                let taken_rid_any = compute::take(&rid_any, perm, None)?;
                let rid = taken_rid_any
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("row_id downcast".into()))?
                    .clone();
                rids.push(rid);
            }
            if vals.is_empty() {
                return Ok(());
            }
            kmerge_coalesced_rev::<$key, _>(
                &vals,
                &mut |a: &$ArrTy| a.len(),
                &mut |a: &$ArrTy, i: usize| <$key>::new(a.value(i)),
                &mut |c, s, l| visitor.$visit(&vals[c], &rids[c], s, l),
            );
            Ok(())
        }
    };
}

sorted_with_rids_float_impl!(
    sorted_visit_with_rids_f64,
    sorted_visit_with_rids_f64_rev,
    Float64Array,
    f64_run_with_rids,
    FloatOrd64
);
sorted_with_rids_float_impl!(
    sorted_visit_with_rids_f32,
    sorted_visit_with_rids_f32_rev,
    Float32Array,
    f32_run_with_rids,
    FloatOrd32
);

pub(crate) trait SortedDispatch {
    fn visit<P>(
        pager: &P,
        metas: &[ChunkMetadata],
        buffers: &SortedChunkBuffers,
        visitor: &mut dyn PrimitiveSortedVisitor,
    ) -> Result<()>
    where
        P: Pager<Blob = EntryHandle>;

    fn visit_rev<P>(
        pager: &P,
        metas: &[ChunkMetadata],
        buffers: &SortedChunkBuffers,
        visitor: &mut dyn PrimitiveSortedVisitor,
    ) -> Result<()>
    where
        P: Pager<Blob = EntryHandle>;

    fn visit_with_rids<P>(
        pager: &P,
        metas_val: &[ChunkMetadata],
        metas_rid: &[ChunkMetadata],
        buffers: &SortedChunkBuffersWithRids,
        visitor: &mut dyn PrimitiveSortedWithRowIdsVisitor,
    ) -> Result<()>
    where
        P: Pager<Blob = EntryHandle>;

    fn visit_with_rids_rev<P>(
        pager: &P,
        metas_val: &[ChunkMetadata],
        metas_rid: &[ChunkMetadata],
        buffers: &SortedChunkBuffersWithRids,
        visitor: &mut dyn PrimitiveSortedWithRowIdsVisitor,
    ) -> Result<()>
    where
        P: Pager<Blob = EntryHandle>;

    fn visit_bounds<P>(
        pager: &P,
        metas: &[ChunkMetadata],
        buffers: &SortedChunkBuffers,
        ir: &IntRanges,
        visitor: &mut dyn PrimitiveSortedVisitor,
    ) -> Result<()>
    where
        P: Pager<Blob = EntryHandle>;

    fn visit_with_rids_bounds<P>(
        pager: &P,
        metas_val: &[ChunkMetadata],
        metas_rid: &[ChunkMetadata],
        buffers: &SortedChunkBuffersWithRids,
        ir: &IntRanges,
        visitor: &mut dyn PrimitiveSortedWithRowIdsVisitor,
    ) -> Result<()>
    where
        P: Pager<Blob = EntryHandle>;
}

macro_rules! impl_sorted_dispatch {
    (
        $ty:ty,
        $visit:ident,
        $visit_rev:ident,
        $with_rids:ident,
        $with_rids_rev:ident,
        $bounds:ident,
        $bounds_with_rids:ident,
        $range_field:ident
    ) => {
        impl SortedDispatch for $ty {
            #[inline]
            fn visit<P>(
                pager: &P,
                metas: &[ChunkMetadata],
                buffers: &SortedChunkBuffers,
                visitor: &mut dyn PrimitiveSortedVisitor,
            ) -> Result<()>
            where
                P: Pager<Blob = EntryHandle>,
            {
                $visit(pager, metas, buffers, visitor)
            }

            #[inline]
            fn visit_rev<P>(
                pager: &P,
                metas: &[ChunkMetadata],
                buffers: &SortedChunkBuffers,
                visitor: &mut dyn PrimitiveSortedVisitor,
            ) -> Result<()>
            where
                P: Pager<Blob = EntryHandle>,
            {
                $visit_rev(pager, metas, buffers, visitor)
            }

            #[inline]
            fn visit_with_rids<P>(
                pager: &P,
                metas_val: &[ChunkMetadata],
                metas_rid: &[ChunkMetadata],
                buffers: &SortedChunkBuffersWithRids,
                visitor: &mut dyn PrimitiveSortedWithRowIdsVisitor,
            ) -> Result<()>
            where
                P: Pager<Blob = EntryHandle>,
            {
                $with_rids(pager, metas_val, metas_rid, buffers, visitor)
            }

            #[inline]
            fn visit_with_rids_rev<P>(
                pager: &P,
                metas_val: &[ChunkMetadata],
                metas_rid: &[ChunkMetadata],
                buffers: &SortedChunkBuffersWithRids,
                visitor: &mut dyn PrimitiveSortedWithRowIdsVisitor,
            ) -> Result<()>
            where
                P: Pager<Blob = EntryHandle>,
            {
                $with_rids_rev(pager, metas_val, metas_rid, buffers, visitor)
            }

            #[inline]
            fn visit_bounds<P>(
                pager: &P,
                metas: &[ChunkMetadata],
                buffers: &SortedChunkBuffers,
                ir: &IntRanges,
                visitor: &mut dyn PrimitiveSortedVisitor,
            ) -> Result<()>
            where
                P: Pager<Blob = EntryHandle>,
            {
                let bounds = ir
                    .$range_field
                    .unwrap_or((Bound::Unbounded, Bound::Unbounded));
                $bounds(pager, metas, buffers, bounds, visitor)
            }

            #[inline]
            fn visit_with_rids_bounds<P>(
                pager: &P,
                metas_val: &[ChunkMetadata],
                metas_rid: &[ChunkMetadata],
                buffers: &SortedChunkBuffersWithRids,
                ir: &IntRanges,
                visitor: &mut dyn PrimitiveSortedWithRowIdsVisitor,
            ) -> Result<()>
            where
                P: Pager<Blob = EntryHandle>,
            {
                let bounds = ir
                    .$range_field
                    .unwrap_or((Bound::Unbounded, Bound::Unbounded));
                $bounds_with_rids(pager, metas_val, metas_rid, buffers, bounds, visitor)
            }
        }
    };
}

impl_sorted_dispatch!(
    arrow::datatypes::UInt64Type,
    sorted_visit_u64,
    sorted_visit_u64_rev,
    sorted_visit_with_rids_u64,
    sorted_visit_with_rids_u64_rev,
    sorted_visit_u64_bounds,
    sorted_visit_with_rids_u64_bounds,
    u64_r
);
impl_sorted_dispatch!(
    arrow::datatypes::UInt32Type,
    sorted_visit_u32,
    sorted_visit_u32_rev,
    sorted_visit_with_rids_u32,
    sorted_visit_with_rids_u32_rev,
    sorted_visit_u32_bounds,
    sorted_visit_with_rids_u32_bounds,
    u32_r
);
impl_sorted_dispatch!(
    arrow::datatypes::UInt16Type,
    sorted_visit_u16,
    sorted_visit_u16_rev,
    sorted_visit_with_rids_u16,
    sorted_visit_with_rids_u16_rev,
    sorted_visit_u16_bounds,
    sorted_visit_with_rids_u16_bounds,
    u16_r
);
impl_sorted_dispatch!(
    arrow::datatypes::UInt8Type,
    sorted_visit_u8,
    sorted_visit_u8_rev,
    sorted_visit_with_rids_u8,
    sorted_visit_with_rids_u8_rev,
    sorted_visit_u8_bounds,
    sorted_visit_with_rids_u8_bounds,
    u8_r
);
impl_sorted_dispatch!(
    arrow::datatypes::Int64Type,
    sorted_visit_i64,
    sorted_visit_i64_rev,
    sorted_visit_with_rids_i64,
    sorted_visit_with_rids_i64_rev,
    sorted_visit_i64_bounds,
    sorted_visit_with_rids_i64_bounds,
    i64_r
);
impl_sorted_dispatch!(
    arrow::datatypes::Int32Type,
    sorted_visit_i32,
    sorted_visit_i32_rev,
    sorted_visit_with_rids_i32,
    sorted_visit_with_rids_i32_rev,
    sorted_visit_i32_bounds,
    sorted_visit_with_rids_i32_bounds,
    i32_r
);
impl_sorted_dispatch!(
    arrow::datatypes::Int16Type,
    sorted_visit_i16,
    sorted_visit_i16_rev,
    sorted_visit_with_rids_i16,
    sorted_visit_with_rids_i16_rev,
    sorted_visit_i16_bounds,
    sorted_visit_with_rids_i16_bounds,
    i16_r
);
impl_sorted_dispatch!(
    arrow::datatypes::Int8Type,
    sorted_visit_i8,
    sorted_visit_i8_rev,
    sorted_visit_with_rids_i8,
    sorted_visit_with_rids_i8_rev,
    sorted_visit_i8_bounds,
    sorted_visit_with_rids_i8_bounds,
    i8_r
);
impl_sorted_dispatch!(
    arrow::datatypes::Float64Type,
    sorted_visit_f64,
    sorted_visit_f64_rev,
    sorted_visit_with_rids_f64,
    sorted_visit_with_rids_f64_rev,
    sorted_visit_f64_bounds,
    sorted_visit_with_rids_f64_bounds,
    f64_r
);
impl_sorted_dispatch!(
    arrow::datatypes::Float32Type,
    sorted_visit_f32,
    sorted_visit_f32_rev,
    sorted_visit_with_rids_f32,
    sorted_visit_with_rids_f32_rev,
    sorted_visit_f32_bounds,
    sorted_visit_with_rids_f32_bounds,
    f32_r
);
impl_sorted_dispatch!(
    arrow::datatypes::BooleanType,
    sorted_visit_bool,
    sorted_visit_bool_rev,
    sorted_visit_with_rids_bool,
    sorted_visit_with_rids_bool_rev,
    sorted_visit_bool_bounds,
    sorted_visit_with_rids_bool_bounds,
    bool_r
);
impl_sorted_dispatch!(
    arrow::datatypes::Date64Type,
    sorted_visit_date64,
    sorted_visit_date64_rev,
    sorted_visit_with_rids_date64,
    sorted_visit_with_rids_date64_rev,
    sorted_visit_date64_bounds,
    sorted_visit_with_rids_date64_bounds,
    i64_r
);
impl_sorted_dispatch!(
    arrow::datatypes::Date32Type,
    sorted_visit_date32,
    sorted_visit_date32_rev,
    sorted_visit_with_rids_date32,
    sorted_visit_with_rids_date32_rev,
    sorted_visit_date32_bounds,
    sorted_visit_with_rids_date32_bounds,
    i32_r
);

macro_rules! sorted_visit_bounds_impl {
    ($name:ident, $ArrTy:ty, $ty:ty, $visit:ident) => {
        fn $name<P: Pager<Blob = EntryHandle>>(
            _pager: &P,
            metas: &[ChunkMetadata],
            buffers: &SortedChunkBuffers,
            bounds: (Bound<$ty>, Bound<$ty>),
            visitor: &mut dyn PrimitiveSortedVisitor,
        ) -> Result<()> {
            let mut arrays: Vec<$ArrTy> = Vec::with_capacity(metas.len());
            for m in metas {
                let data_any = deserialize_array(
                    buffers
                        .value_by_pk(m.chunk_pk)
                        .ok_or(Error::NotFound)?
                        .clone(),
                )?;
                let perm_any = deserialize_array(
                    buffers
                        .perm_by_pk(m.value_order_perm_pk)
                        .ok_or(Error::NotFound)?
                        .clone(),
                )?;
                let perm = perm_any
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let arr_any = data_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("downcast".into()))?;
                let plen = perm.len();
                if plen == 0 {
                    continue;
                }
                let get = |i: usize| -> $ty {
                    let idx = perm.value(i) as usize;
                    arr_any.value(idx) as $ty
                };
                let (lb, ub) = bounds.clone();
                let s = match lb {
                    Bound::Unbounded => 0,
                    Bound::Included(x) => lower_idx_by(0, plen, &x, &get),
                    Bound::Excluded(x) => upper_idx_by(0, plen, &x, &get),
                };
                let e = match ub {
                    Bound::Unbounded => plen,
                    Bound::Included(x) => upper_idx_by(0, plen, &x, &get),
                    Bound::Excluded(x) => lower_idx_by(0, plen, &x, &get),
                };
                if s >= e {
                    continue;
                }
                let pw = perm.slice(s, e - s);
                let taken_any = compute::take(&data_any, &pw, None)?;
                let a = taken_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("sorted downcast".into()))?
                    .clone();
                arrays.push(a);
            }
            if arrays.is_empty() {
                return Ok(());
            }
            kmerge_coalesced::<_, _>(
                &arrays,
                &mut |a: &$ArrTy| a.len(),
                &mut |a: &$ArrTy, i: usize| a.value(i),
                &mut |c, s, l| visitor.$visit(&arrays[c], s, l),
            );
            Ok(())
        }
    };
}

macro_rules! sorted_visit_float_bounds_impl {
    ($name:ident, $ArrTy:ty, $scalar:ty, $key:ty, $visit:ident) => {
        fn $name<P: Pager<Blob = EntryHandle>>(
            _pager: &P,
            metas: &[ChunkMetadata],
            buffers: &SortedChunkBuffers,
            bounds: (Bound<$scalar>, Bound<$scalar>),
            visitor: &mut dyn PrimitiveSortedVisitor,
        ) -> Result<()> {
            let mut arrays: Vec<$ArrTy> = Vec::with_capacity(metas.len());
            for m in metas {
                let data_any = deserialize_array(
                    buffers
                        .value_by_pk(m.chunk_pk)
                        .ok_or(Error::NotFound)?
                        .clone(),
                )?;
                let perm_any = deserialize_array(
                    buffers
                        .perm_by_pk(m.value_order_perm_pk)
                        .ok_or(Error::NotFound)?
                        .clone(),
                )?;
                let perm = perm_any
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let arr_any = data_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("downcast".into()))?;
                let plen = perm.len();
                if plen == 0 {
                    continue;
                }
                let get = |i: usize| -> $key {
                    let idx = perm.value(i) as usize;
                    <$key>::new(arr_any.value(idx))
                };
                let (lb, ub) = bounds.clone();
                let s = match lb {
                    Bound::Unbounded => 0,
                    Bound::Included(x) => lower_idx_by(0, plen, &<$key>::new(x), &get),
                    Bound::Excluded(x) => upper_idx_by(0, plen, &<$key>::new(x), &get),
                };
                let e = match ub {
                    Bound::Unbounded => plen,
                    Bound::Included(x) => upper_idx_by(0, plen, &<$key>::new(x), &get),
                    Bound::Excluded(x) => lower_idx_by(0, plen, &<$key>::new(x), &get),
                };
                if s >= e {
                    continue;
                }
                let pw = perm.slice(s, e - s);
                let taken_any = compute::take(&data_any, &pw, None)?;
                let a = taken_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("sorted downcast".into()))?
                    .clone();
                arrays.push(a);
            }
            if arrays.is_empty() {
                return Ok(());
            }
            kmerge_coalesced::<$key, _>(
                &arrays,
                &mut |a: &$ArrTy| a.len(),
                &mut |a: &$ArrTy, i: usize| <$key>::new(a.value(i)),
                &mut |c, s, l| visitor.$visit(&arrays[c], s, l),
            );
            Ok(())
        }
    };
}

macro_rules! sorted_with_rids_bounds_impl {
    ($name:ident, $ArrTy:ty, $ty:ty, $visit:ident) => {
        fn $name<P: Pager<Blob = EntryHandle>>(
            _pager: &P,
            metas_val: &[ChunkMetadata],
            metas_rid: &[ChunkMetadata],
            buffers: &SortedChunkBuffersWithRids,
            bounds: (Bound<$ty>, Bound<$ty>),
            visitor: &mut dyn PrimitiveSortedWithRowIdsVisitor,
        ) -> Result<()> {
            if metas_val.len() != metas_rid.len() {
                return Err(Error::Internal(
                    "sorted_with_rids: chunk count mismatch".into(),
                ));
            }
            let mut vals: Vec<$ArrTy> = Vec::with_capacity(metas_val.len());
            let mut rids: Vec<UInt64Array> = Vec::with_capacity(metas_val.len());
            for (mv, mr) in metas_val.iter().zip(metas_rid.iter()) {
                let data_any = deserialize_array(
                    buffers
                        .base()
                        .value_by_pk(mv.chunk_pk)
                        .ok_or(Error::NotFound)?
                        .clone(),
                )?;
                let perm_any = deserialize_array(
                    buffers
                        .base()
                        .perm_by_pk(mv.value_order_perm_pk)
                        .ok_or(Error::NotFound)?
                        .clone(),
                )?;
                let perm = perm_any
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let arr_any = data_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("downcast".into()))?;
                let plen = perm.len();
                if plen == 0 {
                    continue;
                }
                let get = |i: usize| -> $ty {
                    let idx = perm.value(i) as usize;
                    arr_any.value(idx) as $ty
                };
                let (lb, ub) = bounds.clone();
                let s = match lb {
                    Bound::Unbounded => 0,
                    Bound::Included(x) => lower_idx_by(0, plen, &x, &get),
                    Bound::Excluded(x) => upper_idx_by(0, plen, &x, &get),
                };
                let e = match ub {
                    Bound::Unbounded => plen,
                    Bound::Included(x) => upper_idx_by(0, plen, &x, &get),
                    Bound::Excluded(x) => lower_idx_by(0, plen, &x, &get),
                };
                if s >= e {
                    continue;
                }
                let pw = perm.slice(s, e - s);
                let taken_any = compute::take(&data_any, &pw, None)?;
                let a = taken_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("sorted downcast".into()))?
                    .clone();
                vals.push(a);
                let rid_any = deserialize_array(
                    buffers
                        .rid_by_pk(mr.chunk_pk)
                        .ok_or(Error::NotFound)?
                        .clone(),
                )?;
                let taken_rid_any = compute::take(&rid_any, &pw, None)?;
                let ra = taken_rid_any
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("rid downcast".into()))?
                    .clone();
                rids.push(ra);
            }
            if vals.is_empty() {
                return Ok(());
            }
            kmerge_coalesced::<_, _>(
                &vals,
                &mut |a: &$ArrTy| a.len(),
                &mut |a: &$ArrTy, i: usize| a.value(i),
                &mut |c, s, l| visitor.$visit(&vals[c], &rids[c], s, l),
            );
            Ok(())
        }
    };
}

macro_rules! sorted_with_rids_float_bounds_impl {
    ($name:ident, $ArrTy:ty, $scalar:ty, $key:ty, $visit:ident) => {
        fn $name<P: Pager<Blob = EntryHandle>>(
            _pager: &P,
            metas_val: &[ChunkMetadata],
            metas_rid: &[ChunkMetadata],
            buffers: &SortedChunkBuffersWithRids,
            bounds: (Bound<$scalar>, Bound<$scalar>),
            visitor: &mut dyn PrimitiveSortedWithRowIdsVisitor,
        ) -> Result<()> {
            if metas_val.len() != metas_rid.len() {
                return Err(Error::Internal(
                    "sorted_with_rids: chunk count mismatch".into(),
                ));
            }
            let mut vals: Vec<$ArrTy> = Vec::with_capacity(metas_val.len());
            let mut rids: Vec<UInt64Array> = Vec::with_capacity(metas_val.len());
            for (mv, mr) in metas_val.iter().zip(metas_rid.iter()) {
                let data_any = deserialize_array(
                    buffers
                        .base()
                        .value_by_pk(mv.chunk_pk)
                        .ok_or(Error::NotFound)?
                        .clone(),
                )?;
                let perm_any = deserialize_array(
                    buffers
                        .base()
                        .perm_by_pk(mv.value_order_perm_pk)
                        .ok_or(Error::NotFound)?
                        .clone(),
                )?;
                let perm = perm_any
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let arr_any = data_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("downcast".into()))?;
                let plen = perm.len();
                if plen == 0 {
                    continue;
                }
                let get = |i: usize| -> $key {
                    let idx = perm.value(i) as usize;
                    <$key>::new(arr_any.value(idx))
                };
                let (lb, ub) = bounds.clone();
                let s = match lb {
                    Bound::Unbounded => 0,
                    Bound::Included(x) => lower_idx_by(0, plen, &<$key>::new(x), &get),
                    Bound::Excluded(x) => upper_idx_by(0, plen, &<$key>::new(x), &get),
                };
                let e = match ub {
                    Bound::Unbounded => plen,
                    Bound::Included(x) => upper_idx_by(0, plen, &<$key>::new(x), &get),
                    Bound::Excluded(x) => lower_idx_by(0, plen, &<$key>::new(x), &get),
                };
                if s >= e {
                    continue;
                }
                let pw = perm.slice(s, e - s);
                let taken_any = compute::take(&data_any, &pw, None)?;
                let a = taken_any
                    .as_any()
                    .downcast_ref::<$ArrTy>()
                    .ok_or_else(|| Error::Internal("sorted downcast".into()))?
                    .clone();
                vals.push(a);
                let rid_any = deserialize_array(
                    buffers
                        .rid_by_pk(mr.chunk_pk)
                        .ok_or(Error::NotFound)?
                        .clone(),
                )?;
                let taken_rid_any = compute::take(&rid_any, &pw, None)?;
                let ra = taken_rid_any
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("rid downcast".into()))?
                    .clone();
                rids.push(ra);
            }
            if vals.is_empty() {
                return Ok(());
            }
            kmerge_coalesced::<$key, _>(
                &vals,
                &mut |a: &$ArrTy| a.len(),
                &mut |a: &$ArrTy, i: usize| <$key>::new(a.value(i)),
                &mut |c, s, l| visitor.$visit(&vals[c], &rids[c], s, l),
            );
            Ok(())
        }
    };
}
sorted_visit_bounds_impl!(sorted_visit_u64_bounds, UInt64Array, u64, u64_run);
sorted_visit_bounds_impl!(sorted_visit_u32_bounds, UInt32Array, u32, u32_run);
sorted_visit_bounds_impl!(sorted_visit_u16_bounds, UInt16Array, u16, u16_run);
sorted_visit_bounds_impl!(sorted_visit_u8_bounds, UInt8Array, u8, u8_run);
sorted_visit_bounds_impl!(sorted_visit_i64_bounds, Int64Array, i64, i64_run);
sorted_visit_bounds_impl!(sorted_visit_i32_bounds, Int32Array, i32, i32_run);
sorted_visit_bounds_impl!(sorted_visit_i16_bounds, Int16Array, i16, i16_run);
sorted_visit_bounds_impl!(sorted_visit_i8_bounds, Int8Array, i8, i8_run);
sorted_visit_bounds_impl!(sorted_visit_bool_bounds, BooleanArray, bool, bool_run);
sorted_visit_bounds_impl!(sorted_visit_date64_bounds, Date64Array, i64, date64_run);
sorted_visit_bounds_impl!(sorted_visit_date32_bounds, Date32Array, i32, date32_run);
sorted_visit_float_bounds_impl!(
    sorted_visit_f64_bounds,
    Float64Array,
    f64,
    FloatOrd64,
    f64_run
);
sorted_visit_float_bounds_impl!(
    sorted_visit_f32_bounds,
    Float32Array,
    f32,
    FloatOrd32,
    f32_run
);

sorted_with_rids_bounds_impl!(
    sorted_visit_with_rids_u64_bounds,
    UInt64Array,
    u64,
    u64_run_with_rids
);
sorted_with_rids_bounds_impl!(
    sorted_visit_with_rids_u32_bounds,
    UInt32Array,
    u32,
    u32_run_with_rids
);
sorted_with_rids_bounds_impl!(
    sorted_visit_with_rids_u16_bounds,
    UInt16Array,
    u16,
    u16_run_with_rids
);
sorted_with_rids_bounds_impl!(
    sorted_visit_with_rids_u8_bounds,
    UInt8Array,
    u8,
    u8_run_with_rids
);
sorted_with_rids_bounds_impl!(
    sorted_visit_with_rids_i64_bounds,
    Int64Array,
    i64,
    i64_run_with_rids
);
sorted_with_rids_bounds_impl!(
    sorted_visit_with_rids_i32_bounds,
    Int32Array,
    i32,
    i32_run_with_rids
);
sorted_with_rids_bounds_impl!(
    sorted_visit_with_rids_i16_bounds,
    Int16Array,
    i16,
    i16_run_with_rids
);
sorted_with_rids_bounds_impl!(
    sorted_visit_with_rids_i8_bounds,
    Int8Array,
    i8,
    i8_run_with_rids
);
sorted_with_rids_bounds_impl!(
    sorted_visit_with_rids_bool_bounds,
    BooleanArray,
    bool,
    bool_run_with_rids
);
sorted_with_rids_bounds_impl!(
    sorted_visit_with_rids_date64_bounds,
    Date64Array,
    i64,
    date64_run_with_rids
);
sorted_with_rids_bounds_impl!(
    sorted_visit_with_rids_date32_bounds,
    Date32Array,
    i32,
    date32_run_with_rids
);
sorted_with_rids_float_bounds_impl!(
    sorted_visit_with_rids_f64_bounds,
    Float64Array,
    f64,
    FloatOrd64,
    f64_run_with_rids
);
sorted_with_rids_float_bounds_impl!(
    sorted_visit_with_rids_f32_bounds,
    Float32Array,
    f32,
    FloatOrd32,
    f32_run_with_rids
);

fn dispatch_sorted_bounds<P, V>(
    dtype: &DataType,
    pager: &P,
    metas: &[ChunkMetadata],
    buffers: &SortedChunkBuffers,
    ir: &IntRanges,
    visitor: &mut V,
) -> Result<()>
where
    P: Pager<Blob = EntryHandle>,
    V: PrimitiveSortedVisitor,
{
    with_integer_arrow_type!(
        dtype,
        |ArrowTy| <ArrowTy as SortedDispatch>::visit_bounds(pager, metas, buffers, ir, visitor),
        Err(Error::Internal("unsupported sorted dtype (builder)".into())),
    )
}

fn dispatch_sorted_with_rids_bounds<P, V>(
    dtype: &DataType,
    pager: &P,
    metas_val: &[ChunkMetadata],
    metas_rid: &[ChunkMetadata],
    buffers: &SortedChunkBuffersWithRids,
    ir: &IntRanges,
    visitor: &mut V,
) -> Result<()>
where
    P: Pager<Blob = EntryHandle>,
    V: PrimitiveSortedWithRowIdsVisitor,
{
    with_integer_arrow_type!(
        dtype,
        |ArrowTy| {
            <ArrowTy as SortedDispatch>::visit_with_rids_bounds(
                pager, metas_val, metas_rid, buffers, ir, visitor,
            )
        },
        Err(Error::Internal("unsupported sorted dtype (builder)".into())),
    )
}

pub fn range_sorted_dispatch<P, V>(
    store: &ColumnStore<P>,
    field_id: LogicalFieldId,
    opts: ScanOptions,
    ir: IntRanges,
    visitor: &mut V,
) -> Result<()>
where
    P: Pager<Blob = EntryHandle>,
    V: PrimitiveVisitor
        + PrimitiveSortedVisitor
        + PrimitiveWithRowIdsVisitor
        + PrimitiveSortedWithRowIdsVisitor,
{
    // Load descriptor metas and blobs (values, perms, and rids when needed)
    let catalog = store.catalog.read().unwrap();
    let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
    let desc_blob = store
        .pager
        .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
        .pop()
        .and_then(|r| match r {
            GetResult::Raw { bytes, .. } => Some(bytes),
            _ => None,
        })
        .ok_or(Error::NotFound)?;
    let desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
    let mut metas_val: Vec<ChunkMetadata> = Vec::new();
    for m in
        crate::store::descriptor::DescriptorIterator::new(store.pager.as_ref(), desc.head_page_pk)
    {
        let meta = m?;
        if meta.row_count > 0 {
            if meta.value_order_perm_pk == 0 {
                return Err(Error::NotFound);
            }
            metas_val.push(meta);
        }
    }

    let (_rid_desc_opt, metas_rid): (
        Option<crate::store::descriptor::ColumnDescriptor>,
        Vec<ChunkMetadata>,
    ) = if opts.with_row_ids {
        let rid_fid = field_id.with_namespace(Namespace::RowIdShadow);
        let rid_pk = *catalog.map.get(&rid_fid).ok_or(Error::NotFound)?;
        let rid_desc_blob = store
            .pager
            .batch_get(&[BatchGet::Raw { key: rid_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let rid_desc =
            crate::store::descriptor::ColumnDescriptor::from_le_bytes(rid_desc_blob.as_ref());
        let mut mr = Vec::new();
        for m in crate::store::descriptor::DescriptorIterator::new(
            store.pager.as_ref(),
            rid_desc.head_page_pk,
        ) {
            let meta = m?;
            if meta.row_count > 0 {
                mr.push(meta);
            }
        }
        (Some(rid_desc), mr)
    } else {
        (None, Vec::new())
    };
    drop(catalog);

    if metas_val.is_empty() {
        return Ok(());
    }

    if opts.with_row_ids {
        let buffers = load_sorted_buffers_with_rids(store.pager.as_ref(), &metas_val, &metas_rid)?;
        let first_any = deserialize_array(buffers.base().value_handle(0).clone())?;
        dispatch_sorted_with_rids_bounds(
            first_any.data_type(),
            store.pager.as_ref(),
            &metas_val,
            &metas_rid,
            &buffers,
            &ir,
            visitor,
        )
    } else {
        let buffers = load_sorted_buffers(store.pager.as_ref(), &metas_val)?;
        let first_any = deserialize_array(buffers.value_handle(0).clone())?;
        dispatch_sorted_bounds(
            first_any.data_type(),
            store.pager.as_ref(),
            &metas_val,
            &buffers,
            &ir,
            visitor,
        )
    }
}
