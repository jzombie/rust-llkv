//! Primitive column scanners and visitor traits used by ColumnStore.
//!
//! Focus: integer primitives with fast unsorted and sorted scans.
//! - Monomorphized per-type paths (no dynamic dispatch in hot loops)
//! - Batches IO with pager, minimal object churn
//! - Coalesces adjacent items into runs for sorted scans
//! - Optional row-id variants
//!
//! This module is intentionally self-contained so higher-level methods in
//! `store::mod` can delegate without code duplication.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use arrow::array::*;
use arrow::compute;
use arrow::datatypes::DataType;

use rustc_hash::FxHashMap;

use crate::error::{Error, Result};
use crate::serialization::deserialize_array;
use crate::storage::pager::{BatchGet, GetResult, Pager};
use crate::store::descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorIterator};
use crate::types::{LogicalFieldId, PhysicalKey};
use simd_r_drive_entry_handle::EntryHandle;
use super::ColumnStore;

// ---------------------------- Visitor Traits ----------------------------

/// Unsorted primitive visitor: one callback per chunk per type.
pub trait PrimitiveVisitor {
    fn u64_chunk(&mut self, _a: &UInt64Array) {}
    fn u32_chunk(&mut self, _a: &UInt32Array) {}
    fn u16_chunk(&mut self, _a: &UInt16Array) {}
    fn u8_chunk(&mut self, _a: &UInt8Array) {}
    fn i64_chunk(&mut self, _a: &Int64Array) {}
    fn i32_chunk(&mut self, _a: &Int32Array) {}
    fn i16_chunk(&mut self, _a: &Int16Array) {}
    fn i8_chunk(&mut self, _a: &Int8Array) {}
}

/// Unsorted primitive visitor with row ids (u64).
pub trait PrimitiveWithRowIdsVisitor {
    fn u64_chunk_with_rids(&mut self, _v: &UInt64Array, _r: &UInt64Array) {}
    fn u32_chunk_with_rids(&mut self, _v: &UInt32Array, _r: &UInt64Array) {}
    fn u16_chunk_with_rids(&mut self, _v: &UInt16Array, _r: &UInt64Array) {}
    fn u8_chunk_with_rids(&mut self, _v: &UInt8Array, _r: &UInt64Array) {}
    fn i64_chunk_with_rids(&mut self, _v: &Int64Array, _r: &UInt64Array) {}
    fn i32_chunk_with_rids(&mut self, _v: &Int32Array, _r: &UInt64Array) {}
    fn i16_chunk_with_rids(&mut self, _v: &Int16Array, _r: &UInt64Array) {}
    fn i8_chunk_with_rids(&mut self, _v: &Int8Array, _r: &UInt64Array) {}
}

/// Sorted visitor fed with coalesced runs (start,len) within a typed array.
pub trait PrimitiveSortedVisitor {
    fn u64_run(&mut self, _a: &UInt64Array, _start: usize, _len: usize) {}
    fn u32_run(&mut self, _a: &UInt32Array, _start: usize, _len: usize) {}
    fn u16_run(&mut self, _a: &UInt16Array, _start: usize, _len: usize) {}
    fn u8_run(&mut self, _a: &UInt8Array, _start: usize, _len: usize) {}
    fn i64_run(&mut self, _a: &Int64Array, _start: usize, _len: usize) {}
    fn i32_run(&mut self, _a: &Int32Array, _start: usize, _len: usize) {}
    fn i16_run(&mut self, _a: &Int16Array, _start: usize, _len: usize) {}
    fn i8_run(&mut self, _a: &Int8Array, _start: usize, _len: usize) {}
}

/// Sorted visitor with row ids.
pub trait PrimitiveSortedWithRowIdsVisitor {
    fn u64_run_with_rids(
        &mut self,
        _v: &UInt64Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
    }
    fn u32_run_with_rids(
        &mut self,
        _v: &UInt32Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
    }
    fn u16_run_with_rids(
        &mut self,
        _v: &UInt16Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
    }
    fn u8_run_with_rids(&mut self, _v: &UInt8Array, _r: &UInt64Array, _start: usize, _len: usize) {}
    fn i64_run_with_rids(&mut self, _v: &Int64Array, _r: &UInt64Array, _start: usize, _len: usize) {
    }
    fn i32_run_with_rids(&mut self, _v: &Int32Array, _r: &UInt64Array, _start: usize, _len: usize) {
    }
    fn i16_run_with_rids(&mut self, _v: &Int16Array, _r: &UInt64Array, _start: usize, _len: usize) {
    }
    fn i8_run_with_rids(&mut self, _v: &Int8Array, _r: &UInt64Array, _start: usize, _len: usize) {}
}

// --------------------------- Unsorted Scanners ---------------------------

pub fn unsorted_visit<P: Pager<Blob = EntryHandle>, V: PrimitiveVisitor>(
    pager: &P,
    catalog: &FxHashMap<LogicalFieldId, PhysicalKey>,
    field_id: LogicalFieldId,
    visitor: &mut V,
) -> Result<()> {
    let descriptor_pk = *catalog.get(&field_id).ok_or(Error::NotFound)?;
    let desc_blob = pager
        .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
        .pop()
        .and_then(|r| match r {
            GetResult::Raw { bytes, .. } => Some(bytes),
            _ => None,
        })
        .ok_or(Error::NotFound)?;
    let desc = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

    // Gather metas and fetch blobs in one batch.
    let mut metas: Vec<ChunkMetadata> = Vec::new();
    for m in DescriptorIterator::new(pager, desc.head_page_pk) {
        let meta = m?;
        if meta.row_count > 0 {
            metas.push(meta);
        }
    }
    if metas.is_empty() {
        return Ok(());
    }
    let gets: Vec<BatchGet> = metas
        .iter()
        .map(|m| BatchGet::Raw { key: m.chunk_pk })
        .collect();
    let results = pager.batch_get(&gets)?;
    let mut blobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
    for r in results {
        if let GetResult::Raw { key, bytes } = r {
            blobs.insert(key, bytes);
        }
    }

    // Inspect dtype of first chunk to monomorphize the loop.
    let first_any = deserialize_array(
        blobs
            .get(&metas[0].chunk_pk)
            .ok_or(Error::NotFound)?
            .clone(),
    )?;
    match first_any.data_type() {
        DataType::UInt64 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast UInt64".into()))?;
                visitor.u64_chunk(a);
            }
            Ok(())
        }
        DataType::UInt32 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("downcast UInt32".into()))?;
                visitor.u32_chunk(a);
            }
            Ok(())
        }
        DataType::UInt16 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<UInt16Array>()
                    .ok_or_else(|| Error::Internal("downcast UInt16".into()))?;
                visitor.u16_chunk(a);
            }
            Ok(())
        }
        DataType::UInt8 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<UInt8Array>()
                    .ok_or_else(|| Error::Internal("downcast UInt8".into()))?;
                visitor.u8_chunk(a);
            }
            Ok(())
        }
        DataType::Int64 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| Error::Internal("downcast Int64".into()))?;
                visitor.i64_chunk(a);
            }
            Ok(())
        }
        DataType::Int32 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| Error::Internal("downcast Int32".into()))?;
                visitor.i32_chunk(a);
            }
            Ok(())
        }
        DataType::Int16 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<Int16Array>()
                    .ok_or_else(|| Error::Internal("downcast Int16".into()))?;
                visitor.i16_chunk(a);
            }
            Ok(())
        }
        DataType::Int8 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<Int8Array>()
                    .ok_or_else(|| Error::Internal("downcast Int8".into()))?;
                visitor.i8_chunk(a);
            }
            Ok(())
        }
        _ => Err(Error::Internal("unsorted_visit: unsupported dtype".into())),
    }
}

pub fn unsorted_with_row_ids_visit<P: Pager<Blob = EntryHandle>, V: PrimitiveWithRowIdsVisitor>(
    pager: &P,
    catalog: &FxHashMap<LogicalFieldId, PhysicalKey>,
    value_fid: LogicalFieldId,
    rowid_fid: LogicalFieldId,
    visitor: &mut V,
) -> Result<()> {
    let v_pk = *catalog.get(&value_fid).ok_or(Error::NotFound)?;
    let r_pk = *catalog.get(&rowid_fid).ok_or(Error::NotFound)?;

    let v_desc_blob = pager
        .batch_get(&[BatchGet::Raw { key: v_pk }])?
        .pop()
        .and_then(|r| match r {
            GetResult::Raw { bytes, .. } => Some(bytes),
            _ => None,
        })
        .ok_or(Error::NotFound)?;
    let r_desc_blob = pager
        .batch_get(&[BatchGet::Raw { key: r_pk }])?
        .pop()
        .and_then(|r| match r {
            GetResult::Raw { bytes, .. } => Some(bytes),
            _ => None,
        })
        .ok_or(Error::NotFound)?;
    let v_desc = ColumnDescriptor::from_le_bytes(v_desc_blob.as_ref());
    let r_desc = ColumnDescriptor::from_le_bytes(r_desc_blob.as_ref());

    let mut v_metas: Vec<ChunkMetadata> = Vec::new();
    for m in DescriptorIterator::new(pager, v_desc.head_page_pk) {
        let meta = m?;
        if meta.row_count > 0 {
            v_metas.push(meta);
        }
    }
    let mut r_metas: Vec<ChunkMetadata> = Vec::new();
    for m in DescriptorIterator::new(pager, r_desc.head_page_pk) {
        let meta = m?;
        if meta.row_count > 0 {
            r_metas.push(meta);
        }
    }
    if v_metas.len() != r_metas.len() {
        return Err(Error::Internal(
            "unsorted_with_row_ids: chunk count mismatch".into(),
        ));
    }
    if v_metas.is_empty() {
        return Ok(());
    }

    let mut gets: Vec<BatchGet> = Vec::with_capacity(v_metas.len() * 2);
    for (vm, rm) in v_metas.iter().zip(r_metas.iter()) {
        gets.push(BatchGet::Raw { key: vm.chunk_pk });
        gets.push(BatchGet::Raw { key: rm.chunk_pk });
    }
    let results = pager.batch_get(&gets)?;
    let mut vals_blobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
    let mut rids_blobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
    for r in results {
        if let GetResult::Raw { key, bytes } = r {
            if r_metas.iter().any(|m| m.chunk_pk == key) {
                rids_blobs.insert(key, bytes);
            } else {
                vals_blobs.insert(key, bytes);
            }
        }
    }

    let first_any = deserialize_array(
        vals_blobs
            .get(&v_metas[0].chunk_pk)
            .ok_or(Error::NotFound)?
            .clone(),
    )?;
    let r_first_any = deserialize_array(
        rids_blobs
            .get(&r_metas[0].chunk_pk)
            .ok_or(Error::NotFound)?
            .clone(),
    )?;
    let _rids = r_first_any
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| Error::Internal("row_id array must be UInt64".into()))?;
    match first_any.data_type() {
        DataType::UInt64 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast u64".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.u64_chunk_with_rids(v, r);
            }
            Ok(())
        }
        DataType::UInt32 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("downcast u32".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.u32_chunk_with_rids(v, r);
            }
            Ok(())
        }
        DataType::UInt16 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<UInt16Array>()
                    .ok_or_else(|| Error::Internal("downcast u16".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.u16_chunk_with_rids(v, r);
            }
            Ok(())
        }
        DataType::UInt8 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<UInt8Array>()
                    .ok_or_else(|| Error::Internal("downcast u8".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.u8_chunk_with_rids(v, r);
            }
            Ok(())
        }
        DataType::Int64 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| Error::Internal("downcast i64".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.i64_chunk_with_rids(v, r);
            }
            Ok(())
        }
        DataType::Int32 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| Error::Internal("downcast i32".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.i32_chunk_with_rids(v, r);
            }
            Ok(())
        }
        DataType::Int16 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<Int16Array>()
                    .ok_or_else(|| Error::Internal("downcast i16".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.i16_chunk_with_rids(v, r);
            }
            Ok(())
        }
        DataType::Int8 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<Int8Array>()
                    .ok_or_else(|| Error::Internal("downcast i8".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.i8_chunk_with_rids(v, r);
            }
            Ok(())
        }
        _ => Err(Error::Internal(
            "unsorted_with_row_ids: unsupported dtype".into(),
        )),
    }
}

// --------------------------- Sorted Scanners ----------------------------

/// Generic k-way merge over sorted per-chunk arrays, coalescing runs.
/// `get` fetches the T value at index for an array, and `emit` sends runs.
fn kmerge_coalesced<T, A, FLen, FGet, FEmit>(
    arrays: &[A],
    mut len_of: FLen,
    mut get: FGet,
    mut emit: FEmit,
) where
    T: Ord + Copy,
    FLen: FnMut(&A) -> usize,
    FGet: FnMut(&A, usize) -> T,
    FEmit: FnMut(usize, usize, usize), // (chunk_idx, start, len)
{
    #[derive(Clone, Copy, Debug)]
    struct H<T> {
        v: T,
        c: usize,
        i: usize,
    }
    impl<T: Ord> PartialEq for H<T> {
        fn eq(&self, o: &Self) -> bool {
            self.v == o.v && self.c == o.c && self.i == o.i
        }
    }
    impl<T: Ord> Eq for H<T> {}
    impl<T: Ord> PartialOrd for H<T> {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    // Max-heap by value; break ties by chunk to keep deterministic.
    impl<T: Ord> Ord for H<T> {
        fn cmp(&self, o: &Self) -> Ordering {
            // Reverse value ordering to get a min-heap behavior via BinaryHeap
            o.v.cmp(&self.v).then_with(|| o.c.cmp(&self.c))
        }
    }

    let mut heap: BinaryHeap<H<T>> = BinaryHeap::new();
    for (ci, a) in arrays.iter().enumerate() {
        let al = len_of(a);
        if al > 0 {
            heap.push(H {
                v: get(a, 0),
                c: ci,
                i: 0,
            });
        }
    }

    while let Some(h) = heap.pop() {
        let c = h.c;
        let a = &arrays[c];
        let s = h.i;
        let mut e = s + 1;
        let thr = heap.peek().map(|x| x.v);
        let al = len_of(a);
        if let Some(t) = thr {
            while e < al && get(a, e) <= t {
                e += 1;
            }
        } else {
            e = al;
        }
        emit(c, s, e - s);
        if e < al {
            heap.push(H {
                v: get(a, e),
                c,
                i: e,
            });
        }
    }
}

/// Reverse (descending) k-way merge over sorted per-chunk arrays, coalescing runs.
fn kmerge_coalesced_rev<T, A, FLen, FGet, FEmit>(
    arrays: &[A],
    mut len_of: FLen,
    mut get: FGet,
    mut emit: FEmit,
) where
    T: Ord + Copy,
    FLen: FnMut(&A) -> usize,
    FGet: FnMut(&A, usize) -> T,
    FEmit: FnMut(usize, usize, usize), // (chunk_idx, start, len) but start..start+len iterates descending via get
{
    #[derive(Clone, Copy, Debug)]
    struct H<T> { v: T, c: usize, i: usize }
    impl<T: Ord> PartialEq for H<T> { fn eq(&self, o: &Self) -> bool { self.v == o.v && self.c == o.c && self.i == o.i } }
    impl<T: Ord> Eq for H<T> {}
    impl<T: Ord> PartialOrd for H<T> { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
    // Max-heap by value (natural ordering)
    impl<T: Ord> Ord for H<T> { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.v.cmp(&o.v).then_with(|| self.c.cmp(&o.c)) } }

    let mut heap: BinaryHeap<H<T>> = BinaryHeap::new();
    for (ci, a) in arrays.iter().enumerate() {
        let al = len_of(a);
        if al > 0 { let idx = al - 1; heap.push(H { v: get(a, idx), c: ci, i: idx }); }
    }

    while let Some(h) = heap.pop() {
        let c = h.c;
        let a = &arrays[c];
        let mut e = h.i; // inclusive end
        let mut s = e;   // inclusive start, will decrease
        let thr = heap.peek().map(|x| x.v);
        if let Some(t) = thr {
            while s > 0 {
                let p = s - 1;
                if get(a, p) >= t { s = p; } else { break; }
            }
        } else {
            // drain remaining
            s = 0;
        }
        // Emit as (start,len) using ascending indices; caller can iterate descending within the run
        emit(c, s, e - s + 1);
        if s > 0 {
            let next = s - 1;
            heap.push(H { v: get(a, next), c, i: next });
        }
    }
}

macro_rules! sorted_visit_impl {
    ($name:ident, $name_rev:ident, $ArrTy:ty, $visit:ident) => {
        pub fn $name<P: Pager<Blob = EntryHandle>, V: PrimitiveSortedVisitor>(
            pager: &P,
            metas: &[ChunkMetadata],
            blobs: &FxHashMap<PhysicalKey, EntryHandle>,
            visitor: &mut V,
        ) -> Result<()> {
            // Materialize sorted arrays per chunk by applying the permutation.
            let mut arrays: Vec<$ArrTy> = Vec::with_capacity(metas.len());
            for m in metas {
                let data_any =
                    deserialize_array(blobs.get(&m.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                let perm_any = deserialize_array(
                    blobs
                        .get(&m.value_order_perm_pk)
                        .ok_or(Error::NotFound)?
                        .clone(),
                )?;
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

            // K-way merge with coalesced runs.
            // Implement len/get inline for the concrete array type (monomorphized).
            if arrays.is_empty() {
                return Ok(());
            }
            kmerge_coalesced::<_, _, _, _, _>(
                &arrays,
                |a: &$ArrTy| a.len(),
                |a: &$ArrTy, i: usize| a.value(i),
                |c, s, l| visitor.$visit(&arrays[c], s, l),
            );
            Ok(())
        }
        pub fn $name_rev<P: Pager<Blob = EntryHandle>, V: PrimitiveSortedVisitor>(
            pager: &P,
            metas: &[ChunkMetadata],
            blobs: &FxHashMap<PhysicalKey, EntryHandle>,
            visitor: &mut V,
        ) -> Result<()> {
            // Materialize sorted arrays per chunk by applying the permutation.
            let mut arrays: Vec<$ArrTy> = Vec::with_capacity(metas.len());
            for m in metas {
                let data_any = deserialize_array(blobs.get(&m.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                let perm_any = deserialize_array(blobs.get(&m.value_order_perm_pk).ok_or(Error::NotFound)?.clone())?;
                let perm = perm_any.as_any().downcast_ref::<UInt32Array>().ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let taken_any = compute::take(&data_any, perm, None)?;
                let arr = taken_any.as_any().downcast_ref::<$ArrTy>().ok_or_else(|| Error::Internal("sorted downcast".into()))?.clone();
                arrays.push(arr);
            }
            if arrays.is_empty() { return Ok(()); }
            kmerge_coalesced_rev::<_, _, _, _, _>(
                &arrays,
                |a: &$ArrTy| a.len(),
                |a: &$ArrTy, i: usize| a.value(i),
                |c, s, l| visitor.$visit(&arrays[c], s, l),
            );
            Ok(())
        }
    };
}

sorted_visit_impl!(sorted_visit_u64, sorted_visit_u64_rev, UInt64Array, u64_run);
sorted_visit_impl!(sorted_visit_u32, sorted_visit_u32_rev, UInt32Array, u32_run);
sorted_visit_impl!(sorted_visit_u16, sorted_visit_u16_rev, UInt16Array, u16_run);
sorted_visit_impl!(sorted_visit_u8,  sorted_visit_u8_rev,  UInt8Array,  u8_run);
sorted_visit_impl!(sorted_visit_i64, sorted_visit_i64_rev, Int64Array,  i64_run);
sorted_visit_impl!(sorted_visit_i32, sorted_visit_i32_rev, Int32Array,  i32_run);
sorted_visit_impl!(sorted_visit_i16, sorted_visit_i16_rev, Int16Array,  i16_run);
sorted_visit_impl!(sorted_visit_i8,  sorted_visit_i8_rev,  Int8Array,   i8_run);

// Note: A sorted-with-row-ids variant can be added similarly if needed.

// ------------------------ Options + Builders ----------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct ScanOptions {
    pub sorted: bool,
    pub reverse: bool,
    pub with_row_ids: bool,
    /// When `with_row_ids` is true, the LogicalFieldId of the row-id column to pair with.
    pub row_id_field: Option<LogicalFieldId>,
}

impl<P> ColumnStore<P>
where
    P: Pager<Blob = EntryHandle>,
{
    /// Unsorted scan with typed visitor callbacks per chunk.
    fn scan_visit<V: crate::store::scan::PrimitiveVisitor>(
        &self,
        field_id: LogicalFieldId,
        visitor: &mut V,
    ) -> Result<()> {
        let catalog = self.catalog.read().unwrap();
        crate::store::scan::unsorted_visit(self.pager.as_ref(), &catalog.map, field_id, visitor)
    }

    /// Convenience: sorted scan with closures over coalesced runs.
    /// Sorted scan with typed visitor callbacks over coalesced runs.
    fn scan_sorted_visit<V: crate::store::scan::PrimitiveSortedVisitor>(
        &self,
        field_id: LogicalFieldId,
        visitor: &mut V,
    ) -> Result<()> {
        let catalog = self.catalog.read().unwrap();
        let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
        let desc_blob = self
            .pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
            .and_then(|r| match r { GetResult::Raw { bytes, .. } => Some(bytes), _ => None })
            .ok_or(Error::NotFound)?;
        let desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
        drop(catalog);

        let mut metas: Vec<crate::store::descriptor::ChunkMetadata> = Vec::new();
        for m in crate::store::descriptor::DescriptorIterator::new(self.pager.as_ref(), desc.head_page_pk) {
            let meta = m?; if meta.row_count == 0 { continue; }
            if meta.value_order_perm_pk == 0 { return Err(Error::NotFound); }
            metas.push(meta);
        }
        if metas.is_empty() { return Ok(()); }
        let mut gets: Vec<BatchGet> = Vec::with_capacity(metas.len()*2);
        for m in &metas { gets.push(BatchGet::Raw{key:m.chunk_pk}); gets.push(BatchGet::Raw{key:m.value_order_perm_pk}); }
        let results = self.pager.batch_get(&gets)?;
        let mut blobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
        for r in results { if let GetResult::Raw{key,bytes} = r { blobs.insert(key, bytes); } }

        let first_any = crate::serialization::deserialize_array(blobs.get(&metas[0].chunk_pk).ok_or(Error::NotFound)?.clone())?;
        match first_any.data_type() {
            DataType::UInt64 => crate::store::scan::sorted_visit_u64(self.pager.as_ref(), &metas, &blobs, visitor),
            DataType::UInt32 => crate::store::scan::sorted_visit_u32(self.pager.as_ref(), &metas, &blobs, visitor),
            DataType::UInt16 => crate::store::scan::sorted_visit_u16(self.pager.as_ref(), &metas, &blobs, visitor),
            DataType::UInt8  => crate::store::scan::sorted_visit_u8 (self.pager.as_ref(), &metas, &blobs, visitor),
            DataType::Int64  => crate::store::scan::sorted_visit_i64(self.pager.as_ref(), &metas, &blobs, visitor),
            DataType::Int32  => crate::store::scan::sorted_visit_i32(self.pager.as_ref(), &metas, &blobs, visitor),
            DataType::Int16  => crate::store::scan::sorted_visit_i16(self.pager.as_ref(), &metas, &blobs, visitor),
            DataType::Int8   => crate::store::scan::sorted_visit_i8 (self.pager.as_ref(), &metas, &blobs, visitor),
            _ => Err(Error::Internal("unsupported sorted dtype".into())),
        }
    }

    /// Sorted scan in reverse (descending) with typed visitor callbacks.
    fn scan_sorted_visit_reverse<V: crate::store::scan::PrimitiveSortedVisitor>(
        &self,
        field_id: LogicalFieldId,
        visitor: &mut V,
    ) -> Result<()> {
        let catalog = self.catalog.read().unwrap();
        let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
        let desc_blob = self
            .pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
            .and_then(|r| match r { GetResult::Raw { bytes, .. } => Some(bytes), _ => None })
            .ok_or(Error::NotFound)?;
        let desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
        drop(catalog);

        let mut metas: Vec<crate::store::descriptor::ChunkMetadata> = Vec::new();
        for m in crate::store::descriptor::DescriptorIterator::new(self.pager.as_ref(), desc.head_page_pk) {
            let meta = m?; if meta.row_count == 0 { continue; }
            if meta.value_order_perm_pk == 0 { return Err(Error::NotFound); }
            metas.push(meta);
        }
        if metas.is_empty() { return Ok(()); }
        let mut gets: Vec<BatchGet> = Vec::with_capacity(metas.len()*2);
        for m in &metas { gets.push(BatchGet::Raw{key:m.chunk_pk}); gets.push(BatchGet::Raw{key:m.value_order_perm_pk}); }
        let results = self.pager.batch_get(&gets)?;
        let mut blobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
        for r in results { if let GetResult::Raw{key,bytes} = r { blobs.insert(key, bytes); } }

        let first_any = crate::serialization::deserialize_array(blobs.get(&metas[0].chunk_pk).ok_or(Error::NotFound)?.clone())?;
        match first_any.data_type() {
            DataType::UInt64 => crate::store::scan::sorted_visit_u64_rev(self.pager.as_ref(), &metas, &blobs, visitor),
            DataType::UInt32 => crate::store::scan::sorted_visit_u32_rev(self.pager.as_ref(), &metas, &blobs, visitor),
            DataType::UInt16 => crate::store::scan::sorted_visit_u16_rev(self.pager.as_ref(), &metas, &blobs, visitor),
            DataType::UInt8  => crate::store::scan::sorted_visit_u8_rev (self.pager.as_ref(), &metas, &blobs, visitor),
            DataType::Int64  => crate::store::scan::sorted_visit_i64_rev(self.pager.as_ref(), &metas, &blobs, visitor),
            DataType::Int32  => crate::store::scan::sorted_visit_i32_rev(self.pager.as_ref(), &metas, &blobs, visitor),
            DataType::Int16  => crate::store::scan::sorted_visit_i16_rev(self.pager.as_ref(), &metas, &blobs, visitor),
            DataType::Int8   => crate::store::scan::sorted_visit_i8_rev (self.pager.as_ref(), &metas, &blobs, visitor),
            _ => Err(Error::Internal("unsupported sorted dtype".into())),
        }
    }

    /// Unified scan entrypoint configured by ScanOptions.
    /// Requires `V` to implement both unsorted and sorted visitor traits; methods are no-ops by default.
    pub fn scan<V>(&self, field_id: LogicalFieldId, opts: ScanOptions, visitor: &mut V) -> Result<()>
    where
        V: crate::store::scan::PrimitiveVisitor
            + crate::store::scan::PrimitiveSortedVisitor
            + crate::store::scan::PrimitiveWithRowIdsVisitor
            + crate::store::scan::PrimitiveSortedWithRowIdsVisitor,
    {
        if !opts.sorted {
            if opts.with_row_ids {
                let row_fid = opts
                    .row_id_field
                    .ok_or_else(|| Error::Internal("row_id field id required when with_row_ids=true".into()))?;
                let catalog = self.catalog.read().unwrap();
                return crate::store::scan::unsorted_with_row_ids_visit(
                    self.pager.as_ref(),
                    &catalog.map,
                    field_id,
                    row_fid,
                    visitor,
                );
            }
            return self.scan_visit(field_id, visitor);
        }

        if opts.reverse {
            self.scan_sorted_visit_reverse(field_id, visitor)
        } else {
            self.scan_sorted_visit(field_id, visitor)
        }
    }
}
