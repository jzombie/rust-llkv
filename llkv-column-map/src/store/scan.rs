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

macro_rules! sorted_with_rids_impl {
    ($name:ident, $name_rev:ident, $ArrTy:ty, $visit:ident) => {
        pub fn $name<P: Pager<Blob = EntryHandle>, V: PrimitiveSortedWithRowIdsVisitor>(
            pager: &P,
            metas_val: &[ChunkMetadata],
            metas_rid: &[ChunkMetadata],
            vblobs: &FxHashMap<PhysicalKey, EntryHandle>,
            rblobs: &FxHashMap<PhysicalKey, EntryHandle>,
            visitor: &mut V,
        ) -> Result<()> {
            if metas_val.len() != metas_rid.len() { return Err(Error::Internal("sorted_with_rids: chunk count mismatch".into())); }
            let mut vals: Vec<$ArrTy> = Vec::with_capacity(metas_val.len());
            let mut rids: Vec<UInt64Array> = Vec::with_capacity(metas_val.len());
            for (mv, mr) in metas_val.iter().zip(metas_rid.iter()) {
                let data_any = deserialize_array(vblobs.get(&mv.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                let perm_any = deserialize_array(vblobs.get(&mv.value_order_perm_pk).ok_or(Error::NotFound)?.clone())?;
                let perm = perm_any.as_any().downcast_ref::<UInt32Array>().ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let taken_any = compute::take(&data_any, perm, None)?;
                let arr = taken_any.as_any().downcast_ref::<$ArrTy>().ok_or_else(|| Error::Internal("sorted downcast".into()))?.clone();
                vals.push(arr);
                let rid_any = deserialize_array(rblobs.get(&mr.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                let taken_rid_any = compute::take(&rid_any, perm, None)?;
                let rid = taken_rid_any.as_any().downcast_ref::<UInt64Array>().ok_or_else(|| Error::Internal("row_id downcast".into()))?.clone();
                rids.push(rid);
            }
            if vals.is_empty() { return Ok(()); }
            kmerge_coalesced::<_, _, _, _, _>(
                &vals,
                |a: &$ArrTy| a.len(),
                |a: &$ArrTy, i: usize| a.value(i),
                |c, s, l| visitor.$visit(&vals[c], &rids[c], s, l),
            );
            Ok(())
        }
        pub fn $name_rev<P: Pager<Blob = EntryHandle>, V: PrimitiveSortedWithRowIdsVisitor>(
            pager: &P,
            metas_val: &[ChunkMetadata],
            metas_rid: &[ChunkMetadata],
            vblobs: &FxHashMap<PhysicalKey, EntryHandle>,
            rblobs: &FxHashMap<PhysicalKey, EntryHandle>,
            visitor: &mut V,
        ) -> Result<()> {
            if metas_val.len() != metas_rid.len() { return Err(Error::Internal("sorted_with_rids: chunk count mismatch".into())); }
            let mut vals: Vec<$ArrTy> = Vec::with_capacity(metas_val.len());
            let mut rids: Vec<UInt64Array> = Vec::with_capacity(metas_val.len());
            for (mv, mr) in metas_val.iter().zip(metas_rid.iter()) {
                let data_any = deserialize_array(vblobs.get(&mv.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                let perm_any = deserialize_array(vblobs.get(&mv.value_order_perm_pk).ok_or(Error::NotFound)?.clone())?;
                let perm = perm_any.as_any().downcast_ref::<UInt32Array>().ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let taken_any = compute::take(&data_any, perm, None)?;
                let arr = taken_any.as_any().downcast_ref::<$ArrTy>().ok_or_else(|| Error::Internal("sorted downcast".into()))?.clone();
                vals.push(arr);
                let rid_any = deserialize_array(rblobs.get(&mr.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                let taken_rid_any = compute::take(&rid_any, perm, None)?;
                let rid = taken_rid_any.as_any().downcast_ref::<UInt64Array>().ok_or_else(|| Error::Internal("row_id downcast".into()))?.clone();
                rids.push(rid);
            }
            if vals.is_empty() { return Ok(()); }
            kmerge_coalesced_rev::<_, _, _, _, _>(
                &vals,
                |a: &$ArrTy| a.len(),
                |a: &$ArrTy, i: usize| a.value(i),
                |c, s, l| visitor.$visit(&vals[c], &rids[c], s, l),
            );
            Ok(())
        }
    };
}

sorted_with_rids_impl!(sorted_visit_with_rids_u64, sorted_visit_with_rids_u64_rev, UInt64Array, u64_run_with_rids);
sorted_with_rids_impl!(sorted_visit_with_rids_u32, sorted_visit_with_rids_u32_rev, UInt32Array, u32_run_with_rids);
sorted_with_rids_impl!(sorted_visit_with_rids_u16, sorted_visit_with_rids_u16_rev, UInt16Array, u16_run_with_rids);
sorted_with_rids_impl!(sorted_visit_with_rids_u8,  sorted_visit_with_rids_u8_rev,  UInt8Array,  u8_run_with_rids);
sorted_with_rids_impl!(sorted_visit_with_rids_i64, sorted_visit_with_rids_i64_rev, Int64Array,  i64_run_with_rids);
sorted_with_rids_impl!(sorted_visit_with_rids_i32, sorted_visit_with_rids_i32_rev, Int32Array,  i32_run_with_rids);
sorted_with_rids_impl!(sorted_visit_with_rids_i16, sorted_visit_with_rids_i16_rev, Int16Array,  i16_run_with_rids);
sorted_with_rids_impl!(sorted_visit_with_rids_i8,  sorted_visit_with_rids_i8_rev,  Int8Array,   i8_run_with_rids);

// ------------------------ Options + Builders ----------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct ScanOptions {
    pub sorted: bool,
    pub reverse: bool,
    pub with_row_ids: bool,
    /// When `with_row_ids` is true, the LogicalFieldId of the row-id column to pair with.
    pub row_id_field: Option<LogicalFieldId>,
}

// ----------------------------- Scan Builder ------------------------------

use std::ops::{Bound, RangeBounds};

#[derive(Default, Clone, Copy)]
struct IntRanges {
    u64_r: Option<(Bound<u64>, Bound<u64>)>,
    u32_r: Option<(Bound<u32>, Bound<u32>)>,
    u16_r: Option<(Bound<u16>, Bound<u16>)>,
    u8_r: Option<(Bound<u8>, Bound<u8>)>,
    i64_r: Option<(Bound<i64>, Bound<i64>)>,
    i32_r: Option<(Bound<i32>, Bound<i32>)>,
    i16_r: Option<(Bound<i16>, Bound<i16>)>,
    i8_r: Option<(Bound<i8>, Bound<i8>)>,
}

pub struct ScanBuilder<'a, P: Pager<Blob = EntryHandle>> {
    store: &'a ColumnStore<P>,
    field_id: LogicalFieldId,
    opts: ScanOptions,
    ir: IntRanges,
}

impl<'a, P> ScanBuilder<'a, P>
where
    P: Pager<Blob = EntryHandle>,
{
    pub fn new(store: &'a ColumnStore<P>, field_id: LogicalFieldId) -> Self {
        Self { store, field_id, opts: ScanOptions::default(), ir: IntRanges::default() }
    }
    pub fn options(mut self, opts: ScanOptions) -> Self { self.opts = opts; self }
    pub fn with_row_ids(mut self, row_id_field: LogicalFieldId) -> Self {
        self.opts.with_row_ids = true;
        self.opts.row_id_field = Some(row_id_field);
        self
    }
    pub fn sorted(mut self, sorted: bool) -> Self { self.opts.sorted = sorted; self }
    pub fn reverse(mut self, reverse: bool) -> Self { self.opts.reverse = reverse; self }

    pub fn range_u64<R: RangeBounds<u64>>(mut self, r: R) -> Self { self.ir.u64_r = Some((r.start_bound().cloned(), r.end_bound().cloned())); self }
    pub fn range_u32<R: RangeBounds<u32>>(mut self, r: R) -> Self { self.ir.u32_r = Some((r.start_bound().cloned(), r.end_bound().cloned())); self }
    pub fn range_u16<R: RangeBounds<u16>>(mut self, r: R) -> Self { self.ir.u16_r = Some((r.start_bound().cloned(), r.end_bound().cloned())); self }
    pub fn range_u8<R: RangeBounds<u8>>(mut self, r: R) -> Self { self.ir.u8_r = Some((r.start_bound().cloned(), r.end_bound().cloned())); self }
    pub fn range_i64<R: RangeBounds<i64>>(mut self, r: R) -> Self { self.ir.i64_r = Some((r.start_bound().cloned(), r.end_bound().cloned())); self }
    pub fn range_i32<R: RangeBounds<i32>>(mut self, r: R) -> Self { self.ir.i32_r = Some((r.start_bound().cloned(), r.end_bound().cloned())); self }
    pub fn range_i16<R: RangeBounds<i16>>(mut self, r: R) -> Self { self.ir.i16_r = Some((r.start_bound().cloned(), r.end_bound().cloned())); self }
    pub fn range_i8<R: RangeBounds<i8>>(mut self, r: R) -> Self { self.ir.i8_r = Some((r.start_bound().cloned(), r.end_bound().cloned())); self }

    pub fn run<V>(self, visitor: &mut V) -> Result<()>
    where
        V: crate::store::scan::PrimitiveVisitor
            + crate::store::scan::PrimitiveSortedVisitor
            + crate::store::scan::PrimitiveWithRowIdsVisitor
            + crate::store::scan::PrimitiveSortedWithRowIdsVisitor,
    {
        // If no ranges configured, delegate directly.
        let has_ranges = self.ir.u64_r.is_some()
            || self.ir.u32_r.is_some()
            || self.ir.u16_r.is_some()
            || self.ir.u8_r.is_some()
            || self.ir.i64_r.is_some()
            || self.ir.i32_r.is_some()
            || self.ir.i16_r.is_some()
            || self.ir.i8_r.is_some();
        if !has_ranges {
            return self.store.scan(self.field_id, self.opts, visitor);
        }
        // If ranges configured and sorted path requested, handle inside the builder
        // with per-chunk pre-windowing to avoid per-run trimming overhead.
        if self.opts.sorted {
            return range_sorted_dispatch(self.store, self.field_id, self.opts, self.ir, visitor);
        }

        // Range-filtering adapter for unsorted runs. (Pass-through today.)
        struct RangeAdapter<'v, V> {
            inner: &'v mut V,
            ir: IntRanges,
        }
        impl<'v, V> crate::store::scan::PrimitiveVisitor for RangeAdapter<'v, V>
        where
            V: crate::store::scan::PrimitiveVisitor,
        {
            fn u64_chunk(&mut self, a: &UInt64Array) { self.inner.u64_chunk(a) }
            fn u32_chunk(&mut self, a: &UInt32Array) { self.inner.u32_chunk(a) }
            fn u16_chunk(&mut self, a: &UInt16Array) { self.inner.u16_chunk(a) }
            fn u8_chunk(&mut self, a: &UInt8Array) { self.inner.u8_chunk(a) }
            fn i64_chunk(&mut self, a: &Int64Array) { self.inner.i64_chunk(a) }
            fn i32_chunk(&mut self, a: &Int32Array) { self.inner.i32_chunk(a) }
            fn i16_chunk(&mut self, a: &Int16Array) { self.inner.i16_chunk(a) }
            fn i8_chunk(&mut self, a: &Int8Array) { self.inner.i8_chunk(a) }
        }
        impl<'v, V> crate::store::scan::PrimitiveWithRowIdsVisitor for RangeAdapter<'v, V>
        where
            V: crate::store::scan::PrimitiveWithRowIdsVisitor,
        {
            fn u64_chunk_with_rids(&mut self, v: &UInt64Array, r: &UInt64Array) { self.inner.u64_chunk_with_rids(v, r) }
            fn u32_chunk_with_rids(&mut self, v: &UInt32Array, r: &UInt64Array) { self.inner.u32_chunk_with_rids(v, r) }
            fn u16_chunk_with_rids(&mut self, v: &UInt16Array, r: &UInt64Array) { self.inner.u16_chunk_with_rids(v, r) }
            fn u8_chunk_with_rids(&mut self, v: &UInt8Array, r: &UInt64Array) { self.inner.u8_chunk_with_rids(v, r) }
            fn i64_chunk_with_rids(&mut self, v: &Int64Array, r: &UInt64Array) { self.inner.i64_chunk_with_rids(v, r) }
            fn i32_chunk_with_rids(&mut self, v: &Int32Array, r: &UInt64Array) { self.inner.i32_chunk_with_rids(v, r) }
            fn i16_chunk_with_rids(&mut self, v: &Int16Array, r: &UInt64Array) { self.inner.i16_chunk_with_rids(v, r) }
            fn i8_chunk_with_rids(&mut self, v: &Int8Array, r: &UInt64Array) { self.inner.i8_chunk_with_rids(v, r) }
        }

        // Binary search helpers for sorted runs
        #[inline]
        fn lower_idx<T: Ord, F: Fn(usize) -> T>(mut lo: usize, mut hi: usize, pred: &T, get: F) -> usize {
            while lo < hi {
                let mid = (lo + hi) >> 1;
                if get(mid) < *pred { lo = mid + 1; } else { hi = mid; }
            }
            lo
        }
        #[inline]
        fn upper_idx<T: Ord, F: Fn(usize) -> T>(mut lo: usize, mut hi: usize, pred: &T, get: F) -> usize {
            while lo < hi {
                let mid = (lo + hi) >> 1;
                if get(mid) <= *pred { lo = mid + 1; } else { hi = mid; }
            }
            lo
        }

        impl<'v, V> crate::store::scan::PrimitiveSortedVisitor for RangeAdapter<'v, V>
        where
            V: crate::store::scan::PrimitiveSortedVisitor,
        {
            fn u64_run(&mut self, a: &UInt64Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.u64_r {
                    let start = match lb { Bound::Unbounded => s, Bound::Included(x) => lower_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => upper_idx(s, s+l, &x, |i| a.value(i)) };
                    let end = match ub { Bound::Unbounded => s + l, Bound::Included(x) => upper_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => lower_idx(s, s+l, &x, |i| a.value(i)) };
                    if start < end { self.inner.u64_run(a, start, end - start); }
                } else { self.inner.u64_run(a, s, l); }
            }
            fn u32_run(&mut self, a: &UInt32Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.u32_r {
                    let start = match lb { Bound::Unbounded => s, Bound::Included(x) => lower_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => upper_idx(s, s+l, &x, |i| a.value(i)) };
                    let end = match ub { Bound::Unbounded => s + l, Bound::Included(x) => upper_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => lower_idx(s, s+l, &x, |i| a.value(i)) };
                    if start < end { self.inner.u32_run(a, start, end - start); }
                } else { self.inner.u32_run(a, s, l); }
            }
            fn u16_run(&mut self, a: &UInt16Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.u16_r {
                    let start = match lb { Bound::Unbounded => s, Bound::Included(x) => lower_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => upper_idx(s, s+l, &x, |i| a.value(i)) };
                    let end = match ub { Bound::Unbounded => s + l, Bound::Included(x) => upper_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => lower_idx(s, s+l, &x, |i| a.value(i)) };
                    if start < end { self.inner.u16_run(a, start, end - start); }
                } else { self.inner.u16_run(a, s, l); }
            }
            fn u8_run(&mut self, a: &UInt8Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.u8_r {
                    let start = match lb { Bound::Unbounded => s, Bound::Included(x) => lower_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => upper_idx(s, s+l, &x, |i| a.value(i)) };
                    let end = match ub { Bound::Unbounded => s + l, Bound::Included(x) => upper_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => lower_idx(s, s+l, &x, |i| a.value(i)) };
                    if start < end { self.inner.u8_run(a, start, end - start); }
                } else { self.inner.u8_run(a, s, l); }
            }
            fn i64_run(&mut self, a: &Int64Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.i64_r {
                    let start = match lb { Bound::Unbounded => s, Bound::Included(x) => lower_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => upper_idx(s, s+l, &x, |i| a.value(i)) };
                    let end = match ub { Bound::Unbounded => s + l, Bound::Included(x) => upper_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => lower_idx(s, s+l, &x, |i| a.value(i)) };
                    if start < end { self.inner.i64_run(a, start, end - start); }
                } else { self.inner.i64_run(a, s, l); }
            }
            fn i32_run(&mut self, a: &Int32Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.i32_r {
                    let start = match lb { Bound::Unbounded => s, Bound::Included(x) => lower_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => upper_idx(s, s+l, &x, |i| a.value(i)) };
                    let end = match ub { Bound::Unbounded => s + l, Bound::Included(x) => upper_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => lower_idx(s, s+l, &x, |i| a.value(i)) };
                    if start < end { self.inner.i32_run(a, start, end - start); }
                } else { self.inner.i32_run(a, s, l); }
            }
            fn i16_run(&mut self, a: &Int16Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.i16_r {
                    let start = match lb { Bound::Unbounded => s, Bound::Included(x) => lower_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => upper_idx(s, s+l, &x, |i| a.value(i)) };
                    let end = match ub { Bound::Unbounded => s + l, Bound::Included(x) => upper_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => lower_idx(s, s+l, &x, |i| a.value(i)) };
                    if start < end { self.inner.i16_run(a, start, end - start); }
                } else { self.inner.i16_run(a, s, l); }
            }
            fn i8_run(&mut self, a: &Int8Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.i8_r {
                    let start = match lb { Bound::Unbounded => s, Bound::Included(x) => lower_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => upper_idx(s, s+l, &x, |i| a.value(i)) };
                    let end = match ub { Bound::Unbounded => s + l, Bound::Included(x) => upper_idx(s, s+l, &x, |i| a.value(i)), Bound::Excluded(x) => lower_idx(s, s+l, &x, |i| a.value(i)) };
                    if start < end { self.inner.i8_run(a, start, end - start); }
                } else { self.inner.i8_run(a, s, l); }
            }
        }
        impl<'v, V> crate::store::scan::PrimitiveSortedWithRowIdsVisitor for RangeAdapter<'v, V>
        where
            V: crate::store::scan::PrimitiveSortedWithRowIdsVisitor,
        {
            fn u64_run_with_rids(&mut self, v: &UInt64Array, r: &UInt64Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.u64_r {
                    let start = match lb { Bound::Unbounded => s, Bound::Included(x) => lower_idx(s, s+l, &x, |i| v.value(i)), Bound::Excluded(x) => upper_idx(s, s+l, &x, |i| v.value(i)) };
                    let end = match ub { Bound::Unbounded => s + l, Bound::Included(x) => upper_idx(s, s+l, &x, |i| v.value(i)), Bound::Excluded(x) => lower_idx(s, s+l, &x, |i| v.value(i)) };
                    if start < end { self.inner.u64_run_with_rids(v, r, start, end - start); }
                } else { self.inner.u64_run_with_rids(v, r, s, l); }
            }
            fn i32_run_with_rids(&mut self, v: &Int32Array, r: &UInt64Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.i32_r {
                    let start = match lb { Bound::Unbounded => s, Bound::Included(x) => lower_idx(s, s+l, &x, |i| v.value(i)), Bound::Excluded(x) => upper_idx(s, s+l, &x, |i| v.value(i)) };
                    let end = match ub { Bound::Unbounded => s + l, Bound::Included(x) => upper_idx(s, s+l, &x, |i| v.value(i)), Bound::Excluded(x) => lower_idx(s, s+l, &x, |i| v.value(i)) };
                    if start < end { self.inner.i32_run_with_rids(v, r, start, end - start); }
                } else { self.inner.i32_run_with_rids(v, r, s, l); }
            }
            // For brevity, other integer widths with row ids fall back to pass-through.
        }

        let mut adapter = RangeAdapter { inner: visitor, ir: self.ir };
        self.store.scan(self.field_id, self.opts, &mut adapter)
    }
}

// ---------------------- Sorted range windowing (builder) ----------------------

#[inline]
fn lower_idx_by<T: Ord, F: Fn(usize) -> T>(mut lo: usize, mut hi: usize, pred: &T, get: F) -> usize {
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if get(mid) < *pred { lo = mid + 1; } else { hi = mid; }
    }
    lo
}
#[inline]
fn upper_idx_by<T: Ord, F: Fn(usize) -> T>(mut lo: usize, mut hi: usize, pred: &T, get: F) -> usize {
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if get(mid) <= *pred { lo = mid + 1; } else { hi = mid; }
    }
    lo
}

macro_rules! sorted_visit_bounds_impl {
    ($name:ident, $ArrTy:ty, $ty:ty, $visit:ident) => {
        fn $name<P: Pager<Blob = EntryHandle>, V: PrimitiveSortedVisitor>(
            pager: &P,
            metas: &[ChunkMetadata],
            blobs: &FxHashMap<PhysicalKey, EntryHandle>,
            bounds: (Bound<$ty>, Bound<$ty>),
            visitor: &mut V,
        ) -> Result<()> {
            let mut arrays: Vec<$ArrTy> = Vec::with_capacity(metas.len());
            for m in metas {
                let data_any = deserialize_array(blobs.get(&m.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                let perm_any = deserialize_array(blobs.get(&m.value_order_perm_pk).ok_or(Error::NotFound)?.clone())?;
                let perm = perm_any.as_any().downcast_ref::<UInt32Array>().ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let arr_any = data_any.as_any().downcast_ref::<$ArrTy>().ok_or_else(|| Error::Internal("downcast".into()))?;
                let plen = perm.len();
                if plen == 0 { continue; }
                let get = |i: usize| -> $ty { let idx = perm.value(i) as usize; arr_any.value(idx) as $ty };
                let (lb, ub) = bounds.clone();
                let s = match lb { Bound::Unbounded => 0, Bound::Included(x) => lower_idx_by(0, plen, &x, &get), Bound::Excluded(x) => upper_idx_by(0, plen, &x, &get) };
                let e = match ub { Bound::Unbounded => plen, Bound::Included(x) => upper_idx_by(0, plen, &x, &get), Bound::Excluded(x) => lower_idx_by(0, plen, &x, &get) };
                if s >= e { continue; }
                let pw = perm.slice(s, e - s);
                let taken_any = compute::take(&data_any, &pw, None)?;
                let a = taken_any.as_any().downcast_ref::<$ArrTy>().ok_or_else(|| Error::Internal("sorted downcast".into()))?.clone();
                arrays.push(a);
            }
            if arrays.is_empty() { return Ok(()); }
            kmerge_coalesced::<_, _, _, _, _>(&arrays, |a: &$ArrTy| a.len(), |a: &$ArrTy, i: usize| a.value(i), |c,s,l| visitor.$visit(&arrays[c], s, l));
            Ok(())
        }
    };
}

macro_rules! sorted_with_rids_bounds_impl {
    ($name:ident, $ArrTy:ty, $ty:ty, $visit:ident) => {
        fn $name<P: Pager<Blob = EntryHandle>, V: PrimitiveSortedWithRowIdsVisitor>(
            pager: &P,
            metas_val: &[ChunkMetadata],
            metas_rid: &[ChunkMetadata],
            vblobs: &FxHashMap<PhysicalKey, EntryHandle>,
            rblobs: &FxHashMap<PhysicalKey, EntryHandle>,
            bounds: (Bound<$ty>, Bound<$ty>),
            visitor: &mut V,
        ) -> Result<()> {
            if metas_val.len() != metas_rid.len() { return Err(Error::Internal("sorted_with_rids: chunk count mismatch".into())); }
            let mut vals: Vec<$ArrTy> = Vec::with_capacity(metas_val.len());
            let mut rids: Vec<UInt64Array> = Vec::with_capacity(metas_val.len());
            for (mv, mr) in metas_val.iter().zip(metas_rid.iter()) {
                let data_any = deserialize_array(vblobs.get(&mv.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                let perm_any = deserialize_array(vblobs.get(&mv.value_order_perm_pk).ok_or(Error::NotFound)?.clone())?;
                let perm = perm_any.as_any().downcast_ref::<UInt32Array>().ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let arr_any = data_any.as_any().downcast_ref::<$ArrTy>().ok_or_else(|| Error::Internal("downcast".into()))?;
                let plen = perm.len(); if plen == 0 { continue; }
                let get = |i: usize| -> $ty { let idx = perm.value(i) as usize; arr_any.value(idx) as $ty };
                let (lb, ub) = bounds.clone();
                let s = match lb { Bound::Unbounded => 0, Bound::Included(x) => lower_idx_by(0, plen, &x, &get), Bound::Excluded(x) => upper_idx_by(0, plen, &x, &get) };
                let e = match ub { Bound::Unbounded => plen, Bound::Included(x) => upper_idx_by(0, plen, &x, &get), Bound::Excluded(x) => lower_idx_by(0, plen, &x, &get) };
                if s >= e { continue; }
                let pw = perm.slice(s, e - s);
                let taken_any = compute::take(&data_any, &pw, None)?;
                let a = taken_any.as_any().downcast_ref::<$ArrTy>().ok_or_else(|| Error::Internal("sorted downcast".into()))?.clone();
                vals.push(a);
                let rid_any = deserialize_array(rblobs.get(&mr.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                let taken_rid_any = compute::take(&rid_any, &pw, None)?;
                let ra = taken_rid_any.as_any().downcast_ref::<UInt64Array>().ok_or_else(|| Error::Internal("rid downcast".into()))?.clone();
                rids.push(ra);
            }
            if vals.is_empty() { return Ok(()); }
            kmerge_coalesced::<_, _, _, _, _>(&vals, |a: &$ArrTy| a.len(), |a: &$ArrTy, i: usize| a.value(i), |c,s,l| visitor.$visit(&vals[c], &rids[c], s, l));
            Ok(())
        }
    };
}

sorted_visit_bounds_impl!(sorted_visit_u64_bounds, UInt64Array, u64, u64_run);
sorted_visit_bounds_impl!(sorted_visit_u32_bounds, UInt32Array, u32, u32_run);
sorted_visit_bounds_impl!(sorted_visit_u16_bounds, UInt16Array, u16, u16_run);
sorted_visit_bounds_impl!(sorted_visit_u8_bounds,  UInt8Array,  u8,  u8_run);
sorted_visit_bounds_impl!(sorted_visit_i64_bounds, Int64Array,  i64, i64_run);
sorted_visit_bounds_impl!(sorted_visit_i32_bounds, Int32Array,  i32, i32_run);
sorted_visit_bounds_impl!(sorted_visit_i16_bounds, Int16Array,  i16, i16_run);
sorted_visit_bounds_impl!(sorted_visit_i8_bounds,  Int8Array,   i8,  i8_run);

sorted_with_rids_bounds_impl!(sorted_visit_with_rids_u64_bounds, UInt64Array, u64, u64_run_with_rids);
sorted_with_rids_bounds_impl!(sorted_visit_with_rids_u32_bounds, UInt32Array, u32, u32_run_with_rids);
sorted_with_rids_bounds_impl!(sorted_visit_with_rids_u16_bounds, UInt16Array, u16, u16_run_with_rids);
sorted_with_rids_bounds_impl!(sorted_visit_with_rids_u8_bounds,  UInt8Array,  u8,  u8_run_with_rids);
sorted_with_rids_bounds_impl!(sorted_visit_with_rids_i64_bounds, Int64Array,  i64, i64_run_with_rids);
sorted_with_rids_bounds_impl!(sorted_visit_with_rids_i32_bounds, Int32Array,  i32, i32_run_with_rids);
sorted_with_rids_bounds_impl!(sorted_visit_with_rids_i16_bounds, Int16Array,  i16, i16_run_with_rids);
sorted_with_rids_bounds_impl!(sorted_visit_with_rids_i8_bounds,  Int8Array,   i8,  i8_run_with_rids);

fn range_sorted_dispatch<P, V>(
    store: &ColumnStore<P>,
    field_id: LogicalFieldId,
    opts: ScanOptions,
    ir: IntRanges,
    visitor: &mut V,
) -> Result<()>
where
    P: Pager<Blob = EntryHandle,>,
    V: PrimitiveVisitor + PrimitiveSortedVisitor + PrimitiveWithRowIdsVisitor + PrimitiveSortedWithRowIdsVisitor,
{
    // Load descriptor metas and blobs (values, perms, and rids when needed)
    let catalog = store.catalog.read().unwrap();
    let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
    let desc_blob = store
        .pager
        .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
        .pop()
        .and_then(|r| match r { GetResult::Raw { bytes, .. } => Some(bytes), _ => None })
        .ok_or(Error::NotFound)?;
    let desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
    let mut metas_val: Vec<ChunkMetadata> = Vec::new();
    for m in crate::store::descriptor::DescriptorIterator::new(store.pager.as_ref(), desc.head_page_pk) { let meta=m?; if meta.row_count>0 { if meta.value_order_perm_pk==0 { return Err(Error::NotFound); } metas_val.push(meta); } }

    let (rid_desc_opt, metas_rid): (Option<crate::store::descriptor::ColumnDescriptor>, Vec<ChunkMetadata>) = if opts.with_row_ids {
        let rid_fid = opts.row_id_field.ok_or_else(|| Error::Internal("row_id field id required when with_row_ids=true".into()))?;
        let rid_pk = *catalog.map.get(&rid_fid).ok_or(Error::NotFound)?;
        let rid_desc_blob = store
            .pager
            .batch_get(&[BatchGet::Raw { key: rid_pk }])?
            .pop()
            .and_then(|r| match r { GetResult::Raw { bytes, .. } => Some(bytes), _ => None })
            .ok_or(Error::NotFound)?;
        let rid_desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(rid_desc_blob.as_ref());
        let mut mr = Vec::new();
        for m in crate::store::descriptor::DescriptorIterator::new(store.pager.as_ref(), rid_desc.head_page_pk) { let meta=m?; if meta.row_count>0 { mr.push(meta); } }
        (Some(rid_desc), mr)
    } else { (None, Vec::new()) };
    drop(catalog);

    if metas_val.is_empty() { return Ok(()); }

    // Batch gets
    let mut gets: Vec<BatchGet> = Vec::with_capacity(metas_val.len() * if opts.with_row_ids {3} else {2});
    for (i, mv) in metas_val.iter().enumerate() {
        gets.push(BatchGet::Raw { key: mv.chunk_pk });
        gets.push(BatchGet::Raw { key: mv.value_order_perm_pk });
        if opts.with_row_ids { gets.push(BatchGet::Raw { key: metas_rid[i].chunk_pk }); }
    }
    let results = store.pager.batch_get(&gets)?;
    let mut vblobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
    let mut rblobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
    for r in results { if let GetResult::Raw{key,bytes} = r { if opts.with_row_ids && metas_rid.iter().any(|m| m.chunk_pk==key) { rblobs.insert(key, bytes); } else { vblobs.insert(key, bytes); } } }
    let first_any = deserialize_array(vblobs.get(&metas_val[0].chunk_pk).ok_or(Error::NotFound)?.clone())?;

    // Dispatch by dtype + bounds for this dtype only
    if opts.with_row_ids {
        return match first_any.data_type() {
            DataType::UInt64 => { let (lb,ub)=ir.u64_r.unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_with_rids_u64_bounds(store.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, (lb,ub), visitor) }
            DataType::UInt32 => { let (lb,ub)=ir.u32_r.unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_with_rids_u32_bounds(store.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, (lb,ub), visitor) }
            DataType::UInt16 => { let (lb,ub)=ir.u16_r.unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_with_rids_u16_bounds(store.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, (lb,ub), visitor) }
            DataType::UInt8  => { let (lb,ub)=ir.u8_r .unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_with_rids_u8_bounds (store.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, (lb,ub), visitor) }
            DataType::Int64  => { let (lb,ub)=ir.i64_r.unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_with_rids_i64_bounds(store.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, (lb,ub), visitor) }
            DataType::Int32  => { let (lb,ub)=ir.i32_r.unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_with_rids_i32_bounds(store.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, (lb,ub), visitor) }
            DataType::Int16  => { let (lb,ub)=ir.i16_r.unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_with_rids_i16_bounds(store.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, (lb,ub), visitor) }
            DataType::Int8   => { let (lb,ub)=ir.i8_r .unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_with_rids_i8_bounds (store.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, (lb,ub), visitor) }
            _ => Err(Error::Internal("unsupported sorted dtype (builder)".into()))
        };
    } else {
        return match first_any.data_type() {
            DataType::UInt64 => { let (lb,ub)=ir.u64_r.unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_u64_bounds(store.pager.as_ref(), &metas_val, &vblobs, (lb,ub), visitor) }
            DataType::UInt32 => { let (lb,ub)=ir.u32_r.unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_u32_bounds(store.pager.as_ref(), &metas_val, &vblobs, (lb,ub), visitor) }
            DataType::UInt16 => { let (lb,ub)=ir.u16_r.unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_u16_bounds(store.pager.as_ref(), &metas_val, &vblobs, (lb,ub), visitor) }
            DataType::UInt8  => { let (lb,ub)=ir.u8_r .unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_u8_bounds (store.pager.as_ref(), &metas_val, &vblobs, (lb,ub), visitor) }
            DataType::Int64  => { let (lb,ub)=ir.i64_r.unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_i64_bounds(store.pager.as_ref(), &metas_val, &vblobs, (lb,ub), visitor) }
            DataType::Int32  => { let (lb,ub)=ir.i32_r.unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_i32_bounds(store.pager.as_ref(), &metas_val, &vblobs, (lb,ub), visitor) }
            DataType::Int16  => { let (lb,ub)=ir.i16_r.unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_i16_bounds(store.pager.as_ref(), &metas_val, &vblobs, (lb,ub), visitor) }
            DataType::Int8   => { let (lb,ub)=ir.i8_r .unwrap_or((Bound::Unbounded, Bound::Unbounded)); sorted_visit_i8_bounds (store.pager.as_ref(), &metas_val, &vblobs, (lb,ub), visitor) }
            _ => Err(Error::Internal("unsupported sorted dtype (builder)".into()))
        };
    }
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

        if opts.with_row_ids {
            let row_fid = opts
                .row_id_field
                .ok_or_else(|| Error::Internal("row_id field id required when with_row_ids=true".into()))?;
            // Prepare value metas and blobs
            let catalog = self.catalog.read().unwrap();
            let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
            let desc_blob = self
                .pager
                .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
                .pop()
                .and_then(|r| match r { GetResult::Raw { bytes, .. } => Some(bytes), _ => None })
                .ok_or(Error::NotFound)?;
            let desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

            let rid_descriptor_pk = *catalog.map.get(&row_fid).ok_or(Error::NotFound)?;
            let rid_desc_blob = self
                .pager
                .batch_get(&[BatchGet::Raw { key: rid_descriptor_pk }])?
                .pop()
                .and_then(|r| match r { GetResult::Raw { bytes, .. } => Some(bytes), _ => None })
                .ok_or(Error::NotFound)?;
            let rid_desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(rid_desc_blob.as_ref());
            drop(catalog);

            let mut metas_val: Vec<crate::store::descriptor::ChunkMetadata> = Vec::new();
            for m in crate::store::descriptor::DescriptorIterator::new(self.pager.as_ref(), desc.head_page_pk) {
                let meta = m?; if meta.row_count == 0 { continue; }
                if meta.value_order_perm_pk == 0 { return Err(Error::NotFound); }
                metas_val.push(meta);
            }
            let mut metas_rid: Vec<crate::store::descriptor::ChunkMetadata> = Vec::new();
            for m in crate::store::descriptor::DescriptorIterator::new(self.pager.as_ref(), rid_desc.head_page_pk) {
                let meta = m?; if meta.row_count == 0 { continue; }
                metas_rid.push(meta);
            }
            if metas_val.is_empty() { return Ok(()); }
            if metas_val.len() != metas_rid.len() { return Err(Error::Internal("sorted_with_row_ids: chunk count mismatch".into())); }

            // Batch get: values (chunks + perms) and rids (chunks)
            let mut gets: Vec<BatchGet> = Vec::with_capacity(metas_val.len()*3);
            for (mv, mr) in metas_val.iter().zip(metas_rid.iter()) {
                gets.push(BatchGet::Raw{key: mv.chunk_pk});
                gets.push(BatchGet::Raw{key: mv.value_order_perm_pk});
                gets.push(BatchGet::Raw{key: mr.chunk_pk});
            }
            let results = self.pager.batch_get(&gets)?;
            let mut vblobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
            let mut rblobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
            for r in results {
                if let GetResult::Raw{key,bytes} = r {
                    // Heuristic: key in rid metas => rid blob, else value blob
                    if metas_rid.iter().any(|m| m.chunk_pk==key) { rblobs.insert(key, bytes); }
                    else { vblobs.insert(key, bytes); }
                }
            }
            let first_any = crate::serialization::deserialize_array(vblobs.get(&metas_val[0].chunk_pk).ok_or(Error::NotFound)?.clone())?;
            if opts.reverse {
                match first_any.data_type() {
                    DataType::UInt64 => sorted_visit_with_rids_u64_rev(self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    DataType::UInt32 => sorted_visit_with_rids_u32_rev(self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    DataType::UInt16 => sorted_visit_with_rids_u16_rev(self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    DataType::UInt8  => sorted_visit_with_rids_u8_rev (self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    DataType::Int64  => sorted_visit_with_rids_i64_rev(self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    DataType::Int32  => sorted_visit_with_rids_i32_rev(self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    DataType::Int16  => sorted_visit_with_rids_i16_rev(self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    DataType::Int8   => sorted_visit_with_rids_i8_rev (self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    _ => Err(Error::Internal("unsupported sorted dtype".into())),
                }
            } else {
                match first_any.data_type() {
                    DataType::UInt64 => sorted_visit_with_rids_u64(self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    DataType::UInt32 => sorted_visit_with_rids_u32(self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    DataType::UInt16 => sorted_visit_with_rids_u16(self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    DataType::UInt8  => sorted_visit_with_rids_u8 (self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    DataType::Int64  => sorted_visit_with_rids_i64(self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    DataType::Int32  => sorted_visit_with_rids_i32(self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    DataType::Int16  => sorted_visit_with_rids_i16(self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    DataType::Int8   => sorted_visit_with_rids_i8 (self.pager.as_ref(), &metas_val, &metas_rid, &vblobs, &rblobs, visitor),
                    _ => Err(Error::Internal("unsupported sorted dtype".into())),
                }
            }
        } else {
            if opts.reverse { self.scan_sorted_visit_reverse(field_id, visitor) } else { self.scan_sorted_visit(field_id, visitor) }
        }
    }
}
