use super::*;

// --------------------------- Sorted Scanners ----------------------------

/// Generic k-way merge over sorted per-chunk arrays, coalescing runs.
/// `get` fetches the T value at index for an array, and `emit` sends runs.
pub(crate) fn kmerge_coalesced<T, A, FLen, FGet, FEmit>(
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
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    // Max-heap by value (natural ordering)
    impl<T: Ord> Ord for H<T> {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.v.cmp(&o.v).then_with(|| self.c.cmp(&o.c))
        }
    }

    let mut heap: BinaryHeap<H<T>> = BinaryHeap::new();
    for (ci, a) in arrays.iter().enumerate() {
        let al = len_of(a);
        if al > 0 {
            let idx = al - 1;
            heap.push(H {
                v: get(a, idx),
                c: ci,
                i: idx,
            });
        }
    }

    while let Some(h) = heap.pop() {
        let c = h.c;
        let a = &arrays[c];
        let e = h.i; // inclusive end
        let mut s = e; // inclusive start, will decrease
        let thr = heap.peek().map(|x| x.v);
        if let Some(t) = thr {
            while s > 0 {
                let p = s - 1;
                if get(a, p) >= t {
                    s = p;
                } else {
                    break;
                }
            }
        } else {
            // drain remaining
            s = 0;
        }
        // Emit as (start,len) using ascending indices; caller can iterate descending within the run
        emit(c, s, e - s + 1);
        if s > 0 {
            let next = s - 1;
            heap.push(H {
                v: get(a, next),
                c,
                i: next,
            });
        }
    }
}

macro_rules! sorted_visit_impl {
    ($name:ident, $name_rev:ident, $ArrTy:ty, $visit:ident) => {
        pub fn $name<P: Pager<Blob = EntryHandle>, V: PrimitiveSortedVisitor>(
            _pager: &P,
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
            _pager: &P,
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
            if arrays.is_empty() {
                return Ok(());
            }
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
sorted_visit_impl!(sorted_visit_u8, sorted_visit_u8_rev, UInt8Array, u8_run);
sorted_visit_impl!(sorted_visit_i64, sorted_visit_i64_rev, Int64Array, i64_run);
sorted_visit_impl!(sorted_visit_i32, sorted_visit_i32_rev, Int32Array, i32_run);
sorted_visit_impl!(sorted_visit_i16, sorted_visit_i16_rev, Int16Array, i16_run);
sorted_visit_impl!(sorted_visit_i8, sorted_visit_i8_rev, Int8Array, i8_run);

// Note: A sorted-with-row-ids variant can be added similarly if needed.

macro_rules! sorted_with_rids_impl {
    ($name:ident, $name_rev:ident, $ArrTy:ty, $visit:ident) => {
        pub fn $name<P: Pager<Blob = EntryHandle>, V: PrimitiveSortedWithRowIdsVisitor>(
            _pager: &P,
            metas_val: &[ChunkMetadata],
            metas_rid: &[ChunkMetadata],
            vblobs: &FxHashMap<PhysicalKey, EntryHandle>,
            rblobs: &FxHashMap<PhysicalKey, EntryHandle>,
            visitor: &mut V,
        ) -> Result<()> {
            if metas_val.len() != metas_rid.len() {
                return Err(Error::Internal(
                    "sorted_with_rids: chunk count mismatch".into(),
                ));
            }
            let mut vals: Vec<$ArrTy> = Vec::with_capacity(metas_val.len());
            let mut rids: Vec<UInt64Array> = Vec::with_capacity(metas_val.len());
            for (mv, mr) in metas_val.iter().zip(metas_rid.iter()) {
                let data_any =
                    deserialize_array(vblobs.get(&mv.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                let perm_any = deserialize_array(
                    vblobs
                        .get(&mv.value_order_perm_pk)
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
                vals.push(arr);
                let rid_any =
                    deserialize_array(rblobs.get(&mr.chunk_pk).ok_or(Error::NotFound)?.clone())?;
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
            kmerge_coalesced::<_, _, _, _, _>(
                &vals,
                |a: &$ArrTy| a.len(),
                |a: &$ArrTy, i: usize| a.value(i),
                |c, s, l| visitor.$visit(&vals[c], &rids[c], s, l),
            );
            Ok(())
        }
        pub fn $name_rev<P: Pager<Blob = EntryHandle>, V: PrimitiveSortedWithRowIdsVisitor>(
            _pager: &P,
            metas_val: &[ChunkMetadata],
            metas_rid: &[ChunkMetadata],
            vblobs: &FxHashMap<PhysicalKey, EntryHandle>,
            rblobs: &FxHashMap<PhysicalKey, EntryHandle>,
            visitor: &mut V,
        ) -> Result<()> {
            if metas_val.len() != metas_rid.len() {
                return Err(Error::Internal(
                    "sorted_with_rids: chunk count mismatch".into(),
                ));
            }
            let mut vals: Vec<$ArrTy> = Vec::with_capacity(metas_val.len());
            let mut rids: Vec<UInt64Array> = Vec::with_capacity(metas_val.len());
            for (mv, mr) in metas_val.iter().zip(metas_rid.iter()) {
                let data_any =
                    deserialize_array(vblobs.get(&mv.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                let perm_any = deserialize_array(
                    vblobs
                        .get(&mv.value_order_perm_pk)
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
                vals.push(arr);
                let rid_any =
                    deserialize_array(rblobs.get(&mr.chunk_pk).ok_or(Error::NotFound)?.clone())?;
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

macro_rules! sorted_visit_bounds_impl {
    ($name:ident, $ArrTy:ty, $ty:ty, $visit:ident) => {
        fn $name<P: Pager<Blob = EntryHandle>, V: PrimitiveSortedVisitor>(
            _pager: &P,
            metas: &[ChunkMetadata],
            blobs: &FxHashMap<PhysicalKey, EntryHandle>,
            bounds: (Bound<$ty>, Bound<$ty>),
            visitor: &mut V,
        ) -> Result<()> {
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
            kmerge_coalesced::<_, _, _, _, _>(
                &arrays,
                |a: &$ArrTy| a.len(),
                |a: &$ArrTy, i: usize| a.value(i),
                |c, s, l| visitor.$visit(&arrays[c], s, l),
            );
            Ok(())
        }
    };
}

macro_rules! sorted_with_rids_bounds_impl {
    ($name:ident, $ArrTy:ty, $ty:ty, $visit:ident) => {
        fn $name<P: Pager<Blob = EntryHandle>, V: PrimitiveSortedWithRowIdsVisitor>(
            _pager: &P,
            metas_val: &[ChunkMetadata],
            metas_rid: &[ChunkMetadata],
            vblobs: &FxHashMap<PhysicalKey, EntryHandle>,
            rblobs: &FxHashMap<PhysicalKey, EntryHandle>,
            bounds: (Bound<$ty>, Bound<$ty>),
            visitor: &mut V,
        ) -> Result<()> {
            if metas_val.len() != metas_rid.len() {
                return Err(Error::Internal(
                    "sorted_with_rids: chunk count mismatch".into(),
                ));
            }
            let mut vals: Vec<$ArrTy> = Vec::with_capacity(metas_val.len());
            let mut rids: Vec<UInt64Array> = Vec::with_capacity(metas_val.len());
            for (mv, mr) in metas_val.iter().zip(metas_rid.iter()) {
                let data_any =
                    deserialize_array(vblobs.get(&mv.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                let perm_any = deserialize_array(
                    vblobs
                        .get(&mv.value_order_perm_pk)
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
                let rid_any =
                    deserialize_array(rblobs.get(&mr.chunk_pk).ok_or(Error::NotFound)?.clone())?;
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
            kmerge_coalesced::<_, _, _, _, _>(
                &vals,
                |a: &$ArrTy| a.len(),
                |a: &$ArrTy, i: usize| a.value(i),
                |c, s, l| visitor.$visit(&vals[c], &rids[c], s, l),
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
        let rid_fid = opts.row_id_field.ok_or_else(|| {
            Error::Internal("row_id field id required when with_row_ids=true".into())
        })?;
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

    // Batch gets
    let mut gets: Vec<BatchGet> =
        Vec::with_capacity(metas_val.len() * if opts.with_row_ids { 3 } else { 2 });
    for (i, mv) in metas_val.iter().enumerate() {
        gets.push(BatchGet::Raw { key: mv.chunk_pk });
        gets.push(BatchGet::Raw {
            key: mv.value_order_perm_pk,
        });
        if opts.with_row_ids {
            gets.push(BatchGet::Raw {
                key: metas_rid[i].chunk_pk,
            });
        }
    }
    let results = store.pager.batch_get(&gets)?;
    let mut vblobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
    let mut rblobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
    for r in results {
        if let GetResult::Raw { key, bytes } = r {
            if opts.with_row_ids && metas_rid.iter().any(|m| m.chunk_pk == key) {
                rblobs.insert(key, bytes);
            } else {
                vblobs.insert(key, bytes);
            }
        }
    }
    let first_any = deserialize_array(
        vblobs
            .get(&metas_val[0].chunk_pk)
            .ok_or(Error::NotFound)?
            .clone(),
    )?;

    // Dispatch by dtype + bounds for this dtype only
    if opts.with_row_ids {
        return match first_any.data_type() {
            DataType::UInt64 => {
                let (lb, ub) = ir.u64_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_with_rids_u64_bounds(
                    store.pager.as_ref(),
                    &metas_val,
                    &metas_rid,
                    &vblobs,
                    &rblobs,
                    (lb, ub),
                    visitor,
                )
            }
            DataType::UInt32 => {
                let (lb, ub) = ir.u32_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_with_rids_u32_bounds(
                    store.pager.as_ref(),
                    &metas_val,
                    &metas_rid,
                    &vblobs,
                    &rblobs,
                    (lb, ub),
                    visitor,
                )
            }
            DataType::UInt16 => {
                let (lb, ub) = ir.u16_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_with_rids_u16_bounds(
                    store.pager.as_ref(),
                    &metas_val,
                    &metas_rid,
                    &vblobs,
                    &rblobs,
                    (lb, ub),
                    visitor,
                )
            }
            DataType::UInt8 => {
                let (lb, ub) = ir.u8_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_with_rids_u8_bounds(
                    store.pager.as_ref(),
                    &metas_val,
                    &metas_rid,
                    &vblobs,
                    &rblobs,
                    (lb, ub),
                    visitor,
                )
            }
            DataType::Int64 => {
                let (lb, ub) = ir.i64_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_with_rids_i64_bounds(
                    store.pager.as_ref(),
                    &metas_val,
                    &metas_rid,
                    &vblobs,
                    &rblobs,
                    (lb, ub),
                    visitor,
                )
            }
            DataType::Int32 => {
                let (lb, ub) = ir.i32_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_with_rids_i32_bounds(
                    store.pager.as_ref(),
                    &metas_val,
                    &metas_rid,
                    &vblobs,
                    &rblobs,
                    (lb, ub),
                    visitor,
                )
            }
            DataType::Int16 => {
                let (lb, ub) = ir.i16_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_with_rids_i16_bounds(
                    store.pager.as_ref(),
                    &metas_val,
                    &metas_rid,
                    &vblobs,
                    &rblobs,
                    (lb, ub),
                    visitor,
                )
            }
            DataType::Int8 => {
                let (lb, ub) = ir.i8_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_with_rids_i8_bounds(
                    store.pager.as_ref(),
                    &metas_val,
                    &metas_rid,
                    &vblobs,
                    &rblobs,
                    (lb, ub),
                    visitor,
                )
            }
            _ => Err(Error::Internal("unsupported sorted dtype (builder)".into())),
        };
    } else {
        return match first_any.data_type() {
            DataType::UInt64 => {
                let (lb, ub) = ir.u64_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_u64_bounds(
                    store.pager.as_ref(),
                    &metas_val,
                    &vblobs,
                    (lb, ub),
                    visitor,
                )
            }
            DataType::UInt32 => {
                let (lb, ub) = ir.u32_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_u32_bounds(
                    store.pager.as_ref(),
                    &metas_val,
                    &vblobs,
                    (lb, ub),
                    visitor,
                )
            }
            DataType::UInt16 => {
                let (lb, ub) = ir.u16_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_u16_bounds(
                    store.pager.as_ref(),
                    &metas_val,
                    &vblobs,
                    (lb, ub),
                    visitor,
                )
            }
            DataType::UInt8 => {
                let (lb, ub) = ir.u8_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_u8_bounds(store.pager.as_ref(), &metas_val, &vblobs, (lb, ub), visitor)
            }
            DataType::Int64 => {
                let (lb, ub) = ir.i64_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_i64_bounds(
                    store.pager.as_ref(),
                    &metas_val,
                    &vblobs,
                    (lb, ub),
                    visitor,
                )
            }
            DataType::Int32 => {
                let (lb, ub) = ir.i32_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_i32_bounds(
                    store.pager.as_ref(),
                    &metas_val,
                    &vblobs,
                    (lb, ub),
                    visitor,
                )
            }
            DataType::Int16 => {
                let (lb, ub) = ir.i16_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_i16_bounds(
                    store.pager.as_ref(),
                    &metas_val,
                    &vblobs,
                    (lb, ub),
                    visitor,
                )
            }
            DataType::Int8 => {
                let (lb, ub) = ir.i8_r.unwrap_or((Bound::Unbounded, Bound::Unbounded));
                sorted_visit_i8_bounds(store.pager.as_ref(), &metas_val, &vblobs, (lb, ub), visitor)
            }
            _ => Err(Error::Internal("unsupported sorted dtype (builder)".into())),
        };
    }
}
