use super::*;

/// Unsorted primitive visitor: one callback per chunk per type.
pub trait PrimitiveVisitor {
    fn u64_chunk(&mut self, _a: &UInt64Array) {
        unimplemented!("`u64_chunk` not implemented")
    }
    fn u32_chunk(&mut self, _a: &UInt32Array) {
        unimplemented!("`u32_chunk` not implemented")
    }
    fn u16_chunk(&mut self, _a: &UInt16Array) {
        unimplemented!("`u16_chunk` not implemented")
    }
    fn u8_chunk(&mut self, _a: &UInt8Array) {
        unimplemented!("`u8_chunk` not implemented")
    }
    fn i64_chunk(&mut self, _a: &Int64Array) {
        unimplemented!("`i64_chunk` not implemented")
    }
    fn i32_chunk(&mut self, _a: &Int32Array) {
        unimplemented!("`i32_chunk` not implemented")
    }
    fn i16_chunk(&mut self, _a: &Int16Array) {
        unimplemented!("`i16_chunk` not implemented")
    }
    fn i8_chunk(&mut self, _a: &Int8Array) {
        unimplemented!("`i8_chunk` not implemented")
    }
}

/// Unsorted primitive visitor with row ids (u64).
pub trait PrimitiveWithRowIdsVisitor {
    fn u64_chunk_with_rids(&mut self, _v: &UInt64Array, _r: &UInt64Array) {
        unimplemented!("`u64_chunk_with_rids` not implemented")
    }
    fn u32_chunk_with_rids(&mut self, _v: &UInt32Array, _r: &UInt64Array) {
        unimplemented!("`u32_chunk_with_rids` not implemented")
    }
    fn u16_chunk_with_rids(&mut self, _v: &UInt16Array, _r: &UInt64Array) {
        unimplemented!("`u16_chunk_with_rids` not implemented")
    }
    fn u8_chunk_with_rids(&mut self, _v: &UInt8Array, _r: &UInt64Array) {
        unimplemented!("`u8_chunk_with_rids` not implemented")
    }
    fn i64_chunk_with_rids(&mut self, _v: &Int64Array, _r: &UInt64Array) {
        unimplemented!("`i64_chunk_with_rids` not implemented")
    }
    fn i32_chunk_with_rids(&mut self, _v: &Int32Array, _r: &UInt64Array) {
        unimplemented!("`i32_chunk_with_rids` not implemented")
    }
    fn i16_chunk_with_rids(&mut self, _v: &Int16Array, _r: &UInt64Array) {
        unimplemented!("`i16_chunk_with_rids` not implemented")
    }
    fn i8_chunk_with_rids(&mut self, _v: &Int8Array, _r: &UInt64Array) {
        unimplemented!("`i8_chunk_with_rids` not implemented")
    }
}

/// Sorted visitor fed with coalesced runs (start,len) within a typed array.
pub trait PrimitiveSortedVisitor {
    fn u64_run(&mut self, _a: &UInt64Array, _start: usize, _len: usize) {
        unimplemented!("`u64_run` not implemented")
    }
    fn u32_run(&mut self, _a: &UInt32Array, _start: usize, _len: usize) {
        unimplemented!("`u32_run` not implemented")
    }
    fn u16_run(&mut self, _a: &UInt16Array, _start: usize, _len: usize) {
        unimplemented!("`u16_run` not implemented")
    }
    fn u8_run(&mut self, _a: &UInt8Array, _start: usize, _len: usize) {
        unimplemented!("`u8_run` not implemented")
    }
    fn i64_run(&mut self, _a: &Int64Array, _start: usize, _len: usize) {
        unimplemented!("`i64_run` not implemented")
    }
    fn i32_run(&mut self, _a: &Int32Array, _start: usize, _len: usize) {
        unimplemented!("`i32_run` not implemented")
    }
    fn i16_run(&mut self, _a: &Int16Array, _start: usize, _len: usize) {
        unimplemented!("`i16_run` not implemented")
    }
    fn i8_run(&mut self, _a: &Int8Array, _start: usize, _len: usize) {
        unimplemented!("`i8_run` not implemented")
    }
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
    /// Null-only run: values are missing; emits only row_ids.
    fn null_run(&mut self, _r: &UInt64Array, _start: usize, _len: usize) {}
}

// Pagination adapter: enforces offset/limit across chunks (unsorted)
// and across coalesced runs (sorted). It wraps an inner visitor that
// implements the same traits and forwards appropriately.
pub struct PaginateVisitor<'a, V> {
    pub inner: &'a mut V,
    // items to skip before emitting
    skip: usize,
    // remaining items to emit (None => unbounded)
    remaining: Option<usize>,
    // when true, interpret sorted runs as descending
    reverse: bool,
}

impl<'a, V> PaginateVisitor<'a, V> {
    pub fn new(inner: &'a mut V, offset: usize, limit: Option<usize>) -> Self {
        Self {
            inner,
            skip: offset,
            remaining: limit,
            reverse: false,
        }
    }

    pub fn new_with_reverse(
        inner: &'a mut V,
        offset: usize,
        limit: Option<usize>,
        reverse: bool,
    ) -> Self {
        Self {
            inner,
            skip: offset,
            remaining: limit,
            reverse,
        }
    }

    #[inline]
    fn split_len(&mut self, len: usize) -> Option<(usize, usize)> {
        // Apply skip first
        if self.skip >= len {
            self.skip -= len;
            return None;
        }
        let start = self.skip;
        let mut take = len - self.skip;
        self.skip = 0;
        if let Some(rem) = self.remaining.as_mut() {
            if *rem == 0 {
                return None;
            }
            if take > *rem {
                take = *rem;
            }
            *rem -= take;
        }
        Some((start, take))
    }

    #[inline]
    fn done(&self) -> bool {
        matches!(self.remaining, Some(0))
    }
}

macro_rules! impl_unsorted_paginate_for_type {
    ($meth:ident, $ArrTy:ty) => {
        fn $meth(&mut self, a: &$ArrTy) {
            if self.done() {
                return;
            }
            let len = a.len();
            if let Some((s, l)) = self.split_len(len) {
                if s == 0 && self.remaining.is_none() {
                    // No slicing needed; forward full array
                    self.inner.$meth(a);
                } else {
                    // Typed slice returns a new array view of the same type
                    let sa: $ArrTy = a.slice(s, l);
                    self.inner.$meth(&sa);
                }
            }
        }
    };
}

impl<'a, V> PrimitiveVisitor for PaginateVisitor<'a, V>
where
    V: PrimitiveVisitor,
{
    impl_unsorted_paginate_for_type!(u64_chunk, UInt64Array);
    impl_unsorted_paginate_for_type!(u32_chunk, UInt32Array);
    impl_unsorted_paginate_for_type!(u16_chunk, UInt16Array);
    impl_unsorted_paginate_for_type!(u8_chunk, UInt8Array);
    impl_unsorted_paginate_for_type!(i64_chunk, Int64Array);
    impl_unsorted_paginate_for_type!(i32_chunk, Int32Array);
    impl_unsorted_paginate_for_type!(i16_chunk, Int16Array);
    impl_unsorted_paginate_for_type!(i8_chunk, Int8Array);
}

macro_rules! impl_unsorted_with_rids_paginate_for_type {
    ($meth:ident, $ArrTy:ty) => {
        fn $meth(&mut self, v: &$ArrTy, r: &UInt64Array) {
            if self.done() {
                return;
            }
            let len = v.len();
            if len != r.len() {
                panic!("value/rowid length mismatch");
            }
            if let Some((s, l)) = self.split_len(len) {
                let va: $ArrTy = v.slice(s, l);
                let ra: UInt64Array = r.slice(s, l);
                self.inner.$meth(&va, &ra);
            }
        }
    };
}

impl<'a, V> PrimitiveWithRowIdsVisitor for PaginateVisitor<'a, V>
where
    V: PrimitiveWithRowIdsVisitor,
{
    impl_unsorted_with_rids_paginate_for_type!(u64_chunk_with_rids, UInt64Array);
    impl_unsorted_with_rids_paginate_for_type!(u32_chunk_with_rids, UInt32Array);
    impl_unsorted_with_rids_paginate_for_type!(u16_chunk_with_rids, UInt16Array);
    impl_unsorted_with_rids_paginate_for_type!(u8_chunk_with_rids, UInt8Array);
    impl_unsorted_with_rids_paginate_for_type!(i64_chunk_with_rids, Int64Array);
    impl_unsorted_with_rids_paginate_for_type!(i32_chunk_with_rids, Int32Array);
    impl_unsorted_with_rids_paginate_for_type!(i16_chunk_with_rids, Int16Array);
    impl_unsorted_with_rids_paginate_for_type!(i8_chunk_with_rids, Int8Array);
}

macro_rules! impl_sorted_paginate_for_type {
    ($meth:ident, $ArrTy:ty) => {
        fn $meth(&mut self, a: &$ArrTy, start: usize, len: usize) {
            if self.done() {
                return;
            }
            if let Some((delta, take)) = self.split_len(len) {
                if take > 0 {
                    if self.reverse {
                        // For descending semantics, take from the end of the run.
                        // The first items to emit correspond to the last indices.
                        let adj_start = start + (len - (delta + take));
                        self.inner.$meth(a, adj_start, take);
                    } else {
                        self.inner.$meth(a, start + delta, take);
                    }
                }
            }
        }
    };
}

impl<'a, V> PrimitiveSortedVisitor for PaginateVisitor<'a, V>
where
    V: PrimitiveSortedVisitor,
{
    impl_sorted_paginate_for_type!(u64_run, UInt64Array);
    impl_sorted_paginate_for_type!(u32_run, UInt32Array);
    impl_sorted_paginate_for_type!(u16_run, UInt16Array);
    impl_sorted_paginate_for_type!(u8_run, UInt8Array);
    impl_sorted_paginate_for_type!(i64_run, Int64Array);
    impl_sorted_paginate_for_type!(i32_run, Int32Array);
    impl_sorted_paginate_for_type!(i16_run, Int16Array);
    impl_sorted_paginate_for_type!(i8_run, Int8Array);
}

macro_rules! impl_sorted_with_rids_paginate_for_type {
    ($meth:ident, $ArrTy:ty) => {
        fn $meth(&mut self, v: &$ArrTy, r: &UInt64Array, start: usize, len: usize) {
            if self.done() {
                return;
            }
            if let Some((delta, take)) = self.split_len(len) {
                if take > 0 {
                    if self.reverse {
                        let adj_start = start + (len - (delta + take));
                        self.inner.$meth(v, r, adj_start, take);
                    } else {
                        self.inner.$meth(v, r, start + delta, take);
                    }
                }
            }
        }
    };
}

impl<'a, V> PrimitiveSortedWithRowIdsVisitor for PaginateVisitor<'a, V>
where
    V: PrimitiveSortedWithRowIdsVisitor,
{
    impl_sorted_with_rids_paginate_for_type!(u64_run_with_rids, UInt64Array);
    impl_sorted_with_rids_paginate_for_type!(u32_run_with_rids, UInt32Array);
    impl_sorted_with_rids_paginate_for_type!(u16_run_with_rids, UInt16Array);
    impl_sorted_with_rids_paginate_for_type!(u8_run_with_rids, UInt8Array);
    impl_sorted_with_rids_paginate_for_type!(i64_run_with_rids, Int64Array);
    impl_sorted_with_rids_paginate_for_type!(i32_run_with_rids, Int32Array);
    impl_sorted_with_rids_paginate_for_type!(i16_run_with_rids, Int16Array);
    impl_sorted_with_rids_paginate_for_type!(i8_run_with_rids, Int8Array);
    fn null_run(&mut self, r: &UInt64Array, start: usize, len: usize) {
        if self.done() {
            return;
        }
        if let Some((delta, take)) = self.split_len(len)
            && take > 0
        {
            if self.reverse {
                // For reverse, take from the end of the run
                let adj_start = start + (len - (delta + take));
                self.inner.null_run(r, adj_start, take);
            } else {
                self.inner.null_run(r, start + delta, take);
            }
        }
    }
}
