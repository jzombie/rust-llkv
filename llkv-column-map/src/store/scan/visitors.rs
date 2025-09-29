use super::*;

macro_rules! declare_primitive_trait_methods {
    (
        $base:ident,
        $chunk_fn:ident,
        $chunk_with_rids_fn:ident,
        $run_fn:ident,
        $run_with_rids_fn:ident,
        $array_ty:ty,
        $physical_ty:ty,
        $dtype:expr,
        $native_ty:ty,
        $cast_expr:expr
    ) => {
        fn $chunk_fn(&mut self, _a: &$array_ty) {
            unimplemented!(concat!("`", stringify!($chunk_fn), "` not implemented"))
        }
    };
}

macro_rules! declare_primitive_with_rids_trait_methods {
    (
        $base:ident,
        $chunk_fn:ident,
        $chunk_with_rids_fn:ident,
        $run_fn:ident,
        $run_with_rids_fn:ident,
        $array_ty:ty,
        $physical_ty:ty,
        $dtype:expr,
        $native_ty:ty,
        $cast_expr:expr
    ) => {
        fn $chunk_with_rids_fn(&mut self, _v: &$array_ty, _r: &UInt64Array) {
            unimplemented!(concat!(
                "`",
                stringify!($chunk_with_rids_fn),
                "` not implemented"
            ))
        }
    };
}

macro_rules! declare_sorted_trait_methods {
    (
        $base:ident,
        $chunk_fn:ident,
        $chunk_with_rids_fn:ident,
        $run_fn:ident,
        $run_with_rids_fn:ident,
        $array_ty:ty,
        $physical_ty:ty,
        $dtype:expr,
        $native_ty:ty,
        $cast_expr:expr
    ) => {
        fn $run_fn(&mut self, _a: &$array_ty, _start: usize, _len: usize) {
            unimplemented!(concat!("`", stringify!($run_fn), "` not implemented"))
        }
    };
}

macro_rules! declare_sorted_with_rids_trait_methods {
    (
        $base:ident,
        $chunk_fn:ident,
        $chunk_with_rids_fn:ident,
        $run_fn:ident,
        $run_with_rids_fn:ident,
        $array_ty:ty,
        $physical_ty:ty,
        $dtype:expr,
        $native_ty:ty,
        $cast_expr:expr
    ) => {
        fn $run_with_rids_fn(
            &mut self,
            _v: &$array_ty,
            _r: &UInt64Array,
            _start: usize,
            _len: usize,
        ) {
        }
    };
}

/// Unsorted primitive visitor: one callback per chunk per type.
pub trait PrimitiveVisitor {
    llkv_for_each_arrow_numeric!(declare_primitive_trait_methods);
}

/// Unsorted primitive visitor with row ids (u64).
pub trait PrimitiveWithRowIdsVisitor {
    llkv_for_each_arrow_numeric!(declare_primitive_with_rids_trait_methods);
}

/// Sorted visitor fed with coalesced runs (start,len) within a typed array.
pub trait PrimitiveSortedVisitor {
    llkv_for_each_arrow_numeric!(declare_sorted_trait_methods);
}

/// Sorted visitor with row ids.
pub trait PrimitiveSortedWithRowIdsVisitor {
    llkv_for_each_arrow_numeric!(declare_sorted_with_rids_trait_methods);
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

macro_rules! expand_unsorted_paginate {
    (
        $base:ident,
        $chunk_fn:ident,
        $chunk_with_rids_fn:ident,
        $run_fn:ident,
        $run_with_rids_fn:ident,
        $array_ty:ty,
        $physical_ty:ty,
        $dtype:expr,
        $native_ty:ty,
        $cast_expr:expr
    ) => {
        impl_unsorted_paginate_for_type!($chunk_fn, $array_ty);
    };
}

impl<'a, V> PrimitiveVisitor for PaginateVisitor<'a, V>
where
    V: PrimitiveVisitor,
{
    llkv_for_each_arrow_numeric!(expand_unsorted_paginate);
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

macro_rules! expand_unsorted_with_rids_paginate {
    (
        $base:ident,
        $chunk_fn:ident,
        $chunk_with_rids_fn:ident,
        $run_fn:ident,
        $run_with_rids_fn:ident,
        $array_ty:ty,
        $physical_ty:ty,
        $dtype:expr,
        $native_ty:ty,
        $cast_expr:expr
    ) => {
        impl_unsorted_with_rids_paginate_for_type!($chunk_with_rids_fn, $array_ty);
    };
}

impl<'a, V> PrimitiveWithRowIdsVisitor for PaginateVisitor<'a, V>
where
    V: PrimitiveWithRowIdsVisitor,
{
    llkv_for_each_arrow_numeric!(expand_unsorted_with_rids_paginate);
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

macro_rules! expand_sorted_paginate {
    (
        $base:ident,
        $chunk_fn:ident,
        $chunk_with_rids_fn:ident,
        $run_fn:ident,
        $run_with_rids_fn:ident,
        $array_ty:ty,
        $physical_ty:ty,
        $dtype:expr,
        $native_ty:ty,
        $cast_expr:expr
    ) => {
        impl_sorted_paginate_for_type!($run_fn, $array_ty);
    };
}

impl<'a, V> PrimitiveSortedVisitor for PaginateVisitor<'a, V>
where
    V: PrimitiveSortedVisitor,
{
    llkv_for_each_arrow_numeric!(expand_sorted_paginate);
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

macro_rules! expand_sorted_with_rids_paginate {
    (
        $base:ident,
        $chunk_fn:ident,
        $chunk_with_rids_fn:ident,
        $run_fn:ident,
        $run_with_rids_fn:ident,
        $array_ty:ty,
        $physical_ty:ty,
        $dtype:expr,
        $native_ty:ty,
        $cast_expr:expr
    ) => {
        impl_sorted_with_rids_paginate_for_type!($run_with_rids_fn, $array_ty);
    };
}

impl<'a, V> PrimitiveSortedWithRowIdsVisitor for PaginateVisitor<'a, V>
where
    V: PrimitiveSortedWithRowIdsVisitor,
{
    llkv_for_each_arrow_numeric!(expand_sorted_with_rids_paginate);
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
