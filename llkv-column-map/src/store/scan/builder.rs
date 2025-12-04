use super::*;
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
struct F64Key(f64);

impl F64Key {
    #[inline]
    fn new(v: f64) -> Self {
        Self(v)
    }
}

impl PartialEq for F64Key {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for F64Key {}

impl PartialOrd for F64Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for F64Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[derive(Clone, Copy, Debug)]
struct F32Key(f32);

impl F32Key {
    #[inline]
    fn new(v: f32) -> Self {
        Self(v)
    }
}

impl PartialEq for F32Key {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for F32Key {}

impl PartialOrd for F32Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for F32Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

macro_rules! impl_sorted_range_filter {
    ($fn_name:ident, $ArrTy:ty, $range_field:ident, $bound_to_key:expr, $value_to_key:expr) => {
        fn $fn_name(&mut self, a: &$ArrTy, s: usize, l: usize) {
            if let Some((lb, ub)) = self.ir.$range_field {
                let start = match lb {
                    Bound::Unbounded => s,
                    Bound::Included(x) => {
                        let key = ($bound_to_key)(x);
                        crate::store::scan::lower_idx_by(s, s + l, &key, &|i| ($value_to_key)(a, i))
                    }
                    Bound::Excluded(x) => {
                        let key = ($bound_to_key)(x);
                        crate::store::scan::upper_idx_by(s, s + l, &key, &|i| ($value_to_key)(a, i))
                    }
                };
                let end = match ub {
                    Bound::Unbounded => s + l,
                    Bound::Included(x) => {
                        let key = ($bound_to_key)(x);
                        crate::store::scan::upper_idx_by(s, s + l, &key, &|i| ($value_to_key)(a, i))
                    }
                    Bound::Excluded(x) => {
                        let key = ($bound_to_key)(x);
                        crate::store::scan::lower_idx_by(s, s + l, &key, &|i| ($value_to_key)(a, i))
                    }
                };
                if start < end {
                    self.inner.$fn_name(a, start, end - start);
                }
            } else {
                self.inner.$fn_name(a, s, l);
            }
        }
    };
}

macro_rules! impl_sorted_with_rids_range_filter {
    ($fn_name:ident, $ArrTy:ty, $range_field:ident, $bound_to_key:expr, $value_to_key:expr) => {
        fn $fn_name(&mut self, v: &$ArrTy, r: &UInt64Array, s: usize, l: usize) {
            if let Some((lb, ub)) = self.ir.$range_field {
                let start = match lb {
                    Bound::Unbounded => s,
                    Bound::Included(x) => {
                        let key = ($bound_to_key)(x);
                        crate::store::scan::lower_idx_by(s, s + l, &key, &|i| ($value_to_key)(v, i))
                    }
                    Bound::Excluded(x) => {
                        let key = ($bound_to_key)(x);
                        crate::store::scan::upper_idx_by(s, s + l, &key, &|i| ($value_to_key)(v, i))
                    }
                };
                let end = match ub {
                    Bound::Unbounded => s + l,
                    Bound::Included(x) => {
                        let key = ($bound_to_key)(x);
                        crate::store::scan::upper_idx_by(s, s + l, &key, &|i| ($value_to_key)(v, i))
                    }
                    Bound::Excluded(x) => {
                        let key = ($bound_to_key)(x);
                        crate::store::scan::lower_idx_by(s, s + l, &key, &|i| ($value_to_key)(v, i))
                    }
                };
                if start < end {
                    self.inner.$fn_name(v, r, start, end - start);
                }
            } else {
                self.inner.$fn_name(v, r, s, l);
            }
        }
    };
}

macro_rules! declare_sorted_range_filter_for_type {
    (u64, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_range_filter!(
            $run_fn,
            $array_ty,
            u64_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (u32, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_range_filter!(
            $run_fn,
            $array_ty,
            u32_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (u16, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_range_filter!(
            $run_fn,
            $array_ty,
            u16_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (u8, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_range_filter!($run_fn, $array_ty, u8_r, |x| x, |array: &$array_ty, idx| {
            array.value(idx)
        });
    };
    (i64, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_range_filter!(
            $run_fn,
            $array_ty,
            i64_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (i32, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_range_filter!(
            $run_fn,
            $array_ty,
            i32_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (i16, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_range_filter!(
            $run_fn,
            $array_ty,
            i16_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (i8, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_range_filter!($run_fn, $array_ty, i8_r, |x| x, |array: &$array_ty, idx| {
            array.value(idx)
        });
    };
    (f64, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_range_filter!(
            $run_fn,
            $array_ty,
            f64_r,
            F64Key::new,
            |array: &$array_ty, idx| F64Key::new(array.value(idx))
        );
    };
    (f32, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_range_filter!(
            $run_fn,
            $array_ty,
            f32_r,
            F32Key::new,
            |array: &$array_ty, idx| F32Key::new(array.value(idx))
        );
    };
    (bool, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_range_filter!(
            $run_fn,
            $array_ty,
            bool_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (date64, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_range_filter!(
            $run_fn,
            $array_ty,
            i64_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (date32, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_range_filter!(
            $run_fn,
            $array_ty,
            i32_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
}

macro_rules! declare_sorted_with_rids_range_filter_for_type {
    (u64, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_with_rids_range_filter!(
            $run_with_rids_fn,
            $array_ty,
            u64_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (u32, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_with_rids_range_filter!(
            $run_with_rids_fn,
            $array_ty,
            u32_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (u16, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_with_rids_range_filter!(
            $run_with_rids_fn,
            $array_ty,
            u16_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (u8, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_with_rids_range_filter!(
            $run_with_rids_fn,
            $array_ty,
            u8_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (i64, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_with_rids_range_filter!(
            $run_with_rids_fn,
            $array_ty,
            i64_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (i32, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_with_rids_range_filter!(
            $run_with_rids_fn,
            $array_ty,
            i32_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (i16, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_with_rids_range_filter!(
            $run_with_rids_fn,
            $array_ty,
            i16_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (i8, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_with_rids_range_filter!(
            $run_with_rids_fn,
            $array_ty,
            i8_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (f64, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_with_rids_range_filter!(
            $run_with_rids_fn,
            $array_ty,
            f64_r,
            F64Key::new,
            |array: &$array_ty, idx| F64Key::new(array.value(idx))
        );
    };
    (f32, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_with_rids_range_filter!(
            $run_with_rids_fn,
            $array_ty,
            f32_r,
            F32Key::new,
            |array: &$array_ty, idx| F32Key::new(array.value(idx))
        );
    };
    (bool, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_with_rids_range_filter!(
            $run_with_rids_fn,
            $array_ty,
            bool_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (date64, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_with_rids_range_filter!(
            $run_with_rids_fn,
            $array_ty,
            i64_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
    (date32, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype_expr:expr, $native_ty:ty, $cast_expr:expr) => {
        impl_sorted_with_rids_range_filter!(
            $run_with_rids_fn,
            $array_ty,
            i32_r,
            |x| x,
            |array: &$array_ty, idx| array.value(idx)
        );
    };
}

/// Configurable scanner for a single logical column.
///
/// Collects [`crate::store::scan::ScanOptions`] and typed range bounds before dispatching work into the visitor
/// pipeline.
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
        Self {
            store,
            field_id,
            opts: ScanOptions::default(),
            ir: IntRanges::default(),
        }
    }
    pub fn options(mut self, opts: ScanOptions) -> Self {
        self.opts = opts;
        self
    }
    pub fn with_row_ids(mut self, _row_id_field: LogicalFieldId) -> Self {
        self.opts.with_row_ids = true;
        self
    }
    pub fn sorted(mut self, sorted: bool) -> Self {
        self.opts.sorted = sorted;
        self
    }
    pub fn reverse(mut self, reverse: bool) -> Self {
        self.opts.reverse = reverse;
        self
    }

    pub fn with_ranges(mut self, ranges: IntRanges) -> Self {
        self.ir = ranges;
        self
    }

    // Generic, monomorphized range setter (no perf impact):
    // Usage: builder.with_range::<u64,_>(2000..=8000)
    pub fn with_range<T, R>(mut self, r: R) -> Self
    where
        T: RangeKey + Copy,
        R: RangeBounds<T>,
    {
        let lb_ref = r.start_bound();
        let ub_ref = r.end_bound();
        // Copy out of Bound<&T> into Bound<T> (T: Copy)
        let lb = match lb_ref {
            Bound::Unbounded => Bound::Unbounded,
            Bound::Included(&x) => Bound::Included(x),
            Bound::Excluded(&x) => Bound::Excluded(x),
        };
        let ub = match ub_ref {
            Bound::Unbounded => Bound::Unbounded,
            Bound::Included(&x) => Bound::Included(x),
            Bound::Excluded(&x) => Bound::Excluded(x),
        };
        T::store(&mut self.ir, lb, ub);
        self
    }

    pub fn run<V>(mut self, visitor: &mut V) -> Result<()>
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
            || self.ir.i8_r.is_some()
            || self.ir.f64_r.is_some()
            || self.ir.f32_r.is_some()
            || self.ir.bool_r.is_some();
        
        // Pass ranges to options so the store can use them for pruning
        self.opts.ranges = self.ir;

        if !has_ranges {
            return self.store.scan(self.field_id, self.opts, visitor);
        }
        // If ranges configured and sorted path requested, handle inside the builder
        // with per-chunk pre-windowing to avoid per-run trimming overhead.
        if self.opts.sorted {
            return range_sorted_dispatch(self.store, self.field_id, self.opts, self.ir, visitor);
        }

        // TODO: Implement unsorted range filtering efficiently.
        // NOTE: Unsorted scans currently defer range pruning until more efficient
        // adapters are available. The adapter below simply forwards all chunks.
        // Range-filtering adapter for unsorted runs. (Pass-through today.)
        struct RangeAdapter<'v, V> {
            inner: &'v mut V,
            ir: IntRanges,
        }
        impl<'v, V> crate::store::scan::PrimitiveVisitor for RangeAdapter<'v, V>
        where
            V: crate::store::scan::PrimitiveVisitor,
        {
            fn u64_chunk(&mut self, a: &UInt64Array) {
                self.inner.u64_chunk(a)
            }
            fn u32_chunk(&mut self, a: &UInt32Array) {
                self.inner.u32_chunk(a)
            }
            fn u16_chunk(&mut self, a: &UInt16Array) {
                self.inner.u16_chunk(a)
            }
            fn u8_chunk(&mut self, a: &UInt8Array) {
                self.inner.u8_chunk(a)
            }
            fn i64_chunk(&mut self, a: &Int64Array) {
                self.inner.i64_chunk(a)
            }
            fn i32_chunk(&mut self, a: &Int32Array) {
                self.inner.i32_chunk(a)
            }
            fn i16_chunk(&mut self, a: &Int16Array) {
                self.inner.i16_chunk(a)
            }
            fn i8_chunk(&mut self, a: &Int8Array) {
                self.inner.i8_chunk(a)
            }
            fn f64_chunk(&mut self, a: &Float64Array) {
                self.inner.f64_chunk(a)
            }
            fn f32_chunk(&mut self, a: &Float32Array) {
                self.inner.f32_chunk(a)
            }
            fn bool_chunk(&mut self, a: &BooleanArray) {
                self.inner.bool_chunk(a)
            }
            fn utf8_chunk(&mut self, a: &StringArray) {
                self.inner.utf8_chunk(a)
            }
            fn date64_chunk(&mut self, a: &Date64Array) {
                self.inner.date64_chunk(a)
            }
            fn date32_chunk(&mut self, a: &Date32Array) {
                self.inner.date32_chunk(a)
            }
        }
        impl<'v, V> crate::store::scan::PrimitiveWithRowIdsVisitor for RangeAdapter<'v, V>
        where
            V: crate::store::scan::PrimitiveWithRowIdsVisitor,
        {
            fn u64_chunk_with_rids(&mut self, v: &UInt64Array, r: &UInt64Array) {
                self.inner.u64_chunk_with_rids(v, r)
            }
            fn u32_chunk_with_rids(&mut self, v: &UInt32Array, r: &UInt64Array) {
                self.inner.u32_chunk_with_rids(v, r)
            }
            fn u16_chunk_with_rids(&mut self, v: &UInt16Array, r: &UInt64Array) {
                self.inner.u16_chunk_with_rids(v, r)
            }
            fn u8_chunk_with_rids(&mut self, v: &UInt8Array, r: &UInt64Array) {
                self.inner.u8_chunk_with_rids(v, r)
            }
            fn i64_chunk_with_rids(&mut self, v: &Int64Array, r: &UInt64Array) {
                self.inner.i64_chunk_with_rids(v, r)
            }
            fn i32_chunk_with_rids(&mut self, v: &Int32Array, r: &UInt64Array) {
                self.inner.i32_chunk_with_rids(v, r)
            }
            fn i16_chunk_with_rids(&mut self, v: &Int16Array, r: &UInt64Array) {
                self.inner.i16_chunk_with_rids(v, r)
            }
            fn i8_chunk_with_rids(&mut self, v: &Int8Array, r: &UInt64Array) {
                self.inner.i8_chunk_with_rids(v, r)
            }
            fn f64_chunk_with_rids(&mut self, v: &Float64Array, r: &UInt64Array) {
                self.inner.f64_chunk_with_rids(v, r)
            }
            fn f32_chunk_with_rids(&mut self, v: &Float32Array, r: &UInt64Array) {
                self.inner.f32_chunk_with_rids(v, r)
            }
            fn bool_chunk_with_rids(&mut self, v: &BooleanArray, r: &UInt64Array) {
                self.inner.bool_chunk_with_rids(v, r)
            }
            fn utf8_chunk_with_rids(&mut self, v: &StringArray, r: &UInt64Array) {
                self.inner.utf8_chunk_with_rids(v, r)
            }
            fn date64_chunk_with_rids(&mut self, v: &Date64Array, r: &UInt64Array) {
                self.inner.date64_chunk_with_rids(v, r)
            }
            fn date32_chunk_with_rids(&mut self, v: &Date32Array, r: &UInt64Array) {
                self.inner.date32_chunk_with_rids(v, r)
            }
        }
        impl<'v, V> crate::store::scan::PrimitiveSortedVisitor for RangeAdapter<'v, V>
        where
            V: crate::store::scan::PrimitiveSortedVisitor,
        {
            llkv_for_each_arrow_numeric!(declare_sorted_range_filter_for_type);
            llkv_for_each_arrow_boolean!(declare_sorted_range_filter_for_type);
        }
        impl<'v, V> crate::store::scan::PrimitiveSortedWithRowIdsVisitor for RangeAdapter<'v, V>
        where
            V: crate::store::scan::PrimitiveSortedWithRowIdsVisitor,
        {
            llkv_for_each_arrow_numeric!(declare_sorted_with_rids_range_filter_for_type);
            llkv_for_each_arrow_boolean!(declare_sorted_with_rids_range_filter_for_type);
        }

        let mut adapter = RangeAdapter {
            inner: visitor,
            ir: self.ir,
        };
        self.store.scan(self.field_id, self.opts, &mut adapter)
    }
}
