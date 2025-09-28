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
                        crate::store::scan::lower_idx_by(s, s + l, &key, |i| ($value_to_key)(a, i))
                    }
                    Bound::Excluded(x) => {
                        let key = ($bound_to_key)(x);
                        crate::store::scan::upper_idx_by(s, s + l, &key, |i| ($value_to_key)(a, i))
                    }
                };
                let end = match ub {
                    Bound::Unbounded => s + l,
                    Bound::Included(x) => {
                        let key = ($bound_to_key)(x);
                        crate::store::scan::upper_idx_by(s, s + l, &key, |i| ($value_to_key)(a, i))
                    }
                    Bound::Excluded(x) => {
                        let key = ($bound_to_key)(x);
                        crate::store::scan::lower_idx_by(s, s + l, &key, |i| ($value_to_key)(a, i))
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
                        crate::store::scan::lower_idx_by(s, s + l, &key, |i| ($value_to_key)(v, i))
                    }
                    Bound::Excluded(x) => {
                        let key = ($bound_to_key)(x);
                        crate::store::scan::upper_idx_by(s, s + l, &key, |i| ($value_to_key)(v, i))
                    }
                };
                let end = match ub {
                    Bound::Unbounded => s + l,
                    Bound::Included(x) => {
                        let key = ($bound_to_key)(x);
                        crate::store::scan::upper_idx_by(s, s + l, &key, |i| ($value_to_key)(v, i))
                    }
                    Bound::Excluded(x) => {
                        let key = ($bound_to_key)(x);
                        crate::store::scan::lower_idx_by(s, s + l, &key, |i| ($value_to_key)(v, i))
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
            || self.ir.i8_r.is_some()
            || self.ir.f64_r.is_some()
            || self.ir.f32_r.is_some();
        if !has_ranges {
            return self.store.scan(self.field_id, self.opts, visitor);
        }
        // If ranges configured and sorted path requested, handle inside the builder
        // with per-chunk pre-windowing to avoid per-run trimming overhead.
        if self.opts.sorted {
            return range_sorted_dispatch(self.store, self.field_id, self.opts, self.ir, visitor);
        }

        // TODO: Implement unsorted range filtering efficiently.
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
        }
        impl<'v, V> crate::store::scan::PrimitiveSortedVisitor for RangeAdapter<'v, V>
        where
            V: crate::store::scan::PrimitiveSortedVisitor,
        {
            impl_sorted_range_filter!(
                u64_run,
                UInt64Array,
                u64_r,
                |x| x,
                |array: &UInt64Array, idx| array.value(idx)
            );
            impl_sorted_range_filter!(
                u32_run,
                UInt32Array,
                u32_r,
                |x| x,
                |array: &UInt32Array, idx| array.value(idx)
            );
            impl_sorted_range_filter!(
                u16_run,
                UInt16Array,
                u16_r,
                |x| x,
                |array: &UInt16Array, idx| array.value(idx)
            );
            impl_sorted_range_filter!(
                u8_run,
                UInt8Array,
                u8_r,
                |x| x,
                |array: &UInt8Array, idx| array.value(idx)
            );
            impl_sorted_range_filter!(
                i64_run,
                Int64Array,
                i64_r,
                |x| x,
                |array: &Int64Array, idx| array.value(idx)
            );
            impl_sorted_range_filter!(
                i32_run,
                Int32Array,
                i32_r,
                |x| x,
                |array: &Int32Array, idx| array.value(idx)
            );
            impl_sorted_range_filter!(
                i16_run,
                Int16Array,
                i16_r,
                |x| x,
                |array: &Int16Array, idx| array.value(idx)
            );
            impl_sorted_range_filter!(i8_run, Int8Array, i8_r, |x| x, |array: &Int8Array, idx| {
                array.value(idx)
            });
            impl_sorted_range_filter!(
                f64_run,
                Float64Array,
                f64_r,
                F64Key::new,
                |array: &Float64Array, idx| F64Key::new(array.value(idx))
            );
            impl_sorted_range_filter!(
                f32_run,
                Float32Array,
                f32_r,
                F32Key::new,
                |array: &Float32Array, idx| F32Key::new(array.value(idx))
            );
        }
        impl<'v, V> crate::store::scan::PrimitiveSortedWithRowIdsVisitor for RangeAdapter<'v, V>
        where
            V: crate::store::scan::PrimitiveSortedWithRowIdsVisitor,
        {
            impl_sorted_with_rids_range_filter!(
                u64_run_with_rids,
                UInt64Array,
                u64_r,
                |x| x,
                |array: &UInt64Array, idx| array.value(idx)
            );
            impl_sorted_with_rids_range_filter!(
                u32_run_with_rids,
                UInt32Array,
                u32_r,
                |x| x,
                |array: &UInt32Array, idx| array.value(idx)
            );
            impl_sorted_with_rids_range_filter!(
                u16_run_with_rids,
                UInt16Array,
                u16_r,
                |x| x,
                |array: &UInt16Array, idx| array.value(idx)
            );
            impl_sorted_with_rids_range_filter!(
                u8_run_with_rids,
                UInt8Array,
                u8_r,
                |x| x,
                |array: &UInt8Array, idx| array.value(idx)
            );
            impl_sorted_with_rids_range_filter!(
                i64_run_with_rids,
                Int64Array,
                i64_r,
                |x| x,
                |array: &Int64Array, idx| array.value(idx)
            );
            impl_sorted_with_rids_range_filter!(
                i32_run_with_rids,
                Int32Array,
                i32_r,
                |x| x,
                |array: &Int32Array, idx| array.value(idx)
            );
            impl_sorted_with_rids_range_filter!(
                i16_run_with_rids,
                Int16Array,
                i16_r,
                |x| x,
                |array: &Int16Array, idx| array.value(idx)
            );
            impl_sorted_with_rids_range_filter!(
                i8_run_with_rids,
                Int8Array,
                i8_r,
                |x| x,
                |array: &Int8Array, idx| array.value(idx)
            );
            impl_sorted_with_rids_range_filter!(
                f64_run_with_rids,
                Float64Array,
                f64_r,
                F64Key::new,
                |array: &Float64Array, idx| F64Key::new(array.value(idx))
            );
            impl_sorted_with_rids_range_filter!(
                f32_run_with_rids,
                Float32Array,
                f32_r,
                F32Key::new,
                |array: &Float32Array, idx| F32Key::new(array.value(idx))
            );
        }

        let mut adapter = RangeAdapter {
            inner: visitor,
            ir: self.ir,
        };
        self.store.scan(self.field_id, self.opts, &mut adapter)
    }
}
