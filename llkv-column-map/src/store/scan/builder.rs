use super::*;
use std::cmp::Ordering;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    UInt8Array, UInt16Array, UInt32Array, UInt64Array, new_null_array,
};
use arrow::datatypes::DataType;

use crate::store::rowid::rowid_fid;

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

        // Binary search helpers for sorted runs
        #[inline]
        fn lower_idx<T: Ord, F: Fn(usize) -> T>(
            mut lo: usize,
            mut hi: usize,
            pred: &T,
            get: F,
        ) -> usize {
            while lo < hi {
                let mid = (lo + hi) >> 1;
                if get(mid) < *pred {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        }
        #[inline]
        fn upper_idx<T: Ord, F: Fn(usize) -> T>(
            mut lo: usize,
            mut hi: usize,
            pred: &T,
            get: F,
        ) -> usize {
            while lo < hi {
                let mid = (lo + hi) >> 1;
                if get(mid) <= *pred {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        }

        impl<'v, V> crate::store::scan::PrimitiveSortedVisitor for RangeAdapter<'v, V>
        where
            V: crate::store::scan::PrimitiveSortedVisitor,
        {
            fn u64_run(&mut self, a: &UInt64Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.u64_r {
                    let start = match lb {
                        Bound::Unbounded => s,
                        Bound::Included(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    let end = match ub {
                        Bound::Unbounded => s + l,
                        Bound::Included(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    if start < end {
                        self.inner.u64_run(a, start, end - start);
                    }
                } else {
                    self.inner.u64_run(a, s, l);
                }
            }
            fn u32_run(&mut self, a: &UInt32Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.u32_r {
                    let start = match lb {
                        Bound::Unbounded => s,
                        Bound::Included(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    let end = match ub {
                        Bound::Unbounded => s + l,
                        Bound::Included(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    if start < end {
                        self.inner.u32_run(a, start, end - start);
                    }
                } else {
                    self.inner.u32_run(a, s, l);
                }
            }
            fn u16_run(&mut self, a: &UInt16Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.u16_r {
                    let start = match lb {
                        Bound::Unbounded => s,
                        Bound::Included(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    let end = match ub {
                        Bound::Unbounded => s + l,
                        Bound::Included(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    if start < end {
                        self.inner.u16_run(a, start, end - start);
                    }
                } else {
                    self.inner.u16_run(a, s, l);
                }
            }
            fn u8_run(&mut self, a: &UInt8Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.u8_r {
                    let start = match lb {
                        Bound::Unbounded => s,
                        Bound::Included(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    let end = match ub {
                        Bound::Unbounded => s + l,
                        Bound::Included(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    if start < end {
                        self.inner.u8_run(a, start, end - start);
                    }
                } else {
                    self.inner.u8_run(a, s, l);
                }
            }
            fn i64_run(&mut self, a: &Int64Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.i64_r {
                    let start = match lb {
                        Bound::Unbounded => s,
                        Bound::Included(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    let end = match ub {
                        Bound::Unbounded => s + l,
                        Bound::Included(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    if start < end {
                        self.inner.i64_run(a, start, end - start);
                    }
                } else {
                    self.inner.i64_run(a, s, l);
                }
            }
            fn i32_run(&mut self, a: &Int32Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.i32_r {
                    let start = match lb {
                        Bound::Unbounded => s,
                        Bound::Included(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    let end = match ub {
                        Bound::Unbounded => s + l,
                        Bound::Included(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    if start < end {
                        self.inner.i32_run(a, start, end - start);
                    }
                } else {
                    self.inner.i32_run(a, s, l);
                }
            }
            fn i16_run(&mut self, a: &Int16Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.i16_r {
                    let start = match lb {
                        Bound::Unbounded => s,
                        Bound::Included(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    let end = match ub {
                        Bound::Unbounded => s + l,
                        Bound::Included(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    if start < end {
                        self.inner.i16_run(a, start, end - start);
                    }
                } else {
                    self.inner.i16_run(a, s, l);
                }
            }
            fn i8_run(&mut self, a: &Int8Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.i8_r {
                    let start = match lb {
                        Bound::Unbounded => s,
                        Bound::Included(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    let end = match ub {
                        Bound::Unbounded => s + l,
                        Bound::Included(x) => upper_idx(s, s + l, &x, |i| a.value(i)),
                        Bound::Excluded(x) => lower_idx(s, s + l, &x, |i| a.value(i)),
                    };
                    if start < end {
                        self.inner.i8_run(a, start, end - start);
                    }
                } else {
                    self.inner.i8_run(a, s, l);
                }
            }
            fn f64_run(&mut self, a: &Float64Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.f64_r {
                    let start = match lb {
                        Bound::Unbounded => s,
                        Bound::Included(x) => {
                            let key = F64Key::new(x);
                            lower_idx_by(s, s + l, &key, |i| F64Key::new(a.value(i)))
                        }
                        Bound::Excluded(x) => {
                            let key = F64Key::new(x);
                            upper_idx_by(s, s + l, &key, |i| F64Key::new(a.value(i)))
                        }
                    };
                    let end = match ub {
                        Bound::Unbounded => s + l,
                        Bound::Included(x) => {
                            let key = F64Key::new(x);
                            upper_idx_by(s, s + l, &key, |i| F64Key::new(a.value(i)))
                        }
                        Bound::Excluded(x) => {
                            let key = F64Key::new(x);
                            lower_idx_by(s, s + l, &key, |i| F64Key::new(a.value(i)))
                        }
                    };
                    if start < end {
                        self.inner.f64_run(a, start, end - start);
                    }
                } else {
                    self.inner.f64_run(a, s, l);
                }
            }
            fn f32_run(&mut self, a: &Float32Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.f32_r {
                    let start = match lb {
                        Bound::Unbounded => s,
                        Bound::Included(x) => {
                            let key = F32Key::new(x);
                            lower_idx_by(s, s + l, &key, |i| F32Key::new(a.value(i)))
                        }
                        Bound::Excluded(x) => {
                            let key = F32Key::new(x);
                            upper_idx_by(s, s + l, &key, |i| F32Key::new(a.value(i)))
                        }
                    };
                    let end = match ub {
                        Bound::Unbounded => s + l,
                        Bound::Included(x) => {
                            let key = F32Key::new(x);
                            upper_idx_by(s, s + l, &key, |i| F32Key::new(a.value(i)))
                        }
                        Bound::Excluded(x) => {
                            let key = F32Key::new(x);
                            lower_idx_by(s, s + l, &key, |i| F32Key::new(a.value(i)))
                        }
                    };
                    if start < end {
                        self.inner.f32_run(a, start, end - start);
                    }
                } else {
                    self.inner.f32_run(a, s, l);
                }
            }
        }
        impl<'v, V> crate::store::scan::PrimitiveSortedWithRowIdsVisitor for RangeAdapter<'v, V>
        where
            V: crate::store::scan::PrimitiveSortedWithRowIdsVisitor,
        {
            fn u64_run_with_rids(&mut self, v: &UInt64Array, r: &UInt64Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.u64_r {
                    let start = match lb {
                        Bound::Unbounded => s,
                        Bound::Included(x) => lower_idx(s, s + l, &x, |i| v.value(i)),
                        Bound::Excluded(x) => upper_idx(s, s + l, &x, |i| v.value(i)),
                    };
                    let end = match ub {
                        Bound::Unbounded => s + l,
                        Bound::Included(x) => upper_idx(s, s + l, &x, |i| v.value(i)),
                        Bound::Excluded(x) => lower_idx(s, s + l, &x, |i| v.value(i)),
                    };
                    if start < end {
                        self.inner.u64_run_with_rids(v, r, start, end - start);
                    }
                } else {
                    self.inner.u64_run_with_rids(v, r, s, l);
                }
            }
            fn i32_run_with_rids(&mut self, v: &Int32Array, r: &UInt64Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.i32_r {
                    let start = match lb {
                        Bound::Unbounded => s,
                        Bound::Included(x) => lower_idx(s, s + l, &x, |i| v.value(i)),
                        Bound::Excluded(x) => upper_idx(s, s + l, &x, |i| v.value(i)),
                    };
                    let end = match ub {
                        Bound::Unbounded => s + l,
                        Bound::Included(x) => upper_idx(s, s + l, &x, |i| v.value(i)),
                        Bound::Excluded(x) => lower_idx(s, s + l, &x, |i| v.value(i)),
                    };
                    if start < end {
                        self.inner.i32_run_with_rids(v, r, start, end - start);
                    }
                } else {
                    self.inner.i32_run_with_rids(v, r, s, l);
                }
            }
            // For brevity, other integer widths with row ids fall back to pass-through.
            fn f64_run_with_rids(&mut self, v: &Float64Array, r: &UInt64Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.f64_r {
                    let start = match lb {
                        Bound::Unbounded => s,
                        Bound::Included(x) => {
                            let key = F64Key::new(x);
                            lower_idx_by(s, s + l, &key, |i| F64Key::new(v.value(i)))
                        }
                        Bound::Excluded(x) => {
                            let key = F64Key::new(x);
                            upper_idx_by(s, s + l, &key, |i| F64Key::new(v.value(i)))
                        }
                    };
                    let end = match ub {
                        Bound::Unbounded => s + l,
                        Bound::Included(x) => {
                            let key = F64Key::new(x);
                            upper_idx_by(s, s + l, &key, |i| F64Key::new(v.value(i)))
                        }
                        Bound::Excluded(x) => {
                            let key = F64Key::new(x);
                            lower_idx_by(s, s + l, &key, |i| F64Key::new(v.value(i)))
                        }
                    };
                    if start < end {
                        self.inner.f64_run_with_rids(v, r, start, end - start);
                    }
                } else {
                    self.inner.f64_run_with_rids(v, r, s, l);
                }
            }
            fn f32_run_with_rids(&mut self, v: &Float32Array, r: &UInt64Array, s: usize, l: usize) {
                if let Some((lb, ub)) = self.ir.f32_r {
                    let start = match lb {
                        Bound::Unbounded => s,
                        Bound::Included(x) => {
                            let key = F32Key::new(x);
                            lower_idx_by(s, s + l, &key, |i| F32Key::new(v.value(i)))
                        }
                        Bound::Excluded(x) => {
                            let key = F32Key::new(x);
                            upper_idx_by(s, s + l, &key, |i| F32Key::new(v.value(i)))
                        }
                    };
                    let end = match ub {
                        Bound::Unbounded => s + l,
                        Bound::Included(x) => {
                            let key = F32Key::new(x);
                            upper_idx_by(s, s + l, &key, |i| F32Key::new(v.value(i)))
                        }
                        Bound::Excluded(x) => {
                            let key = F32Key::new(x);
                            lower_idx_by(s, s + l, &key, |i| F32Key::new(v.value(i)))
                        }
                    };
                    if start < end {
                        self.inner.f32_run_with_rids(v, r, start, end - start);
                    }
                } else {
                    self.inner.f32_run_with_rids(v, r, s, l);
                }
            }
        }

        let mut adapter = RangeAdapter {
            inner: visitor,
            ir: self.ir,
        };
        self.store.scan(self.field_id, self.opts, &mut adapter)
    }

    pub fn project<F>(mut self, mut on_batch: F) -> Result<()>
    where
        F: FnMut(ProjectionBatch) -> Result<()>,
    {
        let dtype = self.store.data_type(self.field_id)?;
        self.opts.with_row_ids = true;
        if self.opts.include_nulls && self.opts.anchor_row_id_field.is_none() {
            self.opts.anchor_row_id_field = Some(rowid_fid(self.field_id));
        }

        let mut visitor = ProjectionVisitor::new(dtype, &mut on_batch);
        self.run(&mut visitor)?;
        visitor.finish()
    }
}

/// A logical batch produced by `ScanBuilder::project`, containing the requested row ids and values.
pub struct ProjectionBatch {
    pub row_ids: Arc<UInt64Array>,
    pub values: ArrayRef,
}

struct ProjectionVisitor<'a, F>
where
    F: FnMut(ProjectionBatch) -> Result<()>,
{
    dtype: DataType,
    on_batch: &'a mut F,
    error: Option<Error>,
}

impl<'a, F> ProjectionVisitor<'a, F>
where
    F: FnMut(ProjectionBatch) -> Result<()>,
{
    fn new(dtype: DataType, on_batch: &'a mut F) -> Self {
        Self {
            dtype,
            on_batch,
            error: None,
        }
    }

    fn emit(&mut self, values: ArrayRef, row_ids: Arc<UInt64Array>) {
        if self.error.is_some() {
            return;
        }
        if let Err(err) = (self.on_batch)(ProjectionBatch { row_ids, values }) {
            self.error = Some(err);
        }
    }

    fn emit_null_run(&mut self, row_ids: &UInt64Array, start: usize, len: usize) {
        let rid = Arc::new(row_ids.slice(start, len));
        let values = new_null_array(&self.dtype, len);
        self.emit(values, rid);
    }

    fn finish(self) -> Result<()> {
        if let Some(err) = self.error {
            Err(err)
        } else {
            Ok(())
        }
    }
}

impl<'a, F> crate::store::scan::PrimitiveVisitor for ProjectionVisitor<'a, F>
where
    F: FnMut(ProjectionBatch) -> Result<()>,
{
    fn u64_chunk(&mut self, _a: &UInt64Array) {}
    fn u32_chunk(&mut self, _a: &UInt32Array) {}
    fn u16_chunk(&mut self, _a: &UInt16Array) {}
    fn u8_chunk(&mut self, _a: &UInt8Array) {}
    fn i64_chunk(&mut self, _a: &Int64Array) {}
    fn i32_chunk(&mut self, _a: &Int32Array) {}
    fn i16_chunk(&mut self, _a: &Int16Array) {}
    fn i8_chunk(&mut self, _a: &Int8Array) {}
    fn f64_chunk(&mut self, _a: &Float64Array) {}
    fn f32_chunk(&mut self, _a: &Float32Array) {}
}

impl<'a, F> crate::store::scan::PrimitiveSortedVisitor for ProjectionVisitor<'a, F>
where
    F: FnMut(ProjectionBatch) -> Result<()>,
{
    fn u64_run(&mut self, _a: &UInt64Array, _start: usize, _len: usize) {}
    fn u32_run(&mut self, _a: &UInt32Array, _start: usize, _len: usize) {}
    fn u16_run(&mut self, _a: &UInt16Array, _start: usize, _len: usize) {}
    fn u8_run(&mut self, _a: &UInt8Array, _start: usize, _len: usize) {}
    fn i64_run(&mut self, _a: &Int64Array, _start: usize, _len: usize) {}
    fn i32_run(&mut self, _a: &Int32Array, _start: usize, _len: usize) {}
    fn i16_run(&mut self, _a: &Int16Array, _start: usize, _len: usize) {}
    fn i8_run(&mut self, _a: &Int8Array, _start: usize, _len: usize) {}
    fn f64_run(&mut self, _a: &Float64Array, _start: usize, _len: usize) {}
    fn f32_run(&mut self, _a: &Float32Array, _start: usize, _len: usize) {}
}

macro_rules! impl_projection_with_rids {
    ($meth:ident, $ArrTy:ty) => {
        fn $meth(&mut self, values: &$ArrTy, row_ids: &UInt64Array) {
            let values_ref = values.clone();
            let row_ids_ref = row_ids.clone();
            self.emit(Arc::new(values_ref) as ArrayRef, Arc::new(row_ids_ref));
        }
    };
}

macro_rules! impl_projection_sorted_with_rids {
    ($meth:ident, $ArrTy:ty) => {
        fn $meth(&mut self, values: &$ArrTy, row_ids: &UInt64Array, start: usize, len: usize) {
            let v_slice = values.slice(start, len);
            let r_slice = row_ids.slice(start, len);
            self.emit(Arc::new(v_slice) as ArrayRef, Arc::new(r_slice));
        }
    };
}

impl<'a, F> crate::store::scan::PrimitiveWithRowIdsVisitor for ProjectionVisitor<'a, F>
where
    F: FnMut(ProjectionBatch) -> Result<()>,
{
    impl_projection_with_rids!(u64_chunk_with_rids, UInt64Array);
    impl_projection_with_rids!(u32_chunk_with_rids, UInt32Array);
    impl_projection_with_rids!(u16_chunk_with_rids, UInt16Array);
    impl_projection_with_rids!(u8_chunk_with_rids, UInt8Array);
    impl_projection_with_rids!(i64_chunk_with_rids, Int64Array);
    impl_projection_with_rids!(i32_chunk_with_rids, Int32Array);
    impl_projection_with_rids!(i16_chunk_with_rids, Int16Array);
    impl_projection_with_rids!(i8_chunk_with_rids, Int8Array);
    impl_projection_with_rids!(f64_chunk_with_rids, Float64Array);
    impl_projection_with_rids!(f32_chunk_with_rids, Float32Array);
}

impl<'a, F> crate::store::scan::PrimitiveSortedWithRowIdsVisitor for ProjectionVisitor<'a, F>
where
    F: FnMut(ProjectionBatch) -> Result<()>,
{
    impl_projection_sorted_with_rids!(u64_run_with_rids, UInt64Array);
    impl_projection_sorted_with_rids!(u32_run_with_rids, UInt32Array);
    impl_projection_sorted_with_rids!(u16_run_with_rids, UInt16Array);
    impl_projection_sorted_with_rids!(u8_run_with_rids, UInt8Array);
    impl_projection_sorted_with_rids!(i64_run_with_rids, Int64Array);
    impl_projection_sorted_with_rids!(i32_run_with_rids, Int32Array);
    impl_projection_sorted_with_rids!(i16_run_with_rids, Int16Array);
    impl_projection_sorted_with_rids!(i8_run_with_rids, Int8Array);
    impl_projection_sorted_with_rids!(f64_run_with_rids, Float64Array);
    impl_projection_sorted_with_rids!(f32_run_with_rids, Float32Array);

    fn null_run(&mut self, row_ids: &UInt64Array, start: usize, len: usize) {
        self.emit_null_run(row_ids, start, len);
    }
}
