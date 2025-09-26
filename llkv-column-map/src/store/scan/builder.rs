use super::*;
use std::cmp::Ordering;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Int64Array, PrimitiveArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array, new_null_array,
};
use arrow::compute::{filter as arrow_filter, is_not_null, take};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_storage::pager::{BatchGet, GetResult};
use llkv_storage::types::PhysicalKey;
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;

use crate::store::{ROW_ID_COLUMN_NAME, rowid::rowid_fid};

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
    projection: Vec<LogicalFieldId>,
}

impl<'a, P> ScanBuilder<'a, P>
where
    P: Pager<Blob = EntryHandle>,
{
    /// Create a scan builder for the given logical column.
    ///
    /// The column passed here becomes the *driver* for subsequent scans: range, pagination, and
    /// index decisions are derived from it, and it will be the first value column in any projection.
    pub fn new(store: &'a ColumnStore<P>, field_id: LogicalFieldId) -> Self {
        Self {
            store,
            field_id,
            opts: ScanOptions::default(),
            ir: IntRanges::default(),
            projection: vec![field_id],
        }
    }

    /// Create a scan builder seeded with the provided projection list.
    ///
    /// `columns[0]` is treated as the driver column. If the slice is empty this falls back to the
    /// single-column constructor.
    pub fn with_columns(store: &'a ColumnStore<P>, columns: &[LogicalFieldId]) -> Self {
        assert!(
            !columns.is_empty(),
            "ScanBuilder::with_columns requires at least one logical field id"
        );
        let mut iter = columns.iter();
        let driver = *iter.next().unwrap();
        let mut builder = Self::new(store, driver);
        for &fid in iter {
            builder.push_projection(fid);
        }
        builder
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

    /// Extend the projection list with additional logical field ids.
    ///
    /// The list always keeps the driver column (passed to [`ScanBuilder::new`]) as the first
    /// entry. Additional ids are appended in the order supplied here; duplicates are ignored.
    /// Projection order controls both output column order and the sequence of chunk fetches for
    /// companion columns.
    pub fn with_projection(mut self, columns: &[LogicalFieldId]) -> Self {
        for &fid in columns {
            self.push_projection(fid);
        }
        self
    }

    fn push_projection(&mut self, fid: LogicalFieldId) {
        if fid == self.field_id || self.projection.contains(&fid) {
            if fid == self.field_id && self.projection.is_empty() {
                self.projection.push(self.field_id);
            }
            return;
        }
        self.projection.push(fid);
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

    /// Stream row-id aligned batches for the configured projection.
    ///
    /// The first projected column (by default the driver passed to [`ScanBuilder::new`]) defines
    /// chunk order and powers range/pagination logic. Additional columns are hydrated per chunk
    /// and combined into an Arrow `RecordBatch` alongside the row-id column. Callers receive each
    /// batch via `on_batch` and can stop early by returning an error.
    pub fn project<F>(mut self, mut on_batch: F) -> Result<()>
    where
        F: FnMut(MultiProjectionBatch) -> Result<()>,
    {
        if self.opts.sorted {
            return Err(Error::Internal(
                "project does not support sorted scans yet".into(),
            ));
        }
        self.opts.with_row_ids = true;
        if self.opts.include_nulls && self.opts.anchor_row_id_field.is_none() {
            self.opts.anchor_row_id_field = Some(rowid_fid(self.field_id));
        }

        if self.projection.is_empty() {
            self.projection.push(self.field_id);
        }
        if self.projection[0] != self.field_id {
            if let Some(pos) = self.projection.iter().position(|&fid| fid == self.field_id) {
                self.projection.swap(0, pos);
            } else {
                self.projection.insert(0, self.field_id);
            }
        }

        let mut seen: FxHashSet<LogicalFieldId> = FxHashSet::default();
        let mut projection: Vec<LogicalFieldId> = Vec::new();
        for &fid in &self.projection {
            if seen.insert(fid) {
                projection.push(fid);
            }
        }

        let base_dtype = self.store.data_type(self.field_id)?;

        let mut extras: Vec<PrefetchedColumn> = Vec::new();
        for &fid in projection.iter().skip(1) {
            extras.push(prefetch_unsorted_column(self.store, fid)?);
        }

        let mut schema_fields: Vec<Field> = Vec::with_capacity(2 + extras.len());
        schema_fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));
        schema_fields.push(Field::new(
            logical_field_name(self.field_id),
            base_dtype.clone(),
            true,
        ));
        for col in &extras {
            schema_fields.push(Field::new(
                logical_field_name(col.field_id),
                col.dtype.clone(),
                true,
            ));
        }
        let schema = Arc::new(Schema::new(schema_fields));

        let projection_arc: Arc<[LogicalFieldId]> = projection.clone().into();

        let mut visitor = MultiProjectionVisitor::new(
            base_dtype,
            schema,
            extras,
            Arc::clone(&projection_arc),
            &mut on_batch,
            self.opts.include_nulls,
        );
        self.run(&mut visitor)?;
        visitor.finish()
    }
}

/// A logical batch emitted by `ScanBuilder::project`, pairing row ids with one or more value
/// columns materialized as an Arrow `RecordBatch`.
pub struct MultiProjectionBatch {
    pub row_ids: Arc<UInt64Array>,
    pub record_batch: RecordBatch,
    pub columns: Arc<[LogicalFieldId]>,
}

struct PrefetchedColumn {
    field_id: LogicalFieldId,
    dtype: DataType,
    values: ArrayRef,
    row_ids: Arc<UInt64Array>,
    cursor: usize,
}

impl PrefetchedColumn {
    fn next_aligned(&mut self, base_row_ids: &UInt64Array) -> Result<ArrayRef> {
        if self.row_ids.is_empty() {
            return Ok(new_null_array(&self.dtype, base_row_ids.len()));
        }

        let mut indices: Vec<Option<u32>> = Vec::with_capacity(base_row_ids.len());
        let mut cursor = self.cursor;
        let rid_slice = self.row_ids.as_ref();
        let total = rid_slice.len();

        for base_idx in 0..base_row_ids.len() {
            let base_rid = base_row_ids.value(base_idx);
            while cursor < total && rid_slice.value(cursor) < base_rid {
                cursor += 1;
            }
            if cursor < total && rid_slice.value(cursor) == base_rid {
                indices.push(Some(cursor as u32));
                cursor += 1;
            } else {
                indices.push(None);
            }
        }

        self.cursor = cursor;

        let indices_array = UInt32Array::from(indices);
        let aligned = take(self.values.as_ref(), &indices_array, None).map_err(Error::from)?;
        Ok(aligned)
    }
}

struct MultiProjectionVisitor<'a, F>
where
    F: FnMut(MultiProjectionBatch) -> Result<()>,
{
    base_dtype: DataType,
    schema: Arc<Schema>,
    extras: Vec<PrefetchedColumn>,
    projection: Arc<[LogicalFieldId]>,
    on_batch: &'a mut F,
    include_nulls: bool,
    error: Option<Error>,
}

impl<'a, F> MultiProjectionVisitor<'a, F>
where
    F: FnMut(MultiProjectionBatch) -> Result<()>,
{
    fn new(
        base_dtype: DataType,
        schema: Arc<Schema>,
        extras: Vec<PrefetchedColumn>,
        projection: Arc<[LogicalFieldId]>,
        on_batch: &'a mut F,
        include_nulls: bool,
    ) -> Self {
        Self {
            base_dtype,
            schema,
            extras,
            projection,
            on_batch,
            include_nulls,
            error: None,
        }
    }

    fn emit_record(
        &mut self,
        row_ids: Arc<UInt64Array>,
        base: ArrayRef,
        extra_arrays: Vec<ArrayRef>,
    ) -> Result<()> {
        if base.len() != row_ids.len() {
            return Err(Error::Internal(
                "project: base column length mismatch".into(),
            ));
        }
        for (idx, arr) in extra_arrays.iter().enumerate() {
            let fid = self.extras.get(idx).map(|c| c.field_id);
            if arr.len() != row_ids.len() {
                return Err(Error::Internal(format!(
                    "project: extra column length mismatch (column {idx}, field {:?}, row_ids {}, extra {})",
                    fid,
                    row_ids.len(),
                    arr.len()
                )));
            }
        }

        let mut columns: Vec<ArrayRef> = Vec::with_capacity(2 + extra_arrays.len());
        columns.push(row_ids.clone() as ArrayRef);
        columns.push(base);
        columns.extend(extra_arrays);

        let batch = RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| Error::Internal(format!("project: {}", e)))?;

        (self.on_batch)(MultiProjectionBatch {
            row_ids,
            record_batch: batch,
            columns: Arc::clone(&self.projection),
        })
    }

    fn emit_chunk(&mut self, values: ArrayRef, row_ids: Arc<UInt64Array>) {
        if self.error.is_some() {
            return;
        }
        let mut extra_arrays: Vec<ArrayRef> = Vec::with_capacity(self.extras.len());
        for col in &mut self.extras {
            match col.next_aligned(row_ids.as_ref()) {
                Ok(aligned) => extra_arrays.push(aligned),
                Err(err) => {
                    self.error = Some(err);
                    return;
                }
            }
        }
        if let Err(err) = self.emit_record(row_ids, values, extra_arrays) {
            self.error = Some(err);
        }
    }

    fn emit_null_run(&mut self, row_ids: &UInt64Array, start: usize, len: usize) {
        if self.error.is_some() || len == 0 {
            return;
        }
        let rid_slice = Arc::new(row_ids.slice(start, len));
        let base = new_null_array(&self.base_dtype, len);
        let extras = self
            .extras
            .iter()
            .map(|col| new_null_array(&col.dtype, len))
            .collect();
        if let Err(err) = self.emit_record(rid_slice, base, extras) {
            self.error = Some(err);
        }
    }

    fn sorted_not_supported(&mut self) {
        if self.error.is_none() {
            self.error = Some(Error::Internal(
                "project: sorted scans are not supported".into(),
            ));
        }
    }

    fn finish(self) -> Result<()> {
        if let Some(err) = self.error {
            Err(err)
        } else {
            Ok(())
        }
    }
}

impl<'a, F> crate::store::scan::PrimitiveVisitor for MultiProjectionVisitor<'a, F>
where
    F: FnMut(MultiProjectionBatch) -> Result<()>,
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

macro_rules! impl_multi_projection_with_rids {
    ($meth:ident, $ArrTy:ty) => {
        fn $meth(&mut self, values: &$ArrTy, row_ids: &UInt64Array) {
            let base = Arc::new(values.clone()) as ArrayRef;
            let rids = Arc::new(row_ids.clone());
            self.emit_chunk(base, rids);
        }
    };
}

impl<'a, F> crate::store::scan::PrimitiveWithRowIdsVisitor for MultiProjectionVisitor<'a, F>
where
    F: FnMut(MultiProjectionBatch) -> Result<()>,
{
    impl_multi_projection_with_rids!(u64_chunk_with_rids, UInt64Array);
    impl_multi_projection_with_rids!(u32_chunk_with_rids, UInt32Array);
    impl_multi_projection_with_rids!(u16_chunk_with_rids, UInt16Array);
    impl_multi_projection_with_rids!(u8_chunk_with_rids, UInt8Array);
    impl_multi_projection_with_rids!(i64_chunk_with_rids, Int64Array);
    impl_multi_projection_with_rids!(i32_chunk_with_rids, Int32Array);
    impl_multi_projection_with_rids!(i16_chunk_with_rids, Int16Array);
    impl_multi_projection_with_rids!(i8_chunk_with_rids, Int8Array);
    impl_multi_projection_with_rids!(f64_chunk_with_rids, Float64Array);
    impl_multi_projection_with_rids!(f32_chunk_with_rids, Float32Array);
}

impl<'a, F> crate::store::scan::PrimitiveSortedVisitor for MultiProjectionVisitor<'a, F>
where
    F: FnMut(MultiProjectionBatch) -> Result<()>,
{
    fn u64_run(&mut self, _a: &UInt64Array, _start: usize, _len: usize) {
        self.sorted_not_supported();
    }
    fn u32_run(&mut self, _a: &UInt32Array, _start: usize, _len: usize) {
        self.sorted_not_supported();
    }
    fn u16_run(&mut self, _a: &UInt16Array, _start: usize, _len: usize) {
        self.sorted_not_supported();
    }
    fn u8_run(&mut self, _a: &UInt8Array, _start: usize, _len: usize) {
        self.sorted_not_supported();
    }
    fn i64_run(&mut self, _a: &Int64Array, _start: usize, _len: usize) {
        self.sorted_not_supported();
    }
    fn i32_run(&mut self, _a: &Int32Array, _start: usize, _len: usize) {
        self.sorted_not_supported();
    }
    fn i16_run(&mut self, _a: &Int16Array, _start: usize, _len: usize) {
        self.sorted_not_supported();
    }
    fn i8_run(&mut self, _a: &Int8Array, _start: usize, _len: usize) {
        self.sorted_not_supported();
    }
    fn f64_run(&mut self, _a: &Float64Array, _start: usize, _len: usize) {
        self.sorted_not_supported();
    }
    fn f32_run(&mut self, _a: &Float32Array, _start: usize, _len: usize) {
        self.sorted_not_supported();
    }
}

impl<'a, F> crate::store::scan::PrimitiveSortedWithRowIdsVisitor for MultiProjectionVisitor<'a, F>
where
    F: FnMut(MultiProjectionBatch) -> Result<()>,
{
    fn u64_run_with_rids(
        &mut self,
        _v: &UInt64Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
        self.sorted_not_supported();
    }
    fn u32_run_with_rids(
        &mut self,
        _v: &UInt32Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
        self.sorted_not_supported();
    }
    fn u16_run_with_rids(
        &mut self,
        _v: &UInt16Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
        self.sorted_not_supported();
    }
    fn u8_run_with_rids(&mut self, _v: &UInt8Array, _r: &UInt64Array, _start: usize, _len: usize) {
        self.sorted_not_supported();
    }
    fn i64_run_with_rids(&mut self, _v: &Int64Array, _r: &UInt64Array, _start: usize, _len: usize) {
        self.sorted_not_supported();
    }
    fn i32_run_with_rids(&mut self, _v: &Int32Array, _r: &UInt64Array, _start: usize, _len: usize) {
        self.sorted_not_supported();
    }
    fn i16_run_with_rids(&mut self, _v: &Int16Array, _r: &UInt64Array, _start: usize, _len: usize) {
        self.sorted_not_supported();
    }
    fn i8_run_with_rids(&mut self, _v: &Int8Array, _r: &UInt64Array, _start: usize, _len: usize) {
        self.sorted_not_supported();
    }
    fn f64_run_with_rids(
        &mut self,
        _v: &Float64Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
        self.sorted_not_supported();
    }
    fn f32_run_with_rids(
        &mut self,
        _v: &Float32Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
        self.sorted_not_supported();
    }

    fn null_run(&mut self, row_ids: &UInt64Array, start: usize, len: usize) {
        if self.include_nulls {
            self.emit_null_run(row_ids, start, len);
        } else {
            self.sorted_not_supported();
        }
    }
}

fn logical_field_name(fid: LogicalFieldId) -> String {
    format!(
        "{:?}-{}-{}",
        fid.namespace(),
        fid.table_id(),
        fid.field_id()
    )
}

fn prefetch_unsorted_column<P>(
    store: &ColumnStore<P>,
    field_id: LogicalFieldId,
) -> Result<PrefetchedColumn>
where
    P: Pager<Blob = EntryHandle>,
{
    let dtype = store.data_type(field_id)?;

    let (descriptor_pk, rowid_descriptor_pk) = {
        let catalog = store.catalog.read().unwrap();
        let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
        let rid_fid = rowid_fid(field_id);
        let rowid_descriptor_pk = *catalog.map.get(&rid_fid).ok_or(Error::Internal(
            "prefetch: missing row-id column for projection".into(),
        ))?;
        (descriptor_pk, rowid_descriptor_pk)
    };

    let desc_blob = store
        .pager
        .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
        .pop()
        .and_then(|r| match r {
            GetResult::Raw { bytes, .. } => Some(bytes),
            _ => None,
        })
        .ok_or_else(|| Error::Internal("prefetch: missing descriptor".into()))?;
    let desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

    let mut metas: Vec<crate::store::descriptor::ChunkMetadata> = Vec::new();
    for m in
        crate::store::descriptor::DescriptorIterator::new(store.pager.as_ref(), desc.head_page_pk)
    {
        let meta = m?;
        if meta.row_count > 0 {
            metas.push(meta);
        }
    }
    if metas.is_empty() {
        let empty_values = new_null_array(&dtype, 0);
        let empty_row_ids = Arc::new(UInt64Array::from(Vec::<u64>::new()));
        return Ok(PrefetchedColumn {
            field_id,
            dtype,
            values: empty_values,
            row_ids: empty_row_ids,
            cursor: 0,
        });
    }

    let rid_desc_blob = store
        .pager
        .batch_get(&[BatchGet::Raw {
            key: rowid_descriptor_pk,
        }])?
        .pop()
        .and_then(|r| match r {
            GetResult::Raw { bytes, .. } => Some(bytes),
            _ => None,
        })
        .ok_or_else(|| Error::Internal("prefetch: missing rowid descriptor".into()))?;
    let rid_desc =
        crate::store::descriptor::ColumnDescriptor::from_le_bytes(rid_desc_blob.as_ref());

    let mut rowid_metas: Vec<crate::store::descriptor::ChunkMetadata> = Vec::new();
    for m in crate::store::descriptor::DescriptorIterator::new(
        store.pager.as_ref(),
        rid_desc.head_page_pk,
    ) {
        let meta = m?;
        if meta.row_count > 0 {
            rowid_metas.push(meta);
        }
    }
    if rowid_metas.len() != metas.len() {
        return Err(Error::Internal(
            "prefetch: row-id chunk count mismatch for projection".into(),
        ));
    }

    let mut gets: Vec<BatchGet> = Vec::with_capacity(metas.len() * 2);
    for m in &metas {
        gets.push(BatchGet::Raw { key: m.chunk_pk });
    }
    for m in &rowid_metas {
        gets.push(BatchGet::Raw { key: m.chunk_pk });
    }

    let results = store.pager.batch_get(&gets)?;
    let data_keys: FxHashSet<PhysicalKey> = metas.iter().map(|m| m.chunk_pk).collect();
    let rowid_keys: FxHashSet<PhysicalKey> = rowid_metas.iter().map(|m| m.chunk_pk).collect();
    let mut data_blobs: FxHashMap<PhysicalKey, Option<EntryHandle>> = FxHashMap::default();
    let mut rowid_blobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
    for r in results {
        match r {
            GetResult::Raw { key, bytes } => {
                if data_keys.contains(&key) {
                    data_blobs.insert(key, Some(bytes));
                } else if rowid_keys.contains(&key) {
                    rowid_blobs.insert(key, bytes);
                }
            }
            GetResult::Missing { key } => {
                if data_keys.contains(&key) {
                    data_blobs.insert(key, None);
                } else if rowid_keys.contains(&key) {
                    return Err(Error::Internal(
                        "prefetch: missing row-id chunk for projection".into(),
                    ));
                }
            }
        }
    }

    let mut chunks: Vec<ArrayRef> = Vec::with_capacity(metas.len());
    let mut row_id_chunks: Vec<Arc<UInt64Array>> = Vec::with_capacity(metas.len());
    for (meta, rid_meta) in metas.into_iter().zip(rowid_metas.into_iter()) {
        let rid_bytes = rowid_blobs
            .remove(&rid_meta.chunk_pk)
            .ok_or_else(|| Error::Internal("prefetch: missing rowid chunk".into()))?;
        let rid_arr_any = llkv_storage::serialization::deserialize_array(rid_bytes)?;
        let rid_arr = rid_arr_any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("prefetch: row-id column type mismatch".into()))?
            .clone();
        let rid_len = rid_arr.len();
        let rid_arr = Arc::new(rid_arr);

        let value_entry = data_blobs
            .remove(&meta.chunk_pk)
            .ok_or_else(|| Error::Internal("prefetch: missing data chunk for projection".into()))?;
        let arr = match value_entry {
            Some(bytes) => llkv_storage::serialization::deserialize_array(bytes)?,
            None => new_null_array(&dtype, rid_len),
        };

        chunks.push(arr);
        row_id_chunks.push(rid_arr);
    }

    let values = if chunks.is_empty() {
        new_null_array(&dtype, 0)
    } else {
        let refs: Vec<&ArrayRef> = chunks.iter().collect();
        crate::store::slicing::concat_many(refs)?
    };

    let row_ids = if row_id_chunks.is_empty() {
        Arc::new(UInt64Array::from(Vec::<u64>::new()))
    } else {
        let row_refs: Vec<ArrayRef> = row_id_chunks
            .iter()
            .map(|arr| Arc::clone(arr) as ArrayRef)
            .collect();
        let ref_refs: Vec<&ArrayRef> = row_refs.iter().collect();
        let any = crate::store::slicing::concat_many(ref_refs)?;
        let arr = any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("prefetch: row-id concat type mismatch".into()))?
            .clone();
        Arc::new(arr)
    };

    Ok(PrefetchedColumn {
        field_id,
        dtype,
        values,
        row_ids,
        cursor: 0,
    })
}

pub fn filter_projection_values<ArrowTy>(
    proj_lfid: LogicalFieldId,
    include_nulls: bool,
    batch: &MultiProjectionBatch,
    predicate: &dyn Fn(ArrowTy::Native) -> bool,
) -> Result<Option<ArrayRef>>
where
    ArrowTy: ArrowPrimitiveType + crate::store::FilterPrimitive,
{
    let base_col = batch.record_batch.column(1).clone();
    let filter_col = base_col
        .as_any()
        .downcast_ref::<PrimitiveArray<ArrowTy>>()
        .ok_or_else(|| Error::Internal("project: filter column type mismatch".into()))?;

    let projection_pos = batch
        .columns
        .iter()
        .position(|fid| *fid == proj_lfid)
        .ok_or_else(|| Error::Internal("project: projection column missing".into()))?;

    let proj_col_idx = 1 + projection_pos;
    let proj_col = if proj_col_idx < batch.record_batch.num_columns() {
        batch.record_batch.column(proj_col_idx).clone()
    } else {
        base_col.clone()
    };

    let mut indices: Vec<u32> = Vec::new();
    for (idx, maybe_value) in filter_col.iter().enumerate() {
        if let Some(value) = maybe_value {
            if predicate(value) {
                indices.push(idx as u32);
            }
        }
    }

    if indices.is_empty() {
        return Ok(None);
    }

    let indices_array = UInt32Array::from(indices);
    let mut filtered = take(proj_col.as_ref(), &indices_array, None).map_err(Error::from)?;

    if !include_nulls {
        if filtered.null_count() > 0 {
            let mask: BooleanArray = is_not_null(filtered.as_ref()).map_err(Error::from)?;
            filtered = arrow_filter(filtered.as_ref(), &mask).map_err(Error::from)?;
        }
    }

    if filtered.len() == 0 {
        return Ok(None);
    }

    Ok(Some(filtered))
}
