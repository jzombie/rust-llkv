use super::*;

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
    pub fn with_row_ids(mut self, row_id_field: LogicalFieldId) -> Self {
        self.opts.with_row_ids = true;
        self.opts.row_id_field = Some(row_id_field);
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
        }

        let mut adapter = RangeAdapter {
            inner: visitor,
            ir: self.ir,
        };
        self.store.scan(self.field_id, self.opts, &mut adapter)
    }
}
