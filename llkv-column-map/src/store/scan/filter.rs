use std::marker::PhantomData;

use arrow::array::{
    Array, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::ArrowPrimitiveType;

use crate::store::descriptor::{ColumnDescriptor, DescriptorIterator};
use crate::store::rowid_fid;
use crate::types::LogicalFieldId;
use llkv_result::{Error, Result};
use simd_r_drive_entry_handle::EntryHandle;

use super::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanBuilder,
};
use crate::store::ColumnStore;
use llkv_storage::pager::{BatchGet, GetResult, Pager};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilterRun {
    pub start_row_id: u64,
    pub len: usize,
}

impl FilterRun {
    #[inline]
    pub fn new(start_row_id: u64) -> Self {
        Self {
            start_row_id,
            len: 1,
        }
    }

    #[inline]
    pub fn end_row_id(&self) -> u64 {
        self.start_row_id + (self.len.saturating_sub(1) as u64)
    }
}

#[derive(Debug)]
pub struct FilterResult {
    runs: Vec<FilterRun>,
    fallback_row_ids: Option<Vec<u64>>,
    total_matches: usize,
}

impl FilterResult {
    pub fn total_matches(&self) -> usize {
        self.total_matches
    }

    pub fn is_dense(&self) -> bool {
        self.fallback_row_ids.is_none()
    }

    pub fn runs(&self) -> &[FilterRun] {
        &self.runs
    }

    pub fn into_runs(self) -> Option<Vec<FilterRun>> {
        if self.is_dense() {
            Some(self.runs)
        } else {
            None
        }
    }

    pub fn into_row_ids(mut self) -> Vec<u64> {
        if let Some(rows) = self.fallback_row_ids.take() {
            return rows;
        }

        let mut out = Vec::with_capacity(self.total_matches);
        for run in &self.runs {
            let mut current = run.start_row_id;
            for _ in 0..run.len {
                out.push(current);
                current += 1;
            }
        }
        out
    }
}

pub(crate) struct FilterVisitor<T: ArrowPrimitiveType, F: FnMut(T::Native) -> bool> {
    predicate: F,
    runs: Vec<FilterRun>,
    fallback_row_ids: Option<Vec<u64>>,
    prev_row_id: Option<u64>,
    total_matches: usize,
    _phantom: PhantomData<T>,
}

impl<T: ArrowPrimitiveType, F: FnMut(T::Native) -> bool> FilterVisitor<T, F> {
    pub(crate) fn new(predicate: F) -> Self {
        Self {
            predicate,
            runs: Vec::new(),
            fallback_row_ids: None,
            prev_row_id: None,
            total_matches: 0,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn into_result(self) -> FilterResult {
        if let Some(rows) = self.fallback_row_ids {
            FilterResult {
                runs: Vec::new(),
                fallback_row_ids: Some(rows),
                total_matches: self.total_matches,
            }
        } else {
            FilterResult {
                runs: self.runs,
                fallback_row_ids: None,
                total_matches: self.total_matches,
            }
        }
    }

    #[inline]
    fn record_match(&mut self, row_id: u64) {
        self.total_matches += 1;

        if let Some(rows) = self.fallback_row_ids.as_mut() {
            rows.push(row_id);
            self.prev_row_id = Some(row_id);
            return;
        }

        match self.prev_row_id {
            None => {
                self.runs.push(FilterRun::new(row_id));
            }
            Some(prev) => {
                if row_id == prev + 1 {
                    if let Some(last) = self.runs.last_mut() {
                        last.len += 1;
                    }
                } else if row_id > prev {
                    self.runs.push(FilterRun::new(row_id));
                } else {
                    self.demote_to_fallback(row_id);
                }
            }
        }

        self.prev_row_id = Some(row_id);
    }

    fn demote_to_fallback(&mut self, row_id: u64) {
        if self.fallback_row_ids.is_none() {
            let mut rows = Vec::with_capacity(self.total_matches);
            for run in &self.runs {
                let mut current = run.start_row_id;
                for _ in 0..run.len {
                    rows.push(current);
                    current += 1;
                }
            }
            self.runs.clear();
            self.fallback_row_ids = Some(rows);
        }

        if let Some(rows) = self.fallback_row_ids.as_mut() {
            rows.push(row_id);
        }
    }
}

impl<T: ArrowPrimitiveType, F: FnMut(T::Native) -> bool> PrimitiveVisitor for FilterVisitor<T, F> {}

impl<T: ArrowPrimitiveType, F: FnMut(T::Native) -> bool> PrimitiveSortedVisitor
    for FilterVisitor<T, F>
{
}

impl<T: ArrowPrimitiveType, F: FnMut(T::Native) -> bool> PrimitiveSortedWithRowIdsVisitor
    for FilterVisitor<T, F>
{
}

macro_rules! impl_filter_visitor {
    ($ty:ty, $arr:ty, $method:ident) => {
        impl<F> PrimitiveWithRowIdsVisitor for FilterVisitor<$ty, F>
        where
            F: FnMut(<$ty as ArrowPrimitiveType>::Native) -> bool,
        {
            fn $method(&mut self, values: &$arr, row_ids: &UInt64Array) {
                let len = values.len();
                debug_assert_eq!(len, row_ids.len());
                debug_assert_eq!(row_ids.null_count(), 0);
                if values.null_count() == 0 {
                    for i in 0..len {
                        // SAFETY: `i < len`, and we already checked that there are no nulls.
                        let value = unsafe { values.value_unchecked(i) };
                        let predicate_passes = {
                            let predicate = &mut self.predicate;
                            predicate(value)
                        };
                        if predicate_passes {
                            // SAFETY: Row ids share the same length and contain no nulls.
                            let row_id = unsafe { row_ids.value_unchecked(i) };
                            self.record_match(row_id);
                        }
                    }
                } else {
                    for i in 0..len {
                        if !values.is_valid(i) {
                            continue;
                        }
                        // SAFETY: guarded by the validity bitmap.
                        let value = unsafe { values.value_unchecked(i) };
                        let predicate_passes = {
                            let predicate = &mut self.predicate;
                            predicate(value)
                        };
                        if predicate_passes {
                            let row_id = unsafe { row_ids.value_unchecked(i) };
                            self.record_match(row_id);
                        }
                    }
                }
            }
        }
    };
}

impl_filter_visitor!(
    arrow::datatypes::UInt64Type,
    UInt64Array,
    u64_chunk_with_rids
);
impl_filter_visitor!(
    arrow::datatypes::UInt32Type,
    UInt32Array,
    u32_chunk_with_rids
);
impl_filter_visitor!(
    arrow::datatypes::UInt16Type,
    UInt16Array,
    u16_chunk_with_rids
);
impl_filter_visitor!(arrow::datatypes::UInt8Type, UInt8Array, u8_chunk_with_rids);
impl_filter_visitor!(arrow::datatypes::Int64Type, Int64Array, i64_chunk_with_rids);
impl_filter_visitor!(arrow::datatypes::Int32Type, Int32Array, i32_chunk_with_rids);
impl_filter_visitor!(arrow::datatypes::Int16Type, Int16Array, i16_chunk_with_rids);
impl_filter_visitor!(arrow::datatypes::Int8Type, Int8Array, i8_chunk_with_rids);
impl_filter_visitor!(
    arrow::datatypes::Float64Type,
    Float64Array,
    f64_chunk_with_rids
);
impl_filter_visitor!(
    arrow::datatypes::Float32Type,
    Float32Array,
    f32_chunk_with_rids
);

pub(crate) struct RowIdFilterVisitor<T: ArrowPrimitiveType, F: FnMut(T::Native) -> bool> {
    predicate: F,
    row_ids: Vec<u64>,
    _phantom: PhantomData<T>,
}

impl<T: ArrowPrimitiveType, F: FnMut(T::Native) -> bool> RowIdFilterVisitor<T, F> {
    pub(crate) fn new(predicate: F) -> Self {
        Self {
            predicate,
            row_ids: Vec::new(),
            _phantom: PhantomData,
        }
    }

    pub(crate) fn into_row_ids(self) -> Vec<u64> {
        self.row_ids
    }
}

impl<T: ArrowPrimitiveType, F: FnMut(T::Native) -> bool> PrimitiveVisitor
    for RowIdFilterVisitor<T, F>
{
}

impl<T: ArrowPrimitiveType, F: FnMut(T::Native) -> bool> PrimitiveSortedVisitor
    for RowIdFilterVisitor<T, F>
{
}

impl<T: ArrowPrimitiveType, F: FnMut(T::Native) -> bool> PrimitiveSortedWithRowIdsVisitor
    for RowIdFilterVisitor<T, F>
{
}

macro_rules! impl_rowid_filter_visitor {
    ($ty:ty, $arr:ty, $method:ident) => {
        impl<F> PrimitiveWithRowIdsVisitor for RowIdFilterVisitor<$ty, F>
        where
            F: FnMut(<$ty as ArrowPrimitiveType>::Native) -> bool,
        {
            fn $method(&mut self, values: &$arr, row_ids: &UInt64Array) {
                let len = values.len();
                debug_assert_eq!(len, row_ids.len());
                debug_assert_eq!(row_ids.null_count(), 0);
                self.row_ids.reserve(len);
                if values.null_count() == 0 {
                    for i in 0..len {
                        let value = unsafe { values.value_unchecked(i) };
                        let predicate_passes = {
                            let predicate = &mut self.predicate;
                            predicate(value)
                        };
                        if predicate_passes {
                            let row_id = unsafe { row_ids.value_unchecked(i) };
                            self.row_ids.push(row_id);
                        }
                    }
                } else {
                    for i in 0..len {
                        if !values.is_valid(i) {
                            continue;
                        }
                        let value = unsafe { values.value_unchecked(i) };
                        let predicate_passes = {
                            let predicate = &mut self.predicate;
                            predicate(value)
                        };
                        if predicate_passes {
                            let row_id = unsafe { row_ids.value_unchecked(i) };
                            self.row_ids.push(row_id);
                        }
                    }
                }
            }
        }
    };
}

impl_rowid_filter_visitor!(
    arrow::datatypes::UInt64Type,
    UInt64Array,
    u64_chunk_with_rids
);
impl_rowid_filter_visitor!(
    arrow::datatypes::UInt32Type,
    UInt32Array,
    u32_chunk_with_rids
);
impl_rowid_filter_visitor!(
    arrow::datatypes::UInt16Type,
    UInt16Array,
    u16_chunk_with_rids
);
impl_rowid_filter_visitor!(arrow::datatypes::UInt8Type, UInt8Array, u8_chunk_with_rids);
impl_rowid_filter_visitor!(arrow::datatypes::Int64Type, Int64Array, i64_chunk_with_rids);
impl_rowid_filter_visitor!(arrow::datatypes::Int32Type, Int32Array, i32_chunk_with_rids);
impl_rowid_filter_visitor!(arrow::datatypes::Int16Type, Int16Array, i16_chunk_with_rids);
impl_rowid_filter_visitor!(arrow::datatypes::Int8Type, Int8Array, i8_chunk_with_rids);
impl_rowid_filter_visitor!(
    arrow::datatypes::Float64Type,
    Float64Array,
    f64_chunk_with_rids
);
impl_rowid_filter_visitor!(
    arrow::datatypes::Float32Type,
    Float32Array,
    f32_chunk_with_rids
);

pub trait FilterPrimitive: ArrowPrimitiveType {
    fn run_filter_with_result<P, F>(
        store: &ColumnStore<P>,
        field_id: LogicalFieldId,
        predicate: F,
    ) -> Result<FilterResult>
    where
        P: Pager<Blob = EntryHandle> + Send + Sync,
        F: FnMut(Self::Native) -> bool;

    fn run_filter<P, F>(
        store: &ColumnStore<P>,
        field_id: LogicalFieldId,
        predicate: F,
    ) -> Result<Vec<u64>>
    where
        P: Pager<Blob = EntryHandle> + Send + Sync,
        F: FnMut(Self::Native) -> bool;
}

macro_rules! impl_filter_primitive {
    ($ty:ty) => {
        impl FilterPrimitive for $ty {
            fn run_filter_with_result<P, F>(
                store: &ColumnStore<P>,
                field_id: LogicalFieldId,
                predicate: F,
            ) -> Result<FilterResult>
            where
                P: Pager<Blob = EntryHandle> + Send + Sync,
                F: FnMut(Self::Native) -> bool,
            {
                run_filter_for_with_result::<P, $ty, F>(store, field_id, predicate)
            }

            fn run_filter<P, F>(
                store: &ColumnStore<P>,
                field_id: LogicalFieldId,
                predicate: F,
            ) -> Result<Vec<u64>>
            where
                P: Pager<Blob = EntryHandle> + Send + Sync,
                F: FnMut(Self::Native) -> bool,
            {
                run_filter_for::<P, $ty, F>(store, field_id, predicate)
            }
        }
    };
}

impl_filter_primitive!(arrow::datatypes::UInt64Type);
impl_filter_primitive!(arrow::datatypes::UInt32Type);
impl_filter_primitive!(arrow::datatypes::UInt16Type);
impl_filter_primitive!(arrow::datatypes::UInt8Type);
impl_filter_primitive!(arrow::datatypes::Int64Type);
impl_filter_primitive!(arrow::datatypes::Int32Type);
impl_filter_primitive!(arrow::datatypes::Int16Type);
impl_filter_primitive!(arrow::datatypes::Int8Type);
impl_filter_primitive!(arrow::datatypes::Float64Type);
impl_filter_primitive!(arrow::datatypes::Float32Type);

pub fn dense_row_runs<P>(
    store: &ColumnStore<P>,
    field_id: LogicalFieldId,
) -> Result<Option<Vec<FilterRun>>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let rid_field = rowid_fid(field_id);
    let catalog = store.catalog.read().unwrap();
    let Some(descriptor_pk) = catalog.map.get(&rid_field).copied() else {
        return Ok(None);
    };

    let desc_blob = store
        .pager
        .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
        .pop()
        .and_then(|res| match res {
            GetResult::Raw { bytes, .. } => Some(bytes),
            _ => None,
        })
        .ok_or(Error::NotFound)?;
    let descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
    drop(catalog);

    if descriptor.head_page_pk == 0 {
        return Ok(Some(Vec::new()));
    }

    let mut runs: Vec<FilterRun> = Vec::new();
    let mut expected_next: Option<u64> = None;

    for meta in DescriptorIterator::new(store.pager.as_ref(), descriptor.head_page_pk) {
        let meta = meta?;
        if meta.row_count == 0 {
            continue;
        }

        let start = meta.min_val_u64;
        let end = meta.max_val_u64;
        if end < start {
            return Ok(None);
        }

        let span = end
            .checked_sub(start)
            .and_then(|delta| delta.checked_add(1))
            .ok_or_else(|| Error::Internal("row_id span overflow in dense_row_runs".into()))?;

        if span != meta.row_count {
            return Ok(None);
        }

        if let Some(expected) = expected_next && start != expected {
            return Ok(None);
        }

        let len = usize::try_from(meta.row_count)
            .map_err(|_| Error::Internal("row_count exceeds usize in dense_row_runs".into()))?;

        runs.push(FilterRun {
            start_row_id: start,
            len,
        });
        expected_next = Some(end + 1);
    }

    Ok(Some(runs))
}

pub(crate) fn run_filter_for_with_result<P, T, F>(
    store: &ColumnStore<P>,
    field_id: LogicalFieldId,
    predicate: F,
) -> Result<FilterResult>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    T: ArrowPrimitiveType,
    F: FnMut(T::Native) -> bool,
    FilterVisitor<T, F>: PrimitiveVisitor
        + PrimitiveSortedVisitor
        + PrimitiveSortedWithRowIdsVisitor
        + PrimitiveWithRowIdsVisitor,
{
    let mut visitor = FilterVisitor::<T, F>::new(predicate);
    ScanBuilder::new(store, field_id)
        .with_row_ids(rowid_fid(field_id))
        .run(&mut visitor)?;
    Ok(visitor.into_result())
}

pub(crate) fn run_filter_for<P, T, F>(
    store: &ColumnStore<P>,
    field_id: LogicalFieldId,
    predicate: F,
) -> Result<Vec<u64>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    T: ArrowPrimitiveType,
    F: FnMut(T::Native) -> bool,
    RowIdFilterVisitor<T, F>: PrimitiveVisitor
        + PrimitiveSortedVisitor
        + PrimitiveSortedWithRowIdsVisitor
        + PrimitiveWithRowIdsVisitor,
{
    let mut visitor = RowIdFilterVisitor::<T, F>::new(predicate);
    ScanBuilder::new(store, field_id)
        .with_row_ids(rowid_fid(field_id))
        .run(&mut visitor)?;
    Ok(visitor.into_row_ids())
}
