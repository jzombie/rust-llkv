use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, GenericStringArray, Int8Array,
    Int16Array, Int32Array, Int64Array, OffsetSizeTrait, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::ArrowPrimitiveType;
use arrow::error::Result as ArrowResult;

use crate::parallel;
use crate::store::descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorIterator};
use crate::store::rowid_fid;
use crate::types::LogicalFieldId;
use llkv_expr::typed_predicate::{Predicate, PredicateValue};
use llkv_result::{Error, Result};
use simd_r_drive_entry_handle::EntryHandle;

use super::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanBuilder,
};
use crate::store::ColumnStore;
use llkv_storage::pager::{BatchGet, GetResult, Pager};
use llkv_storage::serialization::deserialize_array;
use rayon::prelude::*;

// Packed bitset used by fused string predicate evaluation. Stores bits in u64 words
// with LSB corresponding to lower indices. Provides small helpers for common ops.
struct BitMask {
    bits: Vec<u64>,
    len: usize,
}

impl BitMask {
    fn new() -> Self {
        Self {
            bits: Vec::new(),
            len: 0,
        }
    }

    fn resize(&mut self, new_len: usize) {
        self.len = new_len;
        let words = new_len.div_ceil(64);
        if self.bits.len() < words {
            self.bits.resize(words, 0);
        } else if self.bits.len() > words {
            self.bits.truncate(words);
        }
        self.clear_trailing_bits();
    }

    fn clear_trailing_bits(&mut self) {
        if self.len == 0 || self.bits.is_empty() {
            return;
        }
        let words = self.bits.len();
        let valid_bits = self.len % 64;
        if valid_bits == 0 {
            return;
        }
        let mask = (1u64 << valid_bits) - 1u64;
        let last = words - 1;
        self.bits[last] &= mask;
    }

    fn fill_ones(&mut self) {
        for w in &mut self.bits {
            *w = u64::MAX;
        }
        self.clear_trailing_bits();
    }

    fn and_with_iter<I>(&mut self, iter: I) -> usize
    where
        I: IntoIterator<Item = u64>,
    {
        let mut removed = 0usize;
        let mut chunks = iter.into_iter();
        for word in &mut self.bits {
            let before = *word;
            let chunk = chunks.next().unwrap_or(0);
            *word &= chunk;
            removed += (before.count_ones() - word.count_ones()) as usize;
        }
        self.clear_trailing_bits();
        removed
    }

    fn for_each_set_bit<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        for (word_idx, word) in self.bits.iter().enumerate() {
            let mut current = *word;
            while current != 0 {
                let bit = current.trailing_zeros() as usize;
                let idx = word_idx * 64 + bit;
                if idx >= self.len {
                    break;
                }
                f(idx);
                current &= current - 1;
            }
        }
    }
}

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

pub trait StringContainsKernel: OffsetSizeTrait {
    fn contains(array: &GenericStringArray<Self>, needle: &str) -> ArrowResult<BooleanArray>;
}

#[allow(deprecated)]
impl StringContainsKernel for i32 {
    fn contains(array: &GenericStringArray<Self>, needle: &str) -> ArrowResult<BooleanArray> {
        #[allow(deprecated)]
        {
            arrow::compute::kernels::comparison::contains_utf8_scalar(array, needle)
        }
    }
}

#[allow(deprecated)]
impl StringContainsKernel for i64 {
    fn contains(array: &GenericStringArray<Self>, needle: &str) -> ArrowResult<BooleanArray> {
        #[allow(deprecated)]
        {
            arrow::compute::kernels::comparison::contains_utf8_scalar(array, needle)
        }
    }
}

pub trait FilterDispatch {
    type Value: PredicateValue + Clone;

    fn run_filter<P>(
        store: &ColumnStore<P>,
        field_id: LogicalFieldId,
        predicate: &Predicate<Self::Value>,
    ) -> Result<Vec<u64>>
    where
        P: Pager<Blob = EntryHandle> + Send + Sync;

    /// Run multiple predicates fused into a single pass over the data.
    fn run_fused<P>(
        store: &ColumnStore<P>,
        field_id: LogicalFieldId,
        predicates: &[Predicate<Self::Value>],
    ) -> Result<Vec<u64>>
    where
        P: Pager<Blob = EntryHandle> + Send + Sync,
    {
        // Conservative default: run each predicate separately and intersect the resulting
        // row-id lists. Implementors (e.g., Utf8Filter) should override to provide a
        // single-pass, vectorized evaluation for better performance.
        if predicates.is_empty() {
            return Ok(Vec::new());
        }

        // Collect per-predicate result vectors
        let mut sets: Vec<Vec<u64>> = Vec::with_capacity(predicates.len());
        for p in predicates {
            let ids = Self::run_filter(store, field_id, p)?;
            sets.push(ids);
            // early exit: if any predicate matched nothing, intersection is empty
            if sets.last().map(|v| v.is_empty()).unwrap_or(false) {
                return Ok(Vec::new());
            }
        }

        // Intersect all vectors (they are returned in ascending order by run_filter).
        // Start with the smallest vector to minimize work.
        sets.sort_by_key(|v| v.len());
        let mut result = sets[0].clone();
        for other in sets.iter().skip(1) {
            let mut i = 0usize;
            let mut j = 0usize;
            let mut out: Vec<u64> = Vec::with_capacity(result.len().min(other.len()));
            while i < result.len() && j < other.len() {
                match result[i].cmp(&other[j]) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    std::cmp::Ordering::Equal => {
                        out.push(result[i]);
                        i += 1;
                        j += 1;
                    }
                }
            }
            result = out;
            if result.is_empty() {
                break;
            }
        }
        Ok(result)
    }
}

impl<T> FilterDispatch for T
where
    T: FilterPrimitive,
    T::Native: PredicateValue + Clone,
{
    type Value = T::Native;

    fn run_filter<P>(
        store: &ColumnStore<P>,
        field_id: LogicalFieldId,
        predicate: &Predicate<Self::Value>,
    ) -> Result<Vec<u64>>
    where
        P: Pager<Blob = EntryHandle> + Send + Sync,
    {
        let predicate = predicate.clone();
        T::run_filter(store, field_id, move |value| {
            predicate.matches(<T::Native as PredicateValue>::borrowed(&value))
        })
    }
}

pub struct Utf8Filter<O>(PhantomData<O>)
where
    O: OffsetSizeTrait + StringContainsKernel;

impl<O> FilterDispatch for Utf8Filter<O>
where
    O: OffsetSizeTrait + StringContainsKernel,
{
    type Value = String;

    fn run_filter<P>(
        store: &ColumnStore<P>,
        field_id: LogicalFieldId,
        predicate: &Predicate<Self::Value>,
    ) -> Result<Vec<u64>>
    where
        P: Pager<Blob = EntryHandle> + Send + Sync,
    {
        match predicate {
            Predicate::Contains(fragment) => {
                run_filter_for_string_contains::<P, O>(store, field_id, fragment.as_str())
            }
            _ => {
                let predicate = predicate.clone();
                run_filter_for_string::<P, O, _>(store, field_id, move |value| {
                    predicate.matches(value)
                })
            }
        }
    }

    fn run_fused<P>(
        store: &ColumnStore<P>,
        field_id: LogicalFieldId,
        predicates: &[Predicate<Self::Value>],
    ) -> Result<Vec<u64>>
    where
        P: Pager<Blob = EntryHandle> + Send + Sync,
    {
        if predicates.is_empty() {
            return Ok(Vec::new());
        }

        // Separate vectorizable 'contains' predicates from others.
        let mut contains: Vec<String> = Vec::new();
        let mut others: Vec<Predicate<String>> = Vec::new();
        for p in predicates {
            match p {
                Predicate::Contains(s) => contains.push(s.clone()),
                _ => others.push(p.clone()),
            }
        }

        // If there are no contains predicates, fall back to per-row evaluation of all predicates.
        if contains.is_empty() {
            let preds = predicates.to_vec();
            let combined = move |value: &str| {
                for p in &preds {
                    if !p.matches(value) {
                        return false;
                    }
                }
                true
            };
            return run_filter_for_string::<P, O, _>(store, field_id, combined);
        }

        let contains = Arc::new(contains);
        let others = Arc::new(others);

        let (value_metas, row_metas) = string_chunk_metadata(store, field_id)?;

        let chunk_results: Vec<Result<Vec<u64>>> = parallel::with_thread_pool(|| {
            value_metas
                .par_iter()
                .zip(row_metas.par_iter())
                .map(|(value_meta, row_meta)| {
                    let contains = Arc::clone(&contains);
                    let others = Arc::clone(&others);

                    let (value_any, row_any) =
                        load_string_chunk_arrays(store, value_meta, row_meta)?;

                    let value_arr = value_any
                        .as_any()
                        .downcast_ref::<GenericStringArray<O>>()
                        .ok_or_else(|| {
                            Error::Internal("string filter: value chunk dtype mismatch".into())
                        })?;
                    let row_arr =
                        row_any
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .ok_or_else(|| {
                                Error::Internal("string filter: row_id chunk downcast".into())
                            })?;

                    if value_arr.len() != row_arr.len() {
                        return Err(Error::Internal(
                            "string filter: value/row chunk length mismatch".into(),
                        ));
                    }

                    let len: usize = value_arr.len();
                    if len == 0 {
                        return Ok(Vec::new());
                    }

                    let mut bitmask = BitMask::new();
                    bitmask.resize(len);
                    bitmask.fill_ones();

                    let mut candidates: usize = len;

                    if let Some(row_nulls) = row_arr.nulls() {
                        let removed =
                            bitmask.and_with_iter(row_nulls.inner().bit_chunks().iter_padded());
                        candidates = candidates.saturating_sub(removed);
                        if candidates == 0 {
                            return Ok(Vec::new());
                        }
                    }

                    if let Some(value_nulls) = value_arr.nulls() {
                        let removed =
                            bitmask.and_with_iter(value_nulls.inner().bit_chunks().iter_padded());
                        candidates = candidates.saturating_sub(removed);
                        if candidates == 0 {
                            return Ok(Vec::new());
                        }
                    }

                    for frag in contains.as_ref() {
                        if frag.is_empty() {
                            continue;
                        }

                        let arr_mask =
                            <O as StringContainsKernel>::contains(value_arr, frag.as_str())
                                .map_err(Error::from)?;

                        let removed =
                            bitmask.and_with_iter(arr_mask.values().bit_chunks().iter_padded());
                        candidates = candidates.saturating_sub(removed);
                        if candidates == 0 {
                            return Ok(Vec::new());
                        }

                        if let Some(mask_nulls) = arr_mask.nulls() {
                            let removed = bitmask
                                .and_with_iter(mask_nulls.inner().bit_chunks().iter_padded());
                            candidates = candidates.saturating_sub(removed);
                            if candidates == 0 {
                                return Ok(Vec::new());
                            }
                        }
                    }

                    if candidates == 0 {
                        return Ok(Vec::new());
                    }

                    let mut local_matches = Vec::with_capacity(candidates);
                    if others.as_ref().is_empty() {
                        bitmask.for_each_set_bit(|idx| {
                            if idx < len {
                                local_matches.push(row_arr.value(idx));
                            }
                        });
                    } else {
                        let other_preds = others.as_ref();
                        bitmask.for_each_set_bit(|idx| {
                            if idx >= len {
                                return;
                            }
                            let text = value_arr.value(idx);
                            if other_preds.iter().all(|p| p.matches(text)) {
                                local_matches.push(row_arr.value(idx));
                            }
                        });
                    }

                    Ok(local_matches)
                })
                .collect()
        });

        let mut matches = Vec::new();
        for chunk in chunk_results {
            let mut chunk_matches = chunk?;
            matches.append(&mut chunk_matches);
        }

        Ok(matches)
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

        if let Some(expected) = expected_next
            && start != expected
        {
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

fn string_chunk_metadata<P>(
    store: &ColumnStore<P>,
    field_id: LogicalFieldId,
) -> Result<(Vec<ChunkMetadata>, Vec<ChunkMetadata>)>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let row_field = rowid_fid(field_id);
    let catalog = store.catalog.read().unwrap();
    let value_desc_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
    let row_desc_pk = *catalog.map.get(&row_field).ok_or(Error::NotFound)?;
    drop(catalog);

    let descriptors = store.pager.batch_get(&[
        BatchGet::Raw { key: value_desc_pk },
        BatchGet::Raw { key: row_desc_pk },
    ])?;

    let mut value_desc_blob: Option<EntryHandle> = None;
    let mut row_desc_blob: Option<EntryHandle> = None;
    for res in descriptors {
        if let GetResult::Raw { key, bytes } = res {
            if key == value_desc_pk {
                value_desc_blob = Some(bytes);
            } else if key == row_desc_pk {
                row_desc_blob = Some(bytes);
            }
        }
    }

    let value_desc_blob = value_desc_blob.ok_or(Error::NotFound)?;
    let row_desc_blob = row_desc_blob.ok_or(Error::NotFound)?;
    let value_desc = ColumnDescriptor::from_le_bytes(value_desc_blob.as_ref());
    let row_desc = ColumnDescriptor::from_le_bytes(row_desc_blob.as_ref());

    let mut value_metas: Vec<ChunkMetadata> = Vec::new();
    for meta in DescriptorIterator::new(store.pager.as_ref(), value_desc.head_page_pk) {
        let meta = meta?;
        if meta.row_count == 0 {
            continue;
        }
        value_metas.push(meta);
    }

    let mut row_metas: Vec<ChunkMetadata> = Vec::new();
    for meta in DescriptorIterator::new(store.pager.as_ref(), row_desc.head_page_pk) {
        let meta = meta?;
        if meta.row_count == 0 {
            continue;
        }
        row_metas.push(meta);
    }

    if value_metas.len() != row_metas.len() {
        return Err(Error::Internal(
            "string filter: chunk count mismatch between value and row columns".into(),
        ));
    }

    Ok((value_metas, row_metas))
}

fn load_string_chunk_arrays<P>(
    store: &ColumnStore<P>,
    value_meta: &ChunkMetadata,
    row_meta: &ChunkMetadata,
) -> Result<(ArrayRef, ArrayRef)>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let requests = [
        BatchGet::Raw {
            key: value_meta.chunk_pk,
        },
        BatchGet::Raw {
            key: row_meta.chunk_pk,
        },
    ];
    let results = store.pager.batch_get(&requests)?;
    let mut value_blob: Option<EntryHandle> = None;
    let mut row_blob: Option<EntryHandle> = None;
    for res in results {
        if let GetResult::Raw { key, bytes } = res {
            if key == value_meta.chunk_pk {
                value_blob = Some(bytes);
            } else if key == row_meta.chunk_pk {
                row_blob = Some(bytes);
            }
        }
    }

    let value_blob = value_blob.ok_or(Error::NotFound)?;
    let row_blob = row_blob.ok_or(Error::NotFound)?;

    let value_any = deserialize_array(value_blob)?;
    let row_any = deserialize_array(row_blob)?;
    Ok((value_any, row_any))
}

fn visit_string_chunks<P, O, F>(
    store: &ColumnStore<P>,
    field_id: LogicalFieldId,
    mut visit: F,
) -> Result<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    O: OffsetSizeTrait,
    F: FnMut(&GenericStringArray<O>, &UInt64Array) -> Result<()>,
{
    let (value_metas, row_metas) = string_chunk_metadata(store, field_id)?;

    for (value_meta, row_meta) in value_metas.iter().zip(row_metas.iter()) {
        let (value_any, row_any) = load_string_chunk_arrays(store, value_meta, row_meta)?;

        let value_arr = value_any
            .as_any()
            .downcast_ref::<GenericStringArray<O>>()
            .ok_or_else(|| Error::Internal("string filter: value chunk dtype mismatch".into()))?;

        let row_arr = row_any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("string filter: row_id chunk downcast".into()))?;

        if value_arr.len() != row_arr.len() {
            return Err(Error::Internal(
                "string filter: value/row chunk length mismatch".into(),
            ));
        }

        visit(value_arr, row_arr)?;
    }

    Ok(())
}

pub(crate) fn run_filter_for_string<P, O, F>(
    store: &ColumnStore<P>,
    field_id: LogicalFieldId,
    predicate: F,
) -> Result<Vec<u64>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    O: OffsetSizeTrait,
    F: FnMut(&str) -> bool,
{
    let mut matches = Vec::new();
    let mut pred = predicate;
    visit_string_chunks::<P, O, _>(store, field_id, |value_arr, row_arr| {
        for i in 0..value_arr.len() {
            if !row_arr.is_valid(i) || value_arr.is_null(i) {
                continue;
            }
            let row_id = row_arr.value(i);
            let text = value_arr.value(i);
            if pred(text) {
                matches.push(row_id);
            }
        }
        Ok(())
    })?;

    Ok(matches)
}

pub(crate) fn run_filter_for_string_contains<P, O>(
    store: &ColumnStore<P>,
    field_id: LogicalFieldId,
    needle: &str,
) -> Result<Vec<u64>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    O: OffsetSizeTrait + StringContainsKernel,
{
    let mut matches = Vec::new();
    let needle_is_empty = needle.is_empty();

    visit_string_chunks::<P, O, _>(store, field_id, |value_arr, row_arr| {
        if needle_is_empty {
            for i in 0..value_arr.len() {
                if !row_arr.is_valid(i) || value_arr.is_null(i) {
                    continue;
                }
                matches.push(row_arr.value(i));
            }
            return Ok(());
        }

        let mask = <O as StringContainsKernel>::contains(value_arr, needle).map_err(Error::from)?;

        for i in 0..mask.len() {
            if !row_arr.is_valid(i) {
                continue;
            }
            if !mask.is_valid(i) || !mask.value(i) {
                continue;
            }
            matches.push(row_arr.value(i));
        }

        Ok(())
    })?;

    Ok(matches)
}
