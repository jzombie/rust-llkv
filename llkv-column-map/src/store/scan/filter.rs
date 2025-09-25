use std::marker::PhantomData;

use arrow::array::{
    Array, Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::ArrowPrimitiveType;

use crate::error::Result;
use crate::store::rowid_fid;
use crate::types::LogicalFieldId;
use simd_r_drive_entry_handle::EntryHandle;

use super::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanBuilder,
};
use crate::storage::pager::Pager;
use crate::store::ColumnStore;

pub(crate) struct FilterVisitor<T: ArrowPrimitiveType, F: FnMut(T::Native) -> bool> {
    predicate: F,
    row_ids: Vec<u64>,
    _phantom: PhantomData<T>,
}

impl<T: ArrowPrimitiveType, F: FnMut(T::Native) -> bool> FilterVisitor<T, F> {
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
                let predicate = &mut self.predicate;
                for i in 0..len {
                    if values.is_null(i) {
                        continue;
                    }
                    let value = values.value(i);
                    if predicate(value) {
                        self.row_ids.push(row_ids.value(i));
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

pub trait FilterPrimitive: ArrowPrimitiveType {
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

pub(crate) fn run_filter_for<P, T, F>(
    store: &ColumnStore<P>,
    field_id: LogicalFieldId,
    predicate: F,
) -> Result<Vec<u64>>
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
    Ok(visitor.into_row_ids())
}
