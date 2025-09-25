use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    UInt8Array, UInt16Array, UInt32Array, UInt64Array, new_null_array,
};
use arrow::datatypes::DataType;

use crate::store::scan;
use crate::{Error, Result};

/// A logical batch produced by `project_column`, containing the row ids and the projected column
/// values for a contiguous slice of column storage.
pub struct ProjectionBatch {
    pub row_ids: Arc<UInt64Array>,
    pub values: ArrayRef,
}

pub struct ProjectionVisitor<'a, F>
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
    pub fn new(dtype: DataType, on_batch: &'a mut F) -> Self {
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

    pub fn finish(self) -> Result<()> {
        if let Some(err) = self.error {
            Err(err)
        } else {
            Ok(())
        }
    }
}

impl<'a, F> scan::PrimitiveVisitor for ProjectionVisitor<'a, F>
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

impl<'a, F> scan::PrimitiveSortedVisitor for ProjectionVisitor<'a, F>
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

impl<'a, F> scan::PrimitiveWithRowIdsVisitor for ProjectionVisitor<'a, F>
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

impl<'a, F> scan::PrimitiveSortedWithRowIdsVisitor for ProjectionVisitor<'a, F>
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
