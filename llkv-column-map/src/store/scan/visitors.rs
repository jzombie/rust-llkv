use super::*;

/// Unsorted primitive visitor: one callback per chunk per type.
pub trait PrimitiveVisitor {
    fn u64_chunk(&mut self, _a: &UInt64Array) {
        unimplemented!("`u64_chunk` not implemented")
    }
    fn u32_chunk(&mut self, _a: &UInt32Array) {
        unimplemented!("`u32_chunk` not implemented")
    }
    fn u16_chunk(&mut self, _a: &UInt16Array) {
        unimplemented!("`u16_chunk` not implemented")
    }
    fn u8_chunk(&mut self, _a: &UInt8Array) {
        unimplemented!("`u8_chunk` not implemented")
    }
    fn i64_chunk(&mut self, _a: &Int64Array) {
        unimplemented!("`i64_chunk` not implemented")
    }
    fn i32_chunk(&mut self, _a: &Int32Array) {
        unimplemented!("`i32_chunk` not implemented")
    }
    fn i16_chunk(&mut self, _a: &Int16Array) {
        unimplemented!("`i16_chunk` not implemented")
    }
    fn i8_chunk(&mut self, _a: &Int8Array) {
        unimplemented!("`i8_chunk` not implemented")
    }
}

/// Unsorted primitive visitor with row ids (u64).
pub trait PrimitiveWithRowIdsVisitor {
    fn u64_chunk_with_rids(&mut self, _v: &UInt64Array, _r: &UInt64Array) {
        unimplemented!("`u64_chunk_with_rids` not implemented")
    }
    fn u32_chunk_with_rids(&mut self, _v: &UInt32Array, _r: &UInt64Array) {
        unimplemented!("`u32_chunk_with_rids` not implemented")
    }
    fn u16_chunk_with_rids(&mut self, _v: &UInt16Array, _r: &UInt64Array) {
        unimplemented!("`u16_chunk_with_rids` not implemented")
    }
    fn u8_chunk_with_rids(&mut self, _v: &UInt8Array, _r: &UInt64Array) {
        unimplemented!("`u8_chunk_with_rids` not implemented")
    }
    fn i64_chunk_with_rids(&mut self, _v: &Int64Array, _r: &UInt64Array) {
        unimplemented!("`i64_chunk_with_rids` not implemented")
    }
    fn i32_chunk_with_rids(&mut self, _v: &Int32Array, _r: &UInt64Array) {
        unimplemented!("`i32_chunk_with_rids` not implemented")
    }
    fn i16_chunk_with_rids(&mut self, _v: &Int16Array, _r: &UInt64Array) {
        unimplemented!("`i16_chunk_with_rids` not implemented")
    }
    fn i8_chunk_with_rids(&mut self, _v: &Int8Array, _r: &UInt64Array) {
        unimplemented!("`i8_chunk_with_rids` not implemented")
    }
}

/// Sorted visitor fed with coalesced runs (start,len) within a typed array.
pub trait PrimitiveSortedVisitor {
    fn u64_run(&mut self, _a: &UInt64Array, _start: usize, _len: usize) {
        unimplemented!("`u64_run` not implemented")
    }
    fn u32_run(&mut self, _a: &UInt32Array, _start: usize, _len: usize) {
        unimplemented!("`u32_run` not implemented")
    }
    fn u16_run(&mut self, _a: &UInt16Array, _start: usize, _len: usize) {
        unimplemented!("`u16_run` not implemented")
    }
    fn u8_run(&mut self, _a: &UInt8Array, _start: usize, _len: usize) {
        unimplemented!("`u8_run` not implemented")
    }
    fn i64_run(&mut self, _a: &Int64Array, _start: usize, _len: usize) {
        unimplemented!("`i64_run` not implemented")
    }
    fn i32_run(&mut self, _a: &Int32Array, _start: usize, _len: usize) {
        unimplemented!("`i32_run` not implemented")
    }
    fn i16_run(&mut self, _a: &Int16Array, _start: usize, _len: usize) {
        unimplemented!("`i16_run` not implemented")
    }
    fn i8_run(&mut self, _a: &Int8Array, _start: usize, _len: usize) {
        unimplemented!("`i8_run` not implemented")
    }
}

/// Sorted visitor with row ids.
pub trait PrimitiveSortedWithRowIdsVisitor {
    fn u64_run_with_rids(
        &mut self,
        _v: &UInt64Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
    }
    fn u32_run_with_rids(
        &mut self,
        _v: &UInt32Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
    }
    fn u16_run_with_rids(
        &mut self,
        _v: &UInt16Array,
        _r: &UInt64Array,
        _start: usize,
        _len: usize,
    ) {
    }
    fn u8_run_with_rids(&mut self, _v: &UInt8Array, _r: &UInt64Array, _start: usize, _len: usize) {}
    fn i64_run_with_rids(&mut self, _v: &Int64Array, _r: &UInt64Array, _start: usize, _len: usize) {
    }
    fn i32_run_with_rids(&mut self, _v: &Int32Array, _r: &UInt64Array, _start: usize, _len: usize) {
    }
    fn i16_run_with_rids(&mut self, _v: &Int16Array, _r: &UInt64Array, _start: usize, _len: usize) {
    }
    fn i8_run_with_rids(&mut self, _v: &Int8Array, _r: &UInt64Array, _start: usize, _len: usize) {}
}
