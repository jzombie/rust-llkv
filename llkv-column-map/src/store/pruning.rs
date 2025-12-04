//! Pruning logic for column scans.
//!
//! This module contains structures and functions for pruning chunks based on
//! metadata statistics (min/max values) and query predicates.

use arrow::array::ArrayRef;
use rustc_hash::FxHashSet;
use std::ops::Bound;

use crate::codecs::{
    sortable_u64_from_f32, sortable_u64_from_f64, sortable_u64_from_i16, sortable_u64_from_i32,
    sortable_u64_from_i64, sortable_u64_from_i8,
};

/// Statistics for a chunk of data, used for pruning.
#[derive(Debug, Clone, Copy)]
pub struct ChunkStats {
    pub min_u64: u64,
    pub max_u64: u64,
    pub null_count: u64,
    pub distinct_count: u64,
}

/// Type-indexed bounds for range pruning across primitive Arrow scalars.
///
/// Each field captures the caller-provided lower/upper bound pair for a given primitive domain.
#[derive(Default, Clone, Copy, Debug)]
pub struct IntRanges {
    pub u64_r: Option<(Bound<u64>, Bound<u64>)>,
    pub u32_r: Option<(Bound<u32>, Bound<u32>)>,
    pub u16_r: Option<(Bound<u16>, Bound<u16>)>,
    pub u8_r: Option<(Bound<u8>, Bound<u8>)>,
    pub i64_r: Option<(Bound<i64>, Bound<i64>)>,
    pub i32_r: Option<(Bound<i32>, Bound<i32>)>,
    pub i16_r: Option<(Bound<i16>, Bound<i16>)>,
    pub i8_r: Option<(Bound<i8>, Bound<i8>)>,
    pub f64_r: Option<(Bound<f64>, Bound<f64>)>,
    pub f32_r: Option<(Bound<f32>, Bound<f32>)>,
    pub bool_r: Option<(Bound<bool>, Bound<bool>)>,
}

// Internal helper trait mapping a scalar type to the appropriate IntRanges slot.
pub trait RangeKey: Sized {
    fn store(ir: &mut IntRanges, lb: Bound<Self>, ub: Bound<Self>);
}

impl RangeKey for u64 {
    fn store(ir: &mut IntRanges, lb: Bound<u64>, ub: Bound<u64>) {
        ir.u64_r = Some((lb, ub));
    }
}
impl RangeKey for u32 {
    fn store(ir: &mut IntRanges, lb: Bound<u32>, ub: Bound<u32>) {
        ir.u32_r = Some((lb, ub));
    }
}
impl RangeKey for u16 {
    fn store(ir: &mut IntRanges, lb: Bound<u16>, ub: Bound<u16>) {
        ir.u16_r = Some((lb, ub));
    }
}
impl RangeKey for u8 {
    fn store(ir: &mut IntRanges, lb: Bound<u8>, ub: Bound<u8>) {
        ir.u8_r = Some((lb, ub));
    }
}
impl RangeKey for i64 {
    fn store(ir: &mut IntRanges, lb: Bound<i64>, ub: Bound<i64>) {
        ir.i64_r = Some((lb, ub));
    }
}
impl RangeKey for i32 {
    fn store(ir: &mut IntRanges, lb: Bound<i32>, ub: Bound<i32>) {
        ir.i32_r = Some((lb, ub));
    }
}
impl RangeKey for i16 {
    fn store(ir: &mut IntRanges, lb: Bound<i16>, ub: Bound<i16>) {
        ir.i16_r = Some((lb, ub));
    }
}
impl RangeKey for i8 {
    fn store(ir: &mut IntRanges, lb: Bound<i8>, ub: Bound<i8>) {
        ir.i8_r = Some((lb, ub));
    }
}
impl RangeKey for f64 {
    fn store(ir: &mut IntRanges, lb: Bound<f64>, ub: Bound<f64>) {
        ir.f64_r = Some((lb, ub));
    }
}
impl RangeKey for f32 {
    fn store(ir: &mut IntRanges, lb: Bound<f32>, ub: Bound<f32>) {
        ir.f32_r = Some((lb, ub));
    }
}

impl IntRanges {
    /// Check if a chunk with the given min/max values (interpreted as u64)
    /// overlaps with the configured ranges.
    ///
    /// Returns `true` if the chunk *might* contain relevant data (overlap),
    /// or `false` if it definitely does not (disjoint).
    pub fn matches(&self, chunk_min: u64, chunk_max: u64) -> bool {
        // If no ranges are set, we match everything.
        // Note: We check specific type ranges below. If a specific type range is set,
        // we assume the chunk min/max are of that type (cast to u64).

        if let Some((lb, ub)) = self.u64_r
            && !check_overlap(lb, ub, chunk_min, chunk_max)
        {
            return false;
        }
        if let Some((lb, ub)) = self.i64_r
            && !check_overlap(
                map_bound(lb, sortable_u64_from_i64),
                map_bound(ub, sortable_u64_from_i64),
                chunk_min,
                chunk_max,
            )
        {
            return false;
        }
        if let Some((lb, ub)) = self.u32_r
            && !check_overlap(
                map_bound(lb, |v| v as u64),
                map_bound(ub, |v| v as u64),
                chunk_min,
                chunk_max,
            )
        {
            return false;
        }
        if let Some((lb, ub)) = self.i32_r
            && !check_overlap(
                map_bound(lb, sortable_u64_from_i32),
                map_bound(ub, sortable_u64_from_i32),
                chunk_min,
                chunk_max,
            )
        {
            return false;
        }
        if let Some((lb, ub)) = self.u16_r
            && !check_overlap(
                map_bound(lb, |v| v as u64),
                map_bound(ub, |v| v as u64),
                chunk_min,
                chunk_max,
            )
        {
            return false;
        }
        if let Some((lb, ub)) = self.i16_r
            && !check_overlap(
                map_bound(lb, sortable_u64_from_i16),
                map_bound(ub, sortable_u64_from_i16),
                chunk_min,
                chunk_max,
            )
        {
            return false;
        }
        if let Some((lb, ub)) = self.u8_r
            && !check_overlap(
                map_bound(lb, |v| v as u64),
                map_bound(ub, |v| v as u64),
                chunk_min,
                chunk_max,
            )
        {
            return false;
        }
        if let Some((lb, ub)) = self.i8_r
            && !check_overlap(
                map_bound(lb, sortable_u64_from_i8),
                map_bound(ub, sortable_u64_from_i8),
                chunk_min,
                chunk_max,
            )
        {
            return false;
        }
        if let Some((lb, ub)) = self.f64_r
            && !check_overlap(
                map_bound(lb, sortable_u64_from_f64),
                map_bound(ub, sortable_u64_from_f64),
                chunk_min,
                chunk_max,
            )
        {
            return false;
        }
        if let Some((lb, ub)) = self.f32_r
            && !check_overlap(
                map_bound(lb, sortable_u64_from_f32),
                map_bound(ub, sortable_u64_from_f32),
                chunk_min,
                chunk_max,
            )
        {
            return false;
        }

        true
    }
}

fn check_overlap<T: PartialOrd + Copy>(
    lb: Bound<T>,
    ub: Bound<T>,
    chunk_min: T,
    chunk_max: T,
) -> bool {
    // Range is [lb, ub] (inclusive/exclusive based on Bound)
    // Chunk is [chunk_min, chunk_max] (inclusive)

    // Check if Range is strictly less than Chunk
    // Range < Chunk if ub < chunk_min
    match ub {
        Bound::Included(u) => {
            if u < chunk_min {
                return false;
            }
        }
        Bound::Excluded(u) => {
            if u <= chunk_min {
                return false;
            }
        }
        Bound::Unbounded => {}
    }

    // Check if Range is strictly greater than Chunk
    // Range > Chunk if lb > chunk_max
    match lb {
        Bound::Included(l) => {
            if l > chunk_max {
                return false;
            }
        }
        Bound::Excluded(l) => {
            if l >= chunk_max {
                return false;
            }
        }
        Bound::Unbounded => {}
    }

    true
}

fn map_bound<T, U, F>(b: Bound<T>, f: F) -> Bound<U>
where
    F: Fn(T) -> U,
{
    match b {
        Bound::Included(v) => Bound::Included(f(v)),
        Bound::Excluded(v) => Bound::Excluded(f(v)),
        Bound::Unbounded => Bound::Unbounded,
    }
}

fn str_to_u64_prefix(s: &str) -> u64 {
    let bytes = s.as_bytes();
    let mut buf = [0u8; 8];
    let len = bytes.len().min(8);
    buf[..len].copy_from_slice(&bytes[..len]);
    u64::from_be_bytes(buf)
}

pub fn compute_chunk_stats(array: &ArrayRef) -> Option<ChunkStats> {
    use arrow::array::*;
    use arrow::datatypes::*;

    if array.is_empty() {
        return None;
    }

    let null_count = array.null_count() as u64;
    let len = array.len();

    if null_count == len as u64 {
        return Some(ChunkStats {
            min_u64: 0,
            max_u64: 0,
            null_count,
            distinct_count: 0,
        });
    }

    macro_rules! stats_primitive {
        ($ty:ty, $array_ty:ty, $to_u64:expr) => {{
            let arr = array.as_any().downcast_ref::<$array_ty>().unwrap();
            let mut min_val: Option<$ty> = None;
            let mut max_val: Option<$ty> = None;
            let mut distinct = FxHashSet::with_capacity_and_hasher(len, Default::default());

            for i in 0..len {
                if arr.is_null(i) {
                    continue;
                }
                let v = arr.value(i);
                distinct.insert(v);
                match min_val {
                    None => min_val = Some(v),
                    Some(m) => {
                        if v < m {
                            min_val = Some(v)
                        }
                    }
                }
                match max_val {
                    None => max_val = Some(v),
                    Some(m) => {
                        if v > m {
                            max_val = Some(v)
                        }
                    }
                }
            }

            match (min_val, max_val) {
                (Some(min), Some(max)) => Some(ChunkStats {
                    min_u64: $to_u64(min),
                    max_u64: $to_u64(max),
                    null_count,
                    distinct_count: distinct.len() as u64,
                }),
                _ => None,
            }
        }};
    }

    macro_rules! stats_string {
        ($array_ty:ty) => {{
            let arr = array.as_any().downcast_ref::<$array_ty>().unwrap();
            let mut min_val: Option<&str> = None;
            let mut max_val: Option<&str> = None;
            let mut distinct = FxHashSet::with_capacity_and_hasher(len, Default::default());

            for i in 0..len {
                if arr.is_null(i) {
                    continue;
                }
                let v = arr.value(i);
                distinct.insert(v);
                match min_val {
                    None => min_val = Some(v),
                    Some(m) => {
                        if v < m {
                            min_val = Some(v)
                        }
                    }
                }
                match max_val {
                    None => max_val = Some(v),
                    Some(m) => {
                        if v > m {
                            max_val = Some(v)
                        }
                    }
                }
            }

            match (min_val, max_val) {
                (Some(min), Some(max)) => Some(ChunkStats {
                    min_u64: str_to_u64_prefix(min),
                    max_u64: str_to_u64_prefix(max),
                    null_count,
                    distinct_count: distinct.len() as u64,
                }),
                _ => Some(ChunkStats {
                    min_u64: 0,
                    max_u64: 0,
                    null_count,
                    distinct_count: 0,
                }),
            }
        }};
    }

    match array.data_type() {
        DataType::Int8 => stats_primitive!(i8, Int8Array, sortable_u64_from_i8),
        DataType::Int16 => stats_primitive!(i16, Int16Array, sortable_u64_from_i16),
        DataType::Int32 => stats_primitive!(i32, Int32Array, sortable_u64_from_i32),
        DataType::Int64 => stats_primitive!(i64, Int64Array, sortable_u64_from_i64),
        DataType::UInt8 => stats_primitive!(u8, UInt8Array, |v: u8| v as u64),
        DataType::UInt16 => stats_primitive!(u16, UInt16Array, |v: u16| v as u64),
        DataType::UInt32 => stats_primitive!(u32, UInt32Array, |v: u32| v as u64),
        DataType::UInt64 => stats_primitive!(u64, UInt64Array, |v: u64| v),
        DataType::Date32 => stats_primitive!(i32, Date32Array, sortable_u64_from_i32),
        DataType::Date64 => stats_primitive!(i64, Date64Array, sortable_u64_from_i64),

        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            let mut min_val: f32 = f32::INFINITY;
            let mut max_val: f32 = f32::NEG_INFINITY;
            let mut distinct = FxHashSet::with_capacity_and_hasher(len, Default::default());
            let mut has_valid = false;

            for i in 0..len {
                if arr.is_null(i) {
                    continue;
                }
                let v = arr.value(i);
                has_valid = true;
                distinct.insert(v.to_bits());
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
            }

            if !has_valid {
                return Some(ChunkStats {
                    min_u64: 0,
                    max_u64: 0,
                    null_count,
                    distinct_count: 0,
                });
            }

            let min_u64 = sortable_u64_from_f64(min_val as f64);
            let max_u64 = sortable_u64_from_f64(max_val as f64);

            Some(ChunkStats {
                min_u64,
                max_u64,
                null_count,
                distinct_count: distinct.len() as u64,
            })
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let mut min_val: f64 = f64::INFINITY;
            let mut max_val: f64 = f64::NEG_INFINITY;
            let mut distinct = FxHashSet::with_capacity_and_hasher(len, Default::default());
            let mut has_valid = false;

            for i in 0..len {
                if arr.is_null(i) {
                    continue;
                }
                let v = arr.value(i);
                has_valid = true;
                distinct.insert(v.to_bits());
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
            }

            if !has_valid {
                return Some(ChunkStats {
                    min_u64: 0,
                    max_u64: 0,
                    null_count,
                    distinct_count: 0,
                });
            }

            let min_u64 = sortable_u64_from_f64(min_val);
            let max_u64 = sortable_u64_from_f64(max_val);

            Some(ChunkStats {
                min_u64,
                max_u64,
                null_count,
                distinct_count: distinct.len() as u64,
            })
        }

        DataType::Utf8 => stats_string!(StringArray),
        DataType::LargeUtf8 => stats_string!(LargeStringArray),

        _ => None,
    }
}
