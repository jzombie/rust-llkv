use super::*;

#[derive(Default, Clone, Copy)]
pub struct IntRanges {
    pub u64_r: Option<(Bound<u64>, Bound<u64>)>,
    pub u32_r: Option<(Bound<u32>, Bound<u32>)>,
    pub u16_r: Option<(Bound<u16>, Bound<u16>)>,
    pub u8_r: Option<(Bound<u8>, Bound<u8>)>,
    pub i64_r: Option<(Bound<i64>, Bound<i64>)>,
    pub i32_r: Option<(Bound<i32>, Bound<i32>)>,
    pub i16_r: Option<(Bound<i16>, Bound<i16>)>,
    pub i8_r: Option<(Bound<i8>, Bound<i8>)>,
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

#[inline]
pub(crate) fn lower_idx_by<T: Ord, F: Fn(usize) -> T>(
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
pub(crate) fn upper_idx_by<T: Ord, F: Fn(usize) -> T>(
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
