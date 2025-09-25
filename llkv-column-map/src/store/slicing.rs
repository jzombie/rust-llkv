use arrow::array::{
    Array, ArrayRef, BinaryArray, LargeBinaryArray, LargeStringArray, StringArray, make_array,
};
use arrow::datatypes::DataType;
use llkv_result::{Error, Result};

/// Normalize an array so its offset is zero without copying buffers.
#[inline]
pub(crate) fn zero_offset(arr: &ArrayRef) -> ArrayRef {
    if arr.offset() == 0 {
        return arr.clone();
    }
    let data = arr.to_data();
    let sliced = data.slice(arr.offset(), arr.len());
    make_array(sliced)
}

/// Concatenate many arrays; returns a zero-offset array.
#[inline]
pub(crate) fn concat_many(v: Vec<&ArrayRef>) -> Result<ArrayRef> {
    if v.is_empty() {
        return Err(Error::Internal("concat_many: empty".into()));
    }
    if v.len() == 1 {
        return Ok(zero_offset(v[0]));
    }
    let parts: Vec<&dyn Array> = v.iter().map(|a| a.as_ref()).collect();
    Ok(arrow::compute::concat(&parts)?)
}

/// Bytes-per-value for fixed-width primitives (Boolean approximated as 1).
#[inline]
pub(crate) fn fixed_width_bpv(dt: &DataType) -> Option<usize> {
    match dt {
        DataType::Int8 | DataType::UInt8 => Some(1),
        DataType::Int16 | DataType::UInt16 => Some(2),
        DataType::Int32 | DataType::UInt32 | DataType::Float32 => Some(4),
        DataType::Int64 | DataType::UInt64 | DataType::Float64 => Some(8),
        DataType::Boolean => Some(1),
        _ => None,
    }
}

/// Split variable-width arrays (Utf8/Binary) using their offset buffers so each
/// slice is ~target_bytes of values payload. Offsets are 32-bit here.
#[inline]
pub(crate) fn split_varlen_offsets32(
    offsets: &[i32],
    len: usize,
    arr: &ArrayRef,
    target_bytes: usize,
) -> Vec<ArrayRef> {
    let mut out = Vec::new();
    if len == 0 {
        return out;
    }
    let base = offsets[0] as i64;
    let mut i = 0usize;
    // Ensure at least 1 row per slice; pack until ~target_bytes.
    while i < len {
        let start_off = (offsets[i] as i64 - base) as usize;
        let mut j = i + 1;
        while j <= len {
            let end_off = (offsets[j] as i64 - base) as usize;
            let bytes = end_off - start_off;
            if bytes >= target_bytes && j > i {
                break;
            }
            j += 1;
        }
        let take = (j - i).min(len - i);
        out.push(arr.slice(i, take));
        i += take;
    }
    out
}

/// Same as above but for LargeUtf8/LargeBinary (64-bit offsets).
#[inline]
pub(crate) fn split_varlen_offsets64(
    offsets: &[i64],
    len: usize,
    arr: &ArrayRef,
    target_bytes: usize,
) -> Vec<ArrayRef> {
    let mut out = Vec::new();
    if len == 0 {
        return out;
    }
    let base = offsets[0];
    let mut i = 0usize;
    while i < len {
        let start_off = (offsets[i] - base) as usize;
        let mut j = i + 1;
        while j <= len {
            let end_off = (offsets[j] - base) as usize;
            let bytes = end_off - start_off;
            if bytes >= target_bytes && j > i {
                break;
            }
            j += 1;
        }
        let take = (j - i).min(len - i);
        out.push(arr.slice(i, take));
        i += take;
    }
    out
}

/// Split array into ~target_bytes slices. Fixed-width uses bytes-based rows.
/// For var-width (Utf8/Binary), we use offsets to target value-bytes per slice.
/// For other exotic var-width types, we conservatively fall back to a row cap.
pub(crate) fn split_to_target_bytes(
    arr: &ArrayRef,
    target_bytes: usize,
    varwidth_fallback_rows_per_slice: usize,
) -> Vec<ArrayRef> {
    let n = arr.len();
    if n == 0 {
        return vec![];
    }
    if let Some(bpv) = fixed_width_bpv(arr.data_type()) {
        let rows_per = (target_bytes / bpv).max(1);
        let mut out = Vec::with_capacity(n.div_ceil(rows_per));
        let mut off = 0usize;
        while off < n {
            let take = (n - off).min(rows_per);
            out.push(arr.slice(off, take));
            off += take;
        }
        return out;
    }

    // Handle common var-width types using their offsets, after zeroing slice
    // offset.
    let arr0 = zero_offset(arr);
    match arr0.data_type() {
        DataType::Utf8 => {
            let a = arr0.as_any().downcast_ref::<StringArray>().unwrap();
            split_varlen_offsets32(a.value_offsets(), n, &arr0, target_bytes)
        }
        DataType::Binary => {
            let a = arr0.as_any().downcast_ref::<BinaryArray>().unwrap();
            split_varlen_offsets32(a.value_offsets(), n, &arr0, target_bytes)
        }
        DataType::LargeUtf8 => {
            let a = arr0.as_any().downcast_ref::<LargeStringArray>().unwrap();
            split_varlen_offsets64(a.value_offsets(), n, &arr0, target_bytes)
        }
        DataType::LargeBinary => {
            let a = arr0.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            split_varlen_offsets64(a.value_offsets(), n, &arr0, target_bytes)
        }
        // Fallback for other var-width types (e.g., List/Struct/Map):
        // conservative row cap.
        _ => {
            let rows_per = varwidth_fallback_rows_per_slice; // configurable
            let mut out = Vec::with_capacity(n.div_ceil(rows_per));
            let mut off = 0usize;
            while off < n {
                let take = (n - off).min(rows_per);
                out.push(arr.slice(off, take));
                off += take;
            }
            out
        }
    }
}
