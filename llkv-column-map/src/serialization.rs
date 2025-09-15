//! Zero-copy array persistence.
//!
//! We store Arrow buffers directly with a tiny fixed header so we can
//! reconstruct arrays without parsing IPC or cloning buffers.
//!
//! Supported layouts (non-null only for now):
//! - Primitive: UInt64, Int32, UInt32, Float32
//! - FixedSizeList<Float32> with constant list_size
//!
//! Format (little-endian):
//!   [0..4)   : b"ARR0"
//!   [4]      : layout (0=primitive, 1=fsl_float32)
//!   [5]      : type_code (UInt64=1, Int32=2, UInt32=3, Float32=4),
//!              for fsl this is ignored (kept as 0)
//!   [6..8)   : reserved (0)
//!   [8..16)  : len (u64) = number of elements (primitive) or lists (fsl)
//!   [16..20) : extra_a (u32) -> primitive: values_len_bytes
//!                                     fsl: list_size
//!   [20..24) : extra_b (u32) -> primitive: reserved 0
//!                                     fsl: child_values_len_bytes
//!   payload:
//!     primitive: [values_bytes]
//!     fsl:       [child_values_bytes]
//!
//! Note: we deliberately skip nulls to keep the format minimal and fast;
//! your current tests/benches use non-null arrays. We can extend the
//! header with null lengths later without changing the strategy.

use std::convert::TryFrom;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, UInt32Array, UInt64Array,
};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Field};
use bytes::Bytes;

use crate::error::{Error, Result};

const MAGIC: [u8; 4] = *b"ARR0";

#[repr(u8)]
enum Layout {
    Primitive = 0,
    FslFloat32 = 1,
}

#[repr(u8)]
enum PrimType {
    UInt64 = 1,
    Int32 = 2,
    UInt32 = 3,
    Float32 = 4,
}

#[inline]
fn put_u32_le(dst: &mut Vec<u8>, v: u32) {
    dst.extend_from_slice(&v.to_le_bytes());
}
#[inline]
fn put_u64_le(dst: &mut Vec<u8>, v: u64) {
    dst.extend_from_slice(&v.to_le_bytes());
}
#[inline]
fn get_u32_le(src: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes(src[*o..*o + 4].try_into().unwrap());
    *o += 4;
    v
}
#[inline]
fn get_u64_le(src: &[u8], o: &mut usize) -> u64 {
    let v = u64::from_le_bytes(src[*o..*o + 8].try_into().unwrap());
    *o += 8;
    v
}

/// Serialize array buffers with a minimal header.
/// This allocates once for the resulting blob; no per-element work.
pub fn serialize_array(arr: &dyn Array) -> Result<Vec<u8>> {
    match arr.data_type() {
        &DataType::UInt64 => serialize_primitive(arr, PrimType::UInt64),
        &DataType::Int32 => serialize_primitive(arr, PrimType::Int32),
        &DataType::UInt32 => serialize_primitive(arr, PrimType::UInt32),
        &DataType::Float32 => serialize_primitive(arr, PrimType::Float32),

        &DataType::FixedSizeList(ref child, list_size) => {
            // Only support Float32 children now; matches your tests.
            if child.data_type() != &DataType::Float32 {
                return Err(Error::Internal(
                    "Only FixedSizeList<Float32> supported".to_string(),
                ));
            }
            serialize_fsl_float32(arr, i32::try_from(list_size).unwrap())
        }

        other => Err(Error::Internal(format!(
            "Unsupported type in serialize_array: {other}"
        ))),
    }
}

fn serialize_primitive(arr: &dyn Array, code: PrimType) -> Result<Vec<u8>> {
    // Non-null only for now.
    if arr.null_count() != 0 {
        return Err(Error::Internal(
            "nulls not supported in zero-copy format (yet)".to_string(),
        ));
    }

    let data = arr.to_data();
    let len = data.len() as u64;
    let values = data
        .buffers()
        .get(0)
        .ok_or_else(|| Error::Internal("missing values buffer".into()))?;
    let values_bytes = values.as_slice();
    let values_len = u32::try_from(values_bytes.len())
        .map_err(|_| Error::Internal("values too large".into()))?;

    let mut out = Vec::with_capacity(24 + values_bytes.len());
    out.extend_from_slice(&MAGIC);
    out.push(Layout::Primitive as u8);
    out.push(code as u8);
    out.extend_from_slice(&[0u8; 2]); // reserved
    put_u64_le(&mut out, len);
    put_u32_le(&mut out, values_len);
    put_u32_le(&mut out, 0); // extra_b (reserved)
    out.extend_from_slice(values_bytes);
    Ok(out)
}

fn serialize_fsl_float32(arr: &dyn Array, list_size: i32) -> Result<Vec<u8>> {
    // Non-null only for now (both parent and child).
    if arr.null_count() != 0 {
        return Err(Error::Internal(
            "nulls not supported in zero-copy format (yet)".to_string(),
        ));
    }
    let fsl = arr
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| Error::Internal("FSL downcast failed".into()))?;

    let values = fsl.values();
    if values.null_count() != 0 || values.data_type() != &DataType::Float32 {
        return Err(Error::Internal(
            "FSL child must be non-null Float32".to_string(),
        ));
    }
    let child = values.to_data();
    let child_buf = child
        .buffers()
        .get(0)
        .ok_or_else(|| Error::Internal("missing child values".into()))?;
    let child_bytes = child_buf.as_slice();

    let child_len =
        u32::try_from(child_bytes.len()).map_err(|_| Error::Internal("child too large".into()))?;
    let mut out = Vec::with_capacity(24 + child_bytes.len());
    out.extend_from_slice(&MAGIC);
    out.push(Layout::FslFloat32 as u8);
    out.push(0); // type_code unused for FSL
    out.extend_from_slice(&[0u8; 2]); // reserved
    put_u64_le(&mut out, fsl.len() as u64);
    put_u32_le(&mut out, u32::try_from(list_size).unwrap()); // extra_a = list_size
    put_u32_le(&mut out, child_len); // extra_b = child_values_len
    out.extend_from_slice(child_bytes);
    Ok(out)
}

/// Deserialize zero-copy from pager blob bytes (Bytes).
/// Arrow `Buffer`s borrow the pager memory with no cloning or IPC parsing.
pub fn deserialize_array(blob: Bytes) -> Result<ArrayRef> {
    if blob.len() < 24 || &blob[0..4] != &MAGIC {
        return Err(Error::Internal("bad array blob magic/size".into()));
    }
    let layout = blob[4];
    let type_code = blob[5];
    // [6..8) reserved
    let mut o = 8usize;
    let len = get_u64_le(&blob, &mut o) as usize;
    let extra_a = get_u32_le(&blob, &mut o);
    let extra_b = get_u32_le(&blob, &mut o);

    // Create a Buffer that references the whole blob, then slice to payload.
    // (Deprecated in Arrow 56, but still zero-copy. Can be swapped to
    //  Buffer::from(blob.into()) once you bump and want to quiet the warn.)
    let whole = Buffer::from_bytes(blob.into()); // zero-copy
    let payload = whole.slice(o); // Arrow 56 slice(offset)

    match layout {
        x if x == Layout::Primitive as u8 => {
            let values_len = extra_a as usize;
            if payload.len() != values_len {
                return Err(Error::Internal("primitive payload length mismatch".into()));
            }
            let values = payload; // already exact length
            let arr: ArrayRef = match type_code {
                x if x == PrimType::UInt64 as u8 => Arc::new(UInt64Array::new(values.into(), None)),
                x if x == PrimType::Int32 as u8 => Arc::new(Int32Array::new(values.into(), None)),
                x if x == PrimType::UInt32 as u8 => Arc::new(UInt32Array::new(values.into(), None)),
                x if x == PrimType::Float32 as u8 => {
                    Arc::new(Float32Array::new(values.into(), None))
                }
                _ => return Err(Error::Internal("unsupported primitive code".into())),
            };
            if arr.len() != len {
                return Err(Error::Internal("primitive length mismatch".into()));
            }
            Ok(arr)
        }

        x if x == Layout::FslFloat32 as u8 => {
            let list_size = extra_a as i32;
            let child_values_len = extra_b as usize;
            if payload.len() != child_values_len {
                return Err(Error::Internal("fsl child length mismatch".into()));
            }
            let child_values = payload; // already exact length
            let child = Arc::new(Float32Array::new(child_values.into(), None)) as ArrayRef;
            let field = Arc::new(Field::new("item", DataType::Float32, false));
            let arr = FixedSizeListArray::new(field, list_size, child, None);
            if arr.len() != len {
                return Err(Error::Internal("fsl length mismatch".into()));
            }
            Ok(Arc::new(arr))
        }

        _ => Err(Error::Internal("unknown layout".into())),
    }
}
