//! Zero-copy array persistence for fixed/var width Arrow arrays used by the store.
//!
//! Format (little-endian):
//!   [0..4)   : b"ARR0"
//!   [4]      : layout (0=primitive, 1=fsl_float32, 2=varlen)
//!   [5]      : type_code (UInt64=1, Int32=2, ... Binary=5, etc.)
//!   [6..8)   : reserved (0)
//!   [8..16)  : len (u64) = number of elements
//!   [16..20) : extra_a (u32) -> primitive: values_len_bytes; fsl: list_size; varlen: offsets_len_bytes
//!   [20..24) : extra_b (u32) -> primitive: reserved 0;      fsl: child_values_len_bytes; varlen: values_len_bytes
//!   payload  : primitive: [values]
//!              fsl      : [child_values]
//!              varlen   : [offsets][values]

use std::convert::TryFrom;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayData, ArrayRef, FixedSizeListArray, Float32Array, make_array,
};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Field};
use simd_r_drive_entry_handle::EntryHandle;

use crate::error::{Error, Result};

const MAGIC: [u8; 4] = *b"ARR0";

#[repr(u8)]
enum Layout { Primitive = 0, FslFloat32 = 1, Varlen = 2 }

#[repr(u8)]
enum PrimType {
    UInt64 = 1,
    Int32 = 2,
    UInt32 = 3,
    Float32 = 4,
    Binary = 5,
    Int64 = 6,
    Int16 = 7,
    Int8 = 8,
    UInt16 = 9,
    UInt8 = 10,
}

// Re-export convenience helpers from codecs to keep call sites tidy.
use crate::codecs::{read_u32_le, read_u64_le, write_u32_le, write_u64_le};

/// Serialize array buffers with a minimal header (no nulls supported yet).
pub fn serialize_array(arr: &dyn Array) -> Result<Vec<u8>> {
    match arr.data_type() {
        &DataType::UInt64 => serialize_primitive(arr, PrimType::UInt64),
        &DataType::Int64 => serialize_primitive(arr, PrimType::Int64),
        &DataType::Int32 => serialize_primitive(arr, PrimType::Int32),
        &DataType::Int16 => serialize_primitive(arr, PrimType::Int16),
        &DataType::Int8 => serialize_primitive(arr, PrimType::Int8),
        &DataType::UInt32 => serialize_primitive(arr, PrimType::UInt32),
        &DataType::UInt16 => serialize_primitive(arr, PrimType::UInt16),
        &DataType::UInt8 => serialize_primitive(arr, PrimType::UInt8),
        &DataType::Float32 => serialize_primitive(arr, PrimType::Float32),
        &DataType::Binary => serialize_varlen(arr, PrimType::Binary),
        &DataType::FixedSizeList(ref child, list_size) => {
            if child.data_type() != &DataType::Float32 {
                return Err(Error::Internal("Only FixedSizeList<Float32> supported".into()));
            }
            serialize_fsl_float32(arr, i32::try_from(list_size).unwrap())
        }
        other => Err(Error::Internal(format!("Unsupported type in serialize_array: {other}"))),
    }
}

fn serialize_primitive(arr: &dyn Array, code: PrimType) -> Result<Vec<u8>> {
    if arr.null_count() != 0 {
        return Err(Error::Internal("nulls not supported in zero-copy format (yet)".into()));
    }
    let data = arr.to_data();
    let len = data.len() as u64;
    let values = data.buffers().get(0).ok_or_else(|| Error::Internal("missing values buffer".into()))?;
    let values_bytes = values.as_slice();
    let values_len = u32::try_from(values_bytes.len()).map_err(|_| Error::Internal("values too large".into()))?;
    let mut out = Vec::with_capacity(24 + values_bytes.len());
    out.extend_from_slice(&MAGIC);
    out.push(Layout::Primitive as u8);
    out.push(code as u8);
    out.extend_from_slice(&[0u8; 2]);
    write_u64_le(&mut out, len);
    write_u32_le(&mut out, values_len);
    write_u32_le(&mut out, 0);
    out.extend_from_slice(values_bytes);
    Ok(out)
}

fn serialize_varlen(arr: &dyn Array, code: PrimType) -> Result<Vec<u8>> {
    if arr.null_count() != 0 { return Err(Error::Internal("nulls not supported in zero-copy format (yet)".into())); }
    let data = arr.to_data();
    let len = data.len() as u64;
    let offsets_buf = data.buffers().get(0).ok_or_else(|| Error::Internal("missing offsets buffer".into()))?;
    let values_buf = data.buffers().get(1).ok_or_else(|| Error::Internal("missing values buffer for varlen".into()))?;
    let offsets_bytes = offsets_buf.as_slice();
    let values_bytes = values_buf.as_slice();
    let offsets_len = u32::try_from(offsets_bytes.len()).map_err(|_| Error::Internal("offsets buffer too large".into()))?;
    let values_len = u32::try_from(values_bytes.len()).map_err(|_| Error::Internal("values buffer too large".into()))?;
    let mut out = Vec::with_capacity(24 + offsets_bytes.len() + values_bytes.len());
    out.extend_from_slice(&MAGIC);
    out.push(Layout::Varlen as u8);
    out.push(code as u8);
    out.extend_from_slice(&[0u8; 2]);
    write_u64_le(&mut out, len);
    write_u32_le(&mut out, offsets_len);
    write_u32_le(&mut out, values_len);
    out.extend_from_slice(offsets_bytes);
    out.extend_from_slice(values_bytes);
    Ok(out)
}

fn serialize_fsl_float32(arr: &dyn Array, list_size: i32) -> Result<Vec<u8>> {
    if arr.null_count() != 0 { return Err(Error::Internal("nulls not supported in zero-copy format (yet)".into())); }
    let fsl = arr.as_any().downcast_ref::<FixedSizeListArray>().ok_or_else(|| Error::Internal("FSL downcast failed".into()))?;
    let values = fsl.values();
    if values.null_count() != 0 || values.data_type() != &DataType::Float32 {
        return Err(Error::Internal("FSL child must be non-null Float32".into()));
    }
    let child = values.to_data();
    let child_buf = child.buffers().get(0).ok_or_else(|| Error::Internal("missing child values".into()))?;
    let child_bytes = child_buf.as_slice();
    let child_len = u32::try_from(child_bytes.len()).map_err(|_| Error::Internal("child too large".into()))?;
    let mut out = Vec::with_capacity(24 + child_bytes.len());
    out.extend_from_slice(&MAGIC);
    out.push(Layout::FslFloat32 as u8);
    out.push(0);
    out.extend_from_slice(&[0u8; 2]);
    write_u64_le(&mut out, fsl.len() as u64);
    write_u32_le(&mut out, u32::try_from(list_size).unwrap());
    write_u32_le(&mut out, child_len);
    out.extend_from_slice(child_bytes);
    Ok(out)
}

/// Deserialize zero-copy from a pager blob.
pub fn deserialize_array(blob: EntryHandle) -> Result<ArrayRef> {
    let raw = blob.as_ref();
    if raw.len() < 24 || &raw[0..4] != &MAGIC { return Err(Error::Internal("bad array blob magic/size".into())); }
    let layout = raw[4];
    let type_code = raw[5];
    let mut o = 8usize;
    let len = read_u64_le(raw, &mut o) as usize;
    let extra_a = read_u32_le(raw, &mut o);
    let extra_b = read_u32_le(raw, &mut o);
    let whole: Buffer = blob.as_arrow_buffer();
    let payload: Buffer = whole.slice_with_length(o, whole.len() - o);
    match layout {
        x if x == Layout::Primitive as u8 => {
            let values_len = extra_a as usize;
            if payload.len() != values_len { return Err(Error::Internal("primitive payload length mismatch".into())); }
            let data_type = match type_code {
                x if x == PrimType::UInt64 as u8 => DataType::UInt64,
                x if x == PrimType::Int64 as u8 => DataType::Int64,
                x if x == PrimType::Int32 as u8 => DataType::Int32,
                x if x == PrimType::Int16 as u8 => DataType::Int16,
                x if x == PrimType::Int8 as u8 => DataType::Int8,
                x if x == PrimType::UInt32 as u8 => DataType::UInt32,
                x if x == PrimType::UInt16 as u8 => DataType::UInt16,
                x if x == PrimType::UInt8 as u8 => DataType::UInt8,
                x if x == PrimType::Float32 as u8 => DataType::Float32,
                _ => return Err(Error::Internal("unsupported primitive code".into())),
            };
            let data = ArrayData::builder(data_type).len(len).add_buffer(payload).build()?;
            Ok(make_array(data))
        }
        x if x == Layout::FslFloat32 as u8 => {
            let list_size = extra_a as i32;
            let child_values_len = extra_b as usize;
            if payload.len() != child_values_len { return Err(Error::Internal("fsl child length mismatch".into())); }
            let child_values = payload;
            let child_len = len * list_size as usize;
            let child_data = ArrayData::builder(DataType::Float32).len(child_len).add_buffer(child_values).build()?;
            let child = Arc::new(Float32Array::from(child_data)) as ArrayRef;
            let field = Arc::new(Field::new("item", DataType::Float32, false));
            let arr_data = ArrayData::builder(DataType::FixedSizeList(field, list_size)).len(len).add_child_data(child.to_data()).build()?;
            Ok(Arc::new(FixedSizeListArray::from(arr_data)))
        }
        x if x == Layout::Varlen as u8 => {
            let offsets_len = extra_a as usize;
            let values_len = extra_b as usize;
            if payload.len() != offsets_len + values_len { return Err(Error::Internal("varlen payload length mismatch".into())); }
            let offsets = payload.slice_with_length(0, offsets_len);
            let values = payload.slice_with_length(offsets_len, values_len);
            let data_type = match type_code {
                x if x == PrimType::Binary as u8 => DataType::Binary,
                _ => return Err(Error::Internal("unsupported varlen code".into())),
            };
            let data = ArrayData::builder(data_type).len(len).add_buffer(offsets).add_buffer(values).build()?;
            Ok(make_array(data))
        }
        _ => Err(Error::Internal("unknown layout".into())),
    }
}
