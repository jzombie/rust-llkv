//! Zero-copy array persistence for fixed/var width Arrow arrays used by the
//! store.
//!
//! ## Why a custom on-disk format instead of Arrow IPC?
//!
//! This module persists Arrow arrays in a minimal, mmap-friendly container so
//! we can reconstruct `ArrayData` *zero-copy* from a single memory-mapped
//! region. Using Arrow IPC (stream/file) directly would:
//!   - Increase file size from extra framing, padding, and metadata.
//!   - Require additional allocations and buffer copies during decode.
//!   - Prevent us from keeping a single contiguous payload per array, which
//!     hurts scan performance.
//!
//! Design goals of this format:
//! - **Minimal headers**: fixed-size header plus raw buffers only, no framing
//!   or schema objects.
//! - **Predictable contiguous payloads**: each array’s bytes live together in
//!   one region, ideal for mmap and SIMD access.
//! - **True zero-copy rebuild**: deserialization produces `ArrayData` that
//!   references the original mmap directly, avoiding memcpy.
//! - **Simpler invariants**: deliberately omits certain features (e.g., null
//!   bitmaps) to keep the format compact and reconstruction trivial.
//! - **Stable codes**: layout and type tags are explicitly pinned with
//!   compile-time checks to avoid silent corruption.
//!
//! Net effect: smaller files, faster scans, and preserved zero-copy semantics
//! tailored to this storage engine’s access pattern, at the cost of leaving
//! out some of the generality of Arrow IPC.

use std::convert::TryFrom;
use std::sync::Arc;

use arrow::array::{Array, ArrayData, ArrayRef, FixedSizeListArray, Float32Array, make_array};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Field, Schema};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use simd_r_drive_entry_handle::EntryHandle;

use llkv_result::{Error, Result};

const MAGIC: [u8; 4] = *b"ARR0";

/// On-disk layout selector for serialized Arrow arrays.
///
/// The zero-copy header is:
///   bytes 0..=3  : MAGIC = b"ARR0"
///   byte  4      : layout code (see `Layout`)
///   byte  5      : type code (PrimType; may be 0 for non-primitive layouts)
///   bytes 6..=7  : reserved (currently 0)
///   bytes 8..=15 : len (u64) = logical array length (number of elements)
///   bytes 16..=19: extra_a (u32) = layout-specific
///   bytes 20..=23: extra_b (u32) = layout-specific
///   bytes 24..   : payload (layout-specific), no null bitmaps supported yet
///
/// Variants:
///
/// - `Primitive`
///
///   - Fixed-width primitive arrays (e.g., Int32, UInt64, Float32, Float64).
///
///   - Header:
///     - len = element count
///     - extra_a = values_len (u32), byte length of the values buffer
///     - extra_b = 0
///
///   - Payload:
///     - [values buffer bytes]
///
///   - Notes:
///     - `type_code` (header byte 5) is a `PrimType` that chooses the Arrow
///       primitive `DataType`.
///     - Nulls are not supported yet (null_count must be 0).
///
/// - `FslFloat32`
///
///   - Specialized fast-path for `FixedSizeList<Float32>`. Encodes a
///     vector-like column (e.g., embeddings) as one contiguous `Float32`
///     child buffer, without per-list headers.
///
///   - Header:
///     - len = number of lists (rows)
///     - extra_a = list_size (u32), number of f32 values per list
///     - extra_b = child_values_len (u32), total bytes in the child f32
///       buffer
///
///   - Payload:
///     - [child Float32 values as a single buffer]
///       The child has `len * list_size` elements, each 4 bytes (f32).
///
///   - Constraints:
///     - The parent `FixedSizeList` has no nulls.
///     - The child array has `DataType::Float32` and no nulls.
///     - `type_code` (header byte 5) is unused here and written as 0.
///
///   - Rationale:
///     - Many workloads store dense float vectors (embeddings) as
///       `FixedSizeList<Float32>`. This variant avoids extra nesting
///       overhead and allows a direct slice into the contiguous f32 buffer.
///
/// - `Varlen`
///
///   - Variable-length, currently for `Binary` (offsets + values).
///
///   - Header:
///     - len = number of binary values
///     - extra_a = offsets_len (u32) in bytes
///     - extra_b = values_len (u32) in bytes
///
///   - Payload:
///     - [offsets buffer bytes][values buffer bytes]
///
///   - Notes:
///     - `type_code` is a `PrimType` and must be `Binary` for now.
///     - Nulls are not supported yet (null_count must be 0).
///
/// - **Struct** (`Layout::Struct`)
///
///   - Header:
///     - len = number of struct values
///     - extra_a = unused (0)
///     - extra_b = payload_len (u32) in bytes
///
///   - Payload:
///     - [IPC-serialized struct array bytes]
///
///   - Notes:
///     - Uses Arrow IPC format for struct serialization
///     - Nulls are not supported yet (null_count must be 0).
#[repr(u8)]
enum Layout {
    Primitive = 0,
    FslFloat32 = 1,
    Varlen = 2,
    Struct = 3,
}

/// Stable on-disk primitive type codes. Do not reorder. Only append new
/// variants at the end with explicit numeric values.
///
/// Using `num_enum` gives zero-cost Into/TryFrom conversions and avoids
/// manual matches for u8 <-> enum mapping.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, IntoPrimitive, TryFromPrimitive)]
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
    Float64 = 11,
    Utf8 = 12,
    LargeBinary = 13,
    LargeUtf8 = 14,
    Boolean = 15,
    Date32 = 16,
    Date64 = 17,
}

use crate::codecs::{read_u32_le, read_u64_le, write_u32_le, write_u64_le};

/// Map Arrow `DataType` to on-disk `PrimType`.
#[inline]
fn prim_from_datatype(dt: &DataType) -> Result<PrimType> {
    use DataType::*;
    let p = match dt {
        UInt64 => PrimType::UInt64,
        Int64 => PrimType::Int64,
        Int32 => PrimType::Int32,
        Int16 => PrimType::Int16,
        Int8 => PrimType::Int8,
        UInt32 => PrimType::UInt32,
        UInt16 => PrimType::UInt16,
        UInt8 => PrimType::UInt8,
        Float32 => PrimType::Float32,
        Float64 => PrimType::Float64,
        Binary => PrimType::Binary,
        Utf8 => PrimType::Utf8,
        LargeBinary => PrimType::LargeBinary,
        LargeUtf8 => PrimType::LargeUtf8,
        Boolean => PrimType::Boolean,
        Date32 => PrimType::Date32,
        Date64 => PrimType::Date64,
        _ => return Err(Error::Internal("unsupported Arrow type".into())),
    };
    Ok(p)
}

/// Map on-disk `PrimType` to Arrow `DataType`.
#[inline]
fn datatype_from_prim(p: PrimType) -> Result<DataType> {
    use DataType::*;
    let dt = match p {
        PrimType::UInt64 => UInt64,
        PrimType::Int64 => Int64,
        PrimType::Int32 => Int32,
        PrimType::Int16 => Int16,
        PrimType::Int8 => Int8,
        PrimType::UInt32 => UInt32,
        PrimType::UInt16 => UInt16,
        PrimType::UInt8 => UInt8,
        PrimType::Float32 => Float32,
        PrimType::Float64 => Float64,
        PrimType::Binary => Binary,
        PrimType::Utf8 => Utf8,
        PrimType::LargeBinary => LargeBinary,
        PrimType::LargeUtf8 => LargeUtf8,
        PrimType::Boolean => Boolean,
        PrimType::Date32 => Date32,
        PrimType::Date64 => Date64,
    };
    Ok(dt)
}

/// Serialize array buffers with a minimal header (no nulls supported yet).
pub fn serialize_array(arr: &dyn Array) -> Result<Vec<u8>> {
    match arr.data_type() {
        // Var-len path stays explicit to preserve layout.
        &DataType::Binary => serialize_varlen(arr, PrimType::Binary),
        &DataType::Utf8 => serialize_varlen(arr, PrimType::Utf8),
        &DataType::LargeBinary => serialize_varlen(arr, PrimType::LargeBinary),
        &DataType::LargeUtf8 => serialize_varlen(arr, PrimType::LargeUtf8),

        // Special-case fixed-size list of f32 as before.
        &DataType::FixedSizeList(ref child, list_size) => {
            if child.data_type() != &DataType::Float32 {
                return Err(Error::Internal(
                    "Only FixedSizeList<Float32> supported".into(),
                ));
            }
            serialize_fsl_float32(arr, list_size)
        }

        // Struct types use IPC serialization
        DataType::Struct(_) => serialize_struct(arr),

        // All remaining supported fixed-width primitives route here.
        dt => {
            let p = prim_from_datatype(dt)?;
            serialize_primitive(arr, p)
        }
    }
}

fn serialize_primitive(arr: &dyn Array, code: PrimType) -> Result<Vec<u8>> {
    if arr.null_count() != 0 {
        return Err(Error::Internal(
            "nulls not supported in zero-copy format (yet)".into(),
        ));
    }
    let data = arr.to_data();
    let len = data.len() as u64;
    let values = data
        .buffers()
        .first()
        .ok_or_else(|| Error::Internal("missing values buffer".into()))?;
    let values_bytes = values.as_slice();
    let values_len = u32::try_from(values_bytes.len())
        .map_err(|_| Error::Internal("values too large".into()))?;

    let mut out = Vec::with_capacity(24 + values_bytes.len());
    out.extend_from_slice(&MAGIC);
    out.push(Layout::Primitive as u8);
    out.push(u8::from(code));
    // 2 bytes padding reserved (e.g., future versioning).
    out.extend_from_slice(&[0u8; 2]);
    write_u64_le(&mut out, len);
    write_u32_le(&mut out, values_len);
    write_u32_le(&mut out, 0);
    out.extend_from_slice(values_bytes);
    Ok(out)
}

fn serialize_varlen(arr: &dyn Array, code: PrimType) -> Result<Vec<u8>> {
    if arr.null_count() != 0 {
        return Err(Error::Internal(
            "nulls not supported in zero-copy format (yet)".into(),
        ));
    }
    let data = arr.to_data();
    let len = data.len() as u64;

    let offsets_buf = data
        .buffers()
        .first()
        .ok_or_else(|| Error::Internal("missing offsets buffer".into()))?;
    let values_buf = data
        .buffers()
        .get(1)
        .ok_or_else(|| Error::Internal("missing values buffer for varlen".into()))?;

    let offsets_bytes = offsets_buf.as_slice();
    let values_bytes = values_buf.as_slice();

    let offsets_len = u32::try_from(offsets_bytes.len())
        .map_err(|_| Error::Internal("offsets buffer too large".into()))?;
    let values_len = u32::try_from(values_bytes.len())
        .map_err(|_| Error::Internal("values buffer too large".into()))?;

    let mut out = Vec::with_capacity(24 + offsets_bytes.len() + values_bytes.len());
    out.extend_from_slice(&MAGIC);
    out.push(Layout::Varlen as u8);
    out.push(u8::from(code));
    // 2 bytes padding reserved (e.g., future versioning).
    out.extend_from_slice(&[0u8; 2]);
    write_u64_le(&mut out, len);
    write_u32_le(&mut out, offsets_len);
    write_u32_le(&mut out, values_len);
    out.extend_from_slice(offsets_bytes);
    out.extend_from_slice(values_bytes);
    Ok(out)
}

fn serialize_fsl_float32(arr: &dyn Array, list_size: i32) -> Result<Vec<u8>> {
    if arr.null_count() != 0 {
        return Err(Error::Internal(
            "nulls not supported in zero-copy format (yet)".into(),
        ));
    }
    let fsl = arr
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| Error::Internal("FSL downcast failed".into()))?;

    let values = fsl.values();
    if values.null_count() != 0 || values.data_type() != &DataType::Float32 {
        return Err(Error::Internal("FSL child must be non-null Float32".into()));
    }

    let child = values.to_data();
    let child_buf = child
        .buffers()
        .first()
        .ok_or_else(|| Error::Internal("missing child values".into()))?;
    let child_bytes = child_buf.as_slice();

    let child_len =
        u32::try_from(child_bytes.len()).map_err(|_| Error::Internal("child too large".into()))?;

    let mut out = Vec::with_capacity(24 + child_bytes.len());
    out.extend_from_slice(&MAGIC);
    out.push(Layout::FslFloat32 as u8);
    out.push(0); // no PrimType for FSL header slot
    // 2 bytes padding reserved (e.g., future versioning).
    out.extend_from_slice(&[0u8; 2]);
    write_u64_le(&mut out, fsl.len() as u64);
    write_u32_le(&mut out, u32::try_from(list_size).unwrap());
    write_u32_le(&mut out, child_len);
    out.extend_from_slice(child_bytes);
    Ok(out)
}

fn serialize_struct(arr: &dyn Array) -> Result<Vec<u8>> {
    if arr.null_count() != 0 {
        return Err(Error::Internal(
            "nulls not supported in zero-copy format (yet)".into(),
        ));
    }

    // Use Arrow IPC format to serialize the struct array
    use arrow::ipc::writer::StreamWriter;
    use arrow::record_batch::RecordBatch;

    // Create a RecordBatch with a single column containing the struct array
    let schema = Arc::new(Schema::new(vec![Field::new(
        "struct_col",
        arr.data_type().clone(),
        false,
    )]));
    let array_ref = make_array(arr.to_data());
    let batch = RecordBatch::try_new(schema, vec![array_ref])
        .map_err(|e| Error::Internal(format!("failed to create record batch: {}", e)))?;

    // Serialize to IPC format
    let mut ipc_bytes = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut ipc_bytes, &batch.schema())
            .map_err(|e| Error::Internal(format!("failed to create IPC writer: {}", e)))?;
        writer
            .write(&batch)
            .map_err(|e| Error::Internal(format!("failed to write IPC: {}", e)))?;
        writer
            .finish()
            .map_err(|e| Error::Internal(format!("failed to finish IPC: {}", e)))?;
    }

    let payload_len = u32::try_from(ipc_bytes.len())
        .map_err(|_| Error::Internal("IPC payload too large".into()))?;

    let mut out = Vec::with_capacity(24 + ipc_bytes.len());
    out.extend_from_slice(&MAGIC);
    out.push(Layout::Struct as u8);
    out.push(0); // no PrimType for struct
    out.extend_from_slice(&[0u8; 2]); // padding
    write_u64_le(&mut out, arr.len() as u64);
    write_u32_le(&mut out, 0); // extra_a unused
    write_u32_le(&mut out, payload_len);
    out.extend_from_slice(&ipc_bytes);
    Ok(out)
}

/// Deserialize zero-copy from a pager blob.
pub fn deserialize_array(blob: EntryHandle) -> Result<ArrayRef> {
    let raw = blob.as_ref();
    if raw.len() < 24 || raw[0..4] != MAGIC {
        return Err(Error::Internal("bad array blob magic/size".into()));
    }

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
            if payload.len() != values_len {
                return Err(Error::Internal("primitive payload length mismatch".into()));
            }

            let p = PrimType::try_from(type_code)
                .map_err(|_| Error::Internal("unsupported primitive code".into()))?;
            let data_type = datatype_from_prim(p)?;

            let data = ArrayData::builder(data_type)
                .len(len)
                .add_buffer(payload)
                .build()?;
            Ok(make_array(data))
        }

        x if x == Layout::FslFloat32 as u8 => {
            let list_size = extra_a as i32;
            let child_values_len = extra_b as usize;
            if payload.len() != child_values_len {
                return Err(Error::Internal("fsl child length mismatch".into()));
            }

            let child_values = payload;
            let child_len = len * list_size as usize;

            let child_data = ArrayData::builder(DataType::Float32)
                .len(child_len)
                .add_buffer(child_values)
                .build()?;
            let child = Arc::new(Float32Array::from(child_data)) as ArrayRef;

            let field = Arc::new(Field::new("item", DataType::Float32, false));
            let arr_data = ArrayData::builder(DataType::FixedSizeList(field, list_size))
                .len(len)
                .add_child_data(child.to_data())
                .build()?;
            Ok(Arc::new(FixedSizeListArray::from(arr_data)))
        }

        x if x == Layout::Varlen as u8 => {
            let offsets_len = extra_a as usize;
            let values_len = extra_b as usize;
            if payload.len() != offsets_len + values_len {
                return Err(Error::Internal("varlen payload length mismatch".into()));
            }

            let offsets = payload.slice_with_length(0, offsets_len);
            let values = payload.slice_with_length(offsets_len, values_len);

            let p = PrimType::try_from(type_code)
                .map_err(|_| Error::Internal("unsupported varlen code".into()))?;
            let data_type = datatype_from_prim(p)?;

            let data = ArrayData::builder(data_type)
                .len(len)
                .add_buffer(offsets)
                .add_buffer(values)
                .build()?;
            Ok(make_array(data))
        }

        x if x == Layout::Struct as u8 => {
            let payload_len = extra_b as usize;
            if payload.len() != payload_len {
                return Err(Error::Internal("struct payload length mismatch".into()));
            }

            // Deserialize from IPC format
            use arrow::ipc::reader::StreamReader;
            use std::io::Cursor;

            let cursor = Cursor::new(payload.as_slice());
            let mut reader = StreamReader::try_new(cursor, None)
                .map_err(|e| Error::Internal(format!("failed to create IPC reader: {}", e)))?;

            let batch = reader
                .next()
                .ok_or_else(|| Error::Internal("no batch in IPC stream".into()))?
                .map_err(|e| Error::Internal(format!("failed to read IPC batch: {}", e)))?;

            if batch.num_columns() != 1 {
                return Err(Error::Internal(
                    "expected single column in struct batch".into(),
                ));
            }

            Ok(batch.column(0).clone())
        }

        _ => Err(Error::Internal("unknown layout".into())),
    }
}

/* ---- Compile-time pinning of on-disk codes -------------------------------
   Changing any discriminant silently would corrupt persistence. These const
   checks make such edits fail to compile immediately.
*/
#[allow(clippy::no_effect)]
const _: () = {
    // true -> 1, false -> 0; index out of bounds if false.
    ["code changed"][!(PrimType::UInt64 as u8 == 1) as usize];
    ["code changed"][!(PrimType::Int32 as u8 == 2) as usize];
    ["code changed"][!(PrimType::UInt32 as u8 == 3) as usize];
    ["code changed"][!(PrimType::Float32 as u8 == 4) as usize];
    ["code changed"][!(PrimType::Binary as u8 == 5) as usize];
    ["code changed"][!(PrimType::Int64 as u8 == 6) as usize];
    ["code changed"][!(PrimType::Int16 as u8 == 7) as usize];
    ["code changed"][!(PrimType::Int8 as u8 == 8) as usize];
    ["code changed"][!(PrimType::UInt16 as u8 == 9) as usize];
    ["code changed"][!(PrimType::UInt8 as u8 == 10) as usize];
    ["code changed"][!(PrimType::Float64 as u8 == 11) as usize];
    ["code changed"][!(PrimType::Utf8 as u8 == 12) as usize];
    ["code changed"][!(PrimType::LargeBinary as u8 == 13) as usize];
    ["code changed"][!(PrimType::LargeUtf8 as u8 == 14) as usize];
    ["code changed"][!(PrimType::Boolean as u8 == 15) as usize];
    ["code changed"][!(PrimType::Date32 as u8 == 16) as usize];
    ["code changed"][!(PrimType::Date64 as u8 == 17) as usize];
};
