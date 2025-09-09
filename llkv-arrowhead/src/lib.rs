//! Arrow <-> llkv-column-map bridge.
//!
//! Writer:
//!   - Builds Put vectors from an Arrow RecordBatch.
//!   - Keys: BE if using numeric u64; values encoded LE.
//!
//! Reader:
//!   - Builds an Arrow RecordBatch via ColumnStore::get_many.
//!   - Decodes values with codecs (LE numerics, UTF-8, binary,
//!     fixed-size f32 vectors).

#![forbid(unsafe_code)]

use std::borrow::Cow;
use std::sync::Arc;

use arrow::array::ArrayData;
use arrow_array::builder::StringBuilder;
use arrow_array::types::{Int32Type, Int64Type, UInt32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, BinaryArray, FixedSizeListArray, Float32Array, Float64Array,
    GenericBinaryArray, Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray,
    LargeStringArray, PrimitiveArray, RecordBatch, StringArray, UInt8Array, UInt16Array,
    UInt32Array, UInt64Array,
};
use arrow_buffer::{Buffer, NullBuffer, ScalarBuffer, builder::BooleanBufferBuilder};
use arrow_schema::{ArrowError, DataType, Field, Schema};

use llkv_column_map::ColumnStore;
use llkv_column_map::storage::pager::Pager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, LogicalKeyBytes, Put, ValueMode};
use llkv_column_map::views::ValueSlice;

/* ============================== WRITER ============================== */

/// Which column supplies row keys, and how to encode them.
pub enum KeySpec {
    /// Unsigned 64-bit column -> big-endian bytes.
    U64Be { col_idx: usize },
    /// UTF-8 string column -> UTF-8 bytes.
    Utf8 { col_idx: usize },
    /// Variable-width binary column -> raw bytes.
    Bin { col_idx: usize },
}

/// Maps Arrow columns in a batch to logical field IDs in the store.
pub struct ColumnMap {
    /// (arrow_col_idx, field_id)
    pub cols: Vec<(usize, LogicalFieldId)>,
}

/// Convert a RecordBatch into a `Vec<Put>` ready for append_many.
///
/// All columns use ValueMode::Auto (per-column fixed/variable inferred).
pub fn batch_to_puts<'a>(batch: &'a RecordBatch, keys: &KeySpec, map: &ColumnMap) -> Vec<Put<'a>> {
    let rows = batch.num_rows();

    // TODO (ZERO-COPY): This closure could be updated to return Cow<'a, [u8]>
    // to avoid allocating a Vec for every key, especially for Utf8 and Bin variants.
    let key_bytes_for_row = |row: usize| -> Option<Vec<u8>> {
        match *keys {
            KeySpec::U64Be { col_idx } => {
                let arr = batch.column(col_idx);
                let arr = arr
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .expect("key col must be UInt64Array");
                if arr.is_null(row) {
                    return None;
                }
                Some(arr.value(row).to_be_bytes().to_vec())
            }
            KeySpec::Utf8 { col_idx } => {
                let a = batch.column(col_idx).as_any();
                if let Some(sa) = a.downcast_ref::<StringArray>() {
                    if sa.is_null(row) {
                        return None;
                    }
                    return Some(sa.value(row).as_bytes().to_vec());
                }
                if let Some(la) = a.downcast_ref::<LargeStringArray>() {
                    if la.is_null(row) {
                        return None;
                    }
                    return Some(la.value(row).as_bytes().to_vec());
                }
                panic!("key col must be StringArray or LargeStringArray");
            }
            KeySpec::Bin { col_idx } => {
                let a = batch.column(col_idx).as_any();
                if let Some(ba) = a.downcast_ref::<BinaryArray>() {
                    if ba.is_null(row) {
                        return None;
                    }
                    return Some(ba.value(row).to_vec());
                }
                if let Some(la) = a.downcast_ref::<LargeBinaryArray>() {
                    if la.is_null(row) {
                        return None;
                    }
                    return Some(la.value(row).to_vec());
                }
                panic!("key col must be BinaryArray or LargeBinaryArray");
            }
        }
    };

    // Build Put per mapped column.
    let mut out: Vec<Put<'a>> = Vec::with_capacity(map.cols.len());
    for &(col_idx, fid) in &map.cols {
        // CORRECTED: Borrow the column directly from the batch's slice of columns.
        // This ensures the reference `col` has the lifetime 'a, resolving the E0597 error.
        // `batch.columns()` returns `&'a [ArrayRef]`, so `&batch.columns()[col_idx]` is `&'a ArrayRef`.
        let col = &batch.columns()[col_idx];
        let items = gather_items(col, rows, &key_bytes_for_row);
        if !items.is_empty() {
            out.push(Put {
                field_id: fid,
                items,
            });
        }
    }
    out
}

/// Extract (key, value) pairs for one Arrow column.
#[allow(clippy::type_complexity)] // TODO: Use return type alias
fn gather_items<'a>(
    col: &'a ArrayRef,
    rows: usize,
    key_bytes_for_row: &dyn Fn(usize) -> Option<Vec<u8>>,
) -> Vec<(Cow<'a, [u8]>, Cow<'a, [u8]>)> {
    let mut v: Vec<(Cow<[u8]>, Cow<[u8]>)> = Vec::with_capacity(rows);
    for row in 0..rows {
        let k = match key_bytes_for_row(row) {
            Some(k) => k,
            None => continue,
        };
        // The `value_bytes` function now returns a Cow, which can be borrowed for
        // string/binary types, avoiding a copy. Primitives are still owned.
        if let Some(val) = value_bytes(col, row) {
            v.push((Cow::Owned(k), val));
        }
    }
    v
}

/// Turn one cell into bytes. Integers/floats use little-endian (LE).
/// For FixedSizeList of primitives we emit a single fixed-width value
/// per row by concatenating LE bytes of each element.
///
/// ZERO-COPY CHANGE: This function now returns `Cow<'a, [u8]>`. For string
/// and binary types, this allows us to borrow the slice directly from the
/// Arrow array without any new allocation. For primitive types, we still
/// create an owned Vec, but the interface is more efficient.
fn value_bytes<'a>(col: &'a ArrayRef, row: usize) -> Option<Cow<'a, [u8]>> {
    let a = col.as_ref();
    if a.is_null(row) {
        return None;
    }

    // -------- primitives (LE) --------
    macro_rules! prim {
        ($t:ty, $to:ident) => {{
            if let Some(arr) = a.as_any().downcast_ref::<$t>() {
                let x = arr.value(row);
                let bytes = x.$to();
                return Some(Cow::Owned(bytes.to_vec()));
            }
        }};
    }
    prim!(UInt8Array, to_le_bytes);
    prim!(UInt16Array, to_le_bytes);
    prim!(UInt32Array, to_le_bytes);
    prim!(UInt64Array, to_le_bytes);
    prim!(Int8Array, to_le_bytes);
    prim!(Int16Array, to_le_bytes);
    prim!(Int32Array, to_le_bytes);
    prim!(Int64Array, to_le_bytes);

    // Floats: encode IEEE-754 bits in LE.
    if let Some(arr) = a.as_any().downcast_ref::<Float32Array>() {
        let bits = arr.value(row).to_bits();
        return Some(Cow::Owned(bits.to_le_bytes().to_vec()));
    }
    if let Some(arr) = a.as_any().downcast_ref::<Float64Array>() {
        let bits = arr.value(row).to_bits();
        return Some(Cow::Owned(bits.to_le_bytes().to_vec()));
    }

    // -------- strings/binary (variable width) --------
    if let Some(sa) = a.as_any().downcast_ref::<StringArray>() {
        return Some(Cow::Borrowed(sa.value(row).as_bytes()));
    }
    if let Some(sa) = a.as_any().downcast_ref::<LargeStringArray>() {
        return Some(Cow::Borrowed(sa.value(row).as_bytes()));
    }
    if let Some(ba) = a.as_any().downcast_ref::<BinaryArray>() {
        return Some(Cow::Borrowed(ba.value(row)));
    }
    if let Some(ba) = a.as_any().downcast_ref::<LargeBinaryArray>() {
        return Some(Cow::Borrowed(ba.value(row)));
    }

    // -------- fixed-size list of primitives (LE) --------
    if let Some(fsl) = a.as_any().downcast_ref::<FixedSizeListArray>() {
        let width = fsl.value_length() as usize;
        let child = fsl.values();
        let offset = row * width;
        let mut out = Vec::with_capacity(width * 8);

        macro_rules! pack_slice {
            ($child_ty:ty, $encode:expr) => {{
                if let Some(ch) = child.as_any().downcast_ref::<$child_ty>() {
                    for i in 0..width {
                        let idx = offset + i;
                        let b = $encode(ch, idx);
                        out.extend_from_slice(&b);
                    }
                    return Some(Cow::Owned(out));
                }
            }};
        }
        pack_slice!(UInt8Array, |c: &UInt8Array, i| c.value(i).to_le_bytes());
        pack_slice!(Int8Array, |c: &Int8Array, i| c.value(i).to_le_bytes());
        pack_slice!(UInt16Array, |c: &UInt16Array, i| {
            c.value(i).to_le_bytes()
        });
        pack_slice!(Int16Array, |c: &Int16Array, i| c.value(i).to_le_bytes());
        pack_slice!(UInt32Array, |c: &UInt32Array, i| {
            c.value(i).to_le_bytes()
        });
        pack_slice!(Int32Array, |c: &Int32Array, i| c.value(i).to_le_bytes());
        pack_slice!(UInt64Array, |c: &UInt64Array, i| {
            c.value(i).to_le_bytes()
        });
        pack_slice!(Int64Array, |c: &Int64Array, i| c.value(i).to_le_bytes());
        pack_slice!(Float32Array, |c: &Float32Array, i| {
            c.value(i).to_bits().to_le_bytes()
        });
        pack_slice!(Float64Array, |c: &Float64Array, i| {
            c.value(i).to_bits().to_le_bytes()
        });

        panic!("unsupported FixedSizeList child type");
    }

    panic!("unsupported Arrow array type for value conversion");
}

/// Append a batch to the store in one call using mode=Auto.
///
/// This lets each column independently choose fixed vs variable layout.
pub fn append_batch<P: llkv_column_map::storage::pager::Pager>(
    store: &ColumnStore<P>,
    batch: &RecordBatch,
    keys: &KeySpec,
    map: &ColumnMap,
    mut opts: AppendOptions,
) {
    opts.mode = ValueMode::Auto;
    let puts = batch_to_puts(batch, keys, map);
    if !puts.is_empty() {
        store.append_many(puts, opts);
    }
}

/* ============================== READER ============================== */

/// Arrow read bridge built on ColumnStore::get_many.
/// Caller supplies ReadSpec per column. Rows align with `keys`.
/// Numerics decode as LE. Keys remain BE.
#[derive(Clone, Debug)]
pub enum ReadCodec {
    U64Le,
    U32Le,
    I64Le,
    I32Le,
    Utf8,
    Binary,
    /// Fixed-size vector of f32, length `len`, LE elements.
    F32Fixed {
        len: i32,
    },
}

/// Per-column read plan.
#[derive(Clone, Debug)]
pub struct ReadSpec {
    /// Logical field id in your store.
    pub field_id: LogicalFieldId,
    /// Arrow field to emit (name, datatype, nullability).
    pub field: Field,
    /// Codec for decoding bytes -> Arrow array.
    pub codec: ReadCodec,
}

/// Build a RecordBatch for `specs` and `keys` using get_many.
/// Rows align with `keys`. Each spec becomes one Arrow column.
pub fn read_record_batch<P: Pager>(
    store: &ColumnStore<P>,
    keys: &[LogicalKeyBytes],
    specs: &[ReadSpec],
) -> Result<RecordBatch, ArrowError> {
    // 1) get_many for all requested fields.
    let items = specs
        .iter()
        .map(|s| (s.field_id, keys.to_vec()))
        .collect::<Vec<_>>();
    let fetched = store.get_many(items);
    debug_assert_eq!(fetched.len(), specs.len());

    // 2) Decode per column.
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(specs.len());
    for (i, spec) in specs.iter().enumerate() {
        let col_vals = &fetched[i];
        let arr = decode_column::<P>(spec, col_vals)?;
        arrays.push(arr);
    }

    // 3) Assemble schema and batch.
    let fields = specs
        .iter()
        .map(|s| Arc::new(s.field.clone()))
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays)
}

/// Decode one column based on a ReadSpec.
fn decode_column<P: Pager>(
    spec: &ReadSpec,
    vals: &[Option<ValueSlice<P::Blob>>],
) -> Result<ArrayRef, ArrowError> {
    match spec.codec {
        ReadCodec::U64Le => Ok(decode_u64_le::<P>(vals)),
        ReadCodec::U32Le => Ok(decode_u32_le::<P>(vals)),
        ReadCodec::I64Le => Ok(decode_i64_le::<P>(vals)),
        ReadCodec::I32Le => Ok(decode_i32_le::<P>(vals)),
        ReadCodec::Utf8 => decode_utf8::<P>(vals),
        ReadCodec::Binary => Ok(decode_binary::<P>(vals)),
        ReadCodec::F32Fixed { len } => decode_f32_fixed::<P>(vals, len),
    }
}

/// Minimal-copy decoder for u64.
/// It iterates through the value slices, copies their content into a single
/// contiguous buffer, and then creates an Arrow array from that buffer. This
/// avoids per-value heap allocations.
fn decode_u64_le<P: Pager>(vals: &[Option<ValueSlice<P::Blob>>]) -> ArrayRef {
    let mut values_buffer = Vec::<u8>::with_capacity(vals.len() * 8);
    let mut nulls = BooleanBufferBuilder::new(vals.len());

    for v in vals {
        if let Some(vs) = v {
            let s = vs.as_slice();
            if s.len() == 8 {
                values_buffer.extend_from_slice(s);
                nulls.append(true);
            } else {
                values_buffer.extend_from_slice(&[0u8; 8]); // Dummy data for null slot
                nulls.append(false);
            }
        } else {
            values_buffer.extend_from_slice(&[0u8; 8]); // Dummy data for null slot
            nulls.append(false);
        }
    }

    let array_data = ArrayData::builder(DataType::UInt64)
        .len(vals.len())
        .add_buffer(Buffer::from(values_buffer))
        .nulls(Some(NullBuffer::from(nulls.finish())))
        .build()
        .unwrap();

    Arc::new(PrimitiveArray::<UInt64Type>::from(array_data))
}

/// Minimal-copy decoder for u32.
fn decode_u32_le<P: Pager>(vals: &[Option<ValueSlice<P::Blob>>]) -> ArrayRef {
    let mut values_buffer = Vec::<u8>::with_capacity(vals.len() * 4);
    let mut nulls = BooleanBufferBuilder::new(vals.len());

    for v in vals {
        if let Some(vs) = v {
            let s = vs.as_slice();
            if s.len() == 4 {
                values_buffer.extend_from_slice(s);
                nulls.append(true);
            } else {
                values_buffer.extend_from_slice(&[0u8; 4]);
                nulls.append(false);
            }
        } else {
            values_buffer.extend_from_slice(&[0u8; 4]);
            nulls.append(false);
        }
    }

    let array_data = ArrayData::builder(DataType::UInt32)
        .len(vals.len())
        .add_buffer(Buffer::from(values_buffer))
        .nulls(Some(NullBuffer::from(nulls.finish())))
        .build()
        .unwrap();

    Arc::new(PrimitiveArray::<UInt32Type>::from(array_data))
}

/// Minimal-copy decoder for i64.
fn decode_i64_le<P: Pager>(vals: &[Option<ValueSlice<P::Blob>>]) -> ArrayRef {
    let mut values_buffer = Vec::<u8>::with_capacity(vals.len() * 8);
    let mut nulls = BooleanBufferBuilder::new(vals.len());

    for v in vals {
        if let Some(vs) = v {
            let s = vs.as_slice();
            if s.len() == 8 {
                values_buffer.extend_from_slice(s);
                nulls.append(true);
            } else {
                values_buffer.extend_from_slice(&[0u8; 8]);
                nulls.append(false);
            }
        } else {
            values_buffer.extend_from_slice(&[0u8; 8]);
            nulls.append(false);
        }
    }

    let array_data = ArrayData::builder(DataType::Int64)
        .len(vals.len())
        .add_buffer(Buffer::from(values_buffer))
        .nulls(Some(NullBuffer::from(nulls.finish())))
        .build()
        .unwrap();

    Arc::new(PrimitiveArray::<Int64Type>::from(array_data))
}

/// Minimal-copy decoder for i32.
fn decode_i32_le<P: Pager>(vals: &[Option<ValueSlice<P::Blob>>]) -> ArrayRef {
    let mut values_buffer = Vec::<u8>::with_capacity(vals.len() * 4);
    let mut nulls = BooleanBufferBuilder::new(vals.len());

    for v in vals {
        if let Some(vs) = v {
            let s = vs.as_slice();
            if s.len() == 4 {
                values_buffer.extend_from_slice(s);
                nulls.append(true);
            } else {
                values_buffer.extend_from_slice(&[0u8; 4]);
                nulls.append(false);
            }
        } else {
            values_buffer.extend_from_slice(&[0u8; 4]);
            nulls.append(false);
        }
    }

    let array_data = ArrayData::builder(DataType::Int32)
        .len(vals.len())
        .add_buffer(Buffer::from(values_buffer))
        .nulls(Some(NullBuffer::from(nulls.finish())))
        .build()
        .unwrap();

    Arc::new(PrimitiveArray::<Int32Type>::from(array_data))
}

/// Minimal-copy decoder for Utf8.
/// This is more complex because it involves validation. While we can build the
/// buffers directly, we must still validate the UTF-8 content.
fn decode_utf8<P: Pager>(vals: &[Option<ValueSlice<P::Blob>>]) -> Result<ArrayRef, ArrowError> {
    // For UTF-8, using the builder is often safest to ensure validation.
    // A true zero-copy approach would be unsafe if the source bytes are not
    // guaranteed to be valid UTF-8. This implementation prioritizes correctness.
    let mut b = StringBuilder::with_capacity(vals.len(), 0);
    for v in vals {
        if let Some(vs) = v {
            let s = vs.as_slice();
            let strv = std::str::from_utf8(s).map_err(|e| ArrowError::ParseError(e.to_string()))?;
            b.append_value(strv);
        } else {
            b.append_null();
        }
    }
    Ok(Arc::new(b.finish()))
}

/// Minimal-copy decoder for Binary.
/// We build the offset and data buffers directly from the value slices.
fn decode_binary<P: Pager>(vals: &[Option<ValueSlice<P::Blob>>]) -> ArrayRef {
    let mut data_buffer = Vec::<u8>::new();
    let mut offsets = Vec::<i32>::with_capacity(vals.len() + 1);
    offsets.push(0);
    let mut nulls = BooleanBufferBuilder::new(vals.len());

    for v in vals {
        if let Some(vs) = v {
            data_buffer.extend_from_slice(vs.as_slice());
            offsets.push(data_buffer.len() as i32);
            nulls.append(true);
        } else {
            offsets.push(data_buffer.len() as i32); // Offset doesn't change for a null value
            nulls.append(false);
        }
    }

    let array_data = ArrayData::builder(DataType::Binary)
        .len(vals.len())
        .add_buffer(Buffer::from(ScalarBuffer::from(offsets))) // Offsets buffer
        .add_buffer(Buffer::from(data_buffer)) // Data buffer
        .nulls(Some(NullBuffer::from(nulls.finish())))
        .build()
        .unwrap();

    Arc::new(GenericBinaryArray::<i32>::from(array_data))
}

/// Minimal-copy decoder for `FixedSizeList<f32>`.
fn decode_f32_fixed<P: Pager>(
    vals: &[Option<ValueSlice<P::Blob>>],
    len: i32,
) -> Result<ArrayRef, ArrowError> {
    let len_usize = len as usize;
    let expected_byte_len = len_usize * 4;

    // This is a minimal-copy approach. We build one large buffer for the child array.
    // A true zero-copy would require the underlying storage to provide a single,
    // contiguous buffer for all rows, which `get_many` does not guarantee.
    let mut child_buffer = Vec::<u8>::with_capacity(vals.len() * expected_byte_len);
    let mut any_null = false;
    let mut valid_bits = BooleanBufferBuilder::new(vals.len());

    for v in vals {
        if let Some(vs) = v {
            let s = vs.as_slice();
            if s.len() != expected_byte_len {
                // Malformed row: mark parent null, fill child with dummies.
                any_null = true;
                valid_bits.append(false);
                child_buffer.extend_from_slice(&vec![0u8; expected_byte_len]);
            } else {
                // Good row: copy the bytes.
                child_buffer.extend_from_slice(s);
                valid_bits.append(true);
            }
        } else {
            // Missing row: parent null, child gets dummy values.
            any_null = true;
            valid_bits.append(false);
            child_buffer.extend_from_slice(&vec![0u8; expected_byte_len]);
        }
    }

    // Child array data (plain Float32 with no nulls of its own).
    let child_array_data = ArrayData::builder(DataType::Float32)
        .len(vals.len() * len_usize)
        .add_buffer(Buffer::from(child_buffer))
        .build()
        .unwrap();
    let child_arr: ArrayRef = Arc::new(Float32Array::from(child_array_data));

    // Parent validity buffer, only if we had any nulls.
    let nulls = if any_null {
        Some(NullBuffer::from(valid_bits.finish()))
    } else {
        None
    };

    // The child field in the list's data type MUST be non-nullable.
    let item_field = Arc::new(Field::new("item", DataType::Float32, false));

    let arr = FixedSizeListArray::new(item_field, len, child_arr, nulls);
    Ok(Arc::new(arr))
}
