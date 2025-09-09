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

use arrow_array::builder::{
    BinaryBuilder, FixedSizeListBuilder, Float32Builder, Int32Builder, Int64Builder, StringBuilder,
    UInt32Builder, UInt64Builder,
};
use arrow_array::{
    Array, ArrayRef, BinaryArray, FixedSizeListArray, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, LargeBinaryArray, LargeStringArray, RecordBatch,
    StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_buffer::NullBuffer;
use arrow_buffer::builder::BooleanBufferBuilder;
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

/// Convert a RecordBatch into a Vec<Put> ready for append_many.
///
/// All columns use ValueMode::Auto (per-column fixed/variable inferred).
pub fn batch_to_puts<'a>(batch: &'a RecordBatch, keys: &KeySpec, map: &ColumnMap) -> Vec<Put<'a>> {
    let rows = batch.num_rows();

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
        let col = batch.column(col_idx).clone();
        let items = gather_items(&col, rows, &key_bytes_for_row);
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
fn gather_items<'a>(
    col: &ArrayRef,
    rows: usize,
    key_bytes_for_row: &dyn Fn(usize) -> Option<Vec<u8>>,
) -> Vec<(Cow<'a, [u8]>, Cow<'a, [u8]>)> {
    let mut v: Vec<(Cow<[u8]>, Cow<[u8]>)> = Vec::with_capacity(rows);
    for row in 0..rows {
        let k = match key_bytes_for_row(row) {
            Some(k) => k,
            None => continue,
        };
        if let Some(val) = value_bytes(col, row) {
            v.push((Cow::Owned(k), Cow::Owned(val)));
        }
    }
    v
}

/// Turn one cell into bytes. Integers/floats use little-endian (LE).
///
/// For FixedSizeList of primitives (e.g., f32[1024]) we emit a single
/// fixed-width value per row by concatenating LE bytes of each element.
fn value_bytes(col: &ArrayRef, row: usize) -> Option<Vec<u8>> {
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
                return Some(bytes.to_vec());
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
        return Some(bits.to_le_bytes().to_vec());
    }
    if let Some(arr) = a.as_any().downcast_ref::<Float64Array>() {
        let bits = arr.value(row).to_bits();
        return Some(bits.to_le_bytes().to_vec());
    }

    // -------- strings/binary (variable width) --------
    if let Some(sa) = a.as_any().downcast_ref::<StringArray>() {
        return Some(sa.value(row).as_bytes().to_vec());
    }
    if let Some(sa) = a.as_any().downcast_ref::<LargeStringArray>() {
        return Some(sa.value(row).as_bytes().to_vec());
    }
    if let Some(ba) = a.as_any().downcast_ref::<BinaryArray>() {
        return Some(ba.value(row).to_vec());
    }
    if let Some(ba) = a.as_any().downcast_ref::<LargeBinaryArray>() {
        return Some(ba.value(row).to_vec());
    }

    // -------- fixed-size list of primitives (LE) --------
    if let Some(fsl) = a.as_any().downcast_ref::<FixedSizeListArray>() {
        let width = fsl.value_length() as usize;
        let child = fsl.values();
        let offset = (row * width) as usize;
        let mut out = Vec::with_capacity(width * 8);

        macro_rules! pack_slice {
            ($child_ty:ty, $encode:expr) => {{
                if let Some(ch) = child.as_any().downcast_ref::<$child_ty>() {
                    for i in 0..width {
                        let idx = offset + i;
                        let b = $encode(ch, idx);
                        out.extend_from_slice(&b);
                    }
                    return Some(out);
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

/// Return an owned Vec<u8> for a ValueSlice.
/// We cannot return a borrowed slice because `data()` returns an owned
/// blob; this function deliberately copies the window.
fn bytes_of<P: Pager>(vs: &ValueSlice<P::Blob>) -> Vec<u8> {
    let blob = vs.data(); // hold the owned blob so it outlives the slice
    let all = blob.as_ref();
    let a = vs.start() as usize;
    let b = vs.end() as usize;
    all[a..b].to_vec()
}

fn decode_u64_le<P: Pager>(vals: &[Option<ValueSlice<P::Blob>>]) -> ArrayRef {
    let mut b = UInt64Builder::with_capacity(vals.len());
    for v in vals {
        if let Some(vs) = v {
            let s = bytes_of::<P>(vs);
            if s.len() == 8 {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&s);
                b.append_value(u64::from_le_bytes(buf));
            } else {
                b.append_null();
            }
        } else {
            b.append_null();
        }
    }
    Arc::new(b.finish())
}

fn decode_u32_le<P: Pager>(vals: &[Option<ValueSlice<P::Blob>>]) -> ArrayRef {
    let mut b = UInt32Builder::with_capacity(vals.len());
    for v in vals {
        if let Some(vs) = v {
            let s = bytes_of::<P>(vs);
            if s.len() == 4 {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(&s);
                b.append_value(u32::from_le_bytes(buf));
            } else {
                b.append_null();
            }
        } else {
            b.append_null();
        }
    }
    Arc::new(b.finish())
}

fn decode_i64_le<P: Pager>(vals: &[Option<ValueSlice<P::Blob>>]) -> ArrayRef {
    let mut b = Int64Builder::with_capacity(vals.len());
    for v in vals {
        if let Some(vs) = v {
            let s = bytes_of::<P>(vs);
            if s.len() == 8 {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&s);
                b.append_value(i64::from_le_bytes(buf));
            } else {
                b.append_null();
            }
        } else {
            b.append_null();
        }
    }
    Arc::new(b.finish())
}

fn decode_i32_le<P: Pager>(vals: &[Option<ValueSlice<P::Blob>>]) -> ArrayRef {
    let mut b = Int32Builder::with_capacity(vals.len());
    for v in vals {
        if let Some(vs) = v {
            let s = bytes_of::<P>(vs);
            if s.len() == 4 {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(&s);
                b.append_value(i32::from_le_bytes(buf));
            } else {
                b.append_null();
            }
        } else {
            b.append_null();
        }
    }
    Arc::new(b.finish())
}

fn decode_utf8<P: Pager>(vals: &[Option<ValueSlice<P::Blob>>]) -> Result<ArrayRef, ArrowError> {
    let mut b = StringBuilder::with_capacity(vals.len(), 0);
    for v in vals {
        if let Some(vs) = v {
            let s = bytes_of::<P>(vs);
            let strv =
                std::str::from_utf8(&s).map_err(|e| ArrowError::ParseError(e.to_string()))?;
            b.append_value(strv);
        } else {
            b.append_null();
        }
    }
    Ok(Arc::new(b.finish()))
}

fn decode_binary<P: Pager>(vals: &[Option<ValueSlice<P::Blob>>]) -> ArrayRef {
    let mut b = BinaryBuilder::with_capacity(vals.len(), 0);
    for v in vals {
        if let Some(vs) = v {
            let s = bytes_of::<P>(vs);
            b.append_value(&s);
        } else {
            b.append_null();
        }
    }
    Arc::new(b.finish())
}

fn decode_f32_fixed<P: Pager>(
    vals: &[Option<ValueSlice<P::Blob>>],
    len: i32,
) -> Result<ArrayRef, ArrowError> {
    let len_usize = len as usize;

    // Build child values (no nulls in child) and track parent validity.
    let mut child = Float32Builder::with_capacity(vals.len() * len_usize);
    let mut any_null = false;
    let mut valid_bits = BooleanBufferBuilder::new(vals.len());

    for v in vals {
        if let Some(vs) = v {
            let s = bytes_of::<P>(vs);
            if s.len() != len_usize * 4 {
                // Malformed row: mark parent null, fill child with dummies.
                any_null = true;
                valid_bits.append(false);
                for _ in 0..len_usize {
                    child.append_value(0.0);
                }
                continue;
            }
            // Good row: decode len LE f32 values.
            for i in 0..len_usize {
                let off = i * 4;
                let mut buf = [0u8; 4];
                buf.copy_from_slice(&s[off..off + 4]);
                let f = f32::from_le_bytes(buf);
                child.append_value(f);
            }
            valid_bits.append(true);
        } else {
            // Missing row: parent null, child gets dummy values.
            any_null = true;
            valid_bits.append(false);
            for _ in 0..len_usize {
                child.append_value(0.0);
            }
        }
    }

    // Child array: plain Float32 with no nulls.
    let child_arr: ArrayRef = Arc::new(child.finish());

    // Parent validity buffer, only if we had any nulls.
    let nulls = if any_null {
        Some(NullBuffer::from(valid_bits.finish()))
    } else {
        None
    };

    // Child field MUST be non-nullable to match your schema.
    let item_field = Arc::new(Field::new("item", DataType::Float32, false));

    let arr = FixedSizeListArray::new(item_field, len, child_arr, nulls);
    Ok(Arc::new(arr))
}
