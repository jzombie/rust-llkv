// src/arrow_bridge.rs

//! Arrow <-> llkv-column-map ingest bridge.
//!
//! Keys are taken from a user-selected column. For integer keys we
//! encode as big-endian bytes so lex order matches numeric order.
//!
//! Nulls are skipped (no Put for that row). If you want nulls to map
//! to tombstones, pass ValueMode::Auto and write empty slices for
//! variable-width columns only.

use std::borrow::Cow;

use arrow_array::{
    Array, ArrayRef, BinaryArray, FixedSizeListArray, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, LargeBinaryArray, LargeStringArray, RecordBatch,
    StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use llkv_column_map::ColumnStore;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode};

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
        pack_slice!(UInt16Array, |c: &UInt16Array, i| c.value(i).to_le_bytes());
        pack_slice!(Int16Array, |c: &Int16Array, i| c.value(i).to_le_bytes());
        pack_slice!(UInt32Array, |c: &UInt32Array, i| c.value(i).to_le_bytes());
        pack_slice!(Int32Array, |c: &Int32Array, i| c.value(i).to_le_bytes());
        pack_slice!(UInt64Array, |c: &UInt64Array, i| c.value(i).to_le_bytes());
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
    // Ensure ValueMode::Auto so different columns can choose their own
    // layout (fixed vs variable) in one append_many call.
    opts.mode = ValueMode::Auto;
    let puts = batch_to_puts(batch, keys, map);
    if !puts.is_empty() {
        store.append_many(puts, opts);
    }
}
