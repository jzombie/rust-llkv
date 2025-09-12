use std::borrow::Cow;

/// Opaque 64-bit address in the KV/pager namespace.
///
/// - Treated as an **opaque handle** by higher layers (never interpreted).
/// - Stable across process restarts if your pager persists it.
/// - In this crate, **physical key `0` is reserved** for the Bootstrap record.
/// - Typical implementations allocate keys monotonically, but callers MUST NOT
///   rely on contiguity or ordering.
///
/// Examples: page IDs, log sequence numbers, virtual block addresses.
pub type PhysicalKey = u64;

/// Logical column/field identifier chosen by the application.
///
/// - Used to group rows into a “column” in the manifest and to locate that
///   column’s current `ColumnIndex`.
/// - Does not need to be dense or sequential; it only needs to be unique per column.
pub type LogicalFieldId = u64;

/// Already-encoded **logical key bytes** (application format).
///
/// - ColumnStore and index structures treat this as an ordered byte string.
/// - Sorting is **lexicographical on bytes**. If you need numeric ordering,
///   encode numbers in **big-endian** (e.g., `u64::to_be_bytes`) so byte order
///   matches numeric order.
/// - Duplicates may appear in a single append batch; “last write wins” behavior
///   is controlled by `AppendOptions::last_write_wins_in_batch`.
pub type LogicalKeyBytes = Vec<u8>;

/// Number of entries (key/value pairs) in a segment or structure.
///
/// - Kept as `u32` to minimize header size.
/// - Implies a per-segment limit of `< 2^32` entries. If you need more,
///   split into multiple segments.
pub type IndexEntryCount = u32;

/// A **width in bytes** for fixed-width layouts.
///
/// - Used when all values (or all keys) in a segment share a single width.
/// - Valid values are `> 0` for non-empty fixed layouts. A width of `0` is
///   only meaningful for empty segments (no entries).
pub type ByteWidth = u32;

/// A **byte offset** into a concatenated byte blob (prefix-sum array elements).
///
/// - Used by variable-width layouts to delimit slices: slice `i` is
///   `[offsets[i], offsets[i+1])`.
/// - Monotonically non-decreasing; the final element equals the total byte len.
/// - `u32` keeps headers compact and caps any single blob at `< 4 GiB`.
///   If you need to exceed that, use multiple blobs/segments or switch to 64-bit.
pub type ByteOffset = u32;

/// A **byte length** of a single slice/value.
///
/// - Often used to build prefix-sum offset arrays for variable-width values.
/// - Summing `ByteLen`s produces the final `ByteOffset` tail value.
/// - Stored as `u32` for compactness; thus each individual value must be
///   `< 4 GiB` and any single blob must also remain `< 4 GiB`.
pub type ByteLen = u32;

/// Any clonable, thread-safe buffer that can be viewed as `&[u8]`.
pub trait BlobLike: AsRef<[u8]> + Clone + Send + Sync + 'static {}
impl<T> BlobLike for T where T: AsRef<[u8]> + Clone + Send + Sync + 'static {}

/// Wire tag for typed blobs stored in the pager.
///
/// Each persisted metadata object (e.g., `Manifest`, `ColumnIndex`, `IndexSegment`)
/// is saved as a *typed* bitcode blob. `TypedKind` is the small, stable tag
/// that tells the pager which concrete type to decode into on read.
///
/// Why this exists:
/// - **Safety:** Prevents decoding a value as the wrong struct when multiple
///   blob types share the same key space.
/// - **Speed:** Lets the pager pick the right decode path without trial-and-error.
/// - **Clarity:** Callers explicitly state what they expect to read.
///
/// Versioning notes:
/// - Treat this as a wire-level enum. Adding new variants is fine, but removing
///   or reusing existing variants will break compatibility for blobs already
///   written.
/// - If you evolve on-disk layouts, version those structs (or the manifest)
///   rather than reusing a `TypedKind` variant with a new meaning.
#[derive(Clone, Copy, Debug)]
pub enum TypedKind {
    /// Tiny record that points to the current `Manifest` (often at physical key 0).
    Bootstrap,
    /// Top-level directory mapping columns to their current `ColumnIndex`.
    Manifest,
    /// Per-column list of sealed segments (newest-first) with fast-prune bounds.
    ColumnIndex,
    /// Packed logical keys and value layout for one sealed segment (points to data blob).
    IndexSegment,
}

/// Type-erased typed value. Concrete structs live in your crate's
/// `index` module. This avoids `Any`/downcasts at callers.
#[derive(Clone, Debug)]
pub enum TypedValue {
    Bootstrap(crate::column_index::Bootstrap),
    Manifest(crate::column_index::Manifest),
    ColumnIndex(crate::column_index::ColumnIndex),
    IndexSegment(crate::column_index::IndexSegment),
}

/// Type-erased container for all metadata blobs we persist via `bitcode`.
///
/// At write time you construct the concrete value (e.g., `Manifest`) and wrap it
/// in the corresponding `TypedValue` variant before handing it to the pager.
/// At read time, you ask the pager for a given `TypedKind`, and it returns a
/// `TypedValue` you can `match` to regain the concrete type.
///
/// Why not `Any`/downcast?
/// - This enum is **zero-alloc** and **explicit** about what the storage layer
///   can serialize/deserialize. It’s also friendlier to non-std environments.
///
/// Gotchas:
/// - The `TypedKind` you request **must** match the variant that was written,
///   or decode will fail (by design).
/// - Keep these variants aligned with the set of persisted structs in
///   `crate::column_index`; treat changes as on-disk format changes.
impl TypedValue {
    /// Return the wire tag (`TypedKind`) corresponding to this value.
    ///
    /// Useful when building batched puts where the pager needs to know the kind
    /// alongside the encoded bytes.
    #[inline]
    pub fn kind(&self) -> TypedKind {
        match self {
            TypedValue::Bootstrap(_) => TypedKind::Bootstrap,
            TypedValue::Manifest(_) => TypedKind::Manifest,
            TypedValue::ColumnIndex(_) => TypedKind::ColumnIndex,
            TypedValue::IndexSegment(_) => TypedKind::IndexSegment,
        }
    }
}

pub type PutItem<'a> = (Cow<'a, [u8]>, Cow<'a, [u8]>);
pub type PutItems<'a> = Vec<PutItem<'a>>;

// TODO: Document
pub struct Put<'a> {
    pub field_id: LogicalFieldId,
    pub items: PutItems<'a>,
}

// TODO: Document
#[derive(Clone, Copy, Debug)]
pub enum ValueMode {
    Auto,
    ForceFixed(ByteWidth),
    ForceVariable,
}

// TODO: Document
#[derive(Clone, Debug)]
pub struct AppendOptions {
    pub mode: ValueMode,
    pub segment_max_entries: usize,
    pub segment_max_bytes: usize, // data payload budget per segment
    pub last_write_wins_in_batch: bool,
}

impl Default for AppendOptions {
    fn default() -> Self {
        Self {
            mode: ValueMode::Auto,
            segment_max_entries: u16::MAX as usize,
            segment_max_bytes: 8 * 1024 * 1024,
            last_write_wins_in_batch: true,
        }
    }
}
