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
/// - 32-bit keeps on-disk headers compact while allowing plenty of columns.
pub type LogicalFieldId = u32;

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

/// Typed kinds the pager knows how to decode/encode. Keep this set
/// scoped to storage types you persist via `bitcode`.
#[derive(Clone, Copy, Debug)]
pub enum TypedKind {
    Bootstrap,
    Manifest,
    ColumnIndex,
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

impl TypedValue {
    /// Helper to get the kind tag for a value.
    pub fn kind(&self) -> TypedKind {
        match self {
            TypedValue::Bootstrap(_) => TypedKind::Bootstrap,
            TypedValue::Manifest(_) => TypedKind::Manifest,
            TypedValue::ColumnIndex(_) => TypedKind::ColumnIndex,
            TypedValue::IndexSegment(_) => TypedKind::IndexSegment,
        }
    }
}
