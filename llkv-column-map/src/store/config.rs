/// Run-time configuration (no hidden constants).
#[derive(Debug, Clone)]
pub(crate) struct ColumnStoreConfig {
    /// Fallback rows-per-slice for *exotic* variable-width arrays that don't
    /// expose byte offsets we can use (e.g., List/Struct/Map). Used only as
    /// a last resort to avoid pathological slices.
    pub(crate) varwidth_fallback_rows_per_slice: usize,
}

impl Default for ColumnStoreConfig {
    fn default() -> Self {
        Self {
            // TODO: Don't hardcode
            varwidth_fallback_rows_per_slice: 4096,
        }
    }
}
