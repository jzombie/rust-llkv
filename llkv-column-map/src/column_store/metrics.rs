/// Minimal pager-hit metrics collected inside ColumnStore. This counts how
/// many *requests* we send to the pager (not bytes). If you need byte totals,
/// you can extend this to compute sizes for Raw puts/gets and (optionally)
/// encoded sizes for Typed values.
#[derive(Clone, Debug, Default)]
pub struct IoStats {
    pub batches: usize,       // number of times we called Pager::batch_* (get/put)
    pub get_raw_ops: usize,   // number of Raw gets requested
    pub get_typed_ops: usize, // number of Typed gets requested
    pub put_raw_ops: usize,   // number of Raw puts requested
    pub put_typed_ops: usize, // number of Typed puts requested
    pub free_ops: usize,      // number of physical keys freed
}
