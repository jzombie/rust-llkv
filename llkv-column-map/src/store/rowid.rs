use crate::types::{LogicalFieldId, Namespace};

// TODO: Dedupe in tests and benches
/// Sets the shadow row_id tag on a LogicalFieldId using the Namespace enum.
pub fn rowid_fid(fid: LogicalFieldId) -> LogicalFieldId {
    fid.with_namespace(Namespace::RowIdShadow)
}
