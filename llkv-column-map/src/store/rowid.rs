//! Helpers for working with row-id shadow columns.

use crate::types::{LogicalFieldId, Namespace};

/// Sets the shadow row_id tag on a LogicalFieldId using the Namespace enum.
pub fn rowid_fid(fid: LogicalFieldId) -> LogicalFieldId {
    fid.with_namespace(Namespace::RowIdShadow)
}
