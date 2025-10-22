//! Transaction result types.

use super::TransactionKind;
use arrow::datatypes::Schema;
use llkv_executor::SelectExecution;
use llkv_result::{Error as LlkvError, Result as LlkvResult};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

/// Transaction result enum representing the outcome of a transaction operation.
///
/// This enum encapsulates the results of various database operations executed
/// within a transaction context, including DDL and DML statements.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum TransactionResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// CREATE TABLE result
    CreateTable {
        table_name: String,
    },
    /// INSERT result with row count
    Insert {
        rows_inserted: usize,
    },
    /// UPDATE result with match and update counts
    Update {
        rows_matched: usize,
        rows_updated: usize,
    },
    /// DELETE result with row count
    Delete {
        rows_deleted: usize,
    },
    /// CREATE INDEX result
    CreateIndex {
        table_name: String,
        index_name: Option<String>,
    },
    /// SELECT result with streaming execution handle
    Select {
        table_name: String,
        schema: Arc<Schema>,
        execution: SelectExecution<P>,
    },
    /// Transaction control statement result
    Transaction {
        kind: TransactionKind,
    },
    /// No-op result (e.g., for DDL statements that don't produce output)
    NoOp,
}

impl<P> TransactionResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Convert pager type for compatibility across different storage backends.
    pub fn convert_pager_type<P2>(self) -> LlkvResult<TransactionResult<P2>>
    where
        P2: Pager<Blob = EntryHandle> + Send + Sync + 'static,
    {
        match self {
            TransactionResult::CreateTable { table_name } => {
                Ok(TransactionResult::CreateTable { table_name })
            }
            TransactionResult::Insert { rows_inserted } => {
                Ok(TransactionResult::Insert { rows_inserted })
            }
            TransactionResult::Update {
                rows_matched,
                rows_updated,
            } => Ok(TransactionResult::Update {
                rows_matched,
                rows_updated,
            }),
            TransactionResult::Delete { rows_deleted } => {
                Ok(TransactionResult::Delete { rows_deleted })
            }
            TransactionResult::CreateIndex {
                table_name,
                index_name,
            } => Ok(TransactionResult::CreateIndex {
                table_name,
                index_name,
            }),
            TransactionResult::Select { .. } => Err(LlkvError::InvalidArgumentError(
                "cannot convert SELECT result with pager type change".into(),
            )),
            TransactionResult::Transaction { kind } => {
                Ok(TransactionResult::Transaction { kind })
            }
            TransactionResult::NoOp => Ok(TransactionResult::NoOp),
        }
    }
}
