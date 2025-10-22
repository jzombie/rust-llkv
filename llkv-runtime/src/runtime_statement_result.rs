use std::fmt;
use std::sync::Arc;

use arrow::datatypes::Schema;
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use crate::{SelectExecution, TransactionKind};

#[derive(Clone)]
pub enum RuntimeStatementResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    CreateTable {
        table_name: String,
    },
    CreateIndex {
        table_name: String,
        index_name: Option<String>,
    },
    NoOp,
    Insert {
        table_name: String,
        rows_inserted: usize,
    },
    Update {
        table_name: String,
        rows_updated: usize,
    },
    Delete {
        table_name: String,
        rows_deleted: usize,
    },
    Select {
        table_name: String,
        schema: Arc<Schema>,
        execution: Box<SelectExecution<P>>,
    },
    Transaction {
        kind: TransactionKind,
    },
}

impl<P> fmt::Debug for RuntimeStatementResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeStatementResult::CreateTable { table_name } => f
                .debug_struct("CreateTable")
                .field("table_name", table_name)
                .finish(),
            RuntimeStatementResult::CreateIndex {
                table_name,
                index_name,
            } => f
                .debug_struct("CreateIndex")
                .field("table_name", table_name)
                .field("index_name", index_name)
                .finish(),
            RuntimeStatementResult::NoOp => f.debug_struct("NoOp").finish(),
            RuntimeStatementResult::Insert {
                table_name,
                rows_inserted,
            } => f
                .debug_struct("Insert")
                .field("table_name", table_name)
                .field("rows_inserted", rows_inserted)
                .finish(),
            RuntimeStatementResult::Update {
                table_name,
                rows_updated,
            } => f
                .debug_struct("Update")
                .field("table_name", table_name)
                .field("rows_updated", rows_updated)
                .finish(),
            RuntimeStatementResult::Delete {
                table_name,
                rows_deleted,
            } => f
                .debug_struct("Delete")
                .field("table_name", table_name)
                .field("rows_deleted", rows_deleted)
                .finish(),
            RuntimeStatementResult::Select {
                table_name, schema, ..
            } => f
                .debug_struct("Select")
                .field("table_name", table_name)
                .field("schema", schema)
                .finish(),
            RuntimeStatementResult::Transaction { kind } => {
                f.debug_struct("Transaction").field("kind", kind).finish()
            }
        }
    }
}

impl<P> RuntimeStatementResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Convert a StatementResult from one pager type to another.
    /// Only works for non-SELECT results (CreateTable, Insert, Update, Delete, NoOp, Transaction).
    #[allow(dead_code)]
    pub(crate) fn convert_pager_type<Q>(self) -> Result<RuntimeStatementResult<Q>>
    where
        Q: Pager<Blob = EntryHandle> + Send + Sync,
    {
        match self {
            RuntimeStatementResult::CreateTable { table_name } => {
                Ok(RuntimeStatementResult::CreateTable { table_name })
            }
            RuntimeStatementResult::CreateIndex {
                table_name,
                index_name,
            } => Ok(RuntimeStatementResult::CreateIndex {
                table_name,
                index_name,
            }),
            RuntimeStatementResult::NoOp => Ok(RuntimeStatementResult::NoOp),
            RuntimeStatementResult::Insert {
                table_name,
                rows_inserted,
            } => Ok(RuntimeStatementResult::Insert {
                table_name,
                rows_inserted,
            }),
            RuntimeStatementResult::Update {
                table_name,
                rows_updated,
            } => Ok(RuntimeStatementResult::Update {
                table_name,
                rows_updated,
            }),
            RuntimeStatementResult::Delete {
                table_name,
                rows_deleted,
            } => Ok(RuntimeStatementResult::Delete {
                table_name,
                rows_deleted,
            }),
            RuntimeStatementResult::Transaction { kind } => {
                Ok(RuntimeStatementResult::Transaction { kind })
            }
            RuntimeStatementResult::Select { .. } => Err(Error::Internal(
                "Cannot convert SELECT result between pager types in transaction".into(),
            )),
        }
    }
}
