pub type SqlResult<T> = llkv_result::Result<T>;

mod engine;
mod value;

// TODO: Rename to `sql_engine`
pub use engine::{SqlEngine, SqlStatementResult, TransactionKind};
