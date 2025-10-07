pub type SqlResult<T> = llkv_result::Result<T>;

mod sql_engine;
mod value;

pub use sql_engine::{SqlEngine, SqlStatementResult, TransactionKind};
