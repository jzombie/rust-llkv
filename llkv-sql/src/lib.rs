pub type SqlResult<T> = llkv_result::Result<T>;

mod sql_engine;
mod value;

pub use llkv_dsl::{SelectExecution, TransactionKind};
pub use sql_engine::SqlEngine;

pub type SqlStatementResult<P> = llkv_dsl::StatementResult<P>;
