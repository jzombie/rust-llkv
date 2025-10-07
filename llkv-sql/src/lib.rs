pub type SqlResult<T> = llkv_result::Result<T>;

mod engine;
mod value;

pub use engine::{SqlEngine, SqlStatementResult};
