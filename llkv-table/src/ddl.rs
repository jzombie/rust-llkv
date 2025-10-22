//! Shared catalog DDL trait for contexts that can execute plan-based schema operations.
//!
//! This trait lives in `llkv-table` so that the runtime, transaction, and session layers
//! can agree on a single contract without introducing circular dependencies. It captures
//! the common plan-driven schema mutations (create/drop table, rename table, create/drop
//! index) while letting each layer choose its own return types via associated outputs.

use llkv_plan::{
    AlterTablePlan, CreateIndexPlan, CreateTablePlan, DropIndexPlan, DropTablePlan,
    RenameTablePlan,
};
use llkv_result::Result;

/// Common DDL operations that operate on catalog-backed tables and indexes.
///
/// Implementors provide plan-based entry points for schema mutations so higher layers
/// (runtime session, transaction staging, etc.) can share the same method signatures.
/// Associated output types keep the contract generic while still conveying the concrete
/// results used by each layer (e.g., `RuntimeStatementResult`, `TransactionResult`, or
/// unit for operations that only signal success).
pub trait CatalogDdl {
    /// Output of executing a CREATE TABLE plan.
    type CreateTableOutput;
    /// Output of executing a DROP TABLE plan.
    type DropTableOutput;
    /// Output of executing a table rename.
    type RenameTableOutput;
    /// Output of executing an ALTER TABLE plan.
    type AlterTableOutput;
    /// Output of executing a CREATE INDEX plan.
    type CreateIndexOutput;
    /// Output of executing a DROP INDEX plan.
    type DropIndexOutput;

    /// Creates a table described by the given plan.
    fn create_table(&self, plan: CreateTablePlan) -> Result<Self::CreateTableOutput>;

    /// Drops a table identified by the given plan.
    fn drop_table(&self, plan: DropTablePlan) -> Result<Self::DropTableOutput>;

    /// Renames a table using the provided plan.
    fn rename_table(&self, plan: RenameTablePlan) -> Result<Self::RenameTableOutput>;

    /// Alters a table using the provided plan.
    fn alter_table(&self, plan: AlterTablePlan) -> Result<Self::AlterTableOutput>;

    /// Creates an index described by the given plan.
    fn create_index(&self, plan: CreateIndexPlan) -> Result<Self::CreateIndexOutput>;

    /// Drops a single-column index described by the given plan, returning metadata about the
    /// index when it existed.
    fn drop_index(&self, plan: DropIndexPlan) -> Result<Self::DropIndexOutput>;
}
