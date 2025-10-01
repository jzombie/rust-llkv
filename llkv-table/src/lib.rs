#![forbid(unsafe_code)]

pub mod constants;
pub mod join;
mod planner;
mod scalar_eval;
mod sys_catalog;
pub mod expr {
    pub use llkv_expr::expr::*;
}

pub mod table;
pub mod types;

pub use sys_catalog::{ColMeta, TableMeta};
pub use table::Table;
pub use types::{FieldId, RowId};

pub use planner::plan_graph::{
    PLAN_GRAPH_VERSION, PlanAnnotations, PlanEdge, PlanEdgeMetadata, PlanExpression, PlanField,
    PlanGraph, PlanGraphBuilder, PlanGraphError, PlanGraphResult, PlanGraphVersion, PlanInput,
    PlanNode, PlanNodeId, PlanOperator,
};
