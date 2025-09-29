//! Compatibility shim that re-exports the planner IR from the standalone
//! `llkv-plan` crate. New development should depend on `llkv-plan` directly.

pub use llkv_plan::{
    PLAN_GRAPH_VERSION, PlanAnnotations, PlanEdge, PlanEdgeMetadata, PlanExpression, PlanField,
    PlanGraph, PlanGraphBuilder, PlanGraphError, PlanGraphResult, PlanGraphVersion, PlanInput,
    PlanNode, PlanNodeId, PlanOperator,
};
