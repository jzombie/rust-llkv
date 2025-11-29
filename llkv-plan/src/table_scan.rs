use arrow::datatypes::DataType;
use std::ops::Bound;

use llkv_expr::{Expr, Filter, Operator, ScalarExpr};
use llkv_types::{FieldId, LogicalFieldId, TableId};

use crate::{
    PlanEdge, PlanExpression, PlanField, PlanGraph, PlanGraphBuilder, PlanGraphError, PlanNode,
    PlanNodeId, PlanOperator,
};

/// Projection descriptor for building table-scan plan graphs without depending
/// on storage-specific types.
pub enum TableScanProjectionSpec {
    Column {
        logical_field_id: LogicalFieldId,
        data_type: DataType,
        alias: Option<String>,
    },
    Computed {
        expr: ScalarExpr<FieldId>,
        alias: String,
        data_type: DataType,
    },
}

/// Build a `PlanGraph` for a table scan with optional filter and projections.
///
/// This mirrors the graph previously assembled inside `llkv-table` so EXPLAIN
/// output remains stable while allowing the planner to live in `llkv-plan`.
pub fn build_table_scan_plan(
    table_id: TableId,
    projections: &[TableScanProjectionSpec],
    filter_expr: &Expr<'_, FieldId>,
    include_nulls: bool,
) -> Result<PlanGraph, PlanGraphError> {
    let mut builder = PlanGraphBuilder::new();

    let scan_node_id = PlanNodeId::new(1);
    let mut scan_node = PlanNode::new(scan_node_id, PlanOperator::TableScan);
    scan_node
        .metadata
        .insert("table_id", table_id.to_string());
    scan_node
        .metadata
        .insert("projection_count", projections.len().to_string());
    builder.add_node(scan_node)?;
    builder.add_root(scan_node_id)?;

    let mut next_node = 2u32;
    let mut parent = scan_node_id;

    if !is_trivial_filter(filter_expr) {
        let filter_node_id = PlanNodeId::new(next_node);
        next_node += 1;
        let mut filter_node = PlanNode::new(filter_node_id, PlanOperator::Filter);
        filter_node.add_predicate(PlanExpression::new(format_expr(filter_expr)));
        builder.add_node(filter_node)?;
        builder.add_edge(PlanEdge::new(parent, filter_node_id))?;
        parent = filter_node_id;
    }

    let project_node_id = PlanNodeId::new(next_node);
    next_node += 1;
    let mut project_node = PlanNode::new(project_node_id, PlanOperator::Project);

    for projection in projections {
        match projection {
            TableScanProjectionSpec::Column {
                logical_field_id,
                data_type,
                alias,
            } => {
                let fallback = logical_field_id.field_id().to_string();
                let name = alias.clone().unwrap_or(fallback);
                project_node.add_projection(PlanExpression::new(format!("column({name})")));
                project_node.add_field(
                    PlanField::new(name, format!("{data_type:?}")).with_nullability(true),
                );
            }
            TableScanProjectionSpec::Computed {
                expr,
                alias,
                data_type,
            } => {
                project_node.add_projection(PlanExpression::new(format!(
                    "{} := {}",
                    alias,
                    expr.format_display()
                )));
                project_node.add_field(
                    PlanField::new(alias.clone(), format!("{data_type:?}"))
                        .with_nullability(true),
                );
            }
        }
    }

    builder.add_node(project_node)?;
    builder.add_edge(PlanEdge::new(parent, project_node_id))?;
    parent = project_node_id;

    let output_node_id = PlanNodeId::new(next_node);
    let mut output_node = PlanNode::new(output_node_id, PlanOperator::Output);
    output_node
        .metadata
        .insert("include_nulls", include_nulls.to_string());
    builder.add_node(output_node)?;
    builder.add_edge(PlanEdge::new(parent, output_node_id))?;

    let annotations = builder.annotations_mut();
    annotations.description = Some("table.scan_stream".to_string());
    annotations
        .properties
        .insert("table_id".to_string(), table_id.to_string());

    builder.finish()
}

fn is_trivial_filter(filter_expr: &Expr<'_, FieldId>) -> bool {
    matches!(
        filter_expr,
        Expr::Pred(Filter {
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
            ..
        })
    ) || matches!(filter_expr, Expr::Literal(_))
}

fn format_expr(filter_expr: &Expr<'_, FieldId>) -> String {
    filter_expr.format_display()
}
