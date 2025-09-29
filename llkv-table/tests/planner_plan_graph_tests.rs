use llkv_table::{
    PLAN_GRAPH_VERSION, PlanEdge, PlanField, PlanGraphBuilder, PlanNode, PlanNodeId, PlanOperator,
};

#[test]
fn plan_graph_reexports_are_available() {
    let mut builder = PlanGraphBuilder::new();
    let mut node = PlanNode::new(PlanNodeId::new(1), PlanOperator::TableScan);
    node.add_field(PlanField::new("order_id", "UInt64"));
    builder.add_node(node).unwrap();
    builder.add_root(PlanNodeId::new(1)).unwrap();

    let projection = PlanNode::new(PlanNodeId::new(2), PlanOperator::Project);
    builder.add_node(projection).unwrap();
    builder
        .add_edge(PlanEdge::new(PlanNodeId::new(1), PlanNodeId::new(2)))
        .unwrap();

    let graph = builder.finish().expect("finish graph");
    assert_eq!(graph.version(), PLAN_GRAPH_VERSION);
    assert_eq!(graph.root_nodes(), &[PlanNodeId::new(1)]);
}
