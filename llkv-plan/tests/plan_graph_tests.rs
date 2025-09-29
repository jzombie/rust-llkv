use llkv_plan::{
    PLAN_GRAPH_VERSION, PlanEdge, PlanExpression, PlanField, PlanGraph, PlanGraphBuilder,
    PlanGraphError, PlanNode, PlanNodeId, PlanOperator,
};

fn make_scan_node(id: u32) -> PlanNode {
    let mut node = PlanNode::new(PlanNodeId::new(id), PlanOperator::TableScan);
    node.cardinality = Some(1_024);
    node.metadata.insert("table", "orders");
    node.add_field(PlanField::new("order_id", "UInt64").with_nullability(false));
    node.add_field(PlanField::new("total", "Float64"));
    node
}

fn make_filter_node(id: u32) -> PlanNode {
    let mut node = PlanNode::new(PlanNodeId::new(id), PlanOperator::Filter);
    node.add_predicate(
        PlanExpression::new("total > 100.0").with_fingerprint("expr-1-total-gt-100"),
    );
    node
}

fn make_project_node(id: u32) -> PlanNode {
    let mut node = PlanNode::new(PlanNodeId::new(id), PlanOperator::Project);
    node.add_projection(PlanExpression::new("order_id"));
    node.add_projection(PlanExpression::new("total"));
    node
}

#[test]
fn plan_graph_round_trip_json_and_dot() {
    let mut builder = PlanGraph::builder();
    builder.add_node(make_scan_node(1)).unwrap();
    builder.add_node(make_filter_node(2)).unwrap();
    builder.add_node(make_project_node(3)).unwrap();

    builder
        .add_edge(PlanEdge::new(PlanNodeId::new(1), PlanNodeId::new(2)).with_label("scan"))
        .unwrap();
    builder
        .add_edge(PlanEdge::new(PlanNodeId::new(2), PlanNodeId::new(3)))
        .unwrap();
    builder.add_root(PlanNodeId::new(1)).unwrap();

    let graph = builder.finish().expect("finish graph");
    assert_eq!(graph.version(), PLAN_GRAPH_VERSION);
    assert_eq!(graph.root_nodes(), &[PlanNodeId::new(1)]);

    let dot = graph.to_dot().expect("dot");
    assert!(dot.contains("n1"));
    assert!(dot.contains("n2"));
    assert!(dot.contains("scan"));

    let json_first = graph.to_json().expect("json");
    let graph_from_json = PlanGraph::from_json(&json_first).expect("from json");
    assert_eq!(graph, graph_from_json);

    let json_second = graph_from_json.to_json().expect("json round trip");
    assert_eq!(json_first, json_second);
}

#[test]
fn plan_graph_detects_cycles() {
    let mut builder = PlanGraphBuilder::new();
    builder.add_node(make_scan_node(10)).unwrap();
    builder.add_node(make_filter_node(11)).unwrap();

    builder
        .add_edge(PlanEdge::new(PlanNodeId::new(10), PlanNodeId::new(11)))
        .unwrap();
    builder
        .add_edge(PlanEdge::new(PlanNodeId::new(11), PlanNodeId::new(10)))
        .unwrap();

    let err = builder.finish().expect_err("cycle should fail");
    assert!(matches!(err, PlanGraphError::CycleDetected));
}

#[test]
fn builder_rejects_prewired_nodes() {
    let mut node = make_scan_node(42);
    node.outputs.push(PlanNodeId::new(99));

    let mut builder = PlanGraphBuilder::new();
    let err = builder
        .add_node(node)
        .expect_err("should reject node with outputs");
    assert!(matches!(err, PlanGraphError::NodeAlreadyConnected(_)));
}
