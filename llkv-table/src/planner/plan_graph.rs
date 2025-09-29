//! Planner graph intermediate representation (IR).
//!
//! This module defines a directed acyclic graph (DAG) description that the
//! query planner can emit after it has applied logical and physical
//! optimizations. The IR is designed to serve multiple audiences:
//!
//! * **Executors** can materialize runtime operators from the in-memory
//!   [`PlanGraph`] structure.
//! * **Tests and tooling** can capture deterministic [`serde_json`] or DOT
//!   artifacts produced via [`PlanGraph::to_json`] and [`PlanGraph::to_dot`].
//! * **Developers** gain visibility into planner decisions (chosen indices,
//!   predicates, costing) that can be rendered with Graphviz or other
//!   visualizers.
//!
//! The representation concentrates on a few key goals:
//!
//! * Deterministic ordering for reproducible snapshot tests.
//! * Explicit metadata for schema, expressions, and physical properties.
//! * Validation helpers that guarantee the structure is a proper DAG with
//!   internally consistent edges.
//!
//! The `PlanGraph` intentionally avoids referencing heavy Arrow or execution
//! types directly; instead, it records human-readable summaries (for example
//! `DataType` names) so that serialization stays lightweight.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Semantic version identifier for the plan graph payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PlanGraphVersion {
    pub major: u16,
    pub minor: u16,
}

impl PlanGraphVersion {
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }
}

impl fmt::Display for PlanGraphVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{:02}", self.major, self.minor)
    }
}

/// Current version of the planner IR.
pub const PLAN_GRAPH_VERSION: PlanGraphVersion = PlanGraphVersion::new(0, 1);

/// Unique identifier for a planner node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PlanNodeId(pub u32);

impl PlanNodeId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Display for PlanNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// Planner node operator kind.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanOperator {
    TableScan,
    IndexScan,
    Filter,
    Project,
    Aggregate,
    Sort,
    Limit,
    TopK,
    HashJoin,
    NestedLoopJoin,
    MergeJoin,
    Window,
    Union,
    Intersect,
    Difference,
    Values,
    Materialize,
    Output,
    Explain,
}

impl fmt::Display for PlanOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Column description carried with each node's output schema.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanField {
    pub name: String,
    pub data_type: String,
    #[serde(default)]
    pub nullable: bool,
}

impl PlanField {
    pub fn new<N: Into<String>, T: Into<String>>(name: N, data_type: T) -> Self {
        Self {
            name: name.into(),
            data_type: data_type.into(),
            nullable: true,
        }
    }

    pub fn with_nullability(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }
}

/// Expression or projection annotations associated with a node.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanExpression {
    pub display: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot: Option<String>,
}

impl PlanExpression {
    pub fn new<S: Into<String>>(display: S) -> Self {
        Self {
            display: display.into(),
            fingerprint: None,
            slot: None,
        }
    }

    pub fn with_fingerprint<S: Into<String>>(mut self, fingerprint: S) -> Self {
        self.fingerprint = Some(fingerprint.into());
        self
    }

    pub fn with_slot<S: Into<String>>(mut self, slot: S) -> Self {
        self.slot = Some(slot.into());
        self
    }
}

/// Free-form metadata map for nodes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanNodeMetadata {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub properties: BTreeMap<String, String>,
}

impl PlanNodeMetadata {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.properties.is_empty()
    }

    pub fn insert<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.properties.insert(key.into(), value.into());
    }
}

/// Metadata associated with plan edges or inputs.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanEdgeMetadata {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub properties: BTreeMap<String, String>,
}

impl PlanEdgeMetadata {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.properties.is_empty()
    }

    pub fn insert<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.properties.insert(key.into(), value.into());
    }
}

/// Incoming connection for a node.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanInput {
    pub source: PlanNodeId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "PlanEdgeMetadata::is_empty")]
    pub metadata: PlanEdgeMetadata,
}

impl PlanInput {
    pub fn new(source: PlanNodeId) -> Self {
        Self {
            source,
            label: None,
            metadata: PlanEdgeMetadata::default(),
        }
    }

    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_metadata(mut self, metadata: PlanEdgeMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Edge between two planner nodes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanEdge {
    pub from: PlanNodeId,
    pub to: PlanNodeId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "PlanEdgeMetadata::is_empty")]
    pub metadata: PlanEdgeMetadata,
}

impl PlanEdge {
    pub fn new(from: PlanNodeId, to: PlanNodeId) -> Self {
        Self {
            from,
            to,
            label: None,
            metadata: PlanEdgeMetadata::default(),
        }
    }

    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_metadata(mut self, metadata: PlanEdgeMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Aggregate annotations that apply to the whole plan.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanAnnotations {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logical_plan_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub properties: BTreeMap<String, String>,
}

impl PlanAnnotations {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.logical_plan_fingerprint.is_none()
            && self.description.is_none()
            && self.properties.is_empty()
    }
}

/// Planner node with annotations used to describe how operators compose.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanNode {
    pub id: PlanNodeId,
    pub kind: PlanOperator,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<PlanInput>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<PlanNodeId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub schema: Vec<PlanField>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub predicates: Vec<PlanExpression>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub projections: Vec<PlanExpression>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cardinality: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_index: Option<String>,
    #[serde(default, skip_serializing_if = "PlanNodeMetadata::is_empty")]
    pub metadata: PlanNodeMetadata,
}

impl PlanNode {
    pub fn new(id: PlanNodeId, kind: PlanOperator) -> Self {
        Self {
            id,
            kind,
            inputs: Vec::new(),
            outputs: Vec::new(),
            schema: Vec::new(),
            predicates: Vec::new(),
            projections: Vec::new(),
            cost: None,
            cardinality: None,
            chosen_index: None,
            metadata: PlanNodeMetadata::default(),
        }
    }

    pub fn add_field(&mut self, field: PlanField) {
        self.schema.push(field);
    }

    pub fn add_predicate(&mut self, expr: PlanExpression) {
        self.predicates.push(expr);
    }

    pub fn add_projection(&mut self, expr: PlanExpression) {
        self.projections.push(expr);
    }
}

/// Errors raised while constructing or validating a plan graph.
#[derive(Error, Debug)]
pub enum PlanGraphError {
    #[error("duplicate node id {0}")]
    DuplicateNode(PlanNodeId),
    #[error("edge references missing node {0}")]
    MissingNode(PlanNodeId),
    #[error("edge creates self-loop on node {0}")]
    SelfLoop(PlanNodeId),
    #[error("duplicate edge {from:?} -> {to:?}")]
    DuplicateEdge { from: PlanNodeId, to: PlanNodeId },
    #[error("node {0} already contains wired edges; use the builder to manage connections")]
    NodeAlreadyConnected(PlanNodeId),
    #[error("root node {0} not present in graph")]
    UnknownRoot(PlanNodeId),
    #[error("root node {0} receives inputs")]
    RootHasInputs(PlanNodeId),
    #[error("node {node} inputs are inconsistent with edge set")]
    InputsDoNotMatch { node: PlanNodeId },
    #[error("node {node} outputs are inconsistent with edge set")]
    OutputsDoNotMatch { node: PlanNodeId },
    #[error("cycle detected in plan graph")]
    CycleDetected,
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type PlanGraphResult<T> = Result<T, PlanGraphError>;

/// Immutable DAG describing the planner output.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanGraph {
    pub version: PlanGraphVersion,
    pub nodes: Vec<PlanNode>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edges: Vec<PlanEdge>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub root_nodes: Vec<PlanNodeId>,
    #[serde(default, skip_serializing_if = "PlanAnnotations::is_empty")]
    pub annotations: PlanAnnotations,
}

impl PlanGraph {
    pub fn builder() -> PlanGraphBuilder {
        PlanGraphBuilder::default()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn version(&self) -> PlanGraphVersion {
        self.version
    }

    pub fn root_nodes(&self) -> &[PlanNodeId] {
        &self.root_nodes
    }

    pub fn validate(&self) -> PlanGraphResult<()> {
        let mut index_by_id: BTreeMap<PlanNodeId, usize> = BTreeMap::new();
        for (idx, node) in self.nodes.iter().enumerate() {
            if index_by_id.insert(node.id, idx).is_some() {
                return Err(PlanGraphError::DuplicateNode(node.id));
            }
        }

        let mut indegree = vec![0usize; self.nodes.len()];
        let mut outgoing: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        let mut seen_edges: BTreeSet<(PlanNodeId, PlanNodeId, Option<String>)> = BTreeSet::new();

        for edge in &self.edges {
            if edge.from == edge.to {
                return Err(PlanGraphError::SelfLoop(edge.from));
            }
            let from_idx = index_by_id
                .get(&edge.from)
                .copied()
                .ok_or(PlanGraphError::MissingNode(edge.from))?;
            let to_idx = index_by_id
                .get(&edge.to)
                .copied()
                .ok_or(PlanGraphError::MissingNode(edge.to))?;

            let key = (edge.from, edge.to, edge.label.clone());
            if !seen_edges.insert(key) {
                return Err(PlanGraphError::DuplicateEdge {
                    from: edge.from,
                    to: edge.to,
                });
            }

            indegree[to_idx] += 1;
            outgoing[from_idx].push(to_idx);
        }

        // Validate inputs and outputs line up with edges.
        let mut expected_inputs: BTreeMap<
            PlanNodeId,
            Vec<(PlanNodeId, Option<String>, Vec<(String, String)>)>,
        > = BTreeMap::new();
        let mut expected_outputs: BTreeMap<PlanNodeId, BTreeSet<PlanNodeId>> = BTreeMap::new();

        for edge in &self.edges {
            let metadata_vec = metadata_as_vec(&edge.metadata);
            expected_inputs.entry(edge.to).or_default().push((
                edge.from,
                edge.label.clone(),
                metadata_vec.clone(),
            ));
            expected_outputs
                .entry(edge.from)
                .or_default()
                .insert(edge.to);
        }

        for node in &self.nodes {
            let mut actual_inputs: Vec<(PlanNodeId, Option<String>, Vec<(String, String)>)> = node
                .inputs
                .iter()
                .map(|input| {
                    (
                        input.source,
                        input.label.clone(),
                        metadata_as_vec(&input.metadata),
                    )
                })
                .collect();
            let mut expected = expected_inputs.remove(&node.id).unwrap_or_default();
            actual_inputs.sort();
            expected.sort();
            if actual_inputs != expected {
                return Err(PlanGraphError::InputsDoNotMatch { node: node.id });
            }

            let mut actual_outputs: BTreeSet<PlanNodeId> = node.outputs.iter().copied().collect();
            let expected_outputs = expected_outputs.remove(&node.id).unwrap_or_default();
            if actual_outputs != expected_outputs {
                return Err(PlanGraphError::OutputsDoNotMatch { node: node.id });
            }

            actual_outputs.clear();
        }

        // Remaining expected inputs implies an edge pointed to a missing node entry.
        if let Some((node, _)) = expected_inputs.iter().next() {
            return Err(PlanGraphError::MissingNode(*node));
        }

        // Validate roots.
        let mut root_set: BTreeSet<PlanNodeId> = BTreeSet::new();
        for root in &self.root_nodes {
            if !root_set.insert(*root) {
                return Err(PlanGraphError::DuplicateNode(*root));
            }
            if !index_by_id.contains_key(root) {
                return Err(PlanGraphError::UnknownRoot(*root));
            }
        }
        for root in &self.root_nodes {
            let idx = index_by_id[root];
            if indegree[idx] > 0 {
                return Err(PlanGraphError::RootHasInputs(*root));
            }
        }

        // Cycle detection via Kahn's algorithm.
        let mut queue: VecDeque<usize> = indegree
            .iter()
            .enumerate()
            .filter_map(|(idx, &deg)| if deg == 0 { Some(idx) } else { None })
            .collect();
        let mut visited = 0usize;
        let mut indegree_mut = indegree.clone();
        while let Some(node_idx) = queue.pop_front() {
            visited += 1;
            for &child in &outgoing[node_idx] {
                indegree_mut[child] -= 1;
                if indegree_mut[child] == 0 {
                    queue.push_back(child);
                }
            }
        }

        if visited != self.nodes.len() {
            return Err(PlanGraphError::CycleDetected);
        }

        Ok(())
    }

    pub fn to_dot(&self) -> PlanGraphResult<String> {
        self.validate()?;

        let mut dot = String::new();
        dot.push_str("digraph PlanGraph {\n");
        dot.push_str("  graph [rankdir=LR];\n");
        dot.push_str("  node [shape=box, fontname=\"Helvetica\"];\n");

        let root_set: BTreeSet<PlanNodeId> = self.root_nodes.iter().copied().collect();
        for node in &self.nodes {
            let shape_attr = if root_set.contains(&node.id) {
                "shape=doublecircle, style=bold"
            } else {
                "shape=box"
            };
            let label = escape_dot_label(&build_node_label(node));
            dot.push_str(&format!(
                "  \"{id}\" [{shape} label=\"{label}\"];\n",
                id = node.id,
                shape = shape_attr,
                label = label
            ));
        }

        for edge in &self.edges {
            let label = edge
                .label
                .as_ref()
                .map(|label| escape_dot_label(label))
                .or_else(|| {
                    if edge.metadata.is_empty() {
                        None
                    } else {
                        Some(escape_dot_label(&format_edge_metadata(&edge.metadata)))
                    }
                });
            match label {
                Some(text) => {
                    dot.push_str(&format!(
                        "  \"{from}\" -> \"{to}\" [label=\"{label}\"];\n",
                        from = edge.from,
                        to = edge.to,
                        label = text
                    ));
                }
                None => {
                    dot.push_str(&format!(
                        "  \"{from}\" -> \"{to}\";\n",
                        from = edge.from,
                        to = edge.to
                    ));
                }
            }
        }

        dot.push('}');
        Ok(dot)
    }

    pub fn to_json(&self) -> PlanGraphResult<String> {
        self.validate()?;
        let json = serde_json::to_string_pretty(self)?;
        Ok(json)
    }

    pub fn from_json(json: &str) -> PlanGraphResult<Self> {
        let mut graph: PlanGraph = serde_json::from_str(json)?;
        graph.normalize();
        graph.validate()?;
        Ok(graph)
    }

    pub fn topological_order(&self) -> PlanGraphResult<Vec<PlanNodeId>> {
        self.validate()?;

        let mut index_by_id: BTreeMap<PlanNodeId, usize> = BTreeMap::new();
        for (idx, node) in self.nodes.iter().enumerate() {
            index_by_id.insert(node.id, idx);
        }

        let mut indegree = vec![0usize; self.nodes.len()];
        let mut outgoing: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        for edge in &self.edges {
            let from_idx = index_by_id[&edge.from];
            let to_idx = index_by_id[&edge.to];
            indegree[to_idx] += 1;
            outgoing[from_idx].push(to_idx);
        }

        let mut queue: VecDeque<usize> = indegree
            .iter()
            .enumerate()
            .filter_map(|(idx, &deg)| if deg == 0 { Some(idx) } else { None })
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut indegree_mut = indegree;
        while let Some(idx) = queue.pop_front() {
            order.push(self.nodes[idx].id);
            for &child in &outgoing[idx] {
                indegree_mut[child] -= 1;
                if indegree_mut[child] == 0 {
                    queue.push_back(child);
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(PlanGraphError::CycleDetected);
        }

        Ok(order)
    }

    fn normalize(&mut self) {
        self.nodes.sort_by(|a, b| a.id.cmp(&b.id));
        for node in &mut self.nodes {
            node.inputs.sort_by(|left, right| {
                left.source
                    .cmp(&right.source)
                    .then(left.label.cmp(&right.label))
                    .then_with(|| {
                        metadata_as_vec(&left.metadata).cmp(&metadata_as_vec(&right.metadata))
                    })
            });
            node.outputs.sort();
        }
        self.edges.sort_by(|a, b| {
            a.from
                .cmp(&b.from)
                .then(a.to.cmp(&b.to))
                .then(a.label.cmp(&b.label))
        });
        self.root_nodes.sort();
    }
}

/// Builder for `PlanGraph` that enforces DAG invariants while allowing
/// incremental construction.
pub struct PlanGraphBuilder {
    version: PlanGraphVersion,
    nodes: BTreeMap<PlanNodeId, PlanNode>,
    edges: BTreeMap<(PlanNodeId, PlanNodeId, Option<String>), PlanEdge>,
    roots: BTreeSet<PlanNodeId>,
    annotations: PlanAnnotations,
}

impl PlanGraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_version(version: PlanGraphVersion) -> Self {
        Self {
            version,
            ..Self::default()
        }
    }

    pub fn add_node(&mut self, node: PlanNode) -> PlanGraphResult<()> {
        let node_id = node.id;
        if !node.inputs.is_empty() || !node.outputs.is_empty() {
            return Err(PlanGraphError::NodeAlreadyConnected(node_id));
        }
        if self.nodes.insert(node_id, node).is_some() {
            return Err(PlanGraphError::DuplicateNode(node_id));
        }
        Ok(())
    }

    pub fn add_edge(&mut self, edge: PlanEdge) -> PlanGraphResult<()> {
        if edge.from == edge.to {
            return Err(PlanGraphError::SelfLoop(edge.from));
        }
        let key = (edge.from, edge.to, edge.label.clone());
        if self.edges.contains_key(&key) {
            return Err(PlanGraphError::DuplicateEdge {
                from: edge.from,
                to: edge.to,
            });
        }

        let target = self
            .nodes
            .get_mut(&edge.to)
            .ok_or(PlanGraphError::MissingNode(edge.to))?;
        let mut input = PlanInput::new(edge.from);
        if let Some(label) = &edge.label {
            input.label = Some(label.clone());
        }
        if !edge.metadata.is_empty() {
            input.metadata = edge.metadata.clone();
        }
        target.inputs.push(input);

        let source = self
            .nodes
            .get_mut(&edge.from)
            .ok_or(PlanGraphError::MissingNode(edge.from))?;
        if !source.outputs.contains(&edge.to) {
            source.outputs.push(edge.to);
        }

        self.edges.insert(key, edge);
        Ok(())
    }

    pub fn add_root(&mut self, node_id: PlanNodeId) -> PlanGraphResult<()> {
        if !self.nodes.contains_key(&node_id) {
            return Err(PlanGraphError::UnknownRoot(node_id));
        }
        self.roots.insert(node_id);
        Ok(())
    }

    pub fn annotations_mut(&mut self) -> &mut PlanAnnotations {
        &mut self.annotations
    }

    pub fn finish(self) -> PlanGraphResult<PlanGraph> {
        let nodes: Vec<PlanNode> = self.nodes.into_values().collect();
        let edges: Vec<PlanEdge> = self.edges.into_values().collect();
        let root_nodes: Vec<PlanNodeId> = self.roots.into_iter().collect();

        let mut graph = PlanGraph {
            version: self.version,
            nodes,
            edges,
            root_nodes,
            annotations: self.annotations,
        };
        graph.normalize();
        graph.validate()?;
        Ok(graph)
    }
}

impl Default for PlanGraphBuilder {
    fn default() -> Self {
        Self {
            version: PLAN_GRAPH_VERSION,
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            roots: BTreeSet::new(),
            annotations: PlanAnnotations::default(),
        }
    }
}

fn metadata_as_vec(metadata: &PlanEdgeMetadata) -> Vec<(String, String)> {
    metadata
        .properties
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

fn build_node_label(node: &PlanNode) -> String {
    let mut lines = Vec::new();
    lines.push(node.kind.to_string());
    if !node.schema.is_empty() {
        let fields: Vec<String> = node
            .schema
            .iter()
            .map(|field| {
                if field.nullable {
                    format!("{}:{}?", field.name, field.data_type)
                } else {
                    format!("{}:{}", field.name, field.data_type)
                }
            })
            .collect();
        lines.push(format!("schema: {}", fields.join(", ")));
    }
    if let Some(card) = node.cardinality {
        lines.push(format!("card: {}", card));
    }
    if let Some(cost) = node.cost {
        lines.push(format!("cost: {:.4}", cost));
    }
    if let Some(idx) = &node.chosen_index {
        lines.push(format!("index: {idx}"));
    }
    for expr in &node.predicates {
        lines.push(format!("pred: {}", expr.display));
    }
    for expr in &node.projections {
        lines.push(format!("proj: {}", expr.display));
    }
    for (key, value) in &node.metadata.properties {
        lines.push(format!("{key}: {value}"));
    }
    lines.join("\n")
}

fn format_edge_metadata(metadata: &PlanEdgeMetadata) -> String {
    metadata
        .properties
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn escape_dot_label(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
