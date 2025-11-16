use std::collections::HashMap;
use crate::error::Result;

/// Node ID in the compute graph
pub type NodeId = usize;

/// Node type in the compute graph
#[derive(Debug, Clone)]
pub enum NodeType {
    Input,
    Output,
    Operator(String), // Operator name
    Constant,
}

/// A node in the compute graph
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub name: String,
    pub node_type: NodeType,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub attributes: HashMap<String, AttributeValue>,
}

/// Attribute value types
#[derive(Debug, Clone)]
pub enum AttributeValue {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
}

impl Node {
    pub fn new(id: NodeId, name: String, node_type: NodeType) -> Self {
        Self {
            id,
            name,
            node_type,
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    pub fn input(name: &str) -> Self {
        Self::new(0, name.to_string(), NodeType::Input)
    }

    pub fn output(name: &str) -> Self {
        Self::new(0, name.to_string(), NodeType::Output)
    }

    pub fn operator(name: &str, op_type: &str) -> Self {
        Self::new(0, name.to_string(), NodeType::Operator(op_type.to_string()))
    }

    pub fn constant(name: &str) -> Self {
        Self::new(0, name.to_string(), NodeType::Constant)
    }

    pub fn add_attribute(&mut self, key: String, value: AttributeValue) {
        self.attributes.insert(key, value);
    }
}

/// Compute graph representing a neural network
#[derive(Debug)]
pub struct ComputeGraph {
    nodes: Vec<Node>,
    node_map: HashMap<String, NodeId>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

impl ComputeGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_map: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, mut node: Node) -> NodeId {
        let id = self.nodes.len();
        node.id = id;

        self.node_map.insert(node.name.clone(), id);

        match &node.node_type {
            NodeType::Input => self.inputs.push(id),
            NodeType::Output => self.outputs.push(id),
            _ => {}
        }

        self.nodes.push(node);
        id
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(id)
    }

    /// Get a node by name
    pub fn get_node_by_name(&self, name: &str) -> Option<&Node> {
        self.node_map.get(name).and_then(|&id| self.get_node(id))
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) -> Result<()> {
        if let Some(from_node) = self.get_node_mut(from) {
            from_node.outputs.push(to);
        }

        if let Some(to_node) = self.get_node_mut(to) {
            to_node.inputs.push(from);
        }

        Ok(())
    }

    /// Get input nodes
    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    /// Get output nodes
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    /// Get all nodes
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Topological sort for execution order
    pub fn topological_sort(&self) -> Result<Vec<NodeId>> {
        let mut in_degree = vec![0; self.nodes.len()];
        let mut result = Vec::new();

        // Calculate in-degree for each node
        for node in &self.nodes {
            for &output_id in &node.outputs {
                in_degree[output_id] += 1;
            }
        }

        // Find nodes with in-degree 0
        let mut queue: Vec<NodeId> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, &degree)| degree == 0)
            .map(|(id, _)| id)
            .collect();

        while let Some(node_id) = queue.pop() {
            result.push(node_id);

            if let Some(node) = self.get_node(node_id) {
                for &output_id in &node.outputs {
                    in_degree[output_id] -= 1;
                    if in_degree[output_id] == 0 {
                        queue.push(output_id);
                    }
                }
            }
        }

        Ok(result)
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let mut graph = ComputeGraph::new();

        let input = Node::input("x");
        let input_id = graph.add_node(input);

        let relu = Node::operator("relu1", "ReLU");
        let relu_id = graph.add_node(relu);

        let output = Node::output("y");
        let output_id = graph.add_node(output);

        graph.add_edge(input_id, relu_id).unwrap();
        graph.add_edge(relu_id, output_id).unwrap();

        assert_eq!(graph.inputs().len(), 1);
        assert_eq!(graph.outputs().len(), 1);
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = ComputeGraph::new();

        let n0 = graph.add_node(Node::input("input"));
        let n1 = graph.add_node(Node::operator("op1", "ReLU"));
        let n2 = graph.add_node(Node::operator("op2", "Conv2D"));
        let n3 = graph.add_node(Node::output("output"));

        graph.add_edge(n0, n1).unwrap();
        graph.add_edge(n1, n2).unwrap();
        graph.add_edge(n2, n3).unwrap();

        let order = graph.topological_sort().unwrap();
        assert_eq!(order.len(), 4);

        // Verify order: input -> op1 -> op2 -> output
        assert_eq!(order[0], n0);
        assert_eq!(order[1], n1);
        assert_eq!(order[2], n2);
        assert_eq!(order[3], n3);
    }
}
