mod graph;
mod model;
mod runtime;

pub use graph::{ComputeGraph, Node, NodeId};
pub use model::Model;
pub use runtime::Runtime;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_graph() {
        let mut graph = ComputeGraph::new();
        let node = Node::input("x");
        let id = graph.add_node(node);
        assert!(graph.get_node(id).is_some());
    }
}
