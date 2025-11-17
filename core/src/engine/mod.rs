mod graph;
mod model;
mod model_def;
mod model_loader;
mod runtime;

pub use graph::{ComputeGraph, Node, NodeId};
pub use model::Model;
pub use model_def::{ModelDef, GraphDef, NodeDef, EdgeDef, WeightDef, AttributeValue};
pub use model_loader::{ModelLoader, create_simple_relu_model};
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
