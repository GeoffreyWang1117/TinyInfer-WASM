use crate::error::{Result, TinyInferError};
use crate::tensor::{Tensor, Shape};
use crate::ops::*;
use super::{Model, ComputeGraph, Node, model_def::*};
use std::collections::HashMap;

/// Model loader - loads models from JSON format
pub struct ModelLoader;

impl ModelLoader {
    /// Load model from JSON string
    pub fn load_from_json(json_str: &str) -> Result<Model> {
        let model_def: ModelDef = serde_json::from_str(json_str)
            .map_err(|e| TinyInferError::ModelLoadError(format!("JSON parse error: {}", e)))?;

        Self::load_from_def(model_def)
    }

    /// Load model from ModelDef
    pub fn load_from_def(model_def: ModelDef) -> Result<Model> {
        // Validate version
        if model_def.version != "1.0" {
            return Err(TinyInferError::ModelLoadError(
                format!("Unsupported model version: {}", model_def.version)
            ));
        }

        // Build compute graph
        let mut graph = ComputeGraph::new();
        let mut node_id_map: HashMap<String, usize> = HashMap::new();

        // Add nodes
        for node_def in &model_def.graph.nodes {
            let node = Self::create_node(&node_def)?;
            let id = graph.add_node(node);
            node_id_map.insert(node_def.id.clone(), id);
        }

        // Add edges
        for edge_def in &model_def.graph.edges {
            let from_id = node_id_map.get(&edge_def.from)
                .ok_or_else(|| TinyInferError::ModelLoadError(
                    format!("Unknown node: {}", edge_def.from)
                ))?;
            let to_id = node_id_map.get(&edge_def.to)
                .ok_or_else(|| TinyInferError::ModelLoadError(
                    format!("Unknown node: {}", edge_def.to)
                ))?;

            graph.add_edge(*from_id, *to_id)?;
        }

        // Create model
        let mut model = Model::new(graph);

        // Load weights
        for (name, weight_def) in model_def.weights {
            let tensor = Self::create_tensor(&weight_def)?;
            model.add_weight(name, tensor);
        }

        model.set_initialized(true);

        Ok(model)
    }

    /// Create a node from node definition
    fn create_node(node_def: &NodeDef) -> Result<Node> {
        match node_def.op_type.as_str() {
            "Input" => Ok(Node::input(&node_def.id)),
            "Output" => Ok(Node::output(&node_def.id)),
            _ => {
                // Create operator node
                Ok(Node::operator(&node_def.id, &node_def.op_type))
            }
        }
    }

    /// Create a tensor from weight definition
    fn create_tensor(weight_def: &WeightDef) -> Result<Tensor> {
        if weight_def.dtype != "float32" {
            return Err(TinyInferError::ModelLoadError(
                format!("Unsupported dtype: {}", weight_def.dtype)
            ));
        }

        let shape = Shape::new(weight_def.shape.clone());
        let expected_size = shape.size();

        if weight_def.data.len() != expected_size {
            return Err(TinyInferError::ModelLoadError(
                format!("Weight data size mismatch: expected {}, got {}",
                    expected_size, weight_def.data.len())
            ));
        }

        Ok(Tensor::new(weight_def.data.clone(), shape))
    }
}

/// Helper to create a simple model definition
pub fn create_simple_relu_model() -> ModelDef {
    ModelDef {
        version: "1.0".to_string(),
        name: "simple_relu".to_string(),
        graph: GraphDef {
            nodes: vec![
                NodeDef {
                    id: "input".to_string(),
                    op_type: "Input".to_string(),
                    inputs: vec![],
                    outputs: vec!["input".to_string()],
                    attributes: HashMap::new(),
                },
                NodeDef {
                    id: "relu".to_string(),
                    op_type: "ReLU".to_string(),
                    inputs: vec!["input".to_string()],
                    outputs: vec!["relu_out".to_string()],
                    attributes: HashMap::new(),
                },
                NodeDef {
                    id: "output".to_string(),
                    op_type: "Output".to_string(),
                    inputs: vec!["relu_out".to_string()],
                    outputs: vec![],
                    attributes: HashMap::new(),
                },
            ],
            edges: vec![
                EdgeDef {
                    from: "input".to_string(),
                    to: "relu".to_string(),
                },
                EdgeDef {
                    from: "relu".to_string(),
                    to: "output".to_string(),
                },
            ],
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
        },
        weights: HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_simple_model() {
        let model_def = create_simple_relu_model();
        let json = serde_json::to_string(&model_def).unwrap();

        let model = ModelLoader::load_from_json(&json).unwrap();
        assert!(model.is_initialized());
    }

    #[test]
    fn test_load_model_with_weights() {
        let mut weights = HashMap::new();
        weights.insert(
            "weight1".to_string(),
            WeightDef {
                shape: vec![2, 3],
                dtype: "float32".to_string(),
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
        );

        let model_def = ModelDef {
            version: "1.0".to_string(),
            name: "test".to_string(),
            graph: GraphDef {
                nodes: vec![
                    NodeDef {
                        id: "input".to_string(),
                        op_type: "Input".to_string(),
                        inputs: vec![],
                        outputs: vec!["input".to_string()],
                        attributes: HashMap::new(),
                    },
                    NodeDef {
                        id: "output".to_string(),
                        op_type: "Output".to_string(),
                        inputs: vec!["input".to_string()],
                        outputs: vec![],
                        attributes: HashMap::new(),
                    },
                ],
                edges: vec![],
                inputs: vec!["input".to_string()],
                outputs: vec!["output".to_string()],
            },
            weights,
        };

        let model = ModelLoader::load_from_def(model_def).unwrap();
        assert!(model.is_initialized());

        let weight = model.get_weight("weight1").unwrap();
        assert_eq!(weight.shape().dims(), &[2, 3]);
        assert_eq!(weight.data().len(), 6);
    }
}
