use crate::error::{Result, TinyInferError};
use crate::tensor::Tensor;
use super::ComputeGraph;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// A neural network model
#[wasm_bindgen]
pub struct Model {
    graph: ComputeGraph,
    weights: HashMap<String, Tensor>,
    initialized: bool,
}

impl Model {
    pub fn new(graph: ComputeGraph) -> Self {
        Self {
            graph,
            weights: HashMap::new(),
            initialized: false,
        }
    }

    /// Add a weight tensor to the model
    pub fn add_weight(&mut self, name: String, tensor: Tensor) {
        self.weights.insert(name, tensor);
    }

    /// Get a weight tensor by name
    pub fn get_weight(&self, name: &str) -> Option<&Tensor> {
        self.weights.get(name)
    }

    /// Get the compute graph
    pub fn graph(&self) -> &ComputeGraph {
        &self.graph
    }

    /// Mark model as initialized
    pub fn set_initialized(&mut self, initialized: bool) {
        self.initialized = initialized;
    }

    /// Check if model is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

#[wasm_bindgen]
impl Model {
    /// Create a new model (for WASM)
    #[wasm_bindgen(constructor)]
    pub fn new_js() -> Model {
        Model::new(ComputeGraph::new())
    }

    /// Get model info as JSON string
    #[wasm_bindgen(js_name = getInfo)]
    pub fn get_info(&self) -> String {
        format!(
            "{{\"nodes\": {}, \"weights\": {}, \"initialized\": {}}}",
            self.graph.nodes().len(),
            self.weights.len(),
            self.initialized
        )
    }

    /// Check if initialized
    #[wasm_bindgen(js_name = isInitialized)]
    pub fn is_initialized_js(&self) -> bool {
        self.initialized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let graph = ComputeGraph::new();
        let model = Model::new(graph);

        assert!(!model.is_initialized());
        assert_eq!(model.weights.len(), 0);
    }
}
