use wasm_bindgen::prelude::*;
use crate::tensor::{Tensor, Shape};
use crate::engine::{Model, ComputeGraph, Node, Runtime, ModelLoader};
use std::collections::HashMap;

/// TinyInfer API - Main entry point for JavaScript
#[wasm_bindgen]
pub struct TinyInfer {
    model: Model,
    runtime: Runtime,
}

#[wasm_bindgen]
impl TinyInfer {
    /// Create a new TinyInfer instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            model: Model::new(ComputeGraph::new()),
            runtime: Runtime::new(),
        }
    }

    /// Load a simple test model (for demonstration)
    /// In the future, this will load ONNX models
    #[wasm_bindgen(js_name = loadTestModel)]
    pub fn load_test_model(&mut self) {
        // Create a simple graph: input -> ReLU -> output
        let mut graph = ComputeGraph::new();

        let input_node = Node::input("input");
        let input_id = graph.add_node(input_node);

        let relu_node = Node::operator("relu", "ReLU");
        let relu_id = graph.add_node(relu_node);

        let output_node = Node::output("output");
        let output_id = graph.add_node(output_node);

        let _ = graph.add_edge(input_id, relu_id);
        let _ = graph.add_edge(relu_id, output_id);

        self.model = Model::new(graph);
        self.model.set_initialized(true);
    }

    /// Load a model from JSON string
    #[wasm_bindgen(js_name = loadModelFromJSON)]
    pub fn load_model_from_json(&mut self, json: &str) -> Result<(), JsValue> {
        match ModelLoader::load_from_json(json) {
            Ok(model) => {
                self.model = model;
                Ok(())
            }
            Err(e) => Err(JsValue::from_str(&format!("Failed to load model: {}", e))),
        }
    }

    /// Check if model is loaded
    #[wasm_bindgen(js_name = isModelLoaded)]
    pub fn is_model_loaded(&self) -> bool {
        self.model.is_initialized()
    }

    /// Run inference with the loaded model
    /// Takes input data as Float32Array and returns output as Float32Array
    #[wasm_bindgen(js_name = infer)]
    pub fn infer(&mut self, input_data: Vec<f32>, input_shape: Vec<usize>) -> Vec<f32> {
        let shape = Shape::new(input_shape);
        let input_tensor = Tensor::new(input_data, shape);

        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), input_tensor);

        match self.runtime.run(&self.model, inputs) {
            Ok(outputs) => {
                if let Some(output) = outputs.get("output") {
                    output.data().to_vec()
                } else {
                    vec![]
                }
            }
            Err(e) => {
                log::error!("Inference failed: {}", e);
                vec![]
            }
        }
    }

    /// Get model information
    #[wasm_bindgen(js_name = getModelInfo)]
    pub fn get_model_info(&self) -> String {
        self.model.get_info()
    }
}

/// Utility function to create a tensor from JavaScript
#[wasm_bindgen(js_name = createTensor)]
pub fn create_tensor(data: Vec<f32>, dims: Vec<usize>) -> Tensor {
    let shape = Shape::new(dims);
    Tensor::new(data, shape)
}

/// Benchmark a specific operation
#[wasm_bindgen]
pub struct Benchmark {
    results: Vec<f64>,
}

#[wasm_bindgen]
impl Benchmark {
    /// Create a new benchmark
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Benchmark matrix multiplication
    #[wasm_bindgen(js_name = benchmarkMatMul)]
    pub fn benchmark_matmul(&mut self, size: usize, iterations: usize) -> f64 {
        use crate::ops::{MatMul, Operator};
        use crate::utils::Timer;

        let data_a = vec![1.0; size * size];
        let data_b = vec![1.0; size * size];

        let shape = Shape::new(vec![size, size]);
        let tensor_a = Tensor::new(data_a, shape.clone());
        let tensor_b = Tensor::new(data_b, shape);

        let matmul = MatMul::new();

        let timer = Timer::new();
        for _ in 0..iterations {
            let _ = matmul.forward(&[&tensor_a, &tensor_b]);
        }
        let elapsed = timer.elapsed();

        let avg_time = elapsed / iterations as f64;
        self.results.push(avg_time);

        avg_time
    }

    /// Benchmark ReLU activation
    #[wasm_bindgen(js_name = benchmarkReLU)]
    pub fn benchmark_relu(&mut self, size: usize, iterations: usize) -> f64 {
        use crate::ops::{ReLU, Operator};
        use crate::utils::Timer;

        let data = vec![-1.0; size];
        let shape = Shape::new(vec![size]);
        let tensor = Tensor::new(data, shape);

        let relu = ReLU::new();

        let timer = Timer::new();
        for _ in 0..iterations {
            let _ = relu.forward(&[&tensor]);
        }
        let elapsed = timer.elapsed();

        let avg_time = elapsed / iterations as f64;
        self.results.push(avg_time);

        avg_time
    }

    /// Benchmark Conv2D
    #[wasm_bindgen(js_name = benchmarkConv2D)]
    pub fn benchmark_conv2d(&mut self, batch: usize, channels: usize, size: usize, iterations: usize) -> f64 {
        use crate::ops::{Conv2D, Operator};
        use crate::utils::Timer;

        let input_size = batch * channels * size * size;
        let weight_size = channels * channels * 3 * 3;

        let input_data = vec![1.0; input_size];
        let weight_data = vec![1.0; weight_size];

        let input_shape = Shape::new(vec![batch, channels, size, size]);
        let weight_shape = Shape::new(vec![channels, channels, 3, 3]);

        let input = Tensor::new(input_data, input_shape);
        let weight = Tensor::new(weight_data, weight_shape);

        let conv = Conv2D::with_params([1, 1], [1, 1], [1, 1], 1);

        let timer = Timer::new();
        for _ in 0..iterations {
            let _ = conv.forward(&[&input, &weight]);
        }
        let elapsed = timer.elapsed();

        let avg_time = elapsed / iterations as f64;
        self.results.push(avg_time);

        avg_time
    }

    /// Get all benchmark results
    #[wasm_bindgen(js_name = getResults)]
    pub fn get_results(&self) -> Vec<f64> {
        self.results.clone()
    }

    /// Clear benchmark results
    #[wasm_bindgen(js_name = clearResults)]
    pub fn clear_results(&mut self) {
        self.results.clear();
    }
}

impl Default for Benchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tinyinfer_creation() {
        let tinyinfer = TinyInfer::new();
        assert!(!tinyinfer.model.is_initialized());
    }

    #[test]
    fn test_load_test_model() {
        let mut tinyinfer = TinyInfer::new();
        tinyinfer.load_test_model();
        assert!(tinyinfer.model.is_initialized());
    }

    #[test]
    fn test_inference() {
        let mut tinyinfer = TinyInfer::new();
        tinyinfer.load_test_model();

        let input_data = vec![-1.0, 2.0, -3.0, 4.0];
        let input_shape = vec![4];

        let output = tinyinfer.infer(input_data, input_shape);
        assert_eq!(output, vec![0.0, 2.0, 0.0, 4.0]); // ReLU applied
    }
}
