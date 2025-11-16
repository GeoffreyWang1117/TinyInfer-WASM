use crate::error::{Result, TinyInferError};
use crate::ops::*;
use crate::tensor::Tensor;
use super::{Model, graph::NodeType};
use std::collections::HashMap;

/// Runtime for executing models
pub struct Runtime {
    // Tensor cache during execution
    tensors: HashMap<usize, Tensor>,
}

impl Runtime {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }

    /// Execute a model with given inputs
    pub fn run(&mut self, model: &Model, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        // Clear previous execution tensors
        self.tensors.clear();

        // Load input tensors
        for input_id in model.graph().inputs() {
            if let Some(node) = model.graph().get_node(*input_id) {
                if let Some(tensor) = inputs.get(&node.name) {
                    self.tensors.insert(*input_id, tensor.clone_tensor());
                } else {
                    return Err(TinyInferError::InferenceError(format!(
                        "Missing input: {}",
                        node.name
                    )));
                }
            }
        }

        // Get execution order
        let exec_order = model.graph().topological_sort()?;

        // Execute each node in order
        for &node_id in &exec_order {
            if let Some(node) = model.graph().get_node(node_id) {
                match &node.node_type {
                    NodeType::Input | NodeType::Constant => {
                        // Already loaded
                        continue;
                    }
                    NodeType::Output => {
                        // Output nodes just pass through
                        if !node.inputs.is_empty() {
                            let input_tensor = self.tensors.get(&node.inputs[0])
                                .ok_or_else(|| TinyInferError::InferenceError(
                                    format!("Missing input tensor for output node {}", node.name)
                                ))?;
                            self.tensors.insert(node_id, input_tensor.clone_tensor());
                        }
                    }
                    NodeType::Operator(op_type) => {
                        // Execute operator
                        let output = self.execute_operator(model, node_id, op_type)?;
                        self.tensors.insert(node_id, output);
                    }
                }
            }
        }

        // Collect outputs
        let mut outputs = HashMap::new();
        for output_id in model.graph().outputs() {
            if let Some(node) = model.graph().get_node(*output_id) {
                if let Some(tensor) = self.tensors.get(output_id) {
                    outputs.insert(node.name.clone(), tensor.clone_tensor());
                }
            }
        }

        Ok(outputs)
    }

    /// Execute a specific operator
    fn execute_operator(
        &self,
        model: &Model,
        node_id: usize,
        op_type: &str,
    ) -> Result<Tensor> {
        let node = model.graph().get_node(node_id)
            .ok_or_else(|| TinyInferError::InferenceError("Node not found".to_string()))?;

        // Gather input tensors
        let mut input_tensors = Vec::new();
        for &input_id in &node.inputs {
            // Try to get from runtime tensors
            if let Some(tensor) = self.tensors.get(&input_id) {
                input_tensors.push(tensor);
            } else {
                // Try to get from model weights
                if let Some(input_node) = model.graph().get_node(input_id) {
                    if let Some(weight) = model.get_weight(&input_node.name) {
                        input_tensors.push(weight);
                    } else {
                        return Err(TinyInferError::InferenceError(format!(
                            "Missing tensor for node {}",
                            input_node.name
                        )));
                    }
                }
            }
        }

        // Create operator and execute
        let inputs_ref: Vec<&Tensor> = input_tensors.iter().copied().collect();

        match op_type {
            "ReLU" => {
                let op = ReLU::new();
                op.forward(&inputs_ref)
            }
            "ReLU6" => {
                let op = ReLU6::new();
                op.forward(&inputs_ref)
            }
            "Sigmoid" => {
                let op = Sigmoid::new();
                op.forward(&inputs_ref)
            }
            "Tanh" => {
                let op = Tanh::new();
                op.forward(&inputs_ref)
            }
            "GELU" => {
                let op = GELU::new();
                op.forward(&inputs_ref)
            }
            "Add" => {
                let op = Add::new();
                op.forward(&inputs_ref)
            }
            "Sub" => {
                let op = Sub::new();
                op.forward(&inputs_ref)
            }
            "Mul" => {
                let op = Mul::new();
                op.forward(&inputs_ref)
            }
            "Div" => {
                let op = Div::new();
                op.forward(&inputs_ref)
            }
            "MatMul" => {
                let op = MatMul::new();
                op.forward(&inputs_ref)
            }
            "Conv2D" | "Conv" => {
                // Extract attributes for Conv2D
                let padding = node.attributes.get("padding")
                    .and_then(|v| match v {
                        super::graph::AttributeValue::Ints(ints) => {
                            if ints.len() >= 2 {
                                Some([ints[0] as usize, ints[1] as usize])
                            } else {
                                None
                            }
                        }
                        _ => None,
                    })
                    .unwrap_or([0, 0]);

                let stride = node.attributes.get("stride")
                    .and_then(|v| match v {
                        super::graph::AttributeValue::Ints(ints) => {
                            if ints.len() >= 2 {
                                Some([ints[0] as usize, ints[1] as usize])
                            } else {
                                None
                            }
                        }
                        _ => None,
                    })
                    .unwrap_or([1, 1]);

                let op = Conv2D::with_params(padding, stride, [1, 1], 1);
                op.forward(&inputs_ref)
            }
            "MaxPool" | "MaxPool2D" => {
                let kernel_size = node.attributes.get("kernel_size")
                    .and_then(|v| match v {
                        super::graph::AttributeValue::Ints(ints) => {
                            if ints.len() >= 2 {
                                Some([ints[0] as usize, ints[1] as usize])
                            } else {
                                None
                            }
                        }
                        _ => None,
                    })
                    .unwrap_or([2, 2]);

                let op = MaxPool2D::new(kernel_size);
                op.forward(&inputs_ref)
            }
            "AvgPool" | "AvgPool2D" => {
                let kernel_size = node.attributes.get("kernel_size")
                    .and_then(|v| match v {
                        super::graph::AttributeValue::Ints(ints) => {
                            if ints.len() >= 2 {
                                Some([ints[0] as usize, ints[1] as usize])
                            } else {
                                None
                            }
                        }
                        _ => None,
                    })
                    .unwrap_or([2, 2]);

                let op = AvgPool2D::new(kernel_size);
                op.forward(&inputs_ref)
            }
            "GlobalAvgPool" | "GlobalAvgPool2D" => {
                let op = GlobalAvgPool2D::new();
                op.forward(&inputs_ref)
            }
            _ => Err(TinyInferError::UnsupportedOp(format!(
                "Unsupported operator: {}",
                op_type
            ))),
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{ComputeGraph, Node};
    use crate::tensor::Shape;

    #[test]
    fn test_runtime_simple() {
        // Create a simple graph: input -> ReLU -> output
        let mut graph = ComputeGraph::new();

        let input_node = Node::input("x");
        let input_id = graph.add_node(input_node);

        let relu_node = Node::operator("relu", "ReLU");
        let relu_id = graph.add_node(relu_node);

        let output_node = Node::output("y");
        let output_id = graph.add_node(output_node);

        graph.add_edge(input_id, relu_id).unwrap();
        graph.add_edge(relu_id, output_id).unwrap();

        let mut model = Model::new(graph);
        model.set_initialized(true);

        // Create runtime and execute
        let mut runtime = Runtime::new();

        let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let input_tensor = Tensor::new(input_data, Shape::new(vec![5]));

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), input_tensor);

        let outputs = runtime.run(&model, inputs).unwrap();

        assert!(outputs.contains_key("y"));
        let output = &outputs["y"];
        assert_eq!(output.data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }
}
