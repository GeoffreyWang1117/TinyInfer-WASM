use crate::error::{Result, TinyInferError};
use crate::ops::Operator;
use crate::tensor::Tensor;
use crate::simd;

/// ReLU activation function
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }

    /// Apply ReLU element-wise
    pub fn apply(x: f32) -> f32 {
        x.max(0.0)
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for ReLU {
    fn name(&self) -> &str {
        "ReLU"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 1 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "ReLU expects 1 input, got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];

        // Use SIMD-optimized ReLU
        let mut output_data = vec![0.0; input.size()];
        if simd::is_simd_available() {
            simd::simd_relu_f32(input.data(), &mut output_data);
        } else {
            for i in 0..input.size() {
                output_data[i] = input.data()[i].max(0.0);
            }
        }

        Ok(Tensor::new(output_data, input.shape().clone()))
    }
}

/// ReLU6 activation (clamps to [0, 6])
pub struct ReLU6;

impl ReLU6 {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReLU6 {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for ReLU6 {
    fn name(&self) -> &str {
        "ReLU6"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 1 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "ReLU6 expects 1 input, got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];
        let mut output = input.clone_tensor();

        for val in output.data_mut() {
            *val = val.max(0.0).min(6.0);
        }

        Ok(output)
    }
}

/// Sigmoid activation
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }

    pub fn apply(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for Sigmoid {
    fn name(&self) -> &str {
        "Sigmoid"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 1 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "Sigmoid expects 1 input, got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];
        let mut output = input.clone_tensor();

        for val in output.data_mut() {
            *val = Self::apply(*val);
        }

        Ok(output)
    }
}

/// Tanh activation
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for Tanh {
    fn name(&self) -> &str {
        "Tanh"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 1 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "Tanh expects 1 input, got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];
        let mut output = input.clone_tensor();

        for val in output.data_mut() {
            *val = val.tanh();
        }

        Ok(output)
    }
}

/// GELU activation (Gaussian Error Linear Unit)
pub struct GELU;

impl GELU {
    pub fn new() -> Self {
        Self
    }

    /// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    pub fn apply(x: f32) -> f32 {
        const SQRT_2_OVER_PI: f32 = 0.7978845608;
        let x3 = x * x * x;
        0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x3)).tanh())
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for GELU {
    fn name(&self) -> &str {
        "GELU"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 1 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "GELU expects 1 input, got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];
        let mut output = input.clone_tensor();

        for val in output.data_mut() {
            *val = Self::apply(*val);
        }

        Ok(output)
    }
}

/// Softmax activation
pub struct Softmax {
    axis: isize,
}

impl Softmax {
    pub fn new(axis: isize) -> Self {
        Self { axis }
    }
}

impl Operator for Softmax {
    fn name(&self) -> &str {
        "Softmax"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 1 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "Softmax expects 1 input, got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];
        let mut output = input.clone_tensor();
        let data = output.data_mut();

        // For simplicity, assuming axis = -1 (last dimension)
        // In a full implementation, we'd need to handle arbitrary axes

        // Find max for numerical stability
        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max)
        for val in data.iter_mut() {
            *val = (*val - max_val).exp();
        }

        // Compute sum
        let sum: f32 = data.iter().sum();

        // Normalize
        for val in data.iter_mut() {
            *val /= sum;
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    #[test]
    fn test_relu() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let shape = Shape::new(vec![5]);
        let input = Tensor::new(data, shape);

        let relu = ReLU::new();
        let output = relu.forward(&[&input]).unwrap();

        assert_eq!(output.data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let data = vec![0.0];
        let shape = Shape::new(vec![1]);
        let input = Tensor::new(data, shape);

        let sigmoid = Sigmoid::new();
        let output = sigmoid.forward(&[&input]).unwrap();

        assert!((output.data()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = Shape::new(vec![3]);
        let input = Tensor::new(data, shape);

        let softmax = Softmax::new(-1);
        let output = softmax.forward(&[&input]).unwrap();

        // Sum should be 1
        let sum: f32 = output.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Values should be in ascending order
        assert!(output.data()[0] < output.data()[1]);
        assert!(output.data()[1] < output.data()[2]);
    }
}
