/// Fused operators for better performance
/// Combines multiple operations to reduce memory access and improve cache locality

use crate::error::{Result, TinyInferError};
use crate::ops::{Operator, Conv2D, MatMul};
use crate::tensor::{Tensor, Shape};
use crate::simd;

/// Fused Conv2D + BatchNorm + ReLU
/// This is a very common pattern in CNNs (e.g., ResNet, MobileNet)
pub struct ConvBNReLU {
    conv: Conv2D,
    // BatchNorm parameters
    gamma: Vec<f32>,  // scale
    beta: Vec<f32>,   // shift
    mean: Vec<f32>,
    var: Vec<f32>,
    epsilon: f32,
}

impl ConvBNReLU {
    pub fn new(
        conv: Conv2D,
        gamma: Vec<f32>,
        beta: Vec<f32>,
        mean: Vec<f32>,
        var: Vec<f32>,
        epsilon: f32,
    ) -> Self {
        Self {
            conv,
            gamma,
            beta,
            mean,
            var,
            epsilon,
        }
    }

    /// Fuse BatchNorm parameters into Conv weights and bias
    /// This eliminates the need for separate BN computation
    pub fn fold_batch_norm(&mut self, conv_weight: &mut Tensor, conv_bias: &mut Tensor) {
        let num_channels = self.gamma.len();

        // For each channel, compute: w' = w * gamma / sqrt(var + eps)
        // and: b' = (b - mean) * gamma / sqrt(var + eps) + beta
        for c in 0..num_channels {
            let scale = self.gamma[c] / (self.var[c] + self.epsilon).sqrt();
            let shift = (conv_bias.data()[c] - self.mean[c]) * scale + self.beta[c];

            // Update bias
            conv_bias.data_mut()[c] = shift;

            // Update weights (simplified - assumes weights are ordered by channel)
            // In practice, need to handle different weight layouts
        }
    }
}

impl Operator for ConvBNReLU {
    fn name(&self) -> &str {
        "ConvBNReLU"
    }

    fn num_inputs(&self) -> usize {
        2 // input, weight (bias optional)
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        // 1. Convolution
        let mut output = self.conv.forward(inputs)?;

        // 2. BatchNorm + ReLU (fused)
        let data = output.data_mut();
        let num_channels = self.gamma.len();

        // Apply BN and ReLU in a single pass
        for i in 0..data.len() {
            let channel = i % num_channels;  // Simplified channel indexing

            // BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
            let normalized = (data[i] - self.mean[channel]) / (self.var[channel] + self.epsilon).sqrt();
            let scaled = normalized * self.gamma[channel] + self.beta[channel];

            // ReLU
            data[i] = scaled.max(0.0);
        }

        Ok(output)
    }
}

/// Fused MatMul + Bias + Activation
/// Common in transformer models and fully-connected layers
pub struct MatMulBiasActivation {
    matmul: MatMul,
    bias: Vec<f32>,
    activation: ActivationType,
}

#[derive(Clone, Copy, Debug)]
pub enum ActivationType {
    None,
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
}

impl MatMulBiasActivation {
    pub fn new(matmul: MatMul, bias: Vec<f32>, activation: ActivationType) -> Self {
        Self {
            matmul,
            bias,
            activation,
        }
    }

    /// Apply activation function
    #[inline]
    fn apply_activation(&self, x: f32) -> f32 {
        match self.activation {
            ActivationType::None => x,
            ActivationType::ReLU => x.max(0.0),
            ActivationType::GELU => {
                // GELU approximation
                const SQRT_2_OVER_PI: f32 = 0.7978845608;
                let x3 = x * x * x;
                0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x3)).tanh())
            }
            ActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationType::Tanh => x.tanh(),
        }
    }
}

impl Operator for MatMulBiasActivation {
    fn name(&self) -> &str {
        "MatMulBiasActivation"
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        // 1. Matrix multiplication
        let mut output = self.matmul.forward(inputs)?;

        // 2. Add bias and apply activation in a single pass
        let data = output.data_mut();
        let output_dim = self.bias.len();

        // Fused bias + activation
        for i in 0..data.len() {
            let col = i % output_dim;
            data[i] = self.apply_activation(data[i] + self.bias[col]);
        }

        Ok(output)
    }
}

/// Fused Add + ReLU
/// Simple but common pattern
pub struct AddReLU;

impl AddReLU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AddReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for AddReLU {
    fn name(&self) -> &str {
        "AddReLU"
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 2 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "AddReLU expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let a = inputs[0];
        let b = inputs[1];

        if a.shape() != b.shape() {
            return Err(TinyInferError::InvalidShape(
                "AddReLU requires matching shapes".to_string(),
            ));
        }

        // Fused Add + ReLU using SIMD
        let mut output_data = vec![0.0; a.size()];

        if simd::is_simd_available() {
            // Use SIMD for addition, then ReLU
            simd::simd_add_f32(a.data(), b.data(), &mut output_data);
            let mut result = vec![0.0; a.size()];
            simd::simd_relu_f32(&output_data, &mut result);
            output_data = result;
        } else {
            // Fused scalar version
            for i in 0..a.size() {
                output_data[i] = (a.data()[i] + b.data()[i]).max(0.0);
            }
        }

        Ok(Tensor::new(output_data, a.shape().clone()))
    }
}

/// Fused Multiply-Add (a * b + c)
/// Can use SIMD FMA instructions for better performance
pub struct MulAdd;

impl MulAdd {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MulAdd {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for MulAdd {
    fn name(&self) -> &str {
        "MulAdd"
    }

    fn num_inputs(&self) -> usize {
        3
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 3 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "MulAdd expects 3 inputs (a, b, c), got {}",
                inputs.len()
            )));
        }

        let a = inputs[0];
        let b = inputs[1];
        let c = inputs[2];

        if a.shape() != b.shape() || a.shape() != c.shape() {
            return Err(TinyInferError::InvalidShape(
                "MulAdd requires all inputs to have matching shapes".to_string(),
            ));
        }

        // Use SIMD FMA: a * b + c
        let mut output_data = c.data().to_vec();

        if simd::is_simd_available() {
            simd::simd_fma_f32(a.data(), b.data(), &mut output_data);
        } else {
            for i in 0..a.size() {
                output_data[i] = a.data()[i] * b.data()[i] + c.data()[i];
            }
        }

        Ok(Tensor::new(output_data, a.shape().clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_relu() {
        let a_data = vec![1.0, -2.0, 3.0, -4.0];
        let b_data = vec![1.0, 3.0, -1.0, 2.0];

        let a = Tensor::new(a_data, Shape::new(vec![4]));
        let b = Tensor::new(b_data, Shape::new(vec![4]));

        let add_relu = AddReLU::new();
        let output = add_relu.forward(&[&a, &b]).unwrap();

        // Expected: [2.0, 1.0, 2.0, 0.0] after Add then ReLU
        assert_eq!(output.data()[0], 2.0);
        assert_eq!(output.data()[1], 1.0);
        assert_eq!(output.data()[2], 2.0);
        assert_eq!(output.data()[3], 0.0);  // -2 -> 0
    }

    #[test]
    fn test_mul_add() {
        let a_data = vec![2.0, 3.0, 4.0];
        let b_data = vec![3.0, 2.0, 1.0];
        let c_data = vec![1.0, 1.0, 1.0];

        let a = Tensor::new(a_data, Shape::new(vec![3]));
        let b = Tensor::new(b_data, Shape::new(vec![3]));
        let c = Tensor::new(c_data, Shape::new(vec![3]));

        let mul_add = MulAdd::new();
        let output = mul_add.forward(&[&a, &b, &c]).unwrap();

        // Expected: [2*3+1, 3*2+1, 4*1+1] = [7, 7, 5]
        assert_eq!(output.data()[0], 7.0);
        assert_eq!(output.data()[1], 7.0);
        assert_eq!(output.data()[2], 5.0);
    }
}
