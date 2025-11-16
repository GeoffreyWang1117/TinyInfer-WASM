use crate::error::{Result, TinyInferError};
use crate::ops::Operator;
use crate::tensor::Tensor;

/// Batch Normalization
/// y = (x - mean) / sqrt(var + eps) * gamma + beta
pub struct BatchNorm2D {
    eps: f32,
    momentum: f32,
}

impl BatchNorm2D {
    pub fn new() -> Self {
        Self {
            eps: 1e-5,
            momentum: 0.1,
        }
    }

    pub fn with_params(eps: f32, momentum: f32) -> Self {
        Self { eps, momentum }
    }
}

impl Default for BatchNorm2D {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for BatchNorm2D {
    fn name(&self) -> &str {
        "BatchNorm2D"
    }

    fn num_inputs(&self) -> usize {
        5 // input, gamma, beta, running_mean, running_var
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 5 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "BatchNorm2D expects 5 inputs (x, gamma, beta, mean, var), got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];
        let gamma = inputs[1];
        let beta = inputs[2];
        let running_mean = inputs[3];
        let running_var = inputs[4];

        if input.ndim() != 4 {
            return Err(TinyInferError::InvalidShape(
                "BatchNorm2D input must be 4D [N, C, H, W]".to_string(),
            ));
        }

        let shape = input.shape().dims();
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

        if gamma.size() != channels
            || beta.size() != channels
            || running_mean.size() != channels
            || running_var.size() != channels
        {
            return Err(TinyInferError::InvalidShape(
                "BatchNorm2D: gamma, beta, mean, var must have size equal to channels".to_string(),
            ));
        }

        let mut output = input.clone_tensor();
        let input_data = input.data();
        let output_data = output.data_mut();
        let gamma_data = gamma.data();
        let beta_data = beta.data();
        let mean_data = running_mean.data();
        let var_data = running_var.data();

        let spatial_size = height * width;

        for b in 0..batch {
            for c in 0..channels {
                let mean = mean_data[c];
                let var = var_data[c];
                let scale = gamma_data[c] / (var + self.eps).sqrt();
                let shift = beta_data[c] - mean * scale;

                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        output_data[idx] = input_data[idx] * scale + shift;
                    }
                }
            }
        }

        Ok(output)
    }
}

/// Layer Normalization
/// Normalizes over the last dimension(s)
pub struct LayerNorm {
    eps: f32,
    normalized_shape: Vec<usize>,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        Self {
            eps: 1e-5,
            normalized_shape,
        }
    }

    pub fn with_eps(normalized_shape: Vec<usize>, eps: f32) -> Self {
        Self {
            eps,
            normalized_shape,
        }
    }
}

impl Operator for LayerNorm {
    fn name(&self) -> &str {
        "LayerNorm"
    }

    fn num_inputs(&self) -> usize {
        3 // input, gamma, beta
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 3 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "LayerNorm expects 3 inputs (x, gamma, beta), got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];
        let gamma = inputs[1];
        let beta = inputs[2];

        let input_shape = input.shape().dims();

        // Calculate normalized size
        let norm_size: usize = self.normalized_shape.iter().product();

        // For simplicity, assume we normalize over the last dimension
        let batch_size = input.size() / norm_size;

        let mut output = input.clone_tensor();
        let input_data = input.data();
        let output_data = output.data_mut();
        let gamma_data = gamma.data();
        let beta_data = beta.data();

        for b in 0..batch_size {
            let offset = b * norm_size;

            // Calculate mean
            let mut sum = 0.0;
            for i in 0..norm_size {
                sum += input_data[offset + i];
            }
            let mean = sum / norm_size as f32;

            // Calculate variance
            let mut var_sum = 0.0;
            for i in 0..norm_size {
                let diff = input_data[offset + i] - mean;
                var_sum += diff * diff;
            }
            let var = var_sum / norm_size as f32;

            // Normalize
            let std = (var + self.eps).sqrt();
            for i in 0..norm_size {
                let normalized = (input_data[offset + i] - mean) / std;
                output_data[offset + i] = normalized * gamma_data[i] + beta_data[i];
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    #[test]
    fn test_batchnorm2d() {
        let input = Tensor::ones(Shape::new(vec![2, 2, 3, 3]));
        let gamma = Tensor::ones(Shape::new(vec![2]));
        let beta = Tensor::zeros(Shape::new(vec![2]));
        let mean = Tensor::ones(Shape::new(vec![2]));
        let var = Tensor::zeros(Shape::new(vec![2]));

        let bn = BatchNorm2D::new();
        let output = bn.forward(&[&input, &gamma, &beta, &mean, &var]).unwrap();

        assert_eq!(output.shape().dims(), &[2, 2, 3, 3]);

        // With mean=1, var=0, input=1: output should be 0
        for &val in output.data() {
            assert!(val.abs() < 1e-3);
        }
    }

    #[test]
    fn test_layernorm() {
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::new(input_data, Shape::new(vec![2, 3]));

        let gamma = Tensor::ones(Shape::new(vec![3]));
        let beta = Tensor::zeros(Shape::new(vec![3]));

        let ln = LayerNorm::new(vec![3]);
        let output = ln.forward(&[&input, &gamma, &beta]).unwrap();

        assert_eq!(output.shape().dims(), &[2, 3]);

        // Each row should be normalized to mean=0, std=1
        let row1_mean = (output.data()[0] + output.data()[1] + output.data()[2]) / 3.0;
        assert!(row1_mean.abs() < 1e-5);
    }
}
