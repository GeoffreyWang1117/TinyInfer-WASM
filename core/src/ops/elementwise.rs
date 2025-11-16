use crate::error::{Result, TinyInferError};
use crate::ops::Operator;
use crate::tensor::Tensor;
use crate::simd;

/// Element-wise addition
pub struct Add;

impl Add {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Add {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for Add {
    fn name(&self) -> &str {
        "Add"
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 2 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "Add expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let a = inputs[0];
        let b = inputs[1];

        // Check shapes are compatible
        if a.shape() != b.shape() {
            // TODO: Support broadcasting
            return Err(TinyInferError::InvalidShape(
                "Add requires matching shapes (broadcasting not yet supported)".to_string(),
            ));
        }

        // Use SIMD-optimized addition
        let mut output_data = vec![0.0; a.size()];
        if simd::is_simd_available() {
            simd::simd_add_f32(a.data(), b.data(), &mut output_data);
        } else {
            for i in 0..a.size() {
                output_data[i] = a.data()[i] + b.data()[i];
            }
        }

        Ok(Tensor::new(output_data, a.shape().clone()))
    }
}

/// Element-wise subtraction
pub struct Sub;

impl Sub {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Sub {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for Sub {
    fn name(&self) -> &str {
        "Sub"
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 2 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "Sub expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let a = inputs[0];
        let b = inputs[1];

        if a.shape() != b.shape() {
            return Err(TinyInferError::InvalidShape(
                "Sub requires matching shapes".to_string(),
            ));
        }

        // Use SIMD-optimized subtraction
        let mut output_data = vec![0.0; a.size()];
        if simd::is_simd_available() {
            simd::simd_sub_f32(a.data(), b.data(), &mut output_data);
        } else {
            for i in 0..a.size() {
                output_data[i] = a.data()[i] - b.data()[i];
            }
        }

        Ok(Tensor::new(output_data, a.shape().clone()))
    }
}

/// Element-wise multiplication
pub struct Mul;

impl Mul {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Mul {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for Mul {
    fn name(&self) -> &str {
        "Mul"
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 2 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "Mul expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let a = inputs[0];
        let b = inputs[1];

        if a.shape() != b.shape() {
            return Err(TinyInferError::InvalidShape(
                "Mul requires matching shapes".to_string(),
            ));
        }

        // Use SIMD-optimized multiplication
        let mut output_data = vec![0.0; a.size()];
        if simd::is_simd_available() {
            simd::simd_mul_f32(a.data(), b.data(), &mut output_data);
        } else {
            for i in 0..a.size() {
                output_data[i] = a.data()[i] * b.data()[i];
            }
        }

        Ok(Tensor::new(output_data, a.shape().clone()))
    }
}

/// Element-wise division
pub struct Div;

impl Div {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Div {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for Div {
    fn name(&self) -> &str {
        "Div"
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 2 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "Div expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let a = inputs[0];
        let b = inputs[1];

        if a.shape() != b.shape() {
            return Err(TinyInferError::InvalidShape(
                "Div requires matching shapes".to_string(),
            ));
        }

        // Use SIMD-optimized division
        let mut output_data = vec![0.0; a.size()];
        if simd::is_simd_available() {
            simd::simd_div_f32(a.data(), b.data(), &mut output_data);
        } else {
            for i in 0..a.size() {
                output_data[i] = a.data()[i] / b.data()[i];
            }
        }

        Ok(Tensor::new(output_data, a.shape().clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    #[test]
    fn test_add() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![5.0, 6.0, 7.0, 8.0];
        let shape = Shape::new(vec![4]);

        let tensor1 = Tensor::new(data1, shape.clone());
        let tensor2 = Tensor::new(data2, shape);

        let add = Add::new();
        let output = add.forward(&[&tensor1, &tensor2]).unwrap();

        assert_eq!(output.data(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_mul() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 2.0, 2.0, 2.0];
        let shape = Shape::new(vec![4]);

        let tensor1 = Tensor::new(data1, shape.clone());
        let tensor2 = Tensor::new(data2, shape);

        let mul = Mul::new();
        let output = mul.forward(&[&tensor1, &tensor2]).unwrap();

        assert_eq!(output.data(), &[2.0, 4.0, 6.0, 8.0]);
    }
}
