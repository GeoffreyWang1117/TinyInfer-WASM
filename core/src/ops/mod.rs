mod matmul;
mod conv;
mod activation;
mod pool;
mod norm;
mod elementwise;

pub use matmul::*;
pub use conv::*;
pub use activation::*;
pub use pool::*;
pub use norm::*;
pub use elementwise::*;

use crate::tensor::Tensor;
use crate::error::Result;

/// Trait for all operators
pub trait Operator: Send + Sync {
    /// Get operator name
    fn name(&self) -> &str;

    /// Execute the operator
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor>;

    /// Get expected number of inputs
    fn num_inputs(&self) -> usize {
        1
    }

    /// Get expected number of outputs
    fn num_outputs(&self) -> usize {
        1
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
}
