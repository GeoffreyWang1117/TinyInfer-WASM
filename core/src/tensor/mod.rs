mod shape;
mod dtype;
mod tensor;

pub use shape::Shape;
pub use dtype::DataType;
pub use tensor::Tensor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = Shape::new(vec![2, 2]);
        let tensor = Tensor::new(data, shape);

        assert_eq!(tensor.size(), 4);
        assert_eq!(tensor.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_tensor_reshape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = Shape::new(vec![2, 3]);
        let mut tensor = Tensor::new(data, shape);

        let new_shape = Shape::new(vec![3, 2]);
        tensor.reshape(new_shape).unwrap();

        assert_eq!(tensor.shape().dims(), &[3, 2]);
    }
}
