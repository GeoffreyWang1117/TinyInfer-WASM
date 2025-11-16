use crate::error::{Result, TinyInferError};
use super::{DataType, Shape};
use wasm_bindgen::prelude::*;

/// Tensor data structure
/// Currently only supports f32 for simplicity
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
    dtype: DataType,
}

impl Tensor {
    /// Create a new tensor
    pub fn new(data: Vec<f32>, shape: Shape) -> Self {
        assert_eq!(
            data.len(),
            shape.size(),
            "Data length must match shape size"
        );

        Self {
            data,
            shape,
            dtype: DataType::Float32,
        }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Shape) -> Self {
        let size = shape.size();
        Self::new(vec![0.0; size], shape)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Shape) -> Self {
        let size = shape.size();
        Self::new(vec![1.0; size], shape)
    }

    /// Create a tensor filled with a constant value
    pub fn full(shape: Shape, value: f32) -> Self {
        let size = shape.size();
        Self::new(vec![value; size], shape)
    }

    /// Get the shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the data type
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Get the raw data
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable raw data
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get total number of elements
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Reshape the tensor
    pub fn reshape(&mut self, new_shape: Shape) -> Result<()> {
        if new_shape.size() != self.size() {
            return Err(TinyInferError::InvalidShape(format!(
                "Cannot reshape tensor of size {} to shape with size {}",
                self.size(),
                new_shape.size()
            )));
        }

        self.shape = new_shape;
        Ok(())
    }

    /// Get element at multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> Result<f32> {
        let offset = self.shape.index_to_offset(indices)?;
        Ok(self.data[offset])
    }

    /// Set element at multi-dimensional index
    pub fn set(&mut self, indices: &[usize], value: f32) -> Result<()> {
        let offset = self.shape.index_to_offset(indices)?;
        self.data[offset] = value;
        Ok(())
    }

    /// Clone the tensor
    pub fn clone_tensor(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }

    /// Convert to a specific layout (NCHW, NHWC, etc.)
    /// For now, we assume all tensors are in the same layout
    pub fn transpose(&self, axes: &[usize]) -> Result<Self> {
        if axes.len() != self.ndim() {
            return Err(TinyInferError::InvalidShape(format!(
                "Expected {} axes for transpose, got {}",
                self.ndim(),
                axes.len()
            )));
        }

        // Check if axes is a valid permutation
        let mut seen = vec![false; axes.len()];
        for &axis in axes {
            if axis >= axes.len() {
                return Err(TinyInferError::InvalidShape(format!(
                    "Axis {} out of bounds for tensor with {} dimensions",
                    axis,
                    axes.len()
                )));
            }
            if seen[axis] {
                return Err(TinyInferError::InvalidShape(
                    "Duplicate axis in transpose".to_string(),
                ));
            }
            seen[axis] = true;
        }

        // Create new shape
        let old_dims = self.shape.dims();
        let new_dims: Vec<usize> = axes.iter().map(|&i| old_dims[i]).collect();
        let new_shape = Shape::new(new_dims);

        // Create new data with transposed layout
        let mut new_data = vec![0.0; self.size()];
        let old_strides = self.shape.strides();
        let new_strides = new_shape.strides();

        for i in 0..self.size() {
            // Convert flat index to multi-dimensional index
            let mut old_idx = i;
            let mut indices = vec![0; self.ndim()];
            for (j, &stride) in old_strides.iter().enumerate() {
                indices[j] = old_idx / stride;
                old_idx %= stride;
            }

            // Permute indices
            let mut new_indices = vec![0; self.ndim()];
            for (j, &axis) in axes.iter().enumerate() {
                new_indices[j] = indices[axis];
            }

            // Convert back to flat index
            let new_flat_idx: usize = new_indices
                .iter()
                .zip(new_strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum();

            new_data[new_flat_idx] = self.data[i];
        }

        Ok(Tensor::new(new_data, new_shape))
    }
}

// WASM bindings
#[wasm_bindgen]
impl Tensor {
    /// Create a new tensor from JavaScript
    #[wasm_bindgen(constructor)]
    pub fn new_from_js(data: Vec<f32>, dims: Vec<usize>) -> Tensor {
        let shape = Shape::new(dims);
        Tensor::new(data, shape)
    }

    /// Get shape as array
    #[wasm_bindgen(js_name = getShape)]
    pub fn get_shape(&self) -> Vec<usize> {
        self.shape.dims().to_vec()
    }

    /// Get data as array
    #[wasm_bindgen(js_name = getData)]
    pub fn get_data(&self) -> Vec<f32> {
        self.data.clone()
    }

    /// Get size
    #[wasm_bindgen(js_name = getSize)]
    pub fn get_size(&self) -> usize {
        self.size()
    }

    /// Convert to string representation
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_js(&self) -> String {
        format!(
            "Tensor(shape={}, dtype={}, size={})",
            self.shape,
            self.dtype,
            self.size()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = Shape::new(vec![2, 2]);
        let tensor = Tensor::new(data.clone(), shape);

        assert_eq!(tensor.size(), 4);
        assert_eq!(tensor.data(), &data);
    }

    #[test]
    fn test_tensor_zeros() {
        let shape = Shape::new(vec![3, 3]);
        let tensor = Tensor::zeros(shape);

        assert_eq!(tensor.size(), 9);
        assert!(tensor.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_get_set() {
        let shape = Shape::new(vec![2, 3]);
        let mut tensor = Tensor::zeros(shape);

        tensor.set(&[1, 2], 5.0).unwrap();
        assert_eq!(tensor.get(&[1, 2]).unwrap(), 5.0);
    }

    #[test]
    fn test_tensor_transpose() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = Shape::new(vec![2, 3]);
        let tensor = Tensor::new(data, shape);

        let transposed = tensor.transpose(&[1, 0]).unwrap();
        assert_eq!(transposed.shape().dims(), &[3, 2]);

        // Check data layout
        assert_eq!(transposed.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(transposed.get(&[1, 0]).unwrap(), 2.0);
        assert_eq!(transposed.get(&[2, 0]).unwrap(), 3.0);
        assert_eq!(transposed.get(&[0, 1]).unwrap(), 4.0);
    }
}
