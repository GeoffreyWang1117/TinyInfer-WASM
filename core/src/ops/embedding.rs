use crate::error::{Result, TinyInferError};
use crate::ops::Operator;
use crate::tensor::Tensor;

/// Embedding Layer
/// Maps indices to dense vectors
///
/// Input:
///   - indices: [batch_size, seq_len] or [batch_size] tensor of integers
///   - weight: [vocab_size, embedding_dim] embedding matrix
///
/// Output:
///   - [batch_size, seq_len, embedding_dim] or [batch_size, embedding_dim]
pub struct Embedding {
    vocab_size: usize,
    embedding_dim: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        Self {
            vocab_size,
            embedding_dim,
        }
    }
}

impl Operator for Embedding {
    fn name(&self) -> &str {
        "Embedding"
    }

    fn num_inputs(&self) -> usize {
        2 // indices, weight
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 2 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "Embedding expects 2 inputs (indices, weight), got {}",
                inputs.len()
            )));
        }

        let indices = inputs[0];
        let weight = inputs[1];

        // Validate weight shape
        if weight.ndim() != 2 {
            return Err(TinyInferError::InvalidShape(
                "Embedding weight must be 2D [vocab_size, embedding_dim]".to_string(),
            ));
        }

        let weight_shape = weight.shape().dims();
        let vocab_size = weight_shape[0];
        let embedding_dim = weight_shape[1];

        if vocab_size != self.vocab_size || embedding_dim != self.embedding_dim {
            return Err(TinyInferError::InvalidShape(format!(
                "Embedding weight shape mismatch: expected [{}, {}], got [{}, {}]",
                self.vocab_size, self.embedding_dim, vocab_size, embedding_dim
            )));
        }

        // Get indices data
        let indices_data = indices.data();
        let weight_data = weight.data();

        // Calculate output shape
        let mut output_shape = indices.shape().dims().to_vec();
        output_shape.push(embedding_dim);

        // Allocate output
        let output_size = indices.size() * embedding_dim;
        let mut output_data = vec![0.0; output_size];

        // Lookup embeddings
        for (i, &idx_f32) in indices_data.iter().enumerate() {
            let idx = idx_f32 as usize;

            if idx >= vocab_size {
                return Err(TinyInferError::InvalidShape(format!(
                    "Index {} out of bounds for vocab_size {}",
                    idx, vocab_size
                )));
            }

            // Copy embedding vector
            let weight_offset = idx * embedding_dim;
            let output_offset = i * embedding_dim;

            for j in 0..embedding_dim {
                output_data[output_offset + j] = weight_data[weight_offset + j];
            }
        }

        Ok(Tensor::new(
            output_data,
            crate::tensor::Shape::new(output_shape),
        ))
    }
}

/// Gather operator
/// Gathers values along an axis
///
/// Input:
///   - data: [d0, d1, ..., dn] input tensor
///   - indices: [i0, i1, ..., im] indices tensor
///
/// Output:
///   - Tensor with gathered values
pub struct Gather {
    axis: isize,
}

impl Gather {
    pub fn new(axis: isize) -> Self {
        Self { axis }
    }
}

impl Operator for Gather {
    fn name(&self) -> &str {
        "Gather"
    }

    fn num_inputs(&self) -> usize {
        2 // data, indices
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 2 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "Gather expects 2 inputs (data, indices), got {}",
                inputs.len()
            )));
        }

        let data = inputs[0];
        let indices = inputs[1];

        // For simplicity, implement gather along axis 0
        // Full implementation would handle arbitrary axes
        let axis = if self.axis < 0 {
            (data.ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        if axis != 0 {
            return Err(TinyInferError::UnsupportedOp(
                "Gather currently only supports axis=0".to_string(),
            ));
        }

        let data_shape = data.shape().dims();
        let data_data = data.data();
        let indices_data = indices.data();

        // Calculate size of each slice along axis 0
        let slice_size: usize = data_shape[1..].iter().product();

        // Output shape: [indices.size(), data_shape[1], data_shape[2], ...]
        let mut output_shape = vec![indices.size()];
        output_shape.extend_from_slice(&data_shape[1..]);

        let mut output_data = Vec::with_capacity(indices.size() * slice_size);

        // Gather slices
        for &idx_f32 in indices_data {
            let idx = idx_f32 as usize;

            if idx >= data_shape[0] {
                return Err(TinyInferError::InvalidShape(format!(
                    "Index {} out of bounds for dimension {}",
                    idx, data_shape[0]
                )));
            }

            let offset = idx * slice_size;
            output_data.extend_from_slice(&data_data[offset..offset + slice_size]);
        }

        Ok(Tensor::new(
            output_data,
            crate::tensor::Shape::new(output_shape),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    #[test]
    fn test_embedding() {
        // vocab_size = 3, embedding_dim = 2
        let weight_data = vec![
            1.0, 2.0, // embedding for index 0
            3.0, 4.0, // embedding for index 1
            5.0, 6.0, // embedding for index 2
        ];
        let weight = Tensor::new(weight_data, Shape::new(vec![3, 2]));

        // Indices: [0, 2, 1]
        let indices_data = vec![0.0, 2.0, 1.0];
        let indices = Tensor::new(indices_data, Shape::new(vec![3]));

        let emb = Embedding::new(3, 2);
        let output = emb.forward(&[&indices, &weight]).unwrap();

        assert_eq!(output.shape().dims(), &[3, 2]);

        let expected = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0];
        for (i, &val) in output.data().iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gather() {
        // Data: [[1, 2], [3, 4], [5, 6]]
        let data_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data = Tensor::new(data_data, Shape::new(vec![3, 2]));

        // Indices: [2, 0]
        let indices_data = vec![2.0, 0.0];
        let indices = Tensor::new(indices_data, Shape::new(vec![2]));

        let gather = Gather::new(0);
        let output = gather.forward(&[&data, &indices]).unwrap();

        assert_eq!(output.shape().dims(), &[2, 2]);

        let expected = vec![5.0, 6.0, 1.0, 2.0];
        for (i, &val) in output.data().iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-6);
        }
    }
}
