use crate::error::{Result, TinyInferError};
use crate::ops::Operator;
use crate::tensor::Tensor;
use std::f32::consts::PI;

/// Sinusoidal Positional Encoding
///
/// Adds positional information to embeddings using sine and cosine functions.
/// This is the standard positional encoding used in the original Transformer paper.
///
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
///
/// Input:
///   - embeddings: [batch, seq_len, d_model] embeddings tensor
///
/// Output:
///   - [batch, seq_len, d_model] embeddings with positional encoding added
pub struct PositionalEncoding {
    d_model: usize,
    max_len: usize,
    dropout: f32,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize) -> Self {
        Self {
            d_model,
            max_len,
            dropout: 0.0,
        }
    }

    pub fn with_dropout(d_model: usize, max_len: usize, dropout: f32) -> Self {
        Self {
            d_model,
            max_len,
            dropout,
        }
    }

    /// Generate positional encoding matrix
    /// Returns: [max_len, d_model]
    fn generate_positional_encoding(&self) -> Vec<f32> {
        let mut pe = vec![0.0; self.max_len * self.d_model];

        for pos in 0..self.max_len {
            for i in 0..self.d_model {
                let position = pos as f32;
                let div_term = (10000.0_f32).powf((2 * (i / 2)) as f32 / self.d_model as f32);

                let idx = pos * self.d_model + i;

                if i % 2 == 0 {
                    // Even indices: sin
                    pe[idx] = (position / div_term).sin();
                } else {
                    // Odd indices: cos
                    pe[idx] = (position / div_term).cos();
                }
            }
        }

        pe
    }
}

impl Operator for PositionalEncoding {
    fn name(&self) -> &str {
        "PositionalEncoding"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 1 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "PositionalEncoding expects 1 input (embeddings), got {}",
                inputs.len()
            )));
        }

        let embeddings = inputs[0];

        // Validate shape: [batch, seq_len, d_model]
        if embeddings.ndim() != 3 {
            return Err(TinyInferError::InvalidShape(
                "Embeddings must be 3D [batch, seq_len, d_model]".to_string(),
            ));
        }

        let shape = embeddings.shape().dims();
        let batch = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];

        if d_model != self.d_model {
            return Err(TinyInferError::InvalidShape(format!(
                "d_model mismatch: expected {}, got {}",
                self.d_model, d_model
            )));
        }

        if seq_len > self.max_len {
            return Err(TinyInferError::InvalidShape(format!(
                "Sequence length {} exceeds max_len {}",
                seq_len, self.max_len
            )));
        }

        // Generate positional encoding
        let pe = self.generate_positional_encoding();

        // Add positional encoding to embeddings
        let mut output = embeddings.clone_tensor();
        let output_data = output.data_mut();
        let input_data = embeddings.data();

        for b in 0..batch {
            for pos in 0..seq_len {
                for i in 0..d_model {
                    let output_idx = (b * seq_len + pos) * d_model + i;
                    let pe_idx = pos * d_model + i;
                    output_data[output_idx] = input_data[output_idx] + pe[pe_idx];
                }
            }
        }

        Ok(output)
    }
}

/// Learnable Positional Embedding
///
/// Uses learnable position embeddings instead of fixed sinusoidal encoding.
/// This is used in models like BERT.
///
/// Input:
///   - embeddings: [batch, seq_len, d_model]
///   - pos_embeddings: [max_len, d_model] learnable position embeddings
///
/// Output:
///   - [batch, seq_len, d_model] embeddings with position added
pub struct PositionalEmbedding {
    max_len: usize,
    d_model: usize,
}

impl PositionalEmbedding {
    pub fn new(max_len: usize, d_model: usize) -> Self {
        Self { max_len, d_model }
    }
}

impl Operator for PositionalEmbedding {
    fn name(&self) -> &str {
        "PositionalEmbedding"
    }

    fn num_inputs(&self) -> usize {
        2 // embeddings, pos_embeddings
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 2 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "PositionalEmbedding expects 2 inputs (embeddings, pos_embeddings), got {}",
                inputs.len()
            )));
        }

        let embeddings = inputs[0];
        let pos_embeddings = inputs[1];

        // Validate shapes
        if embeddings.ndim() != 3 {
            return Err(TinyInferError::InvalidShape(
                "Embeddings must be 3D [batch, seq_len, d_model]".to_string(),
            ));
        }

        if pos_embeddings.ndim() != 2 {
            return Err(TinyInferError::InvalidShape(
                "Position embeddings must be 2D [max_len, d_model]".to_string(),
            ));
        }

        let emb_shape = embeddings.shape().dims();
        let pos_shape = pos_embeddings.shape().dims();

        let batch = emb_shape[0];
        let seq_len = emb_shape[1];
        let d_model = emb_shape[2];

        if pos_shape[0] != self.max_len || pos_shape[1] != self.d_model {
            return Err(TinyInferError::InvalidShape(format!(
                "Position embeddings shape mismatch: expected [{}, {}], got [{}, {}]",
                self.max_len, self.d_model, pos_shape[0], pos_shape[1]
            )));
        }

        if seq_len > self.max_len {
            return Err(TinyInferError::InvalidShape(format!(
                "Sequence length {} exceeds max_len {}",
                seq_len, self.max_len
            )));
        }

        // Add position embeddings
        let mut output = embeddings.clone_tensor();
        let output_data = output.data_mut();
        let input_data = embeddings.data();
        let pos_data = pos_embeddings.data();

        for b in 0..batch {
            for pos in 0..seq_len {
                for i in 0..d_model {
                    let output_idx = (b * seq_len + pos) * d_model + i;
                    let pos_idx = pos * d_model + i;
                    output_data[output_idx] = input_data[output_idx] + pos_data[pos_idx];
                }
            }
        }

        Ok(output)
    }
}

/// Rotary Position Embedding (RoPE)
///
/// Applies rotary position embeddings as used in models like GPT-Neo, LLaMA.
/// RoPE rotates query and key vectors based on their position.
///
/// This is a simplified implementation for demonstration.
pub struct RotaryPositionEmbedding {
    dim: usize,
    max_len: usize,
    base: f32,
}

impl RotaryPositionEmbedding {
    pub fn new(dim: usize, max_len: usize) -> Self {
        Self {
            dim,
            max_len,
            base: 10000.0,
        }
    }

    pub fn with_base(dim: usize, max_len: usize, base: f32) -> Self {
        Self { dim, max_len, base }
    }

    /// Compute frequency for each dimension
    fn compute_freqs(&self) -> Vec<f32> {
        let mut freqs = Vec::with_capacity(self.dim / 2);
        for i in 0..(self.dim / 2) {
            let freq = 1.0 / self.base.powf(2.0 * i as f32 / self.dim as f32);
            freqs.push(freq);
        }
        freqs
    }
}

impl Operator for RotaryPositionEmbedding {
    fn name(&self) -> &str {
        "RotaryPositionEmbedding"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 1 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "RotaryPositionEmbedding expects 1 input, got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];

        // Expecting [batch, seq_len, dim]
        if input.ndim() != 3 {
            return Err(TinyInferError::InvalidShape(
                "Input must be 3D [batch, seq_len, dim]".to_string(),
            ));
        }

        let shape = input.shape().dims();
        let batch = shape[0];
        let seq_len = shape[1];
        let dim = shape[2];

        if dim != self.dim {
            return Err(TinyInferError::InvalidShape(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dim, dim
            )));
        }

        if seq_len > self.max_len {
            return Err(TinyInferError::InvalidShape(format!(
                "Sequence length {} exceeds max_len {}",
                seq_len, self.max_len
            )));
        }

        let freqs = self.compute_freqs();
        let mut output = input.clone_tensor();
        let input_data = input.data();
        let output_data = output.data_mut();

        // Apply RoPE
        for b in 0..batch {
            for pos in 0..seq_len {
                for i in 0..(dim / 2) {
                    let freq = freqs[i];
                    let angle = pos as f32 * freq;

                    let cos_val = angle.cos();
                    let sin_val = angle.sin();

                    // Indices for the pair of values to rotate
                    let idx1 = (b * seq_len + pos) * dim + 2 * i;
                    let idx2 = (b * seq_len + pos) * dim + 2 * i + 1;

                    let x1 = input_data[idx1];
                    let x2 = input_data[idx2];

                    // Apply rotation
                    output_data[idx1] = x1 * cos_val - x2 * sin_val;
                    output_data[idx2] = x1 * sin_val + x2 * cos_val;
                }
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
    fn test_positional_encoding() {
        let batch = 1;
        let seq_len = 4;
        let d_model = 8;

        // Create random embeddings
        let emb_data = vec![0.1; batch * seq_len * d_model];
        let embeddings = Tensor::new(emb_data, Shape::new(vec![batch, seq_len, d_model]));

        let pe = PositionalEncoding::new(d_model, 10);
        let output = pe.forward(&[&embeddings]).unwrap();

        assert_eq!(output.shape().dims(), &[batch, seq_len, d_model]);

        // Values should be different from input (due to added positional encoding)
        assert_ne!(output.data()[0], 0.1);
    }

    #[test]
    fn test_positional_embedding() {
        let batch = 2;
        let seq_len = 3;
        let d_model = 4;
        let max_len = 10;

        let emb_data = vec![1.0; batch * seq_len * d_model];
        let embeddings = Tensor::new(emb_data, Shape::new(vec![batch, seq_len, d_model]));

        let pos_data = vec![0.1; max_len * d_model];
        let pos_embeddings = Tensor::new(pos_data, Shape::new(vec![max_len, d_model]));

        let pe = PositionalEmbedding::new(max_len, d_model);
        let output = pe.forward(&[&embeddings, &pos_embeddings]).unwrap();

        assert_eq!(output.shape().dims(), &[batch, seq_len, d_model]);

        // First value should be 1.0 + 0.1 = 1.1
        assert!((output.data()[0] - 1.1).abs() < 1e-6);
    }

    #[test]
    fn test_rotary_position_embedding() {
        let batch = 1;
        let seq_len = 2;
        let dim = 4;

        let input_data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let input = Tensor::new(input_data, Shape::new(vec![batch, seq_len, dim]));

        let rope = RotaryPositionEmbedding::new(dim, 10);
        let output = rope.forward(&[&input]).unwrap();

        assert_eq!(output.shape().dims(), &[batch, seq_len, dim]);

        // Values should be rotated
        assert_ne!(output.data()[0], 1.0);
    }
}
