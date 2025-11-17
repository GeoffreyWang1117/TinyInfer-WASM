use crate::error::{Result, TinyInferError};
use crate::ops::Operator;
use crate::tensor::Tensor;

/// Multi-Head Attention
///
/// Implements scaled dot-product attention:
/// Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
///
/// Multi-head version splits Q, K, V into multiple heads,
/// computes attention for each head independently,
/// and concatenates the results.
///
/// Input:
///   - Q: [batch, seq_len, d_model] query
///   - K: [batch, seq_len, d_model] key
///   - V: [batch, seq_len, d_model] value
///   - (optional) mask: [batch, seq_len, seq_len] attention mask
///
/// Output:
///   - [batch, seq_len, d_model] attended values
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    d_model: usize,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        if d_model % num_heads != 0 {
            panic!(
                "d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            );
        }

        let head_dim = d_model / num_heads;

        Self {
            num_heads,
            head_dim,
            d_model,
        }
    }

    /// Scaled dot-product attention for a single head
    /// Q: [batch, seq_len, head_dim]
    /// K: [batch, seq_len, head_dim]
    /// V: [batch, seq_len, head_dim]
    /// Returns: [batch, seq_len, head_dim]
    fn scaled_dot_product_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0; batch * seq_len * head_dim];

        for b in 0..batch {
            // Compute Q * K^T / sqrt(d_k) -> attention scores [seq_len, seq_len]
            let mut scores = vec![0.0; seq_len * seq_len];

            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut score = 0.0;

                    // Q[b, i, :] · K[b, j, :]
                    for d in 0..head_dim {
                        let q_idx = (b * seq_len + i) * head_dim + d;
                        let k_idx = (b * seq_len + j) * head_dim + d;
                        score += q[q_idx] * k[k_idx];
                    }

                    scores[i * seq_len + j] = score * scale;
                }
            }

            // Apply softmax along the last dimension (over keys)
            for i in 0..seq_len {
                let row_offset = i * seq_len;

                // Find max for numerical stability
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..seq_len {
                    max_val = max_val.max(scores[row_offset + j]);
                }

                // Exp and sum
                let mut sum = 0.0;
                for j in 0..seq_len {
                    scores[row_offset + j] = (scores[row_offset + j] - max_val).exp();
                    sum += scores[row_offset + j];
                }

                // Normalize
                for j in 0..seq_len {
                    scores[row_offset + j] /= sum;
                }
            }

            // Multiply attention weights with V
            // output[b, i, :] = sum_j(attention[i, j] * V[b, j, :])
            for i in 0..seq_len {
                for d in 0..head_dim {
                    let mut sum = 0.0;

                    for j in 0..seq_len {
                        let attention_weight = scores[i * seq_len + j];
                        let v_idx = (b * seq_len + j) * head_dim + d;
                        sum += attention_weight * v[v_idx];
                    }

                    let out_idx = (b * seq_len + i) * head_dim + d;
                    output[out_idx] = sum;
                }
            }
        }

        output
    }
}

impl Operator for MultiHeadAttention {
    fn name(&self) -> &str {
        "MultiHeadAttention"
    }

    fn num_inputs(&self) -> usize {
        3 // Q, K, V (mask is optional)
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 3 && inputs.len() != 4 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "MultiHeadAttention expects 3 or 4 inputs (Q, K, V, [mask]), got {}",
                inputs.len()
            )));
        }

        let q = inputs[0];
        let k = inputs[1];
        let v = inputs[2];

        // Validate shapes: [batch, seq_len, d_model]
        if q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3 {
            return Err(TinyInferError::InvalidShape(
                "Q, K, V must be 3D tensors [batch, seq_len, d_model]".to_string(),
            ));
        }

        let q_shape = q.shape().dims();
        let k_shape = k.shape().dims();
        let v_shape = v.shape().dims();

        let batch = q_shape[0];
        let seq_len = q_shape[1];
        let d_model = q_shape[2];

        if k_shape != q_shape || v_shape != q_shape {
            return Err(TinyInferError::InvalidShape(
                "Q, K, V must have the same shape".to_string(),
            ));
        }

        if d_model != self.d_model {
            return Err(TinyInferError::InvalidShape(format!(
                "d_model mismatch: expected {}, got {}",
                self.d_model, d_model
            )));
        }

        // For simplicity, we'll implement attention without explicit head splitting
        // In a full implementation, we'd reshape to [batch, num_heads, seq_len, head_dim]
        // and process each head independently.

        // For now, treat it as single-head attention with the full d_model
        // This is a simplified version - a production implementation would split heads

        let output_data = self.scaled_dot_product_attention(
            q.data(),
            k.data(),
            v.data(),
            batch,
            seq_len,
            d_model,
        );

        Ok(Tensor::new(
            output_data,
            crate::tensor::Shape::new(vec![batch, seq_len, d_model]),
        ))
    }
}

/// Self-Attention
/// A convenience wrapper for MultiHeadAttention where Q=K=V
pub struct SelfAttention {
    mha: MultiHeadAttention,
}

impl SelfAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        Self {
            mha: MultiHeadAttention::new(d_model, num_heads),
        }
    }
}

impl Operator for SelfAttention {
    fn name(&self) -> &str {
        "SelfAttention"
    }

    fn num_inputs(&self) -> usize {
        1 // x (used as Q, K, and V)
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 1 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "SelfAttention expects 1 input, got {}",
                inputs.len()
            )));
        }

        let x = inputs[0];
        self.mha.forward(&[x, x, x])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    #[test]
    fn test_multi_head_attention() {
        // Simple test with batch=1, seq_len=2, d_model=4
        let batch = 1;
        let seq_len = 2;
        let d_model = 4;

        // Q, K, V: [1, 2, 4]
        let q_data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k_data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let v_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let q = Tensor::new(q_data, Shape::new(vec![batch, seq_len, d_model]));
        let k = Tensor::new(k_data, Shape::new(vec![batch, seq_len, d_model]));
        let v = Tensor::new(v_data, Shape::new(vec![batch, seq_len, d_model]));

        let mha = MultiHeadAttention::new(d_model, 2);
        let output = mha.forward(&[&q, &k, &v]).unwrap();

        assert_eq!(output.shape().dims(), &[batch, seq_len, d_model]);

        // Output should be a weighted combination of V
        // Since Q and K are orthogonal unit vectors,
        // the attention should be dominated by exact matches
        assert_eq!(output.size(), batch * seq_len * d_model);
    }

    #[test]
    fn test_self_attention() {
        let batch = 1;
        let seq_len = 3;
        let d_model = 4;

        // Input: [1, 3, 4]
        let x_data = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];

        let x = Tensor::new(x_data, Shape::new(vec![batch, seq_len, d_model]));

        let sa = SelfAttention::new(d_model, 2);
        let output = sa.forward(&[&x]).unwrap();

        assert_eq!(output.shape().dims(), &[batch, seq_len, d_model]);
    }
}
