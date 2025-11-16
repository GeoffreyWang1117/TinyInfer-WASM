use crate::error::{Result, TinyInferError};
use crate::ops::Operator;
use crate::tensor::{Shape, Tensor};

/// Matrix multiplication operator
/// Supports 2D matrix multiplication and batched matrix multiplication
pub struct MatMul {
    transpose_a: bool,
    transpose_b: bool,
}

impl MatMul {
    pub fn new() -> Self {
        Self {
            transpose_a: false,
            transpose_b: false,
        }
    }

    pub fn with_transpose(transpose_a: bool, transpose_b: bool) -> Self {
        Self {
            transpose_a,
            transpose_b,
        }
    }

    /// Basic 2D matrix multiplication: C = A @ B
    /// A: [M, K], B: [K, N] -> C: [M, N]
    fn matmul_2d(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        // Naive implementation - will be optimized later with SIMD and tiling
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    /// Optimized matrix multiplication with loop tiling (cache-friendly)
    fn matmul_2d_tiled(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        const TILE_SIZE: usize = 32;

        // Initialize output to zero
        c.fill(0.0);

        // Tiled matrix multiplication
        for i0 in (0..m).step_by(TILE_SIZE) {
            for j0 in (0..n).step_by(TILE_SIZE) {
                for p0 in (0..k).step_by(TILE_SIZE) {
                    let i_end = (i0 + TILE_SIZE).min(m);
                    let j_end = (j0 + TILE_SIZE).min(n);
                    let p_end = (p0 + TILE_SIZE).min(k);

                    for i in i0..i_end {
                        for p in p0..p_end {
                            let a_val = a[i * k + p];
                            for j in j0..j_end {
                                c[i * n + j] += a_val * b[p * n + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

impl Default for MatMul {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for MatMul {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 2 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "MatMul expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let a = inputs[0];
        let b = inputs[1];

        // For now, only support 2D matrix multiplication
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(TinyInferError::UnsupportedOp(
                "MatMul currently only supports 2D matrices".to_string(),
            ));
        }

        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();

        let (m, k_a) = (a_shape[0], a_shape[1]);
        let (k_b, n) = (b_shape[0], b_shape[1]);

        if k_a != k_b {
            return Err(TinyInferError::InvalidShape(format!(
                "Matrix dimensions incompatible for multiplication: [{}, {}] @ [{}, {}]",
                m, k_a, k_b, n
            )));
        }

        let k = k_a;

        // Create output tensor
        let output_shape = Shape::new(vec![m, n]);
        let mut output = Tensor::zeros(output_shape);

        // Perform matrix multiplication
        Self::matmul_2d_tiled(a.data(), b.data(), output.data_mut(), m, n, k);

        Ok(output)
    }
}

/// Gemm (General Matrix Multiplication): Y = alpha * A @ B + beta * C
/// This is a common operation in neural networks
pub struct Gemm {
    alpha: f32,
    beta: f32,
    transpose_a: bool,
    transpose_b: bool,
}

impl Gemm {
    pub fn new(alpha: f32, beta: f32) -> Self {
        Self {
            alpha,
            beta,
            transpose_a: false,
            transpose_b: false,
        }
    }

    pub fn with_transpose(
        alpha: f32,
        beta: f32,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Self {
        Self {
            alpha,
            beta,
            transpose_a,
            transpose_b,
        }
    }
}

impl Operator for Gemm {
    fn name(&self) -> &str {
        "Gemm"
    }

    fn num_inputs(&self) -> usize {
        3
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 3 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "Gemm expects 3 inputs (A, B, C), got {}",
                inputs.len()
            )));
        }

        // For simplicity, use MatMul and Add
        // In a full implementation, this would be optimized as a single kernel
        let matmul = MatMul::new();
        let mut y = matmul.forward(&[inputs[0], inputs[1]])?;

        // Apply alpha
        if self.alpha != 1.0 {
            for val in y.data_mut() {
                *val *= self.alpha;
            }
        }

        // Add beta * C
        if self.beta != 0.0 {
            let c = inputs[2];
            if y.shape() != c.shape() {
                return Err(TinyInferError::InvalidShape(
                    "Gemm: output and C shapes must match".to_string(),
                ));
            }

            for (out, &c_val) in y.data_mut().iter_mut().zip(c.data().iter()) {
                *out += self.beta * c_val;
            }
        }

        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2x2() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];

        let a = Tensor::new(a_data, Shape::new(vec![2, 2]));
        let b = Tensor::new(b_data, Shape::new(vec![2, 2]));

        let matmul = MatMul::new();
        let c = matmul.forward(&[&a, &b]).unwrap();

        // Expected: [[19, 22], [43, 50]]
        assert_eq!(c.data()[0], 19.0);
        assert_eq!(c.data()[1], 22.0);
        assert_eq!(c.data()[2], 43.0);
        assert_eq!(c.data()[3], 50.0);
    }

    #[test]
    fn test_matmul_non_square() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a = Tensor::new(a_data, Shape::new(vec![2, 3]));
        let b = Tensor::new(b_data, Shape::new(vec![3, 2]));

        let matmul = MatMul::new();
        let c = matmul.forward(&[&a, &b]).unwrap();

        assert_eq!(c.shape().dims(), &[2, 2]);

        // Expected: [[22, 28], [49, 64]]
        assert_eq!(c.data()[0], 22.0);
        assert_eq!(c.data()[1], 28.0);
        assert_eq!(c.data()[2], 49.0);
        assert_eq!(c.data()[3], 64.0);
    }

    #[test]
    fn test_gemm() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));
        let c = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], Shape::new(vec![2, 2]));

        let gemm = Gemm::new(2.0, 0.5);
        let y = gemm.forward(&[&a, &b, &c]).unwrap();

        // Y = 2 * (A @ B) + 0.5 * C
        // A @ B = [[1, 2], [3, 4]]
        // Y = 2 * [[1, 2], [3, 4]] + 0.5 * [[1, 1], [1, 1]]
        //   = [[2.5, 4.5], [6.5, 8.5]]
        assert_eq!(y.data()[0], 2.5);
        assert_eq!(y.data()[1], 4.5);
        assert_eq!(y.data()[2], 6.5);
        assert_eq!(y.data()[3], 8.5);
    }
}
