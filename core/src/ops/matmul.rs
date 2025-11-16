use crate::error::{Result, TinyInferError};
use crate::ops::Operator;
use crate::tensor::{Shape, Tensor};
use crate::simd;

/// Matrix multiplication operator
/// Supports 2D matrix multiplication and batched matrix multiplication
pub struct MatMul {
    transpose_a: bool,
    transpose_b: bool,
    use_simd: bool,
}

impl MatMul {
    pub fn new() -> Self {
        Self {
            transpose_a: false,
            transpose_b: false,
            use_simd: simd::is_simd_available(),
        }
    }

    pub fn with_transpose(transpose_a: bool, transpose_b: bool) -> Self {
        Self {
            transpose_a,
            transpose_b,
            use_simd: simd::is_simd_available(),
        }
    }

    /// SIMD-optimized matrix multiplication with tiling
    /// This version uses SIMD dot products for inner loops
    fn matmul_2d_simd(
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

        // Tiled matrix multiplication with SIMD
        for i0 in (0..m).step_by(TILE_SIZE) {
            for j0 in (0..n).step_by(TILE_SIZE) {
                for p0 in (0..k).step_by(TILE_SIZE) {
                    let i_end = (i0 + TILE_SIZE).min(m);
                    let j_end = (j0 + TILE_SIZE).min(n);
                    let p_end = (p0 + TILE_SIZE).min(k);

                    for i in i0..i_end {
                        // Get row from A
                        let a_row = &a[i * k + p0..i * k + p_end];

                        for j in j0..j_end {
                            // Get column from B (note: B is row-major, so we need stride access)
                            let mut b_col = Vec::with_capacity(p_end - p0);
                            for p in p0..p_end {
                                b_col.push(b[p * n + j]);
                            }

                            // SIMD dot product
                            let dot = simd::simd_dot_f32(a_row, &b_col);
                            c[i * n + j] += dot;
                        }
                    }
                }
            }
        }
    }

    /// Standard tiled matrix multiplication (fallback)
    fn matmul_2d_tiled(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        const TILE_SIZE: usize = 32;

        c.fill(0.0);

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

        let output_shape = Shape::new(vec![m, n]);
        let mut output = Tensor::zeros(output_shape);

        // Use SIMD-optimized version if available
        if self.use_simd {
            log::debug!("Using SIMD-optimized MatMul");
            Self::matmul_2d_simd(a.data(), b.data(), output.data_mut(), m, n, k);
        } else {
            log::debug!("Using standard tiled MatMul");
            Self::matmul_2d_tiled(a.data(), b.data(), output.data_mut(), m, n, k);
        }

        Ok(output)
    }
}

/// Gemm (General Matrix Multiplication): Y = alpha * A @ B + beta * C
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

        let matmul = MatMul::new();
        let mut y = matmul.forward(&[inputs[0], inputs[1]])?;

        // Apply alpha using SIMD if available
        if self.alpha != 1.0 {
            if simd::is_simd_available() {
                let mut scaled = vec![0.0; y.size()];
                simd::simd_scale_f32(y.data(), self.alpha, &mut scaled);
                let new_y = Tensor::new(scaled, y.shape().clone());
                y = new_y;
            } else {
                for val in y.data_mut() {
                    *val *= self.alpha;
                }
            }
        }

        // Add beta * C using SIMD if available
        if self.beta != 0.0 {
            let c = inputs[2];
            if y.shape() != c.shape() {
                return Err(TinyInferError::InvalidShape(
                    "Gemm: output and C shapes must match".to_string(),
                ));
            }

            if simd::is_simd_available() && self.beta == 1.0 {
                // Simple addition
                let mut result = vec![0.0; y.size()];
                simd::simd_add_f32(y.data(), c.data(), &mut result);
                y = Tensor::new(result, y.shape().clone());
            } else {
                for (out, &c_val) in y.data_mut().iter_mut().zip(c.data().iter()) {
                    *out += self.beta * c_val;
                }
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

        assert_eq!(y.data()[0], 2.5);
        assert_eq!(y.data()[1], 4.5);
        assert_eq!(y.data()[2], 6.5);
        assert_eq!(y.data()[3], 8.5);
    }
}
