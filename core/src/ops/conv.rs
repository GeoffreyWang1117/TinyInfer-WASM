use crate::error::{Result, TinyInferError};
use crate::ops::Operator;
use crate::tensor::{Shape, Tensor};

/// 2D Convolution operator
/// Input: [N, C_in, H, W] (NCHW format)
/// Weight: [C_out, C_in, kernel_h, kernel_w]
/// Output: [N, C_out, H_out, W_out]
pub struct Conv2D {
    padding: [usize; 2],  // [pad_h, pad_w]
    stride: [usize; 2],   // [stride_h, stride_w]
    dilation: [usize; 2], // [dilation_h, dilation_w]
    groups: usize,
}

impl Conv2D {
    pub fn new() -> Self {
        Self {
            padding: [0, 0],
            stride: [1, 1],
            dilation: [1, 1],
            groups: 1,
        }
    }

    pub fn with_params(
        padding: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Self {
        Self {
            padding,
            stride,
            dilation,
            groups,
        }
    }

    /// Calculate output spatial dimensions
    fn calc_output_size(
        input_size: usize,
        kernel_size: usize,
        padding: usize,
        stride: usize,
        dilation: usize,
    ) -> usize {
        let effective_kernel = (kernel_size - 1) * dilation + 1;
        (input_size + 2 * padding - effective_kernel) / stride + 1
    }

    /// Im2col transformation: convert image patches to columns
    /// This is a standard technique to convert convolution to matrix multiplication
    #[allow(clippy::too_many_arguments)]
    fn im2col(
        input: &[f32],
        output: &mut [f32],
        channels: usize,
        height: usize,
        width: usize,
        kernel_h: usize,
        kernel_w: usize,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
    ) {
        let output_h = Self::calc_output_size(height, kernel_h, pad_h, stride_h, dilation_h);
        let output_w = Self::calc_output_size(width, kernel_w, pad_w, stride_w, dilation_w);

        let mut col_idx = 0;

        for c in 0..channels {
            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    let h_offset = kh * dilation_h;
                    let w_offset = kw * dilation_w;

                    for oh in 0..output_h {
                        for ow in 0..output_w {
                            let h = oh * stride_h + h_offset;
                            let w = ow * stride_w + w_offset;

                            // Check if position is within bounds (considering padding)
                            if h >= pad_h && h < height + pad_h && w >= pad_w && w < width + pad_w
                            {
                                let h_im = h - pad_h;
                                let w_im = w - pad_w;
                                let im_idx = (c * height + h_im) * width + w_im;
                                output[col_idx] = input[im_idx];
                            } else {
                                output[col_idx] = 0.0; // Padding
                            }

                            col_idx += 1;
                        }
                    }
                }
            }
        }
    }

    /// Basic convolution implementation using im2col + GEMM
    #[allow(clippy::too_many_arguments)]
    fn conv2d_im2col(
        input: &[f32],
        weight: &[f32],
        output: &mut [f32],
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
    ) {
        let out_h = Self::calc_output_size(in_h, kernel_h, pad_h, stride_h, dilation_h);
        let out_w = Self::calc_output_size(in_w, kernel_w, pad_w, stride_w, dilation_w);

        let col_size = in_channels * kernel_h * kernel_w * out_h * out_w;
        let mut col_buffer = vec![0.0; col_size];

        for b in 0..batch {
            let input_offset = b * in_channels * in_h * in_w;
            let output_offset = b * out_channels * out_h * out_w;

            // Im2col transformation
            Self::im2col(
                &input[input_offset..],
                &mut col_buffer,
                in_channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
            );

            // Matrix multiplication: weight @ col_buffer
            // weight: [out_channels, in_channels * kernel_h * kernel_w]
            // col_buffer: [in_channels * kernel_h * kernel_w, out_h * out_w]
            // output: [out_channels, out_h * out_w]

            let m = out_channels;
            let k = in_channels * kernel_h * kernel_w;
            let n = out_h * out_w;

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for p in 0..k {
                        sum += weight[i * k + p] * col_buffer[p * n + j];
                    }
                    output[output_offset + i * n + j] = sum;
                }
            }
        }
    }
}

impl Default for Conv2D {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for Conv2D {
    fn name(&self) -> &str {
        "Conv2D"
    }

    fn num_inputs(&self) -> usize {
        2 // input and weight (bias is optional, handled separately)
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() < 2 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "Conv2D expects at least 2 inputs (input, weight), got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];
        let weight = inputs[1];

        // Validate input dimensions
        if input.ndim() != 4 {
            return Err(TinyInferError::InvalidShape(format!(
                "Conv2D input must be 4D [N, C, H, W], got {}D",
                input.ndim()
            )));
        }

        if weight.ndim() != 4 {
            return Err(TinyInferError::InvalidShape(format!(
                "Conv2D weight must be 4D [out_channels, in_channels, kH, kW], got {}D",
                weight.ndim()
            )));
        }

        let input_shape = input.shape().dims();
        let weight_shape = weight.shape().dims();

        let batch = input_shape[0];
        let in_channels = input_shape[1];
        let in_h = input_shape[2];
        let in_w = input_shape[3];

        let out_channels = weight_shape[0];
        let weight_in_channels = weight_shape[1];
        let kernel_h = weight_shape[2];
        let kernel_w = weight_shape[3];

        // Validate channel dimensions
        if in_channels != weight_in_channels * self.groups {
            return Err(TinyInferError::InvalidShape(format!(
                "Conv2D: input channels ({}) must match weight input channels ({}) * groups ({})",
                in_channels, weight_in_channels, self.groups
            )));
        }

        // Calculate output dimensions
        let out_h = Self::calc_output_size(
            in_h,
            kernel_h,
            self.padding[0],
            self.stride[0],
            self.dilation[0],
        );
        let out_w = Self::calc_output_size(
            in_w,
            kernel_w,
            self.padding[1],
            self.stride[1],
            self.dilation[1],
        );

        // Create output tensor
        let output_shape = Shape::new(vec![batch, out_channels, out_h, out_w]);
        let mut output = Tensor::zeros(output_shape);

        // Perform convolution
        Self::conv2d_im2col(
            input.data(),
            weight.data(),
            output.data_mut(),
            batch,
            in_channels,
            out_channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            self.padding[0],
            self.padding[1],
            self.stride[0],
            self.stride[1],
            self.dilation[0],
            self.dilation[1],
        );

        // Add bias if provided
        if inputs.len() >= 3 {
            let bias = inputs[2];
            if bias.size() != out_channels {
                return Err(TinyInferError::InvalidShape(
                    "Conv2D: bias size must match output channels".to_string(),
                ));
            }

            let bias_data = bias.data();
            let output_data = output.data_mut();

            for b in 0..batch {
                for c in 0..out_channels {
                    let offset = (b * out_channels + c) * out_h * out_w;
                    for i in 0..(out_h * out_w) {
                        output_data[offset + i] += bias_data[c];
                    }
                }
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_simple() {
        // Simple 3x3 convolution with 1x1 kernel (should be identity)
        let input_data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let input = Tensor::new(input_data, Shape::new(vec![1, 1, 3, 3]));

        let weight_data = vec![1.0]; // 1x1 kernel with value 1
        let weight = Tensor::new(weight_data, Shape::new(vec![1, 1, 1, 1]));

        let conv = Conv2D::new();
        let output = conv.forward(&[&input, &weight]).unwrap();

        assert_eq!(output.shape().dims(), &[1, 1, 3, 3]);

        // With 1x1 kernel and weight=1, output should equal input
        for (out_val, in_val) in output.data().iter().zip(input.data().iter()) {
            assert!((out_val - in_val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_conv2d_with_padding() {
        let input = Tensor::ones(Shape::new(vec![1, 1, 4, 4]));
        let weight = Tensor::ones(Shape::new(vec![1, 1, 3, 3]));

        let conv = Conv2D::with_params([1, 1], [1, 1], [1, 1], 1);
        let output = conv.forward(&[&input, &weight]).unwrap();

        assert_eq!(output.shape().dims(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_output_size_calculation() {
        // Test output size calculation
        let out_size = Conv2D::calc_output_size(224, 7, 3, 2, 1);
        assert_eq!(out_size, 112); // (224 + 2*3 - 7) / 2 + 1 = 112
    }
}
