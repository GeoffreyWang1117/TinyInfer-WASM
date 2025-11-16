use crate::error::{Result, TinyInferError};
use crate::ops::Operator;
use crate::tensor::{Shape, Tensor};

/// Max pooling operator
/// Input: [N, C, H, W]
/// Output: [N, C, H_out, W_out]
pub struct MaxPool2D {
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
}

impl MaxPool2D {
    pub fn new(kernel_size: [usize; 2]) -> Self {
        Self {
            kernel_size,
            stride: kernel_size, // Default stride = kernel_size
            padding: [0, 0],
        }
    }

    pub fn with_params(
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    fn calc_output_size(input_size: usize, kernel: usize, padding: usize, stride: usize) -> usize {
        (input_size + 2 * padding - kernel) / stride + 1
    }
}

impl Operator for MaxPool2D {
    fn name(&self) -> &str {
        "MaxPool2D"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 1 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "MaxPool2D expects 1 input, got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];

        if input.ndim() != 4 {
            return Err(TinyInferError::InvalidShape(
                "MaxPool2D input must be 4D [N, C, H, W]".to_string(),
            ));
        }

        let shape = input.shape().dims();
        let (batch, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);

        let out_h = Self::calc_output_size(
            in_h,
            self.kernel_size[0],
            self.padding[0],
            self.stride[0],
        );
        let out_w = Self::calc_output_size(
            in_w,
            self.kernel_size[1],
            self.padding[1],
            self.stride[1],
        );

        let output_shape = Shape::new(vec![batch, channels, out_h, out_w]);
        let mut output = Tensor::full(output_shape, f32::NEG_INFINITY);

        let input_data = input.data();
        let output_data = output.data_mut();

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;

                        for kh in 0..self.kernel_size[0] {
                            for kw in 0..self.kernel_size[1] {
                                let h = oh * self.stride[0] + kh;
                                let w = ow * self.stride[1] + kw;

                                if h >= self.padding[0]
                                    && h < in_h + self.padding[0]
                                    && w >= self.padding[1]
                                    && w < in_w + self.padding[1]
                                {
                                    let h_im = h - self.padding[0];
                                    let w_im = w - self.padding[1];

                                    let idx = ((b * channels + c) * in_h + h_im) * in_w + w_im;
                                    max_val = max_val.max(input_data[idx]);
                                }
                            }
                        }

                        let out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                        output_data[out_idx] = max_val;
                    }
                }
            }
        }

        Ok(output)
    }
}

/// Average pooling operator
pub struct AvgPool2D {
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
}

impl AvgPool2D {
    pub fn new(kernel_size: [usize; 2]) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: [0, 0],
        }
    }

    pub fn with_params(
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    fn calc_output_size(input_size: usize, kernel: usize, padding: usize, stride: usize) -> usize {
        (input_size + 2 * padding - kernel) / stride + 1
    }
}

impl Operator for AvgPool2D {
    fn name(&self) -> &str {
        "AvgPool2D"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 1 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "AvgPool2D expects 1 input, got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];

        if input.ndim() != 4 {
            return Err(TinyInferError::InvalidShape(
                "AvgPool2D input must be 4D [N, C, H, W]".to_string(),
            ));
        }

        let shape = input.shape().dims();
        let (batch, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);

        let out_h = Self::calc_output_size(
            in_h,
            self.kernel_size[0],
            self.padding[0],
            self.stride[0],
        );
        let out_w = Self::calc_output_size(
            in_w,
            self.kernel_size[1],
            self.padding[1],
            self.stride[1],
        );

        let output_shape = Shape::new(vec![batch, channels, out_h, out_w]);
        let mut output = Tensor::zeros(output_shape);

        let input_data = input.data();
        let output_data = output.data_mut();

        let pool_size = (self.kernel_size[0] * self.kernel_size[1]) as f32;

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;

                        for kh in 0..self.kernel_size[0] {
                            for kw in 0..self.kernel_size[1] {
                                let h = oh * self.stride[0] + kh;
                                let w = ow * self.stride[1] + kw;

                                if h >= self.padding[0]
                                    && h < in_h + self.padding[0]
                                    && w >= self.padding[1]
                                    && w < in_w + self.padding[1]
                                {
                                    let h_im = h - self.padding[0];
                                    let w_im = w - self.padding[1];

                                    let idx = ((b * channels + c) * in_h + h_im) * in_w + w_im;
                                    sum += input_data[idx];
                                }
                            }
                        }

                        let out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                        output_data[out_idx] = sum / pool_size;
                    }
                }
            }
        }

        Ok(output)
    }
}

/// Global average pooling - reduces spatial dimensions to 1x1
pub struct GlobalAvgPool2D;

impl GlobalAvgPool2D {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GlobalAvgPool2D {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for GlobalAvgPool2D {
    fn name(&self) -> &str {
        "GlobalAvgPool2D"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 1 {
            return Err(TinyInferError::UnsupportedOp(format!(
                "GlobalAvgPool2D expects 1 input, got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];

        if input.ndim() != 4 {
            return Err(TinyInferError::InvalidShape(
                "GlobalAvgPool2D input must be 4D [N, C, H, W]".to_string(),
            ));
        }

        let shape = input.shape().dims();
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

        let output_shape = Shape::new(vec![batch, channels, 1, 1]);
        let mut output = Tensor::zeros(output_shape);

        let input_data = input.data();
        let output_data = output.data_mut();

        let spatial_size = (height * width) as f32;

        for b in 0..batch {
            for c in 0..channels {
                let mut sum = 0.0;

                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        sum += input_data[idx];
                    }
                }

                let out_idx = b * channels + c;
                output_data[out_idx] = sum / spatial_size;
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxpool2d() {
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = Tensor::new(input_data, Shape::new(vec![1, 1, 4, 4]));

        let pool = MaxPool2D::new([2, 2]);
        let output = pool.forward(&[&input]).unwrap();

        assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
        assert_eq!(output.data(), &[6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_avgpool2d() {
        let input = Tensor::ones(Shape::new(vec![1, 1, 4, 4]));

        let pool = AvgPool2D::new([2, 2]);
        let output = pool.forward(&[&input]).unwrap();

        assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
        // Average of all 1s should be 1
        for &val in output.data() {
            assert!((val - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_global_avgpool() {
        let input = Tensor::full(Shape::new(vec![1, 2, 4, 4]), 2.0);

        let pool = GlobalAvgPool2D::new();
        let output = pool.forward(&[&input]).unwrap();

        assert_eq!(output.shape().dims(), &[1, 2, 1, 1]);
        assert_eq!(output.data(), &[2.0, 2.0]);
    }
}
