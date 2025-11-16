use crate::error::{Result, TinyInferError};
use serde::{Deserialize, Serialize};

/// Tensor shape representation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Get the dimensions
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Calculate total number of elements
    pub fn size(&self) -> usize {
        if self.dims.is_empty() {
            0
        } else {
            self.dims.iter().product()
        }
    }

    /// Get dimension at index
    pub fn dim(&self, index: usize) -> Result<usize> {
        self.dims.get(index).copied().ok_or_else(|| {
            TinyInferError::InvalidShape(format!(
                "Dimension index {} out of bounds for shape with {} dimensions",
                index,
                self.dims.len()
            ))
        })
    }

    /// Check if two shapes are compatible for element-wise operations
    pub fn is_compatible(&self, other: &Shape) -> bool {
        if self.ndim() != other.ndim() {
            return false;
        }

        self.dims.iter().zip(other.dims.iter()).all(|(a, b)| {
            a == b || *a == 1 || *b == 1
        })
    }

    /// Check if shape is empty (scalar)
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty() || (self.dims.len() == 1 && self.dims[0] == 1)
    }

    /// Broadcast two shapes
    pub fn broadcast(&self, other: &Shape) -> Result<Shape> {
        let max_ndim = self.ndim().max(other.ndim());
        let mut result_dims = Vec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let dim1 = if i < self.ndim() {
                self.dims[self.ndim() - 1 - i]
            } else {
                1
            };

            let dim2 = if i < other.ndim() {
                other.dims[other.ndim() - 1 - i]
            } else {
                1
            };

            if dim1 == dim2 || dim1 == 1 || dim2 == 1 {
                result_dims.push(dim1.max(dim2));
            } else {
                return Err(TinyInferError::InvalidShape(format!(
                    "Cannot broadcast shapes {:?} and {:?}",
                    self.dims, other.dims
                )));
            }
        }

        result_dims.reverse();
        Ok(Shape::new(result_dims))
    }

    /// Calculate strides for C-contiguous layout
    pub fn strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.ndim()];
        for i in (0..self.ndim().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Convert multi-dimensional index to flat index
    pub fn index_to_offset(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.ndim() {
            return Err(TinyInferError::InvalidShape(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }

        let strides = self.strides();
        let mut offset = 0;

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.dims[i] {
                return Err(TinyInferError::InvalidShape(format!(
                    "Index {} at dimension {} exceeds dimension size {}",
                    idx, i, self.dims[i]
                )));
            }
            offset += idx * strides[i];
        }

        Ok(offset)
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_size() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.size(), 24);
    }

    #[test]
    fn test_shape_broadcast() {
        let shape1 = Shape::new(vec![3, 1, 4]);
        let shape2 = Shape::new(vec![1, 5, 4]);
        let result = shape1.broadcast(&shape2).unwrap();
        assert_eq!(result.dims(), &[3, 5, 4]);
    }

    #[test]
    fn test_shape_strides() {
        let shape = Shape::new(vec![2, 3, 4]);
        let strides = shape.strides();
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_index_to_offset() {
        let shape = Shape::new(vec![2, 3, 4]);
        let offset = shape.index_to_offset(&[1, 2, 3]).unwrap();
        assert_eq!(offset, 23); // 1*12 + 2*4 + 3*1 = 23
    }
}
