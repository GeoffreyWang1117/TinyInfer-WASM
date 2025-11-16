use serde::{Deserialize, Serialize};

/// Data type for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    /// 32-bit floating point
    Float32,

    /// 16-bit floating point (half precision)
    Float16,

    /// 8-bit signed integer
    Int8,

    /// 8-bit unsigned integer
    UInt8,

    /// 32-bit signed integer
    Int32,

    /// 64-bit signed integer
    Int64,

    /// Boolean
    Bool,
}

impl DataType {
    /// Get the size in bytes for this data type
    pub fn size(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float16 => 2,
            DataType::Int8 => 1,
            DataType::UInt8 => 1,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::Bool => 1,
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &str {
        match self {
            DataType::Float32 => "float32",
            DataType::Float16 => "float16",
            DataType::Int8 => "int8",
            DataType::UInt8 => "uint8",
            DataType::Int32 => "int32",
            DataType::Int64 => "int64",
            DataType::Bool => "bool",
        }
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, DataType::Float32 | DataType::Float16)
    }

    /// Check if this is an integer type
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            DataType::Int8 | DataType::UInt8 | DataType::Int32 | DataType::Int64
        )
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DataType::Float32.size(), 4);
        assert_eq!(DataType::Float16.size(), 2);
        assert_eq!(DataType::Int8.size(), 1);
    }

    #[test]
    fn test_dtype_checks() {
        assert!(DataType::Float32.is_float());
        assert!(!DataType::Float32.is_int());
        assert!(DataType::Int32.is_int());
        assert!(!DataType::Int32.is_float());
    }
}
