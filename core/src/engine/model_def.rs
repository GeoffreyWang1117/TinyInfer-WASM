use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model serialization format (JSON-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDef {
    pub version: String,
    pub name: String,
    pub graph: GraphDef,
    pub weights: HashMap<String, WeightDef>,
}

/// Graph definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDef {
    pub nodes: Vec<NodeDef>,
    pub edges: Vec<EdgeDef>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// Node definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDef {
    pub id: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, AttributeValue>,
}

/// Edge definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeDef {
    pub from: String,
    pub to: String,
}

/// Weight definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightDef {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub data: Vec<f32>,  // For now, only support f32
}

/// Attribute value (for operator parameters)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    Int(i64),
    Float(f64),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Strings(Vec<String>),
}

impl AttributeValue {
    pub fn as_int(&self) -> Option<i64> {
        match self {
            AttributeValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            AttributeValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            AttributeValue::String(v) => Some(v.as_str()),
            _ => None,
        }
    }

    pub fn as_ints(&self) -> Option<&[i64]> {
        match self {
            AttributeValue::Ints(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn as_floats(&self) -> Option<&[f64]> {
        match self {
            AttributeValue::Floats(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribute_value() {
        let int_val = AttributeValue::Int(42);
        assert_eq!(int_val.as_int(), Some(42));

        let float_val = AttributeValue::Float(3.14);
        assert_eq!(float_val.as_float(), Some(3.14));

        let str_val = AttributeValue::String("test".to_string());
        assert_eq!(str_val.as_string(), Some("test"));
    }

    #[test]
    fn test_model_serialization() {
        let model = ModelDef {
            version: "1.0".to_string(),
            name: "test_model".to_string(),
            graph: GraphDef {
                nodes: vec![],
                edges: vec![],
                inputs: vec!["input".to_string()],
                outputs: vec!["output".to_string()],
            },
            weights: HashMap::new(),
        };

        let json = serde_json::to_string(&model).unwrap();
        let deserialized: ModelDef = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.version, "1.0");
        assert_eq!(deserialized.name, "test_model");
    }
}
