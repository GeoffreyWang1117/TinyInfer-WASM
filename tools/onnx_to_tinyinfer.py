#!/usr/bin/env python3
"""
ONNX to TinyInfer Model Converter

Converts ONNX models to TinyInfer's JSON format.

Usage:
    python onnx_to_tinyinfer.py <input.onnx> <output.json>

Requirements:
    pip install onnx numpy
"""

import json
import sys
import struct
from pathlib import Path
from typing import Dict, List, Any

try:
    import onnx
    from onnx import numpy_helper
    import numpy as np
except ImportError:
    print("Error: Required packages not found. Install with:")
    print("  pip install onnx numpy")
    sys.exit(1)


# Mapping from ONNX op types to TinyInfer op types
OP_TYPE_MAP = {
    'Relu': 'ReLU',
    'Sigmoid': 'Sigmoid',
    'Tanh': 'Tanh',
    'Softmax': 'Softmax',
    'MatMul': 'MatMul',
    'Gemm': 'Gemm',
    'Conv': 'Conv2D',
    'MaxPool': 'MaxPool2D',
    'AveragePool': 'AvgPool2D',
    'GlobalAveragePool': 'GlobalAvgPool2D',
    'BatchNormalization': 'BatchNorm2D',
    'Add': 'Add',
    'Sub': 'Sub',
    'Mul': 'Mul',
    'Div': 'Div',
}


def convert_dtype(onnx_dtype: int) -> str:
    """Convert ONNX data type to TinyInfer dtype string"""
    dtype_map = {
        1: 'float32',  # FLOAT
        2: 'uint8',    # UINT8
        3: 'int8',     # INT8
        6: 'int32',    # INT32
        7: 'int64',    # INT64
    }
    return dtype_map.get(onnx_dtype, 'float32')


def convert_tensor(tensor: onnx.TensorProto) -> Dict[str, Any]:
    """Convert ONNX tensor to TinyInfer weight definition"""
    # Convert to numpy array
    np_array = numpy_helper.to_array(tensor)

    # For now, convert everything to float32
    if np_array.dtype != np.float32:
        np_array = np_array.astype(np.float32)

    return {
        'shape': list(np_array.shape),
        'dtype': 'float32',
        'data': np_array.flatten().tolist()
    }


def get_attribute_value(attr: onnx.AttributeProto) -> Any:
    """Extract value from ONNX attribute"""
    if attr.type == onnx.AttributeProto.INT:
        return attr.i
    elif attr.type == onnx.AttributeProto.FLOAT:
        return attr.f
    elif attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode('utf-8')
    elif attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    elif attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == onnx.AttributeProto.STRINGS:
        return [s.decode('utf-8') for s in attr.strings]
    else:
        return None


def convert_node(node: onnx.NodeProto, idx: int) -> Dict[str, Any]:
    """Convert ONNX node to TinyInfer node definition"""
    # Map op type
    op_type = OP_TYPE_MAP.get(node.op_type, node.op_type)

    # Extract attributes
    attributes = {}
    for attr in node.attribute:
        value = get_attribute_value(attr)
        if value is not None:
            attributes[attr.name] = value

    # Generate node ID
    node_id = node.name if node.name else f"node_{idx}"

    return {
        'id': node_id,
        'op_type': op_type,
        'inputs': list(node.input),
        'outputs': list(node.output),
        'attributes': attributes
    }


def convert_onnx_model(onnx_path: str, output_path: str):
    """Convert ONNX model to TinyInfer JSON format"""
    print(f"Loading ONNX model from: {onnx_path}")

    # Load ONNX model
    model = onnx.load(onnx_path)
    graph = model.graph

    print(f"Model: {graph.name}")
    print(f"  Nodes: {len(graph.node)}")
    print(f"  Inputs: {len(graph.input)}")
    print(f"  Outputs: {len(graph.output)}")
    print(f"  Initializers: {len(graph.initializer)}")

    # Convert nodes
    nodes = []

    # Add input nodes
    for inp in graph.input:
        # Skip initializers (weights)
        if inp.name not in [init.name for init in graph.initializer]:
            nodes.append({
                'id': inp.name,
                'op_type': 'Input',
                'inputs': [],
                'outputs': [inp.name],
                'attributes': {}
            })

    # Add operator nodes
    for idx, node in enumerate(graph.node):
        nodes.append(convert_node(node, idx))

    # Add output nodes
    for out in graph.output:
        nodes.append({
            'id': f"{out.name}_output",
            'op_type': 'Output',
            'inputs': [out.name],
            'outputs': [],
            'attributes': {}
        })

    # Build edges from node connections
    edges = []
    for node in graph.node:
        for inp in node.input:
            if inp and not inp in [init.name for init in graph.initializer]:
                edges.append({
                    'from': inp,
                    'to': node.name if node.name else node.output[0]
                })

        # For nodes with outputs, connect to next
        if node.output:
            for out in node.output:
                # Find nodes that use this output
                for next_node in graph.node:
                    if out in next_node.input:
                        edges.append({
                            'from': node.name if node.name else out,
                            'to': next_node.name if next_node.name else next_node.output[0]
                        })

    # Convert weights (initializers)
    weights = {}
    for init in graph.initializer:
        print(f"  Converting weight: {init.name} {list(init.dims)}")
        weights[init.name] = convert_tensor(init)

    # Build model definition
    model_def = {
        'version': '1.0',
        'name': graph.name or Path(onnx_path).stem,
        'graph': {
            'nodes': nodes,
            'edges': edges,
            'inputs': [inp.name for inp in graph.input
                      if inp.name not in [init.name for init in graph.initializer]],
            'outputs': [out.name for out in graph.output]
        },
        'weights': weights
    }

    # Save to JSON
    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(model_def, f, indent=2)

    print(f"✅ Conversion complete!")
    print(f"\nModel summary:")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Total edges: {len(edges)}")
    print(f"  Total weights: {len(weights)}")

    # Calculate total weight size
    total_params = sum(
        len(w['data']) for w in weights.values()
    )
    print(f"  Total parameters: {total_params:,}")
    print(f"  Estimated size: {total_params * 4 / 1024 / 1024:.2f} MB")


def create_simple_model_example():
    """Create a simple example model in TinyInfer format"""
    model = {
        'version': '1.0',
        'name': 'simple_linear',
        'graph': {
            'nodes': [
                {
                    'id': 'input',
                    'op_type': 'Input',
                    'inputs': [],
                    'outputs': ['input'],
                    'attributes': {}
                },
                {
                    'id': 'linear',
                    'op_type': 'MatMul',
                    'inputs': ['input', 'weight'],
                    'outputs': ['linear_out'],
                    'attributes': {}
                },
                {
                    'id': 'relu',
                    'op_type': 'ReLU',
                    'inputs': ['linear_out'],
                    'outputs': ['relu_out'],
                    'attributes': {}
                },
                {
                    'id': 'output',
                    'op_type': 'Output',
                    'inputs': ['relu_out'],
                    'outputs': [],
                    'attributes': {}
                }
            ],
            'edges': [
                {'from': 'input', 'to': 'linear'},
                {'from': 'linear', 'to': 'relu'},
                {'from': 'relu', 'to': 'output'}
            ],
            'inputs': ['input'],
            'outputs': ['relu_out']
        },
        'weights': {
            'weight': {
                'shape': [4, 4],
                'dtype': 'float32',
                'data': [1.0] * 16  # 4x4 identity-like matrix
            }
        }
    }

    return model


def main():
    if len(sys.argv) < 2:
        print("TinyInfer Model Converter")
        print("\nUsage:")
        print(f"  {sys.argv[0]} <input.onnx> <output.json>       # Convert ONNX model")
        print(f"  {sys.argv[0]} --create-example <output.json>   # Create example model")
        print("\nExamples:")
        print(f"  {sys.argv[0]} model.onnx model.json")
        print(f"  {sys.argv[0]} --create-example example.json")
        sys.exit(1)

    if sys.argv[1] == '--create-example':
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'example_model.json'
        print(f"Creating example model: {output_path}")
        model = create_simple_model_example()
        with open(output_path, 'w') as f:
            json.dump(model, f, indent=2)
        print(f"✅ Example model created!")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'model.json'

        if not Path(input_path).exists():
            print(f"Error: File not found: {input_path}")
            sys.exit(1)

        convert_onnx_model(input_path, output_path)


if __name__ == '__main__':
    main()
