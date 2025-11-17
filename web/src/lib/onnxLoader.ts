/**
 * Browser-side ONNX Model Loader for TinyInfer
 *
 * Loads and converts ONNX models directly in the browser without server-side conversion.
 */

import protobuf from 'protobufjs'

// Operator type mapping from ONNX to TinyInfer
const OP_TYPE_MAP: Record<string, string> = {
  Relu: 'ReLU',
  Sigmoid: 'Sigmoid',
  Tanh: 'Tanh',
  Softmax: 'Softmax',
  MatMul: 'MatMul',
  Gemm: 'Gemm',
  Conv: 'Conv2D',
  MaxPool: 'MaxPool2D',
  AveragePool: 'AvgPool2D',
  GlobalAveragePool: 'GlobalAvgPool2D',
  BatchNormalization: 'BatchNorm2D',
  Add: 'Add',
  Sub: 'Sub',
  Mul: 'Mul',
  Div: 'Div',
  Transpose: 'Transpose',
  Reshape: 'Reshape',
  Concat: 'Concat',
  Split: 'Split',
  Gather: 'Gather',
}

// TinyInfer model definition interfaces
export interface AttributeValue {
  float?: number
  int?: number
  string?: string
  floats?: number[]
  ints?: number[]
  strings?: string[]
}

export interface NodeDef {
  id: string
  op_type: string
  inputs: string[]
  outputs: string[]
  attributes: Record<string, AttributeValue>
}

export interface EdgeDef {
  from: string
  to: string
}

export interface WeightDef {
  shape: number[]
  dtype: string
  data: number[]
}

export interface GraphDef {
  nodes: NodeDef[]
  edges: EdgeDef[]
  inputs: string[]
  outputs: string[]
}

export interface ModelDef {
  version: string
  name: string
  graph: GraphDef
  weights: Record<string, WeightDef>
}

/**
 * ONNX Loader class
 */
export class ONNXLoader {
  private root: protobuf.Root | null = null

  /**
   * Initialize the ONNX protobuf schema
   */
  async initialize(): Promise<void> {
    if (this.root) return

    // Load ONNX protobuf schema
    const protoPath = new URL('./onnx.proto', import.meta.url).href
    this.root = await protobuf.load(protoPath)
  }

  /**
   * Load ONNX model from ArrayBuffer
   */
  async loadFromBuffer(buffer: ArrayBuffer): Promise<ModelDef> {
    await this.initialize()

    if (!this.root) {
      throw new Error('Protobuf schema not initialized')
    }

    // Decode ONNX model
    const ModelProto = this.root.lookupType('onnx.ModelProto')
    const message = ModelProto.decode(new Uint8Array(buffer))
    const model = ModelProto.toObject(message, {
      longs: Number,
      enums: String,
      bytes: String,
      defaults: true,
    }) as any

    // Convert to TinyInfer format
    return this.convertToTinyInfer(model)
  }

  /**
   * Load ONNX model from File
   */
  async loadFromFile(file: File): Promise<ModelDef> {
    const buffer = await file.arrayBuffer()
    return this.loadFromBuffer(buffer)
  }

  /**
   * Load ONNX model from URL
   */
  async loadFromURL(url: string): Promise<ModelDef> {
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.statusText}`)
    }
    const buffer = await response.arrayBuffer()
    return this.loadFromBuffer(buffer)
  }

  /**
   * Convert ONNX model to TinyInfer format
   */
  private convertToTinyInfer(onnxModel: any): ModelDef {
    const graph = onnxModel.graph

    if (!graph) {
      throw new Error('Invalid ONNX model: no graph found')
    }

    // Extract weights (initializers)
    const weights: Record<string, WeightDef> = {}
    const initializerNames = new Set<string>()

    if (graph.initializer) {
      for (const init of graph.initializer) {
        initializerNames.add(init.name)
        weights[init.name] = this.convertTensor(init)
      }
    }

    // Convert nodes
    const nodes: NodeDef[] = []

    // Add input nodes (excluding initializers)
    if (graph.input) {
      for (const input of graph.input) {
        if (!initializerNames.has(input.name)) {
          nodes.push({
            id: input.name,
            op_type: 'Input',
            inputs: [],
            outputs: [input.name],
            attributes: {},
          })
        }
      }
    }

    // Add operator nodes
    if (graph.node) {
      for (let i = 0; i < graph.node.length; i++) {
        const node = graph.node[i]
        nodes.push(this.convertNode(node, i))
      }
    }

    // Add output nodes
    if (graph.output) {
      for (const output of graph.output) {
        nodes.push({
          id: `${output.name}_output`,
          op_type: 'Output',
          inputs: [output.name],
          outputs: [],
          attributes: {},
        })
      }
    }

    // Build edges from node connections
    const edges: EdgeDef[] = []
    const nodeMap = new Map<string, NodeDef>()

    for (const node of nodes) {
      if (node.outputs.length > 0) {
        nodeMap.set(node.outputs[0], node)
      }
    }

    if (graph.node) {
      for (const node of graph.node) {
        const nodeId = node.name || node.output?.[0] || `node_${nodes.indexOf(node)}`

        // Connect inputs to this node
        if (node.input) {
          for (const inputName of node.input) {
            // Skip weights
            if (initializerNames.has(inputName)) continue

            const sourceNode = nodeMap.get(inputName)
            if (sourceNode) {
              edges.push({
                from: sourceNode.id,
                to: nodeId,
              })
            }
          }
        }
      }
    }

    // Extract input and output names
    const inputs = graph.input
      ? graph.input
          .filter((inp: any) => !initializerNames.has(inp.name))
          .map((inp: any) => inp.name)
      : []

    const outputs = graph.output ? graph.output.map((out: any) => out.name) : []

    return {
      version: '1.0',
      name: graph.name || 'onnx_model',
      graph: {
        nodes,
        edges,
        inputs,
        outputs,
      },
      weights,
    }
  }

  /**
   * Convert ONNX node to TinyInfer node
   */
  private convertNode(onnxNode: any, idx: number): NodeDef {
    const opType = OP_TYPE_MAP[onnxNode.opType] || onnxNode.opType
    const nodeId = onnxNode.name || `node_${idx}`

    // Convert attributes
    const attributes: Record<string, AttributeValue> = {}
    if (onnxNode.attribute) {
      for (const attr of onnxNode.attribute) {
        attributes[attr.name] = this.convertAttribute(attr)
      }
    }

    return {
      id: nodeId,
      op_type: opType,
      inputs: onnxNode.input || [],
      outputs: onnxNode.output || [],
      attributes,
    }
  }

  /**
   * Convert ONNX attribute to TinyInfer attribute
   */
  private convertAttribute(attr: any): AttributeValue {
    const result: AttributeValue = {}

    if (attr.f !== undefined && attr.f !== null) {
      result.float = attr.f
    }
    if (attr.i !== undefined && attr.i !== null) {
      result.int = Number(attr.i)
    }
    if (attr.s !== undefined && attr.s !== null) {
      result.string = typeof attr.s === 'string' ? attr.s : new TextDecoder().decode(attr.s)
    }
    if (attr.floats && attr.floats.length > 0) {
      result.floats = attr.floats
    }
    if (attr.ints && attr.ints.length > 0) {
      result.ints = attr.ints.map(Number)
    }
    if (attr.strings && attr.strings.length > 0) {
      result.strings = attr.strings.map((s: any) =>
        typeof s === 'string' ? s : new TextDecoder().decode(s)
      )
    }

    return result
  }

  /**
   * Convert ONNX tensor to TinyInfer weight
   */
  private convertTensor(tensor: any): WeightDef {
    const shape = tensor.dims ? tensor.dims.map(Number) : []
    const dtype = this.convertDataType(tensor.dataType)

    // Extract data
    let data: number[] = []

    if (tensor.rawData && tensor.rawData.length > 0) {
      // Parse raw data based on data type
      data = this.parseRawData(tensor.rawData, tensor.dataType, shape)
    } else if (tensor.floatData && tensor.floatData.length > 0) {
      data = Array.from(tensor.floatData)
    } else if (tensor.int32Data && tensor.int32Data.length > 0) {
      data = Array.from(tensor.int32Data).map(Number)
    } else if (tensor.int64Data && tensor.int64Data.length > 0) {
      data = Array.from(tensor.int64Data).map(Number)
    } else if (tensor.doubleData && tensor.doubleData.length > 0) {
      data = Array.from(tensor.doubleData)
    }

    // Convert to float32 for consistency
    return {
      shape,
      dtype: 'float32',
      data,
    }
  }

  /**
   * Convert ONNX data type to string
   */
  private convertDataType(dataType: number): string {
    const typeMap: Record<number, string> = {
      1: 'float32',
      2: 'uint8',
      3: 'int8',
      6: 'int32',
      7: 'int64',
      10: 'float16',
      11: 'float64',
    }
    return typeMap[dataType] || 'float32'
  }

  /**
   * Parse raw tensor data
   */
  private parseRawData(rawData: Uint8Array | string, dataType: number, shape: number[]): number[] {
    // Convert string to Uint8Array if needed
    const bytes = typeof rawData === 'string' ? new Uint8Array(Buffer.from(rawData, 'base64')) : rawData

    const totalElements = shape.reduce((a, b) => a * b, 1)
    const data: number[] = []

    switch (dataType) {
      case 1: // FLOAT
        {
          const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
          for (let i = 0; i < totalElements; i++) {
            data.push(view.getFloat32(i * 4, true))
          }
        }
        break

      case 11: // DOUBLE
        {
          const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
          for (let i = 0; i < totalElements; i++) {
            data.push(view.getFloat64(i * 8, true))
          }
        }
        break

      case 6: // INT32
        {
          const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
          for (let i = 0; i < totalElements; i++) {
            data.push(view.getInt32(i * 4, true))
          }
        }
        break

      case 7: // INT64
        {
          const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
          for (let i = 0; i < totalElements; i++) {
            // Note: JavaScript can't accurately represent int64, so we use Number
            data.push(Number(view.getBigInt64(i * 8, true)))
          }
        }
        break

      default:
        // For other types, attempt to read as bytes
        for (let i = 0; i < Math.min(totalElements, bytes.length); i++) {
          data.push(bytes[i])
        }
    }

    return data
  }
}

/**
 * Create a singleton instance
 */
const onnxLoader = new ONNXLoader()

/**
 * Load ONNX model from File and convert to TinyInfer format
 */
export async function loadONNXFromFile(file: File): Promise<ModelDef> {
  return onnxLoader.loadFromFile(file)
}

/**
 * Load ONNX model from URL and convert to TinyInfer format
 */
export async function loadONNXFromURL(url: string): Promise<ModelDef> {
  return onnxLoader.loadFromURL(url)
}

/**
 * Load ONNX model from ArrayBuffer and convert to TinyInfer format
 */
export async function loadONNXFromBuffer(buffer: ArrayBuffer): Promise<ModelDef> {
  return onnxLoader.loadFromBuffer(buffer)
}
