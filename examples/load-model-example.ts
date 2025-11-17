/**
 * TinyInfer-WASM 真实模型加载示例
 *
 * 本示例展示如何加载和使用真实的神经网络模型
 */

import {
  initWasm,
  createInferenceEngine,
  loadModelFromJSON,
  loadModelFromURL,
  loadModelFromFile,
} from '../web/src/lib/tinyinfer'

/**
 * 示例 1: 从 JSON 字符串加载模型
 */
async function example1_loadFromJSON() {
  console.log('=== Example 1: Load from JSON string ===\n')

  // 初始化 WASM
  await initWasm()

  // 创建推理引擎
  const engine = createInferenceEngine()

  // 定义一个简单的模型 (input -> ReLU -> output)
  const modelJSON = JSON.stringify({
    version: '1.0',
    name: 'simple_relu',
    graph: {
      nodes: [
        {
          id: 'input',
          op_type: 'Input',
          inputs: [],
          outputs: ['input'],
          attributes: {},
        },
        {
          id: 'relu',
          op_type: 'ReLU',
          inputs: ['input'],
          outputs: ['relu_out'],
          attributes: {},
        },
        {
          id: 'output',
          op_type: 'Output',
          inputs: ['relu_out'],
          outputs: [],
          attributes: {},
        },
      ],
      edges: [
        { from: 'input', to: 'relu' },
        { from: 'relu', to: 'output' },
      ],
      inputs: ['input'],
      outputs: ['relu_out'],
    },
    weights: {},
  })

  // 加载模型
  await loadModelFromJSON(engine, modelJSON)

  console.log('Model loaded successfully!')
  console.log('Model info:', engine.getModelInfo())

  // 执行推理
  const input = new Float32Array([-1, 2, -3, 4, -5])
  const output = engine.infer(input, [5])

  console.log('Input:', input)
  console.log('Output:', output) // [0, 2, 0, 4, 0] - ReLU applied

  // 清理
  engine.free()

  console.log('\n✅ Example 1 completed!\n')
}

/**
 * 示例 2: 从 URL 加载模型
 */
async function example2_loadFromURL() {
  console.log('=== Example 2: Load from URL ===\n')

  await initWasm()
  const engine = createInferenceEngine()

  try {
    // 从 URL 加载模型
    await loadModelFromURL(engine, '/models/my_model.json')

    console.log('Model loaded from URL')
    console.log('Model info:', engine.getModelInfo())

    // 执行推理
    const input = new Float32Array([1, 2, 3])
    const output = engine.infer(input, [3])

    console.log('Output:', output)

    engine.free()
    console.log('\n✅ Example 2 completed!\n')
  } catch (error) {
    console.error('Failed to load model from URL:', error)
  }
}

/**
 * 示例 3: 从文件上传加载模型 (浏览器环境)
 */
function example3_setupFileUpload() {
  console.log('=== Example 3: Load from file upload ===\n')

  // 在 HTML 中:
  // <input type="file" id="modelFileInput" accept=".json">

  const fileInput = document.getElementById('modelFileInput') as HTMLInputElement

  if (fileInput) {
    fileInput.addEventListener('change', async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0]

      if (!file) {
        console.log('No file selected')
        return
      }

      console.log(`Selected file: ${file.name}`)

      try {
        await initWasm()
        const engine = createInferenceEngine()

        // 从文件加载模型
        await loadModelFromFile(engine, file)

        console.log('Model loaded successfully!')
        console.log('Model info:', engine.getModelInfo())

        // 检查模型是否已加载
        if (engine.isModelLoaded()) {
          console.log('✅ Model is ready for inference')
        }

        console.log('\n✅ Example 3 completed!\n')
      } catch (error) {
        console.error('Failed to load model:', error)
      }
    })
  }
}

/**
 * 示例 4: 加载带有权重的模型
 */
async function example4_loadModelWithWeights() {
  console.log('=== Example 4: Load model with weights ===\n')

  await initWasm()
  const engine = createInferenceEngine()

  // 定义一个带权重的模型 (input -> MatMul -> ReLU -> output)
  const modelWithWeights = {
    version: '1.0',
    name: 'linear_relu',
    graph: {
      nodes: [
        {
          id: 'input',
          op_type: 'Input',
          inputs: [],
          outputs: ['input'],
          attributes: {},
        },
        {
          id: 'matmul',
          op_type: 'MatMul',
          inputs: ['input', 'weight'],
          outputs: ['matmul_out'],
          attributes: {},
        },
        {
          id: 'relu',
          op_type: 'ReLU',
          inputs: ['matmul_out'],
          outputs: ['relu_out'],
          attributes: {},
        },
        {
          id: 'output',
          op_type: 'Output',
          inputs: ['relu_out'],
          outputs: [],
          attributes: {},
        },
      ],
      edges: [
        { from: 'input', to: 'matmul' },
        { from: 'matmul', to: 'relu' },
        { from: 'relu', to: 'output' },
      ],
      inputs: ['input'],
      outputs: ['relu_out'],
    },
    weights: {
      weight: {
        shape: [4, 4],
        dtype: 'float32',
        // 4x4 权重矩阵
        data: [
          1.0, 0.0, 0.0, 0.0,
          0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0,
          0.0, 0.0, 0.0, 1.0,
        ],
      },
    },
  }

  await loadModelFromJSON(engine, JSON.stringify(modelWithWeights))

  console.log('Model with weights loaded successfully!')

  // 执行推理
  const input = new Float32Array([1, -2, 3, -4])
  const output = engine.infer(input, [4])

  console.log('Input:', input)
  console.log('Output:', output)

  engine.free()

  console.log('\n✅ Example 4 completed!\n')
}

/**
 * 示例 5: 使用 ONNX 转换工具
 */
function example5_convertONNX() {
  console.log('=== Example 5: Convert ONNX model ===\n')

  console.log('Step 1: Install dependencies')
  console.log('  pip install onnx numpy')
  console.log('')

  console.log('Step 2: Convert ONNX model to TinyInfer format')
  console.log('  python tools/onnx_to_tinyinfer.py model.onnx model.json')
  console.log('')

  console.log('Step 3: Load the converted model')
  console.log('  const engine = createInferenceEngine()')
  console.log('  await loadModelFromURL(engine, "/models/model.json")')
  console.log('')

  console.log('Alternative: Create example model')
  console.log('  python tools/onnx_to_tinyinfer.py --create-example example.json')
  console.log('')

  console.log('✅ See tools/onnx_to_tinyinfer.py for more details\n')
}

/**
 * 运行所有示例
 */
async function runAllExamples() {
  try {
    await example1_loadFromJSON()
    // await example2_loadFromURL()
    // example3_setupFileUpload() // 仅浏览器环境
    await example4_loadModelWithWeights()
    example5_convertONNX()

    console.log('========================================')
    console.log('✅ All examples completed successfully!')
    console.log('========================================')
  } catch (error) {
    console.error('Error running examples:', error)
  }
}

// 如果直接运行此文件
if (import.meta.url === `file://${process.argv[1]}`) {
  runAllExamples()
}

// 导出供其他文件使用
export {
  example1_loadFromJSON,
  example2_loadFromURL,
  example3_setupFileUpload,
  example4_loadModelWithWeights,
  example5_convertONNX,
}
