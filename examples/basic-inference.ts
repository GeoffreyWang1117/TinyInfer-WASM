/**
 * 基础推理示例
 * 演示如何加载模型并执行推理
 */

import { initWasm, createInferenceEngine } from '../web/src/lib/tinyinfer'

async function basicInferenceExample() {
  console.log('=== 基础推理示例 ===\n')

  // 1. 初始化 WASM 模块
  console.log('初始化 WASM 模块...')
  await initWasm()
  console.log('✅ WASM 初始化完成\n')

  // 2. 创建推理引擎
  console.log('创建推理引擎...')
  const engine = createInferenceEngine()
  console.log('✅ 推理引擎创建完成\n')

  // 3. 加载测试模型
  console.log('加载测试模型 (ReLU)...')
  engine.loadTestModel()
  console.log('✅ 模型加载完成')
  console.log('模型信息:', engine.getModelInfo(), '\n')

  // 4. 准备输入数据
  const inputData = new Float32Array([-5, -2, 0, 3, 7])
  const inputShape = [5]
  
  console.log('输入数据:', Array.from(inputData))
  console.log('输入形状:', inputShape, '\n')

  // 5. 执行推理
  console.log('执行推理...')
  const startTime = performance.now()
  const output = engine.infer(inputData, inputShape)
  const endTime = performance.now()
  
  console.log('✅ 推理完成')
  console.log('输出数据:', Array.from(output))
  console.log('推理时间:', (endTime - startTime).toFixed(2), 'ms\n')

  // 6. 验证结果
  const expected = [0, 0, 0, 3, 7] // ReLU: max(0, x)
  const isCorrect = output.every((val, idx) => Math.abs(val - expected[idx]) < 1e-6)
  
  if (isCorrect) {
    console.log('✅ 推理结果正确！')
  } else {
    console.log('❌ 推理结果不正确')
    console.log('预期:', expected)
  }

  // 7. 清理资源
  engine.free()
  console.log('\n✅ 示例完成')
}

// 运行示例
basicInferenceExample().catch(console.error)
