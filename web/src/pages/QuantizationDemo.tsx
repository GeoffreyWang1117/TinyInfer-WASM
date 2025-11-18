import { useState } from 'react'
import { quantizeToInt8, dequantizeFromInt8, quantizeModelWeights, calculateSizeReduction, QuantizationParams } from '../lib/quantization'

/**
 * Quantization Demo Page
 *
 * Demonstrates Phase 8 INT8 quantization capabilities:
 * - 4x model size reduction
 * - Accuracy loss < 2%
 * - Per-tensor and per-channel quantization
 */

export default function QuantizationDemo() {
  const [originalData, setOriginalData] = useState<Float32Array | null>(null)
  const [quantizedData, setQuantizedData] = useState<Int8Array | null>(null)
  const [dequantizedData, setDequantizedData] = useState<Float32Array | null>(null)
  const [quantParams, setQuantParams] = useState<QuantizationParams | null>(null)
  const [symmetric, setSymmetric] = useState(true)
  const [dataSize, setDataSize] = useState(100)
  const [dataRange, setDataRange] = useState({ min: -10, max: 10 })
  const [modelSize, setModelSize] = useState({ original: 0, quantized: 0 })

  // Generate random test data
  const generateTestData = () => {
    const data = new Float32Array(dataSize)
    const { min, max } = dataRange
    for (let i = 0; i < dataSize; i++) {
      data[i] = min + Math.random() * (max - min)
    }
    setOriginalData(data)
    setQuantizedData(null)
    setDequantizedData(null)
    setQuantParams(null)
  }

  // Perform quantization
  const performQuantization = () => {
    if (!originalData) return

    const result = quantizeToInt8(originalData, [originalData.length], undefined, symmetric)
    setQuantizedData(result.data)
    setQuantParams(result.params)

    // Dequantize to check accuracy
    const deq = dequantizeFromInt8(result)
    setDequantizedData(deq)

    // Calculate sizes
    const originalSize = originalData.length * 4 // 4 bytes per float32
    const quantizedSize = result.data.length * 1 // 1 byte per int8
    setModelSize({ original: originalSize, quantized: quantizedSize })
  }

  // Calculate accuracy metrics
  const calculateAccuracy = () => {
    if (!originalData || !dequantizedData) return null

    let maxError = 0
    let sumSquaredError = 0
    let sumAbsError = 0

    for (let i = 0; i < originalData.length; i++) {
      const error = Math.abs(originalData[i] - dequantizedData[i])
      maxError = Math.max(maxError, error)
      sumSquaredError += error * error
      sumAbsError += error
    }

    const mse = sumSquaredError / originalData.length
    const rmse = Math.sqrt(mse)
    const mae = sumAbsError / originalData.length
    const maxRelativeError = maxError / Math.max(...Array.from(originalData).map(Math.abs))

    return { maxError, rmse, mae, maxRelativeError }
  }

  const accuracy = calculateAccuracy()

  // Quantize example model
  const quantizeExampleModel = () => {
    const exampleModel = {
      version: '0.8.0',
      name: 'example_model',
      weights: {
        'layer1.weight': {
          shape: [64, 128],
          dtype: 'float32',
          data: Array(64 * 128).fill(0).map(() => Math.random() * 2 - 1),
        },
        'layer2.weight': {
          shape: [128, 256],
          dtype: 'float32',
          data: Array(128 * 256).fill(0).map(() => Math.random() * 2 - 1),
        },
        'layer3.weight': {
          shape: [256, 10],
          dtype: 'float32',
          data: Array(256 * 10).fill(0).map(() => Math.random() * 2 - 1),
        },
      },
    }

    const result = quantizeModelWeights(exampleModel, symmetric)

    const sizeReduction = calculateSizeReduction(
      result.stats.originalSize,
      result.stats.quantizedSize
    )

    alert(
      `Model Quantization Complete!\n\n` +
      `Original Size: ${(result.stats.originalSize / 1024).toFixed(2)} KB\n` +
      `Quantized Size: ${(result.stats.quantizedSize / 1024).toFixed(2)} KB\n` +
      `Compression: ${sizeReduction.compressionFactor.toFixed(2)}x\n` +
      `Size Reduction: ${(sizeReduction.reductionRatio * 100).toFixed(1)}%\n\n` +
      `Weights Quantized: ${result.stats.weightsQuantized}\n` +
      `Check console for detailed results.`
    )

    console.log('Quantization Result:', result)
    console.log('Size Reduction:', sizeReduction)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          INT8 量化演示
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          Phase 8 特性: 4倍模型压缩，精度损失 &lt; 2%
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
          测试数据配置
        </h3>

        <div className="space-y-4">
          {/* Data Size */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              数据大小: {dataSize}
            </label>
            <input
              type="range"
              min="10"
              max="1000"
              value={dataSize}
              onChange={e => setDataSize(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Data Range */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                最小值: {dataRange.min}
              </label>
              <input
                type="range"
                min="-100"
                max="0"
                value={dataRange.min}
                onChange={e => setDataRange({ ...dataRange, min: parseInt(e.target.value) })}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                最大值: {dataRange.max}
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={dataRange.max}
                onChange={e => setDataRange({ ...dataRange, max: parseInt(e.target.value) })}
                className="w-full"
              />
            </div>
          </div>

          {/* Quantization Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              量化类型
            </label>
            <div className="flex space-x-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  checked={symmetric}
                  onChange={() => setSymmetric(true)}
                  className="mr-2"
                />
                <span className="text-gray-700 dark:text-gray-300">对称量化</span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  checked={!symmetric}
                  onChange={() => setSymmetric(false)}
                  className="mr-2"
                />
                <span className="text-gray-700 dark:text-gray-300">非对称量化</span>
              </label>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-3 pt-4">
            <button
              onClick={generateTestData}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              生成测试数据
            </button>
            <button
              onClick={performQuantization}
              disabled={!originalData}
              className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              执行量化
            </button>
            <button
              onClick={quantizeExampleModel}
              className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600"
            >
              量化示例模型
            </button>
          </div>
        </div>
      </div>

      {/* Quantization Parameters */}
      {quantParams && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
            量化参数
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded">
              <div className="text-sm text-gray-600 dark:text-gray-400">Scale (缩放因子)</div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {quantParams.scale.toExponential(4)}
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded">
              <div className="text-sm text-gray-600 dark:text-gray-400">Zero Point (零点)</div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {quantParams.zeroPoint}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Size Comparison */}
      {quantizedData && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
            大小对比
          </h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded">
              <div className="text-sm text-blue-600 dark:text-blue-400">原始大小 (Float32)</div>
              <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                {modelSize.original} bytes
              </div>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded">
              <div className="text-sm text-green-600 dark:text-green-400">量化大小 (INT8)</div>
              <div className="text-2xl font-bold text-green-900 dark:text-green-100">
                {modelSize.quantized} bytes
              </div>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded">
              <div className="text-sm text-purple-600 dark:text-purple-400">压缩比</div>
              <div className="text-2xl font-bold text-purple-900 dark:text-purple-100">
                {(modelSize.original / modelSize.quantized).toFixed(2)}x
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Accuracy Metrics */}
      {accuracy && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
            精度分析
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded">
              <div className="text-sm text-gray-600 dark:text-gray-400">最大误差</div>
              <div className="text-xl font-bold text-gray-900 dark:text-white">
                {accuracy.maxError.toFixed(4)}
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded">
              <div className="text-sm text-gray-600 dark:text-gray-400">RMSE</div>
              <div className="text-xl font-bold text-gray-900 dark:text-white">
                {accuracy.rmse.toFixed(4)}
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded">
              <div className="text-sm text-gray-600 dark:text-gray-400">MAE</div>
              <div className="text-xl font-bold text-gray-900 dark:text-white">
                {accuracy.mae.toFixed(4)}
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded">
              <div className="text-sm text-gray-600 dark:text-gray-400">相对误差 %</div>
              <div className="text-xl font-bold text-gray-900 dark:text-white">
                {(accuracy.maxRelativeError * 100).toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Data Visualization */}
      {originalData && dequantizedData && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
            数据对比 (前 20 个值)
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead>
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    索引
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    原始值
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    量化值 (INT8)
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    反量化值
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    误差
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {Array.from({ length: Math.min(20, originalData.length) }).map((_, i) => {
                  const error = Math.abs(originalData[i] - dequantizedData[i])
                  return (
                    <tr key={i}>
                      <td className="px-4 py-2 text-sm text-gray-900 dark:text-white">{i}</td>
                      <td className="px-4 py-2 text-sm font-mono text-gray-900 dark:text-white">
                        {originalData[i].toFixed(4)}
                      </td>
                      <td className="px-4 py-2 text-sm font-mono text-gray-900 dark:text-white">
                        {quantizedData![i]}
                      </td>
                      <td className="px-4 py-2 text-sm font-mono text-gray-900 dark:text-white">
                        {dequantizedData[i].toFixed(4)}
                      </td>
                      <td className="px-4 py-2 text-sm font-mono text-red-600 dark:text-red-400">
                        {error.toFixed(4)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Information */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-3 text-blue-900 dark:text-blue-100">
          💡 量化原理
        </h3>
        <div className="space-y-2 text-sm text-blue-800 dark:text-blue-200">
          <p>
            <strong>对称量化:</strong> 零点为 0，范围对称 [-max, max]，适合权重
          </p>
          <p>
            <strong>非对称量化:</strong> 零点可调，范围 [min, max]，适合激活值
          </p>
          <p>
            <strong>量化公式:</strong> q = round(x / scale) + zero_point
          </p>
          <p>
            <strong>反量化公式:</strong> x' = (q - zero_point) * scale
          </p>
          <p className="pt-2">
            <strong>优势:</strong>
          </p>
          <ul className="list-disc list-inside pl-4 space-y-1">
            <li>模型大小减少 75% (4x 压缩)</li>
            <li>推理速度提升 2-4x (INT8 SIMD)</li>
            <li>精度损失通常 &lt; 2%</li>
            <li>内存占用减少，适合移动端</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
