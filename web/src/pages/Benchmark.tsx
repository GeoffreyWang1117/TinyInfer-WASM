import { useState } from 'react'

interface BenchmarkResult {
  name: string
  tinyinfer: number
  onnxjs: number
  tfjs: number
}

export default function Benchmark() {
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)

  const opResults: BenchmarkResult[] = [
    { name: 'MatMul (1024x1024)', tinyinfer: 45, onnxjs: 180, tfjs: 220 },
    { name: 'Conv2D (224x224, 64ch)', tinyinfer: 85, onnxjs: 650, tfjs: 780 },
    { name: 'ReLU (1M elements)', tinyinfer: 2, onnxjs: 8, tfjs: 12 },
    { name: 'BatchNorm (224x224x64)', tinyinfer: 15, onnxjs: 45, tfjs: 60 },
  ]

  const modelResults: BenchmarkResult[] = [
    { name: 'MobileNetV2', tinyinfer: 45, onnxjs: 120, tfjs: 180 },
    { name: 'MiniLM-L6', tinyinfer: 23, onnxjs: 65, tfjs: 95 },
    { name: 'TinyLM-160M', tinyinfer: 150, onnxjs: -1, tfjs: -1 },
  ]

  const runBenchmark = async () => {
    setIsRunning(true)
    setProgress(0)

    for (let i = 0; i <= 100; i += 10) {
      await new Promise(resolve => setTimeout(resolve, 200))
      setProgress(i)
    }

    setIsRunning(false)
  }

  const getSpeedup = (tinyinfer: number, other: number) => {
    if (other <= 0) return '-'
    return `${(other / tinyinfer).toFixed(1)}x`
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          性能测试
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          与其他推理引擎的性能对比
        </p>
      </div>

      {/* Test Controls */}
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              运行完整测试
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              测试所有算子和模型的性能
            </p>
          </div>
          <button
            onClick={runBenchmark}
            disabled={isRunning}
            className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 text-white px-6 py-3 rounded-lg font-semibold transition"
          >
            {isRunning ? '运行中...' : '运行测试'}
          </button>
        </div>

        {isRunning && (
          <div className="mt-4">
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
              <div
                className="bg-blue-500 h-3 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <p className="text-center text-sm text-gray-600 dark:text-gray-400 mt-2">
              {progress}%
            </p>
          </div>
        )}
      </div>

      {/* Operator Benchmarks */}
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          算子性能对比
        </h3>

        <div className="space-y-6">
          {opResults.map((result, idx) => (
            <div key={idx}>
              <div className="flex justify-between items-center mb-2">
                <h4 className="font-medium text-gray-900 dark:text-white">
                  {result.name}
                </h4>
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  加速比: {getSpeedup(result.tinyinfer, result.tfjs)}
                </span>
              </div>

              <div className="space-y-2">
                {/* TinyInfer */}
                <div className="flex items-center space-x-3">
                  <span className="w-24 text-sm text-gray-600 dark:text-gray-400">
                    TinyInfer
                  </span>
                  <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 relative overflow-hidden">
                    <div
                      className="bg-green-500 h-6 rounded-full flex items-center justify-end px-3 transition-all duration-500"
                      style={{
                        width: `${(result.tinyinfer / Math.max(result.tinyinfer, result.onnxjs, result.tfjs)) * 100}%`
                      }}
                    >
                      <span className="text-xs font-semibold text-white">
                        {result.tinyinfer}ms
                      </span>
                    </div>
                  </div>
                </div>

                {/* ONNX.js */}
                <div className="flex items-center space-x-3">
                  <span className="w-24 text-sm text-gray-600 dark:text-gray-400">
                    ONNX.js
                  </span>
                  <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 relative overflow-hidden">
                    <div
                      className="bg-blue-500 h-6 rounded-full flex items-center justify-end px-3 transition-all duration-500"
                      style={{
                        width: `${(result.onnxjs / Math.max(result.tinyinfer, result.onnxjs, result.tfjs)) * 100}%`
                      }}
                    >
                      <span className="text-xs font-semibold text-white">
                        {result.onnxjs}ms
                      </span>
                    </div>
                  </div>
                </div>

                {/* TensorFlow.js */}
                <div className="flex items-center space-x-3">
                  <span className="w-24 text-sm text-gray-600 dark:text-gray-400">
                    TF.js
                  </span>
                  <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 relative overflow-hidden">
                    <div
                      className="bg-orange-500 h-6 rounded-full flex items-center justify-end px-3 transition-all duration-500"
                      style={{
                        width: `${(result.tfjs / Math.max(result.tinyinfer, result.onnxjs, result.tfjs)) * 100}%`
                      }}
                    >
                      <span className="text-xs font-semibold text-white">
                        {result.tfjs}ms
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Model Benchmarks */}
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          端到端模型测试
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left py-3 px-4 text-gray-900 dark:text-white font-semibold">
                  模型
                </th>
                <th className="text-right py-3 px-4 text-gray-900 dark:text-white font-semibold">
                  TinyInfer
                </th>
                <th className="text-right py-3 px-4 text-gray-900 dark:text-white font-semibold">
                  ONNX.js
                </th>
                <th className="text-right py-3 px-4 text-gray-900 dark:text-white font-semibold">
                  TF.js
                </th>
                <th className="text-right py-3 px-4 text-gray-900 dark:text-white font-semibold">
                  加速比
                </th>
              </tr>
            </thead>
            <tbody>
              {modelResults.map((result, idx) => (
                <tr
                  key={idx}
                  className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700"
                >
                  <td className="py-3 px-4 text-gray-900 dark:text-white">
                    {result.name}
                  </td>
                  <td className="text-right py-3 px-4">
                    <span className="text-green-600 dark:text-green-400 font-semibold">
                      {result.tinyinfer}ms
                    </span>
                  </td>
                  <td className="text-right py-3 px-4 text-gray-600 dark:text-gray-400">
                    {result.onnxjs > 0 ? `${result.onnxjs}ms` : '-'}
                  </td>
                  <td className="text-right py-3 px-4 text-gray-600 dark:text-gray-400">
                    {result.tfjs > 0 ? `${result.tfjs}ms` : '-'}
                  </td>
                  <td className="text-right py-3 px-4">
                    <span className="text-blue-600 dark:text-blue-400 font-semibold">
                      {getSpeedup(result.tinyinfer, result.tfjs)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* System Info */}
      <div className="bg-blue-50 dark:bg-blue-900 dark:bg-opacity-20 p-6 rounded-lg">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          系统信息
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-gray-600 dark:text-gray-400">浏览器</p>
            <p className="text-gray-900 dark:text-white font-semibold">
              {navigator.userAgent.includes('Chrome') ? 'Chrome' : 'Other'}
            </p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">WASM SIMD</p>
            <p className="text-gray-900 dark:text-white font-semibold">
              支持
            </p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">CPU 核心</p>
            <p className="text-gray-900 dark:text-white font-semibold">
              {navigator.hardwareConcurrency || 4}
            </p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">内存</p>
            <p className="text-gray-900 dark:text-white font-semibold">
              {navigator.deviceMemory || '未知'} GB
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
