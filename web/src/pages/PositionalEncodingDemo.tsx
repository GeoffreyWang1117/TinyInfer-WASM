import { useState, useMemo } from 'react'

/**
 * Positional Encoding Demo Page
 *
 * Demonstrates Phase 8 positional encoding capabilities:
 * - Sinusoidal Positional Encoding (Transformer standard)
 * - Learnable Positional Embedding (BERT style)
 * - Rotary Position Embedding - RoPE (LLaMA/GPT-Neo)
 */

type EncodingType = 'sinusoidal' | 'learnable' | 'rope'

export default function PositionalEncodingDemo() {
  const [encodingType, setEncodingType] = useState<EncodingType>('sinusoidal')
  const [dModel, setDModel] = useState(64)
  const [maxLen, setMaxLen] = useState(32)
  const [visualizeRange, setVisualizeRange] = useState({ start: 0, end: 16 })

  // Generate Sinusoidal Positional Encoding
  const generateSinusoidal = (pos: number, i: number, d: number): number => {
    const divTerm = Math.pow(10000, (2 * Math.floor(i / 2)) / d)
    if (i % 2 === 0) {
      return Math.sin(pos / divTerm)
    } else {
      return Math.cos(pos / divTerm)
    }
  }

  // Generate positional encoding matrix
  const positionalEncoding = useMemo(() => {
    const pe: number[][] = []

    for (let pos = 0; pos < maxLen; pos++) {
      const row: number[] = []
      for (let i = 0; i < dModel; i++) {
        if (encodingType === 'sinusoidal') {
          row.push(generateSinusoidal(pos, i, dModel))
        } else if (encodingType === 'learnable') {
          // Simulated learnable embedding (random initialization)
          row.push(Math.random() * 2 - 1)
        } else if (encodingType === 'rope') {
          // Simplified RoPE visualization
          const theta = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dModel)
          if (i % 2 === 0) {
            row.push(Math.cos(theta))
          } else {
            row.push(Math.sin(theta))
          }
        }
      }
      pe.push(row)
    }

    return pe
  }, [encodingType, dModel, maxLen])

  // Calculate statistics
  const stats = useMemo(() => {
    const flat = positionalEncoding.flat()
    const min = Math.min(...flat)
    const max = Math.max(...flat)
    const mean = flat.reduce((a, b) => a + b, 0) / flat.length
    const variance = flat.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / flat.length
    const std = Math.sqrt(variance)

    return { min, max, mean, std }
  }, [positionalEncoding])

  // Get color for heatmap
  const getColor = (value: number): string => {
    // Normalize to [0, 1]
    const normalized = (value + 1) / 2
    const r = Math.floor(255 * (1 - normalized))
    const b = Math.floor(255 * normalized)
    return `rgb(${r}, 0, ${b})`
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          位置编码演示
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          Phase 8 特性: 三种位置编码实现 (Sinusoidal, Learnable, RoPE)
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
          配置参数
        </h3>

        <div className="space-y-4">
          {/* Encoding Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              编码类型
            </label>
            <div className="flex flex-wrap gap-3">
              <button
                onClick={() => setEncodingType('sinusoidal')}
                className={`px-4 py-2 rounded ${
                  encodingType === 'sinusoidal'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                Sinusoidal (Transformer)
              </button>
              <button
                onClick={() => setEncodingType('learnable')}
                className={`px-4 py-2 rounded ${
                  encodingType === 'learnable'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                Learnable (BERT)
              </button>
              <button
                onClick={() => setEncodingType('rope')}
                className={`px-4 py-2 rounded ${
                  encodingType === 'rope'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                RoPE (LLaMA/GPT-Neo)
              </button>
            </div>
          </div>

          {/* d_model */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              d_model (模型维度): {dModel}
            </label>
            <input
              type="range"
              min="16"
              max="512"
              step="16"
              value={dModel}
              onChange={e => setDModel(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              常用值: 64, 128, 256, 512
            </div>
          </div>

          {/* max_len */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              max_len (最大序列长度): {maxLen}
            </label>
            <input
              type="range"
              min="8"
              max="128"
              step="8"
              value={maxLen}
              onChange={e => setMaxLen(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              常用值: 32, 64, 128, 512
            </div>
          </div>

          {/* Visualization Range */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              可视化范围: 位置 {visualizeRange.start} - {visualizeRange.end}
            </label>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-gray-500 dark:text-gray-400">起始位置</label>
                <input
                  type="range"
                  min="0"
                  max={maxLen - 1}
                  value={visualizeRange.start}
                  onChange={e =>
                    setVisualizeRange({
                      ...visualizeRange,
                      start: Math.min(parseInt(e.target.value), visualizeRange.end - 1),
                    })
                  }
                  className="w-full"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500 dark:text-gray-400">结束位置</label>
                <input
                  type="range"
                  min="1"
                  max={maxLen}
                  value={visualizeRange.end}
                  onChange={e =>
                    setVisualizeRange({
                      ...visualizeRange,
                      end: Math.max(parseInt(e.target.value), visualizeRange.start + 1),
                    })
                  }
                  className="w-full"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Statistics */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
          统计信息
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded">
            <div className="text-sm text-gray-600 dark:text-gray-400">最小值</div>
            <div className="text-xl font-bold text-gray-900 dark:text-white">
              {stats.min.toFixed(4)}
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded">
            <div className="text-sm text-gray-600 dark:text-gray-400">最大值</div>
            <div className="text-xl font-bold text-gray-900 dark:text-white">
              {stats.max.toFixed(4)}
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded">
            <div className="text-sm text-gray-600 dark:text-gray-400">均值</div>
            <div className="text-xl font-bold text-gray-900 dark:text-white">
              {stats.mean.toFixed(4)}
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded">
            <div className="text-sm text-gray-600 dark:text-gray-400">标准差</div>
            <div className="text-xl font-bold text-gray-900 dark:text-white">
              {stats.std.toFixed(4)}
            </div>
          </div>
        </div>
      </div>

      {/* Heatmap Visualization */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
          位置编码热力图
        </h3>
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            <div className="grid gap-0.5" style={{ gridTemplateRows: `repeat(${visualizeRange.end - visualizeRange.start}, minmax(0, 1fr))` }}>
              {positionalEncoding.slice(visualizeRange.start, visualizeRange.end).map((row, pos) => (
                <div key={pos} className="flex gap-0.5">
                  <div className="w-12 flex items-center justify-center text-xs text-gray-600 dark:text-gray-400">
                    {visualizeRange.start + pos}
                  </div>
                  {row.slice(0, Math.min(64, dModel)).map((value, i) => (
                    <div
                      key={i}
                      className="w-3 h-3"
                      style={{ backgroundColor: getColor(value) }}
                      title={`Pos: ${visualizeRange.start + pos}, Dim: ${i}, Value: ${value.toFixed(4)}`}
                    />
                  ))}
                  {dModel > 64 && (
                    <div className="flex items-center text-xs text-gray-500 dark:text-gray-400 ml-2">
                      ... +{dModel - 64} dims
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
              显示维度: 0 - {Math.min(64, dModel)} {dModel > 64 && `(共 ${dModel} 维)`}
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="mt-4 flex items-center justify-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-4" style={{ background: 'linear-gradient(to right, rgb(255,0,0), rgb(0,0,255))' }} />
            <span className="text-sm text-gray-600 dark:text-gray-400">-1.0 → +1.0</span>
          </div>
        </div>
      </div>

      {/* Sample Values */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
          示例值 (位置 0, 前 8 维)
        </h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead>
              <tr>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                  维度
                </th>
                {Array.from({ length: Math.min(8, dModel) }).map((_, i) => (
                  <th
                    key={i}
                    className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase"
                  >
                    {i}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              <tr>
                <td className="px-4 py-2 text-sm text-gray-900 dark:text-white">值</td>
                {positionalEncoding[0].slice(0, Math.min(8, dModel)).map((value, i) => (
                  <td key={i} className="px-4 py-2 text-sm font-mono text-gray-900 dark:text-white">
                    {value.toFixed(4)}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Information */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-3 text-blue-900 dark:text-blue-100">
          💡 位置编码原理
        </h3>
        <div className="space-y-3 text-sm text-blue-800 dark:text-blue-200">
          <div>
            <strong>1. Sinusoidal (正弦) - Transformer 标准:</strong>
            <ul className="list-disc list-inside pl-4 mt-1 space-y-1">
              <li>PE(pos, 2i) = sin(pos / 10000^(2i/d_model))</li>
              <li>PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))</li>
              <li>优点: 无需训练，可以处理任意长度序列</li>
              <li>使用: Transformer, GPT-1</li>
            </ul>
          </div>

          <div>
            <strong>2. Learnable (可学习) - BERT 风格:</strong>
            <ul className="list-disc list-inside pl-4 mt-1 space-y-1">
              <li>位置编码作为可训练的 Embedding 层</li>
              <li>优点: 可以学习到特定任务的位置模式</li>
              <li>缺点: 有最大长度限制</li>
              <li>使用: BERT, GPT-2, GPT-3</li>
            </ul>
          </div>

          <div>
            <strong>3. RoPE (旋转位置编码) - LLaMA/GPT-Neo:</strong>
            <ul className="list-disc list-inside pl-4 mt-1 space-y-1">
              <li>通过旋转操作注入位置信息</li>
              <li>优点: 相对位置编码，外推性能好</li>
              <li>特点: 与 attention 计算结合，更高效</li>
              <li>使用: LLaMA, GPT-J, GPT-Neo</li>
            </ul>
          </div>

          <div className="pt-2">
            <strong>为什么需要位置编码？</strong>
            <p className="mt-1">
              Transformer 的 Self-Attention 机制本身是排列不变的（permutation-invariant），
              无法区分词序。位置编码为每个位置添加唯一的表示，让模型能够理解序列中的顺序信息。
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
