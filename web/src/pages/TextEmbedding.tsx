import { useState } from 'react'

export default function TextEmbedding() {
  const [text1, setText1] = useState('')
  const [text2, setText2] = useState('')
  const [embedding1, setEmbedding1] = useState<number[] | null>(null)
  const [embedding2, setEmbedding2] = useState<number[] | null>(null)
  const [similarity, setSimilarity] = useState<number | null>(null)
  const [inferenceTime, setInferenceTime] = useState<number>(0)
  const [isProcessing, setIsProcessing] = useState(false)

  const generateEmbedding = async () => {
    if (!text1) return

    setIsProcessing(true)

    // Simulate embedding generation
    await new Promise(resolve => setTimeout(resolve, 300))

    // Mock embedding (384 dimensions)
    const mockEmbedding = Array.from({ length: 384 }, () => Math.random() * 2 - 1)
    setEmbedding1(mockEmbedding)
    setInferenceTime(23)
    setIsProcessing(false)
  }

  const calculateSimilarity = async () => {
    if (!text1 || !text2) return

    setIsProcessing(true)

    // Simulate embedding generation and similarity calculation
    await new Promise(resolve => setTimeout(resolve, 500))

    // Mock embeddings
    const emb1 = Array.from({ length: 384 }, () => Math.random() * 2 - 1)
    const emb2 = Array.from({ length: 384 }, () => Math.random() * 2 - 1)

    setEmbedding1(emb1)
    setEmbedding2(emb2)

    // Mock cosine similarity
    const sim = 0.75 + Math.random() * 0.2
    setSimilarity(sim)
    setInferenceTime(46)
    setIsProcessing(false)
  }

  const exampleTexts = [
    'I love machine learning!',
    'Machine learning is awesome!',
    'The weather is nice today.',
    'Natural language processing is fascinating.',
  ]

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          文本嵌入
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          使用 MiniLM-L6 生成文本嵌入向量并计算相似度
        </p>
      </div>

      {/* Text Input */}
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          输入文本
        </h3>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              文本 1
            </label>
            <textarea
              value={text1}
              onChange={(e) => setText1(e.target.value)}
              placeholder="输入第一段文本..."
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={3}
            />
          </div>

          <div className="flex space-x-2">
            <button
              onClick={generateEmbedding}
              disabled={!text1 || isProcessing}
              className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 text-white px-6 py-2 rounded transition"
            >
              生成嵌入
            </button>
            <button
              onClick={() => setText1(exampleTexts[Math.floor(Math.random() * exampleTexts.length)])}
              className="bg-gray-500 hover:bg-gray-600 text-white px-6 py-2 rounded transition"
            >
              使用示例
            </button>
          </div>
        </div>
      </div>

      {/* Embedding Visualization */}
      {embedding1 && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            嵌入向量 (384维)
          </h3>

          <div className="space-y-4">
            <div className="bg-gray-100 dark:bg-gray-900 p-4 rounded overflow-x-auto">
              <code className="text-sm text-gray-800 dark:text-gray-200">
                [{embedding1.slice(0, 10).map(v => v.toFixed(3)).join(', ')}, ...]
              </code>
            </div>

            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                向量分布可视化（前 100 维）
              </p>
              <div className="flex items-end h-32 space-x-1">
                {embedding1.slice(0, 100).map((val, idx) => (
                  <div
                    key={idx}
                    className="flex-1 bg-blue-500 rounded-t"
                    style={{
                      height: `${(Math.abs(val) * 50)}%`,
                      opacity: 0.7,
                    }}
                  />
                ))}
              </div>
            </div>

            <div className="text-sm text-gray-600 dark:text-gray-400">
              推理时间: <span className="text-gray-900 dark:text-white font-semibold">{inferenceTime}ms</span>
            </div>
          </div>
        </div>
      )}

      {/* Similarity Calculation */}
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          计算相似度
        </h3>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              文本 2
            </label>
            <textarea
              value={text2}
              onChange={(e) => setText2(e.target.value)}
              placeholder="输入第二段文本进行对比..."
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={3}
            />
          </div>

          <button
            onClick={calculateSimilarity}
            disabled={!text1 || !text2 || isProcessing}
            className="bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white px-6 py-2 rounded transition"
          >
            {isProcessing ? '计算中...' : '计算相似度'}
          </button>

          {similarity !== null && (
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900 dark:to-purple-900 dark:bg-opacity-20 p-6 rounded-lg">
              <div className="text-center">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  余弦相似度
                </p>
                <p className="text-5xl font-bold text-gray-900 dark:text-white mb-2">
                  {similarity.toFixed(3)}
                </p>
                <div className="w-full max-w-md mx-auto bg-gray-200 dark:bg-gray-700 rounded-full h-3 mt-4">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-300"
                    style={{ width: `${similarity * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Model Info */}
      <div className="bg-blue-50 dark:bg-blue-900 dark:bg-opacity-20 p-6 rounded-lg">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
          模型信息
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-gray-600 dark:text-gray-400">模型</p>
            <p className="text-gray-900 dark:text-white font-semibold">MiniLM-L6</p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">参数量</p>
            <p className="text-gray-900 dark:text-white font-semibold">22.7M</p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">嵌入维度</p>
            <p className="text-gray-900 dark:text-white font-semibold">384</p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">最大长度</p>
            <p className="text-gray-900 dark:text-white font-semibold">512 tokens</p>
          </div>
        </div>
      </div>
    </div>
  )
}
