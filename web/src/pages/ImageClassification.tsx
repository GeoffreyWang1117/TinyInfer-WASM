import { useState } from 'react'

export default function ImageClassification() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [predictions, setPredictions] = useState<Array<{ label: string; confidence: number }>>([])
  const [inferenceTime, setInferenceTime] = useState<number>(0)
  const [isInferring, setIsInferring] = useState(false)

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string)
        setPredictions([])
      }
      reader.readAsDataURL(file)
    }
  }

  const runInference = async () => {
    if (!selectedImage) return

    setIsInferring(true)

    // Simulate inference
    // TODO: Replace with actual WASM inference
    await new Promise(resolve => setTimeout(resolve, 500))

    // Mock predictions
    const mockPredictions = [
      { label: '猫', confidence: 0.953 },
      { label: '老虎', confidence: 0.032 },
      { label: '狮子', confidence: 0.011 },
      { label: '豹子', confidence: 0.003 },
      { label: '猎豹', confidence: 0.001 },
    ]

    setPredictions(mockPredictions)
    setInferenceTime(45)
    setIsInferring(false)
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          图像分类
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          使用 MobileNetV2 对图片进行分类（1000 类 ImageNet）
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Image Upload Section */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            上传图片
          </h3>

          <div className="space-y-4">
            <div className="flex space-x-2">
              <label className="flex-1">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
                <div className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded cursor-pointer text-center">
                  选择图片
                </div>
              </label>
              <button className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded">
                使用摄像头
              </button>
            </div>

            {/* Image Preview */}
            <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-4 min-h-[300px] flex items-center justify-center">
              {selectedImage ? (
                <img
                  src={selectedImage}
                  alt="Selected"
                  className="max-w-full max-h-[400px] rounded"
                />
              ) : (
                <p className="text-gray-400">请上传图片或使用摄像头</p>
              )}
            </div>

            <button
              onClick={runInference}
              disabled={!selectedImage || isInferring}
              className="w-full bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white px-4 py-3 rounded font-semibold transition"
            >
              {isInferring ? '推理中...' : '开始推理'}
            </button>
          </div>
        </div>

        {/* Results Section */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            预测结果
          </h3>

          {predictions.length > 0 ? (
            <div className="space-y-4">
              <div className="space-y-3">
                {predictions.map((pred, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-900 dark:text-white font-medium">
                        {idx + 1}. {pred.label}
                      </span>
                      <span className="text-gray-600 dark:text-gray-400">
                        {(pred.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${pred.confidence * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400">推理时间:</span>
                  <span className="text-gray-900 dark:text-white font-semibold">
                    {inferenceTime}ms
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-400 py-12">
              <p>上传图片并开始推理以查看结果</p>
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
            <p className="text-gray-900 dark:text-white font-semibold">MobileNetV2 1.0</p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">参数量</p>
            <p className="text-gray-900 dark:text-white font-semibold">3.5M</p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">WASM 大小</p>
            <p className="text-gray-900 dark:text-white font-semibold">1.2MB</p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">分类数</p>
            <p className="text-gray-900 dark:text-white font-semibold">1000 类</p>
          </div>
        </div>
      </div>
    </div>
  )
}
