export default function Home() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="text-center">
        <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          欢迎来到 TinyInfer-WASM
        </h2>
        <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
          高性能 WebAssembly 推理引擎，在浏览器中运行神经网络模型
        </p>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-12">
        {/* Feature 1 */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <div className="text-blue-500 text-3xl mb-4">🔒</div>
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            隐私保护
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            数据不离开浏览器，完全本地计算，保护您的隐私
          </p>
        </div>

        {/* Feature 2 */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <div className="text-green-500 text-3xl mb-4">⚡</div>
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            高性能
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            SIMD 加速，性能达到纯 JavaScript 的 10-20 倍
          </p>
        </div>

        {/* Feature 3 */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <div className="text-purple-500 text-3xl mb-4">🌐</div>
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            跨平台
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            一次编写，在任何现代浏览器中运行
          </p>
        </div>

        {/* Feature 4 */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <div className="text-yellow-500 text-3xl mb-4">📦</div>
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            轻量级
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            WASM 二进制文件小于 1MB，快速加载
          </p>
        </div>

        {/* Feature 5 */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <div className="text-red-500 text-3xl mb-4">🚀</div>
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            零延迟
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            无需网络请求，毫秒级响应
          </p>
        </div>

        {/* Feature 6 */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <div className="text-indigo-500 text-3xl mb-4">🎯</div>
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            易使用
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            简洁的 JavaScript API，快速集成
          </p>
        </div>
      </div>

      {/* Quick Start */}
      <div className="bg-white dark:bg-gray-800 p-8 rounded-lg shadow mt-12">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          快速开始
        </h3>
        <div className="space-y-4">
          <div>
            <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              1. 安装依赖
            </h4>
            <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded overflow-x-auto">
              <code className="text-sm text-gray-800 dark:text-gray-200">
                npm install tinyinfer-wasm
              </code>
            </pre>
          </div>

          <div>
            <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              2. 使用示例
            </h4>
            <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded overflow-x-auto">
              <code className="text-sm text-gray-800 dark:text-gray-200">
{`import { TinyInfer } from 'tinyinfer-wasm';

// 加载模型
const model = await TinyInfer.load('model.onnx');

// 执行推理
const input = new Float32Array([...]);
const output = model.infer(input);`}
              </code>
            </pre>
          </div>
        </div>
      </div>

      {/* Demo Links */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-8 rounded-lg shadow mt-12 text-white">
        <h3 className="text-2xl font-bold mb-4">探索演示</h3>
        <p className="mb-6">
          查看 TinyInfer-WASM 的实际应用场景
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <a
            href="/image-classification"
            className="bg-white bg-opacity-20 hover:bg-opacity-30 p-4 rounded-lg transition"
          >
            <div className="text-2xl mb-2">🖼️</div>
            <h4 className="font-semibold">图像分类</h4>
            <p className="text-sm mt-1 opacity-90">使用 MobileNetV2</p>
          </a>
          <a
            href="/text-embedding"
            className="bg-white bg-opacity-20 hover:bg-opacity-30 p-4 rounded-lg transition"
          >
            <div className="text-2xl mb-2">📝</div>
            <h4 className="font-semibold">文本嵌入</h4>
            <p className="text-sm mt-1 opacity-90">使用 MiniLM</p>
          </a>
          <a
            href="/chat"
            className="bg-white bg-opacity-20 hover:bg-opacity-30 p-4 rounded-lg transition"
          >
            <div className="text-2xl mb-2">💬</div>
            <h4 className="font-semibold">对话生成</h4>
            <p className="text-sm mt-1 opacity-90">使用轻量级 LLM</p>
          </a>
        </div>
      </div>
    </div>
  )
}
