import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import Home from './pages/Home'
import ImageClassification from './pages/ImageClassification'
import TextEmbedding from './pages/TextEmbedding'
import ChatGeneration from './pages/ChatGeneration'
import Benchmark from './pages/Benchmark'

function App() {
  const [wasmLoaded, setWasmLoaded] = useState(false)
  const [simdSupported, setSimdSupported] = useState<boolean | null>(null)

  useEffect(() => {
    // Check SIMD support
    const checkSIMD = async () => {
      try {
        const simd = await WebAssembly.validate(
          new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11])
        )
        setSimdSupported(simd)
      } catch {
        setSimdSupported(false)
      }
    }

    checkSIMD()

    // Load WASM module
    const loadWasm = async () => {
      try {
        // We'll implement the actual WASM loading later
        // For now, just mark as loaded
        setWasmLoaded(true)
      } catch (error) {
        console.error('Failed to load WASM:', error)
      }
    }

    loadWasm()
  }, [])

  return (
    <Router>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        {/* Header */}
        <header className="bg-white dark:bg-gray-800 shadow">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                  TinyInfer-WASM
                </h1>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  浏览器端轻量级神经网络推理引擎
                </p>
              </div>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${wasmLoaded ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    WASM: {wasmLoaded ? '已加载' : '加载中...'}
                  </span>
                </div>
                {simdSupported !== null && (
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${simdSupported ? 'bg-green-500' : 'bg-red-500'}`}></div>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      SIMD: {simdSupported ? '支持' : '不支持'}
                    </span>
                  </div>
                )}
                <a
                  href="https://github.com/GeoffreyWang1117/TinyInfer-WASM"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
                >
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                  </svg>
                </a>
              </div>
            </div>
          </div>
        </header>

        {/* Navigation */}
        <nav className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex space-x-8">
              <Link
                to="/"
                className="border-b-2 border-transparent hover:border-blue-500 px-1 py-4 text-sm font-medium text-gray-900 dark:text-white"
              >
                首页
              </Link>
              <Link
                to="/image-classification"
                className="border-b-2 border-transparent hover:border-blue-500 px-1 py-4 text-sm font-medium text-gray-900 dark:text-white"
              >
                图像分类
              </Link>
              <Link
                to="/text-embedding"
                className="border-b-2 border-transparent hover:border-blue-500 px-1 py-4 text-sm font-medium text-gray-900 dark:text-white"
              >
                文本嵌入
              </Link>
              <Link
                to="/chat"
                className="border-b-2 border-transparent hover:border-blue-500 px-1 py-4 text-sm font-medium text-gray-900 dark:text-white"
              >
                对话生成
              </Link>
              <Link
                to="/benchmark"
                className="border-b-2 border-transparent hover:border-blue-500 px-1 py-4 text-sm font-medium text-gray-900 dark:text-white"
              >
                性能测试
              </Link>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/image-classification" element={<ImageClassification />} />
            <Route path="/text-embedding" element={<TextEmbedding />} />
            <Route path="/chat" element={<ChatGeneration />} />
            <Route path="/benchmark" element={<Benchmark />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <p className="text-center text-sm text-gray-600 dark:text-gray-400">
              © 2025 TinyInfer-WASM. MIT License.
            </p>
          </div>
        </footer>
      </div>
    </Router>
  )
}

export default App
