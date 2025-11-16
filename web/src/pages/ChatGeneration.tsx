import { useState } from 'react'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export default function ChatGeneration() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: '你好！我是 TinyLM，一个运行在浏览器中的轻量级对话模型。有什么我可以帮助你的吗？' }
  ])
  const [input, setInput] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [temperature, setTemperature] = useState(0.7)
  const [maxLength, setMaxLength] = useState(100)
  const [topK, setTopK] = useState(50)
  const [stats, setStats] = useState({ tokens: 0, speed: 0 })

  const sendMessage = async () => {
    if (!input.trim() || isGenerating) return

    const userMessage: Message = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsGenerating(true)

    // Simulate response generation
    await new Promise(resolve => setTimeout(resolve, 1000))

    const responses = [
      '这是一个很好的问题！让我来帮你解答。',
      '我理解你的意思。作为一个轻量级模型，我会尽力回答。',
      '感谢你的提问！这个话题很有趣。',
      '让我想想...这个问题需要仔细考虑。',
    ]

    const assistantMessage: Message = {
      role: 'assistant',
      content: responses[Math.floor(Math.random() * responses.length)]
    }

    setMessages(prev => [...prev, assistantMessage])
    setStats({
      tokens: stats.tokens + 15,
      speed: 12 + Math.random() * 3
    })
    setIsGenerating(false)
  }

  const clearChat = () => {
    setMessages([
      { role: 'assistant', content: '你好！我是 TinyLM，一个运行在浏览器中的轻量级对话模型。有什么我可以帮助你的吗？' }
    ])
    setStats({ tokens: 0, speed: 0 })
  }

  const exportChat = () => {
    const chatText = messages.map(m => `${m.role}: ${m.content}`).join('\n\n')
    const blob = new Blob([chatText], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'chat-export.txt'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          对话生成
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          使用轻量级语言模型进行对话（TinyLM 160M）
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chat Area */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-lg shadow">
          {/* Messages */}
          <div className="h-[500px] overflow-y-auto p-6 space-y-4">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-4 ${
                    msg.role === 'user'
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white'
                  }`}
                >
                  <div className="flex items-start space-x-2">
                    <span className="text-xl">
                      {msg.role === 'user' ? '👤' : '🤖'}
                    </span>
                    <p className="flex-1">{msg.content}</p>
                  </div>
                </div>
              </div>
            ))}

            {isGenerating && (
              <div className="flex justify-start">
                <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center space-x-2">
                    <span className="text-xl">🤖</span>
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <div className="border-t border-gray-200 dark:border-gray-700 p-4">
            <div className="flex space-x-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                placeholder="输入消息..."
                className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={isGenerating}
              />
              <button
                onClick={sendMessage}
                disabled={!input.trim() || isGenerating}
                className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg transition"
              >
                发送
              </button>
            </div>
          </div>
        </div>

        {/* Settings & Stats */}
        <div className="space-y-6">
          {/* Settings */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              参数设置
            </h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Temperature: {temperature.toFixed(1)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full"
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  控制输出的随机性
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Max Length: {maxLength}
                </label>
                <input
                  type="range"
                  min="50"
                  max="200"
                  step="10"
                  value={maxLength}
                  onChange={(e) => setMaxLength(parseInt(e.target.value))}
                  className="w-full"
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  最大生成长度（tokens）
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Top-k: {topK}
                </label>
                <input
                  type="range"
                  min="1"
                  max="100"
                  step="1"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full"
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  采样时保留的候选数
                </p>
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              统计信息
            </h3>

            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">消息数:</span>
                <span className="text-gray-900 dark:text-white font-semibold">
                  {messages.length}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">已生成 tokens:</span>
                <span className="text-gray-900 dark:text-white font-semibold">
                  {stats.tokens}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">平均速度:</span>
                <span className="text-gray-900 dark:text-white font-semibold">
                  {stats.speed.toFixed(1)} tok/s
                </span>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 space-y-2">
              <button
                onClick={clearChat}
                className="w-full bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded transition"
              >
                清空对话
              </button>
              <button
                onClick={exportChat}
                className="w-full bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded transition"
              >
                导出对话
              </button>
            </div>
          </div>
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
            <p className="text-gray-900 dark:text-white font-semibold">TinyLM 160M</p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">参数量</p>
            <p className="text-gray-900 dark:text-white font-semibold">160M</p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">上下文长度</p>
            <p className="text-gray-900 dark:text-white font-semibold">2048 tokens</p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">词汇表大小</p>
            <p className="text-gray-900 dark:text-white font-semibold">50K</p>
          </div>
        </div>
      </div>
    </div>
  )
}
