import { useState, useEffect } from 'react'
import { TinyInferWorker } from '../lib/workerClient'

/**
 * Worker Demo Page
 *
 * Demonstrates Phase 8 Web Worker capabilities:
 * - Background inference without blocking UI
 * - Parallel inference execution
 * - UI responsiveness during computation
 */

export default function WorkerDemo() {
  const [worker, setWorker] = useState<TinyInferWorker | null>(null)
  const [workerStatus, setWorkerStatus] = useState<string>('Not initialized')
  const [isRunning, setIsRunning] = useState(false)
  const [results, setResults] = useState<Array<{ id: number; time: number; result: string }>>([])
  const [uiCounter, setUiCounter] = useState(0)
  const [parallelCount, setParallelCount] = useState(5)

  // UI counter to show responsiveness
  useEffect(() => {
    const interval = setInterval(() => {
      setUiCounter(c => c + 1)
    }, 100)
    return () => clearInterval(interval)
  }, [])

  // Initialize worker
  const initializeWorker = async () => {
    try {
      setWorkerStatus('Initializing...')
      const newWorker = new TinyInferWorker()
      await newWorker.initialize()
      setWorker(newWorker)
      setWorkerStatus('Worker initialized')

      // Load a simple model
      setWorkerStatus('Loading model...')
      const simpleModel = JSON.stringify({
        version: '0.8.0',
        name: 'worker_test',
        graph: {
          nodes: [
            { id: 'input', op_type: 'Input', inputs: [], outputs: ['x'], attributes: {} },
            { id: 'relu', op_type: 'ReLU', inputs: ['x'], outputs: ['y'], attributes: {} },
            { id: 'output', op_type: 'Output', inputs: ['y'], outputs: [], attributes: {} },
          ],
          edges: [
            { from: 'input', to: 'relu' },
            { from: 'relu', to: 'output' },
          ],
          inputs: ['x'],
          outputs: ['y'],
        },
        weights: {},
      })

      await newWorker.loadModel(simpleModel)
      setWorkerStatus('Ready')
    } catch (error) {
      setWorkerStatus(`Error: ${error}`)
      console.error('Worker initialization failed:', error)
    }
  }

  // Run single inference
  const runSingleInference = async () => {
    if (!worker) {
      alert('Please initialize worker first')
      return
    }

    try {
      setIsRunning(true)
      const start = performance.now()

      const input = new Float32Array([1.0, -2.0, 3.0, -4.0])
      const output = await worker.infer(input, [4])

      const time = performance.now() - start

      setResults(prev => [
        ...prev,
        {
          id: prev.length + 1,
          time,
          result: `[${Array.from(output).map(v => v.toFixed(2)).join(', ')}]`,
        },
      ])
    } catch (error) {
      console.error('Inference failed:', error)
      alert(`Inference failed: ${error}`)
    } finally {
      setIsRunning(false)
    }
  }

  // Run parallel inferences
  const runParallelInferences = async () => {
    if (!worker) {
      alert('Please initialize worker first')
      return
    }

    try {
      setIsRunning(true)
      setResults([])

      const start = performance.now()
      const promises = []

      // Launch multiple inferences in parallel
      for (let i = 0; i < parallelCount; i++) {
        const input = new Float32Array(Array(4).fill(0).map(() => Math.random() * 10 - 5))
        promises.push(
          worker.infer(input, [4]).then(output => ({
            id: i + 1,
            time: 0,
            result: `[${Array.from(output).map(v => v.toFixed(2)).join(', ')}]`,
          }))
        )
      }

      // Wait for all to complete
      const inferenceResults = await Promise.all(promises)
      const totalTime = performance.now() - start

      setResults(inferenceResults.map((r, i) => ({
        ...r,
        time: totalTime / parallelCount,
      })))

      setWorkerStatus(`Completed ${parallelCount} parallel inferences in ${totalTime.toFixed(2)}ms`)
    } catch (error) {
      console.error('Parallel inference failed:', error)
      alert(`Parallel inference failed: ${error}`)
    } finally {
      setIsRunning(false)
    }
  }

  // Terminate worker
  const terminateWorker = () => {
    if (worker) {
      worker.terminate()
      setWorker(null)
      setWorkerStatus('Worker terminated')
      setResults([])
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          Web Worker Demonstration
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          Phase 8 feature: Background inference without blocking the UI
        </p>
      </div>

      {/* UI Responsiveness Indicator */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg shadow p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold mb-1">UI Responsiveness Test</h3>
            <p className="text-sm opacity-90">
              This counter updates every 100ms. If it freezes during inference, the UI is blocked.
            </p>
          </div>
          <div className="text-right">
            <div className="text-5xl font-bold tabular-nums">{uiCounter}</div>
            <div className="text-sm opacity-90 mt-1">
              {isRunning ? '⚡ Inference Running' : '✓ UI Responsive'}
            </div>
          </div>
        </div>
      </div>

      {/* Worker Control Panel */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
          Worker Controls
        </h3>

        <div className="space-y-4">
          {/* Status */}
          <div className="flex items-center space-x-3">
            <div
              className={`w-3 h-3 rounded-full ${
                worker ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
              }`}
            />
            <span className="text-gray-700 dark:text-gray-300">Status: {workerStatus}</span>
          </div>

          {/* Buttons */}
          <div className="flex flex-wrap gap-3">
            <button
              onClick={initializeWorker}
              disabled={worker !== null || isRunning}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Initialize Worker
            </button>

            <button
              onClick={runSingleInference}
              disabled={!worker || isRunning}
              className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Run Single Inference
            </button>

            <button
              onClick={runParallelInferences}
              disabled={!worker || isRunning}
              className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Run {parallelCount} Parallel Inferences
            </button>

            <button
              onClick={terminateWorker}
              disabled={!worker || isRunning}
              className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Terminate Worker
            </button>
          </div>

          {/* Parallel Count Control */}
          <div className="flex items-center space-x-4">
            <label className="text-gray-700 dark:text-gray-300">Parallel Count:</label>
            <input
              type="range"
              min="1"
              max="20"
              value={parallelCount}
              onChange={e => setParallelCount(parseInt(e.target.value))}
              className="flex-1 max-w-xs"
              disabled={isRunning}
            />
            <span className="text-gray-700 dark:text-gray-300 font-mono w-8 text-right">
              {parallelCount}
            </span>
          </div>
        </div>
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
            Inference Results
          </h3>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead>
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    ID
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Time (ms)
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Output
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {results.map(result => (
                  <tr key={result.id}>
                    <td className="px-4 py-2 text-sm text-gray-900 dark:text-white">
                      #{result.id}
                    </td>
                    <td className="px-4 py-2 text-sm text-gray-900 dark:text-white">
                      {result.time.toFixed(2)}
                    </td>
                    <td className="px-4 py-2 text-sm font-mono text-gray-900 dark:text-white">
                      {result.result}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
            Total results: {results.length}
          </div>
        </div>
      )}

      {/* Information */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-3 text-blue-900 dark:text-blue-100">
          💡 How It Works
        </h3>
        <ul className="space-y-2 text-sm text-blue-800 dark:text-blue-200">
          <li className="flex items-start space-x-2">
            <span className="font-bold mt-0.5">•</span>
            <span>
              <strong>Web Workers</strong> run in a separate thread, keeping the UI responsive
            </span>
          </li>
          <li className="flex items-start space-x-2">
            <span className="font-bold mt-0.5">•</span>
            <span>
              <strong>Transferable Objects</strong> enable zero-copy data transfer for better
              performance
            </span>
          </li>
          <li className="flex items-start space-x-2">
            <span className="font-bold mt-0.5">•</span>
            <span>
              <strong>Message Queue</strong> handles multiple concurrent inference requests
            </span>
          </li>
          <li className="flex items-start space-x-2">
            <span className="font-bold mt-0.5">•</span>
            <span>
              Watch the counter above - it should continue updating smoothly even during inference
            </span>
          </li>
        </ul>
      </div>
    </div>
  )
}
