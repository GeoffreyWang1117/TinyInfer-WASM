/**
 * Browser-side ONNX Model Loading Example
 *
 * This example demonstrates how to load ONNX models directly in the browser
 * without server-side conversion, with automatic IndexedDB caching.
 */

import {
  initWasm,
  createInferenceEngine,
  loadONNXFromFile,
  loadONNXFromURL,
  clearModelCache,
  getModelCacheStats,
} from '../web/src/lib/tinyinfer'

/**
 * Example 1: Load ONNX model from file upload
 */
export function setupONNXFileUpload() {
  console.log('=== Example 1: Load ONNX from File Upload ===\n')

  // HTML setup:
  // <input type="file" id="onnxFileInput" accept=".onnx">
  // <button id="uploadBtn">Load Model</button>
  // <div id="status"></div>

  const fileInput = document.getElementById('onnxFileInput') as HTMLInputElement
  const uploadBtn = document.getElementById('uploadBtn') as HTMLButtonElement
  const statusDiv = document.getElementById('status') as HTMLDivElement

  uploadBtn.addEventListener('click', async () => {
    const file = fileInput.files?.[0]

    if (!file) {
      statusDiv.textContent = 'Please select an ONNX file'
      return
    }

    try {
      statusDiv.textContent = 'Initializing WASM...'

      // Initialize WASM
      await initWasm()

      // Create inference engine
      const engine = createInferenceEngine()

      statusDiv.textContent = 'Loading ONNX model...'

      // Load ONNX model (with automatic caching)
      await loadONNXFromFile(engine, file)

      statusDiv.textContent = `✅ Model loaded: ${file.name}`

      // Check if model is loaded
      if (engine.isModelLoaded()) {
        console.log('Model info:', engine.getModelInfo())

        // Get cache statistics
        const cacheStats = await getModelCacheStats()
        console.log('Cache stats:', cacheStats)

        statusDiv.textContent += `\nCached models: ${cacheStats.count}`
      }

      // Example inference (adjust input based on your model)
      const input = new Float32Array([1, 2, 3, 4])
      const output = engine.infer(input, [4])
      console.log('Inference output:', output)

      // Clean up
      engine.free()
    } catch (error) {
      console.error('Error:', error)
      statusDiv.textContent = `❌ Error: ${error}`
    }
  })
}

/**
 * Example 2: Load ONNX model from URL
 */
export async function loadONNXFromRemote() {
  console.log('=== Example 2: Load ONNX from URL ===\n')

  try {
    // Initialize WASM
    await initWasm()

    // Create inference engine
    const engine = createInferenceEngine()

    // Load ONNX model from URL (with caching)
    const modelURL = 'https://example.com/models/my_model.onnx'
    console.log('Loading model from:', modelURL)

    await loadONNXFromURL(engine, modelURL)

    console.log('✅ Model loaded successfully')
    console.log('Model info:', engine.getModelInfo())

    // Run inference
    const input = new Float32Array([1, 2, 3, 4])
    const output = engine.infer(input, [4])
    console.log('Output:', output)

    // Clean up
    engine.free()
  } catch (error) {
    console.error('Failed to load ONNX model:', error)
  }
}

/**
 * Example 3: Cache management
 */
export async function manageCacheExample() {
  console.log('=== Example 3: Cache Management ===\n')

  try {
    // Get current cache statistics
    const stats = await getModelCacheStats()

    console.log('Cache Statistics:')
    console.log(`  Total models: ${stats.count}`)
    console.log(`  Total size: ${(stats.totalSize / 1024 / 1024).toFixed(2)} MB`)

    console.log('\nCached models:')
    for (const model of stats.models) {
      const date = new Date(model.timestamp).toLocaleString()
      const size = (model.size / 1024 / 1024).toFixed(2)
      console.log(`  - ${model.name} (${size} MB) - Cached at: ${date}`)
    }

    // Clear cache if needed
    // await clearModelCache()
    // console.log('✅ Cache cleared')
  } catch (error) {
    console.error('Error managing cache:', error)
  }
}

/**
 * Example 4: Disable caching for a specific load
 */
export async function loadWithoutCache() {
  console.log('=== Example 4: Load without Cache ===\n')

  try {
    await initWasm()
    const engine = createInferenceEngine()

    const modelURL = 'https://example.com/models/model.onnx'

    // Load without using cache
    await loadONNXFromURL(engine, modelURL, { useCache: false })

    console.log('✅ Model loaded (cache bypassed)')

    engine.free()
  } catch (error) {
    console.error('Error:', error)
  }
}

/**
 * Example 5: Complete browser application with ONNX loading
 */
export function createONNXApp() {
  console.log('=== Example 5: Complete ONNX App ===\n')

  const html = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>TinyInfer ONNX Loader</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          max-width: 800px;
          margin: 50px auto;
          padding: 20px;
        }
        .upload-area {
          border: 2px dashed #ccc;
          border-radius: 8px;
          padding: 40px;
          text-align: center;
          margin: 20px 0;
        }
        .upload-area.dragover {
          border-color: #4CAF50;
          background-color: #f1f8f4;
        }
        button {
          background-color: #4CAF50;
          color: white;
          padding: 12px 24px;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 16px;
          margin: 5px;
        }
        button:hover {
          background-color: #45a049;
        }
        button:disabled {
          background-color: #ccc;
          cursor: not-allowed;
        }
        #status {
          margin: 20px 0;
          padding: 15px;
          border-radius: 4px;
        }
        .success {
          background-color: #d4edda;
          color: #155724;
        }
        .error {
          background-color: #f8d7da;
          color: #721c24;
        }
        .info {
          background-color: #d1ecf1;
          color: #0c5460;
        }
        #cacheStats {
          margin-top: 20px;
          padding: 15px;
          background-color: #f5f5f5;
          border-radius: 4px;
        }
      </style>
    </head>
    <body>
      <h1>🚀 TinyInfer ONNX Model Loader</h1>
      <p>Load ONNX models directly in your browser - no server required!</p>

      <div class="upload-area" id="dropZone">
        <h3>📁 Upload ONNX Model</h3>
        <input type="file" id="fileInput" accept=".onnx" style="display: none">
        <p>Drag and drop an ONNX file here, or</p>
        <button onclick="document.getElementById('fileInput').click()">
          Choose File
        </button>
      </div>

      <div>
        <h3>🌐 Load from URL</h3>
        <input
          type="text"
          id="urlInput"
          placeholder="https://example.com/model.onnx"
          style="width: 60%; padding: 10px; margin-right: 10px;"
        >
        <button id="loadUrlBtn">Load from URL</button>
      </div>

      <div id="status"></div>

      <div id="cacheStats"></div>

      <div style="margin-top: 20px;">
        <button id="refreshCacheBtn">🔄 Refresh Cache Stats</button>
        <button id="clearCacheBtn" style="background-color: #f44336;">
          🗑️ Clear Cache
        </button>
      </div>

      <script type="module">
        import {
          initWasm,
          createInferenceEngine,
          loadONNXFromFile,
          loadONNXFromURL,
          clearModelCache,
          getModelCacheStats,
        } from './lib/tinyinfer.js'

        let engine = null

        // Initialize WASM on page load
        initWasm().then(() => {
          console.log('WASM initialized')
          updateCacheStats()
        })

        // File upload handling
        const fileInput = document.getElementById('fileInput')
        const dropZone = document.getElementById('dropZone')
        const statusDiv = document.getElementById('status')

        fileInput.addEventListener('change', handleFileSelect)

        // Drag and drop
        dropZone.addEventListener('dragover', (e) => {
          e.preventDefault()
          dropZone.classList.add('dragover')
        })

        dropZone.addEventListener('dragleave', () => {
          dropZone.classList.remove('dragover')
        })

        dropZone.addEventListener('drop', async (e) => {
          e.preventDefault()
          dropZone.classList.remove('dragover')

          const file = e.dataTransfer.files[0]
          if (file && file.name.endsWith('.onnx')) {
            await loadModel(file)
          } else {
            showStatus('Please drop an ONNX file', 'error')
          }
        })

        async function handleFileSelect(event) {
          const file = event.target.files[0]
          if (file) {
            await loadModel(file)
          }
        }

        async function loadModel(file) {
          try {
            showStatus('Loading model...', 'info')

            engine = createInferenceEngine()
            await loadONNXFromFile(engine, file)

            showStatus(\`✅ Model loaded: \${file.name}\`, 'success')

            console.log('Model info:', engine.getModelInfo())
            await updateCacheStats()
          } catch (error) {
            showStatus(\`❌ Error: \${error.message}\`, 'error')
          }
        }

        // URL loading
        document.getElementById('loadUrlBtn').addEventListener('click', async () => {
          const url = document.getElementById('urlInput').value
          if (!url) {
            showStatus('Please enter a URL', 'error')
            return
          }

          try {
            showStatus('Loading model from URL...', 'info')

            engine = createInferenceEngine()
            await loadONNXFromURL(engine, url)

            showStatus(\`✅ Model loaded from: \${url}\`, 'success')

            await updateCacheStats()
          } catch (error) {
            showStatus(\`❌ Error: \${error.message}\`, 'error')
          }
        })

        // Cache management
        document.getElementById('refreshCacheBtn').addEventListener('click', updateCacheStats)

        document.getElementById('clearCacheBtn').addEventListener('click', async () => {
          if (confirm('Are you sure you want to clear all cached models?')) {
            await clearModelCache()
            showStatus('✅ Cache cleared', 'success')
            await updateCacheStats()
          }
        })

        async function updateCacheStats() {
          const stats = await getModelCacheStats()
          const cacheDiv = document.getElementById('cacheStats')

          const totalSizeMB = (stats.totalSize / 1024 / 1024).toFixed(2)

          let html = \`
            <h3>💾 Cache Statistics</h3>
            <p>Total models: \${stats.count} | Total size: \${totalSizeMB} MB</p>
          \`

          if (stats.count > 0) {
            html += '<ul>'
            for (const model of stats.models) {
              const date = new Date(model.timestamp).toLocaleString()
              const size = (model.size / 1024 / 1024).toFixed(2)
              html += \`<li>\${model.name} (\${size} MB) - Cached: \${date}</li>\`
            }
            html += '</ul>'
          }

          cacheDiv.innerHTML = html
        }

        function showStatus(message, type) {
          statusDiv.textContent = message
          statusDiv.className = type
        }
      </script>
    </body>
    </html>
  `

  console.log('HTML template for ONNX app:')
  console.log(html)
}

/**
 * Run all examples (for demonstration)
 */
if (typeof window !== 'undefined') {
  // Browser environment
  console.log('Browser ONNX examples loaded')
  console.log('Available functions:')
  console.log('  - setupONNXFileUpload()')
  console.log('  - loadONNXFromRemote()')
  console.log('  - manageCacheExample()')
  console.log('  - loadWithoutCache()')
  console.log('  - createONNXApp()')
}
