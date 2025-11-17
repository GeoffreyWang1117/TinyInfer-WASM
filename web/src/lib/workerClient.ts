/**
 * TinyInfer Web Worker Client
 *
 * High-level API for using TinyInfer in a Web Worker.
 * Provides the same interface as the main thread API, but runs in background.
 */

import type { WorkerRequest, WorkerResponse } from './worker'

export class TinyInferWorker {
  private worker: Worker | null = null
  private requestId = 0
  private pendingRequests = new Map<
    string,
    { resolve: (value: any) => void; reject: (error: Error) => void }
  >()

  /**
   * Initialize the worker
   */
  async initialize(workerPath?: string): Promise<void> {
    if (this.worker) {
      return // Already initialized
    }

    // Create worker
    const path = workerPath || new URL('./worker.ts', import.meta.url).href
    this.worker = new Worker(path, { type: 'module' })

    // Setup message handler
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const { id, success, result, error } = event.data

      // Handle ready message
      if (!id && (event.data as any).type === 'ready') {
        console.log('[TinyInferWorker] Worker ready')
        return
      }

      const pending = this.pendingRequests.get(id)
      if (!pending) {
        console.warn(`[TinyInferWorker] Received response for unknown request: ${id}`)
        return
      }

      this.pendingRequests.delete(id)

      if (success) {
        pending.resolve(result)
      } else {
        pending.reject(new Error(error || 'Unknown error'))
      }
    }

    this.worker.onerror = (error) => {
      console.error('[TinyInferWorker] Worker error:', error)

      // Reject all pending requests
      for (const [id, pending] of this.pendingRequests.entries()) {
        pending.reject(new Error('Worker error'))
        this.pendingRequests.delete(id)
      }
    }

    // Initialize WASM in worker
    await this.sendRequest('init')
  }

  /**
   * Load model from JSON
   */
  async loadModel(modelJSON: string): Promise<void> {
    await this.sendRequest('load_model', { modelJSON })
  }

  /**
   * Run inference
   */
  async infer(inputData: Float32Array | number[], inputShape: number[]): Promise<Float32Array> {
    const result = await this.sendRequest('infer', {
      inputData: inputData instanceof Float32Array ? Array.from(inputData) : inputData,
      inputShape,
    })

    return new Float32Array(result.output)
  }

  /**
   * Get model information
   */
  async getModelInfo(): Promise<{ info: string; isLoaded: boolean }> {
    return await this.sendRequest('get_model_info')
  }

  /**
   * Free resources
   */
  async free(): Promise<void> {
    await this.sendRequest('free')
  }

  /**
   * Terminate the worker
   */
  terminate(): void {
    if (this.worker) {
      this.worker.terminate()
      this.worker = null

      // Reject all pending requests
      for (const [id, pending] of this.pendingRequests.entries()) {
        pending.reject(new Error('Worker terminated'))
        this.pendingRequests.delete(id)
      }
    }
  }

  /**
   * Send request to worker
   */
  private async sendRequest(type: string, payload?: any): Promise<any> {
    if (!this.worker) {
      throw new Error('Worker not initialized. Call initialize() first.')
    }

    const id = `${this.requestId++}`

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject })

      const request: WorkerRequest = {
        id,
        type: type as any,
        payload,
      }

      this.worker!.postMessage(request)

      // Timeout after 30 seconds
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id)
          reject(new Error('Request timeout'))
        }
      }, 30000)
    })
  }

  /**
   * Check if worker is initialized
   */
  isInitialized(): boolean {
    return this.worker !== null
  }
}

/**
 * Create a new TinyInfer worker instance
 */
export async function createWorkerInstance(workerPath?: string): Promise<TinyInferWorker> {
  const worker = new TinyInferWorker()
  await worker.initialize(workerPath)
  return worker
}

export default TinyInferWorker
