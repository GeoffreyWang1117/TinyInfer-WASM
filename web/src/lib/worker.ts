/**
 * Web Worker for TinyInfer Background Inference
 *
 * Runs inference in a background thread to avoid blocking the main UI thread.
 * This is especially useful for large models or long sequences.
 */

import { initWasm, createInferenceEngine, loadModelFromJSON } from './tinyinfer'
import type { TinyInferInstance } from './tinyinfer'

// Message types
export type WorkerMessageType =
  | 'init'
  | 'load_model'
  | 'infer'
  | 'get_model_info'
  | 'free'

export interface WorkerRequest {
  id: string
  type: WorkerMessageType
  payload?: any
}

export interface WorkerResponse {
  id: string
  success: boolean
  result?: any
  error?: string
}

// Worker state
let engine: TinyInferInstance | null = null
let initialized = false

/**
 * Handle incoming messages from main thread
 */
self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const { id, type, payload } = event.data

  try {
    let result: any

    switch (type) {
      case 'init':
        if (!initialized) {
          await initWasm()
          initialized = true
        }
        result = { initialized: true }
        break

      case 'load_model':
        if (!initialized) {
          throw new Error('Worker not initialized. Call init() first.')
        }

        // Create or reuse engine
        if (!engine) {
          engine = createInferenceEngine()
        }

        // Load model
        await loadModelFromJSON(engine, payload.modelJSON)
        result = { loaded: true }
        break

      case 'infer':
        if (!engine) {
          throw new Error('No model loaded. Call load_model() first.')
        }

        const { inputData, inputShape } = payload

        // Convert plain array to Float32Array if needed
        const input =
          inputData instanceof Float32Array ? inputData : new Float32Array(inputData)

        // Run inference
        const output = engine.infer(input, inputShape)

        // Transfer output back (using transferable objects for efficiency)
        result = {
          output: output,
          shape: inputShape, // Simplified - should compute actual output shape
        }

        // Post response with transferable object
        const response: WorkerResponse = {
          id,
          success: true,
          result,
        }

        self.postMessage(response, [output.buffer])
        return // Early return to avoid duplicate postMessage

      case 'get_model_info':
        if (!engine) {
          throw new Error('No model loaded.')
        }

        result = {
          info: engine.getModelInfo(),
          isLoaded: engine.isModelLoaded(),
        }
        break

      case 'free':
        if (engine) {
          engine.free()
          engine = null
        }
        result = { freed: true }
        break

      default:
        throw new Error(`Unknown message type: ${type}`)
    }

    // Send success response
    const response: WorkerResponse = {
      id,
      success: true,
      result,
    }

    self.postMessage(response)
  } catch (error) {
    // Send error response
    const response: WorkerResponse = {
      id,
      success: false,
      error: error instanceof Error ? error.message : String(error),
    }

    self.postMessage(response)
  }
}

// Worker is ready
self.postMessage({ type: 'ready' })
