/**
 * TinyInfer WASM SDK
 * TypeScript wrapper for TinyInfer WASM module
 */

// Import the generated WASM module
import init, {
  TinyInfer as WasmTinyInfer,
  Benchmark as WasmBenchmark,
  version as wasmVersion,
  check_simd_support as wasmCheckSimdSupport,
  get_system_info as wasmGetSystemInfo,
} from '../../public/wasm/tinyinfer_core.js'

export interface WasmModule {
  TinyInfer: new () => TinyInferInstance;
  Benchmark: new () => BenchmarkInstance;
  Tensor: new (data: Float32Array, dims: number[]) => TensorInstance;
  version: () => string;
  check_simd_support: () => boolean;
  get_system_info: () => string;
  createTensor: (data: Float32Array, dims: number[]) => TensorInstance;
}

export interface TinyInferInstance {
  loadTestModel: () => void;
  loadModelFromJSON: (json: string) => void;
  isModelLoaded: () => boolean;
  infer: (inputData: Float32Array, inputShape: number[]) => Float32Array;
  getModelInfo: () => string;
  free: () => void;
}

export interface TensorInstance {
  getShape: () => number[];
  getData: () => Float32Array;
  getSize: () => number;
  toString: () => string;
  free: () => void;
}

export interface BenchmarkInstance {
  benchmarkMatMul: (size: number, iterations: number) => number;
  benchmarkReLU: (size: number, iterations: number) => number;
  benchmarkConv2D: (batch: number, channels: number, size: number, iterations: number) => number;
  getResults: () => Float64Array;
  clearResults: () => void;
  free: () => void;
}

export interface SystemInfo {
  version: string;
  simd_support: boolean;
  target: string;
}

let wasmModule: WasmModule | null = null;

/**
 * Initialize TinyInfer WASM module
 */
export async function initWasm(): Promise<WasmModule> {
  if (wasmModule) {
    return wasmModule;
  }

  try {
    console.log('Initializing TinyInfer WASM module...');

    // Initialize the WASM module
    await init()

    // Create wrapper module
    wasmModule = {
      TinyInfer: WasmTinyInfer as any,
      Benchmark: WasmBenchmark as any,
      Tensor: null as any, // Not exposed in current WASM API
      version: wasmVersion,
      check_simd_support: wasmCheckSimdSupport,
      get_system_info: wasmGetSystemInfo,
      createTensor: (data: Float32Array, dims: number[]) => {
        throw new Error('createTensor not yet implemented in WASM')
      },
    }

    console.log('TinyInfer WASM initialized successfully');
    console.log('Version:', wasmVersion());
    console.log('SIMD Support:', wasmCheckSimdSupport());

    return wasmModule;
  } catch (error) {
    console.error('Failed to initialize WASM:', error);
    // Fallback to mock implementation for development
    console.warn('Falling back to mock implementation');
    wasmModule = createMockWasm();
    return wasmModule;
  }
}

/**
 * Get the loaded WASM module
 */
export function getWasm(): WasmModule {
  if (!wasmModule) {
    throw new Error('WASM module not initialized. Call initWasm() first.');
  }
  return wasmModule;
}

/**
 * Check if WASM is loaded
 */
export function isWasmLoaded(): boolean {
  return wasmModule !== null;
}

/**
 * Get TinyInfer version
 */
export function getVersion(): string {
  return getWasm().version();
}

/**
 * Check SIMD support
 */
export function checkSimdSupport(): boolean {
  return getWasm().check_simd_support();
}

/**
 * Get system information
 */
export function getSystemInfo(): SystemInfo {
  const info = getWasm().get_system_info();
  return JSON.parse(info);
}

/**
 * Create a TinyInfer instance
 */
export function createInferenceEngine(): TinyInferInstance {
  return new (getWasm().TinyInfer)();
}

/**
 * Create a Benchmark instance
 */
export function createBenchmark(): BenchmarkInstance {
  return new (getWasm().Benchmark)();
}

/**
 * Create a tensor
 */
export function createTensor(data: number[] | Float32Array, shape: number[]): TensorInstance {
  const float32Data = data instanceof Float32Array ? data : new Float32Array(data);
  return new (getWasm().Tensor)(float32Data, shape);
}

/**
 * Load model from JSON string
 */
export async function loadModelFromJSON(engine: TinyInferInstance, json: string): Promise<void> {
  try {
    engine.loadModelFromJSON(json);
  } catch (error) {
    console.error('Failed to load model from JSON:', error);
    throw new Error(`Model loading failed: ${error}`);
  }
}

/**
 * Load model from URL
 */
export async function loadModelFromURL(engine: TinyInferInstance, url: string): Promise<void> {
  try {
    console.log(`Loading model from URL: ${url}`);
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const json = await response.text();
    await loadModelFromJSON(engine, json);
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Failed to load model from URL:', error);
    throw new Error(`Failed to load model from ${url}: ${error}`);
  }
}

/**
 * Load model from File object (user upload)
 */
export async function loadModelFromFile(engine: TinyInferInstance, file: File): Promise<void> {
  try {
    console.log(`Loading model from file: ${file.name}`);
    const text = await file.text();
    await loadModelFromJSON(engine, text);
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Failed to load model from file:', error);
    throw new Error(`Failed to load model from file: ${error}`);
  }
}

// ============================================================================
// Mock implementation for development (will be removed when WASM is compiled)
// ============================================================================

function createMockWasm(): WasmModule {
  class MockTinyInfer implements TinyInferInstance {
    private initialized = false;

    loadTestModel(): void {
      console.log('Mock: Loading test model...');
      this.initialized = true;
    }

    loadModelFromJSON(json: string): void {
      console.log('Mock: Loading model from JSON...');
      try {
        JSON.parse(json); // Validate JSON
        this.initialized = true;
        console.log('Mock: Model loaded successfully');
      } catch (error) {
        console.error('Mock: Invalid JSON format');
        throw new Error('Invalid model JSON format');
      }
    }

    isModelLoaded(): boolean {
      return this.initialized;
    }

    infer(inputData: Float32Array, inputShape: number[]): Float32Array {
      console.log('Mock: Running inference...', { inputShape });
      // Mock ReLU: max(0, x)
      const output = new Float32Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        output[i] = Math.max(0, inputData[i]);
      }
      return output;
    }

    getModelInfo(): string {
      return JSON.stringify({
        nodes: 3,
        weights: 0,
        initialized: this.initialized,
      });
    }

    free(): void {
      // No-op for mock
    }
  }

  class MockBenchmark implements BenchmarkInstance {
    private results: number[] = [];

    benchmarkMatMul(size: number, iterations: number): number {
      console.log(`Mock: Benchmarking MatMul ${size}x${size}, ${iterations} iterations`);
      const time = 45 + Math.random() * 10; // Mock time: 45-55ms
      this.results.push(time);
      return time;
    }

    benchmarkReLU(size: number, iterations: number): number {
      console.log(`Mock: Benchmarking ReLU ${size} elements, ${iterations} iterations`);
      const time = 2 + Math.random() * 1; // Mock time: 2-3ms
      this.results.push(time);
      return time;
    }

    benchmarkConv2D(batch: number, channels: number, size: number, iterations: number): number {
      console.log(`Mock: Benchmarking Conv2D [${batch}, ${channels}, ${size}, ${size}], ${iterations} iterations`);
      const time = 85 + Math.random() * 15; // Mock time: 85-100ms
      this.results.push(time);
      return time;
    }

    getResults(): Float64Array {
      return new Float64Array(this.results);
    }

    clearResults(): void {
      this.results = [];
    }

    free(): void {
      // No-op for mock
    }
  }

  class MockTensor implements TensorInstance {
    constructor(private data: Float32Array, private shape: number[]) {}

    getShape(): number[] {
      return this.shape;
    }

    getData(): Float32Array {
      return this.data;
    }

    getSize(): number {
      return this.data.length;
    }

    toString(): string {
      return `Tensor(shape=[${this.shape.join(', ')}], size=${this.data.length})`;
    }

    free(): void {
      // No-op for mock
    }
  }

  return {
    TinyInfer: MockTinyInfer as any,
    Benchmark: MockBenchmark as any,
    Tensor: MockTensor as any,
    version: () => '0.1.0',
    check_simd_support: () => {
      // Check actual SIMD support
      try {
        return typeof WebAssembly !== 'undefined' &&
               WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]));
      } catch {
        return false;
      }
    },
    get_system_info: () => JSON.stringify({
      version: '0.1.0',
      simd_support: true,
      target: 'wasm32-unknown-unknown',
    }),
    createTensor: (data: Float32Array, dims: number[]) => new MockTensor(data, dims) as any,
  };
}

// ============================================================================
// ONNX Loading with Caching Support
// ============================================================================

import {
  loadONNXFromFile as onnxLoadFromFile,
  loadONNXFromURL as onnxLoadFromURL,
  type ModelDef,
} from './onnxLoader'

import {
  getCachedModel,
  cacheModel,
  clearModelCache as clearCache,
  getModelCacheStats as getCacheStats,
  generateCacheKeyFromFile,
  generateCacheKeyFromURL,
} from './modelCache'

/**
 * Load ONNX model from File with caching
 * Automatically caches parsed models for faster subsequent loads
 */
export async function loadONNXFromFile(
  engine: TinyInferInstance,
  file: File,
  options: { useCache?: boolean } = {}
): Promise<void> {
  const { useCache = true } = options

  try {
    console.log(`Loading ONNX model from file: ${file.name}`)

    let modelDef: ModelDef

    // Check cache if enabled
    if (useCache) {
      const cacheKey = await generateCacheKeyFromFile(file)
      const cached = await getCachedModel(cacheKey)

      if (cached) {
        console.log('Using cached model')
        modelDef = cached
      } else {
        console.log('Parsing ONNX model...')
        modelDef = await onnxLoadFromFile(file)

        // Cache the parsed model
        await cacheModel(cacheKey, modelDef, {
          name: file.name,
          size: file.size,
          format: 'onnx',
        })
      }
    } else {
      modelDef = await onnxLoadFromFile(file)
    }

    // Load into engine
    const json = JSON.stringify(modelDef)
    await loadModelFromJSON(engine, json)

    console.log('ONNX model loaded successfully')
  } catch (error) {
    console.error('Failed to load ONNX model from file:', error)
    throw new Error(`Failed to load ONNX model: ${error}`)
  }
}

/**
 * Load ONNX model from URL with caching
 * Automatically caches parsed models for faster subsequent loads
 */
export async function loadONNXFromURL(
  engine: TinyInferInstance,
  url: string,
  options: { useCache?: boolean } = {}
): Promise<void> {
  const { useCache = true } = options

  try {
    console.log(`Loading ONNX model from URL: ${url}`)

    let modelDef: ModelDef

    // Check cache if enabled
    if (useCache) {
      const cacheKey = generateCacheKeyFromURL(url)
      const cached = await getCachedModel(cacheKey)

      if (cached) {
        console.log('Using cached model')
        modelDef = cached
      } else {
        console.log('Fetching and parsing ONNX model...')
        modelDef = await onnxLoadFromURL(url)

        // Estimate size (rough estimate based on JSON size)
        const jsonSize = JSON.stringify(modelDef).length

        // Cache the parsed model
        await cacheModel(cacheKey, modelDef, {
          name: url.split('/').pop() || 'model',
          size: jsonSize,
          format: 'onnx',
        })
      }
    } else {
      modelDef = await onnxLoadFromURL(url)
    }

    // Load into engine
    const json = JSON.stringify(modelDef)
    await loadModelFromJSON(engine, json)

    console.log('ONNX model loaded successfully')
  } catch (error) {
    console.error('Failed to load ONNX model from URL:', error)
    throw new Error(`Failed to load ONNX model from ${url}: ${error}`)
  }
}

/**
 * Clear all cached models
 */
export async function clearModelCache(): Promise<void> {
  await clearCache()
  console.log('Model cache cleared')
}

/**
 * Get model cache statistics
 */
export async function getModelCacheStats() {
  return await getCacheStats()
}

export default {
  initWasm,
  getWasm,
  isWasmLoaded,
  getVersion,
  checkSimdSupport,
  getSystemInfo,
  createInferenceEngine,
  createBenchmark,
  createTensor,
  loadModelFromJSON,
  loadModelFromURL,
  loadModelFromFile,
  loadONNXFromFile,
  loadONNXFromURL,
  clearModelCache,
  getModelCacheStats,
};
