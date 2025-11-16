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
};
