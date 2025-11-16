/* tslint:disable */
/* eslint-disable */
/**
 * Get version information
 */
export function version(): string;
/**
 * Get system information
 */
export function get_system_info(): string;
/**
 * Initialize the WASM module
 */
export function init(): void;
/**
 * Check SIMD support
 */
export function check_simd_support(): boolean;
/**
 * Utility function to create a tensor from JavaScript
 */
export function createTensor(data: Float32Array, dims: Uint32Array): Tensor;
/**
 * Get current timestamp in milliseconds
 */
export function now(): number;
/**
 * Benchmark a specific operation
 */
export class Benchmark {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Get all benchmark results
   */
  getResults(): Float64Array;
  /**
   * Clear benchmark results
   */
  clearResults(): void;
  /**
   * Benchmark ReLU activation
   */
  benchmarkReLU(size: number, iterations: number): number;
  /**
   * Benchmark Conv2D
   */
  benchmarkConv2D(batch: number, channels: number, size: number, iterations: number): number;
  /**
   * Benchmark matrix multiplication
   */
  benchmarkMatMul(size: number, iterations: number): number;
  /**
   * Create a new benchmark
   */
  constructor();
}
/**
 * A neural network model
 */
export class Model {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Check if initialized
   */
  isInitialized(): boolean;
  /**
   * Create a new model (for WASM)
   */
  constructor();
  /**
   * Get model info as JSON string
   */
  getInfo(): string;
}
/**
 * Tensor data structure
 * Currently only supports f32 for simplicity
 */
export class Tensor {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new tensor from JavaScript
   */
  constructor(data: Float32Array, dims: Uint32Array);
  /**
   * Convert to string representation
   */
  toString(): string;
  /**
   * Get data as array
   */
  getData(): Float32Array;
  /**
   * Get size
   */
  getSize(): number;
  /**
   * Get shape as array
   */
  getShape(): Uint32Array;
}
/**
 * TinyInfer API - Main entry point for JavaScript
 */
export class TinyInfer {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Get model information
   */
  getModelInfo(): string;
  /**
   * Load a simple test model (for demonstration)
   * In the future, this will load ONNX models
   */
  loadTestModel(): void;
  /**
   * Create a new TinyInfer instance
   */
  constructor();
  /**
   * Run inference with the loaded model
   * Takes input data as Float32Array and returns output as Float32Array
   */
  infer(input_data: Float32Array, input_shape: Uint32Array): Float32Array;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_benchmark_free: (a: number, b: number) => void;
  readonly __wbg_model_free: (a: number, b: number) => void;
  readonly __wbg_tensor_free: (a: number, b: number) => void;
  readonly __wbg_tinyinfer_free: (a: number, b: number) => void;
  readonly benchmark_benchmarkConv2D: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly benchmark_benchmarkMatMul: (a: number, b: number, c: number) => number;
  readonly benchmark_benchmarkReLU: (a: number, b: number, c: number) => number;
  readonly benchmark_clearResults: (a: number) => void;
  readonly benchmark_getResults: (a: number, b: number) => void;
  readonly benchmark_new: () => number;
  readonly check_simd_support: () => number;
  readonly createTensor: (a: number, b: number, c: number, d: number) => number;
  readonly get_system_info: (a: number) => void;
  readonly init: () => void;
  readonly model_getInfo: (a: number, b: number) => void;
  readonly model_isInitialized: (a: number) => number;
  readonly model_new_js: () => number;
  readonly tensor_getData: (a: number, b: number) => void;
  readonly tensor_getShape: (a: number, b: number) => void;
  readonly tensor_getSize: (a: number) => number;
  readonly tensor_toString: (a: number, b: number) => void;
  readonly tinyinfer_getModelInfo: (a: number, b: number) => void;
  readonly tinyinfer_infer: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly tinyinfer_loadTestModel: (a: number) => void;
  readonly tinyinfer_new: () => number;
  readonly version: (a: number) => void;
  readonly tensor_new_from_js: (a: number, b: number, c: number, d: number) => number;
  readonly now: () => number;
  readonly __wbindgen_export: (a: number, b: number, c: number) => void;
  readonly __wbindgen_export2: (a: number, b: number) => number;
  readonly __wbindgen_export3: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
