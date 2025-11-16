import { useState, useEffect } from 'react';
import { initWasm, isWasmLoaded, getSystemInfo, type SystemInfo } from '../lib/tinyinfer';

interface WasmState {
  loaded: boolean;
  loading: boolean;
  error: string | null;
  systemInfo: SystemInfo | null;
}

/**
 * React hook for managing WASM module lifecycle
 */
export function useWasm() {
  const [state, setState] = useState<WasmState>({
    loaded: false,
    loading: false,
    error: null,
    systemInfo: null,
  });

  useEffect(() => {
    if (isWasmLoaded()) {
      setState({
        loaded: true,
        loading: false,
        error: null,
        systemInfo: getSystemInfo(),
      });
      return;
    }

    setState(prev => ({ ...prev, loading: true }));

    initWasm()
      .then(() => {
        setState({
          loaded: true,
          loading: false,
          error: null,
          systemInfo: getSystemInfo(),
        });
      })
      .catch((error) => {
        console.error('Failed to load WASM:', error);
        setState({
          loaded: false,
          loading: false,
          error: error.message || 'Failed to load WASM module',
          systemInfo: null,
        });
      });
  }, []);

  return state;
}

/**
 * Hook for running inference
 */
export function useInference() {
  const wasm = useWasm();

  const runInference = async (
    inputData: number[] | Float32Array,
    inputShape: number[]
  ): Promise<Float32Array> => {
    if (!wasm.loaded) {
      throw new Error('WASM not loaded');
    }

    const { createInferenceEngine } = await import('../lib/tinyinfer');
    const engine = createInferenceEngine();

    try {
      // Load test model
      engine.loadTestModel();

      // Convert to Float32Array if needed
      const input = inputData instanceof Float32Array
        ? inputData
        : new Float32Array(inputData);

      // Run inference
      const output = engine.infer(input, inputShape);

      return output;
    } finally {
      engine.free();
    }
  };

  return {
    ...wasm,
    runInference,
  };
}

/**
 * Hook for running benchmarks
 */
export function useBenchmark() {
  const wasm = useWasm();
  const [results, setResults] = useState<Record<string, number>>({});
  const [running, setRunning] = useState(false);

  const runMatMulBenchmark = async (size: number = 1024, iterations: number = 10) => {
    if (!wasm.loaded) {
      throw new Error('WASM not loaded');
    }

    setRunning(true);
    try {
      const { createBenchmark } = await import('../lib/tinyinfer');
      const benchmark = createBenchmark();

      const time = benchmark.benchmarkMatMul(size, iterations);
      setResults(prev => ({ ...prev, matmul: time }));

      benchmark.free();
      return time;
    } finally {
      setRunning(false);
    }
  };

  const runReLUBenchmark = async (size: number = 1000000, iterations: number = 100) => {
    if (!wasm.loaded) {
      throw new Error('WASM not loaded');
    }

    setRunning(true);
    try {
      const { createBenchmark } = await import('../lib/tinyinfer');
      const benchmark = createBenchmark();

      const time = benchmark.benchmarkReLU(size, iterations);
      setResults(prev => ({ ...prev, relu: time }));

      benchmark.free();
      return time;
    } finally {
      setRunning(false);
    }
  };

  const runConv2DBenchmark = async (
    batch: number = 1,
    channels: number = 64,
    size: number = 224,
    iterations: number = 10
  ) => {
    if (!wasm.loaded) {
      throw new Error('WASM not loaded');
    }

    setRunning(true);
    try {
      const { createBenchmark } = await import('../lib/tinyinfer');
      const benchmark = createBenchmark();

      const time = benchmark.benchmarkConv2D(batch, channels, size, iterations);
      setResults(prev => ({ ...prev, conv2d: time }));

      benchmark.free();
      return time;
    } finally {
      setRunning(false);
    }
  };

  const runAllBenchmarks = async () => {
    await runMatMulBenchmark();
    await runReLUBenchmark();
    await runConv2DBenchmark();
  };

  const clearResults = () => {
    setResults({});
  };

  return {
    ...wasm,
    results,
    running,
    runMatMulBenchmark,
    runReLUBenchmark,
    runConv2DBenchmark,
    runAllBenchmarks,
    clearResults,
  };
}
