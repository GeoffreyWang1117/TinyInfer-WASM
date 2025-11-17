/**
 * Performance Optimization Examples for TinyInfer
 *
 * Demonstrates Phase 8 optimizations:
 * - Web Workers for background inference
 * - Positional Encoding
 * - Attention Masks
 * - INT8 Quantization
 */

import {
  init Wasm,
  createInferenceEngine,
  loadModelFromJSON,
} from '../web/src/lib/tinyinfer'

import { createWorkerInstance } from '../web/src/lib/workerClient'

import {
  quantizeModelWeights,
  dequantizeModelWeights,
  calculateSizeReduction,
} from '../web/src/lib/quantization'

/**
 * Example 1: Web Worker for Background Inference
 *
 * Demonstrates running inference in a worker thread to avoid blocking the UI.
 */
export async function example1_webWorkerInference() {
  console.log('=== Example 1: Web Worker Background Inference ===\n')

  // Create worker instance
  const worker = await createWorkerInstance()

  // Load model
  const model = {
    version: '1.0',
    name: 'simple_model',
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
  }

  await worker.loadModel(JSON.stringify(model))
  console.log('✅ Model loaded in worker')

  // Run inference in background
  const input = new Float32Array([-1, 2, -3, 4])
  console.log('Input:', Array.from(input))

  console.log('Running inference in worker (non-blocking)...')
  const startTime = performance.now()

  const output = await worker.infer(input, [4])

  const elapsedTime = performance.now() - startTime

  console.log('Output:', Array.from(output))
  console.log(`✅ Inference complete in ${elapsedTime.toFixed(2)}ms`)

  // Run multiple inferences in parallel
  console.log('\nRunning 5 parallel inferences...')
  const parallelStart = performance.now()

  const promises = []
  for (let i = 0; i < 5; i++) {
    const input = new Float32Array([i, -i, i * 2, -i * 2])
    promises.push(worker.infer(input, [4]))
  }

  const results = await Promise.all(promises)
  const parallelTime = performance.now() - parallelStart

  console.log(`✅ All inferences complete in ${parallelTime.toFixed(2)}ms`)
  console.log(`   Average: ${(parallelTime / 5).toFixed(2)}ms per inference`)

  // Clean up
  worker.terminate()
}

/**
 * Example 2: Positional Encoding in Transformer
 *
 * Shows how to use sinusoidal positional encoding.
 */
export async function example2_positionalEncoding() {
  console.log('\n=== Example 2: Positional Encoding ===\n')

  await initWasm()
  const engine = createInferenceEngine()

  const d_model = 128
  const max_len = 512
  const seq_len = 16
  const batch = 1

  // Model with positional encoding
  const model = {
    version: '1.0',
    name: 'positional_encoding_model',
    graph: {
      nodes: [
        // Input
        { id: 'input', op_type: 'Input', inputs: [], outputs: ['x'], attributes: {} },

        // Embedding (simulated - in reality would use Embedding layer)
        { id: 'embedding', op_type: 'MatMul', inputs: ['x', 'emb_weight'], outputs: ['embedded'], attributes: {} },

        // Positional Encoding
        {
          id: 'pos_enc',
          op_type: 'PositionalEncoding',
          inputs: ['embedded'],
          outputs: ['pos_encoded'],
          attributes: {
            d_model: { int: d_model },
            max_len: { int: max_len },
          },
        },

        // Output
        { id: 'output', op_type: 'Output', inputs: ['pos_encoded'], outputs: [], attributes: {} },
      ],
      edges: [
        { from: 'input', to: 'embedding' },
        { from: 'embedding', to: 'pos_enc' },
        { from: 'pos_enc', to: 'output' },
      ],
      inputs: ['x'],
      outputs: ['pos_encoded'],
    },
    weights: {
      emb_weight: {
        shape: [seq_len, d_model],
        dtype: 'float32',
        data: Array(seq_len * d_model).fill(0).map(() => (Math.random() - 0.5) * 0.1),
      },
    },
  }

  await loadModelFromJSON(engine, JSON.stringify(model))
  console.log('✅ Model with positional encoding loaded')

  // Run inference
  const input = new Float32Array(seq_len).fill(1.0)
  const output = engine.infer(input, [batch, seq_len, 1])

  console.log(`Output shape: [${batch}, ${seq_len}, ${d_model}]`)
  console.log(`✅ Positional encoding applied`)

  engine.free()
}

/**
 * Example 3: Attention with Mask
 *
 * Demonstrates using attention masks for padding and causal attention.
 */
export async function example3_attentionMask() {
  console.log('\n=== Example 3: Attention with Mask ===\n')

  await initWasm()
  const engine = createInferenceEngine()

  const d_model = 64
  const seq_len = 8
  const batch = 1

  // Create causal mask (lower triangular)
  const causalMask = new Float32Array(seq_len * seq_len)
  for (let i = 0; i < seq_len; i++) {
    for (let j = 0; j < seq_len; j++) {
      causalMask[i * seq_len + j] = j <= i ? 1.0 : 0.0
    }
  }

  console.log('Causal mask (8x8):')
  for (let i = 0; i < seq_len; i++) {
    const row = Array.from(causalMask.slice(i * seq_len, (i + 1) * seq_len))
    console.log(`  [${row.join(', ')}]`)
  }

  // Create padding mask (mask out last 2 positions)
  const paddingMask = new Float32Array(seq_len * seq_len).fill(1.0)
  for (let i = 0; i < seq_len; i++) {
    for (let j = seq_len - 2; j < seq_len; j++) {
      paddingMask[i * seq_len + j] = 0.0 // Mask these positions
    }
  }

  console.log('\n✅ Masks created')
  console.log('   - Causal mask: Lower triangular')
  console.log('   - Padding mask: Last 2 positions masked')

  console.log('\nNote: Full masked attention example requires complete model setup')
}

/**
 * Example 4: INT8 Quantization
 *
 * Shows how to quantize model weights to reduce size.
 */
export async function example4_int8Quantization() {
  console.log('\n=== Example 4: INT8 Quantization ===\n')

  // Create a model with weights
  const model = {
    version: '1.0',
    name: 'quantization_example',
    graph: {
      nodes: [
        { id: 'input', op_type: 'Input', inputs: [], outputs: ['x'], attributes: {} },
        { id: 'linear', op_type: 'MatMul', inputs: ['x', 'weight'], outputs: ['y'], attributes: {} },
        { id: 'output', op_type: 'Output', inputs: ['y'], outputs: [], attributes: {} },
      ],
      edges: [
        { from: 'input', to: 'linear' },
        { from: 'linear', to: 'output' },
      ],
      inputs: ['x'],
      outputs: ['y'],
    },
    weights: {
      weight: {
        shape: [128, 256],
        dtype: 'float32',
        data: Array(128 * 256).fill(0).map(() => (Math.random() - 0.5) * 2),
      },
    },
  }

  console.log('Original model:')
  console.log(`  Weight shape: [128, 256]`)
  console.log(`  Weight count: ${128 * 256}`)
  console.log(`  Dtype: float32`)

  // Quantize weights
  console.log('\nQuantizing weights to INT8...')
  const { quantizedModel, quantizationInfo } = quantizeModelWeights(model, true)

  console.log('\nQuantized model:')
  console.log(`  Dtype: int8`)
  console.log(`  Quantization params:`)
  for (const [name, params] of Object.entries(quantizationInfo)) {
    console.log(`    ${name}:`)
    console.log(`      scale: ${params.scale.toFixed(6)}`)
    console.log(`      zero_point: ${params.zeroPoint}`)
    console.log(`      range: [${params.min.toFixed(2)}, ${params.max.toFixed(2)}]`)
  }

  // Calculate size reduction
  const stats = calculateSizeReduction(model, quantizedModel)

  console.log('\n📊 Size Reduction:')
  console.log(`   Original size: ${(stats.originalSize / 1024).toFixed(2)} KB`)
  console.log(`   Quantized size: ${(stats.quantizedSize / 1024).toFixed(2)} KB`)
  console.log(`   Reduction: ${(stats.reduction / 1024).toFixed(2)} KB`)
  console.log(`   Compression ratio: ${stats.compressionRatio.toFixed(2)}x`)

  // Dequantize for loading (in real usage, this happens automatically)
  console.log('\nDequantizing for inference...')
  const dequantizedModel = dequantizeModelWeights(quantizedModel)

  console.log('✅ Model quantized and ready for inference')
  console.log(`   Memory savings: ~${((1 - 1 / stats.compressionRatio) * 100).toFixed(1)}%`)
}

/**
 * Example 5: Complete Optimized Transformer
 *
 * Combines all optimizations: worker, positional encoding, masks.
 */
export async function example5_optimizedTransformer() {
  console.log('\n=== Example 5: Optimized Transformer ===\n')

  const d_model = 256
  const num_heads = 8
  const seq_len = 32
  const batch = 1

  // Create model with all optimizations
  const model = {
    version: '1.0',
    name: 'optimized_transformer',
    graph: {
      nodes: [
        // Input
        { id: 'input', op_type: 'Input', inputs: [], outputs: ['x'], attributes: {} },

        // Embedding + Positional Encoding
        {
          id: 'pos_enc',
          op_type: 'PositionalEncoding',
          inputs: ['x'],
          outputs: ['pos_x'],
          attributes: { d_model: { int: d_model }, max_len: { int: 512 } },
        },

        // Self-Attention with Mask
        {
          id: 'self_attn',
          op_type: 'SelfAttention',
          inputs: ['pos_x', 'attn_mask'],
          outputs: ['attn_out'],
          attributes: { d_model: { int: d_model }, num_heads: { int: num_heads } },
        },

        // LayerNorm
        {
          id: 'ln',
          op_type: 'LayerNorm',
          inputs: ['attn_out', 'ln_gamma', 'ln_beta'],
          outputs: ['output'],
          attributes: {},
        },

        { id: 'out', op_type: 'Output', inputs: ['output'], outputs: [], attributes: {} },
      ],
      edges: [
        { from: 'input', to: 'pos_enc' },
        { from: 'pos_enc', to: 'self_attn' },
        { from: 'self_attn', to: 'ln' },
        { from: 'ln', to: 'out' },
      ],
      inputs: ['x'],
      outputs: ['output'],
    },
    weights: {
      attn_mask: {
        shape: [seq_len, seq_len],
        dtype: 'float32',
        data: Array(seq_len * seq_len).fill(0).map((_, i) => {
          const row = Math.floor(i / seq_len)
          const col = i % seq_len
          return col <= row ? 1.0 : 0.0 // Causal mask
        }),
      },
      ln_gamma: {
        shape: [d_model],
        dtype: 'float32',
        data: Array(d_model).fill(1.0),
      },
      ln_beta: {
        shape: [d_model],
        dtype: 'float32',
        data: Array(d_model).fill(0.0),
      },
    },
  }

  console.log('Creating optimized Transformer model:')
  console.log(`  - d_model: ${d_model}`)
  console.log(`  - num_heads: ${num_heads}`)
  console.log(`  - seq_len: ${seq_len}`)
  console.log(`  - Features: Positional Encoding + Attention Mask + LayerNorm`)

  // Quantize the model
  const { quantizedModel } = quantizeModelWeights(model, true)
  const stats = calculateSizeReduction(model, quantizedModel)

  console.log(`\n📊 Quantization: ${stats.compressionRatio.toFixed(2)}x compression`)

  // Use web worker for inference
  console.log('\nLoading into Web Worker...')
  const worker = await createWorkerInstance()
  await worker.loadModel(JSON.stringify(dequantizeModelWeights(quantizedModel)))

  console.log('✅ Model loaded in worker')

  // Run inference
  const input = new Float32Array(seq_len * d_model).fill(0).map(() => Math.random())
  console.log('\nRunning optimized inference...')

  const startTime = performance.now()
  const output = await worker.infer(input, [batch, seq_len, d_model])
  const elapsedTime = performance.now() - startTime

  console.log(`✅ Inference complete in ${elapsedTime.toFixed(2)}ms`)
  console.log(`   Output shape: [${batch}, ${seq_len}, ${d_model}]`)

  // Cleanup
  worker.terminate()

  console.log('\n✨ Optimizations applied:')
  console.log('   ✓ Web Worker (non-blocking)')
  console.log('   ✓ Positional Encoding')
  console.log('   ✓ Attention Mask (causal)')
  console.log('   ✓ INT8 Quantization')
  console.log(`   ✓ ${stats.compressionRatio.toFixed(2)}x smaller model`)
}

/**
 * Run all examples
 */
export async function runAllExamples() {
  console.log('🚀 TinyInfer Performance Optimization Examples (Phase 8)\n')
  console.log('='.repeat(60))

  try {
    await example1_webWorkerInference()
    await example2_positionalEncoding()
    await example3_attentionMask()
    await example4_int8Quantization()
    await example5_optimizedTransformer()

    console.log('\n' + '='.repeat(60))
    console.log('\n✅ All examples completed successfully!\n')
  } catch (error) {
    console.error('\n❌ Error running examples:', error)
  }
}

// Export for browser usage
if (typeof window !== 'undefined') {
  console.log('Performance Optimization examples loaded. Run:')
  console.log('  - example1_webWorkerInference()')
  console.log('  - example2_positionalEncoding()')
  console.log('  - example3_attentionMask()')
  console.log('  - example4_int8Quantization()')
  console.log('  - example5_optimizedTransformer()')
  console.log('  - runAllExamples()')
}
