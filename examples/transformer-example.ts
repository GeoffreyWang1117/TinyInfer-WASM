/**
 * Transformer Model Example for TinyInfer
 *
 * Demonstrates how to build and run a simple Transformer model using TinyInfer.
 * This example shows the core components of a Transformer:
 * - Multi-Head Self-Attention
 * - Layer Normalization
 * - Feed-Forward Network
 * - GELU Activation
 */

import {
  initWasm,
  createInferenceEngine,
  loadModelFromJSON,
} from '../web/src/lib/tinyinfer'

/**
 * Example 1: Build a simple self-attention layer
 *
 * This demonstrates the basic building block of Transformers.
 */
export async function example1_selfAttention() {
  console.log('=== Example 1: Self-Attention Layer ===\n')

  await initWasm()
  const engine = createInferenceEngine()

  // Model configuration
  const d_model = 128 // Hidden dimension
  const seq_len = 16 // Sequence length
  const batch_size = 1

  // Build a simple self-attention model
  const model = {
    version: '1.0',
    name: 'self_attention_layer',
    graph: {
      nodes: [
        {
          id: 'input',
          op_type: 'Input',
          inputs: [],
          outputs: ['x'],
          attributes: {},
        },
        {
          id: 'self_attention',
          op_type: 'SelfAttention',
          inputs: ['x'],
          outputs: ['attn_out'],
          attributes: {
            d_model: { int: d_model },
            num_heads: { int: 4 },
          },
        },
        {
          id: 'output',
          op_type: 'Output',
          inputs: ['attn_out'],
          outputs: [],
          attributes: {},
        },
      ],
      edges: [
        { from: 'input', to: 'self_attention' },
        { from: 'self_attention', to: 'output' },
      ],
      inputs: ['x'],
      outputs: ['attn_out'],
    },
    weights: {},
  }

  // Load model
  await loadModelFromJSON(engine, JSON.stringify(model))

  console.log('✅ Self-attention model loaded')
  console.log(`   d_model: ${d_model}, num_heads: 4, seq_len: ${seq_len}`)

  // Create sample input [batch_size, seq_len, d_model]
  const input_size = batch_size * seq_len * d_model
  const input = new Float32Array(input_size)

  // Initialize with random values
  for (let i = 0; i < input_size; i++) {
    input[i] = (Math.random() - 0.5) * 2 // Range: [-1, 1]
  }

  console.log('\\nRunning inference...')
  const output = engine.infer(input, [batch_size, seq_len, d_model])

  console.log(`✅ Output shape: [${batch_size}, ${seq_len}, ${d_model}]`)
  console.log(`   Output size: ${output.length}`)
  console.log(`   Sample values: [${output.slice(0, 5).join(', ')}...]`)

  engine.free()
}

/**
 * Example 2: Transformer Encoder Layer
 *
 * A complete Transformer encoder layer with:
 * - Multi-Head Attention
 * - Layer Normalization (x2)
 * - Feed-Forward Network
 * - Residual connections
 */
export async function example2_transformerEncoderLayer() {
  console.log('\\n=== Example 2: Transformer Encoder Layer ===\\n')

  await initWasm()
  const engine = createInferenceEngine()

  const d_model = 256
  const d_ff = 1024 // Feed-forward dimension
  const num_heads = 8
  const seq_len = 32
  const batch_size = 1

  const model = {
    version: '1.0',
    name: 'transformer_encoder_layer',
    graph: {
      nodes: [
        // Input
        { id: 'input', op_type: 'Input', inputs: [], outputs: ['x'], attributes: {} },

        // Self-Attention Block
        {
          id: 'self_attn',
          op_type: 'SelfAttention',
          inputs: ['x'],
          outputs: ['attn_out'],
          attributes: { d_model: { int: d_model }, num_heads: { int: num_heads } },
        },

        // Add & Norm 1 (residual connection + layer norm)
        {
          id: 'add1',
          op_type: 'Add',
          inputs: ['x', 'attn_out'],
          outputs: ['add1_out'],
          attributes: {},
        },
        {
          id: 'ln1',
          op_type: 'LayerNorm',
          inputs: ['add1_out', 'ln1_gamma', 'ln1_beta'],
          outputs: ['ln1_out'],
          attributes: { normalized_shape: { ints: [d_model] } },
        },

        // Feed-Forward Network
        {
          id: 'ff1',
          op_type: 'MatMul',
          inputs: ['ln1_out', 'ff1_weight'],
          outputs: ['ff1_out'],
          attributes: {},
        },
        {
          id: 'gelu',
          op_type: 'GELU',
          inputs: ['ff1_out'],
          outputs: ['gelu_out'],
          attributes: {},
        },
        {
          id: 'ff2',
          op_type: 'MatMul',
          inputs: ['gelu_out', 'ff2_weight'],
          outputs: ['ff2_out'],
          attributes: {},
        },

        // Add & Norm 2
        {
          id: 'add2',
          op_type: 'Add',
          inputs: ['ln1_out', 'ff2_out'],
          outputs: ['add2_out'],
          attributes: {},
        },
        {
          id: 'ln2',
          op_type: 'LayerNorm',
          inputs: ['add2_out', 'ln2_gamma', 'ln2_beta'],
          outputs: ['encoder_out'],
          attributes: { normalized_shape: { ints: [d_model] } },
        },

        // Output
        {
          id: 'output',
          op_type: 'Output',
          inputs: ['encoder_out'],
          outputs: [],
          attributes: {},
        },
      ],
      edges: [
        { from: 'input', to: 'self_attn' },
        { from: 'self_attn', to: 'add1' },
        { from: 'input', to: 'add1' },
        { from: 'add1', to: 'ln1' },
        { from: 'ln1', to: 'ff1' },
        { from: 'ff1', to: 'gelu' },
        { from: 'gelu', to: 'ff2' },
        { from: 'ln1', to: 'add2' },
        { from: 'ff2', to: 'add2' },
        { from: 'add2', to: 'ln2' },
        { from: 'ln2', to: 'output' },
      ],
      inputs: ['x'],
      outputs: ['encoder_out'],
    },
    weights: {
      // Layer Norm 1 parameters
      ln1_gamma: {
        shape: [d_model],
        dtype: 'float32',
        data: Array(d_model).fill(1.0),
      },
      ln1_beta: {
        shape: [d_model],
        dtype: 'float32',
        data: Array(d_model).fill(0.0),
      },

      // Feed-Forward weights
      ff1_weight: {
        shape: [d_model, d_ff],
        dtype: 'float32',
        data: Array(d_model * d_ff)
          .fill(0)
          .map(() => (Math.random() - 0.5) * 0.1),
      },
      ff2_weight: {
        shape: [d_ff, d_model],
        dtype: 'float32',
        data: Array(d_ff * d_model)
          .fill(0)
          .map(() => (Math.random() - 0.5) * 0.1),
      },

      // Layer Norm 2 parameters
      ln2_gamma: {
        shape: [d_model],
        dtype: 'float32',
        data: Array(d_model).fill(1.0),
      },
      ln2_beta: {
        shape: [d_model],
        dtype: 'float32',
        data: Array(d_model).fill(0.0),
      },
    },
  }

  await loadModelFromJSON(engine, JSON.stringify(model))

  console.log('✅ Transformer Encoder Layer loaded')
  console.log(`   d_model: ${d_model}, d_ff: ${d_ff}`)
  console.log(`   num_heads: ${num_heads}, seq_len: ${seq_len}`)

  // Create input
  const input_size = batch_size * seq_len * d_model
  const input = new Float32Array(input_size)
  for (let i = 0; i < input_size; i++) {
    input[i] = (Math.random() - 0.5) * 2
  }

  console.log('\\nRunning inference...')
  const start = performance.now()
  const output = engine.infer(input, [batch_size, seq_len, d_model])
  const elapsed = performance.now() - start

  console.log(`✅ Inference complete in ${elapsed.toFixed(2)}ms`)
  console.log(`   Output shape: [${batch_size}, ${seq_len}, ${d_model}]`)

  engine.free()
}

/**
 * Example 3: Text Classification with Transformer
 *
 * A simple Transformer model for text classification:
 * - Embedding layer
 * - Transformer encoder
 * - Classification head
 */
export async function example3_textClassification() {
  console.log('\\n=== Example 3: Text Classification with Transformer ===\\n')

  await initWasm()
  const engine = createInferenceEngine()

  const vocab_size = 1000
  const d_model = 128
  const num_classes = 2 // Binary classification
  const seq_len = 16
  const batch_size = 1

  console.log('Building text classification model...')
  console.log(`  Vocab size: ${vocab_size}`)
  console.log(`  Embedding dim: ${d_model}`)
  console.log(`  Sequence length: ${seq_len}`)
  console.log(`  Classes: ${num_classes}`)

  // Initialize embedding matrix with random values
  const embedding_matrix = Array(vocab_size * d_model)
    .fill(0)
    .map(() => (Math.random() - 0.5) * 0.1)

  // Note: This is a simplified example. A full implementation would include:
  // - Positional encoding
  // - Multiple encoder layers
  // - Proper weight initialization
  // - Dropout (during training)

  const model = {
    version: '1.0',
    name: 'text_classifier',
    graph: {
      nodes: [
        // Input (token indices)
        { id: 'input', op_type: 'Input', inputs: [], outputs: ['tokens'], attributes: {} },

        // Embedding layer
        {
          id: 'embedding',
          op_type: 'Embedding',
          inputs: ['tokens', 'emb_weight'],
          outputs: ['embedded'],
          attributes: {
            vocab_size: { int: vocab_size },
            embedding_dim: { int: d_model },
          },
        },

        // Self-Attention
        {
          id: 'self_attn',
          op_type: 'SelfAttention',
          inputs: ['embedded'],
          outputs: ['attn_out'],
          attributes: { d_model: { int: d_model }, num_heads: { int: 4 } },
        },

        // Layer Norm
        {
          id: 'ln',
          op_type: 'LayerNorm',
          inputs: ['attn_out', 'ln_gamma', 'ln_beta'],
          outputs: ['normalized'],
          attributes: {},
        },

        // Global average pooling (simplified - take mean over sequence)
        // In a real implementation, this would be a proper pooling operator
        {
          id: 'pool',
          op_type: 'GlobalAvgPool2D',
          inputs: ['normalized'],
          outputs: ['pooled'],
          attributes: {},
        },

        // Classification head
        {
          id: 'classifier',
          op_type: 'MatMul',
          inputs: ['pooled', 'clf_weight'],
          outputs: ['logits'],
          attributes: {},
        },

        // Softmax
        {
          id: 'softmax',
          op_type: 'Softmax',
          inputs: ['logits'],
          outputs: ['probs'],
          attributes: {},
        },

        { id: 'output', op_type: 'Output', inputs: ['probs'], outputs: [], attributes: {} },
      ],
      edges: [
        { from: 'input', to: 'embedding' },
        { from: 'embedding', to: 'self_attn' },
        { from: 'self_attn', to: 'ln' },
        { from: 'ln', to: 'pool' },
        { from: 'pool', to: 'classifier' },
        { from: 'classifier', to: 'softmax' },
        { from: 'softmax', to: 'output' },
      ],
      inputs: ['tokens'],
      outputs: ['probs'],
    },
    weights: {
      emb_weight: {
        shape: [vocab_size, d_model],
        dtype: 'float32',
        data: embedding_matrix,
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
      clf_weight: {
        shape: [d_model, num_classes],
        dtype: 'float32',
        data: Array(d_model * num_classes)
          .fill(0)
          .map(() => (Math.random() - 0.5) * 0.1),
      },
    },
  }

  await loadModelFromJSON(engine, JSON.stringify(model))

  console.log('\\n✅ Model loaded successfully')

  // Create sample input (token indices)
  const input = new Float32Array(seq_len)
  for (let i = 0; i < seq_len; i++) {
    input[i] = Math.floor(Math.random() * vocab_size) // Random token indices
  }

  console.log('\\nSample input tokens:', Array.from(input.slice(0, 5)))

  console.log('\\nRunning inference...')
  const output = engine.infer(input, [seq_len])

  console.log(`\\n✅ Classification complete`)
  console.log(`   Class 0 probability: ${(output[0] * 100).toFixed(2)}%`)
  console.log(`   Class 1 probability: ${(output[1] * 100).toFixed(2)}%`)

  engine.free()
}

/**
 * Run all examples
 */
export async function runAllExamples() {
  console.log('🚀 TinyInfer Transformer Examples\\n')
  console.log('='repeat(60))

  try {
    await example1_selfAttention()
    await example2_transformerEncoderLayer()
    await example3_textClassification()

    console.log('\\n' + '='.repeat(60))
    console.log('\\n✅ All examples completed successfully!\\n')
  } catch (error) {
    console.error('\\n❌ Error running examples:', error)
  }
}

// Run examples if in browser
if (typeof window !== 'undefined') {
  console.log('Transformer examples loaded. Run:')
  console.log('  - example1_selfAttention()')
  console.log('  - example2_transformerEncoderLayer()')
  console.log('  - example3_textClassification()')
  console.log('  - runAllExamples()')
}
