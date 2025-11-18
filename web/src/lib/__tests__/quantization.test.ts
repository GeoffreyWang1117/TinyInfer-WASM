/**
 * Quantization Tests
 *
 * Tests for INT8 quantization utilities
 */

import { describe, it, expect } from 'vitest'
import {
  computeQuantizationParams,
  quantizeToInt8,
  dequantizeFromInt8,
  calculateSizeReduction,
} from '../quantization'

describe('Quantization Parameters', () => {
  it('should compute symmetric quantization params', () => {
    const data = new Float32Array([-10, -5, 0, 5, 10])
    const params = computeQuantizationParams(data, true)

    // Symmetric quantization: scale = (max - min) / (qmax - qmin)
    // For [-10, 10]: scale = 20 / 255 = 0.0784...
    expect(params.scale).toBeCloseTo(20 / 255, 4)
    // Zero point should be near 0 for symmetric quantization
    expect(Math.abs(params.zeroPoint)).toBeLessThan(1)
  })

  it('should compute asymmetric quantization params', () => {
    const data = new Float32Array([0, 5, 10])
    const params = computeQuantizationParams(data, false)

    expect(params.scale).toBeGreaterThan(0)
    expect(params.zeroPoint).toBeGreaterThanOrEqual(-128)
    expect(params.zeroPoint).toBeLessThanOrEqual(127)
  })
})

describe('Quantization and Dequantization', () => {
  it('should quantize and dequantize correctly', () => {
    const original = new Float32Array([0, 1, 2, 3, 4, 5])
    const shape = [6]

    // Quantize
    const quantized = quantizeToInt8(original, shape)
    expect(quantized.data.length).toBe(6)
    expect(quantized.shape).toEqual(shape)

    // Dequantize
    const dequantized = dequantizeFromInt8(quantized)

    // Should be close to original (small loss)
    for (let i = 0; i < original.length; i++) {
      expect(Math.abs(dequantized[i] - original[i])).toBeLessThan(0.1)
    }
  })

  it('should handle negative values', () => {
    const original = new Float32Array([-5, -2.5, 0, 2.5, 5])
    const shape = [5]

    const quantized = quantizeToInt8(original, shape)
    const dequantized = dequantizeFromInt8(quantized)

    for (let i = 0; i < original.length; i++) {
      expect(Math.abs(dequantized[i] - original[i])).toBeLessThan(0.2)
    }
  })
})

describe('Size Reduction', () => {
  it('should calculate size reduction correctly', () => {
    const originalModel = {
      weights: {
        w1: { dtype: 'float32', data: new Array(250).fill(0.5), shape: [250] },
      },
    }
    const quantizedModel = {
      weights: {
        w1: { dtype: 'int8', data: new Array(250).fill(64), shape: [250] },
      },
    }

    const result = calculateSizeReduction(originalModel, quantizedModel)

    expect(result.originalSize).toBeGreaterThan(0)
    expect(result.quantizedSize).toBeGreaterThan(0)
    expect(result.quantizedSize).toBeLessThan(result.originalSize)
    expect(result.compressionRatio).toBeGreaterThan(1)
  })
})
