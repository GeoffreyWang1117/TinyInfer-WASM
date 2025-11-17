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

    expect(params.scale).toBeCloseTo(10 / 127, 5)
    expect(params.zeroPoint).toBe(0)
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
    const original = { float32Size: 1000, int8Size: 250 }
    const reduction = calculateSizeReduction(
      original.float32Size,
      original.int8Size
    )

    expect(reduction.reductionRatio).toBeCloseTo(0.75, 2)
    expect(reduction.compressionFactor).toBeCloseTo(4, 1)
  })
})
