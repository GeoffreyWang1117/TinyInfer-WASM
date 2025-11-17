/**
 * TensorUtils Tests
 *
 * Tests for tensor utility functions
 */

import { describe, it, expect } from 'vitest'
import {
  zeros,
  ones,
  full,
  randomTensor,
  calculateSize,
  calculateStrides,
  reshape,
  transpose2D,
  normalize,
  minMaxScale,
  clip,
  computeStats,
  allClose,
} from '../tensorUtils'

describe('Tensor Creation', () => {
  it('should create zeros tensor', () => {
    const result = zeros([2, 3])
    expect(result.length).toBe(6)
    expect(result.every(v => v === 0)).toBe(true)
  })

  it('should create ones tensor', () => {
    const result = ones([2, 3])
    expect(result.length).toBe(6)
    expect(result.every(v => v === 1)).toBe(true)
  })

  it('should create filled tensor', () => {
    const result = full([2, 3], 42)
    expect(result.length).toBe(6)
    expect(result.every(v => v === 42)).toBe(true)
  })

  it('should create random tensor', () => {
    const result = randomTensor([2, 3], 0, 1)
    expect(result.length).toBe(6)
    expect(result.every(v => v >= 0 && v <= 1)).toBe(true)
  })
})

describe('Shape Operations', () => {
  it('should calculate size correctly', () => {
    expect(calculateSize([2, 3, 4])).toBe(24)
    expect(calculateSize([5])).toBe(5)
    expect(calculateSize([2, 2])).toBe(4)
  })

  it('should calculate strides correctly', () => {
    expect(calculateStrides([2, 3, 4])).toEqual([12, 4, 1])
    expect(calculateStrides([5])).toEqual([1])
    expect(calculateStrides([2, 3])).toEqual([3, 1])
  })

  it('should reshape tensor correctly', () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6])
    const result = reshape(data, [2, 3], [3, 2])

    expect(result.shape).toEqual([3, 2])
    expect(Array.from(result.data)).toEqual([1, 2, 3, 4, 5, 6])
  })

  it('should transpose 2D tensor', () => {
    const data = new Float32Array([
      1, 2, 3,
      4, 5, 6
    ])

    const result = transpose2D(data, 2, 3)
    expect(Array.from(result)).toEqual([
      1, 4,
      2, 5,
      3, 6
    ])
  })
})

describe('Normalization', () => {
  it('should normalize with mean and std', () => {
    const data = new Float32Array([0, 1, 2, 3, 4])
    const result = normalize(data, 2, 1)

    expect(Array.from(result)).toEqual([-2, -1, 0, 1, 2])
  })

  it('should min-max scale', () => {
    const data = new Float32Array([0, 5, 10])
    const result = minMaxScale(data, 0, 1)

    expect(Array.from(result)).toEqual([0, 0.5, 1])
  })

  it('should clip values', () => {
    const data = new Float32Array([-5, 0, 5, 10])
    const result = clip(data, 0, 5)

    expect(Array.from(result)).toEqual([0, 0, 5, 5])
  })
})

describe('Statistics', () => {
  it('should compute stats correctly', () => {
    const data = new Float32Array([1, 2, 3, 4, 5])
    const stats = computeStats(data)

    expect(stats.min).toBe(1)
    expect(stats.max).toBe(5)
    expect(stats.mean).toBe(3)
    expect(stats.sum).toBe(15)
    expect(stats.std).toBeCloseTo(1.414, 2)
  })

  it('should check if tensors are close', () => {
    const a = new Float32Array([1.0, 2.0, 3.0])
    const b = new Float32Array([1.0, 2.0, 3.0])
    const c = new Float32Array([1.1, 2.1, 3.1])

    expect(allClose(a, b)).toBe(true)
    expect(allClose(a, c)).toBe(false)
    expect(allClose(a, c, 0.2, 0.2)).toBe(true)
  })
})
