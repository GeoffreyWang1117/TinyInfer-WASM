/**
 * 张量工具库
 * 提供常用的张量操作和转换函数
 */

export interface TensorInfo {
  shape: number[]
  size: number
  dtype: 'float32' | 'int32' | 'uint8'
  strides: number[]
}

/**
 * 计算张量大小
 */
export function calculateSize(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1)
}

/**
 * 计算步长 (strides)
 */
export function calculateStrides(shape: number[]): number[] {
  const strides: number[] = new Array(shape.length)
  let stride = 1

  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride
    stride *= shape[i]
  }

  return strides
}

/**
 * 验证张量形状
 */
export function validateShape(shape: number[]): boolean {
  return shape.every(dim => dim > 0 && Number.isInteger(dim))
}

/**
 * 张量重塑
 */
export function reshape(
  data: Float32Array,
  originalShape: number[],
  newShape: number[]
): Float32Array {
  const originalSize = calculateSize(originalShape)
  const newSize = calculateSize(newShape)

  if (originalSize !== newSize) {
    throw new Error(
      `Cannot reshape tensor of size ${originalSize} to size ${newSize}`
    )
  }

  return new Float32Array(data)
}

/**
 * 张量转置 (2D)
 */
export function transpose2D(
  data: Float32Array,
  rows: number,
  cols: number
): Float32Array {
  const transposed = new Float32Array(data.length)

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      transposed[j * rows + i] = data[i * cols + j]
    }
  }

  return transposed
}

/**
 * 张量切片
 */
export function slice(
  data: Float32Array,
  shape: number[],
  start: number[],
  end: number[]
): { data: Float32Array; shape: number[] } {
  if (shape.length !== start.length || shape.length !== end.length) {
    throw new Error('Dimension mismatch')
  }

  // 简化实现：仅支持连续切片
  const newShape = start.map((s, i) => end[i] - s)
  const size = calculateSize(newShape)
  const result = new Float32Array(size)

  // 对于 1D 切片
  if (shape.length === 1) {
    result.set(data.slice(start[0], end[0]))
  }

  return { data: result, shape: newShape }
}

/**
 * 张量拼接
 */
export function concatenate(
  tensors: Float32Array[],
  shapes: number[][],
  axis: number = 0
): { data: Float32Array; shape: number[] } {
  if (tensors.length === 0) {
    throw new Error('At least one tensor is required')
  }

  // 验证形状兼容性
  const firstShape = shapes[0]
  for (let i = 1; i < shapes.length; i++) {
    for (let j = 0; j < firstShape.length; j++) {
      if (j !== axis && shapes[i][j] !== firstShape[j]) {
        throw new Error('Shape mismatch for concatenation')
      }
    }
  }

  // 计算新形状
  const newShape = [...firstShape]
  newShape[axis] = shapes.reduce((sum, shape) => sum + shape[axis], 0)

  // 拼接数据
  const totalSize = calculateSize(newShape)
  const result = new Float32Array(totalSize)

  let offset = 0
  tensors.forEach(tensor => {
    result.set(tensor, offset)
    offset += tensor.length
  })

  return { data: result, shape: newShape }
}

/**
 * 张量填充
 */
export function pad(
  data: Float32Array,
  shape: number[],
  padding: number[][],
  value: number = 0
): { data: Float32Array; shape: number[] } {
  if (padding.length !== shape.length) {
    throw new Error('Padding dimension mismatch')
  }

  // 计算新形状
  const newShape = shape.map((dim, i) => dim + padding[i][0] + padding[i][1])
  const newSize = calculateSize(newShape)
  const result = new Float32Array(newSize).fill(value)

  // 简化实现：仅支持 2D
  if (shape.length === 2) {
    const [h, w] = shape
    const [newH, newW] = newShape
    const [padTop, padBottom] = padding[0]
    const [padLeft, padRight] = padding[1]

    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        const srcIdx = i * w + j
        const dstIdx = (i + padTop) * newW + (j + padLeft)
        result[dstIdx] = data[srcIdx]
      }
    }
  }

  return { data: result, shape: newShape }
}

/**
 * 归一化
 */
export function normalize(
  data: Float32Array,
  mean: number = 0,
  std: number = 1
): Float32Array {
  const result = new Float32Array(data.length)

  for (let i = 0; i < data.length; i++) {
    result[i] = (data[i] - mean) / std
  }

  return result
}

/**
 * 标准化 (Min-Max)
 */
export function minMaxScale(
  data: Float32Array,
  min: number = 0,
  max: number = 1
): Float32Array {
  const dataMin = Math.min(...Array.from(data))
  const dataMax = Math.max(...Array.from(data))
  const range = dataMax - dataMin

  if (range === 0) {
    return new Float32Array(data.length).fill((min + max) / 2)
  }

  const result = new Float32Array(data.length)

  for (let i = 0; i < data.length; i++) {
    result[i] = ((data[i] - dataMin) / range) * (max - min) + min
  }

  return result
}

/**
 * 裁剪值
 */
export function clip(
  data: Float32Array,
  min: number,
  max: number
): Float32Array {
  const result = new Float32Array(data.length)

  for (let i = 0; i < data.length; i++) {
    result[i] = Math.max(min, Math.min(max, data[i]))
  }

  return result
}

/**
 * 张量统计
 */
export interface TensorStats {
  min: number
  max: number
  mean: number
  std: number
  sum: number
}

export function computeStats(data: Float32Array): TensorStats {
  const arr = Array.from(data)
  const min = Math.min(...arr)
  const max = Math.max(...arr)
  const sum = arr.reduce((a, b) => a + b, 0)
  const mean = sum / data.length

  const variance = arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length
  const std = Math.sqrt(variance)

  return { min, max, mean, std, sum }
}

/**
 * 打印张量信息
 */
export function printTensorInfo(
  data: Float32Array,
  shape: number[],
  name: string = 'Tensor'
): void {
  const size = calculateSize(shape)
  const stats = computeStats(data)

  console.log(`\n📊 ${name}`)
  console.log('─'.repeat(50))
  console.log(`Shape: [${shape.join(', ')}]`)
  console.log(`Size: ${size}`)
  console.log(`Dtype: float32`)
  console.log(`\nStatistics:`)
  console.log(`  Min: ${stats.min.toFixed(4)}`)
  console.log(`  Max: ${stats.max.toFixed(4)}`)
  console.log(`  Mean: ${stats.mean.toFixed(4)}`)
  console.log(`  Std: ${stats.std.toFixed(4)}`)
  console.log(`\nSample (first 10):`)
  console.log(`  [${Array.from(data.slice(0, 10)).map(v => v.toFixed(4)).join(', ')}${size > 10 ? ', ...' : ''}]`)
}

/**
 * 比较两个张量
 */
export function allClose(
  a: Float32Array,
  b: Float32Array,
  rtol: number = 1e-5,
  atol: number = 1e-8
): boolean {
  if (a.length !== b.length) {
    return false
  }

  for (let i = 0; i < a.length; i++) {
    const diff = Math.abs(a[i] - b[i])
    const threshold = atol + rtol * Math.abs(b[i])
    if (diff > threshold) {
      return false
    }
  }

  return true
}

/**
 * 生成随机张量
 */
export function randomTensor(
  shape: number[],
  min: number = 0,
  max: number = 1
): Float32Array {
  const size = calculateSize(shape)
  const data = new Float32Array(size)

  for (let i = 0; i < size; i++) {
    data[i] = Math.random() * (max - min) + min
  }

  return data
}

/**
 * 生成零张量
 */
export function zeros(shape: number[]): Float32Array {
  const size = calculateSize(shape)
  return new Float32Array(size)
}

/**
 * 生成全一张量
 */
export function ones(shape: number[]): Float32Array {
  const size = calculateSize(shape)
  return new Float32Array(size).fill(1)
}

/**
 * 生成填充张量
 */
export function full(shape: number[], value: number): Float32Array {
  const size = calculateSize(shape)
  return new Float32Array(size).fill(value)
}

/**
 * 数据类型转换
 */
export function convertDtype(
  data: Float32Array | Uint8Array | Int32Array,
  targetType: 'float32' | 'uint8' | 'int32'
): Float32Array | Uint8Array | Int32Array {
  switch (targetType) {
    case 'float32':
      return new Float32Array(data)
    case 'uint8':
      return new Uint8Array(data)
    case 'int32':
      return new Int32Array(data)
    default:
      throw new Error(`Unsupported dtype: ${targetType}`)
  }
}

/**
 * NHWC 转 NCHW (图像格式转换)
 */
export function nhwcToNchw(
  data: Float32Array,
  n: number,
  h: number,
  w: number,
  c: number
): Float32Array {
  const result = new Float32Array(data.length)

  for (let batch = 0; batch < n; batch++) {
    for (let channel = 0; channel < c; channel++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const nhwcIdx = ((batch * h + y) * w + x) * c + channel
          const nchwIdx = ((batch * c + channel) * h + y) * w + x
          result[nchwIdx] = data[nhwcIdx]
        }
      }
    }
  }

  return result
}

/**
 * NCHW 转 NHWC
 */
export function nchwToNhwc(
  data: Float32Array,
  n: number,
  c: number,
  h: number,
  w: number
): Float32Array {
  const result = new Float32Array(data.length)

  for (let batch = 0; batch < n; batch++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        for (let channel = 0; channel < c; channel++) {
          const nchwIdx = ((batch * c + channel) * h + y) * w + x
          const nhwcIdx = ((batch * h + y) * w + x) * c + channel
          result[nhwcIdx] = data[nchwIdx]
        }
      }
    }
  }

  return result
}

/**
 * 批量归一化参数
 */
export interface BatchNormParams {
  mean: Float32Array
  variance: Float32Array
  gamma: Float32Array
  beta: Float32Array
  epsilon: number
}

/**
 * 应用批量归一化
 */
export function applyBatchNorm(
  data: Float32Array,
  shape: number[], // [N, C, H, W]
  params: BatchNormParams
): Float32Array {
  const [n, c, h, w] = shape
  const result = new Float32Array(data.length)

  for (let batch = 0; batch < n; batch++) {
    for (let channel = 0; channel < c; channel++) {
      const mean = params.mean[channel]
      const variance = params.variance[channel]
      const gamma = params.gamma[channel]
      const beta = params.beta[channel]

      const scale = gamma / Math.sqrt(variance + params.epsilon)
      const shift = beta - mean * scale

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const idx = ((batch * c + channel) * h + y) * w + x
          result[idx] = data[idx] * scale + shift
        }
      }
    }
  }

  return result
}
