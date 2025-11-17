/**
 * Model Quantization Utilities for TinyInfer
 *
 * Provides INT8 quantization to reduce model size and improve inference speed.
 */

/**
 * Quantization parameters
 */
export interface QuantizationParams {
  scale: number
  zeroPoint: number
  min: number
  max: number
}

/**
 * Quantized tensor
 */
export interface QuantizedTensor {
  data: Int8Array
  shape: number[]
  params: QuantizationParams
}

/**
 * Compute quantization parameters from float data
 */
export function computeQuantizationParams(
  data: Float32Array,
  symmetric: boolean = false
): QuantizationParams {
  // Find min and max values
  let min = Infinity
  let max = -Infinity

  for (let i = 0; i < data.length; i++) {
    min = Math.min(min, data[i])
    max = Math.max(max, data[i])
  }

  // Symmetric quantization: use max(abs(min), abs(max))
  if (symmetric) {
    const absMax = Math.max(Math.abs(min), Math.abs(max))
    min = -absMax
    max = absMax
  }

  // Compute scale and zero point
  // INT8 range: -128 to 127
  const qmin = -128
  const qmax = 127

  const scale = (max - min) / (qmax - qmin)
  const zeroPoint = Math.round(qmin - min / scale)

  return {
    scale,
    zeroPoint: Math.max(qmin, Math.min(qmax, zeroPoint)),
    min,
    max,
  }
}

/**
 * Quantize float32 tensor to int8
 */
export function quantizeToInt8(
  data: Float32Array,
  shape: number[],
  params?: QuantizationParams
): QuantizedTensor {
  // Compute params if not provided
  const quantParams = params || computeQuantizationParams(data, false)

  // Quantize data
  const quantizedData = new Int8Array(data.length)

  for (let i = 0; i < data.length; i++) {
    const quantized = Math.round(data[i] / quantParams.scale + quantParams.zeroPoint)
    quantizedData[i] = Math.max(-128, Math.min(127, quantized))
  }

  return {
    data: quantizedData,
    shape,
    params: quantParams,
  }
}

/**
 * Dequantize int8 tensor to float32
 */
export function dequantizeFromInt8(quantized: QuantizedTensor): Float32Array {
  const { data, params } = quantized

  const dequantized = new Float32Array(data.length)

  for (let i = 0; i < data.length; i++) {
    dequantized[i] = (data[i] - params.zeroPoint) * params.scale
  }

  return dequantized
}

/**
 * Quantize model weights
 *
 * Converts all float32 weights in a model to int8 for reduced size.
 */
export function quantizeModelWeights(
  modelDef: any,
  symmetric: boolean = true
): {
  quantizedModel: any
  quantizationInfo: Record<string, QuantizationParams>
} {
  const quantizedModel = JSON.parse(JSON.stringify(modelDef))
  const quantizationInfo: Record<string, QuantizationParams> = {}

  // Quantize each weight
  for (const [name, weight] of Object.entries(modelDef.weights)) {
    const weightData = weight as any

    if (weightData.dtype === 'float32' && weightData.data) {
      const floatData = new Float32Array(weightData.data)

      // Compute quantization params
      const params = computeQuantizationParams(floatData, symmetric)
      quantizationInfo[name] = params

      // Quantize
      const quantized = quantizeToInt8(floatData, weightData.shape, params)

      // Update model
      quantizedModel.weights[name] = {
        shape: weightData.shape,
        dtype: 'int8',
        data: Array.from(quantized.data),
        quantization: params,
      }
    }
  }

  return {
    quantizedModel,
    quantizationInfo,
  }
}

/**
 * Dequantize model weights (for loading into engine)
 */
export function dequantizeModelWeights(quantizedModel: any): any {
  const dequantizedModel = JSON.parse(JSON.stringify(quantizedModel))

  for (const [name, weight] of Object.entries(quantizedModel.weights)) {
    const weightData = weight as any

    if (weightData.dtype === 'int8' && weightData.quantization) {
      const int8Data = new Int8Array(weightData.data)

      const quantized: QuantizedTensor = {
        data: int8Data,
        shape: weightData.shape,
        params: weightData.quantization,
      }

      const dequantized = dequantizeFromInt8(quantized)

      dequantizedModel.weights[name] = {
        shape: weightData.shape,
        dtype: 'float32',
        data: Array.from(dequantized),
      }
    }
  }

  return dequantizedModel
}

/**
 * Calculate model size reduction
 */
export function calculateSizeReduction(
  originalModel: any,
  quantizedModel: any
): {
  originalSize: number
  quantizedSize: number
  reduction: number
  compressionRatio: number
} {
  // Calculate sizes
  const originalSize = JSON.stringify(originalModel).length
  const quantizedSize = JSON.stringify(quantizedModel).length

  const reduction = originalSize - quantizedSize
  const compressionRatio = originalSize / quantizedSize

  return {
    originalSize,
    quantizedSize,
    reduction,
    compressionRatio,
  }
}

/**
 * Per-channel quantization (for better accuracy)
 *
 * Quantizes each output channel separately for Conv2D and Linear layers.
 */
export function perChannelQuantize(
  weight: Float32Array,
  shape: number[], // [out_channels, ...]
  symmetric: boolean = true
): {
  quantized: Int8Array
  params: QuantizationParams[]
} {
  const outChannels = shape[0]
  const channelSize = weight.length / outChannels

  const quantized = new Int8Array(weight.length)
  const params: QuantizationParams[] = []

  for (let c = 0; c < outChannels; c++) {
    const start = c * channelSize
    const end = start + channelSize

    const channelData = weight.slice(start, end)

    // Compute params for this channel
    const channelParams = computeQuantizationParams(channelData, symmetric)
    params.push(channelParams)

    // Quantize this channel
    for (let i = 0; i < channelSize; i++) {
      const quantizedVal = Math.round(channelData[i] / channelParams.scale + channelParams.zeroPoint)
      quantized[start + i] = Math.max(-128, Math.min(127, quantizedVal))
    }
  }

  return {
    quantized,
    params,
  }
}

export default {
  computeQuantizationParams,
  quantizeToInt8,
  dequantizeFromInt8,
  quantizeModelWeights,
  dequantizeModelWeights,
  calculateSizeReduction,
  perChannelQuantize,
}
