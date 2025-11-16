/**
 * Image preprocessing utilities for neural network inference
 */

export interface PreprocessConfig {
  targetWidth: number
  targetHeight: number
  normalize: boolean
  mean?: [number, number, number]
  std?: [number, number, number]
}

/**
 * Default config for ImageNet models (MobileNet, ResNet, etc.)
 */
export const IMAGENET_CONFIG: PreprocessConfig = {
  targetWidth: 224,
  targetHeight: 224,
  normalize: true,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
}

/**
 * Load image from file and convert to Float32Array
 */
export async function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = URL.createObjectURL(file)
  })
}

/**
 * Load image from URL
 */
export async function loadImageFromURL(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = url
  })
}

/**
 * Preprocess image for inference
 * Returns Float32Array in NCHW format (batch, channels, height, width)
 */
export function preprocessImage(
  image: HTMLImageElement,
  config: PreprocessConfig = IMAGENET_CONFIG
): Float32Array {
  const { targetWidth, targetHeight, normalize, mean, std } = config

  // Create canvas for image processing
  const canvas = document.createElement('canvas')
  canvas.width = targetWidth
  canvas.height = targetHeight
  const ctx = canvas.getContext('2d')!

  // Draw and resize image
  ctx.drawImage(image, 0, 0, targetWidth, targetHeight)

  // Get image data
  const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight)
  const data = imageData.data // RGBA format

  // Convert to Float32Array in NCHW format
  const size = targetWidth * targetHeight
  const tensor = new Float32Array(3 * size)

  for (let i = 0; i < size; i++) {
    const r = data[i * 4] / 255.0
    const g = data[i * 4 + 1] / 255.0
    const b = data[i * 4 + 2] / 255.0

    if (normalize && mean && std) {
      // Normalize: (x - mean) / std
      tensor[i] = (r - mean[0]) / std[0]                    // R channel
      tensor[size + i] = (g - mean[1]) / std[1]             // G channel
      tensor[2 * size + i] = (b - mean[2]) / std[2]         // B channel
    } else {
      tensor[i] = r                                         // R channel
      tensor[size + i] = g                                  // G channel
      tensor[2 * size + i] = b                              // B channel
    }
  }

  return tensor
}

/**
 * Draw bounding box on canvas
 */
export function drawBoundingBox(
  canvas: HTMLCanvasElement,
  x: number,
  y: number,
  width: number,
  height: number,
  label: string,
  confidence: number,
  color: string = '#00ff00'
) {
  const ctx = canvas.getContext('2d')!

  // Draw box
  ctx.strokeStyle = color
  ctx.lineWidth = 3
  ctx.strokeRect(x, y, width, height)

  // Draw label background
  const labelText = `${label} (${(confidence * 100).toFixed(1)}%)`
  ctx.font = '16px Arial'
  const textWidth = ctx.measureText(labelText).width
  ctx.fillStyle = color
  ctx.fillRect(x, y - 25, textWidth + 10, 25)

  // Draw label text
  ctx.fillStyle = '#000000'
  ctx.fillText(labelText, x + 5, y - 7)
}

/**
 * Softmax function for converting logits to probabilities
 */
export function softmax(logits: Float32Array): Float32Array {
  const maxLogit = Math.max(...Array.from(logits))
  const exps = logits.map(x => Math.exp(x - maxLogit))
  const sumExps = exps.reduce((a, b) => a + b, 0)
  return Float32Array.from(exps.map(x => x / sumExps))
}

/**
 * Get top-k predictions from model output
 */
export function getTopKPredictions(
  output: Float32Array,
  labels: string[],
  k: number = 5
): Array<{ label: string; confidence: number; index: number }> {
  // Apply softmax to get probabilities
  const probabilities = softmax(output)

  // Create array of {index, probability} pairs
  const indexed = Array.from(probabilities).map((prob, idx) => ({
    index: idx,
    probability: prob,
  }))

  // Sort by probability (descending) and take top k
  indexed.sort((a, b) => b.probability - a.probability)
  const topK = indexed.slice(0, k)

  // Map to {label, confidence, index}
  return topK.map(item => ({
    label: labels[item.index] || `Class ${item.index}`,
    confidence: item.probability,
    index: item.index,
  }))
}

/**
 * ImageNet class labels (simplified subset)
 * In production, load full 1000-class labels from a file
 */
export const IMAGENET_LABELS: string[] = [
  '泰迪熊', '金毛猎犬', '拉布拉多犬', '德国牧羊犬', '比格犬',
  '波斯猫', '暹罗猫', '老虎', '狮子', '豹',
  '大象', '长颈鹿', '斑马', '熊猫', '考拉',
  '苹果', '香蕉', '橙子', '草莓', '西瓜',
  '玫瑰', '郁金香', '向日葵', '雏菊', '百合',
  '汽车', '自行车', '摩托车', '飞机', '船',
  '椅子', '桌子', '沙发', '床', '电视',
  '笔记本电脑', '键盘', '鼠标', '手机', '平板',
  '书', '笔', '杯子', '瓶子', '碗',
  '披萨', '汉堡', '热狗', '甜甜圈', '蛋糕',
  // ... would include all 1000 classes in production
]

/**
 * Resize image maintaining aspect ratio
 */
export function resizeImageMaintainAspect(
  image: HTMLImageElement,
  maxWidth: number,
  maxHeight: number
): { width: number; height: number } {
  let width = image.width
  let height = image.height

  // Calculate scaling factor
  const widthRatio = maxWidth / width
  const heightRatio = maxHeight / height
  const scale = Math.min(widthRatio, heightRatio)

  // Apply scale
  width = Math.round(width * scale)
  height = Math.round(height * scale)

  return { width, height }
}

/**
 * Create thumbnail from image
 */
export function createThumbnail(
  image: HTMLImageElement,
  size: number = 128
): string {
  const canvas = document.createElement('canvas')
  const { width, height } = resizeImageMaintainAspect(image, size, size)

  canvas.width = width
  canvas.height = height

  const ctx = canvas.getContext('2d')!
  ctx.drawImage(image, 0, 0, width, height)

  return canvas.toDataURL()
}
