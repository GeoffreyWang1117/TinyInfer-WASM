# TinyInfer-WASM 工具库文档

本文档详细介绍 TinyInfer-WASM 提供的各种工具库和实用功能。

## 📊 性能分析工具 (`lib/profiler.ts`)

### PerformanceProfiler

详细的性能分析器，用于跟踪推理过程的各个阶段。

**使用示例:**

```typescript
import { createProfiler } from '@/lib/profiler'

const profiler = createProfiler()

// 开始分析
profiler.startProfiling()

// 记录层级性能
profiler.recordLayer('conv1', 10.5)
profiler.recordLayer('relu1', 2.3)

// 结束分析
const metrics = profiler.endProfiling()

console.log('总时间:', metrics.totalTime)
console.log('吞吐量:', metrics.throughput)

// 生成报告
console.log(profiler.generateReport())
```

**输出指标:**
- `totalTime`: 总执行时间 (ms)
- `inferenceTime`: 推理时间 (ms)
- `memoryUsed`: 内存使用 (MB)
- `throughput`: 吞吐量 (inferences/sec)
- `layerTimings`: 各层执行时间

### BatchProfiler

批量性能测试工具，用于统计分析。

**使用示例:**

```typescript
import { createBatchProfiler } from '@/lib/profiler'

const batchProfiler = createBatchProfiler()

// 运行 100 次推理
await batchProfiler.runBatch(
  async () => {
    // 执行推理
    const output = engine.infer(input, shape)
  },
  100
)

// 获取统计信息
const stats = batchProfiler.getStatistics()

console.log('平均值:', stats.mean)
console.log('中位数:', stats.median)
console.log('P95:', stats.p95)
console.log('标准差:', stats.stdDev)

// 生成报告
console.log(batchProfiler.generateStatisticsReport())
```

**统计指标:**
- `mean`: 平均值
- `median`: 中位数
- `min/max`: 最小/最大值
- `stdDev`: 标准差
- `p50/p90/p95/p99`: 百分位数

### PerformanceComparator

性能对比工具，比较不同实现或框架的性能。

```typescript
import { createComparator } from '@/lib/profiler'

const comparator = createComparator()

// 添加测试结果
comparator.addResult('TinyInfer', metrics1)
comparator.addResult('ONNX.js', metrics2)
comparator.addResult('TF.js', metrics3)

// 生成对比报告
console.log(comparator.generateComparisonReport())
```

### PerformanceAdvisor

性能优化建议系统。

```typescript
import { createAdvisor } from '@/lib/profiler'

const advisor = createAdvisor()

// 分析性能并获取建议
const suggestions = advisor.analyze(metrics)

suggestions.forEach(suggestion => {
  console.log(suggestion)
})

// 示例输出:
// ⚠️ 推理时间较长 (>100ms)，考虑使用更小的模型或优化算子
// 💾 内存使用较高 (>100MB)，考虑使用内存池或减少中间张量
```

---

## 🎯 张量工具库 (`lib/tensorUtils.ts`)

### 基础操作

#### 创建张量

```typescript
import { zeros, ones, full, randomTensor } from '@/lib/tensorUtils'

// 零张量
const z = zeros([3, 3])

// 全一张量
const o = ones([3, 3])

// 填充张量
const f = full([3, 3], 42)

// 随机张量
const r = randomTensor([3, 3], 0, 1)
```

#### 形状操作

```typescript
import { reshape, transpose2D, calculateSize, calculateStrides } from '@/lib/tensorUtils'

// 重塑
const reshaped = reshape(data, [4, 4], [2, 8])

// 转置 (2D)
const transposed = transpose2D(data, rows, cols)

// 计算大小
const size = calculateSize([3, 4, 5]) // 60

// 计算步长
const strides = calculateStrides([3, 4, 5]) // [20, 5, 1]
```

#### 张量切片和拼接

```typescript
import { slice, concatenate } from '@/lib/tensorUtils'

// 切片
const { data, shape } = slice(
  tensorData,
  [10],
  [2],  // 开始索引
  [8]   // 结束索引
)

// 拼接
const result = concatenate(
  [tensor1, tensor2, tensor3],
  [shape1, shape2, shape3],
  0  // axis
)
```

### 数据预处理

#### 归一化

```typescript
import { normalize, minMaxScale, clip } from '@/lib/tensorUtils'

// 标准归一化
const normalized = normalize(data, mean=0.5, std=0.2)

// Min-Max 缩放
const scaled = minMaxScale(data, min=0, max=1)

// 裁剪值
const clipped = clip(data, min=-1, max=1)
```

#### 填充

```typescript
import { pad } from '@/lib/tensorUtils'

const { data, shape } = pad(
  tensorData,
  [28, 28],
  [[2, 2], [2, 2]],  // padding: [[top, bottom], [left, right]]
  0  // padding value
)
```

### 格式转换

#### NHWC ↔ NCHW

```typescript
import { nhwcToNchw, nchwToNhwc } from '@/lib/tensorUtils'

// NHWC -> NCHW (TensorFlow -> PyTorch)
const nchw = nhwcToNchw(data, n=1, h=224, w=224, c=3)

// NCHW -> NHWC
const nhwc = nchwToNhwc(data, n=1, c=3, h=224, w=224)
```

#### 数据类型转换

```typescript
import { convertDtype } from '@/lib/tensorUtils'

// Float32 -> Uint8
const uint8Data = convertDtype(float32Data, 'uint8')

// Uint8 -> Float32
const float32Data = convertDtype(uint8Data, 'float32')
```

### 统计分析

```typescript
import { computeStats, printTensorInfo, allClose } from '@/lib/tensorUtils'

// 计算统计信息
const stats = computeStats(data)
console.log(stats)
// { min: -1.5, max: 2.3, mean: 0.5, std: 0.8, sum: 100.0 }

// 打印张量信息
printTensorInfo(data, [3, 224, 224], 'input')

// 比较两个张量
const isClose = allClose(tensor1, tensor2, rtol=1e-5, atol=1e-8)
```

### 批量归一化

```typescript
import { applyBatchNorm, BatchNormParams } from '@/lib/tensorUtils'

const params: BatchNormParams = {
  mean: new Float32Array([0.5, 0.5, 0.5]),
  variance: new Float32Array([0.2, 0.2, 0.2]),
  gamma: new Float32Array([1.0, 1.0, 1.0]),
  beta: new Float32Array([0.0, 0.0, 0.0]),
  epsilon: 1e-5
}

const normalized = applyBatchNorm(data, [1, 3, 224, 224], params)
```

---

## 🎨 图像预处理 (`lib/imagePreprocessing.ts`)

### 图像加载

```typescript
import { loadImage, loadImageFromURL } from '@/lib/imagePreprocessing'

// 从文件加载
const img = await loadImage(file)

// 从 URL 加载
const img = await loadImageFromURL('https://example.com/image.jpg')
```

### 图像预处理

```typescript
import { preprocessImage, IMAGENET_CONFIG } from '@/lib/imagePreprocessing'

// 预处理图像 (NCHW 格式)
const tensor = preprocessImage(img, IMAGENET_CONFIG)

// 自定义配置
const customConfig = {
  targetWidth: 256,
  targetHeight: 256,
  normalize: true,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225]
}

const tensor = preprocessImage(img, customConfig)
```

### 后处理

```typescript
import { softmax, getTopKPredictions, IMAGENET_LABELS } from '@/lib/imagePreprocessing'

// Softmax
const probabilities = softmax(logits)

// Top-K 预测
const predictions = getTopKPredictions(
  output,
  IMAGENET_LABELS,
  k=5
)

predictions.forEach(pred => {
  console.log(`${pred.label}: ${pred.confidence * 100}%`)
})
```

### 可视化

```typescript
import { drawBoundingBox } from '@/lib/imagePreprocessing'

// 绘制边界框
drawBoundingBox(
  canvas,
  x=100, y=100,
  width=200, height=200,
  label='猫',
  confidence=0.95,
  color='#00ff00'
)
```

---

## 💡 最佳实践

### 1. 性能分析

```typescript
// 完整的性能分析流程
import { createProfiler, createBatchProfiler, createAdvisor } from '@/lib/profiler'

async function analyzePerformance() {
  // 批量测试
  const batchProfiler = createBatchProfiler()
  await batchProfiler.runBatch(runInference, 100)

  const stats = batchProfiler.getStatistics()
  console.log(batchProfiler.generateStatisticsReport())

  // 获取建议
  const advisor = createAdvisor()
  const suggestions = advisor.analyze({
    totalTime: stats.mean,
    inferenceTime: stats.mean,
    throughput: 1000 / stats.mean,
    memoryUsed: 0,
    initTime: 0,
    postProcessTime: 0,
    layerTimings: new Map()
  })

  suggestions.forEach(s => console.log(s))
}
```

### 2. 图像处理管道

```typescript
import { loadImage, preprocessImage, IMAGENET_CONFIG } from '@/lib/imagePreprocessing'
import { initWasm, createInferenceEngine } from '@/lib/tinyinfer'
import { getTopKPredictions, IMAGENET_LABELS } from '@/lib/imagePreprocessing'

async function imageClassificationPipeline(file: File) {
  // 1. 加载图像
  const img = await loadImage(file)

  // 2. 预处理
  const input = preprocessImage(img, IMAGENET_CONFIG)

  // 3. 推理
  await initWasm()
  const engine = createInferenceEngine()
  engine.loadTestModel()
  const output = engine.infer(input, [1, 3, 224, 224])

  // 4. 后处理
  const predictions = getTopKPredictions(output, IMAGENET_LABELS, 5)

  // 5. 显示结果
  predictions.forEach(p => {
    console.log(`${p.label}: ${(p.confidence * 100).toFixed(1)}%`)
  })

  engine.free()
}
```

### 3. 张量操作链

```typescript
import {
  randomTensor,
  normalize,
  reshape,
  nhwcToNchw,
  computeStats
} from '@/lib/tensorUtils'

// 数据处理管道
function processTensor() {
  // 生成随机数据
  let data = randomTensor([224, 224, 3], 0, 255)

  // 归一化
  data = normalize(data, 127.5, 127.5)

  // 转换格式
  data = nhwcToNchw(data, 1, 224, 224, 3)

  // 检查统计
  const stats = computeStats(data)
  console.log(stats)

  return data
}
```

---

## 🔧 调试技巧

### 1. 打印张量信息

```typescript
import { printTensorInfo } from '@/lib/tensorUtils'

printTensorInfo(tensor, [3, 224, 224], '输入张量')
```

### 2. 验证张量值

```typescript
import { allClose, computeStats } from '@/lib/tensorUtils'

// 检查数值稳定性
const stats = computeStats(output)
if (stats.min < -100 || stats.max > 100) {
  console.warn('数值范围异常！')
}

// 比较输出
if (!allClose(output1, output2)) {
  console.error('输出不一致！')
}
```

### 3. 性能监控

```typescript
import { createProfiler } from '@/lib/profiler'

const profiler = createProfiler()

profiler.startProfiling()
// ... 执行操作
const metrics = profiler.endProfiling()

if (metrics.totalTime > 100) {
  console.warn('性能较差，耗时:', metrics.totalTime)
}
```

---

## 📚 相关文档

- [API 文档](./API.md)
- [架构文档](../ARCHITECTURE.md)
- [性能优化](./PERFORMANCE.md)
- [示例代码](../examples/README.md)

---

## ❓ 常见问题

**Q: 如何选择合适的性能分析工具？**

A:
- 单次测试 → `PerformanceProfiler`
- 批量统计 → `BatchProfiler`
- 对比测试 → `PerformanceComparator`
- 优化建议 → `PerformanceAdvisor`

**Q: NHWC 和 NCHW 有什么区别？**

A:
- NHWC: (Batch, Height, Width, Channels) - TensorFlow 默认
- NCHW: (Batch, Channels, Height, Width) - PyTorch/ONNX 默认

**Q: 如何优化张量操作性能？**

A:
1. 使用 TypedArray (Float32Array) 而非普通数组
2. 避免频繁的格式转换
3. 复用缓冲区
4. 批量处理数据

---

## 🤝 贡献

欢迎为工具库添加新功能！请查看 [CONTRIBUTING.md](../CONTRIBUTING.md) 了解详情。
