# TinyInfer-WASM API 参考文档

本文档提供了 TinyInfer-WASM 的完整 API 参考。

## 目录

- [JavaScript API](#javascript-api)
  - [初始化](#初始化)
  - [推理引擎](#推理引擎)
  - [性能测试](#性能测试)
  - [性能分析](#性能分析)
  - [张量工具](#张量工具)
- [Rust API](#rust-api)
  - [Tensor](#tensor)
  - [Operators](#operators)
  - [Engine](#engine)

---

## JavaScript API

### 初始化

#### `initWasm()`

初始化 WebAssembly 模块。必须在使用其他 API 之前调用。

**签名:**
```typescript
function initWasm(): Promise<WasmModule>
```

**返回值:**
- `Promise<WasmModule>` - 加载的 WASM 模块

**示例:**
```typescript
import { initWasm } from '@/lib/tinyinfer'

await initWasm()
```

**错误处理:**
```typescript
try {
  await initWasm()
} catch (error) {
  console.error('WASM 初始化失败:', error)
  // 会自动回退到 mock 实现
}
```

---

### 推理引擎

#### `createInferenceEngine()`

创建一个推理引擎实例。

**签名:**
```typescript
function createInferenceEngine(): InferenceEngine
```

**返回值:**
- `InferenceEngine` - 推理引擎实例

**InferenceEngine 接口:**

##### `loadTestModel()`

加载内置的测试模型（ReLU 激活函数）。

**签名:**
```typescript
loadTestModel(): void
```

**示例:**
```typescript
const engine = createInferenceEngine()
engine.loadTestModel()
```

##### `infer(data, shape)`

执行推理。

**签名:**
```typescript
infer(data: Float32Array, shape: number[]): Float32Array
```

**参数:**
- `data: Float32Array` - 输入数据
- `shape: number[]` - 输入张量的形状

**返回值:**
- `Float32Array` - 输出数据

**示例:**
```typescript
const input = new Float32Array([-1, 2, -3, 4, 5])
const shape = [5]
const output = engine.infer(input, shape)
// output: [0, 2, 0, 4, 5] (ReLU applied)
```

##### `free()`

释放引擎占用的资源。

**签名:**
```typescript
free(): void
```

**重要:** 使用完毕后必须调用此方法，否则会导致内存泄漏。

**示例:**
```typescript
const engine = createInferenceEngine()
try {
  engine.loadTestModel()
  const output = engine.infer(input, shape)
} finally {
  engine.free() // 确保释放资源
}
```

---

### 性能测试

#### `createBenchmark()`

创建性能测试实例。

**签名:**
```typescript
function createBenchmark(): Benchmark
```

**Benchmark 接口:**

##### `benchmarkMatMul(size, iterations)`

测试矩阵乘法性能。

**签名:**
```typescript
benchmarkMatMul(size: number, iterations: number): number
```

**参数:**
- `size: number` - 矩阵大小 (size × size)
- `iterations: number` - 迭代次数

**返回值:**
- `number` - 平均执行时间（毫秒）

**示例:**
```typescript
const bench = createBenchmark()
const time = bench.benchmarkMatMul(512, 10)
console.log(`MatMul (512×512): ${time.toFixed(2)}ms`)
bench.free()
```

##### `benchmarkReLU(size, iterations)`

测试 ReLU 激活函数性能。

**签名:**
```typescript
benchmarkReLU(size: number, iterations: number): number
```

**参数:**
- `size: number` - 张量元素数量
- `iterations: number` - 迭代次数

**返回值:**
- `number` - 平均执行时间（毫秒）

##### `benchmarkConv2D(batchSize, channels, imageSize, iterations)`

测试 2D 卷积性能。

**签名:**
```typescript
benchmarkConv2D(
  batchSize: number,
  channels: number,
  imageSize: number,
  iterations: number
): number
```

**参数:**
- `batchSize: number` - 批次大小
- `channels: number` - 通道数
- `imageSize: number` - 图像尺寸
- `iterations: number` - 迭代次数

**返回值:**
- `number` - 平均执行时间（毫秒）

##### `free()`

释放资源。

---

### 性能分析

#### `createProfiler()`

创建性能分析器实例。

**签名:**
```typescript
function createProfiler(): PerformanceProfiler
```

**PerformanceProfiler 接口:**

##### `startProfiling()`

开始性能分析。

**签名:**
```typescript
startProfiling(): void
```

##### `recordLayer(name, time)`

记录层的执行时间。

**签名:**
```typescript
recordLayer(name: string, time: number): void
```

**参数:**
- `name: string` - 层名称
- `time: number` - 执行时间（毫秒）

##### `endProfiling()`

结束性能分析并返回指标。

**签名:**
```typescript
endProfiling(): PerformanceMetrics
```

**返回值:**
```typescript
interface PerformanceMetrics {
  totalTime: number          // 总时间（毫秒）
  inferenceTime: number      // 推理时间（毫秒）
  throughput: number         // 吞吐量（样本/秒）
  memoryUsage: number        // 内存使用（MB）
  layerTimings: Map<string, number> // 各层平均时间
}
```

**示例:**
```typescript
import { createProfiler } from '@/lib/profiler'

const profiler = createProfiler()
profiler.startProfiling()

profiler.recordLayer('conv1', 10.5)
profiler.recordLayer('relu1', 2.3)

const metrics = profiler.endProfiling()
console.log(profiler.generateReport())
```

#### `createBatchProfiler()`

创建批量性能分析器。

**签名:**
```typescript
function createBatchProfiler(): BatchProfiler
```

**BatchProfiler 接口:**

##### `runBatch(inferenceFunc, iterations)`

运行批量测试。

**签名:**
```typescript
async runBatch(
  inferenceFunc: () => Promise<void>,
  iterations: number
): Promise<void>
```

**参数:**
- `inferenceFunc: () => Promise<void>` - 推理函数
- `iterations: number` - 迭代次数

##### `getStatistics()`

获取统计信息。

**签名:**
```typescript
getStatistics(): BatchStatistics
```

**返回值:**
```typescript
interface BatchStatistics {
  mean: number      // 平均值
  median: number    // 中位数
  min: number       // 最小值
  max: number       // 最大值
  stdDev: number    // 标准差
  p50: number       // 50% 分位数
  p90: number       // 90% 分位数
  p95: number       // 95% 分位数
  p99: number       // 99% 分位数
}
```

**示例:**
```typescript
import { createBatchProfiler } from '@/lib/profiler'

const profiler = createBatchProfiler()
await profiler.runBatch(async () => {
  const output = engine.infer(input, shape)
}, 100)

const stats = profiler.getStatistics()
console.log(`平均: ${stats.mean.toFixed(2)}ms`)
console.log(`P95: ${stats.p95.toFixed(2)}ms`)
```

#### `createPerformanceComparator()`

创建性能对比工具。

**签名:**
```typescript
function createPerformanceComparator(): PerformanceComparator
```

##### `compare(results)`

对比多个性能结果。

**签名:**
```typescript
compare(results: PerformanceResult[]): ComparisonReport
```

---

### 张量工具

#### 创建张量

##### `zeros(shape)`

创建全零张量。

**签名:**
```typescript
function zeros(shape: number[]): Float32Array
```

**示例:**
```typescript
import { zeros } from '@/lib/tensorUtils'

const tensor = zeros([3, 224, 224])
// 创建 3×224×224 的全零张量
```

##### `ones(shape)`

创建全一张量。

##### `randomTensor(shape, min, max)`

创建随机张量。

**签名:**
```typescript
function randomTensor(
  shape: number[],
  min: number = 0,
  max: number = 1
): Float32Array
```

#### 形状操作

##### `reshape(data, oldShape, newShape)`

重塑张量形状。

**签名:**
```typescript
function reshape(
  data: Float32Array,
  oldShape: number[],
  newShape: number[]
): Float32Array
```

##### `transpose2D(data, rows, cols)`

转置 2D 张量。

**签名:**
```typescript
function transpose2D(
  data: Float32Array,
  rows: number,
  cols: number
): Float32Array
```

#### 格式转换

##### `nhwcToNchw(data, n, h, w, c)`

从 NHWC 格式转换为 NCHW 格式。

**签名:**
```typescript
function nhwcToNchw(
  data: Float32Array,
  n: number,  // batch size
  h: number,  // height
  w: number,  // width
  c: number   // channels
): Float32Array
```

**示例:**
```typescript
import { nhwcToNchw } from '@/lib/tensorUtils'

// ImageData 通常是 NHWC 格式
const nhwc = new Float32Array(imageData.data)
const nchw = nhwcToNchw(nhwc, 1, 224, 224, 3)
```

##### `nchwToNhwc(data, n, h, w, c)`

从 NCHW 格式转换为 NHWC 格式。

#### 数据预处理

##### `normalize(data, mean, std)`

标准化张量。

**签名:**
```typescript
function normalize(
  data: Float32Array,
  mean: number | number[],
  std: number | number[]
): Float32Array
```

**示例:**
```typescript
import { normalize } from '@/lib/tensorUtils'

// ImageNet 标准化
const normalized = normalize(
  data,
  [0.485, 0.456, 0.406],  // RGB 均值
  [0.229, 0.224, 0.225]   // RGB 标准差
)
```

##### `minMaxScale(data, min, max)`

最小-最大归一化。

**签名:**
```typescript
function minMaxScale(
  data: Float32Array,
  min: number = 0,
  max: number = 1
): Float32Array
```

##### `clip(data, min, max)`

裁剪张量值到指定范围。

#### 统计分析

##### `computeStats(data)`

计算张量统计信息。

**签名:**
```typescript
function computeStats(data: Float32Array): TensorStats
```

**返回值:**
```typescript
interface TensorStats {
  min: number
  max: number
  mean: number
  std: number
  sum: number
}
```

**示例:**
```typescript
import { computeStats, printTensorInfo } from '@/lib/tensorUtils'

const stats = computeStats(tensor)
console.log(`范围: [${stats.min}, ${stats.max}]`)
console.log(`均值: ${stats.mean}, 标准差: ${stats.std}`)

// 或使用便捷函数
printTensorInfo(tensor, [3, 224, 224], '输入张量')
```

##### `allClose(a, b, rtol, atol)`

检查两个张量是否近似相等。

**签名:**
```typescript
function allClose(
  a: Float32Array,
  b: Float32Array,
  rtol: number = 1e-5,
  atol: number = 1e-8
): boolean
```

---

## Rust API

### Tensor

#### `Tensor::new(data, shape)`

创建新张量。

**签名:**
```rust
pub fn new(data: Vec<f32>, shape: Shape) -> Result<Self>
```

#### `Tensor::zeros(shape)`

创建全零张量。

**签名:**
```rust
pub fn zeros(shape: Shape) -> Self
```

#### `Tensor::shape()`

获取张量形状。

**签名:**
```rust
pub fn shape(&self) -> &Shape
```

### Operators

所有算子实现 `Operator` trait:

```rust
pub trait Operator {
    fn name(&self) -> &str;
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor>;
}
```

#### 激活函数

- `ReLU` - ReLU 激活
- `ReLU6` - ReLU6 激活
- `Sigmoid` - Sigmoid 激活
- `Tanh` - Tanh 激活
- `GELU` - GELU 激活
- `Softmax` - Softmax 激活

#### 线性层

- `MatMul` - 矩阵乘法
- `Gemm` - 通用矩阵乘法（带 alpha/beta）

#### 卷积

- `Conv2D` - 2D 卷积

#### 池化

- `MaxPool2D` - 最大池化
- `AvgPool2D` - 平均池化
- `GlobalAvgPool2D` - 全局平均池化

#### 归一化

- `BatchNorm2D` - 批量归一化
- `LayerNorm` - 层归一化

#### 元素运算

- `Add` - 加法
- `Sub` - 减法
- `Mul` - 乘法
- `Div` - 除法

### Engine

#### `Runtime::new()`

创建运行时。

**签名:**
```rust
pub fn new() -> Self
```

#### `Runtime::execute(graph)`

执行计算图。

**签名:**
```rust
pub fn execute(&mut self, graph: &ComputeGraph) -> Result<Tensor>
```

---

## 类型定义

### TypeScript 类型

```typescript
// WASM 模块接口
interface WasmModule {
  TinyInfer: typeof TinyInfer
  Benchmark: typeof Benchmark
  version: () => string
  checkSimdSupport: () => boolean
  getSystemInfo: () => SystemInfo
}

// 系统信息
interface SystemInfo {
  version: string
  simd_support: boolean
  wasm_memory: number
}

// 性能指标
interface PerformanceMetrics {
  totalTime: number
  inferenceTime: number
  throughput: number
  memoryUsage: number
  layerTimings: Map<string, number>
}

// 批量统计
interface BatchStatistics {
  mean: number
  median: number
  min: number
  max: number
  stdDev: number
  p50: number
  p90: number
  p95: number
  p99: number
}

// 张量统计
interface TensorStats {
  min: number
  max: number
  mean: number
  std: number
  sum: number
}
```

---

## 最佳实践

### 1. 资源管理

始终在使用完毕后释放资源：

```typescript
const engine = createInferenceEngine()
try {
  // 使用引擎...
} finally {
  engine.free()
}
```

### 2. 错误处理

妥善处理 WASM 加载错误：

```typescript
try {
  await initWasm()
} catch (error) {
  console.error('WASM 加载失败:', error)
  // 应用会自动回退到 mock 实现
}
```

### 3. 性能优化

- 复用引擎实例，避免重复创建
- 批量处理多个输入
- 使用性能分析器找出瓶颈

```typescript
// ✅ 好的做法
const engine = createInferenceEngine()
engine.loadTestModel()

for (const input of inputs) {
  const output = engine.infer(input, shape)
  // 处理输出...
}

engine.free()

// ❌ 不好的做法
for (const input of inputs) {
  const engine = createInferenceEngine()
  engine.loadTestModel()
  const output = engine.infer(input, shape)
  engine.free()
}
```

### 4. 类型安全

使用 TypeScript 类型注解：

```typescript
import type { PerformanceMetrics, BatchStatistics } from '@/lib/profiler'

function analyzePerformance(metrics: PerformanceMetrics): void {
  console.log(`推理时间: ${metrics.inferenceTime}ms`)
  console.log(`吞吐量: ${metrics.throughput} 样本/秒`)
}
```

---

## 相关文档

- [快速开始](../QUICKSTART.md)
- [架构设计](../ARCHITECTURE.md)
- [工具文档](./TOOLS.md)
- [算子文档](./OPERATORS.md)
- [示例代码](../examples/README.md)
