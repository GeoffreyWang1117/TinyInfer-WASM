# TinyInfer-WASM 模型加载指南

本文档详细说明如何加载真实模型到 TinyInfer-WASM。

## 目录

- [🆕 浏览器直接加载 ONNX (推荐)](#浏览器直接加载-onnx-推荐)
- [模型格式](#模型格式)
- [加载方式](#加载方式)
- [ONNX 转换](#onnx-转换)
- [JavaScript API](#javascript-api)
- [示例](#示例)
- [最佳实践](#最佳实践)

---

## 🆕 浏览器直接加载 ONNX (推荐)

**新功能！** TinyInfer 现在支持在浏览器中直接加载 ONNX 模型，无需服务器端转换。

### 主要特性

✅ **零服务器依赖** - 完全在浏览器中解析 ONNX 模型
✅ **自动缓存** - IndexedDB 缓存已解析模型，加速后续加载
✅ **即插即用** - 无需 Python 或命令行工具
✅ **支持拖拽** - 直接拖拽 ONNX 文件到浏览器

### 快速开始

#### 1. 从文件加载 ONNX

```typescript
import { initWasm, createInferenceEngine, loadONNXFromFile } from '@/lib/tinyinfer'

// 初始化
await initWasm()
const engine = createInferenceEngine()

// 用户上传 ONNX 文件
const fileInput = document.getElementById('fileInput') as HTMLInputElement
fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0]

  // 直接加载 ONNX (自动缓存)
  await loadONNXFromFile(engine, file)

  console.log('✅ ONNX 模型加载成功！')

  // 运行推理
  const input = new Float32Array([1, 2, 3, 4])
  const output = engine.infer(input, [4])
  console.log('输出:', output)
})
```

#### 2. 从 URL 加载 ONNX

```typescript
import { loadONNXFromURL } from '@/lib/tinyinfer'

await initWasm()
const engine = createInferenceEngine()

// 从远程 URL 加载 ONNX 模型
await loadONNXFromURL(engine, 'https://example.com/models/model.onnx')

console.log('✅ 模型加载成功！')
```

#### 3. 缓存管理

```typescript
import {
  getModelCacheStats,
  clearModelCache
} from '@/lib/tinyinfer'

// 查看缓存统计
const stats = await getModelCacheStats()
console.log(`已缓存模型: ${stats.count}`)
console.log(`缓存大小: ${(stats.totalSize / 1024 / 1024).toFixed(2)} MB`)

// 清除缓存
await clearModelCache()
```

### 支持的 ONNX 算子

浏览器 ONNX 加载器支持以下算子（与 Python 转换工具相同）:

| ONNX 算子 | TinyInfer 算子 | 说明 |
|-----------|---------------|------|
| `Relu` | `ReLU` | ReLU 激活 |
| `Sigmoid` | `Sigmoid` | Sigmoid 激活 |
| `Tanh` | `Tanh` | Tanh 激活 |
| `Softmax` | `Softmax` | Softmax |
| `MatMul` | `MatMul` | 矩阵乘法 |
| `Gemm` | `Gemm` | 通用矩阵乘法 |
| `Conv` | `Conv2D` | 2D 卷积 |
| `MaxPool` | `MaxPool2D` | 最大池化 |
| `AveragePool` | `AvgPool2D` | 平均池化 |
| `GlobalAveragePool` | `GlobalAvgPool2D` | 全局平均池化 |
| `BatchNormalization` | `BatchNorm2D` | 批归一化 |
| `Add` | `Add` | 加法 |
| `Sub` | `Sub` | 减法 |
| `Mul` | `Mul` | 乘法 |
| `Div` | `Div` | 除法 |
| `Transpose` | `Transpose` | 转置 |
| `Reshape` | `Reshape` | 重塑 |
| `Concat` | `Concat` | 拼接 |
| `Split` | `Split` | 分割 |
| `Gather` | `Gather` | 索引收集 |

### 性能对比

| 加载方式 | 首次加载 | 缓存加载 | 依赖 |
|---------|---------|---------|------|
| 浏览器 ONNX | ~200ms | ~50ms | 无 |
| Python 转换 + JSON | ~100ms | ~100ms | Python |

**推荐**: 生产环境使用浏览器直接加载 ONNX，获得最佳用户体验。

### 完整示例

查看 [browser-onnx-example.ts](../examples/browser-onnx-example.ts) 获取完整的浏览器应用示例，包括:

- 拖拽上传 ONNX 文件
- URL 加载
- 缓存管理界面
- 推理演示

---

## 模型格式

TinyInfer 使用 JSON 格式定义模型。模型包含三个主要部分：

### 1. 模型元数据

```json
{
  "version": "1.0",
  "name": "my_model"
}
```

### 2. 计算图

```json
{
  "graph": {
    "nodes": [
      {
        "id": "input",
        "op_type": "Input",
        "inputs": [],
        "outputs": ["input"],
        "attributes": {}
      },
      {
        "id": "relu",
        "op_type": "ReLU",
        "inputs": ["input"],
        "outputs": ["relu_out"],
        "attributes": {}
      },
      {
        "id": "output",
        "op_type": "Output",
        "inputs": ["relu_out"],
        "outputs": [],
        "attributes": {}
      }
    ],
    "edges": [
      {"from": "input", "to": "relu"},
      {"from": "relu", "to": "output"}
    ],
    "inputs": ["input"],
    "outputs": ["relu_out"]
  }
}
```

### 3. 权重数据

```json
{
  "weights": {
    "weight1": {
      "shape": [2, 3],
      "dtype": "float32",
      "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    }
  }
}
```

### 完整示例

```json
{
  "version": "1.0",
  "name": "simple_linear",
  "graph": {
    "nodes": [
      {
        "id": "input",
        "op_type": "Input",
        "inputs": [],
        "outputs": ["input"],
        "attributes": {}
      },
      {
        "id": "matmul",
        "op_type": "MatMul",
        "inputs": ["input", "weight"],
        "outputs": ["matmul_out"],
        "attributes": {}
      },
      {
        "id": "relu",
        "op_type": "ReLU",
        "inputs": ["matmul_out"],
        "outputs": ["relu_out"],
        "attributes": {}
      },
      {
        "id": "output",
        "op_type": "Output",
        "inputs": ["relu_out"],
        "outputs": [],
        "attributes": {}
      }
    ],
    "edges": [
      {"from": "input", "to": "matmul"},
      {"from": "matmul", "to": "relu"},
      {"from": "relu", "to": "output"}
    ],
    "inputs": ["input"],
    "outputs": ["relu_out"]
  },
  "weights": {
    "weight": {
      "shape": [4, 4],
      "dtype": "float32",
      "data": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    }
  }
}
```

---

## 加载方式

TinyInfer 支持三种模型加载方式：

### 1. 从 JSON 字符串加载

**JavaScript:**
```typescript
import { initWasm, createInferenceEngine, loadModelFromJSON } from '@/lib/tinyinfer'

await initWasm()
const engine = createInferenceEngine()

const modelJSON = JSON.stringify({
  version: '1.0',
  name: 'my_model',
  // ... graph and weights ...
})

await loadModelFromJSON(engine, modelJSON)
```

**Rust (WASM):**
```rust
let mut tinyinfer = TinyInfer::new();
tinyinfer.load_model_from_json(json_str)?;
```

### 2. 从 URL 加载

```typescript
import { initWasm, createInferenceEngine, loadModelFromURL } from '@/lib/tinyinfer'

await initWasm()
const engine = createInferenceEngine()

// 从远程 URL 加载
await loadModelFromURL(engine, 'https://example.com/models/my_model.json')

// 或从本地路径加载
await loadModelFromURL(engine, '/models/my_model.json')
```

### 3. 从文件上传加载

**HTML:**
```html
<input type="file" id="modelFile" accept=".json">
```

**JavaScript:**
```typescript
import { initWasm, createInferenceEngine, loadModelFromFile } from '@/lib/tinyinfer'

const fileInput = document.getElementById('modelFile')

fileInput.addEventListener('change', async (event) => {
  const file = event.target.files[0]

  await initWasm()
  const engine = createInferenceEngine()

  await loadModelFromFile(engine, file)
  console.log('Model loaded!')
})
```

---

## ONNX 转换

TinyInfer 提供工具将 ONNX 模型转换为 JSON 格式。

### 安装依赖

```bash
pip install onnx numpy
```

### 转换 ONNX 模型

```bash
python tools/onnx_to_tinyinfer.py model.onnx output.json
```

**参数:**
- `model.onnx`: 输入的 ONNX 模型文件
- `output.json`: 输出的 TinyInfer JSON 文件

### 创建示例模型

```bash
python tools/onnx_to_tinyinfer.py --create-example example.json
```

### 转换脚本功能

转换工具会自动：

1. ✅ 解析 ONNX 模型结构
2. ✅ 转换所有算子类型
3. ✅ 提取权重数据
4. ✅ 构建计算图
5. ✅ 生成 TinyInfer JSON 格式

**算子类型映射:**

| ONNX | TinyInfer |
|------|-----------|
| `Relu` | `ReLU` |
| `Sigmoid` | `Sigmoid` |
| `Tanh` | `Tanh` |
| `Softmax` | `Softmax` |
| `MatMul` | `MatMul` |
| `Gemm` | `Gemm` |
| `Conv` | `Conv2D` |
| `MaxPool` | `MaxPool2D` |
| `AveragePool` | `AvgPool2D` |
| `GlobalAveragePool` | `GlobalAvgPool2D` |
| `BatchNormalization` | `BatchNorm2D` |
| `Add` | `Add` |
| `Sub` | `Sub` |
| `Mul` | `Mul` |
| `Div` | `Div` |

---

## JavaScript API

### initWasm()

初始化 WASM 模块。

```typescript
await initWasm()
```

### createInferenceEngine()

创建推理引擎实例。

```typescript
const engine = createInferenceEngine()
```

### loadModelFromJSON(engine, json)

从 JSON 字符串加载模型。

```typescript
await loadModelFromJSON(engine, jsonString)
```

**参数:**
- `engine: TinyInferInstance` - 推理引擎实例
- `json: string` - JSON 格式的模型定义

**抛出:**
- `Error` - 如果 JSON 格式无效或模型加载失败

### loadModelFromURL(engine, url)

从 URL 加载模型。

```typescript
await loadModelFromURL(engine, url)
```

**参数:**
- `engine: TinyInferInstance` - 推理引擎实例
- `url: string` - 模型 JSON 文件的 URL

**抛出:**
- `Error` - 如果网络请求失败或模型加载失败

### loadModelFromFile(engine, file)

从 File 对象加载模型。

```typescript
await loadModelFromFile(engine, file)
```

**参数:**
- `engine: TinyInferInstance` - 推理引擎实例
- `file: File` - 用户上传的文件对象

**抛出:**
- `Error` - 如果文件读取失败或模型加载失败

### engine.isModelLoaded()

检查模型是否已加载。

```typescript
if (engine.isModelLoaded()) {
  console.log('Model is ready')
}
```

**返回:**
- `boolean` - 模型是否已成功加载

---

## 示例

### 示例 1: 简单 ReLU 模型

```typescript
import { initWasm, createInferenceEngine, loadModelFromJSON } from '@/lib/tinyinfer'

async function loadSimpleReLU() {
  await initWasm()
  const engine = createInferenceEngine()

  const model = {
    version: '1.0',
    name: 'simple_relu',
    graph: {
      nodes: [
        { id: 'input', op_type: 'Input', inputs: [], outputs: ['input'], attributes: {} },
        { id: 'relu', op_type: 'ReLU', inputs: ['input'], outputs: ['relu_out'], attributes: {} },
        { id: 'output', op_type: 'Output', inputs: ['relu_out'], outputs: [], attributes: {} }
      ],
      edges: [
        { from: 'input', to: 'relu' },
        { from: 'relu', to: 'output' }
      ],
      inputs: ['input'],
      outputs: ['relu_out']
    },
    weights: {}
  }

  await loadModelFromJSON(engine, JSON.stringify(model))

  // 执行推理
  const input = new Float32Array([-1, 2, -3, 4])
  const output = engine.infer(input, [4])
  console.log(output) // [0, 2, 0, 4]

  engine.free()
}
```

### 示例 2: 带权重的线性模型

```typescript
async function loadLinearModel() {
  await initWasm()
  const engine = createInferenceEngine()

  const model = {
    version: '1.0',
    name: 'linear',
    graph: {
      nodes: [
        { id: 'input', op_type: 'Input', inputs: [], outputs: ['input'], attributes: {} },
        { id: 'matmul', op_type: 'MatMul', inputs: ['input', 'weight'], outputs: ['output'], attributes: {} },
        { id: 'out', op_type: 'Output', inputs: ['output'], outputs: [], attributes: {} }
      ],
      edges: [
        { from: 'input', to: 'matmul' },
        { from: 'matmul', to: 'out' }
      ],
      inputs: ['input'],
      outputs: ['output']
    },
    weights: {
      weight: {
        shape: [4, 4],
        dtype: 'float32',
        data: Array(16).fill(0).map((_, i) => i === Math.floor(i / 4) * 4 + i % 4 ? 0 : 1)
      }
    }
  }

  await loadModelFromJSON(engine, JSON.stringify(model))

  const input = new Float32Array([1, 2, 3, 4])
  const output = engine.infer(input, [4])
  console.log(output)

  engine.free()
}
```

### 示例 3: 从 ONNX 转换并加载

**步骤 1: 准备 ONNX 模型**

```python
# 使用 PyTorch 导出 ONNX
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

model = SimpleModel()
dummy_input = torch.randn(1, 4)

torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output']
)
```

**步骤 2: 转换为 TinyInfer 格式**

```bash
python tools/onnx_to_tinyinfer.py model.onnx model.json
```

**步骤 3: 加载模型**

```typescript
await initWasm()
const engine = createInferenceEngine()
await loadModelFromURL(engine, '/models/model.json')

const input = new Float32Array([1, 2, 3, 4])
const output = engine.infer(input, [4])
console.log(output)

engine.free()
```

---

## 最佳实践

### 1. 模型验证

加载模型后，始终检查是否成功：

```typescript
await loadModelFromJSON(engine, modelJSON)

if (!engine.isModelLoaded()) {
  throw new Error('Model failed to load')
}
```

### 2. 错误处理

使用 try-catch 处理加载错误：

```typescript
try {
  await loadModelFromURL(engine, url)
} catch (error) {
  console.error('Failed to load model:', error)
  // 显示错误消息给用户
}
```

### 3. 资源清理

使用完毕后释放引擎：

```typescript
const engine = createInferenceEngine()
try {
  await loadModelFromJSON(engine, json)
  const output = engine.infer(input, shape)
} finally {
  engine.free() // 确保释放
}
```

### 4. 模型缓存

对于频繁使用的模型，考虑缓存：

```typescript
const modelCache = new Map<string, string>()

async function loadCachedModel(engine: TinyInferInstance, url: string) {
  if (!modelCache.has(url)) {
    const response = await fetch(url)
    const json = await response.text()
    modelCache.set(url, json)
  }

  await loadModelFromJSON(engine, modelCache.get(url)!)
}
```

### 5. 进度反馈

对于大模型，显示加载进度：

```typescript
async function loadModelWithProgress(engine: TinyInferInstance, url: string) {
  console.log('Loading model...')

  const response = await fetch(url)
  const total = parseInt(response.headers.get('content-length') || '0')
  let loaded = 0

  const reader = response.body!.getReader()
  const chunks: Uint8Array[] = []

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    chunks.push(value)
    loaded += value.length

    const progress = total ? (loaded / total) * 100 : 0
    console.log(`Progress: ${progress.toFixed(2)}%`)
  }

  const blob = new Blob(chunks)
  const text = await blob.text()

  await loadModelFromJSON(engine, text)
  console.log('Model loaded!')
}
```

### 6. 模型版本检查

验证模型版本兼容性：

```typescript
async function loadModelSafely(engine: TinyInferInstance, json: string) {
  const model = JSON.parse(json)

  if (model.version !== '1.0') {
    throw new Error(`Unsupported model version: ${model.version}`)
  }

  await loadModelFromJSON(engine, json)
}
```

---

## 故障排查

### 模型加载失败

**症状:** `Failed to load model` 错误

**检查清单:**
1. JSON 格式是否正确？
2. 版本是否为 "1.0"？
3. 所有节点是否有有效的 op_type？
4. 权重 shape 和 data 长度是否匹配？

### 推理失败

**症状:** `Inference failed` 错误

**检查清单:**
1. 模型是否已加载？`engine.isModelLoaded()`
2. 输入 shape 是否匹配模型输入？
3. 输入数据类型是否为 Float32Array？

### 性能问题

**症状:** 模型加载或推理很慢

**优化建议:**
1. 压缩 JSON (去除空格)
2. 使用 gzip 压缩传输
3. 缓存已加载的模型
4. 使用 CDN 托管模型文件

---

## 相关文档

- [API 参考](./API.md)
- [算子文档](./OPERATORS.md)
- [示例代码](../examples/load-model-example.ts)
- [ONNX 转换工具](../tools/onnx_to_tinyinfer.py)

---

## 下一步

- 查看[完整示例](../examples/load-model-example.ts)
- 尝试转换自己的 ONNX 模型
- 探索[算子文档](./OPERATORS.md)了解支持的操作
- 阅读[性能优化](./PERFORMANCE.md)提升速度
