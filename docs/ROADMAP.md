# TinyInfer-WASM 改进路线图

本文档分析当前项目状态，提出改进方向和实施计划。

## 📊 当前状态分析

### ✅ 已完成功能

**核心引擎:**
- [x] Tensor 数据结构
- [x] 17+ 基础算子 (ReLU, Conv2D, MatMul, etc.)
- [x] 计算图执行引擎
- [x] SIMD 优化
- [x] 算子融合 (部分)
- [x] 内存池管理

**模型加载:**
- [x] JSON 模型格式定义
- [x] 从 JSON 字符串加载
- [x] 从 URL 加载
- [x] 从文件上传加载
- [x] Python ONNX 转换工具

**前端集成:**
- [x] TypeScript SDK
- [x] React 演示应用
- [x] 性能分析工具
- [x] 张量工具库

**文档:**
- [x] 完整的 API 文档
- [x] 算子文档
- [x] 性能优化指南
- [x] 部署指南

### ❌ 待改进问题

**关键问题:**

1. **❌ ONNX 需要服务器端转换**
   - 当前: Python 脚本在服务器端转换
   - 问题: 用户不能直接在浏览器加载 ONNX
   - 影响: 无法实现完全本地化运行

2. **❌ 缺少 Attention 算子**
   - 无法运行 Transformer 模型
   - 无法支持 BERT, GPT 等现代模型

3. **❌ 动态形状支持不足**
   - 只支持静态形状
   - 限制了模型的灵活性

4. **❌ 缺少模型缓存**
   - 每次都需要重新加载和解析
   - 浪费带宽和时间

**次要问题:**

5. **⚠️ 算子覆盖不全**
   - 缺少 Reshape, Transpose, Concat, Split
   - 缺少 Embedding, Gather
   - 限制了可运行的模型类型

6. **⚠️ 单线程执行**
   - 没有 Web Worker 支持
   - 阻塞主线程 UI

7. **⚠️ 缺少量化支持**
   - 只支持 FP32
   - 模型大，推理慢

---

## 🎯 改进方向与优先级

### 优先级 P0 (关键功能 - 浏览器本地化)

#### 1. 浏览器端 ONNX 解析器 ⭐⭐⭐⭐⭐

**目标:** 在浏览器中直接解析 ONNX 模型，无需服务器

**方案 A: 轻量级 ONNX.js 解析器**

```typescript
// web/src/lib/onnxParser.ts
import * as onnx from 'onnxjs'  // 或使用 onnx-web

export async function loadONNXModel(file: File): Promise<ModelDef> {
  // 1. 读取 ONNX 文件
  const arrayBuffer = await file.arrayBuffer()

  // 2. 解析 ONNX (使用 protobuf.js)
  const model = onnx.InferenceSession.create(arrayBuffer)

  // 3. 转换为 TinyInfer JSON 格式
  const tinyInferModel = convertONNXToTinyInfer(model)

  return tinyInferModel
}
```

**优点:**
- ✅ 完全本地化，无需服务器
- ✅ 用户可直接上传 ONNX 文件
- ✅ 实时转换，无需预处理

**缺点:**
- ❌ 增加 bundle 大小 (~200KB)
- ❌ 需要处理 protobuf

**方案 B: WASM ONNX 解析器**

在 Rust 中使用 `tract-onnx` 或自己实现:

```rust
// core/src/onnx/mod.rs
use tract_onnx::prelude::*;

pub fn load_onnx_bytes(data: &[u8]) -> Result<ModelDef> {
    let model = tract_onnx::onnx()
        .model_for_read(&mut &data[..])?;

    convert_to_tinyinfer(model)
}
```

**优点:**
- ✅ 性能更好
- ✅ WASM 中统一处理
- ✅ 类型安全

**缺点:**
- ❌ 增加 WASM 大小 (~500KB+)
- ❌ 编译复杂度增加

**推荐方案:** 方案 A (JavaScript 解析)
- 更灵活，易于调试
- Bundle 大小可接受
- 开发速度快

**实施步骤:**
1. 添加 `onnx-proto` 或 `protobufjs` 依赖
2. 实现 ONNX Protobuf 解析
3. 实现算子映射 (复用 Python 脚本逻辑)
4. 实现权重提取
5. 生成 TinyInfer JSON
6. 添加到前端 SDK

**工作量:** 3-5 天

---

#### 2. IndexedDB 模型缓存 ⭐⭐⭐⭐

**目标:** 缓存已加载的模型，避免重复下载和解析

```typescript
// web/src/lib/modelCache.ts
export class ModelCache {
  private db: IDBDatabase

  async cacheModel(name: string, model: ModelDef): Promise<void> {
    const tx = this.db.transaction(['models'], 'readwrite')
    const store = tx.objectStore('models')

    await store.put({
      name,
      model,
      timestamp: Date.now(),
      size: JSON.stringify(model).length
    })
  }

  async getModel(name: string): Promise<ModelDef | null> {
    const tx = this.db.transaction(['models'], 'readonly')
    const store = tx.objectStore('models')
    const result = await store.get(name)

    return result?.model || null
  }

  async clearOldModels(maxAge: number = 7 * 24 * 3600 * 1000) {
    // 清理超过 7 天的缓存
  }
}
```

**功能:**
- 缓存已解析的模型
- 支持版本管理
- 自动清理过期缓存
- 查看缓存状态

**工作量:** 1-2 天

---

### 优先级 P1 (Transformer 支持)

#### 3. Attention 算子实现 ⭐⭐⭐⭐⭐

**必需算子:**

**3.1 MultiHeadAttention**

```rust
// core/src/ops/attention.rs
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    dropout: f32,
}

impl Operator for MultiHeadAttention {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        // inputs: [query, key, value, mask(optional)]
        // query: [batch, seq_len, embed_dim]

        // 1. Split into heads
        let q_heads = self.split_heads(query)?;  // [batch, num_heads, seq_len, head_dim]
        let k_heads = self.split_heads(key)?;
        let v_heads = self.split_heads(value)?;

        // 2. Scaled dot-product attention
        let scores = self.scaled_dot_product(q_heads, k_heads, v_heads, mask)?;

        // 3. Concat heads
        let output = self.concat_heads(scores)?;  // [batch, seq_len, embed_dim]

        Ok(output)
    }
}
```

**3.2 ScaledDotProductAttention**

```rust
pub struct ScaledDotProductAttention;

impl ScaledDotProductAttention {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // 1. Q @ K^T
        let scores = matmul(q, &transpose(k))?;

        // 2. Scale by sqrt(d_k)
        let d_k = q.shape().dims().last().unwrap();
        let scale = 1.0 / (d_k as f32).sqrt();
        let scores = mul_scalar(&scores, scale)?;

        // 3. Apply mask (optional)
        if let Some(mask) = mask {
            scores = apply_mask(&scores, mask)?;
        }

        // 4. Softmax
        let attn = softmax(&scores, -1)?;

        // 5. Attention @ V
        let output = matmul(&attn, v)?;

        Ok(output)
    }
}
```

**支持算子:**

- [x] MatMul (已有)
- [x] Softmax (已有)
- [ ] Transpose (需新增)
- [ ] Reshape (需新增)
- [ ] Split (需新增)
- [ ] Concat (需新增)

**工作量:** 5-7 天

---

#### 4. Transformer 必需算子 ⭐⭐⭐⭐

**4.1 Transpose**

```rust
pub struct Transpose {
    perm: Vec<usize>,  // 轴的排列顺序
}

// 例: [batch, seq, embed] -> [batch, embed, seq]
// perm = [0, 2, 1]
```

**4.2 Reshape**

```rust
pub struct Reshape {
    target_shape: Vec<i64>,  // -1 表示自动推断
}
```

**4.3 Concat**

```rust
pub struct Concat {
    axis: i32,
}
```

**4.4 Split**

```rust
pub struct Split {
    axis: i32,
    split_sizes: Vec<usize>,
}
```

**4.5 Embedding**

```rust
pub struct Embedding {
    num_embeddings: usize,
    embedding_dim: usize,
}
```

**4.6 Gather**

```rust
pub struct Gather {
    axis: i32,
}
```

**工作量:** 4-6 天

---

### 优先级 P2 (性能优化)

#### 5. Web Worker 支持 ⭐⭐⭐

**目标:** 在 Worker 中运行推理，不阻塞 UI

```typescript
// web/src/lib/workerInference.ts
export class WorkerInferenceEngine {
  private worker: Worker

  constructor() {
    this.worker = new Worker(new URL('./inference.worker.ts', import.meta.url))
  }

  async loadModel(modelUrl: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.worker.postMessage({ type: 'load', url: modelUrl })
      this.worker.onmessage = (e) => {
        if (e.data.type === 'loaded') resolve()
        else reject(e.data.error)
      }
    })
  }

  async infer(input: Float32Array, shape: number[]): Promise<Float32Array> {
    return new Promise((resolve) => {
      this.worker.postMessage({ type: 'infer', input, shape })
      this.worker.onmessage = (e) => {
        if (e.data.type === 'result') resolve(e.data.output)
      }
    })
  }
}
```

**inference.worker.ts:**

```typescript
import { initWasm, createInferenceEngine, loadModelFromURL } from './tinyinfer'

let engine: any = null

self.onmessage = async (e) => {
  switch (e.data.type) {
    case 'load':
      await initWasm()
      engine = createInferenceEngine()
      await loadModelFromURL(engine, e.data.url)
      self.postMessage({ type: 'loaded' })
      break

    case 'infer':
      const output = engine.infer(e.data.input, e.data.shape)
      self.postMessage({ type: 'result', output })
      break
  }
}
```

**工作量:** 2-3 天

---

#### 6. 模型量化支持 ⭐⭐⭐

**目标:** 支持 INT8 量化，减少模型大小和加速推理

**量化方式:**

1. **动态量化** (推理时量化)
2. **静态量化** (预先量化)

```rust
// core/src/ops/quantized.rs
pub struct QuantizedLinear {
    weight: Tensor,        // INT8
    bias: Tensor,          // FP32
    scale: f32,
    zero_point: i8,
}

impl Operator for QuantizedLinear {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        // 1. Quantize input: FP32 -> INT8
        let input_q = quantize(inputs[0], self.scale, self.zero_point)?;

        // 2. INT8 MatMul (使用 SIMD)
        let output_q = matmul_int8(&input_q, &self.weight)?;

        // 3. Dequantize output: INT8 -> FP32
        let output = dequantize(&output_q, self.scale, self.zero_point)?;

        // 4. Add bias
        let output = add(&output, &self.bias)?;

        Ok(output)
    }
}
```

**效果:**
- 模型大小: 减少 75% (FP32 -> INT8)
- 推理速度: 提升 2-4x (SIMD INT8)

**工作量:** 5-7 天

---

#### 7. 动态形状支持 ⭐⭐⭐

**目标:** 支持动态 batch size 和 sequence length

```rust
// core/src/tensor/shape.rs
pub enum DimValue {
    Static(usize),
    Dynamic(String),  // 例: "batch_size", "seq_len"
}

pub struct DynamicShape {
    dims: Vec<DimValue>,
}
```

**运行时绑定:**

```typescript
engine.infer(input, {
  batch_size: 4,
  seq_len: 128
})
```

**工作量:** 4-6 天

---

### 优先级 P3 (用户体验)

#### 8. 模型可视化 ⭐⭐

```typescript
// 可视化计算图
export function visualizeModel(model: ModelDef): void {
  // 使用 D3.js 或 Cytoscape.js 绘制计算图
}
```

#### 9. 推理调试工具 ⭐⭐

```typescript
// 逐层查看中间结果
engine.enableDebug()
const intermediates = engine.inferWithIntermediates(input, shape)
```

#### 10. 进度回调 ⭐⭐

```typescript
await loadModelFromURL(engine, url, {
  onProgress: (loaded, total) => {
    console.log(`${(loaded/total*100).toFixed(2)}%`)
  }
})
```

---

## 📋 实施计划

### Phase 6: 浏览器本地化 (2-3 周)

**Week 1-2: ONNX 解析器**
- [ ] 添加 protobufjs 依赖
- [ ] 实现 ONNX Protobuf 解析
- [ ] 实现算子映射
- [ ] 实现权重提取
- [ ] 添加单元测试
- [ ] 集成到前端 SDK

**Week 2-3: 模型缓存**
- [ ] 实现 IndexedDB 缓存层
- [ ] 添加缓存管理 UI
- [ ] 实现版本控制
- [ ] 添加缓存清理

**交付物:**
- ✅ 用户可直接上传 ONNX 文件
- ✅ 模型自动缓存到本地
- ✅ 完全本地化运行

---

### Phase 7: Transformer 支持 (3-4 周)

**Week 1-2: 基础算子**
- [ ] Transpose
- [ ] Reshape
- [ ] Concat
- [ ] Split
- [ ] Embedding
- [ ] Gather

**Week 3: Attention 算子**
- [ ] ScaledDotProductAttention
- [ ] MultiHeadAttention
- [ ] 单元测试

**Week 4: 集成测试**
- [ ] 加载 BERT-tiny 模型
- [ ] 文本分类测试
- [ ] 性能基准测试

**交付物:**
- ✅ 支持 Attention 机制
- ✅ 可运行 BERT/GPT 等模型
- ✅ 完整的测试覆盖

---

### Phase 8: 性能优化 (2-3 周)

**Week 1: Web Worker**
- [ ] Worker 封装
- [ ] 消息通信
- [ ] SharedArrayBuffer 支持

**Week 2-3: 量化**
- [ ] INT8 量化算子
- [ ] 量化工具
- [ ] 性能测试

**交付物:**
- ✅ 推理不阻塞 UI
- ✅ 模型大小减少 75%
- ✅ 推理速度提升 2-4x

---

## 🎯 关键里程碑

### M1: 完全本地化 (3 周后)
- ✅ 浏览器直接解析 ONNX
- ✅ 模型本地缓存
- ✅ 无需服务器

### M2: Transformer 就绪 (7 周后)
- ✅ Attention 算子支持
- ✅ 可运行 BERT/GPT
- ✅ 性能达标

### M3: 生产就绪 (10 周后)
- ✅ Web Worker 支持
- ✅ 模型量化
- ✅ 完整文档和示例

---

## 🔧 技术方案对比

### ONNX 解析方案

| 方案 | 优点 | 缺点 | Bundle 增加 | 推荐 |
|------|------|------|-------------|------|
| **JavaScript (protobufjs)** | 灵活、易调试 | 性能稍差 | ~200KB | ⭐⭐⭐⭐⭐ |
| **WASM (tract-onnx)** | 性能好、类型安全 | 复杂、bundle 大 | ~500KB | ⭐⭐⭐ |
| **混合 (JS 解析 + WASM 执行)** | 平衡 | 复杂度高 | ~300KB | ⭐⭐⭐⭐ |

**推荐:** JavaScript 方案，优先保证功能完整性和开发速度

---

## 📊 预期效果

### 功能完整性

**Phase 6 后:**
- 100% 本地化运行
- 支持直接上传 ONNX
- 自动模型缓存

**Phase 7 后:**
- 支持 90% 的 Transformer 模型
- BERT, GPT-2, T5 等可运行
- 完整的 Attention 支持

**Phase 8 后:**
- 推理速度提升 2-4x
- 模型大小减少 75%
- UI 响应性提升 100%

### 性能指标

| 指标 | 当前 | Phase 6 | Phase 7 | Phase 8 |
|------|------|---------|---------|---------|
| **模型加载时间** | 2s | 1s (缓存) | 1s | 0.5s |
| **BERT-tiny 推理** | N/A | N/A | 50ms | 20ms |
| **模型大小** | 100% | 100% | 100% | 25% |
| **UI 阻塞** | 是 | 是 | 是 | 否 |

---

## 💡 其他建议

### 1. 模型市场
- 提供预转换的模型库
- 用户可直接下载使用
- 社区贡献模型

### 2. 在线转换服务
- 用户上传 ONNX
- 服务器转换 + 优化
- 下载优化后的模型

### 3. 插件系统
- 允许用户自定义算子
- JavaScript 插件支持
- 社区扩展

---

## 📖 相关资源

### ONNX 解析
- [onnx-web](https://github.com/microsoft/onnxruntime/tree/main/js/web)
- [protobufjs](https://github.com/protobufjs/protobuf.js)
- [onnx.proto](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto)

### Attention 实现参考
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [BERT Implementation](https://github.com/google-research/bert)

### 量化
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [TensorFlow Lite Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)

---

## ❓ 常见问题

### Q: 为什么不直接使用 ONNX Runtime Web?

A: ONNX Runtime Web 很强大，但：
1. Bundle 太大 (~2MB)
2. 不够轻量
3. 我们的目标是极致轻量 (<500KB)
4. 学习目的

### Q: Attention 算子的性能如何？

A: 优化后：
- BERT-tiny (4.4M): ~20ms/inference
- BERT-base (110M): ~200ms/inference
- 与 ONNX.js 相当或更快

### Q: 量化会影响精度吗？

A: 通常影响很小：
- INT8 量化: <1% 精度损失
- 对大多数任务可接受
- 可选择性量化（只量化大层）

---

## 🚀 开始实施

优先实施顺序：

1. **Week 1-2**: 浏览器 ONNX 解析器 (P0)
2. **Week 3**: IndexedDB 缓存 (P0)
3. **Week 4-5**: Transformer 基础算子 (P1)
4. **Week 6-7**: Attention 算子 (P1)
5. **Week 8-9**: Web Worker (P2)
6. **Week 10**: 量化支持 (P2)

**立即开始:** Phase 6 - 浏览器本地化

查看各个 Phase 文档了解详细实施步骤：
- [Phase 6 文档](./PHASE6_BROWSER_ONNX.md)
- [Phase 7 文档](./PHASE7_TRANSFORMER_SUPPORT.md)
- [Phase 8 文档](./PHASE8_PERFORMANCE_OPT.md)
