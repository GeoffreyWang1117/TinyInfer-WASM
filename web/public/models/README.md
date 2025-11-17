# TinyInfer 示例模型

本目录包含用于测试和演示的示例模型。

## 📦 可用模型

### 1. simple_linear.json

**描述**: 简单的线性模型，用于基础测试

**架构**:
```
Input (4) → Linear (4x4) → Add Bias → ReLU → Output (4)
```

**用途**:
- 快速测试 TinyInfer 基础功能
- 验证 MatMul、Add、ReLU 算子
- 性能基准测试基线

**使用示例**:
```typescript
import { initWasm, createInferenceEngine, loadModelFromURL } from '@/lib/tinyinfer'

await initWasm()
const engine = createInferenceEngine()
await loadModelFromURL(engine, '/models/simple_linear.json')

const input = new Float32Array([1.0, 2.0, 3.0, 4.0])
const output = engine.infer(input, [4])
```

**预期输出**:
```
输入: [1.0, 2.0, 3.0, 4.0]
输出: [0.6, 1.1, 1.6, 2.1] (经过 0.5x + 0.1 后 ReLU)
```

---

## 🚀 添加新模型

### 方法 1: 使用 ONNX 转换工具

```bash
# 转换 ONNX 模型
python ../../../tools/onnx_to_tinyinfer.py model.onnx model.json

# 移动到 models 目录
mv model.json web/public/models/
```

### 方法 2: 手动创建

参考 `simple_linear.json` 的格式：

```json
{
  "version": "0.8.0",
  "name": "model_name",
  "description": "Model description",
  "graph": {
    "nodes": [...],
    "edges": [...],
    "inputs": [...],
    "outputs": [...]
  },
  "weights": {...}
}
```

---

## 📊 模型大小限制

由于这些模型会被加载到浏览器中，建议：

- **小型模型** (< 5MB): 适合实时演示
- **中型模型** (5-50MB): 可以使用，但加载较慢
- **大型模型** (> 50MB): 不推荐，考虑使用 CDN 或按需加载

---

## 🎯 推荐的演示模型

### 计算机视觉
- MobileNetV2 (轻量级图像分类)
- TinyYOLO (目标检测)
- SqueezeNet (图像分类)

### 自然语言处理
- BERT-tiny (文本嵌入)
- DistilBERT (文本分类)
- GPT-2-small (文本生成)

### 其他
- Simple RNN (序列处理)
- Small Transformer (Attention 演示)

---

## 🔧 模型优化建议

### 1. 量化模型

使用 Phase 8 的量化工具：

```typescript
import { quantizeModelWeights } from '@/lib/quantization'

const quantizedModel = quantizeModelWeights(originalModel)
// 模型大小减少 75%
```

### 2. 模型剪枝

删除不必要的层和参数。

### 3. 使用 CDN

对于大模型，从 CDN 加载：

```typescript
await loadModelFromURL(engine, 'https://cdn.example.com/models/bert.json')
```

---

## 📚 相关文档

- [模型加载指南](../../../docs/MODEL_LOADING.md)
- [ONNX 转换工具](../../../tools/README.md)
- [量化文档](../../../docs/PHASE8_PERFORMANCE_OPT.md#4-int8-量化工具)

---

## 🤝 贡献模型

如果你有有用的示例模型，欢迎提交 PR！

**要求**:
- 模型大小 < 10MB
- 包含详细的 README 说明
- 提供使用示例
- 已测试可正常运行
