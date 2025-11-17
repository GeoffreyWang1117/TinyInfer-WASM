# TinyInfer-WASM 工具库

本目录包含用于 TinyInfer-WASM 的各种工具和脚本。

## 📦 工具列表

### 1. ONNX 转换工具 (`onnx_to_tinyinfer.py`)

将 ONNX 模型转换为 TinyInfer 的 JSON 格式。

#### 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install onnx numpy
```

#### 使用方法

**转换 ONNX 模型:**

```bash
python onnx_to_tinyinfer.py model.onnx output.json
```

**创建示例模型:**

```bash
python onnx_to_tinyinfer.py --create-example example.json
```

#### 支持的算子

- **激活函数**: ReLU, Sigmoid, Tanh, Softmax, GELU
- **线性层**: MatMul, Gemm
- **卷积层**: Conv2D
- **池化层**: MaxPool2D, AvgPool2D, GlobalAvgPool2D
- **归一化**: BatchNorm, LayerNorm
- **Attention**: MultiHeadAttention (Phase 7)
- **位置编码**: PositionalEncoding, PositionalEmbedding, RoPE (Phase 8)
- **形状操作**: Reshape, Transpose, Concat, Split
- **元素操作**: Add, Sub, Mul, Div

#### 示例

```bash
# 转换 BERT 模型
python onnx_to_tinyinfer.py bert.onnx models/bert.json

# 转换 ResNet 模型
python onnx_to_tinyinfer.py resnet50.onnx models/resnet50.json

# 转换 GPT 模型
python onnx_to_tinyinfer.py gpt2.onnx models/gpt2.json
```

#### 输出格式

转换后的 JSON 文件格式：

```json
{
  "version": "1.0",
  "name": "model_name",
  "graph": {
    "nodes": [...],
    "edges": [...],
    "inputs": ["input"],
    "outputs": ["output"]
  },
  "weights": {
    "weight_name": {
      "shape": [4, 4],
      "dtype": "float32",
      "data": [...]
    }
  }
}
```

## 🔧 故障排查

### 问题 1: ModuleNotFoundError: No module named 'onnx'

**解决方案:**

```bash
pip install onnx
```

### 问题 2: 不支持的算子

**症状:** 转换时显示 "Unsupported operator" 警告

**解决方案:**
- 检查算子是否在支持列表中
- 如果是新算子，需要在 TinyInfer 核心中先实现该算子
- 或在脚本中添加算子映射

### 问题 3: 内存不足

**症状:** 转换大模型时内存溢出

**解决方案:**
- 使用更小的模型
- 增加系统内存
- 考虑模型量化

## 📊 性能提示

### 1. 模型优化

在转换之前，可以使用 ONNX 优化器：

```python
import onnx
from onnxoptimizer import optimize

# 加载模型
model = onnx.load("model.onnx")

# 优化
optimized_model = optimize(model)

# 保存
onnx.save(optimized_model, "model_optimized.onnx")
```

### 2. 量化

对于大模型，建议先进行量化：

```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    "model.onnx",
    "model_quantized.onnx",
    weight_type=QuantType.QUInt8
)
```

然后转换量化后的模型。

## 🤝 贡献

欢迎添加新的工具！如果你有有用的脚本，请提交 PR。

### 建议的新工具

- [ ] 模型验证工具
- [ ] 性能基准测试工具
- [ ] 模型可视化工具
- [ ] 批量转换脚本
- [ ] 模型压缩工具

## 📚 相关文档

- [ONNX 官方文档](https://onnx.ai/)
- [模型加载指南](../docs/MODEL_LOADING.md)
- [算子文档](../docs/OPERATORS.md)
- [API 文档](../docs/API.md)

## 🆘 获取帮助

如果遇到问题：

1. 查看 [FAQ](../docs/FAQ.md)
2. 搜索 [GitHub Issues](https://github.com/GeoffreyWang1117/TinyInfer-WASM/issues)
3. 提交新的 Issue
