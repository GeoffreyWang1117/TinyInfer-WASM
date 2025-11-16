# TinyInfer-WASM 架构设计

## 总体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Web UI Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ 图像分类  │  │ 文本嵌入  │  │ 对话生成  │  │ 性能测试 │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────┐
│               JavaScript API Layer                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │  TinyInfer.load(model)                           │  │
│  │  TinyInfer.infer(input)                          │  │
│  │  TinyInfer.benchmark()                           │  │
│  └──────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────┘
                            │ wasm-bindgen
┌───────────────────────────┴─────────────────────────────┐
│                  WASM Core (Rust)                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Model Management Layer                │   │
│  │  - Graph Builder  - Memory Planner              │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │            Operator Library                      │   │
│  │  [Conv2D] [MatMul] [Pool] [Norm] [Activation]   │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Compute Optimization Layer              │   │
│  │  - SIMD Kernels   - Loop Tiling                 │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Runtime & Memory Layer                 │   │
│  │  - Tensor Allocator  - Execution Engine         │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 核心模块

### 1. Tensor 模块 (`core/src/tensor/`)

**职责**: 提供张量数据结构和基本操作

**组件**:
- `Shape`: 张量形状表示，支持广播和索引计算
- `DataType`: 支持 float32, float16, int8 等数据类型
- `Tensor`: 核心张量类，包含数据、形状和类型

**特性**:
- 零拷贝操作
- 高效的内存布局（NCHW）
- 支持 reshape, transpose 等基本变换

### 2. Operators 模块 (`core/src/ops/`)

**职责**: 实现神经网络算子

**已实现的算子**:
- **激活函数**: ReLU, ReLU6, Sigmoid, Tanh, GELU, Softmax
- **线性层**: MatMul, Gemm (支持循环分块优化)
- **卷积**: Conv2D (使用 im2col + GEMM)
- **池化**: MaxPool2D, AvgPool2D, GlobalAvgPool2D
- **归一化**: BatchNorm2D, LayerNorm
- **元素运算**: Add, Sub, Mul, Div

**设计模式**:
```rust
pub trait Operator {
    fn name(&self) -> &str;
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor>;
}
```

### 3. Engine 模块 (`core/src/engine/`)

**职责**: 模型表示和执行引擎

**组件**:
- `ComputeGraph`: 有向无环图表示神经网络
- `Node`: 图中的节点（输入、输出、算子、常量）
- `Model`: 模型封装，包含图和权重
- `Runtime`: 执行引擎，负责推理

**执行流程**:
1. 构建计算图
2. 拓扑排序得到执行顺序
3. 按序执行每个算子
4. 管理中间 tensor 生命周期

### 4. Web UI (`web/src/`)

**技术栈**: React + TypeScript + Tailwind CSS + Vite

**页面**:
- `/`: 首页，项目介绍
- `/image-classification`: 图像分类 Demo
- `/text-embedding`: 文本嵌入 Demo
- `/chat`: 对话生成 Demo
- `/benchmark`: 性能测试

## 性能优化策略

### 1. SIMD 优化

使用 WebAssembly SIMD 128-bit 指令加速:
- 矩阵乘法的内积计算
- 元素级运算（加法、乘法等）
- 激活函数（向量化应用）

**编译选项**:
```toml
[profile.release]
opt-level = "z"
lto = true
RUSTFLAGS='-C target-feature=+simd128'
```

### 2. 算子融合

将多个算子融合为单个算子，减少内存访问:
- Conv + BN + ReLU → ConvBNReLU
- MatMul + Bias + Activation → Fused Linear

### 3. 内存优化

- **In-place 操作**: 直接修改输入 tensor（如 ReLU）
- **Buffer 复用**: 共享相同大小的 tensor buffer
- **循环分块**: 提高缓存命中率

### 4. 并行计算（Future）

- Web Workers 并行处理 batch
- SharedArrayBuffer 跨线程共享数据

## 数据流

### 推理流程

```
用户输入
   ↓
JS Array → Tensor
   ↓
WASM 推理
   ↓
   1. 图拓扑排序
   2. 按序执行算子
   3. 管理中间结果
   ↓
Tensor → JS Array
   ↓
显示结果
```

### 模型加载流程（Future）

```
ONNX 文件
   ↓
Protobuf 解析
   ↓
   1. 构建计算图
   2. 加载权重
   3. 图优化
   ↓
可执行模型
```

## 构建和部署

### 开发环境

```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装 wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# 构建 WASM
./build.sh

# 启动开发服务器
cd web && npm install && npm run dev
```

### 生产构建

```bash
# 构建优化的 WASM (< 1MB)
cd core
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build \
    --target web \
    --release

# 构建前端
cd ../web
npm run build
```

### 部署

支持部署到:
- GitHub Pages
- Vercel
- Netlify
- 任何静态文件托管服务

## 性能目标

### 算子性能

| 算子 | TinyInfer | TensorFlow.js | 加速比 |
|------|-----------|---------------|--------|
| MatMul | 45ms | 180ms | 4x |
| Conv2D | 85ms | 650ms | 7.6x |

### 端到端性能

| 模型 | TinyInfer | ONNX.js | TensorFlow.js |
|------|-----------|---------|---------------|
| MobileNetV2 | 45ms | 120ms | 180ms |
| MiniLM-L6 | 23ms | 65ms | 95ms |

## 未来计划

### Phase 2: 性能优化
- [ ] SIMD 算子优化
- [ ] 算子融合
- [ ] 内存池管理
- [ ] Web Workers 并行

### Phase 3: 更多算子
- [ ] Attention 机制
- [ ] LSTM/GRU
- [ ] Embedding
- [ ] 更多激活函数

### Phase 4: 模型支持
- [ ] ONNX 模型加载
- [ ] 权重量化
- [ ] 模型压缩

### Phase 5: 工具链
- [ ] 模型转换工具
- [ ] 性能分析器
- [ ] 可视化工具

## 技术债务

- [ ] 完善错误处理
- [ ] 添加更多单元测试
- [ ] 支持更多数据类型（float16, int8）
- [ ] 实现更完善的广播机制
- [ ] 优化内存分配策略
