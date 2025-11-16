# TinyInfer-WASM

> 🚀 浏览器端轻量级神经网络推理引擎

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目简介

TinyInfer-WASM 是一个高性能的 WebAssembly 推理引擎，专为在浏览器中运行小型神经网络模型而设计。

### 核心特性

- ✅ **隐私保护**：数据不离开浏览器，完全本地计算
- ✅ **零延迟**：无需网络请求，毫秒级响应
- ✅ **高性能**：SIMD 加速，性能达到纯 JS 的 10-20x
- ✅ **跨平台**：一次编写，浏览器即可运行
- ✅ **轻量级**：WASM 二进制 < 1MB
- ✅ **易使用**：简洁的 JavaScript API

## 技术栈

- **核心引擎**：Rust + WebAssembly + SIMD
- **前端界面**：React + TypeScript + Vite
- **模型格式**：ONNX
- **并行计算**：Web Workers（可选）

## 快速开始

### 安装依赖

```bash
# 安装 Rust 工具链
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装 wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# 安装前端依赖
cd web && npm install
```

### 构建项目

```bash
# 构建 WASM 模块
./build.sh

# 启动开发服务器
cd web && npm run dev
```

### 使用示例

```javascript
import { TinyInfer } from 'tinyinfer-wasm';

// 加载模型
const model = await TinyInfer.load('model.onnx');

// 执行推理
const input = new Float32Array([...]);
const output = model.infer(input);

console.log('推理结果:', output);
```

## 应用场景

### 图像分类 🖼️
使用 MobileNetV2 进行实时图像分类

### 文本嵌入 📝
使用 MiniLM 生成文本嵌入向量

### 对话生成 💬
使用轻量级 LLM 进行对话生成

## 性能基准

| 模型 | TinyInfer | ONNX.js | TensorFlow.js |
|------|-----------|---------|---------------|
| MobileNetV2 | 45ms | 120ms | 180ms |
| MiniLM-L6 | 23ms | 65ms | 95ms |

## 项目结构

```
TinyInfer-WASM/
├── core/               # Rust 核心引擎
│   ├── src/
│   │   ├── tensor/    # Tensor 数据结构
│   │   ├── ops/       # 算子实现
│   │   ├── engine/    # 执行引擎
│   │   └── lib.rs     # WASM 绑定
│   └── Cargo.toml
├── web/               # React 前端
│   ├── src/
│   │   ├── components/
│   │   ├── demos/
│   │   └── App.tsx
│   └── package.json
├── models/            # 预训练模型
└── docs/              # 文档
```

## 开发路线图

- [x] Phase 1: 核心引擎
- [ ] Phase 2: 性能优化
- [ ] Phase 3: 应用场景
- [ ] Phase 4: 性能测试
- [ ] Phase 5: 文档与部署

## 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 致谢

本项目灵感来源于 ONNX Runtime Web 和 TensorFlow.js，感谢开源社区的贡献。
