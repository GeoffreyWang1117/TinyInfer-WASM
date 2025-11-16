# TinyInfer-WASM 文档中心

欢迎来到 TinyInfer-WASM 文档中心！这里包含了所有你需要的文档。

## 📖 文档导航

### 🚀 入门指南

**新手从这里开始！**

- **[快速开始](../QUICKSTART.md)** ⭐

  5 分钟快速上手，了解如何安装、构建和运行 TinyInfer-WASM

- **[构建指南](../BUILD.md)**

  详细的构建步骤、优化选项和故障排查

- **[架构设计](../ARCHITECTURE.md)**

  了解系统架构、数据流和设计决策

---

### 💻 开发文档

**深入了解 API 和内部实现**

- **[API 参考](./API.md)** ⭐

  完整的 JavaScript 和 Rust API 文档，包含详细示例

  - JavaScript API (initWasm, createInferenceEngine, etc.)
  - 性能分析工具 (Profiler, BatchProfiler)
  - 张量工具库 (tensorUtils)
  - Rust API (Tensor, Operators, Engine)

- **[算子文档](./OPERATORS.md)** ⭐

  所有 17+ 算子的详细说明

  - 激活函数 (ReLU, Sigmoid, GELU, etc.)
  - 线性层 (MatMul, Gemm)
  - 卷积层 (Conv2D)
  - 池化层 (MaxPool, AvgPool)
  - 归一化层 (BatchNorm, LayerNorm)
  - 融合算子 (ConvBNReLU)
  - 如何添加新算子

- **[工具文档](./TOOLS.md)**

  性能分析和张量工具的使用指南

  - PerformanceProfiler - 详细性能分析
  - BatchProfiler - 批量测试和统计
  - PerformanceComparator - 框架对比
  - 30+ 张量操作函数

---

### ⚡ 优化和部署

**生产环境必读**

- **[性能优化指南](./PERFORMANCE.md)** ⭐

  如何榨干每一滴性能

  - 编译优化 (Rust, WASM, 前端)
  - 算子优化 (SIMD, 循环分块, In-place)
  - 应用层优化 (引擎复用, 批处理, Web Workers)
  - 性能分析工具
  - 常见性能问题

- **[部署指南](./DEPLOYMENT.md)** ⭐

  将应用部署到生产环境

  - 构建生产版本
  - 静态网站托管 (Vercel, Netlify, GitHub Pages)
  - CDN 部署
  - Docker 部署
  - 性能优化建议
  - 安全配置
  - 监控和日志

---

### 🔧 其他资源

- **[FAQ](./FAQ.md)**

  常见问题解答

  - 一般问题
  - 安装和构建
  - 使用和开发
  - 性能相关
  - 部署相关
  - 故障排查

- **[CHANGELOG](../CHANGELOG.md)**

  版本更新日志，了解每个版本的新功能和改进

- **[贡献指南](../CONTRIBUTING.md)**

  如何为项目做出贡献

  - 报告 Bug
  - 提出新功能
  - 提交代码
  - 代码风格
  - 开发工作流

- **[示例代码](../examples/README.md)**

  实用的代码示例和最佳实践

  - 基础推理示例
  - 性能基准测试
  - React 集成
  - 图像分类
  - 文本嵌入

---

## 📊 文档统计

| 文档 | 页数 | 主要内容 | 难度 |
|------|------|----------|------|
| **QUICKSTART** | ~12 | 快速上手 | ⭐ 入门 |
| **BUILD** | ~15 | 构建指南 | ⭐⭐ 初级 |
| **ARCHITECTURE** | ~13 | 架构设计 | ⭐⭐⭐ 中级 |
| **API** | ~30 | API 参考 | ⭐⭐ 初级 |
| **OPERATORS** | ~35 | 算子详解 | ⭐⭐⭐ 中级 |
| **TOOLS** | ~18 | 工具使用 | ⭐⭐ 初级 |
| **PERFORMANCE** | ~25 | 性能优化 | ⭐⭐⭐⭐ 高级 |
| **DEPLOYMENT** | ~22 | 部署实践 | ⭐⭐⭐ 中级 |
| **FAQ** | ~20 | 问题解答 | ⭐ 入门 |

**总计:** ~190 页文档 📚

---

## 🎯 学习路径

### 路径 1: 快速体验 (30 分钟)

1. [快速开始](../QUICKSTART.md) - 10 分钟
2. [示例代码](../examples/README.md) - 10 分钟
3. [API 参考](./API.md) - 10 分钟

### 路径 2: 应用开发 (2 小时)

1. [快速开始](../QUICKSTART.md) - 15 分钟
2. [API 参考](./API.md) - 30 分钟
3. [工具文档](./TOOLS.md) - 20 分钟
4. [算子文档](./OPERATORS.md) - 30 分钟
5. [FAQ](./FAQ.md) - 25 分钟

### 路径 3: 深入理解 (1 天)

1. [快速开始](../QUICKSTART.md) - 15 分钟
2. [架构设计](../ARCHITECTURE.md) - 1 小时
3. [算子文档](./OPERATORS.md) - 2 小时
4. [性能优化](./PERFORMANCE.md) - 3 小时
5. [贡献指南](../CONTRIBUTING.md) - 30 分钟
6. 阅读源码 - 剩余时间

### 路径 4: 生产部署 (4 小时)

1. [构建指南](../BUILD.md) - 30 分钟
2. [性能优化](./PERFORMANCE.md) - 1.5 小时
3. [部署指南](./DEPLOYMENT.md) - 1.5 小时
4. [FAQ](./FAQ.md) - 30 分钟

---

## 🔍 快速查找

### 我想...

**...快速上手**
→ [QUICKSTART.md](../QUICKSTART.md)

**...了解 API**
→ [API.md](./API.md)

**...提升性能**
→ [PERFORMANCE.md](./PERFORMANCE.md)

**...部署到生产**
→ [DEPLOYMENT.md](./DEPLOYMENT.md)

**...解决问题**
→ [FAQ.md](./FAQ.md)

**...添加新算子**
→ [OPERATORS.md#算子添加指南](./OPERATORS.md#算子添加指南)

**...贡献代码**
→ [CONTRIBUTING.md](../CONTRIBUTING.md)

**...查看示例**
→ [examples/README.md](../examples/README.md)

---

## 💡 文档改进

发现文档有错误或不清楚的地方？

1. 在 [GitHub Issues](https://github.com/GeoffreyWang1117/TinyInfer-WASM/issues) 报告
2. 提交 Pull Request 改进文档
3. 在 [Discussions](https://github.com/GeoffreyWang1117/TinyInfer-WASM/discussions) 提问

我们欢迎任何文档改进建议！

---

## 📞 获取帮助

**遇到问题？**

1. 📖 查看 [FAQ](./FAQ.md)
2. 🔍 搜索 [GitHub Issues](https://github.com/GeoffreyWang1117/TinyInfer-WASM/issues)
3. 💬 在 [Discussions](https://github.com/GeoffreyWang1117/TinyInfer-WASM/discussions) 提问
4. 🐛 报告新问题

**想要贡献？**

查看 [贡献指南](../CONTRIBUTING.md) 了解如何参与项目！

---

## 🎉 开始你的 TinyInfer 之旅！

选择适合你的学习路径，开始探索 TinyInfer-WASM 吧！

**推荐起点:** [快速开始指南](../QUICKSTART.md) 🚀
