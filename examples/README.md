# TinyInfer-WASM 示例代码

本目录包含各种使用 TinyInfer-WASM 的示例代码。

## 📚 示例列表

### 1. 基础推理 (`basic-inference.ts`)

最简单的示例，演示：
- 初始化 WASM 模块
- 创建推理引擎
- 加载测试模型
- 执行推理
- 验证结果

### 2. 性能基准测试 (`benchmark-example.ts`)

演示如何使用 Benchmark API：
- MatMul 性能测试
- ReLU 性能测试
- Conv2D 性能测试
- 结果分析

### 3. 图像分类 (查看 web/src/pages/ImageClassification.tsx)

完整的图像分类流程：
- 图像预处理
- NCHW 格式转换
- 模型推理
- Top-K 预测提取

## 🚀 快速开始

### 在浏览器控制台中运行

1. 启动开发服务器：
```bash
cd web && npm run dev
```

2. 打开浏览器访问 `http://localhost:5173`

3. 打开开发者工具控制台

4. 复制示例代码并运行

### 在 React 组件中使用

```typescript
import { useEffect } from 'react'
import { initWasm, createInferenceEngine } from '@/lib/tinyinfer'

function MyComponent() {
  useEffect(() => {
    async function runInference() {
      await initWasm()
      const engine = createInferenceEngine()
      engine.loadTestModel()

      const input = new Float32Array([1, 2, 3, 4])
      const output = engine.infer(input, [4])

      console.log('Output:', output)
      engine.free()
    }

    runInference()
  }, [])

  return <div>Check console for results</div>
}
```

## 📖 API 参考

### initWasm()

初始化 WASM 模块。

```typescript
await initWasm()
```

### createInferenceEngine()

创建推理引擎实例。

```typescript
const engine = createInferenceEngine()
engine.loadTestModel()
const output = engine.infer(input, shape)
engine.free()
```

### createBenchmark()

创建性能测试实例。

```typescript
const bench = createBenchmark()
const time = bench.benchmarkMatMul(512, 10)
bench.free()
```

## 💡 最佳实践

1. **始终调用 free()**
   ```typescript
   const engine = createInferenceEngine()
   try {
     // 使用引擎...
   } finally {
     engine.free()  // 释放资源
   }
   ```

2. **复用引擎实例**
   ```typescript
   const engine = createInferenceEngine()
   engine.loadTestModel()

   // 多次推理使用同一实例
   for (let i = 0; i < 100; i++) {
     const output = engine.infer(input, shape)
   }

   engine.free()
   ```

3. **异常处理**
   ```typescript
   try {
     await initWasm()
     const engine = createInferenceEngine()
     // ...
   } catch (error) {
     console.error('WASM 初始化失败:', error)
   }
   ```

## 🔗 相关资源

- [API 文档](../docs/API.md)
- [架构文档](../ARCHITECTURE.md)
- [构建指南](../BUILD.md)

## ❓ 常见问题

**Q: 如何加载自己的模型？**
A: 目前支持测试模型。ONNX 模型加载功能正在开发中。

**Q: WASM 初始化失败怎么办？**
A: 检查浏览器是否支持 WebAssembly 和 SIMD。

**Q: 如何提高推理性能？**
A: 使用支持 SIMD 的浏览器，复用引擎实例，批量处理输入。
