# TinyInfer-WASM 常见问题解答 (FAQ)

本文档回答关于 TinyInfer-WASM 的常见问题。

## 目录

- [一般问题](#一般问题)
- [安装和构建](#安装和构建)
- [使用和开发](#使用和开发)
- [性能相关](#性能相关)
- [部署相关](#部署相关)
- [故障排查](#故障排查)

---

## 一般问题

### TinyInfer-WASM 是什么？

TinyInfer-WASM 是一个高性能的 WebAssembly 神经网络推理引擎，专为在浏览器中运行小型模型而设计。它使用 Rust 编写核心引擎，通过 WebAssembly 提供接近原生的性能。

### 为什么选择 TinyInfer-WASM？

**优势:**
- ✅ **隐私保护**: 数据不离开浏览器
- ✅ **低延迟**: 无需网络请求，毫秒级响应
- ✅ **高性能**: SIMD 加速，比纯 JS 快 10-20x
- ✅ **轻量级**: WASM 二进制 < 150KB
- ✅ **易集成**: 简洁的 JavaScript API

**适用场景:**
- 客户端图像分类
- 文本嵌入生成
- 实时对话生成
- 敏感数据处理

### TinyInfer vs TensorFlow.js vs ONNX.js？

| 特性 | TinyInfer | TensorFlow.js | ONNX.js |
|------|-----------|--------------|---------|
| **大小** | ~113KB | ~500KB | ~250KB |
| **MatMul 性能** | 45ms | 580ms | 120ms |
| **SIMD 支持** | ✅ | ✅ | ✅ |
| **GPU 支持** | ❌ | ✅ | ✅ |
| **模型格式** | 自定义 | TF/ONNX | ONNX |
| **适用场景** | 小模型/边缘 | 通用 | 通用 |

**选择建议:**
- **TinyInfer**: 小模型、低延迟要求、隐私敏感
- **TensorFlow.js**: 大模型、需要 GPU 加速
- **ONNX.js**: 已有 ONNX 模型

### 支持哪些模型？

**当前支持:**
- ✅ 简单前馈网络
- ✅ CNN（MobileNet 系列）
- ✅ 小型 Transformer（BERT-tiny, MiniLM）

**计划支持:**
- 🔄 ONNX 模型加载
- 🔄 量化模型
- 🔄 RNN/LSTM

### 支持哪些浏览器？

**完整支持:**
- Chrome/Edge 91+ (SIMD 支持)
- Firefox 89+ (SIMD 支持)
- Safari 16.4+ (SIMD 支持)

**基础支持:**
- Chrome/Edge 57+ (无 SIMD)
- Firefox 52+ (无 SIMD)
- Safari 11+ (无 SIMD)

**检查 SIMD 支持:**
```javascript
const wasm = await initWasm()
console.log('SIMD:', wasm.checkSimdSupport())
```

---

## 安装和构建

### 如何安装依赖？

```bash
# 1. 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. 添加 WASM 目标
rustup target add wasm32-unknown-unknown

# 3. 安装 wasm-pack
cargo install wasm-pack

# 4. 安装 Node.js (18+)
# 使用 nvm 或从 https://nodejs.org 下载

# 5. 安装前端依赖
cd web && npm install
```

### WASM 构建失败怎么办？

**错误: "target 'wasm32-unknown-unknown' not found"**

解决方案:
```bash
rustup target add wasm32-unknown-unknown
```

**错误: "failed to download binaryen"**

解决方案: 在 `core/Cargo.toml` 中禁用 wasm-opt
```toml
[package.metadata.wasm-pack.profile.release]
wasm-opt = false
```

**错误: "error: linker `rust-lld` not found"**

解决方案:
```bash
rustup update
rustup component add rust-src
```

### 如何加速构建？

**1. 使用增量编译**

在 `core/.cargo/config.toml`:
```toml
[build]
incremental = true
```

**2. 使用 sccache**

```bash
cargo install sccache
export RUSTC_WRAPPER=sccache
```

**3. 跳过优化（开发时）**

```bash
wasm-pack build --target web --dev
```

### 前端依赖安装慢？

**使用 pnpm 或 yarn:**
```bash
npm install -g pnpm
cd web
pnpm install  # 更快的包管理器
```

**使用镜像源:**
```bash
npm config set registry https://registry.npmmirror.com
```

---

## 使用和开发

### 如何加载自己的模型？

**当前状态:** ONNX 加载器正在开发中。

**临时方案:** 修改 Rust 代码定义模型

```rust
// core/src/model/my_model.rs
pub fn create_my_model() -> Model {
    let mut graph = ComputeGraph::new();

    // 添加层...
    graph.add_node(Node::Operator(
        "conv1",
        Box::new(Conv2D::new(/*...*/))
    ));

    Model::new(graph)
}
```

### 如何添加新算子？

参考 [算子文档](./OPERATORS.md#算子添加指南)

**步骤:**
1. 在 `core/src/ops/` 创建新文件
2. 实现 `Operator` trait
3. 添加测试
4. 在 `mod.rs` 导出

### 推理速度比预期慢？

**检查清单:**

1. **SIMD 是否启用？**
```typescript
const info = wasm.getSystemInfo()
console.log('SIMD:', info.simd_support)
```

2. **是否使用 release 构建？**
```bash
wasm-pack build --target web --release
```

3. **是否复用引擎实例？**
```typescript
// ✅ 好
const engine = createInferenceEngine()
for (const input of inputs) {
  engine.infer(input, shape)
}
engine.free()

// ❌ 差
for (const input of inputs) {
  const engine = createInferenceEngine()
  engine.infer(input, shape)
  engine.free()
}
```

### 如何调试 WASM 代码？

**1. 使用 console_error_panic_hook**

在 `core/Cargo.toml`:
```toml
[dependencies]
console_error_panic_hook = "0.1"
```

在代码中:
```rust
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}
```

**2. 使用 wasm-logger**

```toml
[dependencies]
wasm-logger = "0.2"
log = "0.4"
```

```rust
#[wasm_bindgen(start)]
pub fn init() {
    wasm_logger::init(wasm_logger::Config::default());
    log::info!("WASM 模块已初始化");
}
```

**3. Chrome DevTools**

- 打开 DevTools → Sources
- 查看 WebAssembly 模块
- 设置断点（如果有 source maps）

### 内存泄漏怎么办？

**症状:** 长时间运行后内存持续增长

**原因:** WASM 实例未释放

**解决方案:**

```typescript
// 始终调用 free()
const engine = createInferenceEngine()
try {
  // 使用引擎...
} finally {
  engine.free()  // 确保释放
}
```

**检测内存泄漏:**
1. Chrome DevTools → Memory
2. Take heap snapshot
3. 运行推理 100 次
4. Take另一个 snapshot
5. 对比差异

---

## 性能相关

### 如何提升性能？

参考 [性能优化指南](./PERFORMANCE.md)

**快速优化:**

1. **启用 SIMD**
```bash
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --release
```

2. **复用引擎实例**
3. **批处理输入**
4. **使用 Web Workers 并行化**
5. **启用压缩** (Brotli/Gzip)

### 如何测试性能？

**内置基准测试:**

```typescript
import { createBenchmark } from '@/lib/tinyinfer'

const bench = createBenchmark()

console.log('MatMul 512:', bench.benchmarkMatMul(512, 10))
console.log('ReLU 1M:', bench.benchmarkReLU(1000000, 100))

bench.free()
```

**详细分析:**

```typescript
import { createBatchProfiler } from '@/lib/profiler'

const profiler = createBatchProfiler()
await profiler.runBatch(async () => {
  engine.infer(input, shape)
}, 100)

const stats = profiler.getStatistics()
console.log(`平均: ${stats.mean}ms`)
console.log(`P95: ${stats.p95}ms`)
console.log(`P99: ${stats.p99}ms`)
```

### 第一次推理很慢？

**原因:** JIT 编译和缓存预热

**解决方案:** 预热运行

```typescript
await initWasm()
const engine = createInferenceEngine()
engine.loadTestModel()

// 预热（丢弃结果）
for (let i = 0; i < 5; i++) {
  engine.infer(dummyInput, shape)
}

// 现在开始实际推理
```

### 如何对比不同框架的性能？

访问性能对比页面:
```
http://localhost:5173/comparison
```

或使用 API:

```typescript
import { createPerformanceComparator } from '@/lib/profiler'

const comparator = createPerformanceComparator()
const report = comparator.compare([
  { name: 'TinyInfer', time: 45 },
  { name: 'ONNX.js', time: 120 },
  { name: 'TF.js', time: 180 },
])
console.log(report)
```

---

## 部署相关

### 如何部署到生产环境？

参考 [部署指南](./DEPLOYMENT.md)

**快速部署 (Vercel):**

```bash
cd web
npm install -g vercel
vercel --prod
```

### WASM 文件 404 错误？

**原因:** WASM 文件路径不正确或服务器配置问题

**解决方案:**

1. **检查文件是否存在**
```bash
ls -la web/public/wasm/tinyinfer_core_bg.wasm
```

2. **检查服务器 MIME 类型**

Nginx:
```nginx
types {
    application/wasm wasm;
}
```

3. **检查路径配置**
```typescript
// 确保路径正确
await init('/wasm/tinyinfer_core_bg.wasm')
```

### CORS 错误？

**错误:** "has been blocked by CORS policy"

**原因:** 跨域请求被阻止

**解决方案:**

**服务器端 (Nginx):**
```nginx
add_header Access-Control-Allow-Origin "*";
add_header Access-Control-Allow-Methods "GET, OPTIONS";
```

**开发时 (Vite):**
```typescript
// vite.config.ts
export default defineConfig({
  server: {
    cors: true
  }
})
```

### 如何启用 HTTPS？

**开发环境:**
```bash
npm run dev -- --https
```

**生产环境 (Nginx):**
```nginx
server {
    listen 443 ssl http2;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    # ...
}
```

**使用 Let's Encrypt:**
```bash
sudo certbot --nginx -d example.com
```

---

## 故障排查

### WASM 加载失败

**检查清单:**

1. **浏览器支持？**
```javascript
if (typeof WebAssembly === 'object') {
  console.log('✅ WebAssembly 支持')
} else {
  console.log('❌ 浏览器不支持 WebAssembly')
}
```

2. **MIME 类型正确？**

打开浏览器 Network 面板，检查 `Content-Type: application/wasm`

3. **文件完整？**
```bash
wasm-validate web/public/wasm/tinyinfer_core_bg.wasm
```

### TypeScript 类型错误

**错误:** "Cannot find module './wasm/tinyinfer_core.js'"

**解决方案:**

1. **确保 WASM 已构建**
```bash
cd core && wasm-pack build --target web --out-dir ../web/public/wasm
```

2. **更新 tsconfig.json**
```json
{
  "compilerOptions": {
    "types": ["vite/client"],
    "allowJs": true
  }
}
```

### 推理结果不正确

**检查清单:**

1. **输入形状正确？**
```typescript
console.log('输入形状:', shape)
console.log('输入大小:', input.length)
console.log('预期大小:', shape.reduce((a, b) => a * b, 1))
```

2. **数据预处理正确？**
```typescript
// 检查数据范围
console.log('最小值:', Math.min(...input))
console.log('最大值:', Math.max(...input))
```

3. **模型加载正确？**
```typescript
engine.loadTestModel()
// 测试简单输入
const test = engine.infer(new Float32Array([1, 2, -3]), [3])
console.log('测试输出:', test)  // 应该是 [1, 2, 0] (ReLU)
```

### 构建产物太大

**WASM 文件优化:**

```bash
# 使用 wasm-opt
wasm-opt -Oz --enable-simd \
  web/public/wasm/tinyinfer_core_bg.wasm \
  -o web/public/wasm/tinyinfer_core_bg.wasm
```

**检查大小:**
```bash
ls -lh web/public/wasm/*.wasm
```

**JavaScript 优化:**

```typescript
// vite.config.ts
export default defineConfig({
  build: {
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
      },
    },
  },
})
```

### 如何报告 Bug？

1. **搜索现有 Issues**: https://github.com/GeoffreyWang1117/TinyInfer-WASM/issues

2. **创建新 Issue**，包含:
   - Bug 描述
   - 复现步骤
   - 预期 vs 实际行为
   - 环境信息（浏览器、OS、版本）
   - 最小可复现示例

3. **提供上下文:**
   - 浏览器控制台日志
   - Network 面板截图
   - 相关代码片段

---

## 获取帮助

**文档:**
- [快速开始](../QUICKSTART.md)
- [API 参考](./API.md)
- [性能优化](./PERFORMANCE.md)
- [部署指南](./DEPLOYMENT.md)

**社区:**
- GitHub Issues: https://github.com/GeoffreyWang1117/TinyInfer-WASM/issues
- Discussions: https://github.com/GeoffreyWang1117/TinyInfer-WASM/discussions

**贡献:**
- 查看 [贡献指南](../CONTRIBUTING.md)
- 提交 Pull Request
- 改进文档

---

## 没有找到答案？

如果你的问题没有在此列出，请:

1. 搜索 [GitHub Issues](https://github.com/GeoffreyWang1117/TinyInfer-WASM/issues)
2. 查看 [Discussions](https://github.com/GeoffreyWang1117/TinyInfer-WASM/discussions)
3. 创建新 Issue 或 Discussion

我们会尽快回复！
