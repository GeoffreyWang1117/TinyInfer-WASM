# TinyInfer-WASM 快速开始指南

本指南将帮助你快速上手 TinyInfer-WASM 项目。

## 前置要求

### 必需软件

1. **Rust 工具链** (1.70+)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **wasm-pack** (WebAssembly 打包工具)
   ```bash
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   ```

3. **Node.js** (18+) 和 npm
   ```bash
   # 使用 nvm 安装 (推荐)
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
   nvm install 18
   nvm use 18
   ```

4. **WASM 目标平台**
   ```bash
   rustup target add wasm32-unknown-unknown
   ```

## 项目设置

### 1. 克隆项目

```bash
git clone https://github.com/GeoffreyWang1117/TinyInfer-WASM.git
cd TinyInfer-WASM
```

### 2. 构建 WASM 模块

```bash
# 方式 1: 使用构建脚本 (推荐)
chmod +x build.sh
./build.sh

# 方式 2: 手动构建
cd core
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build \
    --target web \
    --out-dir ../web/src/wasm \
    --release
cd ..
```

构建成功后，你会看到：
- `web/src/wasm/tinyinfer_core_bg.wasm` - WASM 二进制文件
- `web/src/wasm/tinyinfer_core.js` - JavaScript 绑定
- `web/src/wasm/tinyinfer_core.d.ts` - TypeScript 类型定义

### 3. 安装前端依赖

```bash
cd web
npm install
```

### 4. 启动开发服务器

```bash
npm run dev
```

访问 http://localhost:3000 即可看到 TinyInfer-WASM 的 Web 界面！

## 项目结构

```
TinyInfer-WASM/
├── core/                   # Rust WASM 核心引擎
│   ├── src/
│   │   ├── tensor/        # Tensor 数据结构
│   │   ├── ops/           # 算子库
│   │   ├── engine/        # 执行引擎
│   │   ├── api.rs         # WASM API
│   │   └── lib.rs         # 入口
│   └── Cargo.toml
│
├── web/                    # React 前端
│   ├── src/
│   │   ├── lib/           # TypeScript SDK
│   │   ├── hooks/         # React Hooks
│   │   ├── pages/         # 页面组件
│   │   └── App.tsx        # 主应用
│   └── package.json
│
├── build.sh               # WASM 构建脚本
└── README.md
```

## 使用示例

### 在 JavaScript 中使用

```typescript
import { initWasm, createInferenceEngine } from './lib/tinyinfer';

// 1. 初始化 WASM 模块
await initWasm();

// 2. 创建推理引擎
const engine = createInferenceEngine();

// 3. 加载测试模型
engine.loadTestModel();

// 4. 准备输入数据
const inputData = new Float32Array([-1, 2, -3, 4]);
const inputShape = [4];

// 5. 执行推理
const output = engine.infer(inputData, inputShape);
console.log('推理结果:', output);
// 输出: [0, 2, 0, 4] (ReLU 应用后)

// 6. 清理
engine.free();
```

### 使用 React Hooks

```tsx
import { useInference } from './hooks/useWasm';

function MyComponent() {
  const inference = useInference();

  const handleInfer = async () => {
    if (!inference.loaded) {
      console.log('WASM 未加载');
      return;
    }

    const input = new Float32Array([1, 2, 3]);
    const shape = [3];

    const output = await inference.runInference(input, shape);
    console.log('结果:', output);
  };

  return (
    <div>
      <button
        onClick={handleInfer}
        disabled={!inference.loaded}
      >
        运行推理
      </button>
    </div>
  );
}
```

### 性能测试

```typescript
import { createBenchmark } from './lib/tinyinfer';

const benchmark = createBenchmark();

// 测试矩阵乘法
const matmulTime = benchmark.benchmarkMatMul(1024, 10);
console.log(`MatMul (1024x1024): ${matmulTime.toFixed(2)}ms`);

// 测试 ReLU
const reluTime = benchmark.benchmarkReLU(1000000, 100);
console.log(`ReLU (1M elements): ${reluTime.toFixed(2)}ms`);

// 测试 Conv2D
const convTime = benchmark.benchmarkConv2D(1, 64, 224, 5);
console.log(`Conv2D (224x224x64): ${convTime.toFixed(2)}ms`);

benchmark.free();
```

## 开发工作流

### 修改 Rust 代码后

1. 重新编译 WASM:
   ```bash
   ./build.sh
   ```

2. 前端会自动热重载

### 修改 TypeScript 代码后

- Vite 会自动热重载，无需手动操作

### 运行测试

```bash
# Rust 测试
cd core
cargo test

# TypeScript 测试 (如果有)
cd web
npm test
```

### 代码检查

```bash
# Rust linting
cd core
cargo fmt
cargo clippy

# TypeScript linting
cd web
npm run lint
```

## 常见问题

### Q: WASM 编译失败怎么办？

A: 确保已安装 wasm32-unknown-unknown 目标:
```bash
rustup target add wasm32-unknown-unknown
```

### Q: 前端显示 "WASM 未加载"？

A: 检查以下几点:
1. 是否成功构建了 WASM (`web/src/wasm/` 目录下是否有文件)
2. 浏览器控制台是否有错误信息
3. 是否使用了支持 WASM 的现代浏览器

### Q: 如何启用 SIMD 优化？

A: 构建时添加 SIMD 特性标志:
```bash
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --target web --release
```

### Q: 性能测试结果不准确？

A: 注意:
1. 首次运行可能较慢（JIT 编译）
2. 在 release 模式下测试性能
3. 关闭浏览器开发工具
4. 运行多次取平均值

## 下一步

- 查看 [ARCHITECTURE.md](ARCHITECTURE.md) 了解架构设计
- 查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何贡献
- 探索 `web/src/pages/` 中的示例页面
- 尝试添加新的算子到 `core/src/ops/`

## 获取帮助

- 查看 [GitHub Issues](https://github.com/GeoffreyWang1117/TinyInfer-WASM/issues)
- 阅读项目文档
- 查看代码注释和测试用例

祝你使用愉快！🚀
