# TinyInfer-WASM 构建指南

本文档详细说明如何构建 TinyInfer-WASM 项目。

## 📋 前置要求

### 必需工具

1. **Rust 工具链** (1.70+)
```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 添加 WebAssembly 目标
rustup target add wasm32-unknown-unknown
```

2. **wasm-pack**
```bash
cargo install wasm-pack
```

3. **Node.js** (18+)
```bash
# 推荐使用 nvm 安装
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18
```

4. **pnpm** (可选，推荐)
```bash
npm install -g pnpm
```

## 🔨 构建步骤

### 1. 构建 WASM 核心模块

```bash
cd core

# 开发构建
wasm-pack build --target web --dev --out-dir ../web/public/wasm

# 生产构建 (优化)
wasm-pack build --target web --release --out-dir ../web/public/wasm
```

**输出文件:**
- `tinyinfer_core_bg.wasm` - WebAssembly 二进制文件 (~113KB)
- `tinyinfer_core.js` - JavaScript 绑定
- `tinyinfer_core.d.ts` - TypeScript 类型定义
- `package.json` - NPM 包配置

### 2. 构建 Web 前端

```bash
cd web

# 安装依赖
pnpm install  # 或 npm install

# 开发服务器
pnpm dev      # 或 npm run dev

# 生产构建
pnpm build    # 或 npm run build
```

**输出目录:** `web/dist/`

### 3. 测试 WASM 模块

#### 方式 1: 独立测试页面

在浏览器中打开:
```
http://localhost:5173/test-wasm.html
```

#### 方式 2: Rust 单元测试

```bash
cd core

# 运行所有测试 (原生环境)
cargo test

# 运行 WASM 测试 (需要浏览器)
wasm-pack test --headless --chrome
```

## 🎯 编译优化

### WASM 大小优化

在 `core/Cargo.toml` 中已配置:

```toml
[profile.release]
opt-level = "z"        # 优化大小
lto = true             # 链接时优化
codegen-units = 1      # 更好的优化
strip = true           # 去除符号
```

### SIMD 支持

SIMD 已自动启用 (WebAssembly SIMD 128-bit):
```rust
#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
```

### 进一步优化 (可选)

如果有 `wasm-opt` 工具:
```bash
# 安装 binaryen
npm install -g binaryen

# 手动优化
wasm-opt -Oz --enable-simd \
  web/public/wasm/tinyinfer_core_bg.wasm \
  -o web/public/wasm/tinyinfer_core_bg_opt.wasm
```

## 🔍 故障排查

### 问题 1: wasm-pack 网络错误

**症状:** 无法下载 binaryen

**解决方案:** 在 `Cargo.toml` 中禁用 wasm-opt
```toml
[package.metadata.wasm-pack.profile.release]
wasm-opt = false
```

### 问题 2: WASM 模块加载失败

**检查清单:**
- ✅ WASM 文件存在于 `web/public/wasm/`
- ✅ 浏览器支持 WebAssembly
- ✅ CORS 设置正确 (本地开发时)
- ✅ 使用 HTTP 服务器提供文件 (不是 file://)

### 问题 3: TypeScript 类型错误

**解决方案:** 确保 tsconfig.json 包含:
```json
{
  "compilerOptions": {
    "types": ["vite/client"],
    "allowJs": true
  }
}
```

## 📦 生产部署

### 1. 构建生产版本

```bash
# 构建 WASM (release 模式)
cd core && wasm-pack build --target web --release --out-dir ../web/public/wasm

# 构建 Web 应用
cd ../web && pnpm build
```

### 2. 静态文件部署

将 `web/dist/` 目录部署到任何静态托管服务:
- Vercel
- Netlify
- GitHub Pages
- AWS S3 + CloudFront

### 3. WASM 文件 MIME 类型

确保服务器配置正确的 MIME 类型:
```
.wasm → application/wasm
```

**Nginx 配置示例:**
```nginx
location ~* \.wasm$ {
    types {
        application/wasm wasm;
    }
    add_header Cache-Control "public, max-age=31536000";
}
```

## 🚀 持续集成 (CI/CD)

### GitHub Actions 示例

```yaml
name: Build and Deploy

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          target: wasm32-unknown-unknown

      - name: Install wasm-pack
        run: cargo install wasm-pack

      - name: Build WASM
        run: |
          cd core
          wasm-pack build --target web --release --out-dir ../web/public/wasm

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: 18

      - name: Build Web App
        run: |
          cd web
          npm install
          npm run build

      - name: Deploy to Vercel
        run: npx vercel --prod
        env:
          VERCEL_TOKEN: ${{ secrets.VERCEL_TOKEN }}
```

## 📊 性能测试

### 基准测试

```bash
cd core

# 运行基准测试
cargo bench

# 或在浏览器中测试
# 访问 http://localhost:5173/#/benchmark
```

### 大小分析

```bash
# 查看 WASM 文件大小
ls -lh web/public/wasm/*.wasm

# 使用 twiggy 分析 WASM
cargo install twiggy
twiggy top web/public/wasm/tinyinfer_core_bg.wasm
```

## 🔗 相关资源

- [Rust and WebAssembly Book](https://rustwasm.github.io/docs/book/)
- [wasm-pack 文档](https://rustwasm.github.io/wasm-pack/)
- [WebAssembly SIMD](https://v8.dev/features/simd)
- [性能优化指南](./PERFORMANCE.md)

## 💡 开发技巧

### 快速开发循环

使用 watch 模式自动重新构建:

```bash
# 终端 1: 监听 Rust 变化
cd core
cargo watch -s "wasm-pack build --target web --dev --out-dir ../web/public/wasm"

# 终端 2: Web 开发服务器
cd web
pnpm dev
```

### 调试 WASM

在浏览器开发者工具中:
1. 打开 Sources 面板
2. 查看 WebAssembly 模块
3. 设置断点 (如果有 source maps)

### 日志调试

使用 `console_error_panic_hook` 和 `wasm-logger`:
```rust
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::default());
    log::info!("WASM module initialized");
}
```
