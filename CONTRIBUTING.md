# 贡献指南

感谢你对 TinyInfer-WASM 的兴趣！我们欢迎所有形式的贡献。

## 如何贡献

### 报告 Bug

如果你发现了 bug，请创建一个 Issue 并包含以下信息：

1. **Bug 描述**: 简洁地描述问题
2. **复现步骤**: 详细的复现步骤
3. **预期行为**: 你期望发生什么
4. **实际行为**: 实际发生了什么
5. **环境信息**:
   - 浏览器和版本
   - 操作系统
   - TinyInfer-WASM 版本

### 提出新功能

我们欢迎新功能建议！请创建一个 Issue 并说明：

1. **功能描述**: 你希望添加什么功能
2. **使用场景**: 这个功能解决什么问题
3. **实现建议**: （可选）你认为如何实现

### 提交代码

1. **Fork 仓库**

```bash
git clone https://github.com/YOUR_USERNAME/TinyInfer-WASM.git
cd TinyInfer-WASM
```

2. **创建分支**

```bash
git checkout -b feature/your-feature-name
```

3. **编写代码**

- 遵循现有代码风格
- 添加必要的注释
- 编写单元测试
- 确保所有测试通过

4. **提交更改**

```bash
git add .
git commit -m "feat: add your feature description"
```

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

- `feat:` 新功能
- `fix:` Bug 修复
- `docs:` 文档更新
- `style:` 代码格式（不影响功能）
- `refactor:` 重构
- `test:` 测试相关
- `chore:` 构建/工具相关

5. **推送到 GitHub**

```bash
git push origin feature/your-feature-name
```

6. **创建 Pull Request**

在 GitHub 上创建 PR，描述你的更改。

## 开发指南

### 环境搭建

```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装 wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# 安装前端依赖
cd web && npm install
```

### 项目结构

```
TinyInfer-WASM/
├── core/               # Rust 核心引擎
│   ├── src/
│   │   ├── tensor/    # Tensor 实现
│   │   ├── ops/       # 算子库
│   │   ├── engine/    # 执行引擎
│   │   └── lib.rs     # WASM 绑定
│   └── Cargo.toml
├── web/               # React 前端
│   ├── src/
│   │   ├── pages/     # 页面组件
│   │   └── App.tsx
│   └── package.json
└── docs/              # 文档
```

### 运行测试

#### Rust 测试

```bash
cd core
cargo test
```

#### 前端测试

```bash
cd web
npm test
```

### 代码风格

#### Rust

使用 `rustfmt`:

```bash
cargo fmt
```

使用 `clippy` 检查:

```bash
cargo clippy
```

#### TypeScript

使用 ESLint:

```bash
npm run lint
```

### 添加新算子

1. 在 `core/src/ops/` 创建新文件
2. 实现 `Operator` trait
3. 添加单元测试
4. 在 `ops/mod.rs` 中导出
5. 在 `engine/runtime.rs` 中注册

示例:

```rust
// core/src/ops/my_op.rs
use crate::error::Result;
use crate::ops::Operator;
use crate::tensor::Tensor;

pub struct MyOp;

impl MyOp {
    pub fn new() -> Self {
        Self
    }
}

impl Operator for MyOp {
    fn name(&self) -> &str {
        "MyOp"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        // 实现算子逻辑
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // 添加测试
}
```

### 性能优化建议

1. **使用 SIMD**: 对于元素级运算，使用 WASM SIMD 指令
2. **循环分块**: 对于矩阵运算，使用分块提高缓存命中
3. **In-place 操作**: 尽可能避免内存分配
4. **Benchmark**: 添加性能测试确保优化有效

### 文档

- 所有公共 API 都应该有文档注释
- 复杂算法应该有详细说明
- 更新 README 和 ARCHITECTURE.md

## 代码审查

所有 PR 都需要经过代码审查。审查者会关注：

- 代码质量和可读性
- 测试覆盖率
- 性能影响
- 文档完整性
- 安全性

## 行为准则

我们致力于提供一个友好、安全和欢迎的环境。请：

- 尊重他人
- 接受建设性批评
- 专注于对项目最有利的事情
- 对社区成员表示同理心

## 许可证

提交代码即表示你同意你的贡献在 MIT 许可证下发布。

## 问题?

如有任何问题，请创建 Issue 或发送邮件至项目维护者。

再次感谢你的贡献！
