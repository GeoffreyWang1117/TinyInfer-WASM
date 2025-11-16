#!/bin/bash

# TinyInfer-WASM 自动化构建脚本
# 用途：一键构建 WASM 模块和前端应用

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << HELP
TinyInfer-WASM 构建脚本

用法:
    ./build.sh [选项]

选项:
    -h, --help          显示此帮助信息
    -d, --dev           开发模式构建 (更快，文件更大)
    -r, --release       生产模式构建 (默认)
    -c, --clean         清理构建产物
    -t, --test          运行测试
    --skip-wasm         跳过 WASM 构建
    --skip-web          跳过 Web 构建

示例:
    ./build.sh                  # 生产构建
    ./build.sh -d               # 开发构建
    ./build.sh -c               # 清理
    ./build.sh -t               # 运行测试

HELP
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."

    if ! command -v rustc &> /dev/null; then
        print_error "Rust 未安装。请访问 https://rustup.rs/"
        exit 1
    fi

    if ! command -v wasm-pack &> /dev/null; then
        print_warning "wasm-pack 未安装，正在安装..."
        cargo install wasm-pack
    fi

    if ! rustup target list | grep -q "wasm32-unknown-unknown (installed)"; then
        print_warning "wasm32 target 未安装，正在安装..."
        rustup target add wasm32-unknown-unknown
    fi

    if ! command -v node &> /dev/null; then
        print_error "Node.js 未安装。请访问 https://nodejs.org/"
        exit 1
    fi

    print_success "所有依赖已就绪"
}

# 清理构建产物
clean() {
    print_info "清理构建产物..."
    cd core && cargo clean && cd ..
    rm -rf web/public/wasm web/dist
    print_success "清理完成"
}

# 构建 WASM
build_wasm() {
    local mode=$1
    print_info "构建 WASM 模块 (${mode} 模式)..."
    cd core
    if [ "$mode" = "dev" ]; then
        wasm-pack build --target web --dev --out-dir ../web/public/wasm
    else
        wasm-pack build --target web --release --out-dir ../web/public/wasm
    fi
    cd ..
    local size=$(du -h web/public/wasm/tinyinfer_core_bg.wasm 2>/dev/null | cut -f1 || echo "未知")
    print_success "WASM 构建完成 (${size})"
}

# 构建 Web
build_web() {
    print_info "构建 Web 应用..."
    cd web
    [ ! -d "node_modules" ] && npm install
    npm run build
    cd ..
    print_success "Web 构建完成"
}

# 运行测试
run_tests() {
    print_info "运行 Rust 测试..."
    cd core && cargo test --lib && cd ..
    print_success "测试完成"
}

# 主函数
main() {
    local mode="release"
    local skip_wasm=false
    local skip_web=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help) show_help; exit 0 ;;
            -d|--dev) mode="dev"; shift ;;
            -r|--release) mode="release"; shift ;;
            -c|--clean) clean; exit 0 ;;
            -t|--test) run_tests; exit 0 ;;
            --skip-wasm) skip_wasm=true; shift ;;
            --skip-web) skip_web=true; shift ;;
            *) print_error "未知选项: $1"; show_help; exit 1 ;;
        esac
    done

    print_info "=========================================="
    print_info "   TinyInfer-WASM 构建系统"
    print_info "=========================================="

    check_dependencies
    [ "$skip_wasm" = false ] && build_wasm "$mode"
    [ "$skip_web" = false ] && build_web

    print_success "=========================================="
    print_success "   构建完成！"
    print_success "=========================================="
    print_info "运行: cd web && npm run dev"
}

main "$@"
