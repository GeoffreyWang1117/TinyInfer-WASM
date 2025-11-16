#!/bin/bash

set -e

echo "🦀 Building TinyInfer-WASM..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack is not installed"
    echo "Please install it with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Build the WASM module
echo "📦 Building WASM module..."
cd core

# Build with SIMD support
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build \
    --target web \
    --out-dir ../web/src/wasm \
    --release

cd ..

echo "✅ Build complete!"
echo "📊 WASM file size:"
ls -lh web/src/wasm/tinyinfer_core_bg.wasm

echo ""
echo "🚀 To start the development server:"
echo "   cd web && npm install && npm run dev"
