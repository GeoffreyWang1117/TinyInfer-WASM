# TinyInfer-WASM Docker Image
FROM rust:1.70 as rust-builder

# Install wasm-pack
RUN cargo install wasm-pack

# Add wasm target
RUN rustup target add wasm32-unknown-unknown

WORKDIR /app

# Copy Rust source
COPY core ./core

# Build WASM
WORKDIR /app/core
RUN wasm-pack build --target web --release --out-dir ../web/public/wasm

# Node.js build stage
FROM node:18 as web-builder

WORKDIR /app

# Copy web source and WASM output
COPY web ./web
COPY --from=rust-builder /app/web/public/wasm ./web/public/wasm

# Install dependencies and build
WORKDIR /app/web
RUN npm install
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files to nginx
COPY --from=web-builder /app/web/dist /usr/share/nginx/html

# Copy nginx configuration
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
