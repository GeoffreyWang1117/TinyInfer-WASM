# TinyInfer-WASM 部署指南

本文档说明如何将 TinyInfer-WASM 应用部署到各种生产环境。

## 目录

- [构建生产版本](#构建生产版本)
- [静态网站托管](#静态网站托管)
- [CDN 部署](#cdn-部署)
- [Docker 部署](#docker-部署)
- [性能优化建议](#性能优化建议)
- [安全配置](#安全配置)
- [监控和日志](#监控和日志)

---

## 构建生产版本

### 1. 构建 WASM 模块

```bash
cd core

# 启用所有优化
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build \
  --target web \
  --release \
  --out-dir ../web/public/wasm

cd ..
```

**输出文件:**
- `tinyinfer_core_bg.wasm` (~113KB)
- `tinyinfer_core.js`
- `tinyinfer_core.d.ts`
- `package.json`

### 2. 优化 WASM (可选)

```bash
# 安装 binaryen
npm install -g binaryen

# 优化 WASM 二进制
wasm-opt -Oz --enable-simd \
  web/public/wasm/tinyinfer_core_bg.wasm \
  -o web/public/wasm/tinyinfer_core_bg.wasm
```

**优化效果:**
- 减小大小 20-30%
- 略微提升性能 5-10%

### 3. 构建前端应用

```bash
cd web

# 安装依赖
npm install

# 生产构建
npm run build
```

**输出目录:** `web/dist/`

**构建产物:**
```
dist/
├── assets/
│   ├── index-[hash].js      # 主应用 JS
│   ├── vendor-[hash].js     # 第三方库
│   └── index-[hash].css     # 样式
├── wasm/
│   ├── tinyinfer_core_bg.wasm
│   ├── tinyinfer_core.js
│   └── tinyinfer_core.d.ts
└── index.html
```

### 4. 验证构建

```bash
# 本地测试生产构建
cd web
npm run preview

# 访问 http://localhost:4173
```

---

## 静态网站托管

### Vercel 部署

#### 方法 1: CLI 部署

```bash
# 安装 Vercel CLI
npm install -g vercel

# 登录
vercel login

# 部署
cd web
vercel --prod
```

#### 方法 2: Git 集成

1. 推送代码到 GitHub
2. 在 [Vercel](https://vercel.com) 导入项目
3. 配置构建设置：

**vercel.json:**
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "framework": "vite",
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ],
  "headers": [
    {
      "source": "/wasm/(.*)",
      "headers": [
        {
          "key": "Content-Type",
          "value": "application/wasm"
        },
        {
          "key": "Cache-Control",
          "value": "public, max-age=31536000, immutable"
        }
      ]
    }
  ]
}
```

**优势:**
- ✅ 自动 HTTPS
- ✅ 全球 CDN
- ✅ Git 推送自动部署
- ✅ 预览部署

### Netlify 部署

#### netlify.toml

```toml
[build]
  command = "npm run build"
  publish = "dist"

[[headers]]
  for = "/wasm/*"
  [headers.values]
    Content-Type = "application/wasm"
    Cache-Control = "public, max-age=31536000, immutable"

[[headers]]
  for = "/*.js"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

**部署步骤:**
1. 在 [Netlify](https://netlify.com) 连接 Git 仓库
2. 设置构建命令: `npm run build`
3. 设置发布目录: `dist`
4. 点击 Deploy

### GitHub Pages 部署

#### .github/workflows/deploy.yml

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: 18
          cache: 'npm'
          cache-dependency-path: web/package-lock.json

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
          RUSTFLAGS='-C target-feature=+simd128' wasm-pack build \
            --target web \
            --release \
            --out-dir ../web/public/wasm

      - name: Build Web
        run: |
          cd web
          npm ci
          npm run build

      - name: Setup Pages
        uses: actions/configure-pages@v3

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v2
        with:
          path: 'web/dist'

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v2
```

**配置:**
1. 在仓库设置中启用 GitHub Pages
2. 选择 "GitHub Actions" 作为源
3. 推送到 main 分支触发部署

### AWS S3 + CloudFront

#### 部署脚本

```bash
#!/bin/bash

# 构建
npm run build

# 上传到 S3
aws s3 sync dist/ s3://your-bucket-name/ \
  --delete \
  --cache-control "public, max-age=31536000, immutable"

# 设置 WASM MIME 类型
aws s3 cp s3://your-bucket-name/wasm/ s3://your-bucket-name/wasm/ \
  --recursive \
  --content-type "application/wasm" \
  --metadata-directive REPLACE

# 刷新 CloudFront 缓存
aws cloudfront create-invalidation \
  --distribution-id YOUR_DISTRIBUTION_ID \
  --paths "/*"
```

---

## CDN 部署

### WASM 文件 CDN

对于大型应用，可以将 WASM 文件单独部署到 CDN：

```typescript
// 修改 web/src/lib/tinyinfer.ts
const WASM_CDN_URL = 'https://cdn.example.com/tinyinfer/v1.0.0/'

export async function initWasm(): Promise<WasmModule> {
  // 从 CDN 加载
  await init(`${WASM_CDN_URL}tinyinfer_core_bg.wasm`)
  // ...
}
```

**CDN 配置:**
- 启用 Brotli/Gzip 压缩
- 设置正确的 MIME 类型
- 配置 CORS 头
- 设置长期缓存

### CDN 配置示例 (CloudFlare)

```javascript
// workers-site/index.js
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  const url = new URL(request.url)

  // WASM 文件特殊处理
  if (url.pathname.endsWith('.wasm')) {
    const response = await fetch(request)
    const headers = new Headers(response.headers)
    headers.set('Content-Type', 'application/wasm')
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp')
    headers.set('Cross-Origin-Opener-Policy', 'same-origin')

    return new Response(response.body, {
      status: response.status,
      headers
    })
  }

  return fetch(request)
}
```

---

## Docker 部署

### Dockerfile

```dockerfile
# 多阶段构建

# 阶段 1: 构建 WASM
FROM rust:1.70 AS wasm-builder

RUN cargo install wasm-pack
RUN rustup target add wasm32-unknown-unknown

WORKDIR /app
COPY core ./core

WORKDIR /app/core
RUN RUSTFLAGS='-C target-feature=+simd128' wasm-pack build \
    --target web \
    --release \
    --out-dir ../web/public/wasm

# 阶段 2: 构建 Web
FROM node:18 AS web-builder

WORKDIR /app
COPY web/package*.json ./web/
COPY web ./web
COPY --from=wasm-builder /app/web/public/wasm ./web/public/wasm

WORKDIR /app/web
RUN npm ci
RUN npm run build

# 阶段 3: 生产镜像
FROM nginx:alpine

# 复制构建产物
COPY --from=web-builder /app/web/dist /usr/share/nginx/html

# 复制 Nginx 配置
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### nginx.conf

```nginx
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # 启用 gzip 压缩
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/wasm;
    gzip_min_length 1000;

    # WASM 文件特殊配置
    location ~* \.wasm$ {
        types {
            application/wasm wasm;
        }
        add_header Cache-Control "public, max-age=31536000, immutable";
        add_header Cross-Origin-Embedder-Policy "require-corp";
        add_header Cross-Origin-Opener-Policy "same-origin";
    }

    # JavaScript 和 CSS 缓存
    location ~* \.(js|css)$ {
        add_header Cache-Control "public, max-age=31536000, immutable";
    }

    # HTML 文件不缓存
    location ~* \.html$ {
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }

    # SPA 路由支持
    location / {
        try_files $uri $uri/ /index.html;
    }

    # 安全头
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  tinyinfer-web:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:80"
    environment:
      - NODE_ENV=production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost/"]
      interval: 30s
      timeout: 3s
      retries: 3
```

**构建和运行:**

```bash
# 构建镜像
docker build -t tinyinfer-wasm:latest .

# 运行容器
docker run -d -p 8080:80 --name tinyinfer tinyinfer-wasm:latest

# 或使用 docker-compose
docker-compose up -d
```

---

## 性能优化建议

### 1. 启用 HTTP/2

HTTP/2 允许多路复用，加速资源加载。

**Nginx 配置:**
```nginx
server {
    listen 443 ssl http2;
    # ...
}
```

### 2. 启用 Brotli 压缩

Brotli 比 Gzip 压缩率更高。

**Nginx 配置:**
```nginx
brotli on;
brotli_comp_level 6;
brotli_types text/plain text/css application/javascript application/json application/wasm;
```

**压缩效果:**
- WASM: 113KB → 35KB (69% 减少)
- JS: 250KB → 70KB (72% 减少)

### 3. 预加载关键资源

```html
<!-- index.html -->
<head>
  <!-- 预加载 WASM -->
  <link rel="modulepreload" href="/wasm/tinyinfer_core.js">
  <link rel="preload" href="/wasm/tinyinfer_core_bg.wasm" as="fetch" crossorigin>

  <!-- 预连接 CDN -->
  <link rel="preconnect" href="https://cdn.example.com">
</head>
```

### 4. Service Worker 缓存

```javascript
// sw.js
const CACHE_NAME = 'tinyinfer-v1'
const WASM_CACHE = 'tinyinfer-wasm-v1'

const WASM_FILES = [
  '/wasm/tinyinfer_core_bg.wasm',
  '/wasm/tinyinfer_core.js'
]

// 安装时缓存 WASM 文件
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(WASM_CACHE).then(cache => cache.addAll(WASM_FILES))
  )
})

// 网络优先策略
self.addEventListener('fetch', event => {
  if (event.request.url.includes('/wasm/')) {
    event.respondWith(
      caches.match(event.request).then(response => {
        return response || fetch(event.request).then(fetchResponse => {
          return caches.open(WASM_CACHE).then(cache => {
            cache.put(event.request, fetchResponse.clone())
            return fetchResponse
          })
        })
      })
    )
  }
})
```

### 5. 资源分离

将 WASM 文件部署到单独的子域名，启用并行下载：

```
https://app.example.com        # 主应用
https://wasm.example.com      # WASM 文件
https://cdn.example.com       # 静态资源
```

---

## 安全配置

### 1. HTTPS 强制

```nginx
server {
    listen 80;
    server_name example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # SSL 配置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # ...
}
```

### 2. CSP (内容安全策略)

```nginx
add_header Content-Security-Policy "
    default-src 'self';
    script-src 'self' 'wasm-unsafe-eval';
    style-src 'self' 'unsafe-inline';
    img-src 'self' data: blob:;
    connect-src 'self' https://api.example.com;
    worker-src 'self' blob:;
" always;
```

### 3. SharedArrayBuffer 支持

如果使用 Web Workers 和 SharedArrayBuffer:

```nginx
add_header Cross-Origin-Embedder-Policy "require-corp" always;
add_header Cross-Origin-Opener-Policy "same-origin" always;
```

### 4. 限流

```nginx
# 限制请求速率
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

location /api/ {
    limit_req zone=api burst=20 nodelay;
}
```

---

## 监控和日志

### 1. 应用监控

#### Sentry 集成

```typescript
// web/src/main.tsx
import * as Sentry from '@sentry/react'

Sentry.init({
  dsn: 'YOUR_SENTRY_DSN',
  environment: import.meta.env.MODE,
  tracesSampleRate: 1.0,
  integrations: [
    new Sentry.BrowserTracing(),
  ],
})
```

### 2. 性能监控

```typescript
// web/src/lib/analytics.ts
export function trackPerformance(metricName: string, value: number) {
  // Google Analytics
  if (window.gtag) {
    window.gtag('event', 'timing_complete', {
      name: metricName,
      value: Math.round(value),
      event_category: 'WASM Performance',
    })
  }

  // 发送到自定义后端
  fetch('/api/metrics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      metric: metricName,
      value,
      timestamp: Date.now(),
    }),
  }).catch(console.error)
}
```

### 3. Nginx 访问日志

```nginx
log_format custom '$remote_addr - $remote_user [$time_local] '
                  '"$request" $status $body_bytes_sent '
                  '"$http_referer" "$http_user_agent" '
                  '$request_time $upstream_response_time';

access_log /var/log/nginx/access.log custom;
```

### 4. 健康检查

```typescript
// web/public/health
export async function healthCheck() {
  return {
    status: 'ok',
    timestamp: new Date().toISOString(),
    wasm: {
      loaded: !!wasmModule,
      simd: wasmModule?.checkSimdSupport() || false,
    },
  }
}
```

```nginx
location /health {
    access_log off;
    return 200 "OK";
    add_header Content-Type text/plain;
}
```

---

## 部署清单

**构建:**
- [ ] WASM 生产构建 (`--release`)
- [ ] 启用 SIMD
- [ ] 运行 wasm-opt
- [ ] 前端生产构建
- [ ] 测试生产构建

**服务器配置:**
- [ ] 正确的 MIME 类型 (application/wasm)
- [ ] HTTPS 启用
- [ ] HTTP/2 启用
- [ ] Gzip/Brotli 压缩
- [ ] 缓存头配置
- [ ] CORS 配置（如需要）

**安全:**
- [ ] CSP 头
- [ ] HTTPS 强制
- [ ] 安全头 (X-Frame-Options, etc.)
- [ ] 限流配置

**监控:**
- [ ] 错误追踪 (Sentry)
- [ ] 性能监控
- [ ] 访问日志
- [ ] 健康检查端点

---

## 相关文档

- [构建指南](../BUILD.md)
- [性能优化](./PERFORMANCE.md)
- [API 参考](./API.md)
- [FAQ](./FAQ.md)
