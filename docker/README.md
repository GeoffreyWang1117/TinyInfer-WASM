# Docker Deployment Guide for TinyInfer-WASM

## 🐳 Quick Start

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t tinyinfer-wasm .

# Run the container
docker run -d -p 8080:80 --name tinyinfer tinyinfer-wasm

# Access the application
open http://localhost:8080
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📋 Configuration

### Environment Variables

You can configure the application using environment variables:

```bash
docker run -d \
  -p 8080:80 \
  -e NODE_ENV=production \
  --name tinyinfer \
  tinyinfer-wasm
```

### Volume Mounts

Mount custom models or configurations:

```bash
docker run -d \
  -p 8080:80 \
  -v $(pwd)/models:/usr/share/nginx/html/models:ro \
  --name tinyinfer \
  tinyinfer-wasm
```

## 🔧 Customization

### Custom Nginx Configuration

Edit `docker/nginx.conf` to customize the web server:

```nginx
# Add custom headers, caching rules, etc.
```

### Multi-stage Build

The Dockerfile uses multi-stage builds to keep the final image small:

1. **rust-builder**: Compiles Rust to WASM
2. **web-builder**: Builds the web application
3. **nginx:alpine**: Serves the static files (~50MB final image)

## 🚀 Production Deployment

### With SSL/TLS

Use a reverse proxy like Traefik or nginx-proxy:

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  tinyinfer:
    build: .
    expose:
      - "80"
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.tinyinfer.rule=Host(`tinyinfer.example.com`)"
      - "traefik.http.routers.tinyinfer.tls=true"
      - "traefik.http.routers.tinyinfer.tls.certresolver=letsencrypt"

  traefik:
    image: traefik:v2.10
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - "/var/run/docker.sock:/var/run/docker.sock:ro"
      - "./traefik.yml:/etc/traefik/traefik.yml:ro"
      - "./acme.json:/acme.json"
```

### Health Checks

The container includes a health check endpoint:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' tinyinfer

# Manual health check
curl http://localhost:8080/health
```

### Resource Limits

Limit resource usage:

```bash
docker run -d \
  -p 8080:80 \
  --memory="512m" \
  --cpus="1.0" \
  --name tinyinfer \
  tinyinfer-wasm
```

## 🔍 Monitoring

### View Logs

```bash
# Follow logs
docker logs -f tinyinfer

# Last 100 lines
docker logs --tail 100 tinyinfer
```

### Container Stats

```bash
# Real-time stats
docker stats tinyinfer
```

## 🛠️ Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs tinyinfer

# Inspect container
docker inspect tinyinfer
```

### WASM Files Not Loading

Ensure WASM MIME type is set correctly in nginx.conf:

```nginx
types {
    application/wasm wasm;
}
```

### Permission Issues

```bash
# Fix permissions
docker exec tinyinfer chmod -R 755 /usr/share/nginx/html
```

## 📊 Performance Tuning

### Nginx Worker Processes

Edit `docker/nginx.conf`:

```nginx
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 4096;
    use epoll;
}
```

### Compression

Gzip compression is enabled by default. Adjust in nginx.conf:

```nginx
gzip_comp_level 6;
gzip_min_length 1024;
```

## 🔄 Updates

### Rebuild and Restart

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

### Zero-Downtime Updates

Use Docker Swarm or Kubernetes for rolling updates.

## 🌐 Cloud Deployment

### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t tinyinfer-wasm .
docker tag tinyinfer-wasm:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/tinyinfer-wasm:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/tinyinfer-wasm:latest
```

### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/<project-id>/tinyinfer-wasm
gcloud run deploy tinyinfer --image gcr.io/<project-id>/tinyinfer-wasm --platform managed
```

### Azure Container Instances

```bash
# Create container group
az container create \
  --resource-group myResourceGroup \
  --name tinyinfer \
  --image <registry>/tinyinfer-wasm:latest \
  --dns-name-label tinyinfer-unique \
  --ports 80
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [nginx Documentation](https://nginx.org/en/docs/)
- [Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)
