# Docker 部署指南

## 镜像构建

### 使用构建脚本 (推荐)

```bash
# 本地测试构建 (当前平台)
./build.sh local [version]

# 多平台构建并推送到 Registry
./build.sh push [version]
```

示例:
```bash
# 本地构建 latest 版本
./build.sh local

# 本地构建 2.0.0 版本
./build.sh local 2.0.0

# 多平台构建并推送 (需要 Docker Hub 登录)
docker login
./build.sh push 2.0.0
```

### 手动构建

```bash
# 单平台本地构建
docker build -t palfans/comfyui-api:2.0.0 .

# 多平台构建 (需要 buildx)
docker buildx build --platform linux/arm64,linux/amd64 \
  -t palfans/comfyui-api:2.0.0 \
  --push .
```

## 运行方式

### 方式 1: Docker Run

```bash
docker run -d \
  --name comfyui-api \
  -p 8000:8000 \
  -e COMFYUI_URL=http://host.docker.internal:8188 \
  palfans/comfyui-api:2.0.0
```

### 方式 2: Docker Compose (推荐)

```bash
# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI 后端地址 |
| `HOST` | `0.0.0.0` | API 监听地址 |
| `PORT` | `8000` | API 监听端口 |
| `PYTHONUNBUFFERED` | `1` | Python 输出缓冲 |

## ComfyUI 后端连接

### 场景 1: ComfyUI 在本机运行

```bash
# Linux/macOS
docker run -d -p 8000:8000 \
  -e COMFYUI_URL=http://172.17.0.1:8188 \
  palfans/comfyui-api:2.0.0

# Windows/macOS (Docker Desktop)
docker run -d -p 8000:8000 \
  -e COMFYUI_URL=http://host.docker.internal:8188 \
  palfans/comfyui-api:2.0.0
```

### 场景 2: ComfyUI 在另一台服务器

```bash
docker run -d -p 8000:8000 \
  -e COMFYUI_URL=http://192.168.1.100:8188 \
  palfans/comfyui-api:2.0.0
```

### 场景 3: 使用 Docker 网络连接

```bash
# 创建网络
docker network create comfyui-network

# 启动 ComfyUI (假设有 ComfyUI 镜像)
docker run -d --name comfyui \
  --network comfyui-network \
  -p 8188:8188 \
  comfyui/comfyui:latest

# 启动 API (通过服务名连接)
docker run -d --name comfyui-api \
  --network comfyui-network \
  -p 8000:8000 \
  -e COMFYUI_URL=http://comfyui:8188 \
  palfans/comfyui-api:2.0.0
```

## 健康检查

容器内置健康检查,每 30 秒检查一次:

```bash
# 查看容器健康状态
docker inspect --format='{{.State.Health.Status}}' comfyui-api

# 手动健康检查
docker exec comfyui-api curl -f http://localhost:8000/health
```

健康状态:
- `healthy`: ComfyUI 可达,所有 workflow 加载成功
- `degraded`: 部分功能不可用 (如 ComfyUI 无法连接)
- `unhealthy`: 服务不可用

## 镜像信息

- **Base Image**: `python:3.12-slim`
- **Package Manager**: `uv` (modern Python package manager)
- **Image Size**: ~203MB
- **Architectures**: `linux/amd64`, `linux/arm64`
- **Python Version**: 3.12
- **Dependencies**:
  - FastAPI >= 0.104.0
  - uvicorn >= 0.24.0
  - aiohttp >= 3.9.0
  - pydantic >= 2.0.0
  - python-multipart >= 0.0.6

## 代理设置

构建脚本自动检测并使用系统代理设置:

```bash
# 设置代理环境变量
export HTTP_PROXY=http://172.17.0.1:1080
export HTTPS_PROXY=http://172.17.0.1:1080
export NO_PROXY=localhost,127.0.0.1

# 构建 (自动使用代理)
./build.sh local
```

脚本会自动将 `host.wsl` 和 `host.docker.internal` 转换为 Docker 网桥地址 `172.17.0.1`。

## 故障排查

### 1. 容器启动失败

```bash
# 查看日志
docker logs comfyui-api

# 检查端口占用
netstat -tuln | grep 8000
```

### 2. ComfyUI 连接失败

```bash
# 检查 ComfyUI 是否运行
curl http://127.0.0.1:8188/system_stats

# 从容器内测试连接
docker exec comfyui-api curl http://172.17.0.1:8188/system_stats

# 检查防火墙
sudo ufw status
```

### 3. 构建失败

```bash
# 清理 buildx 缓存
docker buildx prune -a

# 使用详细日志
docker buildx build --progress=plain -t test .
```

## 多阶段构建说明

Dockerfile 使用多阶段构建优化镜像大小:

1. **Builder Stage**: 使用 uv 安装依赖到虚拟环境
2. **Runtime Stage**: 仅复制必要文件和依赖,减少最终镜像大小

这样做的好处:
- 更小的镜像 (203MB vs 可能的 500MB+)
- 更快的部署速度
- 更好的安全性 (不包含构建工具)

## Registry 推送

推送到 Docker Hub:

```bash
# 登录
docker login

# 使用构建脚本推送
./build.sh push 2.0.0

# 或手动推送
docker push palfans/comfyui-api:2.0.0
docker push palfans/comfyui-api:latest
```

推送到私有 Registry:

```bash
# 重新标记
docker tag palfans/comfyui-api:2.0.0 registry.example.com/comfyui-api:2.0.0

# 推送
docker push registry.example.com/comfyui-api:2.0.0
```

## 更新 README 说明

本项目现已支持 Docker 部署,详细使用方法请参阅本文档。
