#!/bin/bash
set -e

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 [local|push] [version]"
    echo "  local   - Local test build (only build current platform and load locally)"
    echo "  push    - Multi-platform build and push to registry"
    echo "  version - Optional version tag (e.g., 2.0.0), defaults to latest"
    exit 1
fi

MODE=$1
VERSION="${2:-latest}"

echo "Build mode: $MODE"
echo "Build version: $VERSION"

# Setup docker buildx
if ! docker buildx inspect mybuilder > /dev/null 2>&1; then
  echo "Creating buildx builder instance..."
  docker buildx create --name mybuilder --use
else
  docker buildx use mybuilder
fi

echo "Starting image build..."

# Setup proxy build args if proxy environment variables are set
PROXY_ARGS=""
if [ -n "$HTTP_PROXY" ] || [ -n "$http_proxy" ]; then
    PROXY_VAL=${HTTP_PROXY:-$http_proxy}
    
    # Handle WSL proxy issues - replace host.wsl with Docker bridge gateway
    if [[ "$PROXY_VAL" == *"host.wsl"* ]] || [[ "$PROXY_VAL" == *"host.docker.internal"* ]]; then
        echo "Converting proxy host to Docker bridge gateway..."
        DOCKER_GATEWAY=$(docker network inspect bridge --format='{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || echo "172.17.0.1")
        PROXY_VAL=$(echo "$PROXY_VAL" | sed "s/host\.wsl\|host\.docker\.internal/$DOCKER_GATEWAY/g")
        echo "Updated proxy: $PROXY_VAL"
    fi
    
    PROXY_ARGS="$PROXY_ARGS --build-arg HTTP_PROXY=$PROXY_VAL --build-arg http_proxy=$PROXY_VAL"
fi
if [ -n "$HTTPS_PROXY" ] || [ -n "$https_proxy" ]; then
    PROXY_VAL=${HTTPS_PROXY:-$https_proxy}
    
    if [[ "$PROXY_VAL" == *"host.wsl"* ]] || [[ "$PROXY_VAL" == *"host.docker.internal"* ]]; then
        echo "Converting proxy host to Docker bridge gateway..."
        DOCKER_GATEWAY=$(docker network inspect bridge --format='{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || echo "172.17.0.1")
        PROXY_VAL=$(echo "$PROXY_VAL" | sed "s/host\.wsl\|host\.docker\.internal/$DOCKER_GATEWAY/g")
        echo "Updated proxy: $PROXY_VAL"
    fi
    
    PROXY_ARGS="$PROXY_ARGS --build-arg HTTPS_PROXY=$PROXY_VAL --build-arg https_proxy=$PROXY_VAL"
fi
if [ -n "$NO_PROXY" ] || [ -n "$no_proxy" ]; then
    PROXY_VAL=${NO_PROXY:-$no_proxy}
    PROXY_ARGS="$PROXY_ARGS --build-arg NO_PROXY=$PROXY_VAL --build-arg no_proxy=$PROXY_VAL"
fi

if [ -n "$PROXY_ARGS" ]; then
    echo "Detected proxy settings, using proxy for build..."
fi

if [ "$MODE" = "local" ]; then
    echo "=== Local Test Build Mode ==="
    # Local test build, build for current platform and load locally
    PLATFORM="linux/$(uname -m | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/')"
    echo "Build platform: $PLATFORM"
    
    # Build image
    docker buildx build --platform $PLATFORM \
      -t palfans/comfyui-api:latest \
      -t palfans/comfyui-api:$VERSION \
      $PROXY_ARGS \
      --load .
    
    echo "✅ Local image build complete!"
    echo "   Tags:"
    echo "   - palfans/comfyui-api:latest"
    echo "   - palfans/comfyui-api:$VERSION"
    echo ""
    echo "   You can test with:"
    echo "   docker run --rm -p 8000:8000 -e COMFYUI_URL=http://host.docker.internal:8188 palfans/comfyui-api:latest"
    echo ""
    echo "   Or use docker-compose:"
    echo "   docker-compose up"

elif [ "$MODE" = "push" ]; then
    echo "=== Multi-platform Build and Push Mode ==="
    
    # Build and push multi-platform images
    echo "Building and pushing images to registry..."
    docker buildx build --platform linux/arm64,linux/amd64 \
      -t palfans/comfyui-api:latest \
      -t palfans/comfyui-api:$VERSION \
      $PROXY_ARGS \
      --push .
    
    echo "✅ Multi-platform image build and push complete!"
    echo "   Pushed tags:"
    echo "   - palfans/comfyui-api:latest"
    echo "   - palfans/comfyui-api:$VERSION"
    echo ""
    echo "   Pull with:"
    echo "   docker pull palfans/comfyui-api:$VERSION"

else
    echo "Error: Unknown build mode '$MODE'"
    echo "Supported modes: local, push"
    exit 1
fi

echo "Build operation complete!"
