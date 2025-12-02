# ComfyUI API Bridge

OpenAI compatible API for ComfyUI with txt2img and img2img support.

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) installed
- ComfyUI running on `http://127.0.0.1:8188`

### Option 1: Docker (Recommended)

```bash
# Pull and run
docker run -d \
  --name comfyui-api \
  -p 8000:8000 \
  -e COMFYUI_URL=http://host.docker.internal:8188 \
  palfans/comfyui-api:latest

# Or use docker-compose
docker-compose up -d
```

📖 See [DOCKER.md](DOCKER.md) for detailed Docker deployment guide.

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/palfans/comfyui-api.git
cd comfyui-api

# Install dependencies
uv sync

# Start server
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### Using with uv

```bash
# Development mode with auto-reload
uv run uvicorn main:app --reload

# Production mode
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

| Endpoint | Method | Description | Status |
|----------|--------|-------------|--------|
| `/v1/images/generations` | POST | Text to image (OpenAI compatible) | ✅ |
| `/v1/images/edits` | POST | Image to image | 🚧 Placeholder |
| `/v1/images/img2img` | POST | Image to image (form) | 🚧 Placeholder |
| `/v1/models` | GET | List models (OpenAI compatible) | ✅ |
| `/v1/chat/completions` | POST | Chat with image gen (Extended) | ✅ |
| `/health` | GET | Health check | ✅ |
| `/reload` | POST | Reload workflows | ✅ |

## API Compatibility

### OpenAI Standard Endpoints
- **`/v1/images/generations`**: Fully compatible with OpenAI Images API
  - Standard parameters: `prompt`, `model`, `n`, `size`, `response_format`, `quality`, `style`
  
- **`/v1/models`**: Fully compatible with OpenAI Models API

### Extended Endpoints
- **`/v1/chat/completions`**: OpenAI Chat Completions API with image generation extensions
  - **Standard parameters** (OpenAI compatible): `model`, `messages`, `temperature`, `max_tokens`, `stream`
  - **Extension parameters** (optional, for image generation):
    - `size` (string, optional): Image size, defaults to `1024x1024`
    - `n` (integer, optional): Number of images, defaults to `1`, max `1` for chat completions
  - **Streaming**: Fully supported with Server-Sent Events (SSE) for real-time responses
  - **Note**: Standard OpenAI clients can use this endpoint by omitting extension parameters

## Basic Usage

### Text to Image
```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Authorization: Bearer sk-comfyui-z-image-turbo" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cute cat", "size": "1024x1024", "n": 1}'
```

### Chat Completions (Image Generation)

**With extension parameters:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-comfyui-z-image-turbo" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "z-image-turbo",
    "messages": [
      {"role": "user", "content": "a beautiful sunset over mountains"}
    ],
    "size": "1024x1024",
    "n": 1
  }'
```

**Standard OpenAI client (without extensions):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-comfyui-z-image-turbo" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "z-image-turbo",
    "messages": [
      {"role": "user", "content": "a beautiful sunset over mountains"}
    ]
  }'
```
*Note: Omitting `size` and `n` uses defaults (1024x1024, n=1)*

**Streaming response:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-comfyui-z-image-turbo" \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "z-image-turbo",
    "messages": [
      {"role": "user", "content": "画一只小猫"}
    ],
    "stream": true
  }'
```
*Note: Use `-N` flag with curl to disable buffering for streaming responses*

### Health Check
```bash
curl http://localhost:8000/health
```

## Supported Sizes

Z-Image-Turbo supports resolutions from 512px to 1536px:

**Small (Fast generation)**
- `512x512` (Square)
- `512x768`, `768x512` (Portrait/Landscape)

**Medium (Balanced)**
- `576x1024`, `1024x576` (9:16 / 16:9)
- `768x768` (Square)
- `768x1024`, `1024x768` (3:4 / 4:3)
- `768x1344`, `1344x768` (Portrait/Landscape)
- `1024x1024` (Square - Standard)

**Large (Best quality)**
- `1024x1536`, `1536x1024` (2:3 / 3:2)
- `1152x1536`, `1536x1152` (3:4 / 4:3)
- `1536x1536` (Square - Maximum)

## Features

### Token Counting ✅
Chat completions return estimated token usage based on prompt length.

### Concurrent Generation ✅
Multiple images (n > 1) are generated concurrently (max 4 at a time) for better performance.

### Streaming Support ✅
Chat completions support streaming responses via Server-Sent Events (SSE):
- Enable streaming with `stream: true` in the request
- Compatible with OpenAI SDK and other clients that support SSE
- Receive real-time updates during image generation
- Works seamlessly with tools like Cherry Studio, Continue.dev, etc.

## Current Limitations

- **Response Format**: Only `b64_json` is supported for image responses.
- **Image-to-Image**: `/v1/images/edits` and `/v1/images/img2img` are placeholder endpoints (returns 501). Backend model not yet available.
- **Chat Completions**: 
  - Only supports `n=1` for single image generation (extension parameter)
  - For multiple images, use `/v1/images/generations` instead
  - `size` and `n` are extension parameters, not part of OpenAI standard
- **Token Accuracy**: Token counting is a simple estimation (characters / 4), not exact.

## WebUI

A web interface is available at `index.html`. Simply open it in your browser:

```bash
# Start the API server
uv run uvicorn main:app --host 0.0.0.0 --port 8000

# Open index.html in your browser
open index.html  # macOS
xdg-open index.html  # Linux
start index.html  # Windows
```

Features:
- **Text to Image**: Generate 1-4 images with various sizes
- **Chat Completion**: Generate single image through chat interface
- **Gallery**: View, download, and manage generated images
- **Connection Test**: Verify API connectivity and health

## Configuration

Set environment variables to customize behavior:

```bash
export API_KEY="your-secret-key"
export COMFYUI_HOST="127.0.0.1"
export COMFYUI_PORT="8188"
export WORKFLOW_TXT2IMG="workflow_txt2img.json"
export WORKFLOW_IMG2IMG="workflow_img2img.json"
```

