# ComfyUI API Bridge

OpenAI compatible API for ComfyUI with txt2img and img2img support.

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) installed
- ComfyUI running on `http://127.0.0.1:8188`

### Installation

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
| `/v1/images/generations` | POST | Text to image | ✅ |
| `/v1/images/edits` | POST | Image to image | 🚧 Placeholder |
| `/v1/images/img2img` | POST | Image to image (form) | 🚧 Placeholder |
| `/v1/models` | GET | List models | ✅ |
| `/v1/chat/completions` | POST | Chat with image gen | ✅ (n=1 only) |
| `/health` | GET | Health check | ✅ |
| `/reload` | POST | Reload workflows | ✅ |

## Basic Usage

### Text to Image
```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Authorization: Bearer sk-comfyui-z-image-turbo" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cute cat", "size": "1024x1024", "n": 1}'
```

### Chat Completions (Image Generation)
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

### Health Check
```bash
curl http://localhost:8000/health
```

## Supported Sizes

- `1024x1024` (square)
- `1024x1792` (portrait)
- `1792x1024` (landscape)

## Features

### Token Counting ✅
Chat completions return estimated token usage based on prompt length.

### Concurrent Generation ✅
Multiple images (n > 1) are generated concurrently (max 4 at a time) for better performance.

## Current Limitations

- **Response Format**: Only `b64_json` is supported for image responses.
- **Image-to-Image**: `/v1/images/edits` and `/v1/images/img2img` are placeholder endpoints (returns 501). Backend model not yet available.
- **Chat Completions**: Only supports `n=1` for single image generation. For multiple images, use `/v1/images/generations`.
- **Streaming**: Streaming responses are not yet supported (returns 400 if stream=true).
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

