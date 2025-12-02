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
| `/v1/images/edits` | POST | Image to image | ✅ |
| `/v1/images/img2img` | POST | Image to image (form) | ✅ |
| `/v1/models` | GET | List models | ✅ |
| `/v1/chat/completions` | POST | Chat with image gen | 🚧 Phase 2 |
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

### Image to Image
```bash
curl -X POST http://localhost:8000/v1/images/edits \
  -H "Authorization: Bearer sk-comfyui-z-image-turbo" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "anime style", "image": "<base64>", "strength": 0.6}'
```

### Health Check
```bash
curl http://localhost:8000/health
```
## Configuration

Set environment variables to customize behavior:

```bash
export API_KEY="your-secret-key"
export COMFYUI_HOST="127.0.0.1"
export COMFYUI_PORT="8188"
export WORKFLOW_TXT2IMG="workflow_txt2img.json"
export WORKFLOW_IMG2IMG="workflow_img2img.json"
```

