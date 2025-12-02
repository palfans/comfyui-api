# Multi-stage build for ComfyUI API
# Stage 1: Base image with uv
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files (README.md is required by pyproject.toml)
COPY pyproject.toml uv.lock* README.md ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Stage 2: Runtime image
FROM python:3.12-slim

# Install curl for health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy uv from builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application files
COPY main.py ./
COPY workflow_txt2img.json ./
COPY index.html ./

# Expose API port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    COMFYUI_URL=http://127.0.0.1:8188 \
    PORT=8000 \
    HOST=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application with uv
CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
