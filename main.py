"""
ComfyUI API Bridge - txt2img and img2img support
OpenAI compatible image generation API
"""

import os
import json
import uuid
import time
import base64
import asyncio
import aiohttp
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# ==================== CONFIG ====================
# ComfyUI connection
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = os.getenv("COMFYUI_PORT", "8188")
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

# API authentication
API_KEY = os.getenv("API_KEY", "sk-comfyui-z-image-turbo")

# Workflow templates
WORKFLOW_TXT2IMG = os.getenv("WORKFLOW_TXT2IMG", "workflow_txt2img.json")
WORKFLOW_IMG2IMG = os.getenv("WORKFLOW_IMG2IMG", "workflow_img2img.json")

# Model identifier
MODEL_NAME = "z-image-turbo"

# Default generation parameters
DEFAULT_SIZE = "1024x1024"
DEFAULT_N = 1
MAX_N = 4
DEFAULT_STRENGTH = 0.6
DEFAULT_RESPONSE_FORMAT = "b64_json"
DEFAULT_QUALITY = "standard"
DEFAULT_STYLE = "natural"

# Timeout settings
REQUEST_TIMEOUT = 30  # seconds for HTTP requests
GENERATION_TIMEOUT = 300  # seconds for image generation


def parse_size(size_str: str) -> tuple[int, int]:
    """Parse size string (e.g., '1024x1024') with fallback to default."""
    try:
        parts = size_str.lower().split("x")
        if len(parts) == 2:
            width, height = int(parts[0]), int(parts[1])
            if width > 0 and height > 0:
                return width, height
    except (ValueError, AttributeError):
        pass

    print(f"[WARN] Invalid size '{size_str}', using default {DEFAULT_SIZE}")
    return parse_size(DEFAULT_SIZE)


def clamp_n(n: int) -> int:
    """Clamp n value to valid range [1, MAX_N]."""
    if n < 1:
        print(f"[WARN] n={n} < 1, using n=1")
        return 1
    if n > MAX_N:
        print(f"[WARN] n={n} > {MAX_N}, using n={MAX_N}")
        return MAX_N
    return n


def clamp_strength(strength: float) -> float:
    """Clamp strength value to valid range [0.0, 1.0]."""
    if strength < 0.0:
        print(f"[WARN] strength={strength} < 0, using 0.0")
        return 0.0
    if strength > 1.0:
        print(f"[WARN] strength={strength} > 1, using 1.0")
        return 1.0
    return strength


# ==================== Request Models ====================
class ImageGenerationRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    n: int = Field(default=DEFAULT_N, ge=1, le=MAX_N)
    size: str = DEFAULT_SIZE
    response_format: str = DEFAULT_RESPONSE_FORMAT
    quality: str = DEFAULT_QUALITY
    style: str = DEFAULT_STYLE


class Img2ImgRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    image: str  # base64 encoded image
    n: int = Field(default=DEFAULT_N, ge=1, le=MAX_N)
    size: str = DEFAULT_SIZE
    strength: float = Field(default=DEFAULT_STRENGTH, ge=0.0, le=1.0)
    response_format: str = DEFAULT_RESPONSE_FORMAT


class ImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    max_tokens: Optional[int] = None
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


# ==================== Workflow Manager ====================
class WorkflowManager:
    def __init__(self):
        self.txt2img_template = None
        self.img2img_template = None
        self.load_workflows()

    def load_workflows(self):
        try:
            with open(WORKFLOW_TXT2IMG, "r", encoding="utf-8") as f:
                self.txt2img_template = json.load(f)
            print(f"[OK] txt2img loaded: {WORKFLOW_TXT2IMG}")
        except FileNotFoundError:
            print(f"[WARN] txt2img not found: {WORKFLOW_TXT2IMG}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] txt2img JSON error: {e}")

        try:
            with open(WORKFLOW_IMG2IMG, "r", encoding="utf-8") as f:
                self.img2img_template = json.load(f)
            print(f"[OK] img2img loaded: {WORKFLOW_IMG2IMG}")
        except FileNotFoundError:
            print(f"[WARN] img2img not found: {WORKFLOW_IMG2IMG}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] img2img JSON error: {e}")

    def find_prompt_node(self, workflow: dict) -> Optional[str]:
        for node_id, node in workflow.items():
            if node.get("class_type") == "CLIPTextEncode":
                meta = node.get("_meta", {})
                title = meta.get("title", "").lower()
                if any(kw in title for kw in ["正向", "positive", "prompt"]):
                    return node_id

        for node_id, node in workflow.items():
            if "KSampler" in node.get("class_type", ""):
                positive_input = node.get("inputs", {}).get("positive", [])
                if isinstance(positive_input, list) and len(positive_input) >= 1:
                    return str(positive_input[0])
        return None

    def find_sampler_node(self, workflow: dict) -> Optional[str]:
        for node_id, node in workflow.items():
            if "KSampler" in node.get("class_type", ""):
                return node_id
        return None

    def find_latent_node(self, workflow: dict) -> Optional[str]:
        for node_id, node in workflow.items():
            if node.get("class_type") == "EmptyLatentImage":
                return node_id
        return None

    def find_load_image_node(self, workflow: dict) -> Optional[str]:
        for node_id, node in workflow.items():
            if node.get("class_type") == "LoadImage":
                return node_id
        return None

    def prepare_txt2img(
        self, prompt: str, width: int, height: int, seed: int = None
    ) -> dict:
        if self.txt2img_template is None:
            raise ValueError("txt2img workflow not loaded")

        workflow = json.loads(json.dumps(self.txt2img_template))

        prompt_node = self.find_prompt_node(workflow)
        if prompt_node and prompt_node in workflow:
            workflow[prompt_node]["inputs"]["text"] = prompt

        latent_node = self.find_latent_node(workflow)
        if latent_node and latent_node in workflow:
            workflow[latent_node]["inputs"]["width"] = width
            workflow[latent_node]["inputs"]["height"] = height

        if seed is None:
            seed = int(time.time() * 1000) % (2**32)
        sampler_node = self.find_sampler_node(workflow)
        if sampler_node and sampler_node in workflow:
            workflow[sampler_node]["inputs"]["seed"] = seed

        return workflow

    def prepare_img2img(
        self, prompt: str, image_name: str, strength: float, seed: int = None
    ) -> dict:
        if self.img2img_template is None:
            raise ValueError("img2img workflow not loaded")

        workflow = json.loads(json.dumps(self.img2img_template))

        prompt_node = self.find_prompt_node(workflow)
        if prompt_node and prompt_node in workflow:
            workflow[prompt_node]["inputs"]["text"] = prompt

        load_image_node = self.find_load_image_node(workflow)
        if load_image_node and load_image_node in workflow:
            workflow[load_image_node]["inputs"]["image"] = image_name

        sampler_node = self.find_sampler_node(workflow)
        if sampler_node and sampler_node in workflow:
            workflow[sampler_node]["inputs"]["denoise"] = strength
            if seed is None:
                seed = int(time.time() * 1000) % (2**32)
            workflow[sampler_node]["inputs"]["seed"] = seed

        return workflow


# ==================== ComfyUI Client ====================
class ComfyUIClient:
    def __init__(self, base_url: str, session: aiohttp.ClientSession = None):
        self.base_url = base_url
        self.client_id = str(uuid.uuid4())
        self.session = session
        self._owns_session = session is None

    async def upload_image(self, image_data: bytes, filename: str) -> str:
        form_data = aiohttp.FormData()
        form_data.add_field(
            "image", image_data, filename=filename, content_type="image/png"
        )
        form_data.add_field("overwrite", "true")

        session = self.session or aiohttp.ClientSession()
        try:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            async with session.post(
                f"{self.base_url}/upload/image", data=form_data, timeout=timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=500, detail=f"Upload failed: {error_text}"
                    )

                result = await response.json()
                return result.get("name", filename)
        finally:
            if self._owns_session and session:
                await session.close()

    async def queue_prompt(self, workflow: dict) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}

        session = self.session or aiohttp.ClientSession()
        try:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            async with session.post(
                f"{self.base_url}/prompt", json=payload, timeout=timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=500,
                        detail=f"ComfyUI queue_prompt failed: {error_text}",
                    )

                result = await response.json()
                return result.get("prompt_id")
        finally:
            if self._owns_session and session:
                await session.close()

    async def wait_for_completion(
        self, prompt_id: str, timeout: int = GENERATION_TIMEOUT
    ) -> dict:
        start_time = time.time()

        session = self.session or aiohttp.ClientSession()
        try:
            req_timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            while time.time() - start_time < timeout:
                async with session.get(
                    f"{self.base_url}/history/{prompt_id}", timeout=req_timeout
                ) as response:
                    if response.status == 200:
                        history = await response.json()
                        if prompt_id in history:
                            return history[prompt_id]

                await asyncio.sleep(0.5)

            raise HTTPException(
                status_code=504, detail=f"Generation timeout after {timeout}s"
            )
        finally:
            if self._owns_session and session:
                await session.close()

    async def get_image(
        self, filename: str, subfolder: str = "", folder_type: str = "output"
    ) -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}

        session = self.session or aiohttp.ClientSession()
        try:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            async with session.get(
                f"{self.base_url}/view", params=params, timeout=timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=500, detail=f"Failed to get image: {error_text}"
                    )
                return await response.read()
        finally:
            if self._owns_session and session:
                await session.close()


# ==================== FastAPI App ====================
workflow_manager: WorkflowManager = None
comfyui_client: ComfyUIClient = None
http_session: aiohttp.ClientSession = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global workflow_manager, comfyui_client, http_session

    # Create shared HTTP session
    http_session = aiohttp.ClientSession()

    # Initialize components
    workflow_manager = WorkflowManager()
    comfyui_client = ComfyUIClient(COMFYUI_URL, session=http_session)

    # Startup banner
    print("")
    print("========================================")
    print("  ComfyUI API Bridge v2.0 (Phase 1)")
    print("========================================")
    print(f"  ComfyUI:  {COMFYUI_URL}")
    print(f"  API Key:  {API_KEY[:15]}...")
    print(
        f"  txt2img:  {WORKFLOW_TXT2IMG} [{'OK' if workflow_manager.txt2img_template else 'MISSING'}]"
    )
    print(
        f"  img2img:  {WORKFLOW_IMG2IMG} [{'OK' if workflow_manager.img2img_template else 'MISSING'}]"
    )
    print(
        f"  Defaults: size={DEFAULT_SIZE}, n={DEFAULT_N}, strength={DEFAULT_STRENGTH}"
    )
    print("========================================")

    yield

    # Cleanup
    print("[INFO] Shutting down API Bridge...")
    if http_session:
        await http_session.close()
    print("[INFO] API Bridge stopped")


app = FastAPI(
    title="ComfyUI API Bridge",
    description="OpenAI compatible API with txt2img and img2img",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = authorization.replace("Bearer ", "").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return token


# ==================== API Endpoints ====================


@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    verify_api_key(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def generate_images(
    request: ImageGenerationRequest, authorization: str = Header(None)
):
    """Text to image (txt2img) - OpenAI compatible"""
    verify_api_key(authorization)

    if workflow_manager.txt2img_template is None:
        raise HTTPException(status_code=500, detail="txt2img workflow not configured")

    # Parse and validate parameters
    width, height = parse_size(request.size)
    n = clamp_n(request.n)

    print(f"[txt2img] prompt='{request.prompt[:50]}...', size={width}x{height}, n={n}")

    generated_images = []

    for i in range(n):
        seed = int(time.time() * 1000 + i) % (2**32)
        workflow = workflow_manager.prepare_txt2img(
            prompt=request.prompt, width=width, height=height, seed=seed
        )

        prompt_id = await comfyui_client.queue_prompt(workflow)
        print(f"[txt2img] Task {i + 1}/{n}: prompt_id={prompt_id}, seed={seed}")

        result = await comfyui_client.wait_for_completion(prompt_id)

        outputs = result.get("outputs", {})
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    image_data = await comfyui_client.get_image(
                        filename=img_info["filename"],
                        subfolder=img_info.get("subfolder", ""),
                        folder_type=img_info.get("type", "output"),
                    )
                    b64_data = base64.b64encode(image_data).decode("utf-8")
                    generated_images.append(
                        ImageData(b64_json=b64_data, revised_prompt=request.prompt)
                    )
                    break
            if generated_images:
                break

    if not generated_images:
        raise HTTPException(status_code=500, detail="Failed to generate image")

    return ImageGenerationResponse(created=int(time.time()), data=generated_images)


@app.post("/v1/images/edits", response_model=ImageGenerationResponse)
async def edit_images(request: Img2ImgRequest, authorization: str = Header(None)):
    """Image to image (img2img) - OpenAI compatible"""
    verify_api_key(authorization)

    if workflow_manager.img2img_template is None:
        raise HTTPException(status_code=500, detail="img2img workflow not configured")

    # Validate image data
    try:
        image_data = base64.b64decode(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    # Parse and validate parameters
    n = clamp_n(request.n)
    strength = clamp_strength(request.strength)

    print(f"[img2img] prompt='{request.prompt[:50]}...', n={n}, strength={strength}")

    generated_images = []

    for i in range(n):
        filename = f"input_{int(time.time())}_{i}.png"
        uploaded_name = await comfyui_client.upload_image(image_data, filename)
        print(f"[img2img] Task {i + 1}/{n}: uploaded={uploaded_name}")

        seed = int(time.time() * 1000 + i) % (2**32)
        workflow = workflow_manager.prepare_img2img(
            prompt=request.prompt,
            image_name=uploaded_name,
            strength=strength,
            seed=seed,
        )

        prompt_id = await comfyui_client.queue_prompt(workflow)
        print(f"[img2img] Task {i + 1}/{n}: prompt_id={prompt_id}, seed={seed}")

        result = await comfyui_client.wait_for_completion(prompt_id)

        outputs = result.get("outputs", {})
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    result_image = await comfyui_client.get_image(
                        filename=img_info["filename"],
                        subfolder=img_info.get("subfolder", ""),
                        folder_type=img_info.get("type", "output"),
                    )
                    b64_data = base64.b64encode(result_image).decode("utf-8")
                    generated_images.append(
                        ImageData(b64_json=b64_data, revised_prompt=request.prompt)
                    )
                    break
            if len(generated_images) > i:
                break

    if not generated_images:
        raise HTTPException(status_code=500, detail="Failed to generate image")

    return ImageGenerationResponse(created=int(time.time()), data=generated_images)


@app.post("/v1/images/img2img", response_model=ImageGenerationResponse)
async def img2img_form(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    strength: float = Form(DEFAULT_STRENGTH),
    size: str = Form(DEFAULT_SIZE),
    n: int = Form(DEFAULT_N),
    authorization: str = Header(None),
):
    """Image to image with form upload"""
    verify_api_key(authorization)

    if workflow_manager.img2img_template is None:
        raise HTTPException(status_code=500, detail="img2img workflow not configured")

    # Validate parameters
    n = clamp_n(n)
    strength = clamp_strength(strength)

    print(f"[img2img-form] prompt='{prompt[:50]}...', n={n}, strength={strength}")

    image_data = await image.read()
    generated_images = []

    for i in range(n):
        filename = f"input_{int(time.time())}_{i}.png"
        uploaded_name = await comfyui_client.upload_image(image_data, filename)
        print(f"[img2img-form] Task {i + 1}/{n}: uploaded={uploaded_name}")

        seed = int(time.time() * 1000 + i) % (2**32)
        workflow = workflow_manager.prepare_img2img(
            prompt=prompt, image_name=uploaded_name, strength=strength, seed=seed
        )

        prompt_id = await comfyui_client.queue_prompt(workflow)
        print(f"[img2img-form] Task {i + 1}/{n}: prompt_id={prompt_id}, seed={seed}")

        result = await comfyui_client.wait_for_completion(prompt_id)

        outputs = result.get("outputs", {})
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    result_image = await comfyui_client.get_image(
                        filename=img_info["filename"],
                        subfolder=img_info.get("subfolder", ""),
                        folder_type=img_info.get("type", "output"),
                    )
                    b64_data = base64.b64encode(result_image).decode("utf-8")
                    generated_images.append(
                        ImageData(b64_json=b64_data, revised_prompt=prompt)
                    )
                    break
            if len(generated_images) > i:
                break

    if not generated_images:
        raise HTTPException(status_code=500, detail="Failed to generate image")

    return ImageGenerationResponse(created=int(time.time()), data=generated_images)


@app.get("/health")
async def health_check():
    """Health check endpoint with ComfyUI connectivity and workflow status"""
    comfyui_reachable = False
    comfyui_error = None

    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with http_session.get(
            f"{COMFYUI_URL}/system_stats", timeout=timeout
        ) as response:
            comfyui_reachable = response.status == 200
            if not comfyui_reachable:
                comfyui_error = f"HTTP {response.status}"
    except Exception as e:
        comfyui_error = str(e)

    txt2img_loaded = workflow_manager.txt2img_template is not None
    img2img_loaded = workflow_manager.img2img_template is not None

    overall_status = "healthy" if (comfyui_reachable and txt2img_loaded) else "degraded"

    response = {
        "status": overall_status,
        "comfyui": {"reachable": comfyui_reachable, "url": COMFYUI_URL},
        "workflows": {"txt2img": txt2img_loaded, "img2img": img2img_loaded},
    }

    if comfyui_error:
        response["comfyui"]["error"] = comfyui_error

    return response


@app.post("/reload")
async def reload_workflows(authorization: str = Header(None)):
    """Reload workflow templates from disk"""
    verify_api_key(authorization)

    print("[INFO] Reloading workflow templates...")
    workflow_manager.load_workflows()

    txt2img_loaded = workflow_manager.txt2img_template is not None
    img2img_loaded = workflow_manager.img2img_template is not None

    return {
        "status": "ok",
        "workflows": {"txt2img": txt2img_loaded, "img2img": img2img_loaded},
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest, authorization: str = Header(None)
):
    """Chat completions endpoint (placeholder for future implementation)"""
    verify_api_key(authorization)

    # Phase 1: Return placeholder response
    raise HTTPException(
        status_code=501,
        detail="Chat completions with image generation will be implemented in a future phase. Currently only /v1/images/* endpoints are supported.",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
