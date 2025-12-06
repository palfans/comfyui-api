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
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# ==================== CONFIG ====================
# ComfyUI connection
# Support both COMFYUI_URL (full URL) or COMFYUI_HOST+COMFYUI_PORT
COMFYUI_URL = os.getenv("COMFYUI_URL")
if not COMFYUI_URL:
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

# Supported image sizes (Extended to support up to 2048x2048)
SUPPORTED_SIZES = [
    "512x512",
    "512x768",
    "768x512",
    "576x1024",
    "1024x576",
    "768x768",
    "768x1024",
    "1024x768",
    "768x1344",
    "1344x768",
    "1024x1024",
    "1024x1536",
    "1536x1024",
    "1152x1536",
    "1536x1152",
    "1536x1536",
    "1024x2048",
    "2048x1024",
    "1536x2048",
    "2048x1536",
    "2048x2048",
]

# Timeout settings
REQUEST_TIMEOUT = 30  # seconds for HTTP requests
GENERATION_TIMEOUT = 300  # seconds for image generation

# Concurrent generation limits
MAX_CONCURRENT_GENERATIONS = 4


def parse_size(size_str: str, strict: bool = False) -> tuple[int, int]:
    """
    Parse size string (e.g., '1024x1024').

    Args:
        size_str: Size string in format 'WIDTHxHEIGHT'
        strict: If True, raise ValueError for unsupported sizes. If False, fallback to default.

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If strict=True and size is not supported
    """
    try:
        parts = size_str.lower().split("x")
        if len(parts) != 2:
            raise ValueError("Invalid format")

        width, height = int(parts[0]), int(parts[1])
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")

        size_normalized = f"{width}x{height}"
        if size_normalized in SUPPORTED_SIZES:
            return width, height

        # Size not in supported list
        if strict:
            raise ValueError(
                f"Size '{size_normalized}' is not supported. "
                f"Supported sizes: {', '.join(SUPPORTED_SIZES)}"
            )
        else:
            print(
                f"[WARN] Size '{size_normalized}' not in supported sizes, using default {DEFAULT_SIZE}"
            )
            default_width, default_height = DEFAULT_SIZE.split("x")
            return int(default_width), int(default_height)

    except ValueError as e:
        if strict:
            # Check if error message already contains our custom message
            if "not supported" in str(e):
                raise
            raise ValueError(
                f"Invalid size format '{size_str}'. Expected format: 'WIDTHxHEIGHT' (e.g., '1024x1024')"
            )

        print(f"[WARN] Invalid size '{size_str}', using default {DEFAULT_SIZE}")
        default_width, default_height = DEFAULT_SIZE.split("x")
        return int(default_width), int(default_height)


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


def estimate_tokens(text: str) -> int:
    """Estimate token count based on character length (simple approximation)."""
    import math

    return math.ceil(len(text) / 4)


# ==================== Request Models ====================
class ImageGenerationRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    n: int = Field(default=DEFAULT_N, ge=1, le=MAX_N)
    size: str = DEFAULT_SIZE
    response_format: str = DEFAULT_RESPONSE_FORMAT
    quality: str = DEFAULT_QUALITY
    style: str = DEFAULT_STYLE
    seed: Optional[int] = None  # Custom seed for reproducible generation


class Img2ImgRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    image: str  # base64 encoded image
    n: int = Field(default=DEFAULT_N, ge=1, le=MAX_N)
    size: str = DEFAULT_SIZE
    strength: float = Field(default=DEFAULT_STRENGTH, ge=0.0, le=1.0)
    response_format: str = DEFAULT_RESPONSE_FORMAT


class ImageData(BaseModel):
    b64_json: str
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict]  # Support both string and content list


class ChatCompletionRequest(BaseModel):
    # OpenAI standard parameters
    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    max_tokens: Optional[int] = None
    stream: bool = False

    # Extension parameters (not part of OpenAI standard)
    size: Optional[str] = (
        None  # Image generation size (extension, defaults to 1024x1024)
    )
    n: Optional[int] = None  # Number of images to generate (extension, defaults to 1)
    seed: Optional[int] = None  # Custom seed for reproducible generation (extension)


class ChatCompletionChoice(BaseModel):
    index: int
    message: dict
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str | list[dict]] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Optional[ChatCompletionUsage] = None


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
        # Handle ComfyUI UI workflow format (with nodes array)
        if "nodes" in workflow:
            nodes = workflow["nodes"]
            for node in nodes:
                if node.get("type") == "CLIPTextEncode":
                    return str(node["id"])
            return None

        # Handle API workflow format (dict of nodes)
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
        # Handle ComfyUI UI workflow format
        if "nodes" in workflow:
            nodes = workflow["nodes"]
            for node in nodes:
                node_type = node.get("type")
                if node_type in ["EmptyLatentImage", "EmptySD3LatentImage"]:
                    return str(node["id"])
            return None

        # Handle API workflow format
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

        if seed is None:
            seed = int(time.time() * 1000) % (2**32)

        # Handle ComfyUI UI workflow format (with nodes array)
        if "nodes" in workflow:
            nodes = workflow["nodes"]
            for node in nodes:
                node_type = node.get("type")

                # Update prompt
                if node_type == "CLIPTextEncode":
                    if "widgets_values" in node:
                        node["widgets_values"] = [prompt]

                # Update size
                elif node_type in ["EmptyLatentImage", "EmptySD3LatentImage"]:
                    if "widgets_values" in node:
                        node["widgets_values"] = [width, height, 1]

                # Update seed
                elif "KSampler" in node_type:
                    if "widgets_values" in node and len(node["widgets_values"]) > 0:
                        node["widgets_values"][0] = seed
            return workflow

        # Handle API workflow format (dict of nodes)
        prompt_node = self.find_prompt_node(workflow)
        if prompt_node and prompt_node in workflow:
            workflow[prompt_node]["inputs"]["text"] = prompt

        latent_node = self.find_latent_node(workflow)
        if latent_node and latent_node in workflow:
            workflow[latent_node]["inputs"]["width"] = width
            workflow[latent_node]["inputs"]["height"] = height

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
        # Convert ComfyUI UI workflow format to API format if needed
        api_workflow = workflow
        if "nodes" in workflow:
            api_workflow = self._convert_ui_to_api_format(workflow)

        payload = {"prompt": api_workflow, "client_id": self.client_id}

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

    def _convert_ui_to_api_format(self, ui_workflow: dict) -> dict:
        """Convert ComfyUI UI workflow format to API format."""
        api_workflow = {}
        nodes = ui_workflow.get("nodes", [])
        links_array = ui_workflow.get("links", [])

        # Skip non-executable node types
        skip_node_types = ["Note", "MarkdownNote", "PrimitiveNode"]

        # Build links lookup: link_id -> [source_node_id, source_output_index]
        links_lookup = {}
        for link in links_array:
            if len(link) >= 5:
                link_id, source_node, source_slot, target_node, target_slot = link[:5]
                links_lookup[link_id] = [source_node, source_slot]

        # Convert each node
        for node in nodes:
            node_id = str(node["id"])
            node_type = node.get("type")

            # Skip non-executable nodes
            if node_type in skip_node_types:
                continue

            # Initialize API node
            api_workflow[node_id] = {"class_type": node_type, "inputs": {}}

            # Process inputs
            for inp in node.get("inputs", []):
                input_name = inp.get("name")
                link_id = inp.get("link")

                if link_id is not None and link_id in links_lookup:
                    # Input from another node
                    source_node_id, source_output_index = links_lookup[link_id]
                    api_workflow[node_id]["inputs"][input_name] = [
                        str(source_node_id),
                        source_output_index,
                    ]

            # Add widget values as inputs based on node type
            widgets_values = node.get("widgets_values", [])
            if node_type == "CLIPTextEncode" and len(widgets_values) > 0:
                api_workflow[node_id]["inputs"]["text"] = widgets_values[0]
            elif (
                node_type in ["EmptyLatentImage", "EmptySD3LatentImage"]
                and len(widgets_values) >= 3
            ):
                api_workflow[node_id]["inputs"]["width"] = widgets_values[0]
                api_workflow[node_id]["inputs"]["height"] = widgets_values[1]
                api_workflow[node_id]["inputs"]["batch_size"] = widgets_values[2]
            elif "KSampler" in node_type and len(widgets_values) >= 7:
                # KSampler widgets: [seed, control_after_generate, steps, cfg, sampler_name, scheduler, denoise]
                api_workflow[node_id]["inputs"]["seed"] = widgets_values[0]
                api_workflow[node_id]["inputs"]["steps"] = widgets_values[2]
                api_workflow[node_id]["inputs"]["cfg"] = widgets_values[3]
                api_workflow[node_id]["inputs"]["sampler_name"] = widgets_values[4]
                api_workflow[node_id]["inputs"]["scheduler"] = widgets_values[5]
                api_workflow[node_id]["inputs"]["denoise"] = widgets_values[6]
            elif node_type == "VAELoader" and len(widgets_values) > 0:
                api_workflow[node_id]["inputs"]["vae_name"] = widgets_values[0]
            elif node_type == "UNETLoader" and len(widgets_values) > 0:
                api_workflow[node_id]["inputs"]["unet_name"] = widgets_values[0]
                if len(widgets_values) >= 2:
                    api_workflow[node_id]["inputs"]["weight_dtype"] = widgets_values[1]
            elif node_type == "CLIPLoaderGGUF" and len(widgets_values) >= 2:
                # CLIPLoaderGGUF widgets: [clip_name, type]
                api_workflow[node_id]["inputs"]["clip_name"] = widgets_values[0]
                api_workflow[node_id]["inputs"]["type"] = widgets_values[1]
            elif node_type == "ModelSamplingAuraFlow" and len(widgets_values) >= 1:
                # ModelSamplingAuraFlow widgets: [shift]
                api_workflow[node_id]["inputs"]["shift"] = widgets_values[0]
            elif node_type == "SaveImage" and len(widgets_values) > 0:
                api_workflow[node_id]["inputs"]["filename_prefix"] = widgets_values[0]

        return api_workflow

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
    print("  ComfyUI API Bridge v2.0 (Phase 3)")
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
    print(f"  Sizes: {', '.join(SUPPORTED_SIZES)}")
    print(f"  Max Concurrent: {MAX_CONCURRENT_GENERATIONS}")
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

    # Validate response_format
    if request.response_format != "b64_json":
        raise HTTPException(
            status_code=400,
            detail=f"Only 'b64_json' response_format is supported. Got: {request.response_format}",
        )

    # Parse and validate parameters
    try:
        width, height = parse_size(request.size, strict=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    n = clamp_n(request.n)

    print(f"[txt2img] prompt='{request.prompt[:50]}...', size={width}x{height}, n={n}")

    start_time = time.time()

    # Define async generation function for concurrent execution
    async def generate_single_image(index: int):
        # Use custom seed if provided, otherwise generate random seed
        if request.seed is not None:
            seed = request.seed
        else:
            seed = int(time.time() * 1000 + index) % (2**32)

        workflow = workflow_manager.prepare_txt2img(
            prompt=request.prompt, width=width, height=height, seed=seed
        )

        prompt_id = await comfyui_client.queue_prompt(workflow)
        print(f"[txt2img] Task {index + 1}/{n}: prompt_id={prompt_id}, seed={seed}")

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
                    return ImageData(b64_json=b64_data, revised_prompt=request.prompt)
        return None

    # Execute generations concurrently with limit
    if n <= MAX_CONCURRENT_GENERATIONS:
        # Generate all concurrently
        tasks = [generate_single_image(i) for i in range(n)]
        generated_images = await asyncio.gather(*tasks)
    else:
        # Generate in batches to respect concurrency limit
        generated_images = []
        for i in range(0, n, MAX_CONCURRENT_GENERATIONS):
            batch_size = min(MAX_CONCURRENT_GENERATIONS, n - i)
            tasks = [generate_single_image(i + j) for j in range(batch_size)]
            batch_results = await asyncio.gather(*tasks)
            generated_images.extend(batch_results)

    # Filter out None values
    generated_images = [img for img in generated_images if img is not None]

    elapsed = time.time() - start_time
    print(f"[txt2img] Completed {len(generated_images)}/{n} images in {elapsed:.2f}s")

    if not generated_images:
        raise HTTPException(status_code=500, detail="Failed to generate image")

    return ImageGenerationResponse(created=int(time.time()), data=generated_images)


@app.post("/v1/images/edits", response_model=ImageGenerationResponse)
async def edit_images(request: Img2ImgRequest, authorization: str = Header(None)):
    """Image to image (img2img) - OpenAI compatible [PLACEHOLDER]"""
    verify_api_key(authorization)

    # img2img backend model not yet available
    raise HTTPException(
        status_code=501,
        detail="Image-to-image (img2img) functionality is not yet implemented. Backend model is not available.",
    )


@app.post("/v1/images/img2img", response_model=ImageGenerationResponse)
async def img2img_form(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    strength: float = Form(DEFAULT_STRENGTH),
    size: str = Form(DEFAULT_SIZE),
    n: int = Form(DEFAULT_N),
    authorization: str = Header(None),
):
    """Image to image with form upload [PLACEHOLDER]"""
    verify_api_key(authorization)

    # img2img backend model not yet available
    raise HTTPException(
        status_code=501,
        detail="Image-to-image (img2img) functionality is not yet implemented. Backend model is not available.",
    )


@app.get("/")
async def root():
    """Serve the web UI"""
    return FileResponse("index.html")


@app.get("/index.html")
async def index():
    """Serve the web UI (alternative route)"""
    return FileResponse("index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint with ComfyUI connectivity and workflow status"""
    # Check if application is still initializing
    if http_session is None or workflow_manager is None:
        return {
            "status": "initializing",
            "message": "Application is starting up, please retry in a moment",
        }

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


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, authorization: str = Header(None)
):
    """Chat completions endpoint - generates images based on conversation

    Supports standard OpenAI parameters plus optional extension parameters:
    - size (optional): Image dimensions, defaults to 1024x1024
    - n (optional): Number of images, defaults to 1, must be 1 for chat completions
    - stream (optional): Enable streaming response
    """
    verify_api_key(authorization)

    # Apply defaults for extension parameters
    size = request.size or DEFAULT_SIZE
    n = request.n or DEFAULT_N

    # Chat completions only supports single image generation
    if n != 1:
        raise HTTPException(
            status_code=400,
            detail="Chat completions only supports n=1. For multiple images, use /v1/images/generations.",
        )

    if workflow_manager.txt2img_template is None:
        raise HTTPException(status_code=500, detail="txt2img workflow not configured")

    # Validate messages
    if not request.messages or len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")

    # Extract prompt from messages - concatenate all user and system messages
    prompt_parts = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            if msg.role in ["user", "system"]:
                prompt_parts.append(msg.content)
        elif isinstance(msg.content, list):
            # Handle content as list (multimodal format)
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    prompt_parts.append(item.get("text", ""))

    if not prompt_parts:
        raise HTTPException(status_code=400, detail="No text content found in messages")

    prompt = " ".join(prompt_parts).strip()

    # Parse and validate parameters
    try:
        width, height = parse_size(size, strict=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    n = clamp_n(n)

    print(
        f"[chat/completions] prompt='{prompt[:50]}...', size={width}x{height}, n={n}, stream={request.stream}"
    )

    # Generate image using txt2img
    async def generate_image():
        generated_images = []

        for i in range(n):
            # Use custom seed if provided, otherwise generate random seed
            if request.seed is not None:
                seed = request.seed
            else:
                seed = int(time.time() * 1000 + i) % (2**32)

            workflow = workflow_manager.prepare_txt2img(
                prompt=prompt, width=width, height=height, seed=seed
            )

            prompt_id = await comfyui_client.queue_prompt(workflow)
            print(
                f"[chat/completions] Task {i + 1}/{n}: prompt_id={prompt_id}, seed={seed}"
            )

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
                        generated_images.append(b64_data)
                        break
                if generated_images:
                    break

        if not generated_images:
            raise HTTPException(status_code=500, detail="Failed to generate image")

        return generated_images[0]

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Handle streaming response
    if request.stream:

        async def stream_generator():
            try:
                # Send initial chunk with role
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(role="assistant"),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

                # Send status message
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(
                                content="Generating image..."
                            ),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

                # Generate image
                b64_image = await generate_image()

                # Send image as markdown with base64 data URL
                # This format works with most chat clients
                image_markdown = (
                    f"\n\n![Generated Image](data:image/png;base64,{b64_image})"
                )

                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content=image_markdown),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

                # Send final chunk with finish_reason
                prompt_tokens = estimate_tokens(prompt)
                completion_tokens = 100  # Estimate per image

                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(),
                            finish_reason="stop",
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

                # Send [DONE] marker
                yield "data: [DONE]\n\n"

            except Exception as e:
                print(f"[ERROR] Stream error: {e}")
                import traceback

                traceback.print_exc()
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "internal_error",
                        "code": "internal_error",
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Handle non-streaming response
    b64_image = await generate_image()

    # Construct message content with image and text
    message_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
        },
        {"type": "text", "text": f"Generated image for: {prompt}"},
    ]

    choice = ChatCompletionChoice(
        index=0,
        message={"role": "assistant", "content": message_content},
        finish_reason="stop",
    )

    # Calculate token usage (simple estimation)
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = 100  # Estimate per image

    usage = ChatCompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[choice],
        usage=usage,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
