"""
FastAPI inference server for PelagicGPT.

Loads fine-tuned checkpoint and serves text generation via REST API.

Usage:
  uvicorn serve:app --port 8000 --reload

Endpoints:
  GET  /health   — Model status
  POST /generate — Text generation
"""

import os
import sys
import time
import asyncio
import logging
from contextlib import asynccontextmanager

import torch
import tiktoken
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from model import GPTConfig, GPT

# API key authentication
API_KEY = os.environ.get('API_KEY', '')
api_key_header = APIKeyHeader(name='X-API-Key', auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify the API key from the X-API-Key header."""
    if not API_KEY:
        # If no API_KEY is configured, skip auth (development mode)
        return api_key
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=401, detail='Invalid or missing API key')
    return api_key

# Force unbuffered output for Railway logs
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("pelagic-gpt")

# Configuration
CHECKPOINT_DIR = os.environ.get('PELAGIC_CHECKPOINT_DIR', 'out-pelagic')
CHECKPOINT_URL = os.environ.get('CHECKPOINT_URL', '')
DEVICE = os.environ.get('PELAGIC_DEVICE', 'cpu')
MAX_TOKENS_LIMIT = 500

# Global state
model = None
enc = None
checkpoint_info = {}


TRUSTED_DOMAINS = {'huggingface.co', 'github.com', 'railway.app', 'cdn.openai.com'}


def _validate_checkpoint_url(url: str):
    """Validate that URL is HTTPS and from a trusted domain to prevent SSRF."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme != 'https':
        raise ValueError(f"Only HTTPS URLs are allowed, got: {parsed.scheme}")
    hostname = parsed.hostname or ''
    if not any(hostname == domain or hostname.endswith('.' + domain) for domain in TRUSTED_DOMAINS):
        raise ValueError(
            f"Untrusted domain: {hostname}. Allowed: {', '.join(sorted(TRUSTED_DOMAINS))}"
        )


def download_checkpoint(url: str, dest: str):
    """Download checkpoint from URL if not already present."""
    if os.path.exists(dest):
        logger.info(f"Checkpoint already exists at {dest}, skipping download.")
        return

    _validate_checkpoint_url(url)

    import urllib.request
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info(f"Downloading checkpoint from {url}")
    logger.info("This may take a few minutes for a 1.4GB file...")
    sys.stdout.flush()

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            if block_num % 2000 == 0:
                logger.info(f"  {mb:.0f}/{total_mb:.0f} MB ({pct}%)")
                sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    logger.info(f"Download complete: {os.path.getsize(dest) / (1024*1024):.0f} MB")


def load_model():
    """Load model from checkpoint. Downloads from URL if needed."""
    global model, enc, checkpoint_info

    ckpt_path = os.path.join(CHECKPOINT_DIR, 'ckpt.pt')

    # Download from URL if checkpoint doesn't exist locally
    if not os.path.exists(ckpt_path) and CHECKPOINT_URL:
        # Use the filename from the URL, but save as ckpt.pt
        download_checkpoint(CHECKPOINT_URL, ckpt_path)

    # Also check for ckpt_slim.pt as fallback
    if not os.path.exists(ckpt_path):
        slim_path = os.path.join(CHECKPOINT_DIR, 'ckpt_slim.pt')
        if os.path.exists(slim_path):
            ckpt_path = slim_path

    if not os.path.exists(ckpt_path):
        logger.warning(f"No checkpoint at {ckpt_path}. Set CHECKPOINT_URL env var to download.")
        logger.warning(f"CHECKPOINT_URL is currently: '{CHECKPOINT_URL}'")
        return

    logger.info(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    gptconf = GPTConfig(**checkpoint['model_args'])
    model_instance = GPT(gptconf)

    state_dict = checkpoint['model']
    # Remove torch.compile prefix if present
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model_instance.load_state_dict(state_dict)
    model_instance.eval()
    model_instance.to(DEVICE)
    model = model_instance

    enc = tiktoken.get_encoding("gpt2")

    checkpoint_info = {
        'iter_num': checkpoint.get('iter_num', 0),
        'best_val_loss': round(float(checkpoint.get('best_val_loss', 0)), 4),
        'n_params': model.get_num_params(),
        'n_layer': gptconf.n_layer,
        'n_head': gptconf.n_head,
        'n_embd': gptconf.n_embd,
        'block_size': gptconf.block_size,
        'checkpoint_path': ckpt_path,
        'checkpoint_size_mb': round(os.path.getsize(ckpt_path) / (1024 * 1024), 1),
    }

    logger.info(f"Model loaded: {checkpoint_info['n_params']:,} params, "
               f"val_loss={checkpoint_info['best_val_loss']}, "
               f"iter={checkpoint_info['iter_num']}")


import threading

model_loading = False

def load_model_background():
    """Load model in background thread so server starts immediately."""
    global model_loading
    model_loading = True
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    finally:
        model_loading = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start model loading in background so health checks pass immediately
    thread = threading.Thread(target=load_model_background, daemon=True)
    thread.start()
    yield


app = FastAPI(
    title="PelagicGPT",
    description="Blue economy domain language model",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "https://pelagic.bluenova.vc",
        "https://blue-economy-data-bala-kamallakharans-projects.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Request / Response Models ===

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field(
        default='freeform',
        description='Generation mode: market_commentary, signal_narrative, price_analysis, vessel_intelligence, freeform'
    )
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_k: int = Field(default=200, ge=1, le=500)
    max_tokens: int = Field(default=200, ge=10, le=500)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)
    source_filter: str | None = Field(
        default=None,
        description='Filter by source: financial, company, news, signal, quota, annual_report'
    )
    max_tokens: int = Field(default=1000, ge=100, le=4000)


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    chunks_used: int
    model: str
    query_time_ms: int


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    generation_time_ms: int
    mode: str
    model_info: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    rag_available: bool


# === Mode Tags ===

MODE_TAGS = {
    'market_commentary': '[MARKET COMMENTARY]',
    'signal_narrative': '[SIGNAL NARRATIVE]',
    'price_analysis': '[PRICE ANALYSIS]',
    'vessel_intelligence': '[VESSEL INTELLIGENCE]',
    'freeform': '',
}


# === Endpoints ===

def _rag_available() -> bool:
    """Check if RAG dependencies are configured."""
    return bool(
        os.environ.get('VITE_SUPABASE_URL')
        and os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
        and os.environ.get('OPENAI_API_KEY')
        and os.environ.get('ANTHROPIC_API_KEY')
    )


@app.get('/health', response_model=HealthResponse)
async def health():
    if model_loading:
        status = 'loading'
    elif model is not None:
        status = 'ok'
    else:
        status = 'no_model'
    return HealthResponse(
        status=status,
        model_loaded=model is not None,
        rag_available=_rag_available(),
    )


@app.post('/ask', response_model=AskResponse)
async def ask(req: AskRequest, _key: str = Security(verify_api_key)):
    """RAG endpoint: retrieves relevant data and generates a grounded answer using Claude."""
    if not _rag_available():
        raise HTTPException(
            status_code=503,
            detail='RAG not configured. Set VITE_SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY.'
        )

    import rag

    start_time = time.time()
    result = rag.ask(
        query=req.question,
        top_k=req.top_k,
        source_filter=req.source_filter,
        max_tokens=req.max_tokens,
    )
    elapsed_ms = int((time.time() - start_time) * 1000)

    return AskResponse(
        answer=result['answer'],
        sources=result['sources'],
        chunks_used=result['chunks_used'],
        model=result['model'],
        query_time_ms=elapsed_ms,
    )


@app.post('/generate', response_model=GenerateResponse)
async def generate(req: GenerateRequest, _key: str = Security(verify_api_key)):
    if model is None:
        detail = 'Model is loading, please wait...' if model_loading else 'Model not loaded. Set CHECKPOINT_URL and restart.'
        raise HTTPException(status_code=503, detail=detail)

    # Prepend mode tag
    tag = MODE_TAGS.get(req.mode, '')
    full_prompt = f"{tag}\n{req.prompt}" if tag else req.prompt

    # Encode
    token_ids = enc.encode(full_prompt, allowed_special={"<|endoftext|>"})

    # Truncate if prompt is too long for context window
    max_prompt_len = checkpoint_info.get('block_size', 1024) - req.max_tokens
    if len(token_ids) > max_prompt_len:
        token_ids = token_ids[-max_prompt_len:]

    x = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

    # Generate (run in thread to avoid blocking the async event loop)
    eot_token = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    def _generate():
        with torch.no_grad():
            return model.generate(x, req.max_tokens, temperature=req.temperature, top_k=req.top_k, eot_token=eot_token)

    start_time = time.time()
    y = await asyncio.to_thread(_generate)

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Decode only the new tokens
    generated_ids = y[0][len(token_ids):].tolist()
    generated_text = enc.decode(generated_ids)

    # Clean up: stop at <|endoftext|> if present
    eot_pos = generated_text.find('<|endoftext|>')
    if eot_pos >= 0:
        generated_text = generated_text[:eot_pos]

    return GenerateResponse(
        text=generated_text.strip(),
        tokens_generated=len(generated_ids),
        generation_time_ms=elapsed_ms,
        mode=req.mode,
        model_info={
            'params': checkpoint_info.get('n_params', 0),
            'val_loss': checkpoint_info.get('best_val_loss', 0),
        },
    )
