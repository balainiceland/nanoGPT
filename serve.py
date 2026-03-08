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
import time
from contextlib import asynccontextmanager

import torch
import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import GPTConfig, GPT

# Configuration
CHECKPOINT_DIR = os.environ.get('PELAGIC_CHECKPOINT_DIR', 'out-pelagic')
DEVICE = os.environ.get('PELAGIC_DEVICE', 'cpu')
MAX_TOKENS_LIMIT = 500

# Global state
model = None
enc = None
checkpoint_info = {}


def load_model():
    """Load model from checkpoint."""
    global model, enc, checkpoint_info

    ckpt_path = os.path.join(CHECKPOINT_DIR, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"WARNING: No checkpoint at {ckpt_path}. Server will return errors on /generate.")
        return

    print(f"Loading checkpoint from {ckpt_path}...")
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

    print(f"Model loaded: {checkpoint_info['n_params']:,} params, "
          f"val_loss={checkpoint_info['best_val_loss']}, "
          f"iter={checkpoint_info['iter_num']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
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
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Request / Response Models ===

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field(
        default='freeform',
        description='Generation mode: market_commentary, signal_narrative, price_analysis, freeform'
    )
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_k: int = Field(default=200, ge=1, le=500)
    max_tokens: int = Field(default=200, ge=10, le=500)


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    generation_time_ms: int
    mode: str
    model_info: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: dict


# === Mode Tags ===

MODE_TAGS = {
    'market_commentary': '[MARKET COMMENTARY]',
    'signal_narrative': '[SIGNAL NARRATIVE]',
    'price_analysis': '[PRICE ANALYSIS]',
    'freeform': '',
}


# === Endpoints ===

@app.get('/health', response_model=HealthResponse)
async def health():
    return HealthResponse(
        status='ok' if model is not None else 'no_model',
        model_loaded=model is not None,
        model_info=checkpoint_info,
    )


@app.post('/generate', response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail='Model not loaded. Place checkpoint at out-pelagic/ckpt.pt and restart.'
        )

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

    # Generate
    start_time = time.time()
    with torch.no_grad():
        y = model.generate(x, req.max_tokens, temperature=req.temperature, top_k=req.top_k)

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
