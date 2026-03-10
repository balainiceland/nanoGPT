"""
FastAPI inference server for PelagicGPT.

RAG-powered Q&A over blue economy data using Claude + Supabase pgvector.

Usage:
  uvicorn serve:app --port 8000 --reload

Endpoints:
  GET  /health   — Service status
  POST /ask      — RAG: retrieve + Claude answer
"""

import os
import sys
import time
import logging

from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Sanitize env vars: strip \r that Railway sometimes injects from copy-paste
for _k, _v in list(os.environ.items()):
    if '\r' in _v:
        os.environ[_k] = _v.replace('\r', '')

# API key authentication
API_KEY = os.environ.get('API_KEY', '')
api_key_header = APIKeyHeader(name='X-API-Key', auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify the API key from the X-API-Key header."""
    if not API_KEY:
        return api_key
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=401, detail='Invalid or missing API key')
    return api_key

# Force unbuffered output for Railway logs
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("pelagic-gpt")


app = FastAPI(
    title="PelagicGPT",
    description="Blue economy RAG-powered Q&A using Claude + Supabase pgvector",
    version="0.2.0",
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


class HealthResponse(BaseModel):
    status: str
    rag_available: bool


# === Endpoints ===

def _rag_available() -> bool:
    """Check if RAG dependencies are configured."""
    return bool(
        os.environ.get('VITE_SUPABASE_URL')
        and os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
        and os.environ.get('OPENAI_API_KEY')
        and os.environ.get('ANTHROPIC_API_KEY')
    )


@app.get('/debug/connectivity')
async def debug_connectivity():
    """Test connectivity to external APIs."""
    import httpx
    results = {}
    for name, url in [
        ('anthropic', 'https://api.anthropic.com'),
        ('openai', 'https://api.openai.com'),
        ('supabase', os.environ.get('VITE_SUPABASE_URL', '')),
    ]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(url)
                results[name] = {'status': r.status_code, 'ok': True}
        except Exception as e:
            results[name] = {'error': f"{type(e).__name__}: {e}", 'ok': False}

    try:
        import anthropic
        results['anthropic_sdk_version'] = anthropic.__version__
    except Exception:
        results['anthropic_sdk_version'] = 'unknown'

    return results


@app.get('/health', response_model=HealthResponse)
async def health():
    return HealthResponse(
        status='ok',
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

    try:
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
    except Exception as e:
        logger.error(f"RAG /ask error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"RAG error: {type(e).__name__}: {e}")
