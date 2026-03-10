"""RAG (Retrieval-Augmented Generation) module for PelagicGPT.

Retrieves relevant chunks from Supabase pgvector and generates
grounded answers using Claude.
"""

import os
import logging

logger = logging.getLogger("pelagic-gpt")

SUPABASE_URL = os.environ.get('VITE_SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY', '')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

EMBEDDING_MODEL = 'text-embedding-3-small'
TOP_K = 8

# Prompt injection mitigation patterns
import re
_INJECTION_PATTERNS = re.compile(
    r'^\s*('
    r'(system\s*:)|(instruction\s*:)|(you\s+are\b)|(ignore\s+(all\s+)?previous)|(forget\s+(all\s+)?previous)|'
    r'(disregard\s+(all\s+)?previous)|(override\s+(all\s+)?previous)|(new\s+instructions?\s*:)|'
    r'(SYSTEM\s*:)|(INSTRUCTION\s*:)|(###\s*system)|(###\s*instruction)|'
    r'(\[SYSTEM\])|(\\n\\nsystem)'
    r')',
    re.IGNORECASE | re.MULTILINE
)


def sanitize_query(query: str) -> str:
    """Sanitize user query to mitigate prompt injection attempts."""
    # Strip system/instruction-like prefixes
    cleaned = _INJECTION_PATTERNS.sub('', query).strip()
    # Remove any remaining leading special characters that might be used for injection
    cleaned = cleaned.lstrip('#[]<>')
    # If sanitization removed everything, return a safe default
    if not cleaned:
        return "general market overview"
    return cleaned

# System prompt for Claude RAG generation
SYSTEM_PROMPT = """You are PelagicGPT, an AI analyst specializing in the blue economy —
seafood, aquaculture, ocean technology, marine biotech, and related industries.

You answer questions using ONLY the retrieved context provided below.
If the context doesn't contain enough information to fully answer, say so honestly.

Guidelines:
- Be specific with numbers, tickers, and company names from the context
- When discussing financials, cite the period/date from the data
- For market commentary, reference the source and date
- Keep answers concise and data-driven
- Use bullet points for comparisons
- If multiple sources conflict, note the discrepancy"""


def _get_supabase():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _get_openai():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def _get_anthropic():
    import anthropic
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def embed_query(query: str) -> list[float]:
    """Embed a query string."""
    try:
        client = _get_openai()
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
        return resp.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"OpenAI embedding failed: {type(e).__name__}: {e}") from e


def retrieve_chunks(query: str, top_k: int = TOP_K, source_filter: str = None) -> list[dict]:
    """Retrieve relevant chunks from Supabase pgvector."""
    query_embedding = embed_query(query)

    try:
        db = _get_supabase()
        result = db.rpc('match_rag_chunks', {
            'query_embedding': query_embedding,
            'match_count': top_k,
            'filter_source': source_filter,
        }).execute()
        return result.data or []
    except RuntimeError:
        raise  # Re-raise embedding errors as-is
    except Exception as e:
        raise RuntimeError(f"Supabase retrieval failed: {type(e).__name__}: {e}") from e


def generate_answer(query: str, chunks: list[dict], max_tokens: int = 1000) -> dict:
    """Generate a grounded answer using Claude with retrieved context."""
    # Format context
    context_parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.get('source', 'unknown')
        similarity = chunk.get('similarity', 0)
        content = chunk.get('content', '')
        context_parts.append(
            f"--- Source {i+1} [{source}] (relevance: {similarity:.2f}) ---\n{content}"
        )

    context = '\n\n'.join(context_parts)

    try:
        client = _get_anthropic()
        message = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{
                'role': 'user',
                'content': f"""Retrieved context:

{context}

---

Question: {query}

Answer based on the retrieved context above:"""
            }],
        )
        answer = message.content[0].text
    except Exception as e:
        raise RuntimeError(f"Claude generation failed: {type(e).__name__}: {e}") from e

    return {
        'answer': answer,
        'sources': [
            {
                'source': c.get('source'),
                'similarity': round(c.get('similarity', 0), 3),
                'metadata': c.get('metadata', {}),
                'excerpt': c.get('content', '')[:200],
            }
            for c in chunks
        ],
        'chunks_used': len(chunks),
        'model': 'claude-sonnet-4-20250514',
    }


def ask(query: str, top_k: int = TOP_K, source_filter: str = None, max_tokens: int = 1000) -> dict:
    """Full RAG pipeline: sanitize → embed → retrieve → generate."""
    # 0. Sanitize query to mitigate prompt injection
    query = sanitize_query(query)

    # 1. Retrieve relevant chunks
    logger.info(f"RAG: embedding query ({len(query)} chars)...")
    chunks = retrieve_chunks(query, top_k=top_k, source_filter=source_filter)
    logger.info(f"RAG: retrieved {len(chunks)} chunks")

    if not chunks:
        return {
            'answer': 'No relevant data found in the knowledge base for this query.',
            'sources': [],
            'chunks_used': 0,
            'model': 'none',
        }

    # 2. Generate grounded answer
    logger.info(f"RAG: generating answer with Claude...")
    result = generate_answer(query, chunks, max_tokens=max_tokens)
    logger.info(f"RAG: done")
    return result
