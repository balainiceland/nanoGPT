FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install dependencies — split into two layers for better caching
# Layer 1: PyTorch CPU-only (largest dep, changes least)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Layer 2: Everything else (smaller, faster)
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Copy model code and server
COPY model.py serve.py configurator.py rag.py ./

# Checkpoint is downloaded at startup via CHECKPOINT_URL env var
RUN mkdir -p out-pelagic && chown -R appuser:appuser /app

ENV PYTHONUNBUFFERED=1
ENV PELAGIC_DEVICE=cpu
ENV PELAGIC_CHECKPOINT_DIR=out-pelagic
# Set CHECKPOINT_URL in Railway to download ckpt.pt at startup
# RAG requires: VITE_SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY

# Run as non-root user
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn serve:app --host 0.0.0.0 --port ${PORT:-8000}"]
