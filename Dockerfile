FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU-only (~200MB) + dependencies
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    numpy \
    tiktoken \
    fastapi \
    uvicorn[standard]

# Copy model code and server
COPY model.py serve.py configurator.py ./

# Checkpoint is downloaded at startup via CHECKPOINT_URL env var
RUN mkdir -p out-pelagic

ENV PYTHONUNBUFFERED=1
ENV PELAGIC_DEVICE=cpu
ENV PELAGIC_CHECKPOINT_DIR=out-pelagic
# Set CHECKPOINT_URL in Railway to download ckpt.pt at startup

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
