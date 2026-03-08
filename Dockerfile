FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU-only (~200MB) + dependencies
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    tiktoken \
    fastapi \
    uvicorn[standard]

# Copy model code and checkpoint
COPY model.py serve.py ./
COPY out-pelagic/ckpt.pt out-pelagic/ckpt.pt

ENV PELAGIC_DEVICE=cpu
ENV PELAGIC_CHECKPOINT_DIR=out-pelagic

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
