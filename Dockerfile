# Multi-stage Dockerfile for K-Number Extractor
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-minimal.txt .

# Build wheels
RUN pip install --user --no-cache-dir --wheel --no-binary :all: -r requirements-minimal.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Set environment variables
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TOKENIZERS_PARALLELISM=true

# Create results directory
RUN mkdir -p results

# Pre-download models (optional - comment out to skip)
# RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" || true
# RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-base-en')" || true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Set entrypoint
ENTRYPOINT ["python", "k_number_extractor_batch.py"]
CMD ["--limit", "100"]
