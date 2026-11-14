# =============================================================================
# Optional application image (reproducible CPU environment for the pipeline).
# GPU training is expected to run on the host / a CUDA-enabled machine; this
# image is primarily for running preprocessing, correlation, and evaluation.
# Build from project root:  docker build -f docker/Dockerfile -t cti-platform .
# =============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for lxml / scientific wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "-c", "print('CTI platform image ready. Mount data/ and run modules under src/.')"]
