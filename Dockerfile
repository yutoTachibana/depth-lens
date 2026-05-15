# depth-lens GPU image.
#
# Defaults to CUDA 12.4 + PyTorch 2.5.1 + the OpenMythos and HuggingFace
# extras. API-only users (Anthropic / OpenAI / Gemini) can use a slim Python
# image instead; see Dockerfile.api below in this file.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
 && rm -rf /var/lib/apt/lists/*

# Install heavyweight deps first so source changes don't bust the layer.
RUN pip install \
    numpy>=1.26 matplotlib>=3.7 pydantic>=2.6 click>=8.1 tqdm>=4.66 \
    transformers>=4.40 \
    open-mythos

COPY pyproject.toml README.md ./
COPY depth_lens ./depth_lens
COPY tests ./tests
COPY examples ./examples

RUN pip install -e ".[anthropic,openai,gemini,dashboard]" --no-deps

CMD ["depth-lens", "--help"]


# -----------------------------------------------------------------------------
# API-only slim image. Use for laptops without GPUs that only probe remote
# reasoning APIs.
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY depth_lens ./depth_lens

RUN pip install -e ".[anthropic,openai,gemini,dashboard]"

CMD ["depth-lens", "--help"]
