FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
 && rm -rf /var/lib/apt/lists/*

# depth-lens core deps
RUN pip install numpy matplotlib pydantic click tqdm pytest

# OpenMythos as an installable dep so the openmythos adapter works
RUN pip install open-mythos

COPY . .

RUN pip install -e . --no-deps

CMD ["depth-lens", "--help"]
