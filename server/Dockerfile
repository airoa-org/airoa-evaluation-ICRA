FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential clang ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY packages/policy-client ./packages/policy-client
COPY src ./src
COPY server ./server
COPY runtime_core ./runtime_core

# Create virtual environment
RUN uv venv --python 3.11 /workspace/.venv
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev --no-install-project
RUN uv pip install --python /workspace/.venv/bin/python pytest

# Fix 2: install CLIP and matching torch for CUDA 12.8
RUN /workspace/.venv/bin/pip install --upgrade pip
RUN /workspace/.venv/bin/pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN /workspace/.venv/bin/pip install \
    ftfy regex tqdm \
    git+https://github.com/openai/CLIP.git

# Fix 6: pre-download CLIP weights at build time (no network at runtime)
RUN /workspace/.venv/bin/python -c "import clip; clip.load('ViT-B/32')"

ENV PATH="/workspace/.venv/bin:${PATH}"
ENV PYTHONPATH="/workspace:/workspace/src:/workspace/packages/policy-client/src"

ENTRYPOINT ["/workspace/server/entrypoint.sh"]
