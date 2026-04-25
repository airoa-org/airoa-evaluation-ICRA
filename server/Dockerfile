FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential clang ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY packages/policy-client ./packages/policy-client
COPY src/openpi ./src/openpi
COPY server ./server
COPY runtime_core ./runtime_core

# Create virtual environment and install locked deps including local workspace packages.
RUN uv venv --python 3.11 /workspace/.venv
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev
# Reinstall torch for CUDA 12.8 (Blackwell sm_120)
RUN /workspace/.venv/bin/pip install --force-reinstall \
    torch==2.7.1 torchvision \
    --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --python /workspace/.venv/bin/python pytest

# Support PyTorch checkpoints (same behavior as upstream serve_policy image).
RUN /workspace/.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /workspace/src/openpi/models_pytorch/transformers_replace/* {}

ENV PATH="/workspace/.venv/bin:${PATH}"
ENV PYTHONPATH="/workspace:/workspace/src:/workspace/packages/policy-client/src"
ENV OPENPI_DATA_HOME="/policy_assets"

ENTRYPOINT ["/workspace/server/entrypoint.sh"]
