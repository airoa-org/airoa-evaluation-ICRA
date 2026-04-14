# Blackwell (RTX 5070 Ti / sm_120) support: CUDA 12.8 runtime base.
# The evaluation GPU is compute capability 12.0 (sm_120); older CUDA bases
# do not ship kernels for it and torch fails at runtime with
# "no kernel image is available for execution on the device".
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
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

# Create virtual environment and install locked deps including local workspace packages.
RUN uv venv --python 3.11 /workspace/.venv
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev --no-install-project
RUN uv pip install --python /workspace/.venv/bin/python pytest

# Replace the PyPI torch wheel (cu126, no sm_120 kernels) with the cu128 build
# from the official PyTorch index. Required for Blackwell (RTX 5070 Ti);
# without it, any CUDA op raises "no kernel image is available for execution
# on the device". --no-deps keeps the rest of uv sync's resolution intact.
RUN /workspace/.venv/bin/pip install --force-reinstall --no-deps \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128

ENV PATH="/workspace/.venv/bin:${PATH}"
ENV PYTHONPATH="/workspace:/workspace/src:/workspace/packages/policy-client/src"

ENTRYPOINT ["/workspace/server/entrypoint.sh"]
