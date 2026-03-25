# Blackwell (RTX 5070 Ti / SM_120) 対応: CUDA 12.8.1
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
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

# HVLA: PA マップ + config をイメージに含める
COPY pa_decomposition_v2.json ./pa_decomposition.json
COPY hierarchical_config_optimized.yaml ./hierarchical_config.yaml

# Create virtual environment and install locked deps including local workspace packages.
RUN uv venv --python 3.11 /workspace/.venv
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev
RUN uv pip install --python /workspace/.venv/bin/python pytest

# Support PyTorch checkpoints (same behavior as upstream serve_policy image).
RUN /workspace/.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /workspace/src/openpi/models_pytorch/transformers_replace/* {}

ENV PATH="/workspace/.venv/bin:${PATH}"
ENV PYTHONPATH="/workspace:/workspace/src:/workspace/server:/workspace/packages/policy-client/src"
ENV OPENPI_DATA_HOME="/policy_assets"

ENTRYPOINT ["/workspace/server/entrypoint.sh"]
