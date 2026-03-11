FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential clang ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY packages/policy-client ./packages/policy-client

# Create virtual environment and install locked deps including local workspace packages.
# CHANGE_LOG: uvのキャッシュをマウントして高速化
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.11 /workspace/.venv
RUN --mount=type=cache,target=/root/.cache/uv \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev --no-install-project --link-mode=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --link-mode=copy --python /workspace/.venv/bin/python pytest

COPY src ./src
COPY server ./server
COPY runtime_core ./runtime_core

ENV PATH="/workspace/.venv/bin:${PATH}"
ENV PYTHONPATH="/workspace:/workspace/src:/workspace/packages/policy-client/src"

ENTRYPOINT ["/workspace/server/entrypoint.sh"]
