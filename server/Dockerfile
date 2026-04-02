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
COPY src/hierarchical_vla ./src/hierarchical_vla
COPY src/moe ./src/moe
COPY server ./server
COPY runtime_core ./runtime_core

# HVLA: PA マップ + config をイメージに含める
COPY pa_decomposition_v2.json ./pa_decomposition.json
COPY hierarchical_config_optimized.yaml ./hierarchical_config.yaml

# Create virtual environment and install locked deps including local workspace packages.
RUN uv venv --python 3.12 /workspace/.venv
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev && \
    uv pip install --python /workspace/.venv/bin/python pytest && \
    uv pip install --python /workspace/.venv/bin/python \
        torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128 --reinstall && \
    rm -rf /root/.cache/uv

# NOTE: transformers_replace は transformers 4.53.x 用。
# transformers 5.3.0 + lerobot ramen ブランチでは不要（LossKwargs 削除済み）。

# paligemma tokenizer をリポジトリ同梱ファイルからローカルパスに配置（HF_TOKEN 不要）
# entrypoint.sh でチェックポイントの tokenizer_name をこのパスに書き換える
COPY tokenizer/paligemma-3b-pt-224 /workspace/tokenizer/paligemma-3b-pt-224/

ENV PATH="/workspace/.venv/bin:${PATH}"
ENV PYTHONPATH="/workspace:/workspace/src:/workspace/server:/workspace/packages/policy-client/src"
ENV OPENPI_DATA_HOME="/policy_assets"

ENTRYPOINT ["/workspace/server/entrypoint.sh"]
