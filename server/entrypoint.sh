#!/usr/bin/env bash
set -euo pipefail

# HuggingFace 認証（paligemma tokenizer 等のゲート付きモデルに必要）
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[INFO] Logging in to HuggingFace..."
  huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null || \
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" 2>/dev/null || \
    echo "[WARN] HuggingFace login failed, continuing anyway"
fi

: "${POLICY_CHECKPOINT_DIR:?POLICY_CHECKPOINT_DIR is required}"

BACKEND="${POLICY_BACKEND:-openpi}"
HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
PORT="${POLICY_SERVER_PORT:-8000}"

ARGS=(
  "--backend" "${BACKEND}"
  "--checkpoint-dir" "${POLICY_CHECKPOINT_DIR}"
  "--host" "${HOST}"
  "--port" "${PORT}"
)

# --config-name is required for openpi, optional for lerobot
if [[ -n "${POLICY_CONFIG_NAME:-}" ]]; then
  ARGS+=("--config-name" "${POLICY_CONFIG_NAME}")
elif [[ "${BACKEND}" = "openpi" ]]; then
  echo "ERROR: POLICY_CONFIG_NAME is required for openpi backend" >&2
  exit 1
fi

if [[ -n "${POLICY_DEFAULT_PROMPT:-}" ]]; then
  ARGS+=("--default-prompt" "${POLICY_DEFAULT_PROMPT}")
fi

if [[ -n "${POLICY_RECORD_DIR:-}" ]]; then
  ARGS+=("--record-dir" "${POLICY_RECORD_DIR}")
fi

if [[ -n "${POLICY_PYTORCH_DEVICE:-}" ]]; then
  ARGS+=("--pytorch-device" "${POLICY_PYTORCH_DEVICE}")
fi

# HVLA モード
MODE="${POLICY_MODE:-e2e}"
if [[ "${MODE}" = "hierarchical" ]]; then
  ARGS+=("--mode" "hierarchical")
  ARGS+=("--pa-decomposition" "${PA_DECOMPOSITION:-/workspace/pa_decomposition.json}")
  ARGS+=("--policy-config" "${POLICY_CONFIG:-/workspace/hierarchical_config.yaml}")
  if [[ -n "${FM_MODEL:-}" ]]; then
    ARGS+=("--fm-model" "${FM_MODEL}")
  fi
  if [[ -n "${FM_SCALER:-}" ]]; then
    ARGS+=("--fm-scaler" "${FM_SCALER}")
  fi
  if [[ -n "${LLM_API_HOST:-}" ]]; then
    ARGS+=("--llm-api-host" "${LLM_API_HOST}")
  fi
  if [[ -n "${LLM_API_PORT:-}" ]]; then
    ARGS+=("--llm-api-port" "${LLM_API_PORT}")
  fi
fi

exec /workspace/.venv/bin/python /workspace/server/serve_hsr_policy_ws.py "${ARGS[@]}"
