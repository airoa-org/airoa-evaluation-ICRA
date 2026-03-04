#!/usr/bin/env bash
set -euo pipefail

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

exec /workspace/.venv/bin/python /workspace/server/serve_hsr_policy_ws.py "${ARGS[@]}"
