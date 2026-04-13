#!/usr/bin/env bash
set -euo pipefail

: "${POLICY_CHECKPOINT_DIR:?POLICY_CHECKPOINT_DIR is required}"
HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
PORT="${POLICY_SERVER_PORT:-8000}"
PYTORCH_DEVICE="${POLICY_PYTORCH_DEVICE:-}"

# pi05_hsr_task47_ep50_v2, pi0_hsr_airoa-moma
POLICY_CONFIG_NAME="${POLICY_CONFIG_NAME:-0412_pi05_airoa_hsr_fullfinetuning_statediff_horizon8_relocate}"

ARGS=(  
  "--checkpoint-dir" "${POLICY_CHECKPOINT_DIR}"
  "--config-name" "${POLICY_CONFIG_NAME}"
  "--host" "${HOST}"
  "--port" "${PORT}"
)

if [[ -n "${PYTORCH_DEVICE}" ]]; then
  ARGS+=("--pytorch-device" "${PYTORCH_DEVICE}")
fi

exec /workspace/.venv/bin/python /workspace/server/serve_hsr_policy_ws.py "${ARGS[@]}"
