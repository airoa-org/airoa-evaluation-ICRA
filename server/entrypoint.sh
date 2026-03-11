#!/usr/bin/env bash
set -euo pipefail

: "${POLICY_CHECKPOINT_DIR:?POLICY_CHECKPOINT_DIR is required}"
HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
PORT="${POLICY_SERVER_PORT:-8000}"

# pi05_hsr_task47_ep50_v2, pi0_hsr_airoa-moma
POLICY_CONFIG_NAME="${POLICY_CONFIG_NAME:-pi0_hsr_airoa-moma}"

ARGS=(  
  "--checkpoint-dir" "${POLICY_CHECKPOINT_DIR}"
  "--config-name" "${POLICY_CONFIG_NAME}"
  "--host" "${HOST}"
  "--port" "${PORT}"
)

exec /workspace/.venv/bin/python /workspace/server/serve_hsr_policy_ws.py "${ARGS[@]}"
