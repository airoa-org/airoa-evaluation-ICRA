#!/usr/bin/env bash
set -euo pipefail

# POLICY_CHECKPOINT_DIR is set by docker-compose.yml (defaults to /policy_checkpoint).
# It is OPTIONAL for the placeholder ZeroPolicy on the `base` branch — your own
# policy module is expected to read it (and may fail if it is empty / missing).
HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
PORT="${POLICY_SERVER_PORT:-8000}"

ARGS=(
  "--host" "${HOST}"
  "--port" "${PORT}"
)

if [[ -n "${POLICY_CHECKPOINT_DIR:-}" ]]; then
  ARGS+=("--checkpoint-dir" "${POLICY_CHECKPOINT_DIR}")
fi

# Set POLICY_MODULE to swap in your own policy class without editing
# serve_hsr_policy_ws.py, e.g. POLICY_MODULE="my_policy.adapter:MyPolicyAdapter".
if [[ -n "${POLICY_MODULE:-}" ]]; then
  ARGS+=("--policy-module" "${POLICY_MODULE}")
fi

exec /workspace/.venv/bin/python /workspace/server/serve_hsr_policy_ws.py "${ARGS[@]}"
