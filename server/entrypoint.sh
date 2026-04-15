#!/usr/bin/env bash
set -euo pipefail

: "${POLICY_CHECKPOINT_DIR:?POLICY_CHECKPOINT_DIR is required}"

exec /workspace/.venv/bin/python /workspace/server/serve_hsr_policy_ws.py
