# airoa-evaluation-ICRA — Team RAMEN (Team 11)

Participant evaluation runtime for ICRA 2026 VLA Workshop Competition.

**Branch**: `feat/lerobot-pi05`
**Backend**: LeRobot PI05Policy + Hierarchical VLA (HVLA)

## Quick Start (R3)

```bash
# 1. Set environment
export POLICY_CHECKPOINT_PATH=/abs/path/to/checkpoint_dir
export HF_TOKEN=<your_huggingface_token>

# 2. Start containers
./RUN-DOCKER-CONTAINER.sh up

# 3. Enter client shell
./RUN-DOCKER-CONTAINER.sh shell

# 4. Launch (inside container)
roslaunch hsr_policy_client hsr_policy_client.launch

# 5. Stop
./RUN-DOCKER-CONTAINER.sh down
```

See [R3_REPRODUCTION_STEPS.md](R3_REPRODUCTION_STEPS.md) for detailed reproduction steps.

## Architecture

```
Docker Container (airoa_policy_server)
├── LeRobotHSRPolicy          π0.5 fine-tuned model (transformers 4.53.2)
├── HierarchicalHSRPolicy     HVLA wrapper (PA decomposition + PA Monitor + FM + Retry)
├── WebsocketPolicyServer      msgpack protocol, port 8000
└── LLMAPIClient               HTTP client to external LLM server (optional, port 8001)

External (optional)
└── LLM API Server             Qwen3.5-4B for unknown task decomposition (port 8001)
```

## Modes

| Mode | POLICY_MODE | Description |
|------|:-----------:|-------------|
| **E2E** | `e2e` | End-to-end: SHT prompt directly to π0.5 |
| **HVLA** | `hierarchical` | Hierarchical VLA: SHT → PA decomposition → PA-level inference |

Set via `.env` or `export POLICY_MODE=hierarchical`.

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `POLICY_CHECKPOINT_PATH` | Absolute path to checkpoint directory |
| `HF_TOKEN` | HuggingFace token (required for paligemma tokenizer) |

### Optional

| Variable | Default | Description |
|----------|:-------:|-------------|
| `POLICY_BACKEND` | `lerobot` | Backend: `openpi` or `lerobot` |
| `POLICY_MODE` | `e2e` | Inference mode: `e2e` or `hierarchical` |
| `POLICY_CONFIG_NAME` | — | Train config name (required for openpi) |
| `POLICY_SERVER_PORT` | `8000` | WebSocket server port |
| `LLM_API_HOST` | — | LLM API server host (for unknown task decomposition) |
| `LLM_API_PORT` | `8001` | LLM API server port |
| `FM_MODEL` | — | Failure Monitor model path (joblib) |
| `FM_SCALER` | — | FM scaler path (joblib) |

## Key Files

| File | Description |
|------|-------------|
| `server/Dockerfile` | CUDA 12.8.1 (Blackwell compatible) |
| `server/entrypoint.sh` | HF auth + HVLA mode support |
| `server/serve_hsr_policy_ws.py` | WebSocket server (E2E / HVLA) |
| `server/lerobot_hsr_policy.py` | LeRobot PI05Policy wrapper |
| `server/hierarchical_hsr_policy.py` | HVLA wrapper (PA Monitor + FM + Retry) |
| `server/llm_api_client.py` | LLM API client for external Qwen3.5-4B |
| `pa_decomposition_v2.json` | PA decomposition map (93 SHTs) |
| `hierarchical_config_optimized.yaml` | HVLA config (max_steps_short, FM, etc.) |

## HVLA Features

- **PA Planner**: Rule (93 SHT) → Fuzzy match → LLM API → E2E fallback
- **PA Monitor**: Convergence detection + gripper transition + max_steps adaptive control
- **Failure Monitor**: State-based RF (AUROC=0.881) → RetryController
- **PA-R3**: Automatic pronoun resolution in LLM-generated PAs

## WebSocket I/O Contract

Request fields:
- `head_rgb`: `(H, W, 3)` uint8
- `hand_rgb`: `(H, W, 3)` uint8
- `state`: `(8,)` float32
- `prompt`: `str`

Response field:
- `actions`: `(T, 11)` float32, `T >= 1`

Action order:
`[arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll, gripper, head_pan, head_tilt, base_x, base_y, base_t]`

## Host Requirements

- Linux
- Docker Engine + Docker Compose v2
- NVIDIA driver + NVIDIA Container Toolkit
- GPU: NVIDIA RTX 5070 Ti (Blackwell) or compatible

## Important Notes

- `config.json` in checkpoint must have `"compile_model": false`
- HF_TOKEN is required (paligemma tokenizer is gated)
- CUDA 12.8.1 base image for Blackwell compatibility
