# R3 Reproduction Steps — Team RAMEN (Team 11)

## Overview

| Item | Value |
|------|-------|
| Model | π0.5 fine-tuned FFN-only MoE 7 Expert |
| Checkpoint | `ICRA-2026-RAMEN/pi05-moe-ffn-only-7expert` (11.5 GB) |
| Fork Repository | `https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA` |
| Branch | `feat/lerobot-pi05` |
| Backend | `POLICY_BACKEND=lerobot` |
| CUDA | 12.8.1 (Blackwell / RTX 5070 Ti compatible) |
| LeRobot | ramen branch (transformers 5.3.0) |
| Mode | HVLA (hierarchical) — PA-level instruction with action postprocessing |
| VRAM | ~11.5 GB (fits RTX 5070 Ti 16 GB) |
| RAM | 64 GB recommended (31 GB minimum with low_cpu_mem mode) |
| SSD | Docker ~13 GB + Checkpoint 11.5 GB = ~24.5 GB (fits 30 GB limit) |

## Prerequisites

- NVIDIA GPU with Blackwell architecture support (RTX 5070 Ti, 16 GB VRAM)
- Docker with NVIDIA Container Toolkit
- HuggingFace token (for `google/paligemma-3b-pt-224` gated tokenizer)

## Step-by-step Reproduction

### 1. Clone the fork repository

```bash
git clone https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout feat/lerobot-pi05
```

### 2. Download the checkpoint

```bash
mkdir -p checkpoints/r3
huggingface-cli download ICRA-2026-RAMEN/pi05-moe-ffn-only-7expert \
    --local-dir checkpoints/r3
```

### 2b. Verify and fix compile_model setting

```bash
# compile_model must be false (true causes >300s timeout on first inference)
python3 -c "import json; c=json.load(open('checkpoints/r3/config.json')); print('compile_model:', c.get('compile_model', 'NOT SET'))"

# If compile_model is true, fix it:
python3 -c "
import json
with open('checkpoints/r3/config.json') as f: c = json.load(f)
c['compile_model'] = False
with open('checkpoints/r3/config.json', 'w') as f: json.dump(c, f, indent=2)
print('compile_model set to False')
"
```

### 3. Set environment variables

```bash
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoints/r3
export POLICY_BACKEND=lerobot
export POLICY_CONFIG_NAME=pi05_hsr
export POLICY_MODE=hierarchical
export HF_TOKEN=<your_huggingface_token>
```

### 4. Start the Docker containers

```bash
./RUN-DOCKER-CONTAINER.sh up
```

This will:
- Build the Docker image (CUDA 12.8.1 base, ~13 GB)
- Start the policy server (WebSocket on port 8000)
- Automatically login to HuggingFace using HF_TOKEN
- Detect `moe_config.json` in checkpoint → load FFN-only MoE with 7 Experts
- Auto-select Expert based on PA instruction keywords

### 5. Open a shell in the HSR client container

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

### 6. Launch the HSR policy client (inside the container)

```bash
roslaunch hsr_policy_client hsr_policy_client.launch
```

### 7. Stop the containers

```bash
./RUN-DOCKER-CONTAINER.sh down
```

## Important Notes

- `config.json` in the checkpoint **must** have `"compile_model": false`. If set to `true` (max-autotune), the first inference will timeout (>300s).
- The HF_TOKEN is required for downloading the `google/paligemma-3b-pt-224` tokenizer (gated model).
- The Docker image uses CUDA 12.8.1 for Blackwell (RTX 5070 Ti) compatibility.
- HVLA mode requires `pa_decomposition_v2.json` and `hierarchical_config_optimized.yaml` (included in fork repo).
- FFN-only MoE (11.5 GB) fits RTX 5070 Ti (16 GB). MoE mode activates automatically when `moe_config.json` is present in the checkpoint.
- **SSD limit: 30 GB**. Docker image (~13 GB) + checkpoint (11.5 GB) = ~24.5 GB, within limit.
- **RAM: 64 GB or more recommended**. The MoE model loading requires ~21 GB for PyTorch/CUDA initialization + ~12 GB for weights. With 31 GB RAM, the `low_cpu_mem` mode (meta device + direct GPU loading) is used automatically, but 64 GB provides more stability.
- **PA-level evaluation**: Confirmed by organizers. Instructions are sent at PA level. No LLM Planner needed.

## Checkpoint Files

```
pi05-moe-ffn-only-7expert/
├── config.json              (compile_model=false)
├── model.safetensors        (11.5 GB, FFN-only 7 Expert weights)
├── moe_config.json          (Expert routing config, ffn_only_moe=true)
├── policy_postprocessor.json
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
├── policy_preprocessor.json
├── policy_preprocessor_step_2_normalizer_processor.safetensors
└── train_config.json
```

## HVLA Features

- **MoE Routing**: Auto-select Expert from PA instruction keywords (e.g., "pick coffee" → Expert 0)
- **Action Smoothing**: EMA (alpha=0.3, Jerk -40%, MSE -6.8%)
- **Gripper Clipping**: [0.0, 1.0]
- **Navigate**: base_theta damping (scale=0.1) + distance-based completion (NAV-1, ≤0.6m)
- **Pick/Place**: gripper transition AND arm convergence (EE-2) + base=[0,0,0] stop
- **GRP-1**: Gripper amplitude check (≥0.05) to prevent false positives

## Verification

After starting the containers, verify the policy server is running:

```bash
# Check server logs
docker logs airoa_policy_server 2>&1 | tail -5
# Expected: "server listening on 0.0.0.0:8000"
# MoE mode: "MoE Policy ロード完了: 7 Experts"
```
