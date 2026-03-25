# R3 Reproduction Steps — Team RAMEN (Team 11)

## Overview

| Item | Value |
|------|-------|
| Model | π0.5 fine-tuned (SFT 20h, step 021438) |
| Checkpoint | `s3://airoa-icra-team-11/r3-pi05-run16-sft-20h-021k/` |
| Fork Repository | `https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA` |
| Branch | `feat/lerobot-pi05` |
| Backend | `POLICY_BACKEND=lerobot` |
| CUDA | 12.8.1 (Blackwell / RTX 5070 Ti compatible) |

## Prerequisites

- NVIDIA GPU with Blackwell architecture support (RTX 5070 Ti)
- Docker with NVIDIA Container Toolkit
- AWS CLI (for checkpoint download)
- HuggingFace token (for paligemma tokenizer)

## Step-by-step Reproduction

### 1. Clone the fork repository

```bash
git clone https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout feat/lerobot-pi05
```

### 2. Download the checkpoint from S3

```bash
mkdir -p checkpoints/r3
aws s3 cp s3://airoa-icra-team-11/r3-pi05-run16-sft-20h-021k/ checkpoints/r3/ \
    --recursive \
    --endpoint-url https://eabeb2a5516ef53a191452e5714fc16b.r2.cloudflarestorage.com
```

### 3. Set environment variables

```bash
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoints/r3
export POLICY_BACKEND=lerobot
export POLICY_CONFIG_NAME=pi05_hsr
export HF_TOKEN=<your_huggingface_token>
export TEST_MODE=true
```

### 4. Start the Docker containers

```bash
./RUN-DOCKER-CONTAINER.sh up
```

This will:
- Build the Docker image (CUDA 12.8.1 base)
- Start the policy server (WebSocket on port 8000)
- Automatically login to HuggingFace using HF_TOKEN
- Load the PI05Policy checkpoint

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

- `config.json` in the checkpoint must have `"compile_model": false`. If set to `true` (max-autotune), the first inference will timeout (>300s).
- The HF_TOKEN is required for downloading the `google/paligemma-3b-pt-224` tokenizer (gated model).
- The Docker image uses CUDA 12.8.1 for Blackwell (RTX 5070 Ti) compatibility.

## Checkpoint Files

```
r3-pi05-run16-sft-20h-021k/
├── config.json
├── model.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
├── policy_preprocessor.json
├── policy_preprocessor_step_2_normalizer_processor.safetensors
└── train_config.json
```

## Verification

After starting the containers, verify the policy server is running:

```bash
# Check server logs
docker logs airoa_policy_server 2>&1 | tail -5
# Expected: "server listening on 0.0.0.0:8000"
```
