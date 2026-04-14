# REPRODUCTION_STEPS — Team XX (submission note template)

> Submissions are sent to the organizers as **fork URL + branch + note** (a description of how to run your model). This file is the recommended structure for your **note** — copy it to your repository root as `REPRODUCTION_STEPS.md`, fill every section, commit it, and link (or paste) it as your note. The evaluator will follow it **literally**, so commands and paths must be correct. Placeholders in parentheses are guidance — replace with your actual values.
>
> 日本語版: [REPRODUCTION_STEPS.template_ja.md](REPRODUCTION_STEPS.template_ja.md)

---

## 1. Overview

| Item | Value |
|---|---|
| Model summary | (one line — what is this? e.g. "π0.5 SFT fine-tuned on public_tasks/task6911") |
| Framework | (e.g. OpenPI / LeRobot / custom PyTorch / custom JAX) |
| Repository | `https://github.com/<you>/airoa-evaluation-ICRA` |
| Branch | `feat/my-policy` |
| Commit hash | `abcd1234…` (optional but recommended) |
| Checkpoint S3 path | `s3://<bucket>/<path>/` |
| Expected VRAM | (e.g. ~9 GB) |

---

## 2. Prerequisites

- NVIDIA GPU with ≥ 16 GB VRAM (**Blackwell-compatible**: RTX 5070 Ti / compute 12.0)
- Docker Engine + Docker Compose v2
- NVIDIA Container Toolkit
- External credentials required (e.g. `HF_TOKEN`, S3 credentials)? **State "none required" if not applicable.**

> If your model relies on anything not bundled in the repo/image at build time (gated HuggingFace models, external services, etc.), list every external resource the evaluator must access. The evaluation environment may be offline/locked.

---

## 3. Reproduction Steps

### 3.1 Clone and checkout

```bash
git clone https://github.com/<you>/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout <branch>
git rev-parse HEAD  # optional: should match the commit hash in §1
```

### 3.2 Download checkpoint

```bash
mkdir -p checkpoints/<name>
aws --profile <profile> --endpoint-url <url> \
    s3 sync s3://<bucket>/<path>/ checkpoints/<name>/
```

(Replace with whatever download method you actually use — HF CLI, `curl`, etc.)

### 3.3 Environment variables

The harness reads only three variables. List the ones you need:

```bash
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoints/<name>       # MUST be a directory
export POLICY_PYTORCH_DEVICE=cuda                             # optional
export POLICY_CONFIG_NAME=<openpi_config_name>                # only if you use the default OpenPI loader
```

If your server implementation reads any additional variables, list every one here (and make sure your `docker-compose.yml` / `.env` set them).

### 3.4 Start containers

```bash
./RUN-DOCKER-CONTAINER.sh up
```

### 3.5 Verify

```bash
# Wait for the policy server to be ready:
until curl -s http://localhost:8000/healthz 2>/dev/null | grep -q OK; do sleep 5; done
echo "READY"

# GPU usage sanity check:
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

### 3.6 Stop

```bash
./RUN-DOCKER-CONTAINER.sh down
```

---

## 4. Files modified relative to the base repo

List paths you edited beyond the core scope (`server/`, `src/`) and explain why.

| Path | Reason |
|---|---|
| `server/serve_hsr_policy_ws.py` | (e.g. loads MyPolicyAdapter instead of the default OpenPI loader) |
| `server/Dockerfile` | (e.g. added torch 2.5.1 + transformers 4.46.0 + custom deps) |
| `src/<your_policy>/` | (e.g. new model implementation) |
| `docker-compose.yml` | (e.g. added one env var your server reads) |
| `client/Dockerfile` | (e.g. CUDA 12.8.1 base for Blackwell compatibility) |

---

## 5. Important notes

Anything the evaluator must know that is not obvious from the commands above. Examples:

- (e.g. "first startup takes ~2 min due to weight loading")
- (e.g. "tokenizer is bundled at `<path>`, no `HF_TOKEN` required")
- (e.g. "model compile is disabled to avoid 300s first-inference timeout")

---

## 6. Checkpoint file layout

List what's inside your checkpoint directory. Layout depends on your framework — two common patterns:

**OpenPI-style:**

```
<name>/
├── params/                            # or model.safetensors
├── assets/<asset_id>/
│   └── norm_stats.json
└── config.yaml                        # or config.json
```

**Custom PyTorch-style:**

```
<name>/
├── model.pt                           # or model.safetensors
├── config.json
└── (any tokenizer / preprocessor files)
```

Include the actual layout and total size of your submission:

```
(paste your `tree` or `ls` output here)
```

Total size: ~X GB

---

## 7. Smoke test expected output

Copy 1–3 log lines from your own successful run so the evaluator knows what "working" looks like:

```
(paste actual log lines here, e.g.:)
[INFO] server listening on 0.0.0.0:8000
[INFO] Action executed.
[INFO] Action executed.
```

---

## 8. Contact

- Team: (team name)
- Representative: (name) <email>
- Submission date: YYYY-MM-DD
