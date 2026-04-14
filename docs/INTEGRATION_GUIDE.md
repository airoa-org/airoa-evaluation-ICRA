# Integration Guide

> This document walks you through adapting your VLA/policy model to the AIRoA evaluation pipeline, step by step. Follow it in order; each step builds on the previous one.
>
> **Read this before asking questions.** Every "why doesn't my model load?" issue we have seen so far is caused by skipping one of these steps.

---

## Prerequisites

- [ ] You have a working policy model (any framework: PyTorch, JAX, …).
- [ ] You have a machine with NVIDIA GPU ≥ 16 GB VRAM, Docker, NVIDIA Container Toolkit.
- [ ] You have read [README.md](../README.md) §5 (WebSocket I/O Contract) and §6 (environment variables).
- [ ] You can run `./RUN-DOCKER-CONTAINER.sh up` on the **unmodified** base repo with the default OpenPI checkpoint and see `Action executed.` in the logs. (This confirms your host setup is correct before you add your own code.)

---

## Step 0. Fork and branch

**Submit from a fork of `airoa-org/airoa-evaluation-ICRA`.** Not a separate repository — a fork. The harness (`RUN-DOCKER-CONTAINER.sh`, `docker-compose.yml`, `runtime_core/`, `packages/`, `deploy/`) lives only here; your separate repo cannot be evaluated.

**Pick the right branch to start from:**

| Branch | Use when |
|---|---|
| **`base`** *(recommended default)* | You're using PyTorch / JAX / LeRobot / your own framework. Minimal harness + a `ZeroPolicy` placeholder so the smoke test passes immediately. |
| **`sample-openpi`** | Your model is `PI0Pytorch` / OpenPI-compatible and you want the loader as a worked example. Ships the full `src/openpi/` source tree. |

```bash
# On GitHub: Fork airoa-org/airoa-evaluation-ICRA → <your-org>/airoa-evaluation-ICRA
git clone https://github.com/<your-org>/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout base                  # or `sample-openpi`
git checkout -b feat/my-policy     # your submission branch
```

Verify you got the branch you wanted:

```bash
ls server/serve_hsr_policy_ws.py src/
# - On `base`: serve_hsr_policy_ws.py uses ZeroPolicy, src/ has only README.md.
# - On `sample-openpi`: serve_hsr_policy_ws.py uses the OpenPI loader, src/openpi/… is populated.
```

---

## Step 1. Place your model code under `src/`

```
src/
└── my_policy/
    ├── __init__.py
    ├── model.py              # your network
    ├── adapter.py            # bridges WebSocket contract ↔ your model
    └── (any supporting files)
```

If your policy depends on an additional local package (e.g. a custom data-loading library), add it under `packages/` and register it in `pyproject.toml` as a workspace member. See the example fork layout in §9 for a concrete structure.

---

## Step 2. Write the adapter

The pipeline calls exactly one method on your policy object:

```python
policy.infer(obs: dict) -> dict
```

Nothing else. No `reset()`, no `predict()`, no `__call__`. Duck-typing is fine; you do not need to inherit from any base class (though `policy_client.base_policy.BasePolicy` is available if you want type-checking).

Minimal adapter template:

```python
# src/my_policy/adapter.py
import numpy as np
import torch
import os
from .model import MyModel

class MyPolicyAdapter:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = MyModel(...).to(self.device).eval()

        # checkpoint_path is a DIRECTORY. Load your weights from inside it.
        weights = os.path.join(checkpoint_path, "model.pt")
        state = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state.get("model_state_dict", state), strict=False)

    @torch.inference_mode()
    def infer(self, obs: dict) -> dict:
        # Contract (see README §5):
        head_rgb = obs["head_rgb"]                 # (480, 640, 3) uint8
        hand_rgb = obs["hand_rgb"]                 # (480, 640, 3) uint8
        state    = obs["state"].astype(np.float32) # (8,) float32
        prompt   = obs.get("prompt", "")           # str

        actions = self._run(head_rgb, hand_rgb, state, prompt)  # your forward pass
        actions = np.asarray(actions, dtype=np.float32)         # (T, 11) float32
        assert actions.ndim == 2 and actions.shape[1] == 11 and actions.shape[0] >= 1
        return {"actions": actions}
```

**Common mistakes**:
- Returning `{"actions": (11,)}` (1-D). Must be `(T, 11)` — 2-D — even if `T=1`.
- Returning 32-dim actions padded with zeros. Must be exactly 11 dims in the order specified in README §5.
- Reading `obs["task_index"]` — there is no such key. The task description is in `obs["prompt"]` (string). If your model needs a task ID, map `prompt → task_id` yourself inside `infer`.
- Accepting `state` as 32-dim. It is **always 8-dim**.

---

## Step 3. Wire your adapter into `serve_hsr_policy_ws.py`

The base file uses the default OpenPI loader. For non-`PI0Pytorch` models, replace that block with your adapter.

**Original (relevant excerpt)** in `server/serve_hsr_policy_ws.py`:
```python
from openpi.policies.policy_config import create_trained_policy
...
policy = create_trained_policy(...)
server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata={})
server.serve_forever()
```

**Replace with** (example):
```python
from my_policy.adapter import MyPolicyAdapter
...
policy = MyPolicyAdapter(
    checkpoint_path=args.checkpoint_dir,
    device=args.pytorch_device or "cuda",
)
server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata={})
server.serve_forever()
```

Keep the rest of the file (argument parsing, logging, `WebsocketPolicyServer` instantiation) intact. The server only needs your `policy` to expose `.infer(obs)`.

> 🔍 You can dispatch between backends (OpenPI / LeRobot / custom) by reading an env var such as `POLICY_BACKEND` at the top of `serve_hsr_policy_ws.py` and selecting the loader accordingly.

---

## Step 4. Install dependencies in `server/Dockerfile`

Add your Python deps to the `server/Dockerfile` so the evaluator container has them.

Example diff:

```dockerfile
# server/Dockerfile
...
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        torch==2.5.1 \
        transformers==4.46.0 \
        your-custom-dep==1.2.3
```

Pin versions. The evaluation machine is offline/locked during scoring — if `pip install` fails on missing packages, your run fails.

> ⚠️ **Blackwell GPUs (RTX 5070 Ti)** require CUDA 12.8+ and PyTorch built against it. See [§8](#8-hardware-specific-notes-blackwell-huggingface-etc).

If you add dependencies via `uv`, refresh `uv.lock`:
```bash
uv lock
git add uv.lock pyproject.toml
```

---

## Step 5. Package your checkpoint as a directory

The harness validates `POLICY_CHECKPOINT_PATH` as a **directory**:

```bash
# RUN-DOCKER-CONTAINER.sh:
if [[ ! -d "${POLICY_CHECKPOINT_PATH}" ]]; then
    echo "[ERROR] POLICY_CHECKPOINT_PATH does not exist: ${POLICY_CHECKPOINT_PATH}"
    exit 1
fi
```

A bare `model.pt` file will fail immediately. Ship a directory like:

```
my_checkpoint/
├── model.pt                   # or model.safetensors
├── config.json
├── tokenizer/                 # if applicable
└── normalization_stats.json   # if applicable
```

Read files out of this directory in your adapter (`os.path.join(checkpoint_path, "model.pt")`).

---

## Step 6. Smoke test locally (mandatory)

```bash
cd airoa-evaluation-ICRA   # your fork, on your submission branch
export POLICY_CHECKPOINT_PATH=/abs/path/to/my_checkpoint
export POLICY_PYTORCH_DEVICE=cuda

# 1. Build & start containers (TEST_MODE=true by default, no real robot needed)
./RUN-DOCKER-CONTAINER.sh up

# 2. Check server loaded your model
./RUN-DOCKER-CONTAINER.sh logs policy_server
# Expect: "server listening on 0.0.0.0:8000"

# 3. Drive it with synthetic observations
./RUN-DOCKER-CONTAINER.sh shell
# Inside the container:
roslaunch hsr_policy_client hsr_policy_client.launch
# Expect in the policy_server logs: "Action executed." (repeatedly)

# 4. Stop
./RUN-DOCKER-CONTAINER.sh down
```

**What you are testing**:
- Your checkpoint loads (`Dockerfile` has the right deps, `adapter.__init__` works).
- Your `infer()` accepts the synthetic observations and returns the right shape.

If anything fails, the remote evaluation will fail the same way. **Do not submit without passing this.**

---

## Step 7. Write your submission note

Participants submit a **fork URL + branch + note** (a short description of how to run your submission). Write this note now while the build is fresh in your head.

The recommended approach is:

1. Copy [REPRODUCTION_STEPS.template.md](REPRODUCTION_STEPS.template.md) to your repo root as `REPRODUCTION_STEPS.md`.
2. Fill in every section (overview, prerequisites, run commands, env vars, edited files, special notes, expected output).
3. Commit it to your submission branch.
4. When you submit, paste a link to that file (or copy its contents) as your note.

This is not strictly required, but it is what gives the evaluator the best chance of reproducing your run on the first try. If you prefer a free-form note, make sure it still covers every field listed in the template.

---

## Step 8. Upload checkpoint to S3/Wasabi

```bash
aws --profile <your-profile> \
    --endpoint-url <url-given-by-organizers> \
    s3 sync /abs/path/to/my_checkpoint/ \
    s3://<your-bucket>/<your-path>/

# Verify
aws --profile <your-profile> \
    --endpoint-url <url> \
    s3 ls s3://<your-bucket>/<your-path>/
```

Confirm all required files are listed. Common miss: large `.safetensors` not fully uploaded due to retries.

---

## Step 9. Submit

Share three things with the organizers:

1. **Fork URL** — e.g. `https://github.com/<you>/airoa-evaluation-ICRA`
2. **Branch name** — e.g. `feat/my-policy`
3. **Note** — how to run your submission: the S3 checkpoint path, any env vars your server needs, any special setup (e.g. bundled tokenizer, non-default GPU memory requirements). The cleanest option is to link to the `REPRODUCTION_STEPS.md` you committed in Step 7.

Optional (recommended for precision): include the **commit hash** (`git rev-parse HEAD`) in your note, so the evaluator can check out the exact state you tested, even if you push more commits afterwards.

---

## 8. Hardware-specific notes (Blackwell, HuggingFace, etc.)

The evaluation machine has an **RTX 5070 Ti (Blackwell, sm_120)**. If your current wheels are for Ada/Ampere/Hopper only, they will silently fail at runtime with `no kernel image is available for execution on the device`.

### PyTorch / CUDA on Blackwell

You need CUDA 12.8+ and a Torch build that supports Blackwell. Two practical options:

**(a)** Use a CUDA 12.8+ base image in `server/Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
```
Then install torch wheels built against that CUDA:
```dockerfile
RUN uv pip install --system \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu128
```

**(b)** If `client/` needs GPU too (your model requires GPU in the client), same applies to `client/Dockerfile`. **This is the one case where editing `client/Dockerfile` is acceptable.** Document it in your submission note.

### HuggingFace tokens

If your model requires a gated HF model (e.g. PaliGemma tokenizer), **do not rely on `HF_TOKEN` being set on the evaluator**. Either:

- Bake the tokenizer / weights into your Docker image at build time (`COPY`), or
- Bundle them into your checkpoint directory,

so the container does not need internet access at run time.

### Long-running model compilation

If your model uses `torch.compile` or JAX JIT with heavy first-call cost (> 30 s), set the compile cache to a persistent path so repeat runs are fast, or disable compilation. The evaluator uses a 300 s timeout for first inference; exceeding it aborts the run.

---

## 9. Example fork layout

Here is an anonymous example of what a cleanly-adapted fork can look like. Use it as a structural reference — your actual filenames and directory names will differ.

```
airoa-evaluation-ICRA/
├── server/
│   ├── Dockerfile                      # custom: CUDA base, your deps
│   ├── entrypoint.sh                   # custom: dispatch between backends (optional)
│   ├── serve_hsr_policy_ws.py          # custom: loads your policy
│   ├── my_policy_loader.py             # thin wrapper that instantiates your model
│   └── my_controller.py                # optional: any high-level control logic
├── src/
│   ├── openpi/…                        # base stays
│   └── my_policy/                      # your new code lives here
├── packages/policy-client/              # unchanged (protocol)
├── docker-compose.yml                   # + any env vars your server reads
├── client/Dockerfile                    # only edited if hardware compat requires it
├── .env                                 # local defaults for your env vars
├── task_config.json                     # optional: task-specific config
├── controller_config.yaml               # optional: controller-specific config
├── tokenizer/<your_tokenizer>/          # bundled, so no external auth at runtime
└── REPRODUCTION_STEPS.md                # step-by-step, exactly what the evaluator runs
```

Key takeaways:

- **New code lives in `src/<your_policy>/`** — not scattered across `server/`.
- **`server/` contains thin wrappers** that call into `src/…` — easy to audit.
- **Config externalized to YAML/JSON + `.env`** — the evaluator doesn't need to read code to reproduce your run.
- **Dependencies (tokenizers, adapters) bundled**, no external auth required.
- **`REPRODUCTION_STEPS.md` is minimal and copy-pasteable** — this is the single biggest factor in whether your run reproduces on the first try.

---

## 10. Common pitfalls we keep seeing

1. **Submitting a separate repo.** You must submit from a fork of this repo. Standalone repos have no harness.
2. **Wrong method name.** `infer`, not `predict` or `__call__`.
3. **Wrong state shape.** 8-dim, not 32.
4. **Single-step action.** Return `(T, 11)`, not `(11,)`.
5. **Bare `.pt` file as `POLICY_CHECKPOINT_PATH`.** Must be a directory.
6. **Custom env var names** (`MODEL_CHECKPOINT`, `DEVICE`, `STATE_DIM`). Only the three in README §6 are read.
7. **Editing `client/Dockerfile` without stating why.** Fine for Blackwell CUDA, but call it out in your submission note.
8. **No smoke test.** If `./RUN-DOCKER-CONTAINER.sh up` + `roslaunch` does not print `Action executed.` on your laptop, the remote run will fail the same way.
9. **Checkpoint not fully uploaded to S3.** Always `aws s3 ls` after `sync` to confirm every file is there and the total size matches.
10. **Dependencies on host Python.** Everything must run inside the server container. The evaluator does not install anything on the host.

---

## 11. Where to go next

- ❓ Specific error? → [FAQ](FAQ.md) ([Japanese](FAQ_ja.md))
- 📝 Writing the reproduction doc? → [REPRODUCTION_STEPS.template.md](REPRODUCTION_STEPS.template.md)
- 🏷 Base repo reference? → [../README.md](../README.md)
