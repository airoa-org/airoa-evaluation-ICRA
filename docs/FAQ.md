# FAQ / Errata

> Consolidated clarifications to the public `airoa-org/airoa-evaluation-ICRA` README, distributed to all participating teams. This does not change the evaluation protocol; it restates points that multiple teams have asked about, plus fixes for recurring submission mistakes.

Authoritative references in the repo:
- [README.md](../README.md) §3 (What you may edit), §5 (WebSocket I/O Contract), §6 (env vars)
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) — step-by-step
- `server/serve_hsr_policy_ws.py`, `server/entrypoint.sh`, `server/Dockerfile`
- `RUN-DOCKER-CONTAINER.sh`
- `deploy/hsr_policy_client/launch/hsr_policy_client.launch`

---

## 0. Which branch should I fork from?

Two branches are intended for participants. Pick the one that matches your stack — forking the wrong one is the single most common reason teams spend days "stripping the harness" before they can start.

| Branch | What it ships | Pick this if … |
|---|---|---|
| **`base`** *(default starting point)* | Minimal harness + a `ZeroPolicy` placeholder. `src/` is empty. | You're using PyTorch / JAX / LeRobot / your own framework — i.e. **most participants**. |
| **`sample-openpi`** | Same harness plus the OpenPI loader pre-wired in `serve_hsr_policy_ws.py` and the full `src/openpi/` source tree. Requires `POLICY_CONFIG_NAME`. | You're integrating a `PI0Pytorch`-compatible model and want the OpenPI loader as a worked example. |

If you don't actively want OpenPI, **fork from `base`**. The `sample-openpi` branch contains ~14k lines of OpenPI code that you would otherwise have to delete.

```bash
git clone https://github.com/<you>/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout base                  # or `sample-openpi`
git checkout -b feat/my-policy
```

---

## 1. Scope of what you may edit

Submit from a fork of `airoa-org/airoa-evaluation-ICRA` (see §0 for which branch). The **core** editable scope is:

- `server/` — including `serve_hsr_policy_ws.py`, `Dockerfile`, `entrypoint.sh`
- `src/` — your model/policy code

Additional paths you may edit when **genuinely necessary** (not just for convenience):

- `docker-compose.yml` — to add env vars or volumes your server needs
- `client/Dockerfile` — **only** for hardware compatibility (e.g. CUDA base image for Blackwell)
- `packages/<your_package>/` — additional local packages
- `pyproject.toml`, `uv.lock` — add dependencies

Paths that must **stay unchanged**:

- `runtime_core/` — WebSocket server protocol
- `packages/policy-client/` — WebSocket client protocol
- `deploy/hsr_policy_client/` — ROS client implementation
- `RUN-DOCKER-CONTAINER.sh` — harness entrypoint

**A standalone repository that only contains your model cannot be evaluated on its own** — the harness scripts live only in the base repository. See §6 for what happens if you try.

If you modify an "additional" path above, explain why in your submission note (see [REPRODUCTION_STEPS.template.md](REPRODUCTION_STEPS.template.md)) so evaluators can audit.

---

## 2. PyTorch is supported

The pipeline is not JAX-only. Two paths for PyTorch models:

**(a) Default `openpi` loader.** `serve_hsr_policy_ws.py` calls `openpi.policies.policy_config.create_trained_policy(...)`, which auto-detects PyTorch by the presence of `model.safetensors` in the checkpoint directory. This path works only for the `PI0Pytorch` architecture under `src/openpi/models_pytorch/`.

**(b) Custom adapter (recommended for non-`PI0Pytorch` models).** Edit `server/serve_hsr_policy_ws.py` to instantiate your own policy class and pass it to `WebsocketPolicyServer`. The server requires only:

```python
policy.infer(obs: dict) -> dict
```

Duck typing is sufficient; inheriting from `policy_client.base_policy.BasePolicy` is optional. See [INTEGRATION_GUIDE §2-3](INTEGRATION_GUIDE.md#step-2-write-the-adapter).

---

## 3. WebSocket I/O Contract (restatement of README §5)

### 3.1 Observation dict passed to `policy.infer(obs)`

```python
{
  "head_rgb": np.ndarray,  # shape (480, 640, 3),  dtype uint8
  "hand_rgb": np.ndarray,  # shape (480, 640, 3),  dtype uint8
  "state":    np.ndarray,  # shape (8,),           dtype float32
  "prompt":   str,
}
```

`state` is **8-dimensional** in the following order:

```
[arm_lift_joint, arm_flex_joint, arm_roll_joint,
 wrist_flex_joint, wrist_roll_joint, gripper,
 head_pan_joint, head_tilt_joint]
```

Transport is msgpack-numpy over WebSocket, so arrays arrive already deserialized as `np.ndarray`.

### 3.2 Action dict returned from `policy.infer(obs)`

```python
{"actions": np.ndarray}   # shape (T, 11), dtype float32, T >= 1, all finite
```

Action order (11 dims):

```
[arm_lift_joint, arm_flex_joint, arm_roll_joint,
 wrist_flex_joint, wrist_roll_joint, gripper,
 head_pan_joint, head_tilt_joint,
 base_x, base_y, base_t]
```

Any `T >= 1` is accepted. A single-step output is valid **only** when shaped `(1, 11)` (2-D array), not `(11,)`.

### 3.3 Episode boundaries

The server does **not** call `reset()` on the policy. If you need per-episode state, detect a new prompt (or an idle gap) inside `infer()` and reset internally.

### 3.4 What `obs` does NOT contain

- No `task_index` / `task_id` — infer task from `prompt` if needed.
- No `episode_step` / `timestep` — track internally if needed.
- No depth image, no point cloud (camera RGB only).

---

## 4. Running the pipeline

### 4.1 `POLICY_CHECKPOINT_PATH` must be a **directory**

`RUN-DOCKER-CONTAINER.sh` validates:

```bash
if [[ ! -d "${POLICY_CHECKPOINT_PATH}" ]]; then
    echo "[ERROR] POLICY_CHECKPOINT_PATH does not exist: ${POLICY_CHECKPOINT_PATH}"
    exit 1
fi
```

Passing a bare `.pt` / `.safetensors` file fails immediately. Package your checkpoint as a directory (and load weight files from inside it in your adapter).

### 4.2 Environment variables actually read by `server/entrypoint.sh`

| Variable | Purpose |
|---|---|
| `POLICY_CHECKPOINT_PATH` | Directory passed to the server as `--checkpoint-dir` |
| `POLICY_PYTORCH_DEVICE` | Optional; passed as `--pytorch-device` (e.g. `cuda`) |
| `POLICY_CONFIG_NAME` | Optional; openpi config name for the default loader |

Any other variable names (e.g. `MODEL_CHECKPOINT`, `DEVICE`, `STATE_DIM`, `ACTION_DIM`) are **not** read by the pipeline and have no effect.

If your server implementation needs additional env vars (e.g. `POLICY_BACKEND=lerobot`), add them to **your** `docker-compose.yml` (and your `.env` for local smoke tests), then list them in your submission note.

### 4.3 Which container loads the checkpoint

`docker compose` starts two containers:

- `airoa_policy_server` — loads your model from `POLICY_CHECKPOINT_PATH` and runs the WebSocket server
- `airoa_hsr_client` — ROS node that collects observations and executes actions

The checkpoint is loaded by the **server**, not the client. `roslaunch hsr_policy_client hsr_policy_client.launch` only accepts the args declared in `deploy/hsr_policy_client/launch/hsr_policy_client.launch` (`policy_server_host`, `policy_server_port`, `test_mode`, etc.). Arbitrary args like `checkpoint:=…` or `device:=…` are silently dropped by ROS.

### 4.4 Which Dockerfile is built

`./RUN-DOCKER-CONTAINER.sh up` runs `docker compose up --build`, which builds `server/Dockerfile` (and `client/Dockerfile`) from **your fork of the base repo**. A Dockerfile living in a separate repository is never built. Install your model's Python dependencies by editing `server/Dockerfile` in your fork.

---

## 5. Local smoke test before uploading to S3

```bash
cd airoa-evaluation-ICRA   # your fork, on your submission branch
export POLICY_CHECKPOINT_PATH=/abs/path/to/your_checkpoint_dir
export POLICY_PYTORCH_DEVICE=cuda
./RUN-DOCKER-CONTAINER.sh up
./RUN-DOCKER-CONTAINER.sh shell
roslaunch hsr_policy_client hsr_policy_client.launch    # test_mode=true by default
./RUN-DOCKER-CONTAINER.sh logs policy_server            # expect "Action executed."
```

In `test_mode=true` the client sends synthetic observations matching the shapes in §3.1. If your adapter errors on shapes, keys, or dtypes here, the remote evaluation will fail the same way.

---

## 6. Common mistakes we keep seeing

### 6.1 Submitting from a standalone repository

**Symptom.** The evaluator reports `./RUN-DOCKER-CONTAINER.sh: No such file or directory` or cannot build.

**Cause.** You submitted from a repo like `<your-org>/MyPolicy`, which only contains model code. The harness (`RUN-DOCKER-CONTAINER.sh`, `docker-compose.yml`, `runtime_core/`, `packages/`) lives only in a fork of `airoa-org/airoa-evaluation-ICRA`.

**Fix.** Fork this repo, put your model code under `src/`, edit `server/`, and submit from that fork.

### 6.2 Wrong method name (`predict`, `__call__`)

**Symptom.** Server loads the checkpoint but fails on the first inference call with `AttributeError: 'MyAdapter' object has no attribute 'infer'`.

**Fix.** Rename your method to `infer`. No other method is called.

### 6.3 Wrong state shape (32 dims instead of 8)

**Symptom.** `ValueError: could not broadcast` or shape mismatch inside your adapter.

**Fix.** Expect `obs["state"]` to be `(8,)` float32. If your training pipeline expects 32 dims, project or pad/trim inside `infer()`.

### 6.4 Single-step action `(11,)` instead of `(T, 11)`

**Symptom.** Server error after infer returns, or client refuses to execute.

**Fix.** Return a 2-D array. The minimum valid shape is `(1, 11)`, not `(11,)`.

### 6.5 Custom env var names have no effect

**Symptom.** You set `MODEL_CHECKPOINT=...` and `DEVICE=cuda` but server can't find the checkpoint.

**Fix.** Only `POLICY_CHECKPOINT_PATH`, `POLICY_PYTORCH_DEVICE`, `POLICY_CONFIG_NAME` are read. Rename your env vars.

### 6.6 Passing `.pt` as `POLICY_CHECKPOINT_PATH`

**Symptom.** `[ERROR] POLICY_CHECKPOINT_PATH does not exist: /path/to/model.pt`.

**Fix.** Package as a directory. Read the file from inside the directory in your adapter.

### 6.7 Forgetting to edit `server/Dockerfile`

**Symptom.** `ImportError` / `ModuleNotFoundError` at server startup for your own deps.

**Fix.** Install everything inside `server/Dockerfile`. The evaluation machine is treated as immutable — anything not in the Docker image is not available.

### 6.8 No smoke test before submission

**Symptom.** The evaluator runs your submission and gets the same error you would have seen in `./RUN-DOCKER-CONTAINER.sh up` + `roslaunch`.

**Fix.** Do the smoke test first. Without it, you are submitting code you have not verified runs.

### 6.9 Gated HuggingFace model requires `HF_TOKEN`

**Symptom.** `OSError: You are trying to access a gated repo`.

**Fix.** Do not rely on `HF_TOKEN` being set. Either bake the files into the Docker image at build time, or bundle them into the checkpoint directory. See [INTEGRATION_GUIDE §8](INTEGRATION_GUIDE.md#huggingface-tokens).

---

## 7. Hardware-specific: Blackwell (RTX 5070 Ti)

The evaluation GPU is Blackwell (compute capability 12.0, sm_120). CUDA 12.8+ and matching Torch wheels are required; older wheels fail with `no kernel image is available for execution on the device`.

- Set `server/Dockerfile` (and `client/Dockerfile` if your model runs in the client) base to `nvidia/cuda:12.8.1-*`.
- Install `torch` with `--index-url https://download.pytorch.org/whl/cu128`.
- See [INTEGRATION_GUIDE §8](INTEGRATION_GUIDE.md#pytorch--cuda-on-blackwell) for a working example.

---

## 8. Pre-submission checklist

Before you submit, confirm every item:

- [ ] Your submission is a branch of a fork of `airoa-org/airoa-evaluation-ICRA` — not a separate repo.
- [ ] `server/serve_hsr_policy_ws.py` instantiates your policy and passes it to `WebsocketPolicyServer`.
- [ ] Your policy object exposes `infer(obs: dict) -> dict`.
- [ ] `infer` accepts `head_rgb`, `hand_rgb`, `state` (shape `(8,)`), `prompt` (str).
- [ ] `infer` returns `{"actions": np.ndarray}` with `actions.shape == (T, 11)`, `dtype=float32`, `T >= 1`, all finite.
- [ ] Checkpoint is a **directory**; your adapter reads files from inside it.
- [ ] Dependencies pinned in `server/Dockerfile`; no reliance on host Python.
- [ ] Blackwell-compatible CUDA + Torch wheels if your model runs on GPU.
- [ ] Local smoke test passed: `./RUN-DOCKER-CONTAINER.sh up` → `roslaunch hsr_policy_client hsr_policy_client.launch` → `Action executed.`.
- [ ] Checkpoint fully uploaded to S3 (verified with `aws s3 ls` and total size check).
- [ ] Submission note prepared — covers how to run (S3 checkpoint path, env vars, special setup). [REPRODUCTION_STEPS.template.md](REPRODUCTION_STEPS.template.md) is the recommended structure; committing a filled-in `REPRODUCTION_STEPS.md` and linking it as your note is the cleanest option.
- [ ] Submission message / form contains: **fork URL**, **branch name**, **note**. (Commit hash is optional but recommended in the note for precision.)

---

## 9. Submission workflow (summary)

1. Fork `airoa-org/airoa-evaluation-ICRA`.
2. Place your model code under `src/…` in the fork.
3. Edit `server/serve_hsr_policy_ws.py` for a non-`PI0Pytorch` loader.
4. Edit `server/Dockerfile` / `server/entrypoint.sh` for extra deps.
5. Package your checkpoint as a directory.
6. Run the local smoke test in §5.
7. Write your submission note (recommended: fill in [REPRODUCTION_STEPS.template.md](REPRODUCTION_STEPS.template.md) and commit it as `REPRODUCTION_STEPS.md`).
8. Upload the checkpoint to S3.
9. Submit **fork URL + branch + note** to the organizers.

---

If you have further questions, please reply to the organizers' channel so that the FAQ can be extended for everyone.
