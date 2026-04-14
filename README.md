# airoa-evaluation-ICRA

Participant evaluation runtime for the ICRA 2026 AIRoA VLA Workshop Competition.

**For participants:** work from a fork of this repository, edit `server/` and `src/`, pass the local smoke test, then submit.

- 📖 [Integration Guide](docs/INTEGRATION_GUIDE.md) ([Japanese](docs/INTEGRATION_GUIDE_ja.md)) — step-by-step walkthrough with a real example
- ❓ [FAQ / Errata](docs/FAQ.md) ([Japanese](docs/FAQ_ja.md)) — common mistakes and clarifications
- 📝 [Reproduction template](docs/REPRODUCTION_STEPS.template.md) ([Japanese](docs/REPRODUCTION_STEPS.template_ja.md)) — recommended structure for your submission note

---

## 0. Which branch should I fork from?

This repository has two branches participants typically use:

| Branch | What you get | When to use |
|---|---|---|
| **`base`** *(this branch — recommended default)* | Minimal harness + a `ZeroPolicy` placeholder so smoke test passes immediately. No model-specific code. | **Most participants.** Fork from here regardless of framework (PyTorch, JAX, LeRobot, custom). |
| **`sample-openpi`** | Same harness + a worked OpenPI loader and the full `src/openpi/` source tree pre-integrated. | **Only if your policy is `PI0Pytorch` / OpenPI-compatible** and you want the loader as a reference. Otherwise the OpenPI code just gets in the way. |

```bash
# Fork airoa-org/airoa-evaluation-ICRA on GitHub, then:
git clone https://github.com/<you>/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout base                  # or `sample-openpi` if you want the OpenPI sample
git checkout -b feat/my-policy     # your submission branch
```

---

## 1. TL;DR

```bash
# 1. Fork airoa-org/airoa-evaluation-ICRA on GitHub (see §0 for which branch), then:
git clone https://github.com/<you>/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout base                  # the minimal starting point
git checkout -b feat/my-policy

# 2. Put your model code under src/<your_policy>/
# 3. Edit server/serve_hsr_policy_ws.py to load your policy (see Integration Guide §3)
# 4. Edit server/Dockerfile if you need extra dependencies
# 5. Package your checkpoint as a DIRECTORY, not a single .pt file

# 6. Smoke test locally (mandatory before submission):
export POLICY_CHECKPOINT_PATH=/abs/path/to/your_checkpoint_dir
./RUN-DOCKER-CONTAINER.sh up
./RUN-DOCKER-CONTAINER.sh shell           # open client container shell
# Inside the container:
roslaunch hsr_policy_client hsr_policy_client.launch
# Expect "Action executed." in the logs.

# 7. Upload checkpoint to the S3 bucket you were given, then share with the organizers:
#    - fork URL
#    - branch name
#    - a short note on how to run (S3 checkpoint path, env vars, any special setup)
#      → use docs/REPRODUCTION_STEPS.template.md as the structure for your note
```

If any step above is unclear, **read [docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) before asking**. Most questions we receive are already answered there.

---

## 2. Host Requirements

- Linux (verified on Ubuntu 24.04)
- Docker Engine + Docker Compose v2
- NVIDIA driver + NVIDIA Container Toolkit
- NVIDIA GPU with ≥ 16 GB VRAM

```bash
docker --version
docker compose version
nvidia-smi
```

Verified environment (the machine your submission will run on):

- OS: Ubuntu 24.04.3 LTS
- GPU: NVIDIA GeForce RTX 5070 Ti (Blackwell, 16 GB)
- NVIDIA driver: 580.126.09
- Docker: 29.0.1
- Docker Compose: v2.40.3

> ⚠️ Note: the evaluation GPU is **Blackwell (sm_120)**. PyTorch/CUDA wheels must support it. See [Integration Guide §8](docs/INTEGRATION_GUIDE.md#8-hardware-specific-notes-blackwell-huggingface-etc).

---

## 3. What You May Edit

| Path | Edit? | Purpose |
|---|---|---|
| `server/serve_hsr_policy_ws.py` | **Yes** — required | Instantiate your policy and pass it to `WebsocketPolicyServer` |
| `server/Dockerfile`, `server/entrypoint.sh` | **Yes** | Install your Python dependencies; tweak server startup |
| `src/<your_policy>/…` | **Yes** | Your model implementation |
| `packages/<your_package>/…` | Yes, if needed | Additional local packages |
| `pyproject.toml` / `uv.lock` | Yes, if you add deps | Workspace + lock |
| `docker-compose.yml` | **Discouraged but allowed** | Only to add env vars / volumes needed by your server. Document the change. |
| `client/Dockerfile` | **Discouraged but allowed** | Only for hardware compatibility (e.g. CUDA base image). Document the change. |
| `deploy/hsr_policy_client/**` | **No** | Pipeline-managed client logic |
| `runtime_core/**`, `packages/policy-client/**` | **No** | WebSocket server & client protocol |
| `RUN-DOCKER-CONTAINER.sh` | **No** | Harness entrypoint |

If you must change a "discouraged" file, mention it in your submission note (see [REPRODUCTION_STEPS.template.md](docs/REPRODUCTION_STEPS.template.md)) so the evaluators can reproduce your run.

---

## 4. Submission Requirements

Your submission **must** satisfy all of the following. The evaluator will check each one.

- [ ] Submitted from a fork of `airoa-org/airoa-evaluation-ICRA` (not a separate repo)
- [ ] `server/serve_hsr_policy_ws.py` instantiates your policy object that exposes `policy.infer(obs) -> dict`
- [ ] Your policy accepts the observation dict documented in [§5](#5-websocket-io-contract) and returns `{"actions": np.ndarray of shape (T, 11), dtype float32}`
- [ ] Checkpoint is packaged as a **directory** (not a bare `.pt` / `.safetensors` file)
- [ ] Dependencies installed via `server/Dockerfile` (don't rely on host Python)
- [ ] Local smoke test passes: `./RUN-DOCKER-CONTAINER.sh up` → `roslaunch hsr_policy_client hsr_policy_client.launch` prints `Action executed.`
- [ ] Checkpoint uploaded to the S3 bucket/path provided by organizers
- [ ] Submission note prepared (fork URL, branch, how to run) — [docs/REPRODUCTION_STEPS.template.md](docs/REPRODUCTION_STEPS.template.md) is a recommended structure

At submission time, share three things with the organizers:

1. **Fork URL** (e.g. `https://github.com/<you>/airoa-evaluation-ICRA`)
2. **Branch name** (e.g. `feat/my-policy`)
3. **Note** — how to run it: S3 checkpoint path, required env vars, any special setup. Using [docs/REPRODUCTION_STEPS.template.md](docs/REPRODUCTION_STEPS.template.md) as the structure is strongly recommended, since it already includes every field the evaluator needs.

> 💡 Committing a `REPRODUCTION_STEPS.md` (based on the template) to your branch and pasting its link as your note is the cleanest option. Pinning the commit hash there also helps the evaluator reproduce the exact state.

---

## 5. WebSocket I/O Contract

The server wraps your policy object and exposes a single method over WebSocket:

```python
policy.infer(obs) -> dict
```

**Observation dict** (what the client sends, already deserialized into numpy):

```python
{
  "head_rgb": np.ndarray,  # (480, 640, 3) uint8
  "hand_rgb": np.ndarray,  # (480, 640, 3) uint8
  "state":    np.ndarray,  # (8,)          float32
  "prompt":   str,
}
```

`state` order (8 dims):
```
[arm_lift_joint, arm_flex_joint, arm_roll_joint,
 wrist_flex_joint, wrist_roll_joint, gripper,
 head_pan_joint, head_tilt_joint]
```

**Action dict** (what your policy returns):

```python
{"actions": np.ndarray}   # shape (T, 11), dtype float32, T >= 1, all finite
```

`actions` order (11 dims):
```
[arm_lift_joint, arm_flex_joint, arm_roll_joint,
 wrist_flex_joint, wrist_roll_joint, gripper,
 head_pan_joint, head_tilt_joint,
 base_x, base_y, base_t]
```

Common mistakes: returning a 1-D `(11,)` array, 32-dim state, calling the method `predict()` instead of `infer()`. See [FAQ §3](docs/FAQ.md).

---

## 6. Environment Variables Consumed by the Pipeline

Only these are read by `server/entrypoint.sh`. **Other names are silently ignored** — it's a common source of confusion.

| Variable | Required? | Purpose |
|---|---|---|
| `POLICY_CHECKPOINT_PATH` | Yes (in practice) | Absolute path to your checkpoint **directory** on the host (mounted as `/policy_checkpoint` in the container; the `ZeroPolicy` placeholder works without it but every real policy needs it). |
| `POLICY_MODULE` | Optional | Import path of your policy class in `'module:Class'` form, e.g. `'my_policy.adapter:MyPolicyAdapter'`. If unset, the placeholder `ZeroPolicy` is used. |
| `POLICY_PYTORCH_DEVICE` | Optional | E.g. `cuda`. Currently informational; pass it through to your policy via your own server code if you need it. |

> The `sample-openpi` branch additionally requires `POLICY_CONFIG_NAME` (the OpenPI config name, e.g. `pi05_hsr`) because that branch's `serve_hsr_policy_ws.py` is hard-wired to the OpenPI loader.

---

## 7. Test Flow

```bash
export POLICY_CHECKPOINT_PATH=/abs/path/to/checkpoint_dir
./RUN-DOCKER-CONTAINER.sh up               # builds & starts containers (uses TEST_MODE=true)
./RUN-DOCKER-CONTAINER.sh shell            # opens client container shell
roslaunch hsr_policy_client hsr_policy_client.launch    # sends synthetic observations
./RUN-DOCKER-CONTAINER.sh logs policy_server
./RUN-DOCKER-CONTAINER.sh logs hsr_client
./RUN-DOCKER-CONTAINER.sh down
```

In `test_mode=true` (the default), the client generates synthetic random `head_rgb`, `hand_rgb`, `state`, and sends them to the server in a loop. A healthy server prints `Action executed.` for each inference.

**If the smoke test fails on your laptop, the remote evaluation will fail the same way.** Do not skip this step.

---

## 8. Getting Help

1. Read [docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) / [_ja](docs/INTEGRATION_GUIDE_ja.md) end to end.
2. Check [docs/FAQ.md](docs/FAQ.md) / [_ja](docs/FAQ_ja.md).
3. Run the smoke test locally and capture full logs before asking.
4. Contact organizers on the designated channel with: fork URL + branch + commit hash + full `docker compose logs` output. (For bug reports, the commit hash matters since branches move.)

---

## License

See [LICENSE](LICENSE).
