# airoa-evaluation-ICRA

## To the person responsible for AIROA deployment
### Design choices

We use `action_mode=absolute` in our final submission. Among the action parameterizations we tested, the absolute formulation achieved the lowest training loss, so we selected it as our final setting.
During local verification, we found that the original deployment-side `hsr_policy.py` did not fully support absolute-style execution out of the box. In particular, modes such as `absolute_arm_head_relative_gripper_base` are not handled correctly by the original `act()` implementation, because the arm/head commands are always applied additively to the current state.
To make inference consistent with training, we modified the action restoration/execution logic in `act()` to branch based on the action mode. When the mode is `absolute`, the predicted action is used directly. Relative components, when present, continue to follow the original relative update rule.

We verified this implementation locally before submission.

### Reproduction / Inference
First, You need to confirm the checkpoint is under the airoa-evaluation-ICRA directory.

```bash
export POLICY_CHECKPOINT_PATH=./pi05_hsr_lora_horizon8_absolute_GTrue_curated_relocate/8000
export POLICY_PYTORCH_DEVICE=cuda
export POLICY_CONFIG_NAME=pi05_hsr_lora_horizon8_absolute_GTrue_curated_relocate
./RUN-DOCKER-CONTAINER.sh up
./RUN-DOCKER-CONTAINER.sh shell
roslaunch hsr_policy_client hsr_policy_client.launch
```


Participant evaluation runtime for ICRA 2026 VLA Workshop Competition.

## 1. Editable Scope

Main implementation targets:

- `server/`
- `src/`

## 2. Host Requirements

- Linux
- Docker Engine
- Docker Compose v2
- NVIDIA driver
- NVIDIA Container Toolkit

```bash
docker --version
docker compose version
nvidia-smi
```

Verified environment (2026-02-20):

- OS: Ubuntu 24.04.3 LTS
- GPU: NVIDIA GeForce RTX 5070 Ti
- NVIDIA driver: 580.126.09
- Docker: 29.0.1
- Docker Compose: v2.40.3

## 3. Implementation Workflow

1. Create a feature branch from the prepared base branch.
2. Add your model repository under `src/`.
3. Update `server/serve_hsr_policy_ws.py` (currently the OpenPI version) and server-side dependencies for your model runtime.
4. Add an adapter in `src/` or `server/`.

Adapter definition:
An adapter is a thin conversion layer that maps between the HSR client contract and your model's native
input/output format. For required fields and shapes, just follow the `WebSocket I/O Contract` section below.

5. Run the test flow and confirm `Action executed.` appears in logs.
6. Submit your branch.

## 4. Required Environment Variables

```bash
export POLICY_CHECKPOINT_PATH=/abs/path/to/checkpoint_dir
```

## 5. Optional Policy-Specific Variables

Set policy-specific variables only when your server implementation requires them.

OpenPI example:

```bash
export POLICY_CONFIG_NAME=<openpi_config_name>
export POLICY_PYTORCH_DEVICE=cuda
```

## 6. Test Flow

Start containers:

```bash
./RUN-DOCKER-CONTAINER.sh up
```

Enter client shell:

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

Run launch inside the container:

```bash
roslaunch hsr_policy_client hsr_policy_client.launch
```

By default, `test_mode` is `true`. In this mode, the client uses synthetic random observations
(`head_rgb`, `hand_rgb`, and `state`) in an infinite loop and prints language/action logs.

## 7. Logs and Stop

```bash
./RUN-DOCKER-CONTAINER.sh logs policy_server
./RUN-DOCKER-CONTAINER.sh down
```

## 8. WebSocket I/O Contract

Inference request fields:

- `head_rgb`: image array `(H, W, 3)` (current HSR dataset profile: `(480, 640, 3)`)
- `hand_rgb`: image array `(H, W, 3)` (current HSR dataset profile: `(480, 640, 3)`)
- `state`: `(8,)`
- `prompt`: `str`

Inference response field:

- `actions` with shape `(T, 11)`, `T >= 1`

Action order:

- `[arm_lift_joint, arm_flex_joint, arm_roll_joint, wrist_flex_joint, wrist_roll_joint, gripper, head_pan_joint, head_tilt_joint, base_x, base_y, base_t]`

Value requirement:

- finite numeric values only
