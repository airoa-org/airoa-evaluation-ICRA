#!/usr/bin/env python3
"""Multi-policy competition evaluator with GUI orchestration."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    import tkinter as tk
    from tkinter import messagebox
except ModuleNotFoundError:
    tk = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]


ROS_SETUP = "source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash"
INSTRUCTION_SERVICE = "/hsr_policy_client/update_instruction"
MOTION_SERVICE = "/hsr_policy_client/set_motion_enabled"
ACTION_OUTPUT_SERVICE = "/hsr_policy_client/has_action_output"
RESET_ACTION_OUTPUT_SERVICE = "/hsr_policy_client/reset_action_output_flag"
ROSBAG_TOPICS = [
    "/tf",
    "/tf_static",
    "/hsrb/command_velocity",
    "/hsrb/omni_base_controller/command",
    "/hsrb/arm_trajectory_controller/command",
    "/hsrb/gripper_controller/command",
    "/hsrb/head_trajectory_controller/command",
    "/hsrb/gripper_controller/grasp/goal",
    "/hsrb/gripper_controller/follow_joint_trajectory/goal",
    "/hsrb/hand_camera/image_raw/compressed",
    "/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed",
    "/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw",
    "/hsrb/base_scan",
    "/hsrb/odom",
    "/hsrb/wrist_wrench/raw",
    "/hsrb/joint_states",
    "/hsrb/servo_states",
    "/hsrb/joy",
    "/control_mode",
    "/hsr_policy_client/original_action_chunk",
]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "unknown"


def _iso_now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _safe_float(value: float) -> str:
    return f"{value:.4f}"


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        text=True,
        capture_output=capture_output,
    )
    if check and proc.returncode != 0:
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            "Command failed:\n"
            f"  {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"  exit={proc.returncode}\n"
            f"  stdout={stdout}\n"
            f"  stderr={stderr}"
        )
    return proc


def _detect_host_ipv4_candidates() -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def _push(ip: str) -> None:
        ip = ip.strip()
        if not ip or ip.startswith("127."):
            return
        if ip in seen:
            return
        seen.add(ip)
        candidates.append(ip)

    # Prefer `ip` command output order (similar to RUN-DOCKER-CONTAINER.sh fallback path).
    try:
        proc = _run(["ip", "-4", "-o", "addr", "show"], check=False)
    except FileNotFoundError:
        proc = None
    if proc and proc.returncode == 0:
        for line in (proc.stdout or "").splitlines():
            m = re.search(r"\binet\s+([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)/", line)
            if m:
                _push(m.group(1))

    # Fallback to ifconfig if needed.
    if not candidates:
        try:
            proc = _run(["ifconfig"], check=False)
        except FileNotFoundError:
            proc = None
        if proc and proc.returncode == 0:
            for line in (proc.stdout or "").splitlines():
                m = re.search(r"\binet\s(?:addr:)?([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)\b", line)
                if m:
                    _push(m.group(1))

    return candidates


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclasses.dataclass(frozen=True)
class Participant:
    name: str
    worktree_path: Path
    checkpoint_path: Path
    config_name: str
    env: dict[str, str]
    policy_cache_dir: Path
    hf_cache_dir: Path


@dataclasses.dataclass(frozen=True)
class PAConfig:
    name: str
    prompt: str


@dataclasses.dataclass(frozen=True)
class SHTConfig:
    name: str
    pas: list[PAConfig]


@dataclasses.dataclass
class ServerRuntime:
    image_tag: str
    container_name: str
    port: int
    running: bool = False


@dataclasses.dataclass
class PARecord:
    policy: str
    sht: str
    repeat_index: int
    pa: str
    prompt: str
    success: bool
    elapsed_sec: float
    recorded_at: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class SHTRecord:
    policy: str
    sht: str
    repeat_index: int
    success: bool
    pa_successes: int
    pa_total: int
    recorded_at: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class DockerManager:
    def __init__(
        self,
        *,
        repo_root: Path,
        participants: list[Participant],
        base_port: int,
        max_running: int,
        result_root: Path,
        client_container_name: str,
        gpu_enabled: bool,
        client_env_overrides: dict[str, str] | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.participants = participants
        self.base_port = int(base_port)
        self.max_running = max(1, int(max_running))
        self.result_root = result_root
        self.client_container_name = client_container_name
        self.gpu_enabled = bool(gpu_enabled)
        self.client_env_overrides = dict(client_env_overrides or {})

        self.runtimes: dict[int, ServerRuntime] = {}
        self._running_indices: set[int] = set()
        self._pending_indices: list[int] = list(range(len(participants)))

    def ensure_client_up(self) -> None:
        names = _run(["docker", "ps", "--format", "{{.Names}}"]).stdout.splitlines()
        if self.client_container_name in names:
            if self.client_env_overrides:
                print(
                    f"[WARN] {self.client_container_name} is already running. "
                    "CLI ROS env overrides are not applied to existing container. "
                    "Run `docker rm -f airoa_hsr_client` (or compose down) and retry."
                )
            return

        env = os.environ.copy()
        env["EVAL_RESULT_DIR"] = str(self.result_root.resolve())
        env.update(self.client_env_overrides)
        self.result_root.mkdir(parents=True, exist_ok=True)
        _run(
            ["docker", "compose", "up", "--build", "-d", "--no-deps", "hsr_client"],
            cwd=self.repo_root,
            env=env,
            capture_output=False,
        )

    def prewarm_servers(self) -> None:
        self._fill_pool()

    def get_runtime(self, participant_idx: int) -> ServerRuntime:
        if participant_idx not in self.runtimes:
            self.runtimes[participant_idx] = self._build_runtime(participant_idx)
        return self.runtimes[participant_idx]

    def ensure_server_running(self, participant_idx: int) -> ServerRuntime:
        rt = self.get_runtime(participant_idx)
        if rt.running:
            return rt
        if len(self._running_indices) >= self.max_running:
            raise RuntimeError(
                "No server capacity left. Complete a running policy first or increase --max-running."
            )
        self._start_server(participant_idx)
        if participant_idx in self._pending_indices:
            self._pending_indices.remove(participant_idx)
        return self.runtimes[participant_idx]

    def complete_policy(self, participant_idx: int) -> None:
        self.stop_server(participant_idx)
        self._fill_pool()

    def stop_server(self, participant_idx: int) -> None:
        rt = self.runtimes.get(participant_idx)
        if rt is None or not rt.running:
            return
        _run(["docker", "rm", "-f", rt.container_name], check=False)
        rt.running = False
        self._running_indices.discard(participant_idx)

    def stop_all_servers(self) -> None:
        for idx in sorted(list(self._running_indices)):
            self.stop_server(idx)

    def _fill_pool(self) -> None:
        while len(self._running_indices) < self.max_running and self._pending_indices:
            idx = self._pending_indices[0]
            self._start_server(idx)
            self._pending_indices.pop(0)

    def _build_runtime(self, participant_idx: int) -> ServerRuntime:
        participant = self.participants[participant_idx]
        slug = _slugify(participant.name)
        port = self.base_port + participant_idx
        image_tag = f"airoa-policy-eval-{slug}:{participant_idx}"
        container_name = f"airoa_policy_{slug}_{participant_idx}"
        return ServerRuntime(image_tag=image_tag, container_name=container_name, port=port)

    def _start_server(self, participant_idx: int) -> None:
        participant = self.participants[participant_idx]
        runtime = self.get_runtime(participant_idx)

        if not participant.worktree_path.exists():
            raise FileNotFoundError(f"worktree_path not found: {participant.worktree_path}")
        if not participant.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint_path not found: {participant.checkpoint_path}")
        participant.policy_cache_dir.mkdir(parents=True, exist_ok=True)
        participant.hf_cache_dir.mkdir(parents=True, exist_ok=True)

        dockerfile = participant.worktree_path / "server" / "Dockerfile"
        if not dockerfile.exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile}")

        _run(
            [
                "docker",
                "build",
                "-t",
                runtime.image_tag,
                "-f",
                str(dockerfile),
                str(participant.worktree_path),
            ],
            capture_output=False,
        )

        _run(["docker", "rm", "-f", runtime.container_name], check=False)

        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            runtime.container_name,
            "--network",
            "host",
            "--restart",
            "unless-stopped",
            "-e",
            "POLICY_SERVER_HOST=0.0.0.0",
            "-e",
            f"POLICY_SERVER_PORT={runtime.port}",
            "-e",
            "POLICY_CHECKPOINT_DIR=/policy_checkpoint",
            "-e",
            f"POLICY_CONFIG_NAME={participant.config_name}",
            "-e",
            "POLICY_DATA_HOME=/policy_assets",
            "-e",
            "OPENPI_DATA_HOME=/policy_assets",
            "-e",
            "HF_HOME=/policy_hf_cache",
        ]

        for key, value in sorted(participant.env.items()):
            cmd.extend(["-e", f"{key}={value}"])

        cmd.extend(
            [
                "-v",
                f"{participant.checkpoint_path}:/policy_checkpoint:ro",
                "-v",
                f"{participant.policy_cache_dir}:/policy_assets",
                "-v",
                f"{participant.hf_cache_dir}:/policy_hf_cache",
            ]
        )

        if self.gpu_enabled:
            cmd.extend(["--gpus", "all"])

        cmd.append(runtime.image_tag)
        _run(cmd, capture_output=False)
        runtime.running = True
        self._running_indices.add(participant_idx)


class RosLaunchController:
    def __init__(
        self,
        *,
        repo_root: Path,
        result_root: Path,
        client_container_name: str,
        test_mode: bool,
    ) -> None:
        self.repo_root = repo_root
        self.result_root = result_root
        self.client_container_name = client_container_name
        self.test_mode = bool(test_mode)
        self._proc: subprocess.Popen[Any] | None = None
        self._rosbag_proc: subprocess.Popen[Any] | None = None
        self._last_host_log_path: Path | None = None
        self._last_runtime_log_name: str | None = None
        self._host_log_candidates: list[Path] = []
        self._active_rosbag_host_path: Path | None = None
        self._active_rosbag_pattern: str | None = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(
        self,
        *,
        policy_name: str,
        policy_server_host: str,
        policy_server_port: int,
        trace_base_dir: str,
        config_name: str,
        initial_instruction: str,
    ) -> None:
        self.stop()

        runtime_log_name = f"{_slugify(policy_name)}_{_slugify(config_name)}.log"
        host_log_dir = self.result_root / "runtime_logs"
        host_log_dir.mkdir(parents=True, exist_ok=True)
        host_log_path = host_log_dir / runtime_log_name
        container_log_dir = "/root/eval_results/runtime_logs"
        container_log_path = f"{container_log_dir}/{runtime_log_name}"
        self._last_host_log_path = host_log_path
        self._last_runtime_log_name = runtime_log_name
        self._host_log_candidates = self._build_host_log_candidates(runtime_log_name)

        args = [
            f"test_mode:={'true' if self.test_mode else 'false'}",
            f"policy_server_host:={policy_server_host}",
            f"policy_server_port:={policy_server_port}",
            "save_exec_trace:=true",
            f"trace_base_dir:={shlex.quote(trace_base_dir)}",
            f"config_name:={shlex.quote(config_name)}",
            f"instruction:={shlex.quote(initial_instruction)}",
        ]
        roslaunch_core = "roslaunch hsr_policy_client hsr_policy_client.launch " + " ".join(args)
        # Route roslaunch output to:
        # 1) container main stdout so `docker logs airoa_hsr_client` shows it
        # 2) mounted runtime log file under /root/eval_results/runtime_logs
        roslaunch_cmd = (
            "set -o pipefail && "
            f"{ROS_SETUP} && "
            f"mkdir -p {shlex.quote(container_log_dir)} && "
            f"{roslaunch_core} 2>&1 | tee {shlex.quote(container_log_path)} > /proc/1/fd/1"
        )
        cmd = ["docker", "exec", self.client_container_name, "bash", "-lc", roslaunch_cmd]

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self.repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        time.sleep(1.5)
        if self._proc.poll() is not None:
            raise RuntimeError(
                "roslaunch exited early. Check "
                f"`docker logs {self.client_container_name}` or {host_log_path}"
            )

        self._wait_for_required_services(timeout_sec=90.0)
        self.set_motion_enabled(True)

    def set_instruction(self, instruction: str) -> None:
        payload = f"message: {json.dumps(instruction)}"
        cmd = (
            f"{ROS_SETUP} && rosservice call {INSTRUCTION_SERVICE} "
            f"{shlex.quote(payload)}"
        )
        proc = self._docker_bash(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to update instruction: {proc.stderr.strip()}")
        if "success: True" not in proc.stdout and "success: true" not in proc.stdout:
            raise RuntimeError(f"Instruction service returned unexpected output: {proc.stdout.strip()}")

    def set_motion_enabled(self, enabled: bool) -> None:
        mode = "enable" if enabled else "disable"
        payload = f"message: {json.dumps(mode)}"
        cmd = (
            f"{ROS_SETUP} && rosservice call {MOTION_SERVICE} "
            f"{shlex.quote(payload)}"
        )
        proc = self._docker_bash(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to set motion state ({mode}): {proc.stderr.strip()}")
        if "success: True" not in proc.stdout and "success: true" not in proc.stdout:
            raise RuntimeError(f"Motion service returned unexpected output: {proc.stdout.strip()}")

    def reset_action_output_flag(self) -> None:
        payload = "message: ''"
        cmd = (
            f"{ROS_SETUP} && rosservice call {RESET_ACTION_OUTPUT_SERVICE} "
            f"{shlex.quote(payload)}"
        )
        proc = self._docker_bash(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to reset action output flag: {proc.stderr.strip()}")
        if "success: True" not in proc.stdout and "success: true" not in proc.stdout:
            raise RuntimeError(f"Reset action output service returned unexpected output: {proc.stdout.strip()}")

    def has_action_output(self) -> bool:
        payload = "message: ''"
        cmd = (
            f"{ROS_SETUP} && rosservice call {ACTION_OUTPUT_SERVICE} "
            f"{shlex.quote(payload)}"
        )
        proc = self._docker_bash(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to query action output flag: {proc.stderr.strip()}")
        m = re.search(r"success:\s*(True|False|true|false)", proc.stdout or "")
        if not m:
            raise RuntimeError(f"Action output service returned unexpected output: {proc.stdout.strip()}")
        return m.group(1).lower() == "true"

    def start_pa_rosbag(
        self,
        *,
        policy_name: str,
        sht_name: str,
        repeat_index: int,
        pa_index: int,
        pa_name: str,
    ) -> Path:
        self.stop_pa_rosbag()

        policy_slug = _slugify(policy_name)
        sht_slug = _slugify(sht_name)
        bag_dir_rel = Path(policy_slug) / "rosbags" / sht_slug / f"eval{int(repeat_index):02d}"
        bag_stem = f"pa{int(pa_index) + 1:02d}"

        host_bag_dir = self.result_root / bag_dir_rel
        host_bag_dir.mkdir(parents=True, exist_ok=True)
        host_bag_path = host_bag_dir / f"{bag_stem}.bag"

        container_bag_dir = Path("/root/eval_results") / bag_dir_rel
        record_pattern = f"rosbag record -O {bag_stem}"
        topic_args = " ".join(shlex.quote(topic) for topic in ROSBAG_TOPICS)
        rosbag_cmd = (
            f"{ROS_SETUP} && "
            f"mkdir -p {shlex.quote(str(container_bag_dir))} && "
            f"cd {shlex.quote(str(container_bag_dir))} && "
            f"exec rosbag record -O {shlex.quote(bag_stem)} {topic_args}"
        )
        cmd = ["docker", "exec", self.client_container_name, "bash", "-lc", rosbag_cmd]

        self._rosbag_proc = subprocess.Popen(
            cmd,
            cwd=str(self.repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._active_rosbag_host_path = host_bag_path
        self._active_rosbag_pattern = record_pattern

        time.sleep(1.0)
        if self._rosbag_proc.poll() is not None:
            self._rosbag_proc = None
            self._active_rosbag_pattern = None
            active_path = self._active_rosbag_host_path
            self._active_rosbag_host_path = None
            raise RuntimeError(
                "rosbag record exited early. Check "
                f"`docker logs {self.client_container_name}` or {active_path}"
            )
        return host_bag_path

    def stop_pa_rosbag(self) -> Path | None:
        host_bag_path = self._active_rosbag_host_path
        record_pattern = self._active_rosbag_pattern

        if self._rosbag_proc is not None and self._rosbag_proc.poll() is None:
            try:
                self._rosbag_proc.send_signal(signal.SIGINT)
            except Exception:
                pass

        def _wait_for_container_rosbag_exit(timeout_sec: float) -> bool:
            if not record_pattern:
                return True
            deadline = time.time() + timeout_sec
            probe = f"pgrep -f {shlex.quote(record_pattern)} >/dev/null"
            while time.time() < deadline:
                proc = self._docker_bash(probe, check=False)
                if proc.returncode != 0:
                    return True
                time.sleep(0.2)
            return False

        if record_pattern:
            self._docker_bash(
                f"pkill -INT -f {shlex.quote(record_pattern)} || true",
                check=False,
            )
            if not _wait_for_container_rosbag_exit(timeout_sec=10.0):
                self._docker_bash(
                    f"pkill -TERM -f {shlex.quote(record_pattern)} || true",
                    check=False,
                )
                if not _wait_for_container_rosbag_exit(timeout_sec=5.0):
                    self._docker_bash(
                        f"pkill -KILL -f {shlex.quote(record_pattern)} || true",
                        check=False,
                    )
                    _wait_for_container_rosbag_exit(timeout_sec=2.0)

        if self._rosbag_proc is not None and self._rosbag_proc.poll() is None:
            try:
                self._rosbag_proc.wait(timeout=5)
            except Exception:
                try:
                    self._rosbag_proc.terminate()
                    self._rosbag_proc.wait(timeout=5)
                except Exception:
                    self._rosbag_proc.kill()

        self._rosbag_proc = None
        self._active_rosbag_pattern = None
        self._active_rosbag_host_path = None

        if host_bag_path is not None:
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if host_bag_path.exists():
                    break
                time.sleep(0.1)

        return host_bag_path

    def stop(self) -> None:
        self.stop_pa_rosbag()

        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.send_signal(signal.SIGINT)
                self._proc.wait(timeout=8)
            except Exception:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
                except Exception:
                    self._proc.kill()

        cleanup_cmd = (
            "pkill -f 'roslaunch hsr_policy_client hsr_policy_client.launch' || true; "
            "pkill -f '/hsr_policy_client/scripts/hsr_policy.py' || true; "
            "pkill -INT -f 'rosbag record -O' || true"
        )
        self._docker_bash(cleanup_cmd, check=False)

        self._proc = None

    def get_last_host_log_path(self) -> Path | None:
        if self._host_log_candidates:
            existing = [p for p in self._host_log_candidates if p.exists()]
            if existing:
                existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return existing[0]
            return self._host_log_candidates[0]
        return self._last_host_log_path

    def _build_host_log_candidates(self, runtime_log_name: str) -> list[Path]:
        candidates: list[Path] = []
        seen: set[str] = set()

        def _push(path: Path) -> None:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                return
            seen.add(key)
            candidates.append(path)

        _push(self.result_root / "runtime_logs" / runtime_log_name)
        results_root = self.repo_root / "results"
        _push(results_root / "runtime_logs" / runtime_log_name)

        if results_root.exists():
            for runtime_dir in results_root.glob("result_*/runtime_logs"):
                _push(runtime_dir / runtime_log_name)

        parent = self.result_root.parent
        if parent.exists() and parent != results_root:
            for runtime_dir in parent.glob("result_*/runtime_logs"):
                _push(runtime_dir / runtime_log_name)

        return candidates

    def _wait_for_required_services(self, timeout_sec: float) -> None:
        deadline = time.time() + timeout_sec
        probe = (
            f"{ROS_SETUP} && "
            f"rosservice list | grep -qx '{INSTRUCTION_SERVICE}' && "
            f"rosservice list | grep -qx '{MOTION_SERVICE}' && "
            f"rosservice list | grep -qx '{ACTION_OUTPUT_SERVICE}' && "
            f"rosservice list | grep -qx '{RESET_ACTION_OUTPUT_SERVICE}'"
        )
        while time.time() < deadline:
            proc = self._docker_bash(probe, check=False)
            if proc.returncode == 0:
                return
            time.sleep(1.0)
        raise TimeoutError(
            "Timed out waiting for required services: "
            f"{INSTRUCTION_SERVICE}, {MOTION_SERVICE}, {ACTION_OUTPUT_SERVICE}, {RESET_ACTION_OUTPUT_SERVICE}"
        )

    def _docker_bash(self, command: str, *, check: bool = False) -> subprocess.CompletedProcess[str]:
        return _run(
            ["docker", "exec", self.client_container_name, "bash", "-lc", command],
            check=check,
        )


def _compute_policy_summary(
    *,
    policy_name: str,
    pa_records: list[PARecord],
    sht_records: list[SHTRecord],
) -> dict[str, Any]:
    policy_pa = [r for r in pa_records if r.policy == policy_name]
    policy_sht = [r for r in sht_records if r.policy == policy_name]

    pa_total = len(policy_pa)
    pa_success = sum(1 for r in policy_pa if r.success)
    pa_success_rate = (pa_success / pa_total) if pa_total else 0.0

    sht_total = len(policy_sht)
    sht_success = sum(1 for r in policy_sht if r.success)
    sht_success_rate = (sht_success / sht_total) if sht_total else 0.0

    per_sht: dict[str, dict[str, Any]] = {}
    for row in policy_sht:
        bucket = per_sht.setdefault(
            row.sht,
            {
                "trials": 0,
                "successes": 0,
                "success_rate": 0.0,
                "pa_total": 0,
                "pa_successes": 0,
                "pa_breakdown": {},
            },
        )
        bucket["trials"] += 1
        bucket["successes"] += 1 if row.success else 0
        bucket["pa_total"] += row.pa_total
        bucket["pa_successes"] += row.pa_successes

    for row in policy_pa:
        bucket = per_sht.setdefault(
            row.sht,
            {
                "trials": 0,
                "successes": 0,
                "success_rate": 0.0,
                "pa_total": 0,
                "pa_successes": 0,
                "pa_breakdown": {},
            },
        )
        pa_breakdown = bucket.setdefault("pa_breakdown", {})
        pa_bucket = pa_breakdown.setdefault(
            row.pa,
            {
                "total": 0,
                "successes": 0,
                "success_rate": 0.0,
            },
        )
        pa_bucket["total"] += 1
        pa_bucket["successes"] += 1 if row.success else 0

    for bucket in per_sht.values():
        bucket["success_rate"] = (bucket["successes"] / bucket["trials"]) if bucket["trials"] else 0.0
        pa_breakdown = bucket.get("pa_breakdown", {})
        if not isinstance(pa_breakdown, dict):
            continue
        for pa_bucket in pa_breakdown.values():
            if not isinstance(pa_bucket, dict):
                continue
            total = int(pa_bucket.get("total", 0))
            successes = int(pa_bucket.get("successes", 0))
            pa_bucket["success_rate"] = (float(successes) / float(total)) if total > 0 else 0.0

    return {
        "policy": policy_name,
        "pa_total": pa_total,
        "pa_success": pa_success,
        "pa_success_rate": pa_success_rate,
        "sht_total": sht_total,
        "sht_success": sht_success,
        "sht_success_rate": sht_success_rate,
        "average_success_rate": sht_success_rate,
        "per_sht": per_sht,
        "updated_at": _iso_now(),
    }


def _write_sr_txt(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"policy: {summary['policy']}",
        f"pa_success: {summary['pa_success']}/{summary['pa_total']}",
        f"pa_success_rate: {_safe_float(summary['pa_success_rate'])}",
        f"sht_success: {summary['sht_success']}/{summary['sht_total']}",
        f"sht_success_rate: {_safe_float(summary['sht_success_rate'])}",
        f"average_success_rate: {_safe_float(summary['average_success_rate'])}",
        f"updated_at: {summary['updated_at']}",
    ]
    per_sht = summary.get("per_sht", {})
    if isinstance(per_sht, dict) and per_sht:
        lines.append("")
        lines.append("per_sht:")
        for sht_name in sorted(per_sht.keys()):
            bucket = per_sht.get(sht_name, {})
            if not isinstance(bucket, dict):
                continue
            trials = int(bucket.get("trials", 0))
            successes = int(bucket.get("successes", 0))
            success_rate = float(bucket.get("success_rate", 0.0))
            lines.append(
                f"- {sht_name}: "
                f"sht_success={successes}/{trials}, "
                f"sht_success_rate={_safe_float(success_rate)}"
            )
            pa_breakdown = bucket.get("pa_breakdown", {})
            if not isinstance(pa_breakdown, dict) or not pa_breakdown:
                lines.append("  - pa: n/a")
                continue
            for pa_name in sorted(pa_breakdown.keys()):
                pa_bucket = pa_breakdown.get(pa_name, {})
                if not isinstance(pa_bucket, dict):
                    continue
                pa_total = int(pa_bucket.get("total", 0))
                pa_successes = int(pa_bucket.get("successes", 0))
                pa_success_rate = float(pa_bucket.get("success_rate", 0.0)) if pa_total > 0 else 0.0
                lines.append(
                    f"  - {pa_name}: "
                    f"pa_success={pa_successes}/{pa_total}, "
                    f"pa_success_rate={_safe_float(pa_success_rate)}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _render_histogram(scores: dict[str, float], out_svg: Path) -> None:
    width = 1100
    height = 640
    margin_l = 80
    margin_r = 30
    margin_t = 70
    margin_b = 110
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    bins = 10
    counts = [0] * bins
    for v in scores.values():
        clipped = max(0.0, min(1.0, float(v)))
        idx = min(int(clipped * bins), bins - 1)
        counts[idx] += 1
    max_count = max(counts) if counts else 1
    if max_count == 0:
        max_count = 1

    bar_w = plot_w / bins

    svg: list[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg.append(
        f'<text x="{margin_l}" y="40" font-size="26" font-family="sans-serif" font-weight="bold">'
        "Performance Distribution (Average Success Rate)</text>"
    )
    svg.append(
        f'<text x="{margin_l}" y="62" font-size="14" font-family="sans-serif" fill="#555">'
        f"Generated at {_escape_xml(_iso_now())}</text>"
    )

    x0 = margin_l
    y0 = margin_t + plot_h
    svg.append(f'<line x1="{x0}" y1="{margin_t}" x2="{x0}" y2="{y0}" stroke="#222" stroke-width="2"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + plot_w}" y2="{y0}" stroke="#222" stroke-width="2"/>')

    for i in range(0, max_count + 1):
        y = y0 - (plot_h * (i / max_count))
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_w}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>')
        svg.append(
            f'<text x="{x0 - 10}" y="{y + 5:.1f}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#444">'
            f"{i}</text>"
        )

    for i, cnt in enumerate(counts):
        bar_h = 0.0 if max_count == 0 else (plot_h * (cnt / max_count))
        x = x0 + i * bar_w + 2
        y = y0 - bar_h
        svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 4:.1f}" height="{bar_h:.1f}" '
            'fill="#2f8fdd" opacity="0.9"/>'
        )
        label = f"{i/10:.1f}-{(i+1)/10:.1f}"
        svg.append(
            f'<text x="{x + (bar_w - 4) / 2:.1f}" y="{y0 + 18}" text-anchor="middle" '
            f'font-size="11" font-family="sans-serif" fill="#444">{label}</text>'
        )
        svg.append(
            f'<text x="{x + (bar_w - 4) / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" '
            f'font-size="11" font-family="sans-serif" fill="#1b4f8a">{cnt}</text>'
        )

    legend_y = y0 + 58
    svg.append(
        f'<text x="{margin_l}" y="{legend_y}" font-size="13" font-family="sans-serif" fill="#333">'
        f"Total policies: {len(scores)}</text>"
    )
    sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_text = ", ".join(f"{name}:{value:.3f}" for name, value in sorted_items[:8])
    svg.append(
        f'<text x="{margin_l}" y="{legend_y + 24}" font-size="12" font-family="monospace" fill="#555">'
        f"{_escape_xml(top_text)}</text>"
    )

    svg.append("</svg>")
    out_svg.write_text("\n".join(svg) + "\n", encoding="utf-8")


class EvaluationApp:
    def __init__(
        self,
        *,
        root: tk.Tk,
        participants: list[Participant],
        tasks: list[SHTConfig],
        runs_per_sht: int,
        result_root: Path,
        docker_manager: DockerManager,
        ros_controller: RosLaunchController,
    ) -> None:
        self.root = root
        self.participants = participants
        self.tasks = tasks
        self.runs_per_sht = max(1, int(runs_per_sht))
        self.result_root = result_root
        self.docker_manager = docker_manager
        self.ros = ros_controller
        self.font_family = "Arial"
        self.font_head = (self.font_family, 20, "bold")
        self.font_head_value = (self.font_family, 20)
        self.font_section = (self.font_family, 17, "bold")
        self.font_body = (self.font_family, 15)
        self.font_button = (self.font_family, 15, "bold")
        self.font_mono = ("Courier New", 14)

        self.policy_idx = 0
        self.sht_idx = 0
        self.repeat_idx = 1
        self.pa_idx = 0
        self.phase = "idle"
        self.current_pa_started: float | None = None
        self.awaiting_first_action = False
        self.pending_sht_success: bool | None = None
        self.current_run_pa_records: list[PARecord] = []

        self.pa_records: list[PARecord] = []
        self.sht_records: list[SHTRecord] = []
        self._record_index_map: list[int] = []
        self._popup_specs: dict[tk.Toplevel, tuple[int, int]] = {}
        self._waiting_action_popup: tk.Toplevel | None = None
        self._action_log_path: Path | None = None
        self._action_log_offset = 0
        self.motion_enabled = True
        self.motion_btn_var: tk.StringVar | None = None

        self.result_root.mkdir(parents=True, exist_ok=True)
        self._build_ui()
        self._update_header()
        self._refresh_records_list()
        self._tick_elapsed()
        self._tick_action_output()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(300, self._show_start_policy_popup)

    def _build_ui(self) -> None:
        self.root.title("AIROA Competition Evaluator")
        self.root.geometry("1400x920")
        self.root.minsize(1260, 820)
        self.root.configure(bg="#e8eef4")
        try:
            self.root.tk.call("tk", "scaling", 1.25)
        except Exception:
            pass
        self.root.bind("<Configure>", self._on_root_configure, add="+")

        container = tk.Frame(self.root, padx=20, pady=20, bg="#e8eef4")
        container.pack(fill=tk.BOTH, expand=True)

        self.policy_var = tk.StringVar(value="-")
        self.sht_var = tk.StringVar(value="-")
        self.pa_var = tk.StringVar(value="-")
        self.elapsed_var = tk.StringVar(value="0.0 sec")
        self.status_var = tk.StringVar(value="Ready")
        self.motion_btn_var = tk.StringVar(value="Robot STOP (no-send)")

        header_card = tk.Frame(container, bg="#ffffff", bd=1, relief=tk.GROOVE, padx=18, pady=14)
        header_card.pack(fill=tk.X, pady=(0, 14))
        header = tk.Frame(header_card, bg="#ffffff")
        header.pack(fill=tk.X)
        tk.Label(header, text="Policy:", font=self.font_head, bg="#ffffff", fg="#1f2a38").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=(0, 4)
        )
        tk.Label(header, textvariable=self.policy_var, font=self.font_head_value, bg="#ffffff", fg="#1f2a38").grid(
            row=0, column=1, sticky="w"
        )
        tk.Label(header, text="SHT:", font=self.font_head, bg="#ffffff", fg="#1f2a38").grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=(0, 4)
        )
        tk.Label(header, textvariable=self.sht_var, font=self.font_head_value, bg="#ffffff", fg="#1f2a38").grid(
            row=1, column=1, sticky="w"
        )
        tk.Label(header, text="PA:", font=self.font_head, bg="#ffffff", fg="#1f2a38").grid(
            row=2, column=0, sticky="w", padx=(0, 10), pady=(0, 4)
        )
        tk.Label(header, textvariable=self.pa_var, font=self.font_head_value, bg="#ffffff", fg="#1f2a38").grid(
            row=2, column=1, sticky="w"
        )
        tk.Label(header, text="Elapsed:", font=self.font_head, bg="#ffffff", fg="#1f2a38").grid(
            row=3, column=0, sticky="w", padx=(0, 10), pady=(0, 4)
        )
        tk.Label(header, textvariable=self.elapsed_var, font=self.font_head_value, bg="#ffffff", fg="#1f2a38").grid(
            row=3, column=1, sticky="w"
        )
        tk.Label(header, text="Status:", font=self.font_section, bg="#ffffff", fg="#1f2a38").grid(
            row=4, column=0, sticky="w", padx=(0, 10), pady=(6, 0)
        )
        tk.Label(header, textvariable=self.status_var, font=self.font_body, bg="#ffffff", fg="#0c5a8a").grid(
            row=4, column=1, sticky="w"
        )

        action_row = tk.Frame(container, bg="#e8eef4", pady=4)
        action_row.pack(fill=tk.X, pady=(0, 14))
        self.success_btn = tk.Button(
            action_row,
            text="Success",
            width=16,
            height=2,
            state=tk.DISABLED,
            bg="#cfeecf",
            activebackground="#b4dfb4",
            font=self.font_button,
            relief=tk.RAISED,
            command=lambda: self._on_pa_result(True),
        )
        self.success_btn.pack(side=tk.LEFT, padx=(0, 14))
        self.fail_btn = tk.Button(
            action_row,
            text="Fail",
            width=16,
            height=2,
            state=tk.DISABLED,
            bg="#f7cccc",
            activebackground="#efb5b5",
            font=self.font_button,
            relief=tk.RAISED,
            command=lambda: self._on_pa_result(False),
        )
        self.fail_btn.pack(side=tk.LEFT, padx=(0, 14))
        self.start_btn = tk.Button(
            action_row,
            text="Start / Resume",
            width=19,
            height=2,
            font=self.font_button,
            bg="#cde0f9",
            activebackground="#b6d0f2",
            relief=tk.RAISED,
            command=self._on_start_button,
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 14))
        self.motion_btn = tk.Button(
            action_row,
            textvariable=self.motion_btn_var,
            width=22,
            height=2,
            font=self.font_button,
            bg="#ffd2d2",
            activebackground="#f8b5b5",
            relief=tk.RAISED,
            command=self._toggle_motion_forwarding,
        )
        self.motion_btn.pack(side=tk.LEFT)
        self._refresh_motion_button()

        mid = tk.Frame(container, bg="#e8eef4")
        mid.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(mid, bg="#ffffff", bd=1, relief=tk.GROOVE, padx=12, pady=10)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        header_row = tk.Frame(left, bg="#ffffff")
        header_row.pack(fill=tk.X, pady=(0, 8))
        tk.Label(header_row, text="PA Result Records", font=self.font_section, bg="#ffffff").pack(side=tk.LEFT)
        self.records_edit_mode = tk.BooleanVar(value=False)
        self.records_edit_label = tk.StringVar(value="Edit Locked")
        self.records_edit_btn = tk.Checkbutton(
            header_row,
            textvariable=self.records_edit_label,
            variable=self.records_edit_mode,
            indicatoron=False,
            font=self.font_body,
            width=12,
            relief=tk.RAISED,
            bg="#eceff4",
            selectcolor="#ffe4a8",
            activebackground="#dee3eb",
            command=self._toggle_records_edit_mode,
        )
        self.records_edit_btn.pack(side=tk.RIGHT)

        self.records_canvas = tk.Canvas(left, bg="#f8fafc", highlightthickness=0)
        self.records_scrollbar = tk.Scrollbar(left, orient=tk.VERTICAL, command=self.records_canvas.yview)
        self.records_inner = tk.Frame(self.records_canvas, bg="#f8fafc")
        self.records_canvas.configure(yscrollcommand=self.records_scrollbar.set)

        self.records_inner_window = self.records_canvas.create_window((0, 0), window=self.records_inner, anchor="nw")
        self.records_inner.bind(
            "<Configure>",
            lambda _e: self.records_canvas.configure(scrollregion=self.records_canvas.bbox("all")),
        )
        self.records_canvas.bind(
            "<Configure>",
            lambda e: self.records_canvas.itemconfigure(self.records_inner_window, width=e.width),
        )
        self.records_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.records_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _toggle_records_edit_mode(self) -> None:
        if self.records_edit_mode.get():
            self.records_edit_label.set("Edit Enabled")
            self.records_edit_btn.configure(bg="#ffe4a8")
        else:
            self.records_edit_label.set("Edit Locked")
            self.records_edit_btn.configure(bg="#eceff4")
        self._refresh_records_list()

    def _refresh_motion_button(self) -> None:
        if self.motion_btn_var is None:
            return
        if self.motion_enabled:
            self.motion_btn_var.set("Robot STOP (no-send)")
            self.motion_btn.configure(bg="#ffd2d2", activebackground="#f8b5b5")
        else:
            self.motion_btn_var.set("Robot RESUME")
            self.motion_btn.configure(bg="#d8f5d8", activebackground="#bde8bd")

    def _toggle_motion_forwarding(self) -> None:
        if not self.ros.is_running():
            self._set_status("roslaunch is not running. Start a policy/SHT first.")
            messagebox.showinfo("Motion Control", "roslaunch is not running yet.")
            return
        target_enabled = not self.motion_enabled
        try:
            self.ros.set_motion_enabled(target_enabled)
        except Exception as exc:
            self._set_status(f"Failed to change motion forwarding: {exc}")
            messagebox.showerror("Motion Control Failed", str(exc))
            return
        self.motion_enabled = target_enabled
        self._refresh_motion_button()
        if self.motion_enabled:
            self._set_status("Robot command forwarding resumed.")
        else:
            self._set_status("Robot command forwarding stopped (inference keeps running).")

    def _on_root_configure(self, event: tk.Event[Any]) -> None:
        if event.widget is not self.root:
            return
        for popup, (width, height) in list(self._popup_specs.items()):
            if popup.winfo_exists():
                self._position_popup(popup, width=width, height=height)
                try:
                    popup.attributes("-topmost", True)
                except Exception:
                    pass
            else:
                self._popup_specs.pop(popup, None)

    def _register_popup(self, popup: tk.Toplevel, *, width: int, height: int) -> None:
        self._popup_specs[popup] = (width, height)
        popup.transient(self.root)
        try:
            popup.attributes("-topmost", True)
        except Exception:
            pass
        self._position_popup(popup, width=width, height=height)

        def _cleanup(event: tk.Event[Any]) -> None:
            if event.widget is popup:
                self._popup_specs.pop(popup, None)

        popup.bind("<Destroy>", _cleanup, add="+")

    def _position_popup(self, popup: tk.Toplevel, *, width: int, height: int) -> None:
        self.root.update_idletasks()
        root_x = self.root.winfo_x()
        root_y = self.root.winfo_y()
        root_w = self.root.winfo_width()
        root_h = self.root.winfo_height()
        x = max(root_x + max((root_w - width) // 2, 10), 0)
        y = max(root_y + max((root_h - height) // 2, 10), 0)
        popup.geometry(f"{width}x{height}+{x}+{y}")
        popup.transient(self.root)
        popup.lift()
        popup.focus_force()

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)
        print(f"[STATUS] {message}")

    def _display_prompt(self, prompt: str, *, max_chars: int = 120) -> str:
        text = " ".join(str(prompt).split())
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "..."

    def _reset_action_output_for_new_run(self) -> None:
        self._action_log_path = self.ros.get_last_host_log_path()
        self._action_log_offset = 0
        if self._action_log_path is not None and self._action_log_path.exists():
            try:
                self._action_log_offset = self._action_log_path.stat().st_size
            except Exception:
                self._action_log_offset = 0

    def _show_wait_action_popup(self) -> None:
        self._close_wait_action_popup()
        participant = self._current_participant()
        sht = self._current_sht()
        pa = self._current_pa()

        popup = tk.Toplevel(self.root)
        popup.title("Waiting for First Action")
        popup.resizable(False, False)
        self._register_popup(popup, width=820, height=340)

        msg = (
            f"Policy: {participant.name}\n"
            f"SHT: {sht.name} (eval {self.repeat_idx}/{self.runs_per_sht})\n"
            f"Prompt: {self._display_prompt(pa.prompt, max_chars=220)}\n\n"
            "Waiting until first action output is observed.\n"
            "Timer and Success/Fail buttons will start after action output begins."
        )
        tk.Label(
            popup,
            text=msg,
            justify="left",
            font=self.font_section,
            wraplength=760,
        ).pack(anchor="w", padx=24, pady=(24, 14))

        tk.Button(
            popup,
            text="Hide Window",
            width=20,
            font=self.font_button,
            command=popup.destroy,
        ).pack(pady=(0, 20))
        self._waiting_action_popup = popup

    def _close_wait_action_popup(self) -> None:
        popup = self._waiting_action_popup
        self._waiting_action_popup = None
        if popup is None:
            return
        if not popup.winfo_exists():
            return
        try:
            popup.grab_release()
        except Exception:
            pass
        popup.destroy()

    def _on_first_action_detected(self) -> None:
        if not self.awaiting_first_action:
            return
        self.awaiting_first_action = False
        self.current_pa_started = time.monotonic()
        self.phase = "running_pa"
        self._enable_pa_buttons(True)
        self._close_wait_action_popup()
        self._set_status("First action detected. Timer started; Success/Fail enabled.")

    def _tick_action_output(self) -> None:
        try:
            self._poll_action_output()
        finally:
            self.root.after(400, self._tick_action_output)

    def _poll_action_output(self) -> None:
        log_path = self.ros.get_last_host_log_path()
        if log_path is None:
            return
        if self._action_log_path != log_path:
            self._action_log_path = log_path
            self._action_log_offset = 0
            if log_path.exists():
                try:
                    self._action_log_offset = log_path.stat().st_size
                except Exception:
                    self._action_log_offset = 0
        if not log_path.exists():
            return

        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(self._action_log_offset)
                new_lines = f.readlines()
                self._action_log_offset = f.tell()
        except Exception:
            return

        if not new_lines:
            return

        saw_action = False
        for raw in new_lines:
            line = raw.strip()
            if not line:
                continue
            lower = line.lower()
            if "action:" in lower:
                saw_action = True
            elif "action executed." in lower or "action not executed." in lower:
                saw_action = True
        if saw_action:
            self._on_first_action_detected()

    def _current_participant(self) -> Participant:
        return self.participants[self.policy_idx]

    def _current_sht(self) -> SHTConfig:
        return self.tasks[self.sht_idx]

    def _current_pa(self) -> PAConfig:
        return self._current_sht().pas[self.pa_idx]

    def _on_start_button(self) -> None:
        if self.phase in {"idle", "policy_ready"}:
            self._show_start_policy_popup()
            return
        if self.phase == "sht_ready":
            self._show_start_sht_popup()
            return
        if self.phase == "running_pa_wait_action":
            self._set_status("Still waiting for first action output...")
            return
        if self.phase in {"awaiting_sht_result", "awaiting_sht_review"}:
            if self.pending_sht_success is None:
                self._show_sht_result_popup()
            else:
                self._show_sht_review_popup()
            return
        messagebox.showinfo("State", f"Current phase is '{self.phase}'.")

    def _show_start_policy_popup(self) -> None:
        if self.phase == "done":
            return
        participant = self._current_participant()
        self.phase = "policy_ready"
        self._set_status(f"Ready to start evaluating policy: {participant.name}")
        self._show_popup_button(
            title="Start Policy",
            message=(
                f"Start evaluating {participant.name}.\n\n"
                "Clicking this button will immediately launch the client deploy node\n"
                "for the model-assigned policy server port."
            ),
            button_text=f"Start evaluating {participant.name}",
            on_click=self._start_policy_run,
        )

    def _start_policy_run(self) -> None:
        self.pa_idx = 0
        self._begin_sht_run()

    def _show_start_sht_popup(self) -> None:
        if self.phase == "done":
            return
        participant = self._current_participant()
        sht = self._current_sht()
        pa = self._current_pa()
        self.phase = "sht_ready"
        self._set_status(f"Ready: {participant.name} / {sht.name} / eval {self.repeat_idx}")
        self._show_popup_button(
            title="Start SHT",
            message=(
                f"Reset robot to initial pose, then start.\n\n"
                f"Policy: {participant.name}\nSHT: {sht.name}\nEval: {self.repeat_idx}\n"
                f"Prompt: {self._display_prompt(pa.prompt, max_chars=220)}"
            ),
            button_text=f"Start {sht.name} eval {self.repeat_idx}",
            on_click=self._begin_sht_run,
        )

    def _begin_sht_run(self) -> None:
        participant = self._current_participant()
        sht = self._current_sht()
        pa = self._current_pa()
        self._set_status("Starting server/session...")

        try:
            runtime = self.docker_manager.ensure_server_running(self.policy_idx)
            self._set_status(
                f"Launching client roslaunch for {participant.name} on ws://127.0.0.1:{runtime.port}"
            )
            trace_base_dir = f"/root/eval_results/{_slugify(participant.name)}/traces"
            config_name = f"{_slugify(participant.name)}_{_slugify(sht.name)}_eval{self.repeat_idx}"
            self.ros.start(
                policy_name=participant.name,
                policy_server_host="127.0.0.1",
                policy_server_port=runtime.port,
                trace_base_dir=trace_base_dir,
                config_name=config_name,
                initial_instruction=pa.prompt,
            )
            self.ros.set_instruction(pa.prompt)
            self.ros.set_motion_enabled(True)
        except Exception as exc:
            self.phase = "sht_ready"
            self.motion_enabled = True
            self.awaiting_first_action = False
            self._refresh_motion_button()
            self._set_status(f"Failed to start run: {exc}")
            messagebox.showerror("Start failed", str(exc))
            return

        self.current_run_pa_records = []
        self.current_pa_started = None
        self.awaiting_first_action = True
        self._reset_action_output_for_new_run()
        self.motion_enabled = True
        self._refresh_motion_button()
        self.phase = "running_pa_wait_action"
        self._enable_pa_buttons(False)
        self._update_header()
        self._set_status("Waiting for first action output...")

    def _enable_pa_buttons(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.success_btn.configure(state=state)
        self.fail_btn.configure(state=state)

    def _on_pa_result(self, success: bool) -> None:
        if self.phase != "running_pa":
            return

        # Stop robot command forwarding immediately when operator submits PA result.
        try:
            if self.ros.is_running():
                self.ros.set_motion_enabled(False)
        except Exception as exc:
            self._set_status(f"Warning: failed to stop robot motion immediately: {exc}")
        self.motion_enabled = False
        self._refresh_motion_button()

        elapsed = 0.0
        if self.current_pa_started is not None:
            elapsed = max(0.0, time.monotonic() - self.current_pa_started)
        pa = self._current_pa()
        rec = PARecord(
            policy=self._current_participant().name,
            sht=self._current_sht().name,
            repeat_index=self.repeat_idx,
            pa=pa.name,
            prompt=pa.prompt,
            success=bool(success),
            elapsed_sec=elapsed,
            recorded_at=_iso_now(),
        )
        self.phase = "awaiting_pa_next"
        self._enable_pa_buttons(False)

        label = "Success" if success else "Fail"
        prompt_label = self._display_prompt(pa.prompt, max_chars=200)
        self._show_popup_button(
            title="PA Result",
            message=f"{prompt_label}\nResult: {label}\nElapsed: {elapsed:.1f} sec",
            button_text="Next PA",
            on_click=lambda: self._commit_pa_and_continue(rec),
        )

    def _commit_pa_and_continue(self, record: PARecord) -> None:
        self.pa_records.append(record)
        self.current_run_pa_records.append(record)
        self._persist_policy_outputs(record.policy)
        self._refresh_records_list()

        pa_count = len(self._current_sht().pas)
        if self.pa_idx + 1 < pa_count:
            self.pa_idx += 1
            next_pa = self._current_pa()
            try:
                self.ros.set_motion_enabled(True)
                self.ros.set_instruction(next_pa.prompt)
            except Exception as exc:
                self._set_status(f"Failed to prepare next PA: {exc}")
                messagebox.showerror("Next PA prepare failed", str(exc))
                return
            self.motion_enabled = True
            self._refresh_motion_button()
            self.phase = "running_pa_wait_action"
            self.current_pa_started = None
            self.awaiting_first_action = True
            self._enable_pa_buttons(False)
            self._update_header()
            self._set_status(f"Moved to next PA. Waiting for action: {self._display_prompt(next_pa.prompt, max_chars=80)}")
            return

        self.phase = "awaiting_sht_result"
        self.awaiting_first_action = False
        self._show_sht_result_popup()

    def _show_sht_result_popup(self) -> None:
        popup = tk.Toplevel(self.root)
        popup.title("SHT Result")
        popup.resizable(False, False)
        self._register_popup(popup, width=560, height=300)
        popup.grab_set()
        sht = self._current_sht()
        tk.Label(
            popup,
            text=f"Select SHT result for {sht.name} (eval {self.repeat_idx})",
            font=self.font_section,
            wraplength=520,
            justify="left",
        ).pack(anchor="w", padx=20, pady=(20, 14))

        def choose(result: bool) -> None:
            popup.destroy()
            self.pending_sht_success = result
            self._show_sht_review_popup()

        row = tk.Frame(popup)
        row.pack(fill=tk.X, padx=20, pady=14)
        tk.Button(row, text="SHT Success", width=16, bg="#d7f5d7", font=self.font_button, command=lambda: choose(True)).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        tk.Button(row, text="SHT Fail", width=16, bg="#ffd7d7", font=self.font_button, command=lambda: choose(False)).pack(side=tk.LEFT)

    def _show_sht_review_popup(self) -> None:
        self.phase = "awaiting_sht_review"
        lines = []
        for rec in self.current_run_pa_records:
            mark = "Success" if rec.success else "Fail"
            lines.append(f"- {self._display_prompt(rec.prompt, max_chars=180)}: {mark} ({rec.elapsed_sec:.1f}s)")
        detail = "\n".join(lines) if lines else "(no PA records)"

        popup = tk.Toplevel(self.root)
        popup.title("Review SHT Records")
        popup.resizable(True, True)
        self._register_popup(popup, width=820, height=560)
        popup.grab_set()
        tk.Label(popup, text="Review PA results before moving next.", font=self.font_section).pack(
            anchor="w", padx=20, pady=(18, 10)
        )
        txt = tk.Text(popup, height=14, width=90, font=self.font_mono)
        txt.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 12))
        txt.insert("1.0", detail)
        txt.configure(state=tk.DISABLED)

        button_row = tk.Frame(popup)
        button_row.pack(fill=tk.X, padx=20, pady=(0, 16))

        def confirm_next() -> None:
            popup.destroy()
            self._finalize_current_sht_run()

        def close_later() -> None:
            popup.destroy()
            self.phase = "awaiting_sht_review"
            self._set_status("SHT review closed. Press Start / Resume to continue.")

        tk.Button(button_row, text="Confirm and Next SHT", font=self.font_button, command=confirm_next).pack(side=tk.LEFT, padx=(0, 12))
        tk.Button(button_row, text="Close (Edit Later)", font=self.font_body, command=close_later).pack(side=tk.LEFT)

    def _finalize_current_sht_run(self) -> None:
        if self.pending_sht_success is None:
            messagebox.showerror("Internal error", "SHT result is not selected.")
            return

        pa_total = len(self.current_run_pa_records)
        pa_successes = sum(1 for r in self.current_run_pa_records if r.success)
        sht_record = SHTRecord(
            policy=self._current_participant().name,
            sht=self._current_sht().name,
            repeat_index=self.repeat_idx,
            success=self.pending_sht_success,
            pa_successes=pa_successes,
            pa_total=pa_total,
            recorded_at=_iso_now(),
        )
        self.sht_records.append(sht_record)
        self.pending_sht_success = None
        self._persist_policy_outputs(sht_record.policy)
        self._refresh_records_list()

        self.ros.stop()
        self.awaiting_first_action = False
        self.motion_enabled = True
        self._refresh_motion_button()
        self._set_status("SHT run completed. Exec trace should be saved under result directory.")

        if self.repeat_idx < self.runs_per_sht:
            self.repeat_idx += 1
            self.pa_idx = 0
            self.phase = "sht_ready"
            self._update_header()
            self._show_start_sht_popup()
            return

        if self.sht_idx + 1 < len(self.tasks):
            self.sht_idx += 1
            self.repeat_idx = 1
            self.pa_idx = 0
            self.phase = "sht_ready"
            self._update_header()
            self._show_start_sht_popup()
            return

        self._complete_policy()

    def _complete_policy(self) -> None:
        finished = self._current_participant()
        self._set_status(f"Policy completed: {finished.name}")
        self._persist_policy_outputs(finished.name)
        self.docker_manager.complete_policy(self.policy_idx)

        if self.policy_idx + 1 < len(self.participants):
            self.policy_idx += 1
            self.sht_idx = 0
            self.repeat_idx = 1
            self.pa_idx = 0
            self.phase = "idle"
            self._update_header()
            self._show_start_policy_popup()
            return

        self.phase = "done"
        self._update_histogram()
        self.docker_manager.stop_all_servers()
        self._update_header()
        self._set_status("All policies completed.")
        messagebox.showinfo("Completed", f"All evaluations finished.\nResults: {self.result_root}")

    def _persist_policy_outputs(self, policy_name: str) -> None:
        policy_dir = self.result_root / _slugify(policy_name)
        policy_dir.mkdir(parents=True, exist_ok=True)

        pa_payload = [r.to_dict() for r in self.pa_records if r.policy == policy_name][::-1]
        sht_payload = [r.to_dict() for r in self.sht_records if r.policy == policy_name][::-1]
        _write_json(policy_dir / "records_pa.json", pa_payload)
        _write_json(policy_dir / "records_sht.json", sht_payload)

        summary = _compute_policy_summary(policy_name=policy_name, pa_records=self.pa_records, sht_records=self.sht_records)
        _write_json(policy_dir / "summary.json", summary)
        _write_sr_txt(policy_dir / "SR.txt", summary)
        self._update_histogram()

    def _update_histogram(self) -> None:
        scores: dict[str, float] = {}
        leaderboard: list[dict[str, Any]] = []
        for participant in self.participants:
            policy_dir = self.result_root / _slugify(participant.name)
            summary_path = policy_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = _read_json(summary_path)
            if int(summary.get("sht_total", 0)) <= 0:
                continue
            score = float(summary.get("average_success_rate", 0.0))
            scores[participant.name] = score
            leaderboard.append({"policy": participant.name, "average_success_rate": score})

        leaderboard.sort(key=lambda x: x["average_success_rate"], reverse=True)
        _write_json(self.result_root / "leaderboard.json", leaderboard)
        if scores:
            _render_histogram(scores, self.result_root / "histogram.svg")
        else:
            (self.result_root / "histogram.svg").write_text(
                "<svg xmlns='http://www.w3.org/2000/svg' width='640' height='220'>"
                "<rect width='100%' height='100%' fill='white'/>"
                "<text x='20' y='50' font-size='26' font-family='sans-serif'>No completed policy yet.</text>"
                "</svg>\n",
                encoding="utf-8",
            )

    def _refresh_records_list(self) -> None:
        current_policy = self._current_participant().name
        for child in self.records_inner.winfo_children():
            child.destroy()

        editable = self.records_edit_mode.get()
        state = tk.NORMAL if editable else tk.DISABLED
        self._record_index_map = list(
            reversed([idx for idx, rec in enumerate(self.pa_records) if rec.policy == current_policy])
        )
        if not self._record_index_map:
            tk.Label(
                self.records_inner,
                text="No records yet for this model.",
                font=self.font_body,
                bg="#f8fafc",
                fg="#5d6b7c",
                pady=16,
            ).pack(anchor="w")
            return

        for row_no, global_idx in enumerate(self._record_index_map, start=1):
            rec = self.pa_records[global_idx]
            row = tk.Frame(self.records_inner, bg="#ffffff", bd=1, relief=tk.GROOVE, padx=10, pady=8)
            row.pack(fill=tk.X, padx=6, pady=5)

            prompt_display = self._display_prompt(rec.prompt, max_chars=130)
            summary = (
                f"{row_no:03d}. {rec.sht} / eval {rec.repeat_index}   ({rec.elapsed_sec:.1f}s)\n"
                f"prompt: {prompt_display}"
            )
            tk.Label(
                row,
                text=summary,
                font=self.font_mono,
                bg="#ffffff",
                fg="#1f2a38",
                anchor="w",
                justify="left",
            ).grid(row=0, column=0, columnspan=3, sticky="w")

            status_var = tk.IntVar(value=1 if rec.success else 0)
            tk.Radiobutton(
                row,
                text="Success",
                variable=status_var,
                value=1,
                indicatoron=False,
                width=12,
                font=self.font_body,
                bg="#d7f2d7",
                activebackground="#c3e7c3",
                selectcolor="#bfe8bf",
                state=state,
                command=lambda idx=global_idx, v=status_var: self._on_record_radio_changed(idx, v),
            ).grid(row=1, column=0, sticky="w", pady=(8, 0))
            tk.Radiobutton(
                row,
                text="Fail",
                variable=status_var,
                value=0,
                indicatoron=False,
                width=12,
                font=self.font_body,
                bg="#f6d1d1",
                activebackground="#efbebe",
                selectcolor="#efbcbc",
                state=state,
                command=lambda idx=global_idx, v=status_var: self._on_record_radio_changed(idx, v),
            ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
            tk.Label(
                row,
                text=f"Recorded: {rec.recorded_at}",
                font=(self.font_family, 12),
                bg="#ffffff",
                fg="#6c7888",
            ).grid(row=1, column=2, sticky="e", padx=(10, 0), pady=(8, 0))

            row.grid_columnconfigure(2, weight=1)

    def _on_record_radio_changed(self, global_idx: int, status_var: tk.IntVar) -> None:
        if not self.records_edit_mode.get():
            return
        if global_idx < 0 or global_idx >= len(self.pa_records):
            return
        rec = self.pa_records[global_idx]
        new_val = bool(status_var.get())
        if rec.success == new_val:
            return
        rec.success = new_val
        rec.recorded_at = _iso_now()
        self._persist_policy_outputs(rec.policy)
        self._set_status(f"Edited PA result: {self._display_prompt(rec.prompt, max_chars=80)} -> {'Success' if new_val else 'Fail'}")
        self._refresh_records_list()

    def _tick_elapsed(self) -> None:
        if self.phase == "running_pa" and self.current_pa_started is not None:
            elapsed = max(0.0, time.monotonic() - self.current_pa_started)
            self.elapsed_var.set(f"{elapsed:.1f} sec")
        else:
            self.elapsed_var.set("0.0 sec")
        self.root.after(500, self._tick_elapsed)

    def _show_popup_button(
        self,
        *,
        title: str,
        message: str,
        button_text: str,
        on_click: Any,
    ) -> None:
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.resizable(False, False)
        self._register_popup(popup, width=760, height=320)
        popup.grab_set()
        tk.Label(
            popup,
            text=message,
            justify="left",
            font=self.font_section,
            wraplength=700,
        ).pack(anchor="w", padx=24, pady=(24, 20))

        def _clicked() -> None:
            popup.destroy()
            on_click()

        tk.Button(popup, text=button_text, width=28, font=self.font_button, command=_clicked).pack(pady=(0, 20))

    def _update_header(self) -> None:
        participant = self._current_participant()
        sht = self._current_sht()
        pa = self._current_pa()
        self.policy_var.set(participant.name)
        self.sht_var.set(f"{sht.name} (eval {self.repeat_idx}/{self.runs_per_sht})")
        self.pa_var.set(f"{self._display_prompt(pa.prompt, max_chars=90)} ({self.pa_idx + 1}/{len(sht.pas)})")

    def _on_close(self) -> None:
        if self.phase == "running_pa":
            if not messagebox.askyesno("Exit", "Evaluation is running. Stop and exit?"):
                return
        self._set_status("Shutting down. Stopping roslaunch and server containers...")
        try:
            self.ros.stop()
        finally:
            self.docker_manager.stop_all_servers()
        self.awaiting_first_action = False
        self.motion_enabled = True
        self._refresh_motion_button()
        self.root.destroy()


def _parse_bool(value: str) -> bool:
    val = str(value).strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


def _load_models(path: Path, repo_root: Path) -> list[Participant]:
    payload = _read_json(path)
    rows = payload.get("models")
    used_legacy_participants_key = False
    if rows is None:
        rows = payload.get("participants")
        used_legacy_participants_key = rows is not None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"'models' must be a non-empty list: {path}")
    if used_legacy_participants_key:
        print("[WARN] Deprecated key 'participants' detected. Please rename it to 'models'.")

    participants: list[Participant] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"models[{idx}] must be object")
        name = str(row.get("name", "")).strip()
        worktree_path = str(row.get("worktree_path", "")).strip()
        checkpoint_path = str(row.get("checkpoint_path", "")).strip()
        config_name = str(row.get("config_name", "")).strip()
        if not (name and worktree_path and checkpoint_path and config_name):
            raise ValueError(
                f"models[{idx}] requires name/worktree_path/checkpoint_path/config_name"
            )

        env = row.get("env", {})
        if env is None:
            env = {}
        if not isinstance(env, dict):
            raise ValueError(f"models[{idx}].env must be object")
        env_cast = {str(k): str(v) for k, v in env.items()}

        policy_cache_dir = str(row.get("policy_cache_dir", repo_root / ".docker_cache" / "policy_cache"))
        hf_cache_dir = str(row.get("hf_cache_dir", repo_root / ".docker_cache" / "hf"))

        participants.append(
            Participant(
                name=name,
                worktree_path=Path(worktree_path).expanduser().resolve(),
                checkpoint_path=Path(checkpoint_path).expanduser().resolve(),
                config_name=config_name,
                env=env_cast,
                policy_cache_dir=Path(policy_cache_dir).expanduser().resolve(),
                hf_cache_dir=Path(hf_cache_dir).expanduser().resolve(),
            )
        )
    return participants


def _load_tasks(path: Path) -> list[SHTConfig]:
    payload = _read_json(path)
    rows = payload.get("shts", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"'shts' must be a non-empty list: {path}")

    out: list[SHTConfig] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"shts[{i}] must be object")
        name = str(row.get("name", "")).strip()
        pa_rows = row.get("pas", [])
        if not name:
            raise ValueError(f"shts[{i}].name is required")
        if not isinstance(pa_rows, list) or not pa_rows:
            raise ValueError(f"shts[{i}].pas must be non-empty list")
        pas: list[PAConfig] = []
        for j, pa in enumerate(pa_rows):
            if not isinstance(pa, dict):
                raise ValueError(f"shts[{i}].pas[{j}] must be object")
            pa_name = str(pa.get("name", "")).strip()
            prompt = str(pa.get("prompt", "")).strip()
            if not pa_name or not prompt:
                raise ValueError(f"shts[{i}].pas[{j}] requires name and prompt")
            pas.append(PAConfig(name=pa_name, prompt=prompt))
        out.append(SHTConfig(name=name, pas=pas))
    return out


def _default_result_root(repo_root: Path) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d")
    return repo_root / "results" / f"result_{stamp}"


def _create_tk_root() -> tk.Tk:
    display = os.environ.get("DISPLAY", "")
    try:
        return tk.Tk()
    except tk.TclError as exc:  # type: ignore[union-attr]
        raise RuntimeError(
            "Failed to open GUI display for tkinter.\n"
            f"DISPLAY={display!r}\n"
            "Run this command in a graphical session, or configure X11 forwarding.\n"
            "Examples:\n"
            "  - Local desktop: export DISPLAY=:0\n"
            "  - SSH: reconnect with `ssh -Y <host>` and ensure local X server is running."
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Competition evaluator GUI with multi-model server orchestration.")
    parser.add_argument("--models", default="", help="Path to models JSON")
    parser.add_argument("--participants", default="", help=argparse.SUPPRESS)  # deprecated alias
    parser.add_argument("--tasks", required=True, help="Path to SHT/PA tasks JSON")
    parser.add_argument("--runs-per-sht", type=int, default=1, help="Evaluation repeats per SHT")
    parser.add_argument("--start-port", type=int, default=8100, help="Base policy server port")
    parser.add_argument("--max-running", type=int, default=1, help="Max simultaneous model server containers")
    parser.add_argument(
        "--result-root",
        default="",
        help="Result root directory (default: results/result_YYYYMMDD)",
    )
    parser.add_argument("--client-container-name", default="airoa_hsr_client")
    parser.add_argument("--skip-client-up", action="store_true", help="Skip docker compose up for hsr_client")
    parser.add_argument("--test-mode", default="false", help="Pass test_mode to roslaunch (true/false)")
    parser.add_argument(
        "--hsr-ip",
        default="",
        help="HSR robot IP (used to derive ROS_MASTER_URI=http://<hsr_ip>:11311 when --ros-master-uri is not set)",
    )
    parser.add_argument("--ros-master-uri", default="", help="ROS master URI for client container")
    parser.add_argument("--ros-ip", default="", help="ROS_IP for client container")
    parser.add_argument(
        "--ros-ip-choice",
        type=int,
        default=2,
        help="1-based host IP candidate index (e.g. 2 selects the 2nd candidate). Ignored when --ros-ip is set.",
    )
    parser.add_argument(
        "--disable-gpu",
        action="store_true",
        help="Do not pass --gpus all to policy server containers",
    )
    args = parser.parse_args()

    if args.models and args.participants and args.models != args.participants:
        parser.error("Use either --models or --participants (deprecated), not both with different values.")
    models_path_str = args.models or args.participants
    if not models_path_str:
        parser.error("--models is required.")
    if args.participants and not args.models:
        print("[WARN] --participants is deprecated. Use --models.")

    if tk is None or messagebox is None:
        raise RuntimeError(
            "tkinter is required for GUI evaluator. Install it via conda (`conda install tk`) or apt (`sudo apt-get install python3-tk`)."
        )

    repo_root = Path(__file__).resolve().parents[1]
    result_root = Path(args.result_root).expanduser().resolve() if args.result_root else _default_result_root(repo_root)

    models = _load_models(Path(models_path_str).expanduser().resolve(), repo_root)
    tasks = _load_tasks(Path(args.tasks).expanduser().resolve())
    test_mode = _parse_bool(args.test_mode)
    root = _create_tk_root()

    client_env_overrides: dict[str, str] = {}
    ros_master_uri = args.ros_master_uri.strip()
    hsr_ip = args.hsr_ip.strip()
    ros_ip = args.ros_ip.strip()
    ros_ip_choice = int(args.ros_ip_choice)
    if ros_ip and ros_ip_choice > 0:
        parser.error("Use either --ros-ip or --ros-ip-choice, not both.")
    if ros_master_uri:
        client_env_overrides["ROS_MASTER_URI"] = ros_master_uri
    elif hsr_ip:
        client_env_overrides["ROS_MASTER_URI"] = f"http://{hsr_ip}:11311"
        client_env_overrides["HSR_IP"] = hsr_ip
    if ros_ip:
        client_env_overrides["ROS_IP"] = ros_ip
    elif ros_ip_choice > 0:
        candidates = _detect_host_ipv4_candidates()
        if not candidates:
            raise RuntimeError(
                "Failed to auto-select ROS_IP: no non-loopback host IPv4 candidate found. "
                "Set --ros-ip explicitly."
            )
        if ros_ip_choice > len(candidates):
            raise RuntimeError(
                f"--ros-ip-choice={ros_ip_choice} is out of range. "
                f"Available candidates ({len(candidates)}): {', '.join(candidates)}"
            )
        chosen_ros_ip = candidates[ros_ip_choice - 1]
        client_env_overrides["ROS_IP"] = chosen_ros_ip
        print(
            f"[INFO] Auto-selected ROS_IP={chosen_ros_ip} "
            f"(candidate #{ros_ip_choice} of {len(candidates)}: {', '.join(candidates)})"
        )
    if not test_mode and "ROS_MASTER_URI" not in client_env_overrides and not os.environ.get("ROS_MASTER_URI"):
        print(
            "[WARN] ROS_MASTER_URI is not set. hsr_client may default to http://127.0.0.1:11311. "
            "Pass --hsr-ip or --ros-master-uri, or export ROS_MASTER_URI."
        )
    if not test_mode and "ROS_IP" not in client_env_overrides and not os.environ.get("ROS_IP"):
        print(
            "[WARN] ROS_IP is not set. hsr_client may default to 127.0.0.1. "
            "Pass --ros-ip, --ros-ip-choice (e.g. 2), or export ROS_IP."
        )

    docker_manager = DockerManager(
        repo_root=repo_root,
        participants=models,
        base_port=args.start_port,
        max_running=args.max_running,
        result_root=result_root,
        client_container_name=args.client_container_name,
        gpu_enabled=not args.disable_gpu,
        client_env_overrides=client_env_overrides,
    )

    if not args.skip_client_up:
        print("[INFO] Starting hsr_client container (if needed)...")
        docker_manager.ensure_client_up()

    print("[INFO] Prewarming model servers...")
    docker_manager.prewarm_servers()

    ros = RosLaunchController(
        repo_root=repo_root,
        result_root=result_root,
        client_container_name=args.client_container_name,
        test_mode=test_mode,
    )

    app = EvaluationApp(
        root=root,
        participants=models,
        tasks=tasks,
        runs_per_sht=args.runs_per_sht,
        result_root=result_root,
        docker_manager=docker_manager,
        ros_controller=ros,
    )
    app._set_status(f"Result root: {result_root}")
    root.mainloop()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
