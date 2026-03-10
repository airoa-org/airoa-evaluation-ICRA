#!/usr/bin/env python3
"""Web server version of the competition evaluator.

Runs a LAN-accessible browser UI while preserving evaluator flow/logic.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from eval_competition import (
    DockerManager,
    PARecord,
    Participant,
    RosLaunchController,
    SHTConfig,
    SHTRecord,
    _compute_policy_summary,
    _default_result_root,
    _detect_host_ipv4_candidates,
    _load_models,
    _load_tasks,
    _parse_bool,
    _read_json,
    _render_histogram,
    _slugify,
    _write_json,
    _write_sr_txt,
)


def _iso_now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class WebEvaluationState:
    def __init__(
        self,
        *,
        participants: list[Participant],
        tasks: list[SHTConfig],
        runs_per_sht: int,
        result_root: Path,
        docker_manager: DockerManager,
        ros_controller: RosLaunchController,
    ) -> None:
        self.participants = participants
        self.tasks = tasks
        self.runs_per_sht = max(1, int(runs_per_sht))
        self.result_root = result_root
        self.docker_manager = docker_manager
        self.ros = ros_controller

        self.policy_idx: int | None = None
        self.completed_policy_indices: set[int] = set()
        self.sht_idx = 0
        self.repeat_idx = 1
        self.pa_idx = 0
        self.phase = "policy_select"

        self.current_pa_started: float | None = None
        self.awaiting_first_action = False
        self.pending_sht_success: bool | None = None
        self.pending_pa_record: PARecord | None = None
        self.pending_pa_label: str | None = None

        self.current_run_pa_records: list[PARecord] = []
        self.pa_records: list[PARecord] = []
        self.sht_records: list[SHTRecord] = []

        self.motion_enabled = True
        self.records_edit_enabled = False
        self.status = "Ready"
        self.done_message = f"All evaluations finished.\nResults: {self.result_root}"

        self._post_reset_phase: str | None = None
        self._pending_policy_summary: dict[str, Any] | None = None
        self._pending_sht_summary: dict[str, Any] | None = None
        self._post_sht_summary_phase: str | None = None
        self._wait_action_started_mono: float | None = None
        self._last_wait_hint_mono = 0.0
        self._startup_action_gate_pending = True
        self._pending_pa_rosbag_args: dict[str, Any] | None = None

        self._lock = threading.RLock()
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, name="eval-web-poller", daemon=True)
        self._poll_thread.start()

        self.result_root.mkdir(parents=True, exist_ok=True)
        self._set_status("Ready. Select a model to start evaluation.")

    def shutdown(self) -> None:
        with self._lock:
            self._running = False
        self._poll_thread.join(timeout=1.5)
        with self._lock:
            try:
                self.ros.stop()
            except Exception:
                pass
            try:
                self.docker_manager.stop_all_servers()
            except Exception:
                pass

    def _set_status(self, message: str) -> None:
        self.status = str(message)
        print(f"[STATUS] {self.status}")

    def _display_prompt(self, prompt: str, *, max_chars: int = 120) -> str:
        text = " ".join(str(prompt).split())
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "..."

    def _current_participant(self) -> Participant:
        if self.policy_idx is None:
            raise RuntimeError("No policy selected.")
        return self.participants[self.policy_idx]

    def _has_active_policy_locked(self) -> bool:
        return self.policy_idx is not None and 0 <= self.policy_idx < len(self.participants)

    def _remaining_policy_indices_locked(self) -> list[int]:
        return [i for i in range(len(self.participants)) if i not in self.completed_policy_indices]

    def _policy_options_locked(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for i, participant in enumerate(self.participants):
            done = i in self.completed_policy_indices
            items.append({"index": i, "name": participant.name, "done": done})
        return items

    def _select_policy_locked(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.participants):
            raise RuntimeError(f"Invalid policy index: {idx}")

        self.policy_idx = idx
        self.sht_idx = 0
        self.repeat_idx = 1
        self.pa_idx = 0
        self.pending_pa_record = None
        self.pending_pa_label = None
        self.pending_sht_success = None
        self.current_run_pa_records = []
        self.awaiting_first_action = False
        self._wait_action_started_mono = None
        self.current_pa_started = None
        self.motion_enabled = True
        self._post_reset_phase = None
        self._pending_policy_summary = None
        self._pending_sht_summary = None
        self._post_sht_summary_phase = None
        self._pending_pa_rosbag_args = None
        # Require first-action detection when a model is newly loaded/selected.
        self._startup_action_gate_pending = True
        self.phase = "policy_ready"
        self._set_status(f"Ready to start evaluating policy: {self._current_participant().name}")

    def _build_sht_loop_summary_locked(self, *, policy_name: str, sht_name: str) -> dict[str, Any]:
        summary = _compute_policy_summary(
            policy_name=policy_name,
            pa_records=self.pa_records,
            sht_records=self.sht_records,
        )
        per_sht_raw = summary.get("per_sht", {})
        bucket = per_sht_raw.get(sht_name, {}) if isinstance(per_sht_raw, dict) else {}
        if not isinstance(bucket, dict):
            bucket = {}

        trials = int(bucket.get("trials", 0))
        successes = int(bucket.get("successes", 0))
        success_rate = float(bucket.get("success_rate", 0.0)) if trials > 0 else 0.0

        pa_breakdown_raw = bucket.get("pa_breakdown", {})
        pa_breakdown = pa_breakdown_raw if isinstance(pa_breakdown_raw, dict) else {}

        ordered_pa_names: list[str] = []
        seen: set[str] = set()
        for sht in self.tasks:
            key = str(sht.name).strip()
            if key != sht_name:
                continue
            for pa in sht.pas:
                pa_name = str(pa.name).strip()
                if not pa_name or pa_name in seen:
                    continue
                if pa_name in pa_breakdown:
                    ordered_pa_names.append(pa_name)
                    seen.add(pa_name)
            break
        for pa_name in sorted(pa_breakdown.keys()):
            key = str(pa_name).strip()
            if not key or key in seen:
                continue
            ordered_pa_names.append(key)
            seen.add(key)

        pa_rows: list[dict[str, Any]] = []
        for pa_name in ordered_pa_names:
            pa_bucket = pa_breakdown.get(pa_name, {})
            if not isinstance(pa_bucket, dict):
                continue
            total = int(pa_bucket.get("total", 0))
            pa_successes = int(pa_bucket.get("successes", 0))
            pa_rate = float(pa_bucket.get("success_rate", 0.0)) if total > 0 else 0.0
            pa_rows.append(
                {
                    "pa": pa_name,
                    "successes": pa_successes,
                    "total": total,
                    "success_rate": pa_rate,
                }
            )

        return {
            "policy": policy_name,
            "sht": sht_name,
            "sht_successes": successes,
            "sht_trials": trials,
            "sht_success_rate": success_rate,
            "pa_rows": pa_rows,
        }

    def _build_policy_summary_locked(self, policy_name: str) -> dict[str, Any]:
        pa_records = [r for r in self.pa_records if r.policy == policy_name]
        summary = _compute_policy_summary(
            policy_name=policy_name,
            pa_records=self.pa_records,
            sht_records=self.sht_records,
        )

        task_pa_order: list[str] = []
        seen_names: set[str] = set()
        for sht in self.tasks:
            for pa in sht.pas:
                key = str(pa.name).strip()
                if not key or key in seen_names:
                    continue
                seen_names.add(key)
                task_pa_order.append(key)

        grouped: dict[str, list[bool]] = {}
        for rec in pa_records:
            grouped.setdefault(rec.pa, []).append(bool(rec.success))

        ordered_names = [name for name in task_pa_order if name in grouped]
        for name in grouped:
            if name not in ordered_names:
                ordered_names.append(name)

        pa_rates: list[float] = []
        for name in ordered_names:
            rows = grouped.get(name, [])
            if not rows:
                continue
            pa_rates.append(sum(1.0 for v in rows if v) / float(len(rows)))

        per_sht_raw = summary.get("per_sht", {})
        sht_rates: list[dict[str, Any]] = []
        seen_sht: set[str] = set()
        if isinstance(per_sht_raw, dict):
            for sht in self.tasks:
                sht_name = str(sht.name).strip()
                if not sht_name:
                    continue
                bucket = per_sht_raw.get(sht_name)
                if not isinstance(bucket, dict):
                    continue
                trials = int(bucket.get("trials", 0))
                successes = int(bucket.get("successes", 0))
                rate = float(bucket.get("success_rate", 0.0)) if trials > 0 else 0.0
                sht_rates.append(
                    {
                        "name": sht_name,
                        "rate": rate,
                        "rate_percent": rate * 100.0,
                        "successes": successes,
                        "trials": trials,
                    }
                )
                seen_sht.add(sht_name)

            for sht_name, bucket in per_sht_raw.items():
                key = str(sht_name).strip()
                if not key or key in seen_sht or not isinstance(bucket, dict):
                    continue
                trials = int(bucket.get("trials", 0))
                successes = int(bucket.get("successes", 0))
                rate = float(bucket.get("success_rate", 0.0)) if trials > 0 else 0.0
                sht_rates.append(
                    {
                        "name": key,
                        "rate": rate,
                        "rate_percent": rate * 100.0,
                        "successes": successes,
                        "trials": trials,
                    }
                )

        return {
            "policy": policy_name,
            "pa_rates": pa_rates,
            "sht_rates": sht_rates,
            "sht_avg_percent": float(summary.get("average_success_rate", 0.0)) * 100.0,
        }

    def _current_sht(self) -> SHTConfig:
        return self.tasks[self.sht_idx]

    def _current_pa(self):
        return self._current_sht().pas[self.pa_idx]

    def _poll_loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    break
                should_poll = self.awaiting_first_action and self.phase == "running_pa_wait_action"
            if should_poll:
                try:
                    with self._lock:
                        self._poll_action_output_locked()
                        self._maybe_emit_wait_hint_locked()
                except Exception as exc:
                    with self._lock:
                        self._set_status(f"Action polling warning: {exc}")
            time.sleep(0.4)

    def _reset_action_output_for_new_run_locked(self) -> None:
        self.ros.reset_action_output_flag()
        self._last_wait_hint_mono = 0.0

    def _set_pending_pa_rosbag_locked(
        self,
        *,
        policy_name: str,
        sht_name: str,
        repeat_index: int,
        pa_index: int,
        pa_name: str,
    ) -> None:
        self._pending_pa_rosbag_args = {
            "policy_name": policy_name,
            "sht_name": sht_name,
            "repeat_index": int(repeat_index),
            "pa_index": int(pa_index),
            "pa_name": pa_name,
        }

    def _start_pending_pa_rosbag_locked(self) -> None:
        if self._pending_pa_rosbag_args is None:
            return
        rosbag_args = dict(self._pending_pa_rosbag_args)
        self.ros.start_pa_rosbag(**rosbag_args)
        self._pending_pa_rosbag_args = None

    def _poll_action_output_locked(self) -> None:
        try:
            has_action_output = self.ros.has_action_output()
        except Exception as exc:
            now_mono = time.monotonic()
            if now_mono - self._last_wait_hint_mono >= 5.0:
                self._last_wait_hint_mono = now_mono
                self._set_status(f"Waiting for action output flag service... ({exc})")
            return
        if has_action_output:
            self._on_first_action_detected_locked()

    def _maybe_emit_wait_hint_locked(self) -> None:
        if not self.awaiting_first_action or self.phase != "running_pa_wait_action":
            return
        if self._wait_action_started_mono is None:
            return
        now_mono = time.monotonic()
        waited = max(0.0, now_mono - self._wait_action_started_mono)
        if waited < 8.0:
            return
        if now_mono - self._last_wait_hint_mono < 10.0:
            return
        self._last_wait_hint_mono = now_mono
        self._set_status(
            f"Still waiting for first action output flag ({waited:.1f}s)..."
        )

    def _on_first_action_detected_locked(self) -> None:
        if not self.awaiting_first_action:
            return
        self._start_pending_pa_rosbag_locked()
        self._startup_action_gate_pending = False
        self.awaiting_first_action = False
        self._wait_action_started_mono = None
        self.current_pa_started = time.monotonic()
        self.phase = "running_pa"
        self._set_status("First action detected. Timer started; Success/Fail enabled.")

    def _persist_policy_outputs_locked(self, policy_name: str) -> None:
        policy_dir = self.result_root / _slugify(policy_name)
        policy_dir.mkdir(parents=True, exist_ok=True)

        pa_payload = [r.to_dict() for r in self.pa_records if r.policy == policy_name][::-1]
        sht_payload = [r.to_dict() for r in self.sht_records if r.policy == policy_name][::-1]
        _write_json(policy_dir / "records_pa.json", pa_payload)
        _write_json(policy_dir / "records_sht.json", sht_payload)

        summary = _compute_policy_summary(policy_name=policy_name, pa_records=self.pa_records, sht_records=self.sht_records)
        _write_json(policy_dir / "summary.json", summary)
        _write_sr_txt(policy_dir / "SR.txt", summary)
        self._update_histogram_locked()

    def _update_histogram_locked(self) -> None:
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

    def _begin_sht_run_locked(self) -> None:
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
            self._set_pending_pa_rosbag_locked(
                policy_name=participant.name,
                sht_name=sht.name,
                repeat_index=self.repeat_idx,
                pa_index=self.pa_idx,
                pa_name=pa.name,
            )
            if not self._startup_action_gate_pending:
                self._start_pending_pa_rosbag_locked()
            self.ros.set_instruction(pa.prompt)
            self.ros.set_motion_enabled(True)
            if self._startup_action_gate_pending:
                self._reset_action_output_for_new_run_locked()
        except Exception as exc:
            try:
                self.ros.stop()
            except Exception:
                pass
            self.phase = "sht_ready"
            self.motion_enabled = True
            self.awaiting_first_action = False
            self._wait_action_started_mono = None
            self._pending_pa_rosbag_args = None
            self._set_status(f"Failed to start run: {exc}")
            raise

        self.current_run_pa_records = []
        self.current_pa_started = None
        self._post_reset_phase = None
        self.motion_enabled = True
        if self._startup_action_gate_pending:
            self.awaiting_first_action = True
            self._wait_action_started_mono = time.monotonic()
            self.phase = "running_pa_wait_action"
            self._set_status("Waiting for first action output...")
            # Try once immediately so UI can switch to running without waiting for poll loop tick.
            self._poll_action_output_locked()
        else:
            self.awaiting_first_action = False
            self._wait_action_started_mono = None
            self.phase = "running_pa"
            self.current_pa_started = time.monotonic()
            self._set_status(
                f"SHT started. Timer started: {self._display_prompt(pa.prompt, max_chars=80)}"
            )

    def _commit_pa_and_continue_locked(self, record: PARecord) -> None:
        self.pa_records.append(record)
        self.current_run_pa_records.append(record)
        self._persist_policy_outputs_locked(record.policy)

        pa_count = len(self._current_sht().pas)
        if self.pa_idx + 1 < pa_count:
            self.pa_idx += 1
            next_pa = self._current_pa()
            try:
                self.ros.start_pa_rosbag(
                    policy_name=record.policy,
                    sht_name=record.sht,
                    repeat_index=record.repeat_index,
                    pa_index=self.pa_idx,
                    pa_name=next_pa.name,
                )
                self.ros.set_motion_enabled(True)
                self.ros.set_instruction(next_pa.prompt)
            except Exception as exc:
                self._set_status(f"Failed to prepare next PA: {exc}")
                raise
            self.motion_enabled = True
            self.phase = "running_pa"
            self.current_pa_started = time.monotonic()
            self.awaiting_first_action = False
            self._wait_action_started_mono = None
            self._set_status(
                f"Moved to next PA. Timer started: {self._display_prompt(next_pa.prompt, max_chars=80)}"
            )
            return

        self.phase = "awaiting_sht_result"
        self.awaiting_first_action = False
        self._wait_action_started_mono = None

    def _enter_reset_phase_locked(self, *, next_phase: str, policy_name: str | None = None) -> None:
        self.phase = "awaiting_robot_reset"
        self._post_reset_phase = next_phase
        if next_phase in {"policy_select", "policy_summary", "sht_summary"}:
            name = (policy_name or "").strip() or "current model"
            if next_phase == "policy_summary":
                self._set_status(
                    f"Please reset robot to initial pose before showing summary: {name}"
                )
                return
            if next_phase == "sht_summary":
                participant = self._current_participant().name if self._has_active_policy_locked() else name
                sht = self._current_sht().name if self._has_active_policy_locked() else "SHT"
                self._set_status(
                    f"Please reset robot to initial pose before showing SHT summary: {participant} / {sht}"
                )
                return
            self._set_status("Please reset robot to initial pose before selecting next model.")
            return
        participant = self._current_participant().name
        sht = self._current_sht().name
        if next_phase == "policy_ready":
            self._set_status(
                f"Please reset robot to initial pose before next policy: {participant}"
            )
        else:
            self._set_status(
                f"Please reset robot to initial pose before next run: {participant} / {sht} / eval {self.repeat_idx}"
            )

    def _finalize_current_sht_run_locked(self) -> None:
        if self.pending_sht_success is None:
            raise RuntimeError("SHT result is not selected.")

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
        self._persist_policy_outputs_locked(sht_record.policy)

        self.ros.stop()
        self.awaiting_first_action = False
        self._wait_action_started_mono = None
        self.motion_enabled = True
        self._pending_pa_rosbag_args = None
        self._set_status("SHT run completed. Exec trace should be saved under result directory.")

        if self.repeat_idx < self.runs_per_sht:
            self.repeat_idx += 1
            self.pa_idx = 0
            self.pending_pa_record = None
            self.pending_pa_label = None
            self._enter_reset_phase_locked(next_phase="sht_ready")
            return

        # SHT loop is complete (all repeats finished): show summary before advancing.
        policy_name = self._current_participant().name
        completed_sht_name = self._current_sht().name
        self._pending_sht_summary = self._build_sht_loop_summary_locked(
            policy_name=policy_name,
            sht_name=completed_sht_name,
        )
        if self.sht_idx + 1 < len(self.tasks):
            self._post_sht_summary_phase = "next_sht"
        else:
            self._post_sht_summary_phase = "complete_policy"
        self._enter_reset_phase_locked(next_phase="sht_summary", policy_name=policy_name)

    def _can_force_sht_fail_locked(self) -> bool:
        if not self._has_active_policy_locked():
            return False
        if self.phase != "awaiting_pa_next":
            return False
        if self.pending_pa_record is None:
            return False
        return not bool(self.pending_pa_record.success)

    def _append_fail_for_remaining_pas_locked(self) -> int:
        if not self._has_active_policy_locked():
            return 0
        sht = self._current_sht()
        participant = self._current_participant().name
        recorded_count = max(0, min(len(self.current_run_pa_records), len(sht.pas)))
        now = _iso_now()
        added = 0
        for idx in range(recorded_count, len(sht.pas)):
            pa = sht.pas[idx]
            rec = PARecord(
                policy=participant,
                sht=sht.name,
                repeat_index=self.repeat_idx,
                pa=pa.name,
                prompt=pa.prompt,
                success=False,
                elapsed_sec=0.0,
                recorded_at=now,
            )
            self.pa_records.append(rec)
            self.current_run_pa_records.append(rec)
            added += 1
        return added

    def _complete_policy_locked(self) -> None:
        if self.policy_idx is None:
            raise RuntimeError("No active policy to complete.")
        finished = self._current_participant()
        self._set_status(f"Policy completed: {finished.name}")
        self._persist_policy_outputs_locked(finished.name)
        self.docker_manager.stop_server(self.policy_idx)
        self.completed_policy_indices.add(self.policy_idx)

        self.policy_idx = None
        self.sht_idx = 0
        self.repeat_idx = 1
        self.pa_idx = 0
        self.pending_pa_record = None
        self.pending_pa_label = None
        self.pending_sht_success = None
        self.current_run_pa_records = []
        self.awaiting_first_action = False
        self._wait_action_started_mono = None
        self.current_pa_started = None
        self.motion_enabled = True
        self._post_reset_phase = None
        self._pending_policy_summary = None
        self._pending_sht_summary = None
        self._post_sht_summary_phase = None
        self._pending_pa_rosbag_args = None
        self.phase = "policy_select"
        self._set_status("Select a model to start evaluation.")

    def _terminate_session_locked(self, *, status_message: str, done_message: str) -> None:
        try:
            self.ros.stop()
        except Exception:
            pass
        try:
            self.docker_manager.stop_all_servers()
        except Exception:
            pass
        self.phase = "done"
        self.awaiting_first_action = False
        self._wait_action_started_mono = None
        self.motion_enabled = True
        self.pending_pa_record = None
        self.pending_pa_label = None
        self.pending_sht_success = None
        self._post_reset_phase = None
        self._pending_policy_summary = None
        self._pending_sht_summary = None
        self._post_sht_summary_phase = None
        self._pending_pa_rosbag_args = None
        self.done_message = done_message
        self._update_histogram_locked()
        self._set_status(status_message)

    def handle_action(self, action: str, payload: dict[str, Any]) -> tuple[bool, str | None, dict[str, Any]]:
        try:
            with self._lock:
                if action == "start_resume":
                    if self.phase in {"idle", "policy_select"}:
                        self.phase = "policy_select"
                        self._set_status("Select a model to start evaluation.")
                    elif self.phase == "policy_summary":
                        self._set_status("Review summary and continue to model selection.")
                    elif self.phase == "sht_summary":
                        self._set_status("Review SHT summary and continue.")
                    elif self.phase == "policy_ready":
                        if not self._has_active_policy_locked():
                            self.phase = "policy_select"
                            self._set_status("No model selected. Select a model first.")
                        else:
                            self._set_status(f"Ready to start evaluating policy: {self._current_participant().name}")
                    elif self.phase == "sht_ready":
                        p = self._current_participant().name
                        s = self._current_sht().name
                        self._set_status(f"Ready: {p} / {s} / eval {self.repeat_idx}")
                    elif self.phase == "running_pa_wait_action":
                        self._set_status("Still waiting for first action output...")
                    elif self.phase == "awaiting_robot_reset":
                        self._set_status("Robot reset confirmation is pending.")
                    elif self.phase == "awaiting_sht_review":
                        self._set_status("SHT review pending. Confirm next SHT or close for later.")
                    elif self.phase == "sht_review_closed":
                        self.phase = "awaiting_sht_review"
                        self._set_status("SHT review reopened. Confirm next SHT when ready.")
                    elif self.phase == "awaiting_sht_result":
                        self._set_status("Please select SHT Success/Fail.")
                    elif self.phase == "done":
                        self._set_status("All policies completed.")
                elif action == "select_policy":
                    if self.phase != "policy_select":
                        raise RuntimeError(f"Cannot select policy while phase={self.phase}")
                    idx_raw = payload.get("index")
                    try:
                        idx = int(idx_raw)
                    except Exception:
                        raise RuntimeError(f"Invalid policy index payload: {idx_raw}")
                    self._select_policy_locked(idx)
                elif action == "start_policy":
                    if self.phase not in {"idle", "policy_ready"}:
                        raise RuntimeError(f"Cannot start policy while phase={self.phase}")
                    if not self._has_active_policy_locked():
                        raise RuntimeError("No model selected.")
                    self.pa_idx = 0
                    self._begin_sht_run_locked()
                elif action == "reselect_policy":
                    if self.phase != "policy_ready":
                        raise RuntimeError(f"Cannot reselect policy while phase={self.phase}")
                    self.policy_idx = None
                    self.sht_idx = 0
                    self.repeat_idx = 1
                    self.pa_idx = 0
                    self.pending_pa_record = None
                    self.pending_pa_label = None
                    self.pending_sht_success = None
                    self.current_run_pa_records = []
                    self.awaiting_first_action = False
                    self._wait_action_started_mono = None
                    self.current_pa_started = None
                    self.motion_enabled = True
                    self._post_reset_phase = None
                    self._pending_sht_summary = None
                    self._post_sht_summary_phase = None
                    self.phase = "policy_select"
                    self._set_status("Model selection reopened. Choose a model.")
                elif action == "confirm_policy_summary":
                    if self.phase != "policy_summary":
                        raise RuntimeError(f"Cannot confirm policy summary while phase={self.phase}")
                    self._pending_policy_summary = None
                    self.phase = "policy_select"
                    self._set_status("Select a model to start evaluation.")
                elif action == "start_sht":
                    if self.phase != "sht_ready":
                        raise RuntimeError(f"Cannot start SHT while phase={self.phase}")
                    self._begin_sht_run_locked()
                elif action == "toggle_motion":
                    if not self.ros.is_running():
                        raise RuntimeError("roslaunch is not running yet.")
                    target_enabled = not self.motion_enabled
                    self.ros.set_motion_enabled(target_enabled)
                    self.motion_enabled = target_enabled
                    if self.motion_enabled:
                        self._set_status("Robot command forwarding resumed.")
                    else:
                        self._set_status("Robot command forwarding stopped (inference keeps running).")
                elif action == "pa_result":
                    if self.phase != "running_pa":
                        raise RuntimeError(f"Cannot submit PA result while phase={self.phase}")
                    success = bool(payload.get("success", False))
                    try:
                        if self.ros.is_running():
                            self.ros.set_motion_enabled(False)
                    except Exception as exc:
                        self._set_status(f"Warning: failed to stop robot motion immediately: {exc}")
                    try:
                        if self.ros.is_running():
                            self.ros.stop_pa_rosbag()
                    except Exception as exc:
                        self._set_status(f"Warning: failed to finalize rosbag for current PA: {exc}")
                    self.motion_enabled = False

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
                        success=success,
                        elapsed_sec=elapsed,
                        recorded_at=_iso_now(),
                    )
                    pa_total = len(self._current_sht().pas)
                    has_next_pa = (self.pa_idx + 1) < pa_total
                    if has_next_pa:
                        self.pending_pa_record = rec
                        self.pending_pa_label = "Success" if success else "Fail"
                        self.phase = "awaiting_pa_next"
                    else:
                        self.pending_pa_record = None
                        self.pending_pa_label = None
                        self._commit_pa_and_continue_locked(rec)
                elif action == "next_pa":
                    if self.phase != "awaiting_pa_next" or self.pending_pa_record is None:
                        raise RuntimeError(f"Cannot move next PA while phase={self.phase}")
                    record = self.pending_pa_record
                    self.pending_pa_record = None
                    self.pending_pa_label = None
                    self._commit_pa_and_continue_locked(record)
                elif action == "select_sht_result":
                    if self.phase != "awaiting_sht_result":
                        raise RuntimeError(f"Cannot select SHT result while phase={self.phase}")
                    self.pending_sht_success = bool(payload.get("success", False))
                    self.phase = "awaiting_sht_review"
                elif action == "confirm_robot_reset":
                    if self.phase != "awaiting_robot_reset":
                        raise RuntimeError(f"Cannot confirm robot reset while phase={self.phase}")
                    next_phase = self._post_reset_phase or ("policy_ready" if self._has_active_policy_locked() else "policy_select")
                    self._post_reset_phase = None
                    self.phase = next_phase
                    if next_phase == "policy_select":
                        self._set_status("Select a model to start evaluation.")
                    elif next_phase == "policy_summary":
                        self._set_status("Policy summary is ready.")
                    elif next_phase == "sht_summary":
                        self._set_status("SHT summary is ready.")
                    elif next_phase == "policy_ready":
                        self._set_status(
                            f"Ready to start evaluating policy: {self._current_participant().name}"
                        )
                    else:
                        p = self._current_participant().name
                        s = self._current_sht().name
                        self._set_status(f"Ready: {p} / {s} / eval {self.repeat_idx}")
                elif action == "confirm_next_sht":
                    if self.phase != "awaiting_sht_review":
                        raise RuntimeError(f"Cannot confirm next SHT while phase={self.phase}")
                    self._finalize_current_sht_run_locked()
                elif action == "confirm_sht_summary":
                    if self.phase != "sht_summary":
                        raise RuntimeError(f"Cannot confirm SHT summary while phase={self.phase}")
                    next_step = (self._post_sht_summary_phase or "").strip()
                    self._pending_sht_summary = None
                    self._post_sht_summary_phase = None
                    if next_step == "next_sht":
                        if self.sht_idx + 1 < len(self.tasks):
                            self.sht_idx += 1
                            self.repeat_idx = 1
                            self.pa_idx = 0
                            self.pending_pa_record = None
                            self.pending_pa_label = None
                            self.pending_sht_success = None
                            self.current_run_pa_records = []
                            self.phase = "sht_ready"
                            self._set_status(
                                f"Ready: {self._current_participant().name} / {self._current_sht().name} / eval {self.repeat_idx}"
                            )
                        else:
                            self._complete_policy_locked()
                    elif next_step == "complete_policy":
                        self._complete_policy_locked()
                    else:
                        self._complete_policy_locked()
                elif action == "close_review":
                    if self.phase != "awaiting_sht_review":
                        raise RuntimeError(f"Cannot close review while phase={self.phase}")
                    self.phase = "sht_review_closed"
                    self._set_status("SHT review closed. Press Start / Resume to continue.")
                elif action == "force_sht_fail":
                    if not self._can_force_sht_fail_locked():
                        raise RuntimeError(f"Cannot force SHT fail while phase={self.phase}")
                    try:
                        if self.ros.is_running():
                            self.ros.set_motion_enabled(False)
                    except Exception as exc:
                        self._set_status(f"Warning: failed to stop robot motion immediately: {exc}")
                    try:
                        if self.ros.is_running():
                            self.ros.stop_pa_rosbag()
                    except Exception as exc:
                        self._set_status(f"Warning: failed to finalize rosbag for forced-fail PA: {exc}")
                    self.motion_enabled = False
                    self.awaiting_first_action = False
                    self._wait_action_started_mono = None
                    # Commit current PA result first (the operator already chose Fail),
                    # then fill the rest of PAs in this run as Fail.
                    if self.pending_pa_record is not None:
                        self.pa_records.append(self.pending_pa_record)
                        self.current_run_pa_records.append(self.pending_pa_record)
                    self.pending_pa_record = None
                    self.pending_pa_label = None

                    added = self._append_fail_for_remaining_pas_locked()
                    self.pending_sht_success = False
                    self._set_status(
                        f"SHT forced to Fail. Added {added} unrecorded PA result(s) as Fail."
                    )
                    self._finalize_current_sht_run_locked()
                elif action == "set_records_edit":
                    self.records_edit_enabled = bool(payload.get("enabled", False))
                elif action == "edit_record":
                    global_idx = int(payload.get("global_idx"))
                    new_success = bool(payload.get("success", False))
                    if global_idx < 0 or global_idx >= len(self.pa_records):
                        raise RuntimeError(f"Invalid record index: {global_idx}")
                    rec = self.pa_records[global_idx]
                    if rec.success != new_success:
                        rec.success = new_success
                        rec.recorded_at = _iso_now()
                        self._persist_policy_outputs_locked(rec.policy)
                        self._set_status(
                            f"Edited PA result: {self._display_prompt(rec.prompt, max_chars=80)} -> "
                            f"{'Success' if new_success else 'Fail'}"
                        )
                elif action == "terminate_session":
                    if self.phase == "done":
                        self._set_status("Session is already terminated.")
                    else:
                        self._terminate_session_locked(
                            status_message="Session terminated by operator.",
                            done_message=f"Evaluation session terminated by operator.\nResults: {self.result_root}",
                        )
                else:
                    raise RuntimeError(f"Unknown action: {action}")

                return True, None, self.snapshot_locked()
        except Exception as exc:
            traceback.print_exc()
            with self._lock:
                self._set_status(f"Action failed ({action}): {exc}")
                return False, str(exc), self.snapshot_locked()

    def _build_modal_locked(self) -> dict[str, Any] | None:
        if self.phase in {"idle", "policy_select"}:
            return {
                "type": "policy_select",
                "title": "Select Model",
                "message": "Choose a model to evaluate next.",
                "items": self._policy_options_locked(),
            }
        if self.phase == "policy_summary":
            summary = dict(self._pending_policy_summary or {})
            pa_rates = list(summary.get("pa_rates") or [])
            pa_text = " - ".join(f"{float(v):.2f}" for v in pa_rates) if pa_rates else "n/a"
            sht_rates = list(summary.get("sht_rates") or [])
            if sht_rates:
                parts: list[str] = []
                for row in sht_rates:
                    try:
                        name = str(row.get("name", "")).strip() or "SHT"
                        rate_percent = float(row.get("rate_percent", 0.0))
                        successes = int(row.get("successes", 0))
                        trials = int(row.get("trials", 0))
                    except Exception:
                        continue
                    percent_text = f"{rate_percent:.1f}".rstrip("0").rstrip(".")
                    parts.append(f"{name}: {percent_text}% ({successes}/{trials})")
                sht_detail_text = "\n".join(parts) if parts else "n/a"
            else:
                sht_detail_text = "n/a"
            sht_avg_percent = float(summary.get("sht_avg_percent", 0.0))
            sht_avg_text = f"{sht_avg_percent:.1f}".rstrip("0").rstrip(".")
            policy_name = str(summary.get("policy", "")).strip() or "Policy"
            return {
                "type": "policy_summary",
                "title": f"Summary: {policy_name}",
                "message": (
                    f"PA: {pa_text}\n"
                    f"SHT:\n{sht_detail_text}\n"
                    f"SHT AVG: {sht_avg_text}%"
                ),
                "button_text": "Continue to model selection",
            }
        if self.phase == "sht_summary":
            summary = dict(self._pending_sht_summary or {})
            policy_name = str(summary.get("policy", "")).strip() or "Policy"
            sht_name = str(summary.get("sht", "")).strip() or "SHT"
            successes = int(summary.get("sht_successes", 0))
            trials = int(summary.get("sht_trials", 0))
            sht_rate = float(summary.get("sht_success_rate", 0.0)) if trials > 0 else 0.0
            pa_rows = list(summary.get("pa_rows") or [])
            lines = [
                f"sht_success: {successes}/{trials}",
                f"sht_success_rate: {sht_rate:.4f}",
                "",
                "per_pa:",
            ]
            if pa_rows:
                for row in pa_rows:
                    try:
                        pa_name = str(row.get("pa", "")).strip() or "PA"
                        pa_successes = int(row.get("successes", 0))
                        pa_total = int(row.get("total", 0))
                        pa_rate = float(row.get("success_rate", 0.0)) if pa_total > 0 else 0.0
                    except Exception:
                        continue
                    lines.append(
                        f"- {pa_name}: pa_success={pa_successes}/{pa_total}, pa_success_rate={pa_rate:.4f}"
                    )
            else:
                lines.append("- n/a")
            button_text = "Continue to next SHT"
            if self._post_sht_summary_phase == "complete_policy":
                button_text = "Continue to model selection"
            return {
                "type": "sht_summary",
                "title": f"SHT Summary: {policy_name} / {sht_name}",
                "message": "\n".join(lines),
                "button_text": button_text,
            }
        if self.phase == "awaiting_robot_reset":
            return {
                "type": "robot_reset",
                "title": "Reset Robot",
                "message": "Reset robot to initial pose, then continue.",
                "button_text": "Reset done",
            }
        if self.phase == "policy_ready":
            if not self._has_active_policy_locked():
                return {
                    "type": "policy_select",
                    "title": "Select Model",
                    "message": "Choose a model to evaluate next.",
                    "items": self._policy_options_locked(),
                }
            participant = self._current_participant()
            return {
                "type": "start_policy",
                "title": "Start Policy",
                "message": (
                    f"Start evaluating {participant.name}.\n\n"
                    "Clicking this button will immediately launch the client deploy node "
                    "for the model-assigned policy server port."
                ),
                "button_text": f"Start evaluating {participant.name}",
            }
        if self.phase == "sht_ready":
            participant = self._current_participant()
            sht = self._current_sht()
            pa = self._current_pa()
            return {
                "type": "start_sht",
                "title": "Start SHT",
                "message": (
                    "Press start when ready.\n\n"
                    f"Policy: {participant.name}\n"
                    f"SHT: {sht.name}\n"
                    f"Eval: {self.repeat_idx}\n"
                    f"Prompt: {self._display_prompt(pa.prompt, max_chars=220)}"
                ),
                "button_text": f"Start {sht.name} eval {self.repeat_idx}",
            }
        if self.phase == "awaiting_pa_next" and self.pending_pa_record is not None:
            rec = self.pending_pa_record
            label = self.pending_pa_label or ("Success" if rec.success else "Fail")
            return {
                "type": "pa_result",
                "title": "PA Result",
                "message": (
                    f"{self._display_prompt(rec.prompt, max_chars=220)}\n"
                    f"Result: {label}\n"
                    f"Elapsed: {rec.elapsed_sec:.1f} sec"
                ),
                "button_text": "Next PA",
                "show_force_sht_fail": bool(not rec.success),
            }
        if self.phase == "awaiting_sht_result":
            sht = self._current_sht()
            return {
                "type": "sht_result",
                "title": "SHT Result",
                "message": f"Select SHT result for {sht.name} (eval {self.repeat_idx})",
            }
        if self.phase == "awaiting_sht_review":
            items: list[dict[str, Any]] = []
            for i, rec in enumerate(self.current_run_pa_records, start=1):
                pa_label = str(rec.pa).strip() or f"PA_{i}"
                items.append(
                    {
                        "pa": pa_label,
                        "success": bool(rec.success),
                    }
                )
            return {
                "type": "sht_review",
                "title": "Review SHT Records",
                "message": "Review PA results before moving next.",
                "items": items,
            }
        if self.phase == "done":
            return {
                "type": "done",
                "title": "Completed",
                "message": self.done_message,
            }
        return None

    def snapshot_locked(self) -> dict[str, Any]:
        participant = self._current_participant() if self._has_active_policy_locked() else None
        sht = self._current_sht() if participant is not None else None
        pa = self._current_pa() if participant is not None else None

        elapsed = 0.0
        if self.phase == "running_pa" and self.current_pa_started is not None:
            elapsed = max(0.0, time.monotonic() - self.current_pa_started)

        records: list[dict[str, Any]] = []
        if participant is not None:
            indices = list(reversed([i for i, r in enumerate(self.pa_records) if r.policy == participant.name]))
            for idx in indices:
                rec = self.pa_records[idx]
                records.append(
                    {
                        "global_idx": idx,
                        "sht": rec.sht,
                        "repeat_index": rec.repeat_index,
                        "pa": rec.pa,
                        "prompt": rec.prompt,
                        "prompt_display": self._display_prompt(rec.prompt, max_chars=130),
                        "elapsed_sec": rec.elapsed_sec,
                        "success": rec.success,
                        "recorded_at": rec.recorded_at,
                    }
                )

        return {
            "phase": self.phase,
            "status": self.status,
            "motion_enabled": self.motion_enabled,
            "policy": participant.name if participant is not None else "-",
            "sht": sht.name if sht is not None else "-",
            "pa_prompt": self._display_prompt(pa.prompt, max_chars=220) if pa is not None else "-",
            "pa_index": (self.pa_idx + 1) if pa is not None else 0,
            "pa_total": len(sht.pas) if sht is not None else 0,
            "repeat_index": self.repeat_idx if participant is not None else 0,
            "runs_per_sht": self.runs_per_sht,
            "elapsed_sec": elapsed,
            "records_edit_enabled": self.records_edit_enabled,
            "policy_options": self._policy_options_locked(),
            "records": records,
            "modal": self._build_modal_locked(),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self.snapshot_locked()


HTML_PAGE_TEMPLATE = """<!doctype html>
<html lang=\"en\"> 
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>AIROA Competition Evaluator (Web)</title>
  <style>
    :root {
      --bg: #e8eef4;
      --card: #ffffff;
      --text: #1f2a38;
      --muted: #5d6b7c;
      --ok: #cfeecf;
      --fail: #f7cccc;
      --btn: #cde0f9;
      --warn: #ffd2d2;
      --line: #d9e2ec;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .wrap {
      max-width: 1320px;
      margin: 0 auto;
      padding: 18px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px 16px;
      margin-bottom: 14px;
    }
    .head-row {
      display: grid;
      grid-template-columns: 130px 1fr;
      gap: 8px 12px;
      align-items: center;
      margin-bottom: 4px;
    }
    .head-key { font-weight: 700; font-size: 22px; }
    .head-val { font-size: 21px; }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
    }
    button {
      font-size: 18px;
      font-weight: 700;
      border: 1px solid #b7c5d1;
      border-radius: 10px;
      padding: 12px 18px;
      cursor: pointer;
      background: #eef3f9;
    }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    .btn-ok { background: var(--ok); }
    .btn-fail { background: var(--fail); }
    .btn-main { background: var(--btn); }
    .btn-motion-stop { background: var(--warn); }
    .btn-motion-run { background: #d8f5d8; }
    .btn-end { background: #f5cfaa; }
    .btn-done {
      background: #e2e8ef;
      color: #5b6775;
      border-color: #c6d0db;
    }
    .session-row {
      display: flex;
      justify-content: flex-end;
      align-items: center;
    }

    .records-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .records-title { font-size: 24px; font-weight: 700; }
    .records-list {
      max-height: calc(100vh - 320px);
      overflow: auto;
      padding-right: 4px;
    }
    .row {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      padding: 10px;
      margin-bottom: 8px;
    }
    .summary {
      font-family: "Courier New", monospace;
      font-size: 15px;
      white-space: pre-wrap;
      margin-bottom: 8px;
    }
    .prompt {
      font-family: "Courier New", monospace;
      font-size: 15px;
      white-space: pre-wrap;
      color: #2a3b52;
      margin-bottom: 8px;
    }
    .row-bottom {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .radio-group label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin-right: 12px;
      font-size: 15px;
      font-weight: 700;
      padding: 6px 10px;
      border-radius: 8px;
      border: 1px solid #cbd5df;
      background: #f6f9fc;
    }
    .timestamp { color: var(--muted); font-size: 13px; }
    .muted { color: var(--muted); }

    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.42);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 1000;
      padding: 18px;
    }
    .modal {
      width: min(860px, 95vw);
      background: #fff;
      border-radius: 12px;
      border: 1px solid #b9c8d6;
      padding: 18px;
      box-shadow: 0 18px 40px rgba(0,0,0,0.2);
    }
    .modal-title { font-size: 24px; font-weight: 700; margin-bottom: 10px; }
    .modal-text { white-space: pre-wrap; font-size: 18px; line-height: 1.45; margin-bottom: 14px; }
    .modal-actions { display: flex; flex-wrap: wrap; gap: 10px; }
    .busy-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.42);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 1100;
      padding: 18px;
    }
    .busy-box {
      width: min(560px, 92vw);
      background: #fff;
      border-radius: 12px;
      border: 1px solid #b9c8d6;
      padding: 22px 18px;
      box-shadow: 0 18px 40px rgba(0,0,0,0.2);
      text-align: center;
    }
    .busy-title { font-size: 24px; font-weight: 700; margin-bottom: 10px; }
    .busy-text { white-space: pre-wrap; font-size: 18px; line-height: 1.45; }
    .review-box {
      max-height: 280px;
      overflow: auto;
      background: #f8fafc;
      border: 1px solid #d7e1ea;
      border-radius: 8px;
      padding: 10px;
      margin-bottom: 12px;
    }
    .review-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      border: 1px solid #d3dde7;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 19px;
      margin-bottom: 8px;
      background: #ffffff;
    }
    .review-item:last-child { margin-bottom: 0; }
    .review-item-ok {
      background: #eaf8ea;
      border-color: #b9dfb9;
    }
    .review-item-fail {
      background: #fdeeee;
      border-color: #efbcbc;
    }
    .review-pa {
      font-weight: 800;
      color: #243549;
    }
    .review-result {
      font-weight: 900;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid transparent;
    }
    .review-result-ok {
      color: #1f6b1f;
      background: #dff2df;
      border-color: #b9dfb9;
    }
    .review-result-fail {
      color: #8f1f1f;
      background: #f9dede;
      border-color: #efbcbc;
    }

    @media (max-width: 860px) {
      .head-key { font-size: 18px; }
      .head-val { font-size: 18px; }
      button { width: 100%; }
      .controls { display: grid; grid-template-columns: 1fr; }
      .records-list { max-height: none; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"card\">
      <div class=\"head-row\"><div class=\"head-key\">Policy:</div><div id=\"policy\" class=\"head-val\">-</div></div>
      <div class=\"head-row\"><div class=\"head-key\">SHT:</div><div id=\"sht\" class=\"head-val\">-</div></div>
      <div class=\"head-row\"><div class=\"head-key\">PA:</div><div id=\"pa\" class=\"head-val\">-</div></div>
      <div class=\"head-row\"><div class=\"head-key\">Elapsed:</div><div id=\"elapsed\" class=\"head-val\">0.0 sec</div></div>
      <div class=\"head-row\"><div class=\"head-key\">Status:</div><div id=\"status\" class=\"head-val muted\">Ready</div></div>
    </div>

    <div class=\"controls\">
      <button id=\"btn-success\" class=\"btn-ok\">Success</button>
      <button id=\"btn-fail\" class=\"btn-fail\">Fail</button>
      <button id=\"btn-start\" class=\"btn-main\">Start / Resume</button>
      <button id=\"btn-motion\" class=\"btn-motion-stop\">Robot STOP (no-send)</button>
    </div>

    <div class=\"card\">
      <div class=\"records-head\">
        <div class=\"records-title\">PA Result Records</div>
        <label><input id=\"edit-toggle\" type=\"checkbox\" /> Edit Enabled</label>
      </div>
      <div id=\"records\" class=\"records-list\"></div>
    </div>

    <div class=\"card\">
      <div class=\"records-head\">
        <div class=\"records-title\">Session Controls</div>
      </div>
      <div class=\"session-row\">
        <button id=\"btn-end\" class=\"btn-end\">End Session</button>
      </div>
    </div>
  </div>

  <div id=\"modal-backdrop\" class=\"modal-backdrop\">
    <div class=\"modal\">
      <div id=\"modal-title\" class=\"modal-title\"></div>
      <div id=\"modal-text\" class=\"modal-text\"></div>
      <div id=\"modal-review\" class=\"review-box\" style=\"display:none\"></div>
      <div id=\"modal-actions\" class=\"modal-actions\"></div>
    </div>
  </div>

  <div id=\"busy-backdrop\" class=\"busy-backdrop\">
    <div class=\"busy-box\">
      <div id=\"busy-title\" class=\"busy-title\">Saving PA Rosbag</div>
      <div id=\"busy-text\" class=\"busy-text\">Finalizing the rosbag for this PA.\nPlease wait...</div>
    </div>
  </div>

  <script id="initial-state" type="application/json">__INITIAL_STATE_JSON__</script>
  <script>
    (function () {
      var stateRef = { value: null };
      var lastModalSig = null;
      var requestInFlight = false;
      var fetchInFlight = false;
      var lastActionStartedAtMs = 0;
      var INITIAL_STATE = null;
      var suppressedModalSig = null;
      var suppressModalUntilMs = 0;

      function byId(id) {
        return document.getElementById(id);
      }

      function setTextIfPresent(id, text) {
        var el = byId(id);
        if (!el) return;
        el.textContent = text;
      }

      function parseInitialState() {
        var el = byId("initial-state");
        if (!el) return null;
        try {
          return JSON.parse(el.textContent || el.innerText || "{}");
        } catch (err) {
          console.error("initial state parse failed", err);
          return null;
        }
      }

      function esc(text) {
        return String(text == null ? "" : text)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/\"/g, "&quot;")
          .replace(/'/g, "&#39;");
      }

      function fmtSec(v) {
        var n = Number(v || 0);
        return n.toFixed(1) + " sec";
      }

      function pad3(n) {
        var s = String(n);
        while (s.length < 3) s = "0" + s;
        return s;
      }

      function httpJSON(method, path, payload, cb) {
        var xhr = new XMLHttpRequest();
        xhr.open(method, path, true);
        if (method === "POST") {
          xhr.setRequestHeader("Content-Type", "application/json");
        }
        xhr.onreadystatechange = function () {
          if (xhr.readyState !== 4) return;
          var data = null;
          try {
            data = JSON.parse(xhr.responseText || "{}");
          } catch (err) {
            cb(err, null);
            return;
          }
          cb(null, data);
        };
        xhr.onerror = function () {
          cb(new Error("network error"), null);
        };
        try {
          xhr.send(method === "POST" ? JSON.stringify(payload || {}) : null);
        } catch (err) {
          cb(err, null);
        }
      }

      function fetchState() {
        if (requestInFlight || fetchInFlight) return;
        fetchInFlight = true;
        var startedAtMs = Date.now();
        httpJSON("GET", "/api/state", null, function (err, data) {
          fetchInFlight = false;
          if (startedAtMs < lastActionStartedAtMs) return;
          if (requestInFlight) return;
          if (err) {
            console.error("fetchState failed", err);
            return;
          }
          if (!data || !data.ok) {
            console.error("state fetch failed", data && data.error);
            return;
          }
          render(data.state);
        });
      }

      function showBusyOverlay(title, message) {
        var backdrop = byId("busy-backdrop");
        var titleEl = byId("busy-title");
        var textEl = byId("busy-text");
        if (!backdrop || !titleEl || !textEl) return;
        titleEl.textContent = title || "Saving PA Rosbag";
        textEl.textContent = message || "Finalizing the rosbag for this PA.\\nPlease wait...";
        backdrop.style.display = "flex";
      }

      function hideBusyOverlay() {
        var backdrop = byId("busy-backdrop");
        if (!backdrop) return;
        backdrop.style.display = "none";
      }

      function postAction(action, payload, opts) {
        opts = opts || {};
        if (requestInFlight) return;
        lastActionStartedAtMs = Date.now();
        requestInFlight = true;
        if (opts.busy) {
          showBusyOverlay(
            opts.busyTitle || "Saving PA Rosbag",
            opts.busyMessage || "Finalizing the rosbag for this PA.\\nPlease wait..."
          );
        }
        httpJSON("POST", "/api/action/" + encodeURIComponent(action), payload || {}, function (err, data) {
          requestInFlight = false;
          if (opts.busy) {
            hideBusyOverlay();
          }
          if (err) {
            console.error("postAction failed", action, err);
            alert("Request failed: " + action);
            return;
          }
          if (!data || !data.ok) {
            alert((data && data.error) || ("Action failed: " + action));
          }
          if (data && data.state) {
            render(data.state);
          }
        });
      }

      function render(state) {
        stateRef.value = state;
        setTextIfPresent("policy", state.policy);
        setTextIfPresent("sht", state.sht + " (eval " + state.repeat_index + "/" + state.runs_per_sht + ")");
        setTextIfPresent("pa", state.pa_prompt + " (" + state.pa_index + "/" + state.pa_total + ")");
        setTextIfPresent("elapsed", fmtSec(state.elapsed_sec));
        setTextIfPresent("status", state.status);

        var running = state.phase === "running_pa";
        var btnSuccess = byId("btn-success");
        if (btnSuccess) btnSuccess.disabled = !running;
        var btnFail = byId("btn-fail");
        if (btnFail) btnFail.disabled = !running;
        var btnEnd = byId("btn-end");
        if (btnEnd) btnEnd.disabled = state.phase === "done";

        var motionBtn = byId("btn-motion");
        if (motionBtn && state.motion_enabled) {
          motionBtn.textContent = "Robot STOP (no-send)";
          motionBtn.className = "btn-motion-stop";
        } else if (motionBtn) {
          motionBtn.textContent = "Robot RESUME";
          motionBtn.className = "btn-motion-run";
        }

        var editToggle = byId("edit-toggle");
        if (editToggle) {
          editToggle.checked = !!state.records_edit_enabled;
        }
        renderRecords(state.records, !!state.records_edit_enabled);
        renderModal(state.modal);
      }

      function renderRecords(records, editEnabled) {
        var root = byId("records");
        if (!records || records.length === 0) {
          root.innerHTML = '<div class="muted">No records yet for this model.</div>';
          return;
        }
        var html = "";
        for (var i = 0; i < records.length; i += 1) {
          var rec = records[i];
          var checkedS = rec.success ? "checked" : "";
          var checkedF = !rec.success ? "checked" : "";
          var disabled = editEnabled ? "" : "disabled";
          html += '<div class="row">';
          html += '<div class="summary">' + pad3(i + 1) + ". " + esc(rec.sht) + " / eval " + rec.repeat_index + " (" + Number(rec.elapsed_sec).toFixed(1) + "s)</div>";
          html += '<div class="prompt">prompt: ' + esc(rec.prompt_display || rec.prompt) + "</div>";
          html += '<div class="row-bottom"><div class="radio-group">';
          html += '<label><input type="radio" name="rec-' + rec.global_idx + '" value="1" ' + checkedS + " " + disabled + ' data-idx="' + rec.global_idx + '" /> Success</label>';
          html += '<label><input type="radio" name="rec-' + rec.global_idx + '" value="0" ' + checkedF + " " + disabled + ' data-idx="' + rec.global_idx + '" /> Fail</label>';
          html += '</div><div class="timestamp">Recorded: ' + esc(rec.recorded_at) + "</div></div>";
          html += "</div>";
        }
        root.innerHTML = html;

        if (editEnabled) {
          var radios = root.querySelectorAll('input[type="radio"]');
          for (var j = 0; j < radios.length; j += 1) {
            radios[j].addEventListener("change", function (ev) {
              var idx = Number(ev.target.getAttribute("data-idx"));
              var success = ev.target.value === "1";
              postAction("edit_record", { global_idx: idx, success: success });
            });
          }
        }
      }

      function renderModal(modal) {
        var backdrop = byId("modal-backdrop");
        var title = byId("modal-title");
        var text = byId("modal-text");
        var actions = byId("modal-actions");
        var review = byId("modal-review");

        function suppressCurrentModal(ms) {
          if (lastModalSig) {
            suppressedModalSig = lastModalSig;
            suppressModalUntilMs = Date.now() + Number(ms || 0);
          }
        }

        function closeModalImmediate() {
          lastModalSig = null;
          backdrop.style.display = "none";
          title.textContent = "";
          text.textContent = "";
          actions.innerHTML = "";
          review.style.display = "none";
          review.innerHTML = "";
        }

        if (!modal) {
          closeModalImmediate();
          return;
        }

        var sig = "";
        try {
          sig = JSON.stringify(modal);
        } catch (err) {
          sig = String(modal.type || "modal");
        }
        if (suppressedModalSig && Date.now() > suppressModalUntilMs) {
          suppressedModalSig = null;
          suppressModalUntilMs = 0;
        }
        if (suppressedModalSig && sig === suppressedModalSig) {
          backdrop.style.display = "none";
          return;
        }
        if (sig === lastModalSig) return;
        lastModalSig = sig;

        backdrop.style.display = "flex";
        title.textContent = modal.title || "";
        text.textContent = modal.message || "";
        actions.innerHTML = "";
        review.style.display = "none";
        review.innerHTML = "";

        function addBtn(label, cls, onClick, opts) {
          opts = opts || {};
          var b = document.createElement("button");
          b.textContent = label;
          if (cls) b.className = cls;
          if (opts.disabled) b.disabled = true;
          b.onclick = function (ev) {
            if (requestInFlight) return;
            if (opts.disableAll !== false) {
              var btns = actions.querySelectorAll("button");
              for (var i = 0; i < btns.length; i += 1) {
                btns[i].disabled = true;
              }
            }
            if (opts.closeFirst) {
              suppressCurrentModal(opts.suppressMs || 1800);
              closeModalImmediate();
            }
            onClick(ev);
          };
          actions.appendChild(b);
        }

        if (modal.type === "policy_select") {
          var policyItems = Array.isArray(modal.items) ? modal.items : [];
          if (policyItems.length === 0) {
            var fallbackItems = stateRef.value && Array.isArray(stateRef.value.policy_options)
              ? stateRef.value.policy_options
              : [];
            policyItems = fallbackItems;
          }
          if (policyItems.length === 0) {
            review.style.display = "block";
            review.textContent = "(no models)";
          } else {
            for (var p = 0; p < policyItems.length; p += 1) {
              (function (item) {
                var done = !!item.done;
                var label = done ? (String(item.name || ("Model " + item.index)) + " (Done)") : String(item.name || ("Model " + item.index));
                var cls = done ? "btn-done" : "btn-main";
                addBtn(label, cls, function () {
                  postAction("select_policy", { index: Number(item.index) });
                }, { closeFirst: true, disableAll: false });
              })(policyItems[p]);
            }
          }
        } else if (modal.type === "robot_reset") {
          addBtn(modal.button_text || "Reset done", "btn-main", function () {
            postAction("confirm_robot_reset", {});
          }, { closeFirst: true });
        } else if (modal.type === "start_policy") {
          addBtn(modal.button_text || "Start evaluating", "btn-main", function (ev) {
            if (ev && ev.target) ev.target.disabled = true;
            closeModalImmediate();
            byId("status").textContent = "Starting server/session...";
            postAction("start_policy", {});
          });
          addBtn("Choose another model", "", function () {
            suppressCurrentModal(1200);
            closeModalImmediate();
            postAction("reselect_policy", {});
          }, { disableAll: false });
        } else if (modal.type === "start_sht") {
          addBtn(modal.button_text || "Start SHT", "btn-main", function (ev) {
            if (ev && ev.target) ev.target.disabled = true;
            closeModalImmediate();
            byId("status").textContent = "Starting server/session...";
            postAction("start_sht", {});
          });
        } else if (modal.type === "policy_summary") {
          addBtn(modal.button_text || "Continue", "btn-main", function () {
            postAction("confirm_policy_summary", {});
          }, { closeFirst: true });
        } else if (modal.type === "sht_summary") {
          addBtn(modal.button_text || "Continue", "btn-main", function () {
            postAction("confirm_sht_summary", {});
          }, { closeFirst: true });
        } else if (modal.type === "pa_result") {
          addBtn(modal.button_text || "Next PA", "btn-main", function () {
            postAction("next_pa", {});
          }, { closeFirst: true });
          if (modal.show_force_sht_fail) {
            addBtn("SHT Fail (Skip Rest)", "btn-fail", function () {
              var ok = window.confirm("Mark all unrecorded PAs in this run as Fail and move to next SHT?");
              if (!ok) return;
              suppressCurrentModal(1200);
              closeModalImmediate();
              postAction("force_sht_fail", {});
            }, { disableAll: false });
          }
        } else if (modal.type === "sht_result") {
          addBtn("SHT Success", "btn-ok", function () {
            postAction("select_sht_result", { success: true });
          }, { closeFirst: true });
          addBtn("SHT Fail", "btn-fail", function () {
            postAction("select_sht_result", { success: false });
          }, { closeFirst: true });
        } else if (modal.type === "sht_review") {
          review.style.display = "block";
          var items = Array.isArray(modal.items) ? modal.items : [];
          if (items.length === 0) {
            review.textContent = "(no PA records)";
          } else {
            var html = "";
            for (var k = 0; k < items.length; k += 1) {
              var item = items[k] || {};
              var paLabel = esc(item.pa || ("PA_" + (k + 1)));
              var ok = !!item.success;
              var rowClass = ok ? "review-item review-item-ok" : "review-item review-item-fail";
              var resultClass = ok ? "review-result review-result-ok" : "review-result review-result-fail";
              var resultLabel = ok ? "Success" : "Fail";
              html += '<div class="' + rowClass + '">';
              html += '<span class="review-pa">' + paLabel + '</span>';
              html += '<span class="' + resultClass + '">' + resultLabel + "</span>";
              html += "</div>";
            }
            review.innerHTML = html;
          }
          addBtn("Confirm and Next SHT", "btn-main", function () {
            postAction("confirm_next_sht", {});
          }, { closeFirst: true });
          addBtn("Close (Edit Later)", "", function () {
            postAction("close_review", {});
          }, { closeFirst: true });
        }

        if (modal.type !== "done") {
          addBtn("End Session", "btn-end", function () {
            var ok = window.confirm("End current evaluation session?\\nThis stops roslaunch and all model server containers.");
            if (!ok) return;
            suppressCurrentModal(1200);
            closeModalImmediate();
            postAction("terminate_session", {});
          }, { disableAll: false });
        }
      }

      function setup() {
        var btnSuccess = byId("btn-success");
        if (btnSuccess) {
          btnSuccess.onclick = function () {
            postAction("pa_result", { success: true }, {
              busy: true,
              busyTitle: "Saving PA Rosbag",
              busyMessage: "Finalizing the rosbag for this PA.\\nPlease wait..."
            });
          };
        }
        var btnFail = byId("btn-fail");
        if (btnFail) {
          btnFail.onclick = function () {
            postAction("pa_result", { success: false }, {
              busy: true,
              busyTitle: "Saving PA Rosbag",
              busyMessage: "Finalizing the rosbag for this PA.\\nPlease wait..."
            });
          };
        }
        var btnStart = byId("btn-start");
        if (btnStart) {
          btnStart.onclick = function () { postAction("start_resume", {}); };
        }
        var btnMotion = byId("btn-motion");
        if (btnMotion) {
          btnMotion.onclick = function () { postAction("toggle_motion", {}); };
        }
        var btnEnd = byId("btn-end");
        if (btnEnd) {
          btnEnd.onclick = function () {
            var ok = window.confirm("End current evaluation session?\\nThis stops roslaunch and all model server containers.");
            if (!ok) return;
            postAction("terminate_session", {});
          };
        }
        var editToggle = byId("edit-toggle");
        if (editToggle) {
          editToggle.addEventListener("change", function (ev) {
            postAction("set_records_edit", { enabled: !!ev.target.checked });
          });
        }

        INITIAL_STATE = parseInitialState();
        if (INITIAL_STATE && typeof INITIAL_STATE === "object") {
          try {
            render(INITIAL_STATE);
          } catch (err) {
            console.error("initial render failed", err);
          }
        }

        window.addEventListener("pageshow", function () {
          requestInFlight = false;
          fetchInFlight = false;
          lastModalSig = null;
          suppressedModalSig = null;
          suppressModalUntilMs = 0;
          hideBusyOverlay();
          if (stateRef.value) {
            try {
              render(stateRef.value);
            } catch (err) {
              console.error("pageshow render failed", err);
            }
          }
          fetchState();
        });

        fetchState();
        setInterval(fetchState, 500);
      }

      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", setup);
      } else {
        setup();
      }
    })();
  </script>
</body>
</html>
"""


def _render_html_page(initial_state: dict[str, Any]) -> str:
    blob = json.dumps(initial_state, ensure_ascii=False)
    blob = blob.replace("</", "<\\/")
    return HTML_PAGE_TEMPLATE.replace("__INITIAL_STATE_JSON__", blob)


class EvaluatorHttpHandler(BaseHTTPRequestHandler):
    state: WebEvaluationState

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)

    def _send_html(self, status: int, text: str) -> None:
        blob = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(blob)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep stdout cleaner than BaseHTTPRequestHandler default.
        print("[HTTP] " + (format % args))

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            try:
                state = self.state.snapshot()
            except Exception as exc:
                traceback.print_exc()
                self._send_html(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"<html><body><h1>State error</h1><pre>{exc}</pre></body></html>",
                )
                return
            self._send_html(HTTPStatus.OK, _render_html_page(state))
            return
        if path == "/api/state":
            try:
                state = self.state.snapshot()
                self._send_json(HTTPStatus.OK, {"ok": True, "state": state})
            except Exception as exc:
                traceback.print_exc()
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown path: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        prefix = "/api/action/"
        if not path.startswith(prefix):
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown path: {path}"})
            return

        action = path[len(prefix) :]
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            payload = {}

        ok, error, state = self.state.handle_action(action, payload if isinstance(payload, dict) else {})
        status = HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST
        self._send_json(status, {"ok": ok, "error": error, "state": state})


class EvaluatorHttpServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler: type[EvaluatorHttpHandler], state: WebEvaluationState):
        self.state = state
        handler.state = state
        super().__init__(server_address, handler)


def _build_state_from_args(args: argparse.Namespace) -> tuple[WebEvaluationState, Path]:
    if args.models and args.participants and args.models != args.participants:
        raise SystemExit("Use either --models or --participants (deprecated), not both with different values.")
    models_path_str = args.models or args.participants
    if not models_path_str:
        raise SystemExit("--models is required.")
    if args.participants and not args.models:
        print("[WARN] --participants is deprecated. Use --models.")

    repo_root = Path(__file__).resolve().parents[1]
    result_root = Path(args.result_root).expanduser().resolve() if args.result_root else _default_result_root(repo_root)

    models = _load_models(Path(models_path_str).expanduser().resolve(), repo_root)
    tasks = _load_tasks(Path(args.tasks).expanduser().resolve())
    test_mode = _parse_bool(args.test_mode)

    client_env_overrides: dict[str, str] = {}
    ros_master_uri = args.ros_master_uri.strip()
    hsr_ip = args.hsr_ip.strip()
    ros_ip = args.ros_ip.strip()
    ros_ip_choice = int(args.ros_ip_choice)
    if ros_ip and ros_ip_choice > 0:
        raise SystemExit("Use either --ros-ip or --ros-ip-choice, not both.")
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
            raise SystemExit(
                "Failed to auto-select ROS_IP: no non-loopback host IPv4 candidate found. Set --ros-ip explicitly."
            )
        if ros_ip_choice > len(candidates):
            raise SystemExit(
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

    print("[INFO] Client is up. Policy servers will be started on demand after model selection.")

    ros = RosLaunchController(
        repo_root=repo_root,
        result_root=result_root,
        client_container_name=args.client_container_name,
        test_mode=test_mode,
    )

    state = WebEvaluationState(
        participants=models,
        tasks=tasks,
        runs_per_sht=args.runs_per_sht,
        result_root=result_root,
        docker_manager=docker_manager,
        ros_controller=ros,
    )
    return state, result_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Competition evaluator Web UI with multi-model server orchestration.")
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
    parser.add_argument("--host", default="0.0.0.0", help="Web server bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Web server bind port")
    args = parser.parse_args()

    state, result_root = _build_state_from_args(args)

    server = EvaluatorHttpServer((args.host, args.port), EvaluatorHttpHandler, state)
    print(f"[INFO] Web evaluator started: http://{args.host}:{args.port}")
    print("[INFO] Access from same network via: http://<this-machine-ip>:%d" % args.port)
    print(f"[INFO] Result root: {result_root}")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Shutting down...")
    finally:
        server.shutdown()
        server.server_close()
        state.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
