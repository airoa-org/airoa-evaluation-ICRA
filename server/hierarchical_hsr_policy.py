"""Hierarchical VLA ラッパー for デプロイ Docker (BasePolicy 互換)

LeRobotHSRPolicy を PA 単位で駆動する HVLA ポリシー。
WebsocketPolicyServer の policy.infer(obs) インターフェースに準拠。

obs keys (クライアントから):
    hand_rgb: [H,W,3] uint8
    head_rgb: [H,W,3] uint8
    state: [8] float
    prompt: str

returns:
    {"actions": np.ndarray [action_horizon, 11]}
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PAMonitor:
    """PA 完了判定（ルールベース）。

    - min_steps 保証
    - 収束判定（関節状態のスライディングウィンドウ）
    - グリッパー遷移量判定（grasp/place）
    - max_steps 適応制御（途中 PA は短縮、最終 PA は通常値）
    - IM-2: 成功確認ウィンドウ（多数決）
    """

    def __init__(self, config: dict):
        cfg = config.get("pa_monitor", {})
        self._max_steps = cfg.get("max_steps", {
            "navigate": 200, "grasp": 250, "place": 200,
            "open": 80, "close": 80, "default": 150,
        })
        self._max_steps_short = cfg.get("max_steps_short", None)
        self._min_steps = cfg.get("min_steps", {"default": 30})

        # 収束判定
        self._convergence_threshold = cfg.get("joint_convergence_threshold", 0.06)
        self._convergence_window = cfg.get("convergence_window", 10)

        # グリッパー遷移
        self._gripper_grasp_threshold = cfg.get("gripper_grasp_transition_threshold", -0.9)
        self._gripper_place_threshold = cfg.get("gripper_place_transition_threshold", 0.9)
        self._gripper_window = cfg.get("gripper_window", 5)

        # IM-2: 成功確認ウィンドウ
        self._confirmation_window = cfg.get("confirmation_window", 1)
        self._confirmation_threshold = cfg.get("confirmation_threshold", 1)

    def is_done(self, pa: str, step: int, state_history: list,
                gripper_history: list, is_last: bool) -> bool:
        pa_type = get_pa_type(pa)
        min_s = self._min_steps.get(pa_type, self._min_steps.get("default", 30))

        if step < min_s:
            return False

        # max_steps
        if not is_last and self._max_steps_short:
            max_s = self._max_steps_short.get(pa_type, self._max_steps_short.get("default", 75))
        else:
            max_s = self._max_steps.get(pa_type, self._max_steps.get("default", 150))

        if step >= max_s:
            return True

        # 収束判定
        if len(state_history) >= self._convergence_window:
            window = np.array(state_history[-self._convergence_window:])
            std = np.std(window, axis=0).mean()
            if std < self._convergence_threshold:
                return True

        # グリッパー遷移量判定
        if len(gripper_history) >= self._gripper_window:
            g_window = gripper_history[-self._gripper_window:]
            transition = g_window[-1] - g_window[0]
            if pa_type == "grasp" and transition <= self._gripper_grasp_threshold:
                return True
            if pa_type == "place" and transition >= self._gripper_place_threshold:
                return True

        return False


class FailureMonitorWrapper:
    """State ベース FM (FIX-6b)。"""

    def __init__(self, model, scaler=None, threshold: float = 0.6,
                 min_steps: int = 10):
        self._model = model
        self._scaler = scaler
        self.threshold = threshold
        self.min_steps = min_steps

    def compute_risk(self, state_history: list,
                     gripper_history: list) -> Optional[float]:
        if len(state_history) < 5:
            return None

        states = np.array(state_history)
        n_steps = len(states)
        gripper = np.array(gripper_history) if gripper_history else np.zeros(n_steps)

        velocity = np.diff(states, axis=0)
        velocity_norms = np.linalg.norm(velocity, axis=1)

        if n_steps >= 4:
            jerk = np.diff(states, n=3, axis=0)
            state_jerk = float(np.mean(np.linalg.norm(jerk, axis=1)))
        else:
            state_jerk = 0.0

        displacement = float(np.linalg.norm(states[-1] - states[0]))
        stillness_ratio = float(np.mean(velocity_norms < 0.001)) if len(velocity_norms) > 0 else 1.0

        features = np.array([
            np.log1p(n_steps),
            float(velocity_norms.mean()),
            float(velocity_norms.std()),
            displacement,
            state_jerk,
            float(gripper.std()) if len(gripper) > 1 else 0.0,
            float(gripper.max() - gripper.min()) if len(gripper) > 1 else 0.0,
            stillness_ratio,
        ], dtype=np.float32).reshape(1, -1)

        if self._scaler is not None:
            features = self._scaler.transform(features)
        return float(self._model.predict_proba(features)[0, 1])


class RetryController:
    """PA 失敗時の retry/skip/abort 制御。"""

    MAX_RETRIES = {"grasp": 3, "navigate": 2, "place": 2, "default": 2}

    def __init__(self, max_total: int = 10):
        self._max_total = max_total
        self._total = 0
        self._pa_counts: Dict[str, int] = {}

    def decide(self, pa: str, pa_type: str) -> str:
        """returns: "retry", "skip", or "abort" """
        if self._total >= self._max_total:
            return "abort"
        max_r = self.MAX_RETRIES.get(pa_type, self.MAX_RETRIES["default"])
        count = self._pa_counts.get(pa, 0)
        if count < max_r:
            self._pa_counts[pa] = count + 1
            self._total += 1
            return "retry"
        return "skip"

    def reset(self):
        self._total = 0
        self._pa_counts.clear()


def get_pa_type(pa: str) -> str:
    pa_lower = pa.lower()
    for kw in ["navigate", "go to", "go back", "move to", "return"]:
        if kw in pa_lower:
            return "navigate"
    for kw in ["grasp", "grab", "pick up", "pick an", "take"]:
        if kw in pa_lower:
            return "grasp"
    for kw in ["place", "store", "put", "drop", "hang"]:
        if kw in pa_lower:
            return "place"
    for kw in ["open"]:
        if kw in pa_lower:
            return "open"
    for kw in ["close"]:
        if kw in pa_lower:
            return "close"
    return "default"


def _fuzzy_match(sht: str, pa_map: dict) -> Optional[List[str]]:
    """正規化マッチング（大小文字・空白・末尾ピリオド無視）。"""
    sht_norm = sht.lower().strip().rstrip(".")
    for key, pa_list in pa_map.items():
        if key.lower().strip().rstrip(".") == sht_norm:
            return pa_list
    return None


class HierarchicalHSRPolicy:
    """BasePolicy 互換の HVLA ラッパー。

    デプロイ Docker の WebsocketPolicyServer から呼ばれる。
    infer(obs) で PA 分解 → PA プロンプト差替 → LeRobotHSRPolicy.infer() を呼ぶ。
    """

    def __init__(self, base_policy, pa_map: dict, config: dict,
                 fm_model=None, fm_scaler=None, llm_api_client=None):
        self._base = base_policy
        self._pa_map = pa_map
        self._config = config
        self._llm_client = llm_api_client  # LLM API クライアント（別プロセス）

        self._monitor = PAMonitor(config)

        # FM
        fm_cfg = config.get("failure_monitor", {})
        self._fm = None
        if fm_model is not None:
            self._fm = FailureMonitorWrapper(
                fm_model, fm_scaler,
                threshold=fm_cfg.get("failure_risk_threshold", 0.6),
                min_steps=fm_cfg.get("min_actions_for_risk", 10),
            )
        # FIX-7: FM チャンク境界実行
        self._fm_check_interval = fm_cfg.get("fm_check_interval", 10)

        # RetryController
        retry_cfg = config.get("retry_controller", {})
        self._retry = RetryController(
            max_total=retry_cfg.get("max_total_retries", 10),
        ) if retry_cfg.get("enabled", True) else None

        # HVLA 状態
        self._current_sht = None
        self._pa_queue: List[str] = []
        self._current_pa: Optional[str] = None
        self._step_count = 0
        self._state_history: List[np.ndarray] = []
        self._gripper_history: List[float] = []

    def infer(self, obs: dict) -> dict:
        prompt = obs.get("prompt", "")
        state = np.asarray(obs.get("state", np.zeros(8)), dtype=np.float32)

        # SHT 開始 / 切替
        if prompt and prompt != self._current_sht:
            self._start_sht(prompt)

        # PA 完了判定 + 遷移
        self._step_count += 1
        self._state_history.append(state.copy())
        if len(state) > 5:
            self._gripper_history.append(float(state[5]))

        if self._current_pa:
            self._check_transition(state)

        # 有効プロンプトで推論
        effective_prompt = self._current_pa or prompt
        obs_with_pa = dict(obs)
        obs_with_pa["prompt"] = effective_prompt

        # MoE Expert selection based on current PA prompt
        if hasattr(self._base, "select_expert"):
            self._base.select_expert(effective_prompt)

        if self._step_count <= 3 or self._step_count % 20 == 0:
            logger.info(
                "HVLA step=%d PA='%s' queue=%d prompt='%s'",
                self._step_count, (self._current_pa or "?")[:40],
                len(self._pa_queue), effective_prompt[:40],
            )

        return self._base.infer(obs_with_pa)

    def reset(self) -> None:
        self._base.reset()
        self._current_sht = None
        self._pa_queue = []
        self._current_pa = None
        self._step_count = 0
        self._state_history = []
        self._gripper_history = []
        if self._retry:
            self._retry.reset()

    @property
    def metadata(self) -> dict:
        base_meta = self._base.metadata if hasattr(self._base, "metadata") else {}
        return {**base_meta, "mode": "hierarchical"}

    def _start_sht(self, sht: str):
        self._current_sht = sht
        self._base.reset()
        pa_list = self._decompose(sht)
        self._pa_queue = list(pa_list)
        self._advance_pa()
        if self._retry:
            self._retry.reset()
        logger.info("HVLA SHT: '%s' -> %d PA: %s",
                     sht[:50], len(self._pa_queue) + 1,
                     [self._current_pa] + self._pa_queue)

    def _decompose(self, sht: str) -> List[str]:
        # 完全一致
        if sht in self._pa_map:
            return self._pa_map[sht]
        # ファジーマッチ
        result = _fuzzy_match(sht, self._pa_map)
        if result:
            return result
        # LLM API フォールバック（別プロセス）
        if self._llm_client is not None:
            try:
                pa_list = self._llm_client.decompose(sht)
                if pa_list and pa_list != [sht]:
                    logger.info("HVLA LLM API: '%s' -> %d PA", sht[:50], len(pa_list))
                    return pa_list
            except Exception as e:
                logger.warning("LLM API error: %s", e)
        # E2E フォールバック
        logger.warning("HVLA: '%s' PA 分解なし -> E2E", sht[:50])
        return [sht]

    def _check_transition(self, state: np.ndarray):
        pa_type = get_pa_type(self._current_pa)
        is_last = len(self._pa_queue) == 0

        # FM 判定（FIX-7: チャンク境界でのみ）
        should_check_fm = (self._fm_check_interval <= 0) or (self._step_count % self._fm_check_interval == 0)
        if (self._fm is not None and should_check_fm
                and self._step_count >= self._fm.min_steps
                and self._step_count >= self._monitor._min_steps.get(
                    pa_type, self._monitor._min_steps.get("default", 30))
                and not is_last):
            risk = self._fm.compute_risk(self._state_history, self._gripper_history)
            if risk is not None and risk >= self._fm.threshold:
                if self._retry:
                    decision = self._retry.decide(self._current_pa, pa_type)
                    if decision == "retry":
                        logger.info("HVLA RETRY (FM risk=%.3f): '%s'",
                                     risk, self._current_pa[:40])
                        self._step_count = 0
                        self._state_history = []
                        self._gripper_history = []
                        return
                    elif decision == "abort":
                        logger.warning("HVLA ABORT: '%s'", self._current_pa[:40])
                        return

                prev = self._current_pa
                self._advance_pa()
                logger.info("HVLA PA transition (FM risk=%.3f): '%s' -> '%s'",
                             risk, prev[:40],
                             (self._current_pa or "END")[:40])
                return

        # PA Monitor 判定
        if self._monitor.is_done(
            self._current_pa, self._step_count,
            self._state_history, self._gripper_history, is_last,
        ) and not is_last:
            prev = self._current_pa
            self._advance_pa()
            logger.info("HVLA PA transition (monitor): '%s' -> '%s'",
                         prev[:40], (self._current_pa or "END")[:40])

    def _advance_pa(self):
        if self._pa_queue:
            self._current_pa = self._pa_queue.pop(0)
            self._step_count = 0
            self._state_history = []
            self._gripper_history = []
