"""アクション後処理: グリッパー制御ガード + テンポラルアンサンブル

任意の BasePolicy をラップし、出力アクションに後処理を適用する。
Hierarchical VLA / E2E 両モードで使用可能。

グリッパー保護機構:
  1. 力覚ガード: wrist_wrench の力ノルムが上限超過時、閉じ方向をブロック
  2. 接触検知ガード: 閉じ指令を出しても位置が動かない場合、接触と判定して閉じ方向をブロック
     （追加データ購読不要。observation.state のみで動作）
"""

import logging
from collections import deque
from typing import Dict

import numpy as np
from policy_client.base_policy import BasePolicy

logger = logging.getLogger(__name__)


class ActionPostprocessor(BasePolicy):
    """アクション後処理ラッパー

    Args:
        policy: 内部ポリシー（BasePolicy 互換）
        gripper_binarize: グリッパー二値化を有効にする
        gripper_threshold: 二値化閾値（これ以上で開=1.0、未満で閉=0.0）
        gripper_idx: アクション配列中のグリッパーインデックス
        gripper_force_guard: 力覚センサー閾値ガードを有効にする
        gripper_force_limit: 把持力上限 [N]（超過時にグリッパー指令を現在値に固定）
        gripper_contact_guard: 接触検知ガードを有効にする
        gripper_contact_tolerance: 位置変化の閾値（これ未満で「動いていない」と判定）
        gripper_contact_patience: 連続何ステップ動かなければ接触と判定するか
        temporal_ensemble: テンポラルアンサンブルを有効にする
        ensemble_window: アンサンブル窓幅（過去 N ステップ）
        ensemble_decay: 指数減衰率（0〜1、1 に近いほど直近を重視）
    """

    def __init__(
        self,
        policy: BasePolicy,
        *,
        gripper_binarize: bool = True,
        gripper_threshold: float = 0.5,
        gripper_idx: int = 5,
        gripper_force_guard: bool = False,
        gripper_force_limit: float = 10.0,
        gripper_contact_guard: bool = False,
        gripper_contact_tolerance: float = 0.005,
        gripper_contact_patience: int = 3,
        gripper_clip: bool = False,
        gripper_clip_min: float = 0.0,
        gripper_clip_max: float = 1.0,
        action_smoothing: bool = False,
        action_smoothing_alpha: float = 0.3,
        action_smoothing_exclude_dims: list[int] | None = None,
        temporal_ensemble: bool = False,
        ensemble_window: int = 5,
        ensemble_decay: float = 0.8,
    ):
        self._policy = policy
        self._gripper_binarize = gripper_binarize
        self._gripper_threshold = gripper_threshold
        self._gripper_idx = gripper_idx
        self._gripper_clip = gripper_clip
        self._gripper_clip_min = gripper_clip_min
        self._gripper_clip_max = gripper_clip_max
        self._action_smoothing = action_smoothing
        self._action_smoothing_alpha = action_smoothing_alpha
        self._action_smoothing_exclude = action_smoothing_exclude_dims or [5]  # default: gripper 除外
        self._prev_action: np.ndarray | None = None
        self._gripper_force_guard = gripper_force_guard
        self._gripper_force_limit = gripper_force_limit
        self._gripper_contact_guard = gripper_contact_guard
        self._gripper_contact_tolerance = gripper_contact_tolerance
        self._gripper_contact_patience = gripper_contact_patience
        self._last_gripper_value: float | None = None
        self._gripper_stall_count: int = 0
        self._last_gripper_state: float | None = None
        self._contact_detected: bool = False
        self._temporal_ensemble = temporal_ensemble
        self._ensemble_window = ensemble_window
        self._ensemble_decay = ensemble_decay

        # アンサンブルバッファ（各要素は 1D アクション配列）
        self._action_buffer: deque[np.ndarray] = deque(maxlen=ensemble_window)

        # 重み（指数減衰: 最新=1, 1つ前=decay, 2つ前=decay^2, ...）
        self._weights = np.array(
            [ensemble_decay ** i for i in range(ensemble_window)],
            dtype=np.float64,
        )

    @property
    def metadata(self) -> dict:
        inner_meta = getattr(self._policy, "metadata", {})
        return {
            **inner_meta,
            "postprocessor": {
                "gripper_binarize": self._gripper_binarize,
                "gripper_threshold": self._gripper_threshold,
                "gripper_force_guard": self._gripper_force_guard,
                "gripper_force_limit": self._gripper_force_limit,
                "gripper_contact_guard": self._gripper_contact_guard,
                "gripper_contact_tolerance": self._gripper_contact_tolerance,
                "gripper_contact_patience": self._gripper_contact_patience,
                "gripper_clip": self._gripper_clip,
                "gripper_clip_min": self._gripper_clip_min,
                "gripper_clip_max": self._gripper_clip_max,
                "action_smoothing": self._action_smoothing,
                "action_smoothing_alpha": self._action_smoothing_alpha,
                "temporal_ensemble": self._temporal_ensemble,
                "ensemble_window": self._ensemble_window,
                "ensemble_decay": self._ensemble_decay,
            },
        }

    def infer(self, obs: Dict) -> Dict:
        result = self._policy.infer(obs)
        actions = result.get("actions")
        if actions is None:
            return result

        actions = np.asarray(actions, dtype=np.float64)

        # テンポラルアンサンブル（アクション horizon の先頭ステップに適用）
        if self._temporal_ensemble and actions.ndim >= 1:
            # horizon がある場合は先頭行、1D ならそのまま
            first_action = actions[0] if actions.ndim == 2 else actions
            self._action_buffer.appendleft(first_action.copy())
            first_action = self._compute_ensemble()
            if actions.ndim == 2:
                actions[0] = first_action
            else:
                actions = first_action

        # グリッパークリッピング（二値化の前に適用）
        if self._gripper_clip:
            actions = self._clip_gripper(actions)

        # Action Smoothing (EMA, gripper 除外)
        if self._action_smoothing:
            actions = self._smooth_action(actions)

        # グリッパー二値化
        if self._gripper_binarize:
            actions = self._binarize_gripper(actions)

        # 力覚センサー閾値ガード
        if self._gripper_force_guard:
            actions = self._apply_force_guard(actions, obs)

        # 接触検知ガード（observation.state ベース）
        if self._gripper_contact_guard:
            actions = self._apply_contact_guard(actions, obs)

        result["actions"] = actions
        return result

    def reset(self) -> None:
        """ポリシーとアンサンブルバッファをリセット"""
        self._action_buffer.clear()
        self._last_gripper_value = None
        self._gripper_stall_count = 0
        self._last_gripper_state = None
        self._contact_detected = False
        self._prev_action = None
        self._policy.reset()

    def reset_ensemble_buffer(self) -> None:
        """PA 切替時など、アンサンブルバッファのみリセット"""
        self._action_buffer.clear()

    def _compute_ensemble(self) -> np.ndarray:
        """指数加重平均を計算"""
        n = len(self._action_buffer)
        if n == 0:
            raise ValueError("アンサンブルバッファが空です")
        if n == 1:
            return self._action_buffer[0].copy()

        w = self._weights[:n]
        w = w / w.sum()
        stacked = np.stack(list(self._action_buffer), axis=0)
        return np.einsum("i,i...->...", w, stacked)

    def _smooth_action(self, actions: np.ndarray) -> np.ndarray:
        """OPT-5: EMA ベースのアクションスムージング。

        隣接アクション間の急激な変化を抑制し、滑らかな動作を実現する。
        Team 18 分析: 滑らかな出力が動作安定性の鍵。
        gripper 等の指定次元は除外（クリスプな遷移が必要なため）。

        alpha=0.3 で jerk -40%, MSE -6.8% の改善を確認 (run24 オフライン検証)。
        """
        first_action = actions[0] if actions.ndim == 2 else actions
        if self._prev_action is not None and len(self._prev_action) == len(first_action):
            alpha = self._action_smoothing_alpha
            smoothed = first_action.copy()
            mask = np.ones(len(first_action), dtype=bool)
            for dim in self._action_smoothing_exclude:
                if dim < len(mask):
                    mask[dim] = False
            smoothed[mask] = alpha * first_action[mask] + (1 - alpha) * self._prev_action[mask]
            if actions.ndim == 2:
                actions[0] = smoothed
            else:
                actions = smoothed
        self._prev_action = (actions[0] if actions.ndim == 2 else actions).copy()
        return actions

    def _clip_gripper(self, actions: np.ndarray) -> np.ndarray:
        """グリッパー次元を [clip_min, clip_max] にクリッピング"""
        idx = self._gripper_idx
        if actions.ndim == 2:
            actions[:, idx] = np.clip(
                actions[:, idx], self._gripper_clip_min, self._gripper_clip_max
            )
        elif actions.ndim == 1:
            actions[idx] = np.clip(
                actions[idx], self._gripper_clip_min, self._gripper_clip_max
            )
        return actions

    def _binarize_gripper(self, actions: np.ndarray) -> np.ndarray:
        """グリッパー次元を閾値で二値化"""
        idx = self._gripper_idx
        if actions.ndim == 2:
            actions[:, idx] = np.where(
                actions[:, idx] >= self._gripper_threshold, 1.0, 0.0
            )
        elif actions.ndim == 1:
            actions[idx] = 1.0 if actions[idx] >= self._gripper_threshold else 0.0
        return actions

    def _apply_force_guard(self, actions: np.ndarray, obs: Dict) -> np.ndarray:
        """力覚センサー閾値ガード

        観測に wrist wrench データが含まれる場合、把持力（fx, fy, fz のノルム）が
        gripper_force_limit を超えていたら、グリッパー指令をこれ以上閉じないよう
        直前の値に固定する。

        対応する観測キー: "wrist_wrench" (6D: [fx, fy, fz, tx, ty, tz])
        キーが存在しない場合は何もしない（フォールバック安全）。
        """
        idx = self._gripper_idx

        # 現在のグリッパー指令値を取得
        if actions.ndim == 2:
            current_gripper = float(actions[0, idx])
        else:
            current_gripper = float(actions[idx])

        # wrench データの取得（なければスキップ）
        wrench = obs.get("wrist_wrench")
        if wrench is not None:
            wrench = np.asarray(wrench, dtype=np.float64)
            force_norm = np.linalg.norm(wrench[:3])  # fx, fy, fz のノルム

            if force_norm > self._gripper_force_limit:
                # 力の上限超過: グリッパーをこれ以上閉じない
                # （現在値より閉じる方向の指令のみブロック、開く方向は許可）
                if self._last_gripper_value is not None and current_gripper > self._last_gripper_value:
                    logger.warning(
                        "力覚ガード発動: force=%.2fN > limit=%.2fN, "
                        "gripper %.3f → %.3f に制限",
                        force_norm, self._gripper_force_limit,
                        current_gripper, self._last_gripper_value,
                    )
                    if actions.ndim == 2:
                        actions[:, idx] = self._last_gripper_value
                    else:
                        actions[idx] = self._last_gripper_value
                    current_gripper = self._last_gripper_value

        self._last_gripper_value = current_gripper
        return actions

    def _apply_contact_guard(self, actions: np.ndarray, obs: Dict) -> np.ndarray:
        """接触検知ガード

        閉じ指令を出しているのにグリッパーの実位置が動かない場合、
        物体に接触していると判定し、これ以上閉じないようブロックする。

        判定ロジック:
          1. 閉じ方向の指令（action[idx] > state[idx]）が出ている
          2. 実位置の変化が tolerance 未満
          3. 上記が patience ステップ連続 → 接触と判定
          4. 接触判定後は閉じ方向をブロック、開く方向は常に許可

        対応する観測キー: "state" (8D+, index=gripper_idx にグリッパー位置)
        キーが存在しない場合は何もしない（フォールバック安全）。
        """
        idx = self._gripper_idx

        # 現在のグリッパー指令値
        if actions.ndim == 2:
            cmd_gripper = float(actions[0, idx])
        else:
            cmd_gripper = float(actions[idx])

        # 観測からグリッパーの実位置を取得
        state = obs.get("state")
        if state is None:
            return actions

        state = np.asarray(state, dtype=np.float64)
        if state.ndim > 1:
            state = state[0]
        if len(state) <= idx:
            return actions

        current_state = float(state[idx])

        # 接触判定のリセット: 開く方向の指令が出たら接触解除
        if self._last_gripper_value is not None and cmd_gripper < self._last_gripper_value:
            if self._contact_detected:
                logger.info("接触検知ガード解除: 開き方向の指令を検出")
            self._contact_detected = False
            self._gripper_stall_count = 0

        # 接触判定中: 閉じ方向をブロック
        if self._contact_detected:
            if self._last_gripper_value is not None and cmd_gripper > self._last_gripper_value:
                logger.debug(
                    "接触検知ガード: gripper %.3f → %.3f に制限",
                    cmd_gripper, self._last_gripper_value,
                )
                if actions.ndim == 2:
                    actions[:, idx] = self._last_gripper_value
                else:
                    actions[idx] = self._last_gripper_value
            self._last_gripper_state = current_state
            self._last_gripper_value = float(actions[0, idx]) if actions.ndim == 2 else float(actions[idx])
            return actions

        # ストール検知: 閉じ指令中に位置が動かない
        if self._last_gripper_state is not None and self._last_gripper_value is not None:
            is_closing = cmd_gripper > current_state  # 閉じ方向の指令
            position_change = abs(current_state - self._last_gripper_state)
            is_stalled = position_change < self._gripper_contact_tolerance

            if is_closing and is_stalled:
                self._gripper_stall_count += 1
                if self._gripper_stall_count >= self._gripper_contact_patience:
                    self._contact_detected = True
                    logger.warning(
                        "接触検知ガード発動: %d ステップ連続ストール "
                        "(position_change=%.4f < tolerance=%.4f), "
                        "gripper を %.3f に固定",
                        self._gripper_stall_count,
                        position_change,
                        self._gripper_contact_tolerance,
                        current_state,
                    )
                    # 現在の実位置で固定
                    if actions.ndim == 2:
                        actions[:, idx] = current_state
                    else:
                        actions[idx] = current_state
            else:
                self._gripper_stall_count = 0

        self._last_gripper_state = current_state
        self._last_gripper_value = float(actions[0, idx]) if actions.ndim == 2 else float(actions[idx])
        return actions
