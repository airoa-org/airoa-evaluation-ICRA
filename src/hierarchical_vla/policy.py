"""Hierarchical VLA: SHT→PA 分解 + PA 単位 VLA 推論 + PA 完了判定

任意の BasePolicy 互換 VLA をラップし、以下の機能を追加する:
  1. PA Planner   — SHT プロンプトを順序付き PA 列に分解
  2. VLA Executor — 現在の PA プロンプトに差し替えて内部 VLA に委譲
  3. PA Monitor   — 複合ルールベースで PA 完了を判定
"""

import logging
from typing import Dict

import numpy as np
from policy_client.base_policy import BasePolicy

logger = logging.getLogger(__name__)

# PA 種別ごとのデフォルト最低ステップ数（完了判定を開始するまでの待機）
# M1-2: DPO 成功エピソード p10 の半分に基づくチューニング
# 旧値 (3-5) では収束判定の早期誤判定が 84-98% 発生
DEFAULT_MIN_STEPS = {
    "navigate": 28,
    "go": 28,
    "move": 28,
    "return": 25,
    "grasp": 54,
    "grab": 54,
    "pick": 54,
    "place": 45,
    "store": 45,
    "put": 45,
    "drop": 45,
    "hang": 45,
    "open": 20,
    "close": 20,
    "default": 30,
}

# PA 種別ごとのデフォルト最大ステップ数（安全弁）
DEFAULT_MAX_STEPS = {
    "navigate": 200,   # 移動（成功平均 111-135, 〜20 秒 @ 10Hz）
    "grasp": 250,      # 把持（成功平均 204, 〜25 秒）— M1-1 DPO チューニング
    "place": 200,      # 配置（成功平均 176, 〜20 秒）— M1-1 DPO チューニング
    "open": 80,        # 開動作（〜8 秒）
    "close": 80,       # 閉動作（〜8 秒）
    "default": 150,    # 未分類（〜15 秒）
}

# 状態履歴のウィンドウサイズ（スライディングウィンドウ収束判定用）
DEFAULT_CONVERGENCE_WINDOW = 10

# グリッパー遷移検知のウィンドウサイズ
DEFAULT_GRIPPER_WINDOW = 5


class HierarchicalVLAPolicy(BasePolicy):
    """Hierarchical VLA: SHT→PA 分解 + PA 単位推論のポリシー

    BasePolicy インターフェースを実装し、WebSocket ポリシーサーバーに
    そのまま組み込める。

    Args:
        vla_policy: 内部 VLA ポリシー（BasePolicy 互換）
        pa_decomposition: SHT プロンプト → PA プロンプト列の辞書
        config: オプション設定辞書（hierarchical_config.yaml の pa_monitor セクション）
        failure_monitor: predict_risk(features) → float を持つオブジェクト（FM 統合用）
        retry_controller: decide(pa, pa_type, risk) → Decision を持つオブジェクト
        pa_planner: decompose(sht) → list[str] を持つオブジェクト（LLM PA Planner）
            未知 SHT をルール→Fuzzy→LLM→E2E のフォールバックチェーンで分解する。
            hierarchical_config.yaml の pa_planner.enabled=true で有効化。
    """

    def __init__(
        self,
        vla_policy: BasePolicy,
        pa_decomposition: dict[str, list[str]],
        config: dict | None = None,
        failure_monitor=None,
        retry_controller=None,
        pa_planner=None,
        on_pa_change=None,
    ):
        self._vla = vla_policy
        self._pa_map = pa_decomposition
        self._on_pa_change = on_pa_change  # MoE Expert 切替コールバック

        # 設定
        cfg = config or {}
        self._min_steps = cfg.get("min_steps", DEFAULT_MIN_STEPS)
        self._max_steps = cfg.get("max_steps", DEFAULT_MAX_STEPS)
        # 後方互換: max_steps_per_pa が指定された場合は全 PA 種別のフォールバックとして使用
        self._max_steps_fallback = cfg.get("max_steps_per_pa", None)
        # P7-4: 途中 PA 用の短縮 max_steps（最後の PA は通常値を使用）
        self._max_steps_short = cfg.get("max_steps_short", None)
        self._gripper_close_thresh = cfg.get("gripper_close_threshold", 0.3)
        self._gripper_open_thresh = cfg.get("gripper_open_threshold", 0.7)
        # IM-3: グリッパー遷移量ベース判定（DPO 分析: 成功 -1.72 / 失敗 -0.16）
        self._gripper_grasp_transition_thresh = cfg.get("gripper_grasp_transition_threshold", -0.9)
        self._gripper_place_transition_thresh = cfg.get("gripper_place_transition_threshold", 0.9)
        self._joint_convergence_thresh = cfg.get("joint_convergence_threshold", 0.01)
        self._convergence_window = cfg.get("convergence_window", DEFAULT_CONVERGENCE_WINDOW)
        self._gripper_window = cfg.get("gripper_window", DEFAULT_GRIPPER_WINDOW)
        # IM-2: 成功確認ウィンドウ（多数決で誤検出防止）
        self._confirmation_window = cfg.get("confirmation_window", 1)  # デフォルト 1 = 無効化
        self._confirmation_threshold = cfg.get("confirmation_threshold", 1)
        # VS-1: navigate 完了後にヘッドスキャン PA を自動挿入
        self._insert_scan_after_navigate = cfg.get("insert_scan_after_navigate", False)
        self._scan_pa_prompt = cfg.get("scan_pa_prompt", "Look around to find the target object")
        # TT-VF3: Action Entropy（チャンク内分散）による PA 完了判定の補強
        self._action_entropy_threshold = cfg.get("action_entropy_threshold", 5e-4)
        self._action_entropy_window = cfg.get("action_entropy_window", 10)
        # P9-B: Navigate 専用収束判定（action の base 次元 8,9,10 の magnitude ベース）
        self._base_action_indices = cfg.get("base_action_indices", [8, 9, 10])
        self._base_convergence_threshold = cfg.get("base_convergence_threshold", 0.005)
        # P9-C: Navigate 中のグリッパー開動作検出（物体前到達の指標）
        self._navigate_gripper_ready_threshold = cfg.get("navigate_gripper_ready_threshold", 0.5)
        self._navigate_gripper_ready_window = cfg.get("navigate_gripper_ready_window", 3)

        # FM 統合 (M2-1): failure_monitor は predict_risk(action_stats) → float を持つオブジェクト
        self._failure_monitor = failure_monitor
        self._failure_risk_threshold = cfg.get("failure_risk_threshold", 0.6)
        # FIX-7: FM 実行間隔（チャンク境界でのみ判定）
        # eh=10 の場合、10 ステップごとに FM を実行。0 = 毎ステップ（従来動作）
        self._fm_check_interval = cfg.get("fm_check_interval", 10)
        # RetryController (M2-2)
        self._retry_controller = retry_controller
        # IM-1: PA Planner 参照（replan 用）
        self._pa_planner = pa_planner
        self._completed_pas: list[str] = []  # 完了済み PA 列（replan に渡す）
        self._action_history: list[np.ndarray] = []  # 予測アクション履歴（failure_risk 計算用）
        self._pa_start_gripper: float | None = None  # IM-3: PA 開始時の gripper 値

        # ステートマシン
        self._current_sht: str | None = None
        self._pa_queue: list[str] = []
        self._current_pa: str | None = None
        self._step_count: int = 0
        self._prev_state: np.ndarray | None = None
        self._total_steps: int = 0
        self._state_history: list[np.ndarray] = []  # スライディングウィンドウ用
        self._gripper_history: list[float] = []  # グリッパー遷移検知用
        self._done_votes: list[bool] = []  # IM-2: 確認ウィンドウ用

        # 外部の ActionPostprocessor への参照（PA 切替時にバッファリセット用）
        self._postprocessor = None

    def set_postprocessor(self, postprocessor) -> None:
        """ActionPostprocessor を登録し、PA 切替時にバッファリセットを連動させる"""
        self._postprocessor = postprocessor

    @property
    def metadata(self) -> dict:
        inner_meta = getattr(self._vla, "metadata", {})
        return {
            **inner_meta,
            "mode": "hierarchical",
            "total_shts": len(self._pa_map),
        }

    def infer(self, obs: Dict) -> Dict:
        sht_prompt = obs.get("prompt", "")

        # SHT が変わったら PA 列を再構築
        if sht_prompt != self._current_sht:
            self._current_sht = sht_prompt
            pa_list = list(self._decompose(sht_prompt))
            # VS-1: navigate PA の後にスキャン PA を自動挿入
            if self._insert_scan_after_navigate:
                pa_list = self._insert_scan_pas(pa_list)
            self._pa_queue = pa_list
            self._advance_pa()
            self._total_steps = 0
            logger.info(
                "新しい SHT: %s → %d PA: %s",
                sht_prompt[:60],
                len(self._pa_queue) + 1,  # current_pa 含む
                [self._current_pa] + self._pa_queue,
            )

        # PA 完了判定
        if self._current_pa and self._is_pa_done(obs):
            if self._pa_queue:
                prev_pa = self._current_pa
                self._advance_pa()
                logger.info(
                    "PA 遷移: '%s' → '%s'（%d ステップ後）",
                    prev_pa,
                    self._current_pa,
                    self._step_count,
                )
            else:
                logger.debug("全 PA 完了、最後の PA を継続")

        # プロンプトを現在の PA に差し替えて内部 VLA に委譲
        effective_prompt = self._current_pa if self._current_pa else sht_prompt
        modified_obs = {**obs, "prompt": effective_prompt}
        result = self._vla.infer(modified_obs)

        # 状態追跡の更新
        state = obs.get("state")
        if state is not None:
            state_arr = np.asarray(state, dtype=np.float32)
            self._prev_state = state_arr
            self._state_history.append(state_arr)
            self._gripper_history.append(float(state_arr[5]))

        # 予測アクション追跡（FM 統合用）
        action = result.get("action")
        if action is not None:
            self._action_history.append(np.asarray(action, dtype=np.float32))

        self._step_count += 1
        self._total_steps += 1

        return result

    def reset(self) -> None:
        self._current_sht = None
        self._pa_queue = []
        self._current_pa = None
        self._step_count = 0
        self._prev_state = None
        self._total_steps = 0
        self._state_history = []
        self._gripper_history = []
        self._action_history = []
        self._completed_pas = []
        self._done_votes = []
        if self._retry_controller is not None:
            self._retry_controller.reset()
        self._vla.reset()

    # --- PA Planner ---

    def _insert_scan_pas(self, pa_list: list[str]) -> list[str]:
        """VS-1: navigate PA の後にスキャン PA を自動挿入する。

        navigate 完了後に物体がカメラに映らない場合に備えて、
        「Look around」PA を挿入してヘッドスキャンを促す。
        Return PA の直前には挿入しない。
        """
        result = []
        for i, pa in enumerate(pa_list):
            result.append(pa)
            pa_type = self._get_pa_type(pa)
            # navigate の後に scan を挿入（最後の PA と Return PA の前は除く）
            if pa_type == "navigate" and i + 1 < len(pa_list):
                next_pa = pa_list[i + 1]
                next_type = self._get_pa_type(next_pa)
                if next_type != "navigate":  # 連続 navigate には挿入しない
                    result.append(self._scan_pa_prompt)
                    logger.debug("VS-1: navigate '%s' の後にスキャン PA 挿入", pa[:40])
        return result

    def _decompose(self, sht_prompt: str) -> list[str]:
        """SHT プロンプトを順序付き PA 列に分解する。

        フォールバックチェーン:
          1. ルール完全一致 (pa_decomposition.json)
          2. ファジーマッチング (case-insensitive)
          3. LLMPAPlanner (pa_planner が設定されている場合)
          4. E2E フォールバック (SHT をそのまま PA 1 個として使用)
        """
        # 完全一致
        if sht_prompt in self._pa_map:
            return self._pa_map[sht_prompt]

        # ファジーマッチング: 大小文字・前後空白・末尾ピリオドを無視
        sht_norm = sht_prompt.lower().strip().rstrip(".")
        for key, pa_list in self._pa_map.items():
            if key.lower().strip().rstrip(".") == sht_norm:
                return pa_list

        # LLM PA Planner（設定されている場合）
        if self._pa_planner is not None:
            try:
                pa_list = self._pa_planner.decompose(sht_prompt)
                if pa_list and len(pa_list) > 0:
                    logger.info(
                        "LLM PA Planner で分解: '%s' → %d PA: %s",
                        sht_prompt[:60], len(pa_list), pa_list,
                    )
                    return pa_list
            except Exception as e:
                logger.warning(
                    "LLM PA Planner エラー: %s — E2E にフォールバック", e,
                )

        # フォールバック: SHT をそのまま使用（End-to-End 動作）
        logger.warning(
            "SHT '%s' の PA 分解が見つかりません。E2E にフォールバックします。",
            sht_prompt,
        )
        return [sht_prompt]

    # --- ステートマシン ---

    def _advance_pa(self) -> None:
        """次の PA に進む。"""
        if self._current_pa:
            self._completed_pas.append(self._current_pa)
        if self._pa_queue:
            self._current_pa = self._pa_queue.pop(0)
            self._step_count = 0
            self._prev_state = None
            self._state_history = []
            self._gripper_history = []
            self._action_history = []
            self._pa_start_gripper = None  # IM-3: PA 開始時の gripper 値を記録
            self._done_votes = []  # IM-2: 確認ウィンドウリセット
            # PA 切替時にテンポラルアンサンブルバッファをリセット
            if self._postprocessor is not None:
                self._postprocessor.reset_ensemble_buffer()
            # MoE: PA 遷移時に Expert を切り替え
            if self._on_pa_change is not None:
                try:
                    self._on_pa_change(self._current_pa)
                except Exception as e:
                    logger.warning("MoE Expert 切替失敗: %s", e)

    # --- PA Monitor ---

    def _get_pa_type(self, pa_name: str) -> str:
        """プロンプトのキーワードから PA 種別を推定する。"""
        pa_lower = pa_name.lower()

        for keyword in ["navigate", "go to", "go back", "move to", "return",
                        "move near", "find and move", "approach"]:
            if keyword in pa_lower:
                return "navigate"

        for keyword in ["grasp", "grab", "pick up", "pick an", "pick out",
                        "take the", "take a", "slide"]:
            if keyword in pa_lower:
                return "grasp"

        for keyword in ["place", "store", "put", "stock", "drop", "hang",
                        "discard", "insert", "stack"]:
            if keyword in pa_lower:
                return "place"

        if "open" in pa_lower:
            return "open"

        if "close" in pa_lower:
            return "close"

        return "default"

    def _min_steps_for_pa(self, pa_type: str) -> int:
        return self._min_steps.get(pa_type, self._min_steps.get("default", 8))

    def _max_steps_for_pa(self, pa_type: str) -> int:
        """PA 種別ごとの最大ステップ数を返す。

        P7-4: max_steps_short が設定されている場合、途中の PA には短縮値を使用し、
        最後の PA（タスク完了動作）には通常の max_steps を使用する。
        """
        is_last_pa = len(self._pa_queue) == 0
        if not is_last_pa and self._max_steps_short is not None:
            max_s = self._max_steps_short.get(pa_type, self._max_steps_short.get("default", 75))
        else:
            max_s = self._max_steps.get(pa_type, self._max_steps.get("default", 150))
        # 後方互換: max_steps_per_pa が明示指定されていればフォールバック
        if self._max_steps_fallback is not None:
            return min(max_s, self._max_steps_fallback)
        return max_s

    def _is_converged(self, window: int | None = None, threshold: float | None = None) -> bool:
        """スライディングウィンドウによる関節収束判定。

        直近 window フレームの各関節の標準偏差の最大値が threshold 未満なら収束と判定。
        1 フレーム差分と異なり、センサーノイズにロバスト。

        M1-1: DPO データ分析により threshold を 0.01→0.06 に緩和。
        成功エピソード p90 = 0.06 に基づく。
        """
        w = window or self._convergence_window
        t = threshold or self._joint_convergence_thresh
        if len(self._state_history) < w:
            return False
        recent = np.array(self._state_history[-w:])
        max_std = float(np.max(np.std(recent, axis=0)))
        return max_std < t

    def _is_action_entropy_low(self) -> bool:
        """TT-VF3: アクション分散が閾値以下かを判定。

        直近 action_entropy_window ステップのアクション履歴の分散平均が
        action_entropy_threshold 未満なら、ロボットが動いていない（タスク完了の可能性）と判定。
        """
        w = self._action_entropy_window
        if len(self._action_history) < w:
            return False
        recent = np.array(self._action_history[-w:])
        var_mean = float(np.mean(np.var(recent, axis=0)))
        return var_mean < self._action_entropy_threshold

    def _is_base_action_converged(self) -> bool:
        """P9-B: Navigate 専用収束判定。

        action の base 次元 (base_x, base_y, base_theta) の magnitude が
        閾値未満なら、ロボットの移動が収束した（目標付近に到達）と判定。

        observation.state に base 位置情報がないため、action 出力で代替する。
        base action が小さい = モデルが「移動不要」と判断 = Navigate 完了。
        """
        w = self._convergence_window
        if len(self._action_history) < w:
            return False
        recent = np.array(self._action_history[-w:])
        if recent.shape[1] <= max(self._base_action_indices):
            return False  # action 次元が足りない場合はフォールバック
        base_actions = recent[:, self._base_action_indices]
        base_mag = float(np.mean(np.abs(base_actions)))
        return base_mag < self._base_convergence_threshold

    def _is_arm_converged(self) -> bool:
        """EE-2: arm 関節 (dim 0-4) の収束判定。

        EE 位置が安定 = arm 操作が完了したことを示す。
        Pick/Place の完了判定の補助条件として使用。
        """
        w = self._convergence_window
        if len(self._state_history) < w:
            return False
        recent_arm = np.array([s[:5] for s in self._state_history[-w:]])
        max_std = float(np.max(np.std(recent_arm, axis=0)))
        return max_std < self._joint_convergence_thresh

    def _is_gripper_closing(self) -> bool:
        """EE-2: グリッパーが閉じ方向に動いているかを判定。"""
        if len(self._gripper_history) < 3:
            return False
        recent = self._gripper_history[-3:]
        return recent[-1] < recent[0]  # 直近で減少傾向

    def _is_gripper_opening(self) -> bool:
        """EE-2: グリッパーが開き方向に動いているかを判定。"""
        if len(self._gripper_history) < 3:
            return False
        recent = self._gripper_history[-3:]
        return recent[-1] > recent[0]  # 直近で増加傾向

    def _is_navigate_gripper_ready(self) -> bool:
        """P9-C: Navigate PA 中にグリッパー開動作を検出。

        Navigate 中にモデルが gripper を大きく開く action を出力した場合、
        「物体の前に到達して把持準備を始めた」と解釈し Navigate 完了とする。

        R2 データ分析: 成功 ep44 では Navigate→Pick 前に gripper が 0.4+ まで開く。
        シミュレータ検証: Step 30-40 で grip=0.39→0.88 (正面 1m 配置時)。
        """
        w = self._navigate_gripper_ready_window
        if len(self._action_history) < w:
            return False
        recent = self._action_history[-w:]
        gripper_idx = 5  # hand_motor_joint
        for action in recent:
            if len(action) > gripper_idx:
                if float(action[gripper_idx]) >= self._navigate_gripper_ready_threshold:
                    return True
        return False

    def _detect_gripper_transition(self, direction: str) -> bool:
        """グリッパーの遷移イベントを検知する。

        絶対値の閾値判定ではなく、直近ウィンドウ内での開→閉 / 閉→開の
        状態変化を検知する。実機のグリッパー物理状態に対してロバスト。

        Args:
            direction: "close"（開→閉）または "open"（閉→開）
        """
        w = self._gripper_window
        if len(self._gripper_history) < w * 2:
            return False
        prev_mean = float(np.mean(self._gripper_history[-w * 2:-w]))
        curr_mean = float(np.mean(self._gripper_history[-w:]))
        mid = (self._gripper_close_thresh + self._gripper_open_thresh) / 2
        if direction == "close":
            return prev_mean > mid and curr_mean < self._gripper_close_thresh
        else:  # "open"
            return prev_mean < mid and curr_mean > self._gripper_open_thresh

    def _detect_gripper_transition_magnitude(self, direction: str) -> bool:
        """IM-3: グリッパー遷移量ベースの PA 完了判定。

        DPO データ分析 (IN-B) に基づく:
          grasp 成功: gripper_transition = -1.72（強い閉動作）
          grasp 失敗: gripper_transition = -0.16（ほぼ動かず）
          place 成功: gripper_transition = +1.60（強い開動作）
          place 失敗: gripper_transition = +0.23（弱い）

        PA 開始時の gripper 値からの遷移量で判定する。
        絶対値閾値（< 0.3 / > 0.7）よりも初期状態に依存しないためロバスト。

        Args:
            direction: "close"（grasp: 閉方向遷移）または "open"（place: 開方向遷移）
        """
        if len(self._gripper_history) < 2:
            return False

        # PA 開始時の gripper 値を記録（初回のみ）
        if self._pa_start_gripper is None:
            self._pa_start_gripper = self._gripper_history[0]

        current_gripper = self._gripper_history[-1]
        transition = current_gripper - self._pa_start_gripper

        if direction == "close":
            # grasp: 閉方向（負の遷移）。閾値 -0.9
            return transition <= self._gripper_grasp_transition_thresh
        else:
            # place: 開方向（正の遷移）。閾値 +0.9
            return transition >= self._gripper_place_transition_thresh

    def _compute_failure_risk(self) -> float | None:
        """現在の PA の observation.state 履歴から failure risk を計算する。

        FIX-6: 予測アクション統計量から obs state ベースに変更。
        旧方式は同一チャンクのアクションで計算するため独立性がなかった。
        state は実際のロボット状態を反映するため、リアルタイム判定に適切。

        特徴量（8D、state ベース）:
          0. length:             log1p(ステップ数)
          1. state_velocity_mean: 関節変化量（1階差分ノルム）の平均
          2. state_velocity_std:  関節変化量のばらつき
          3. state_displacement:  初期位置からの変位ノルム
          4. state_jerk:          状態の三階差分ノルム平均
          5. gripper_std:         グリッパー状態の標準偏差
          6. gripper_range:       グリッパー状態の max - min
          7. stillness_ratio:     動いていないステップの割合（変化量 < 0.001）

        Returns:
            failure risk [0, 1] or None (monitor 未設定 or データ不足)
        """
        if self._failure_monitor is None or len(self._state_history) < 5:
            return None

        states = np.array(self._state_history)
        n_steps = len(states)
        gripper = np.array(self._gripper_history)

        # 関節変化量（1階差分）
        velocity = np.diff(states, axis=0)
        velocity_norms = np.linalg.norm(velocity, axis=1)

        # 三階差分（jerk）
        if n_steps >= 4:
            jerk = np.diff(states, n=3, axis=0)
            state_jerk = float(np.mean(np.linalg.norm(jerk, axis=1)))
        else:
            state_jerk = 0.0

        # 初期位置からの変位
        displacement = float(np.linalg.norm(states[-1] - states[0]))

        # 動いていないステップの割合
        stillness_ratio = float(np.mean(velocity_norms < 0.001)) if len(velocity_norms) > 0 else 1.0

        features = np.array([
            np.log1p(n_steps),                     # length
            float(velocity_norms.mean()),          # state_velocity_mean
            float(velocity_norms.std()),           # state_velocity_std
            displacement,                          # state_displacement
            state_jerk,                            # state_jerk
            float(gripper.std()),                  # gripper_std
            float(gripper.max() - gripper.min()),  # gripper_range
            stillness_ratio,                       # stillness_ratio
        ], dtype=np.float32).reshape(1, -1)

        return float(self._failure_monitor.predict_risk(features))

    def _is_pa_done(self, obs: Dict) -> bool:
        """複合ルールベースの PA 完了判定。

        改善点（v2）:
          - 1c: PA 種別ごとの動的 max_steps（navigate: 200, grasp: 100 等）
          - 1a: スライディングウィンドウによる関節収束判定（navigate, open/close）
          - 1b: グリッパー遷移イベント検知（grasp, place）
        M2-1: failure_risk ベースの失敗検知を追加
        """
        if not self._current_pa:
            return False

        pa_type = self._get_pa_type(self._current_pa)

        # M2-1: failure risk が閾値を超えたら失敗と判定 → RetryController に委譲
        # FIX-7: チャンク境界でのみ FM を実行（eh=10 なら 10 ステップごと）
        fm_interval = self._fm_check_interval
        should_check_fm = (fm_interval <= 0) or (self._step_count % fm_interval == 0)
        failure_risk = self._compute_failure_risk() if should_check_fm else None
        if failure_risk is not None and failure_risk >= self._failure_risk_threshold:
            if self._retry_controller is not None:
                decision = self._retry_controller.decide(self._current_pa, pa_type, failure_risk)
                if decision.action == "retry":
                    logger.info(
                        "FM 失敗検知 (risk=%.3f): PA '%s' を retry (%s)",
                        failure_risk, self._current_pa, decision.reason,
                    )
                    # retry: 現在の PA をリセットして再実行
                    self._step_count = 0
                    self._action_history.clear()
                    self._state_history.clear()
                    self._gripper_history.clear()
                    self._vla.reset()
                    if self._postprocessor is not None:
                        self._postprocessor.reset_ensemble_buffer()
                    return False  # PA 完了ではない（retry 中）
                elif decision.action == "abort":
                    logger.warning(
                        "FM abort (risk=%.3f): PA '%s' — %s",
                        failure_risk, self._current_pa, decision.reason,
                    )
                    return True  # 次の PA に進む（abort = skip）
                elif decision.action == "replan":
                    # IM-1: LLM に再計画を依頼
                    if self._pa_planner is not None and hasattr(self._pa_planner, 'replan'):
                        new_remaining = self._pa_planner.replan(
                            sht=self._current_sht or "",
                            completed_pas=list(self._completed_pas),
                            failed_pa=self._current_pa,
                            failure_reason="high_failure_risk",
                            remaining_pas=list(self._pa_queue),
                        )
                        self._pa_queue = new_remaining
                        logger.info(
                            "FM replan (risk=%.3f): PA '%s' → %d new PAs: %s",
                            failure_risk, self._current_pa, len(new_remaining), new_remaining,
                        )
                    else:
                        logger.info(
                            "FM replan (risk=%.3f): PA '%s' — no planner, skip",
                            failure_risk, self._current_pa,
                        )
                    return True
                elif decision.action == "skip":
                    logger.info(
                        "FM skip (risk=%.3f): PA '%s' — %s",
                        failure_risk, self._current_pa, decision.reason,
                    )
                    return True
            else:
                logger.debug("FM failure risk=%.3f (threshold=%.2f), no RetryController",
                             failure_risk, self._failure_risk_threshold)

        # 安全弁: PA 種別ごとの最大ステップ超過で強制遷移
        max_steps = self._max_steps_for_pa(pa_type)
        if self._step_count >= max_steps:
            logger.debug(
                "PA '%s' (%s) が最大ステップ数 (%d) に到達、強制遷移",
                self._current_pa,
                pa_type,
                max_steps,
            )
            return True

        # 最低ステップ数保証（早期遷移を防止）
        if self._step_count < self._min_steps_for_pa(pa_type):
            return False

        if pa_type == "grasp":
            gripper_done = self._detect_gripper_transition_magnitude("close")
            # EE-2: gripper 遷移 AND arm 収束（安全弁）
            # arm がまだ動いている場合は gripper 遷移だけでは完了しない
            raw_done = gripper_done and self._is_arm_converged()
        elif pa_type == "place":
            gripper_done = self._detect_gripper_transition_magnitude("open")
            raw_done = gripper_done and self._is_arm_converged()
        elif pa_type == "navigate":
            raw_done = (self._is_converged()
                        or self._is_action_entropy_low()
                        or self._is_base_action_converged()
                        or self._is_navigate_gripper_ready())
        elif pa_type in ("open", "close"):
            w = self._convergence_window
            if len(self._state_history) < w:
                raw_done = False
            else:
                recent_arm = np.array([s[:5] for s in self._state_history[-w:]])
                raw_done = float(np.max(np.std(recent_arm, axis=0))) < 0.01
        else:
            raw_done = self._is_converged() or self._is_action_entropy_low()

        # IM-2: 成功確認ウィンドウ（多数決で誤検出防止）
        self._done_votes.append(raw_done)
        window = self._confirmation_window
        threshold = self._confirmation_threshold
        if window <= 1:
            return raw_done  # 無効化（デフォルト）
        recent_votes = self._done_votes[-window:]
        return sum(recent_votes) >= threshold

    # --- 診断用 ---

    def get_state(self) -> dict:
        """現在の Hierarchical 状態を返す（デバッグ・ログ用）。"""
        state = {
            "current_sht": self._current_sht,
            "current_pa": self._current_pa,
            "pa_queue_remaining": list(self._pa_queue),
            "step_count": self._step_count,
            "total_steps": self._total_steps,
            "failure_monitor_enabled": self._failure_monitor is not None,
        }
        if self._retry_controller is not None:
            state["retry_stats"] = self._retry_controller.get_stats()
        return state
