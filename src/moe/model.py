"""VLA-MoE Model: Action Expert を MoE 化した π₀.₅ モデル。

設計方針:
- PaliGemmaWithExpertModel / PI05Pytorch を継承
- FFN-only MoE: Attention + LayerNorm は共有、MLP のみ per-expert
- select_expert() で gemma_expert 内の MLP 参照を差し替え
- forward() / sample_actions() / denoise_step() は変更不要
"""

from __future__ import annotations

import logging
import re
from typing import Literal

import torch
import torch.nn as nn

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import (
    PI05Pytorch,
    PaliGemmaWithExpertModel,
    get_gemma_config,
)
try:
    from lerobot.policies.pi_gemma import PiGemmaForCausalLM
except ImportError:
    # ramen ブランチ固有モジュール。Docker 内で見つからない場合は
    # transformers の GemmaForCausalLM でフォールバック
    from transformers import GemmaForCausalLM as PiGemmaForCausalLM
from transformers.models.gemma.modeling_gemma import GemmaMLP

from .config import MoEConfig
from .router import BaseRouter, RuleBasedRouter

logger = logging.getLogger(__name__)


class PaliGemmaWithExpertMoEModel(PaliGemmaWithExpertModel):
    """PaliGemmaWithExpertModel の MoE 拡張版。

    FFN-only MoE (ffn_only_moe=True):
      - VE + VLM + Expert Attention + LayerNorm は共有（1 セット）
      - MLP (gate/up/down proj) のみ per-expert で expert_ffns に保持
      - select_expert() で gemma_expert.model.layers[i].mlp を差し替え

    Full-expert MoE (ffn_only_moe=False, 後方互換):
      - 従来通り PiGemmaForCausalLM 丸ごとを per-expert で保持
    """

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        num_experts: int,
        ffn_only_moe: bool = True,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = 224,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        # 親クラスで base expert (self.gemma_expert) を1つ作成
        super().__init__(
            vlm_config=vlm_config,
            action_expert_config=action_expert_config,
            use_adarms=use_adarms,
            precision=precision,
            image_size=image_size,
            freeze_vision_encoder=freeze_vision_encoder,
            train_expert_only=train_expert_only,
        )

        self.num_experts = num_experts
        self.ffn_only_moe = ffn_only_moe

        from transformers import CONFIG_MAPPING

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1] if use_adarms else False,
            adarms_cond_dim=action_expert_config.width if (use_adarms and use_adarms[1]) else None,
        )

        if ffn_only_moe:
            # FFN-only MoE: Attention + LayerNorm は gemma_expert 内で共有、MLP のみ per-expert
            depth = action_expert_config.depth

            # Expert 0: gemma_expert 内の既存 MLP への参照
            expert_0_mlps = nn.ModuleList([
                self.gemma_expert.model.layers[i].mlp for i in range(depth)
            ])
            # Expert 1〜N-1: 新規 GemmaMLP
            additional_ffns = [
                nn.ModuleList([
                    GemmaMLP(action_expert_config_hf) for _ in range(depth)
                ])
                for _ in range(num_experts - 1)
            ]
            self.expert_ffns = nn.ModuleList([expert_0_mlps] + additional_ffns)
        else:
            # Full-expert MoE（後方互換）
            additional_experts = [
                PiGemmaForCausalLM(config=action_expert_config_hf)
                for _ in range(num_experts - 1)
            ]
            for expert in additional_experts:
                expert.model.embed_tokens = None

            self.gemma_experts = nn.ModuleList(
                [self.gemma_expert] + additional_experts
            )

        # precision 適用
        self.to_bfloat16_for_selected_params(precision)
        # requires_grad 設定
        self._set_requires_grad()

        # デフォルトは Expert 0
        self._active_expert_idx = 0

    def select_expert(self, idx: int) -> None:
        """Active Expert を切り替える。

        FFN-only: gemma_expert.model.layers[i].mlp を expert_ffns[idx][i] に差し替え。
        Full-expert: self.gemma_expert への参照を差し替え。
        """
        if idx < 0 or idx >= self.num_experts:
            raise ValueError(
                f"Expert index {idx} out of range [0, {self.num_experts})"
            )
        self._active_expert_idx = idx

        if self.ffn_only_moe:
            for i in range(len(self.gemma_expert.model.layers)):
                self.gemma_expert.model.layers[i].mlp = self.expert_ffns[idx][i]
        else:
            self.gemma_expert = self.gemma_experts[idx]

        logger.debug("Selected Expert %d", idx)

    # MLP alias キーのパターン（FFN-only 用）
    _MLP_ALIAS_PATTERN = re.compile(
        r"\.gemma_expert\.model\.layers\.\d+\.mlp\."
    )

    def state_dict(self, *args, **kwargs):
        """エイリアスキーを除外して safetensors の重複を防ぐ。

        FFN-only: gemma_expert.*.mlp.* を除外（expert_ffns.* が canonical）。
                  gemma_expert.*.self_attn.* 等の共有重みは保持。
        Full-expert: gemma_expert.* を全て除外（gemma_experts.* が canonical）。
        """
        sd = super().state_dict(*args, **kwargs)
        if self.ffn_only_moe:
            return {
                k: v for k, v in sd.items()
                if not self._MLP_ALIAS_PATTERN.search(k)
            }
        else:
            return {
                k: v for k, v in sd.items()
                if ".gemma_expert." not in k
            }


class PI05MoEPytorch(PI05Pytorch):
    """PI05Pytorch の MoE 拡張版。

    per-expert の projection 層（action_in_proj, action_out_proj, time_mlp_in/out）を
    ModuleList で保持し、select_expert() で差し替える。
    """

    def __init__(self, config: PI05Config, moe_config: MoEConfig, **kwargs):
        # 親クラスの __init__ を呼ぶと、以下が作成される:
        # - self.paligemma_with_expert (PaliGemmaWithExpertModel)
        # - self.action_in_proj, action_out_proj, time_mlp_in, time_mlp_out
        # ただし、paligemma_with_expert を MoE 版に差し替える必要がある

        # 親クラスの __init__ をスキップして手動構築
        nn.Module.__init__(self)
        self.config = config
        self.rtc_processor = None

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution: {config.image_resolution}"
            )

        # MoE 版の PaliGemmaWithExpertModel を使用
        self.paligemma_with_expert = PaliGemmaWithExpertMoEModel(
            paligemma_config,
            action_expert_config,
            num_experts=moe_config.num_experts,
            ffn_only_moe=moe_config.ffn_only_moe,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        # Expert 0 の projection（親クラスと同様）
        base_proj_in = nn.Linear(config.max_action_dim, action_expert_config.width)
        base_proj_out = nn.Linear(action_expert_config.width, config.max_action_dim)
        base_time_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        base_time_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # per-expert projection 層
        self.action_in_projs = nn.ModuleList(
            [base_proj_in] + [
                nn.Linear(config.max_action_dim, action_expert_config.width)
                for _ in range(moe_config.num_experts - 1)
            ]
        )
        self.action_out_projs = nn.ModuleList(
            [base_proj_out] + [
                nn.Linear(action_expert_config.width, config.max_action_dim)
                for _ in range(moe_config.num_experts - 1)
            ]
        )
        self.time_mlp_ins = nn.ModuleList(
            [base_time_in] + [
                nn.Linear(action_expert_config.width, action_expert_config.width)
                for _ in range(moe_config.num_experts - 1)
            ]
        )
        self.time_mlp_outs = nn.ModuleList(
            [base_time_out] + [
                nn.Linear(action_expert_config.width, action_expert_config.width)
                for _ in range(moe_config.num_experts - 1)
            ]
        )

        # active Expert の projection への参照（forward() で使用される名前）
        self.action_in_proj = self.action_in_projs[0]
        self.action_out_proj = self.action_out_projs[0]
        self.time_mlp_in = self.time_mlp_ins[0]
        self.time_mlp_out = self.time_mlp_outs[0]

        # Router
        self.moe_config = moe_config
        self.router: BaseRouter = RuleBasedRouter(
            default_expert=moe_config.default_expert,
        )

        # gradient checkpointing フラグ
        self.gradient_checkpointing_enabled = False

        # Correlated noise キャッシュ
        self._correlated_noise_cache: dict[tuple[str, torch.dtype], tuple] = {}

        # torch.compile（推論時に有効化する場合）
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(
                self.sample_actions, mode=config.compile_mode, backend="aot_eager"
            )

    @property
    def num_experts(self) -> int:
        return self.moe_config.num_experts

    @property
    def active_expert_idx(self) -> int:
        return self.paligemma_with_expert._active_expert_idx

    def select_expert(self, idx: int) -> None:
        """Expert を切り替える（gemma_expert + projection 全て）。"""
        self.paligemma_with_expert.select_expert(idx)
        self.action_in_proj = self.action_in_projs[idx]
        self.action_out_proj = self.action_out_projs[idx]
        self.time_mlp_in = self.time_mlp_ins[idx]
        self.time_mlp_out = self.time_mlp_outs[idx]
        logger.info("PI05MoE: Selected Expert %d", idx)

    def select_expert_by_instruction(self, instruction: str) -> int:
        """タスク指示テキストから Expert を選択する。

        Returns:
            選択された Expert のインデックス。
        """
        idx = self.router.route(instruction)
        self.select_expert(idx)
        return idx

    # MLP alias キーのパターン（FFN-only 用）
    _MLP_ALIAS_PATTERN = re.compile(
        r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.mlp\."
    )

    def state_dict(self, *args, **kwargs):
        """エイリアスキーを除外して重複を防ぐ。

        FFN-only: gemma_expert.*.mlp.* のみ除外（共有 Attention/LayerNorm は保持）。
        Full-expert: gemma_expert.* を全て除外（gemma_experts.* が canonical）。
        Projection alias (action_in_proj 等) は両モードで除外。
        """
        sd = super().state_dict(*args, **kwargs)

        # Projection alias は両モードで除外
        proj_alias_markers = [
            "action_in_proj.",
            "action_out_proj.",
            "time_mlp_in.",
            "time_mlp_out.",
        ]

        if self.paligemma_with_expert.ffn_only_moe:
            # FFN-only: MLP alias のみ除外、共有 Attention/LayerNorm は保持
            return {
                k: v for k, v in sd.items()
                if not any(m in k for m in proj_alias_markers)
                and not self._MLP_ALIAS_PATTERN.search(k)
            }
        else:
            # Full-expert: gemma_expert.* を全て除外
            alias_markers = [".gemma_expert."] + proj_alias_markers
            return {
                k: v for k, v in sd.items()
                if not any(m in k for m in alias_markers)
            }

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """エイリアスキーが無い state_dict をロードした後、select_expert で参照を復元。"""
        result = super().load_state_dict(state_dict, strict=False, **kwargs)
        # ロード後にデフォルト Expert への参照を再設定
        self.select_expert(0)
        return result

    def gradient_checkpointing_enable(self):
        """全 Expert に対して gradient checkpointing を有効化。"""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        if self.paligemma_with_expert.ffn_only_moe:
            self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        else:
            for expert in self.paligemma_with_expert.gemma_experts:
                expert.model.gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing for PI05MoEPytorch")

    def gradient_checkpointing_disable(self):
        """全 Expert に対して gradient checkpointing を無効化。"""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        if self.paligemma_with_expert.ffn_only_moe:
            self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        else:
            for expert in self.paligemma_with_expert.gemma_experts:
                expert.model.gradient_checkpointing = False
        logger.info("Disabled gradient checkpointing for PI05MoEPytorch")
