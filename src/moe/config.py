"""VLA-MoE Config: MoE モデルの設定を管理する。

PI05Config とは独立した MoE 固有の設定。
JSON でシリアライズ/デシリアライズ可能。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# MoE config ファイル名（checkpoint ディレクトリ内）
MOE_CONFIG_FILENAME = "moe_config.json"


@dataclass
class ExpertConfig:
    """個別 Expert の設定。"""

    name: str  # Expert 名（例: "pick_coffee", "general"）
    checkpoint: str  # 元の checkpoint パス or HF repo ID
    description: str = ""  # 説明（任意）


@dataclass
class MoEConfig:
    """MoE モデル全体の設定。"""

    num_experts: int = 7
    experts: list[ExpertConfig] = field(default_factory=list)
    routing_type: str = "rule_based"  # "rule_based" | "learned"
    default_expert: int = 6  # マッチしない場合のフォールバック Expert ID
    base_checkpoint: str = ""  # 共有重み（VE+VLM）のソース checkpoint
    ffn_only_moe: bool = False  # True: MLP のみ per-expert, False: Expert 全体を複製
    shared_source_expert: int = -1  # 共有 Attention+LayerNorm のソース Expert ID (-1: base_checkpoint)

    def save(self, save_dir: str | Path) -> None:
        """MoE config を JSON ファイルに保存する。"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        config_path = save_dir / MOE_CONFIG_FILENAME

        data = {
            "num_experts": self.num_experts,
            "experts": [
                {
                    "name": e.name,
                    "checkpoint": e.checkpoint,
                    "description": e.description,
                }
                for e in self.experts
            ],
            "routing_type": self.routing_type,
            "default_expert": self.default_expert,
            "base_checkpoint": self.base_checkpoint,
            "ffn_only_moe": self.ffn_only_moe,
            "shared_source_expert": self.shared_source_expert,
        }

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, load_dir: str | Path) -> MoEConfig:
        """MoE config を JSON ファイルから読み込む。"""
        load_dir = Path(load_dir)
        config_path = load_dir / MOE_CONFIG_FILENAME

        with open(config_path) as f:
            data = json.load(f)

        experts = [ExpertConfig(**e) for e in data.get("experts", [])]

        return cls(
            num_experts=data["num_experts"],
            experts=experts,
            routing_type=data.get("routing_type", "rule_based"),
            default_expert=data.get("default_expert", len(experts) - 1),
            base_checkpoint=data.get("base_checkpoint", ""),
            ffn_only_moe=data.get("ffn_only_moe", False),
            shared_source_expert=data.get("shared_source_expert", -1),
        )

    @property
    def expert_names(self) -> list[str]:
        """Expert 名のリストを返す。"""
        return [e.name for e in self.experts]
