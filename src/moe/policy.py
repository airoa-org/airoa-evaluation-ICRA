"""VLA-MoE Policy: MoE 対応の PI05Policy。

PI05Policy を継承し、MoE モデルのロード・Expert 選択・推論を提供。
eval チームは from_pretrained() + select() + select_action() の3メソッドだけ使えば良い。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TypeVar

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

from .config import MOE_CONFIG_FILENAME, MoEConfig
from .model import PI05MoEPytorch

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="PI05MoEPolicy")


class PI05MoEPolicy(PI05Policy):
    """MoE 対応の PI05Policy。

    使い方:
        policy = PI05MoEPolicy.from_pretrained("path/to/moe_checkpoint")
        policy.select("Pick the coffee from the table")
        action = policy.select_action(batch)
    """

    def __init__(self, config: PI05Config, moe_config: MoEConfig, _skip_model_build: bool = False, **kwargs):
        # PI05Policy.__init__ を完全にバイパスし、手動で構築
        # （PI05Policy.__init__ は PI05Pytorch を作るが、我々は PI05MoEPytorch が必要）
        from lerobot.policies.pretrained import PreTrainedPolicy

        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.moe_config = moe_config

        if _skip_model_build:
            # low_cpu_mem モード: モデル構築を遅延（from_pretrained 内で GPU 上に直接構築）
            self.model = None
            return

        # MoE モデルを構築
        self.init_rtc_processor()
        self.model = PI05MoEPytorch(
            config,
            moe_config,
            rtc_processor=self.rtc_processor if hasattr(self, "rtc_processor") else None,
        )

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    def select(self, task_instruction: str) -> int:
        """タスク指示テキストから Expert を選択する。

        Args:
            task_instruction: タスク指示テキスト（例: "Pick the coffee"）

        Returns:
            選択された Expert のインデックス。
        """
        return self.model.select_expert_by_instruction(task_instruction)

    def select_expert(self, idx: int) -> None:
        """Expert インデックスを直接指定して切り替える。"""
        self.model.select_expert(idx)

    @property
    def num_experts(self) -> int:
        return self.model.num_experts

    @property
    def active_expert_idx(self) -> int:
        return self.model.active_expert_idx

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        strict: bool = False,
        low_cpu_mem: bool = True,
        **kwargs,
    ) -> T:
        """MoE checkpoint からモデルをロードする。

        checkpoint ディレクトリ構成:
            model.safetensors  — 統合モデル重み
            config.json        — PI05Config
            moe_config.json    — MoE 設定

        Args:
            low_cpu_mem: True の場合、safetensors を直接 GPU にロードし
                CPU RAM の消費を最小化する（RAM 31GB 環境対応）。
        """
        import gc

        pretrained_path = Path(pretrained_name_or_path)

        # PI05Config をロード
        # draccus は config.json の "type" フィールドでエラーになるため、
        # 一時ファイルから "type" を除去してパースする
        if config is None:
            import tempfile
            config_path = pretrained_path / "config.json"
            with open(config_path) as f:
                cfg_dict = json.load(f)
            cfg_dict.pop("type", None)
            cfg_dict["compile_model"] = False
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
                json.dump(cfg_dict, tmp)
                tmp_path = tmp.name
            try:
                import draccus
                with draccus.config_type("json"):
                    config = draccus.parse(PI05Config, tmp_path, args=[])
            finally:
                os.remove(tmp_path)
            logger.info("Loaded PI05Config (type removed, compile_model=False)")

        # MoEConfig をロード
        moe_config_path = pretrained_path / MOE_CONFIG_FILENAME
        if not moe_config_path.exists():
            raise FileNotFoundError(
                f"MoE config not found: {moe_config_path}. "
                "This checkpoint may not be a MoE model."
            )
        moe_config = MoEConfig.load(pretrained_path)
        logger.info(
            "Loaded MoE config: %d experts (%s)",
            moe_config.num_experts,
            moe_config.expert_names,
        )

        # 重みファイルの確認
        model_path = pretrained_path / "model.safetensors"
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        if low_cpu_mem:
            # RAM 節約モード:
            # 1. meta デバイスでモデル構造を作成（RAM 0, VRAM 0）
            # 2. safetensors を直接 GPU にロード
            # 3. load_state_dict(assign=True) で meta テンソルを GPU テンソルに置換
            logger.info("Low CPU memory mode: building model on meta device...")
            model = cls(config, moe_config, _skip_model_build=True, **kwargs)
            model.init_rtc_processor()
            with torch.device("meta"):
                model.model = PI05MoEPytorch(
                    config,
                    moe_config,
                    rtc_processor=model.rtc_processor if hasattr(model, "rtc_processor") else None,
                )
            gc.collect()
            logger.info("Model structure created on meta device (0 RAM, 0 VRAM)")

            logger.info("Loading weights directly to GPU...")
            state_dict = load_file(str(model_path), device="cuda")
        else:
            # 通常モード: CPU 上でモデル構築 + CPU ロード → GPU 転送
            model = cls(config, moe_config, **kwargs)
            state_dict = load_file(str(model_path))

        logger.info("Loaded state dict: %d keys", len(state_dict))

        # "model." プレフィックスの追加（PI05Policy と同じ処理）
        remapped = {}
        for key, value in state_dict.items():
            new_key = f"model.{key}" if not key.startswith("model.") else key
            remapped[new_key] = value
        del state_dict
        gc.collect()

        # assign=True: meta テンソルを GPU テンソルで直接置換（コピーではなく参照差替）
        missing, unexpected = model.load_state_dict(remapped, strict=strict, assign=True)
        del remapped
        gc.collect()
        torch.cuda.empty_cache()

        if missing:
            logger.warning("Missing keys: %d", len(missing))
            for k in missing[:10]:
                logger.warning("  - %s", k)
        if unexpected:
            logger.warning("Unexpected keys: %d", len(unexpected))
            for k in unexpected[:10]:
                logger.warning("  - %s", k)

        if not missing and not unexpected:
            logger.info("All keys loaded successfully!")

        # デフォルトで Expert 0 を選択（FFN-only: expert_ffns[0] の MLP を差し替え）
        model.model.select_expert(0)

        # low_cpu_mem モード: meta テンソルが残っている場合、空の GPU テンソルで初期化
        if low_cpu_mem:
            for name, param in model.named_parameters():
                if param.device == torch.device("meta"):
                    logger.debug("Replacing meta param: %s", name)
                    param.data = torch.zeros(param.shape, device="cuda", dtype=param.dtype)

        # _skip_model_build=True の場合 reset() が未実行のため、ここで呼ぶ
        model.reset()

        return model

    def save_pretrained(self, save_directory: str | Path, **kwargs) -> None:
        """MoE モデルを checkpoint ディレクトリに保存する。"""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # PI05Config を保存
        self.config._save_pretrained(save_dir)

        # MoE Config を保存
        self.moe_config.save(save_dir)

        # モデル重みを保存
        state_dict = {}
        for key, value in self.model.state_dict().items():
            state_dict[key] = value

        save_file(state_dict, str(save_dir / "model.safetensors"))
        logger.info(
            "Saved MoE model to %s (%d keys, %d experts)",
            save_dir,
            len(state_dict),
            self.num_experts,
        )
