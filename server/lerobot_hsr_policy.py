#!/usr/bin/env python3
"""LeRobot PI05Policy wrapper for HSR WebSocket serving.

Wraps a merged LeRobot PI05Policy checkpoint to serve over the WebSocket
protocol expected by the AIRoA evaluation HSR client.

Camera mapping (training data → HSR client):
    hand_rgb  → observation.images.left_wrist_0_rgb
    head_rgb  → observation.images.base_0_rgb
    right_wrist_0_rgb, empty_camera_0 → dummy (-1.0)

Action padding:
    Model outputs 8D → padded to 11D (3 zeros for base_x, base_y, base_t)
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from policy_client.base_policy import BasePolicy

logger = logging.getLogger(__name__)

_IMAGE_SIZE = (224, 224)
_HSR_ACTION_DIM = 11  # 8 joints + 3 base twist


class LeRobotHSRPolicy(BasePolicy):
    """LeRobot PI05Policy wrapper for HSR WebSocket serving."""

    def __init__(self, checkpoint_dir: str, device: str = "cuda", default_prompt: str | None = None):
        self._device = device
        self._checkpoint_dir = checkpoint_dir
        self._default_prompt = default_prompt or ""

        logger.info("Loading PI05Policy from %s on %s", checkpoint_dir, device)

        from lerobot.processor.pipeline import DataProcessorPipeline

        # MoE auto-detection: use PI05MoEPolicy if moe_config.json exists
        moe_config_path = Path(checkpoint_dir) / "moe_config.json"
        if moe_config_path.exists():
            logger.info("Detected moe_config.json → loading PI05MoEPolicy")
            from moe.policy import PI05MoEPolicy
            self._policy = PI05MoEPolicy.from_pretrained(checkpoint_dir)
        else:
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            self._policy = PI05Policy.from_pretrained(checkpoint_dir)

        self._policy.eval()
        self._policy.to(device)

        self._preprocessor = DataProcessorPipeline.from_pretrained(
            checkpoint_dir,
            "policy_preprocessor.json",
            overrides={"device_processor": {"device": device}},
        )
        self._postprocessor = DataProcessorPipeline.from_pretrained(
            checkpoint_dir,
            "policy_postprocessor.json",
            overrides={"device_processor": {"device": device}},
        )

        logger.info("PI05Policy loaded successfully")

    @property
    def supports_moe(self) -> bool:
        """Whether the loaded policy supports MoE expert selection."""
        return hasattr(self._policy, "select")

    def select_expert(self, instruction: str) -> int | None:
        """Select MoE expert based on instruction. Returns expert index or None if not MoE."""
        if not self.supports_moe:
            return None
        # Route only (no weight swap yet)
        idx = self._policy.model.router.route(instruction)
        # Skip redundant weight swap if same expert is already active
        if idx != self._policy.model.active_expert_idx:
            self._policy.model.select_expert(idx)
        return idx

    def infer(self, obs: dict) -> dict:
        """Convert HSR observation to LeRobot format, run inference, and return actions.

        Input obs keys:
            hand_rgb: [H, W, 3] uint8
            head_rgb: [H, W, 3] uint8
            state: [8] float
            prompt: str

        Returns:
            {"actions": np.ndarray [action_horizon, 11]}
        """
        hand_img = self._prepare_image(obs["hand_rgb"])
        head_img = self._prepare_image(obs["head_rgb"])
        dummy_img = torch.full((3, *_IMAGE_SIZE), -1.0, device=self._device)

        prompt = obs.get("prompt", self._default_prompt) or self._default_prompt

        batch = {
            "observation.images.left_wrist_0_rgb": hand_img.unsqueeze(0),
            "observation.images.base_0_rgb": head_img.unsqueeze(0),
            "observation.images.right_wrist_0_rgb": dummy_img.unsqueeze(0),
            "observation.images.empty_camera_0": dummy_img.unsqueeze(0),
            "observation.state": torch.tensor(
                np.asarray(obs["state"], dtype=np.float32),
                device=self._device,
            ).unsqueeze(0),
            "task": prompt,
        }

        batch = self._preprocessor(batch)
        with torch.inference_mode():
            action = self._policy.select_action(batch)
        result = self._postprocessor({"action": action})

        action_out = result["action"].cpu().numpy()

        # Ensure 2D: [action_horizon, action_dim]
        if action_out.ndim == 1:
            action_out = action_out[np.newaxis, :]

        # Pad model output → HSR 11D (append zeros for base_x, base_y, base_t)
        if action_out.shape[-1] < _HSR_ACTION_DIM:
            pad_width = _HSR_ACTION_DIM - action_out.shape[-1]
            action_out = np.pad(action_out, ((0, 0), (0, pad_width)), mode="constant")

        return {"actions": action_out}

    def reset(self) -> None:
        self._policy.reset()

    def _prepare_image(self, img: np.ndarray) -> torch.Tensor:
        """Convert raw image to [3, 224, 224] float tensor."""
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        if np.issubdtype(img.dtype, np.floating):
            img = (img * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        img = img.resize(_IMAGE_SIZE, Image.Resampling.BICUBIC)
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).to(self._device)

    @property
    def metadata(self) -> dict:
        return {
            "backend": "lerobot",
            "checkpoint_dir": self._checkpoint_dir,
        }
