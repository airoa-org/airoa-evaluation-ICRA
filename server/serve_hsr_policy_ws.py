#!/usr/bin/env python3
"""Minimal websocket policy server for the AIRoA HSR evaluation harness.

This is the **starting template** distributed on the `base` branch. It
serves a `ZeroPolicy` placeholder that returns valid-shaped zero actions so
the harness round-trip (client → server → client) can be smoke-tested out
of the box, before you have integrated your model.

Replace `ZeroPolicy` (or swap it via `--policy-module`) with your own policy
that exposes a single method:

    policy.infer(obs: dict) -> dict

See `docs/INTEGRATION_GUIDE.md` for the full integration walkthrough and
`docs/FAQ.md` §3 for the WebSocket I/O contract this server implements.

For an OpenPI-loader example, fork from the `sample-openpi` branch instead.
"""

import argparse
import importlib
import logging
import os
from pathlib import Path

import numpy as np

from runtime_core.websocket_policy_server import WebsocketPolicyServer


class ZeroPolicy:
    """Placeholder policy that returns zero actions of the correct shape.

    Useful only for verifying the harness round-trip. Replace with your own
    policy class for real evaluation.
    """

    def __init__(self, checkpoint_dir: str | None = None) -> None:
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            logging.info("ZeroPolicy: checkpoint_dir=%s (not loaded — placeholder)", checkpoint_dir)

    @property
    def metadata(self) -> dict:
        return {"policy": "zero", "actions_shape": [1, 11]}

    def infer(self, obs: dict) -> dict:
        # Contract reminder (see docs/FAQ.md §3):
        #   obs["head_rgb"]: (480, 640, 3) uint8
        #   obs["hand_rgb"]: (480, 640, 3) uint8
        #   obs["state"]:    (8,)          float32
        #   obs["prompt"]:   str
        # Return: {"actions": np.ndarray of shape (T, 11), dtype=float32}
        return {"actions": np.zeros((1, 11), dtype=np.float32)}


def _load_policy(policy_module: str | None, checkpoint_dir: str | None):
    """Load `policy_module:Class` if provided, else fall back to ZeroPolicy."""
    if not policy_module:
        return ZeroPolicy(checkpoint_dir=checkpoint_dir)

    if ":" not in policy_module:
        raise ValueError(
            f"--policy-module must be in 'module:Class' form, got: {policy_module!r}"
        )
    module_name, class_name = policy_module.split(":", 1)
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(checkpoint_dir=checkpoint_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIRoA HSR evaluation websocket policy server")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Path to checkpoint directory (passed to your policy class).",
    )
    parser.add_argument(
        "--policy-module",
        default=os.environ.get("POLICY_MODULE"),
        help=(
            "Import path of your policy class in 'module:Class' form, e.g. "
            "'my_policy.adapter:MyPolicyAdapter'. Defaults to the placeholder ZeroPolicy."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoint_dir:
        checkpoint_dir = str(Path(args.checkpoint_dir).expanduser())
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")
    else:
        checkpoint_dir = None

    policy = _load_policy(args.policy_module, checkpoint_dir)
    metadata = dict(getattr(policy, "metadata", {}))
    metadata.update(
        {
            "checkpoint_dir": checkpoint_dir,
            "policy_module": args.policy_module or "ZeroPolicy",
            "server_host": args.host,
            "server_port": args.port,
        }
    )

    logging.info(
        "Serving policy=%s checkpoint=%s on %s:%s",
        args.policy_module or "ZeroPolicy",
        checkpoint_dir,
        args.host,
        args.port,
    )
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
