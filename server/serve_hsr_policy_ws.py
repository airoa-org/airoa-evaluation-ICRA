#!/usr/bin/env python3
"""Serve a policy as a WebSocket server for the HSR client.

Supports two backends:
  --backend openpi   (default) OpenPI framework (JAX/PyTorch, config-driven)
  --backend lerobot  LeRobot PI05Policy (merged checkpoint)
"""
import argparse
import logging
import os
from pathlib import Path

from runtime_core.websocket_policy_server import WebsocketPolicyServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve policy as WebSocket server for HSR client")
    parser.add_argument("--backend", choices=["openpi", "lerobot"], default="openpi", help="Policy backend")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--config-name", default=None, help="Train config name (required for openpi backend)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--default-prompt", default=None, help="Fallback prompt if prompt key is missing")
    parser.add_argument("--record-dir", default=None, help="Optional directory for policy records")
    parser.add_argument(
        "--pytorch-device",
        default=None,
        help='Torch device override (e.g. "cuda", "cuda:0", "cpu")',
    )
    return parser.parse_args()


def _create_openpi_policy(args):
    """Create policy using the OpenPI framework."""
    from openpi.policies import policy as policy_lib
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    if not args.config_name:
        raise ValueError("--config-name is required for openpi backend")

    config = train_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )

    if args.record_dir:
        policy = policy_lib.PolicyRecorder(policy, args.record_dir)

    return policy


def _create_lerobot_policy(args):
    """Create policy using LeRobot PI05Policy."""
    from lerobot_hsr_policy import LeRobotHSRPolicy

    device = args.pytorch_device or "cuda"
    return LeRobotHSRPolicy(
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        default_prompt=args.default_prompt,
    )


def main() -> None:
    args = parse_args()

    checkpoint_dir = str(Path(args.checkpoint_dir).expanduser())
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")
    args.checkpoint_dir = checkpoint_dir

    if args.backend == "lerobot":
        policy = _create_lerobot_policy(args)
    else:
        policy = _create_openpi_policy(args)

    metadata = dict(policy.metadata)
    metadata.update(
        {
            "backend": args.backend,
            "config_name": args.config_name or "",
            "checkpoint_dir": checkpoint_dir,
            "server_host": args.host,
            "server_port": args.port,
        }
    )

    logging.info(
        "Serving policy backend=%s checkpoint=%s on %s:%s",
        args.backend, checkpoint_dir, args.host, args.port,
    )
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
