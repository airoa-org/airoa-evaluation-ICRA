#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path

from openpi.policies import policy as policy_lib
from openpi.policies import policy_config
from openpi.training import config as train_config
from runtime_core.websocket_policy_server import WebsocketPolicyServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve OpenPI policy as websocket server for HSR client")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--config-name", required=True, help="Train config name (e.g. pi05_hsr)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--default-prompt", default=None, help="Fallback prompt if prompt key is missing")
    parser.add_argument("--record-dir", default=None, help="Optional directory for policy records")
    parser.add_argument(
        "--pytorch-device",
        default=None,
        help='Optional torch device override (e.g. "cuda", "cuda:0", "cpu")',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_dir = str(Path(args.checkpoint_dir).expanduser())
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")

    config_name = args.config_name
    config = train_config.get_config(config_name)

    policy = policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )

    if args.record_dir:
        policy = policy_lib.PolicyRecorder(policy, args.record_dir)

    metadata = dict(policy.metadata)
    data_cfg = getattr(config, "data", None)
    metadata.update(
        {
            "config_name": config_name,
            "checkpoint_dir": checkpoint_dir,
            "server_host": args.host,
            "server_port": args.port,
            "action_mode": getattr(data_cfg, "action_mode", None),
            "convert_gripper": getattr(data_cfg, "convert_gripper", None),
            "adapt_to_pi": getattr(data_cfg, "adapt_to_pi", None),
            "base_action_dim": getattr(data_cfg, "base_action_dim", None),
        }
    )

    logging.info(
        "Serving policy config=%s action_mode=%s checkpoint=%s on %s:%s",
        config_name,
        metadata.get("action_mode"),
        checkpoint_dir,
        args.host,
        args.port,
    )
    # NOTE: Keep the OpenPI implementation as needed, but do not change the next two lines.
    # They are the fixed websocket serving contract for the HSR client runtime.
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
