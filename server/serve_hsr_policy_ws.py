#!/usr/bin/env python3
"""Serve a policy as a WebSocket server for the HSR client.

Supports two backends:
  --backend openpi   (default) OpenPI framework (JAX/PyTorch, config-driven)
  --backend lerobot  LeRobot PI05Policy (merged checkpoint)

Supports two modes:
  --mode e2e          (default) End-to-end inference
  --mode hierarchical HVLA: PA decomposition + PA-level inference
"""
import argparse
import json
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
    # HVLA
    parser.add_argument("--mode", choices=["e2e", "hierarchical"], default="e2e",
                        help="Inference mode: e2e or hierarchical (HVLA)")
    parser.add_argument("--pa-decomposition", default="/workspace/pa_decomposition.json",
                        help="PA decomposition JSON file")
    parser.add_argument("--policy-config", default="/workspace/hierarchical_config.yaml",
                        help="HVLA config YAML file")
    parser.add_argument("--fm-model", default=None, help="FM RF model path (joblib)")
    parser.add_argument("--fm-scaler", default=None, help="FM StandardScaler path (joblib)")
    parser.add_argument("--llm-api-host", default="localhost", help="LLM API server host")
    parser.add_argument("--llm-api-port", type=int, default=8001, help="LLM API server port")
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


def _wrap_with_hvla(base_policy, args):
    """Wrap base policy with HVLA controller."""
    import yaml
    from hierarchical_hsr_policy import HierarchicalHSRPolicy

    # PA decomposition map
    with open(args.pa_decomposition) as f:
        pa_map = json.load(f)
    logging.info("PA map loaded: %d SHTs from %s", len(pa_map), args.pa_decomposition)

    # Policy config
    policy_cfg = {}
    if os.path.exists(args.policy_config):
        with open(args.policy_config) as f:
            policy_cfg = yaml.safe_load(f) or {}
        logging.info("Policy config loaded: %s", args.policy_config)
    else:
        logging.warning("Policy config not found: %s (using defaults)", args.policy_config)

    # FM model
    fm_model = None
    fm_scaler = None
    if args.fm_model and os.path.exists(args.fm_model):
        try:
            import joblib
            fm_model = joblib.load(args.fm_model)
            if args.fm_scaler and os.path.exists(args.fm_scaler):
                fm_scaler = joblib.load(args.fm_scaler)
            logging.info("FM loaded: %s", args.fm_model)
        except Exception as e:
            logging.warning("FM load failed: %s", e)

    # LLM API クライアント（別プロセスの Qwen3.5-4B）
    llm_client = None
    try:
        from llm_api_client import LLMAPIClient
        llm_client = LLMAPIClient(
            host=args.llm_api_host,
            port=args.llm_api_port,
            pa_map=pa_map,
        )
        if llm_client.is_available():
            logging.info("LLM API client connected: %s:%d", args.llm_api_host, args.llm_api_port)
        else:
            logging.warning("LLM API server not available at %s:%d (PA マップ + E2E フォールバックで動作)",
                          args.llm_api_host, args.llm_api_port)
    except ImportError:
        logging.warning("LLM API client not available (llm_api_client.py not found)")

    hvla_policy = HierarchicalHSRPolicy(
        base_policy=base_policy,
        pa_map=pa_map,
        config=policy_cfg,
        fm_model=fm_model,
        fm_scaler=fm_scaler,
        llm_api_client=llm_client,
    )
    logging.info("HVLA mode: %d SHT, FM=%s, LLM=%s, Retry=enabled",
                 len(pa_map),
                 "enabled" if fm_model else "disabled",
                 "API" if (llm_client and llm_client.is_available()) else "disabled")
    return hvla_policy


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

    # HVLA ラップ
    if args.mode == "hierarchical":
        policy = _wrap_with_hvla(policy, args)

    metadata = dict(policy.metadata) if hasattr(policy, "metadata") else {}
    metadata.update(
        {
            "backend": args.backend,
            "mode": args.mode,
            "config_name": args.config_name or "",
            "checkpoint_dir": checkpoint_dir,
            "server_host": args.host,
            "server_port": args.port,
        }
    )

    logging.info(
        "Serving policy backend=%s mode=%s checkpoint=%s on %s:%s",
        args.backend, args.mode, checkpoint_dir, args.host, args.port,
    )
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
