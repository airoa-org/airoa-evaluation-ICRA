#!/usr/bin/env python3

import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from discrete_hybrid_vla.adapter import HSRAdapter
from runtime_core.websocket_policy_server import WebsocketPolicyServer


def main() -> None:
    checkpoint_dir = Path(os.environ.get("POLICY_CHECKPOINT_DIR", "/policy_checkpoint"))
    checkpoint_file = checkpoint_dir / "model.pt"
    host = os.environ.get("POLICY_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("POLICY_SERVER_PORT", "8000"))

    print(f"[Server] Loading checkpoint from directory: {checkpoint_dir}")
    print(f"[Server] Checkpoint file: {checkpoint_file}")

    adapter = HSRAdapter(checkpoint_path=str(checkpoint_file))
    print("[Server] Ready.")
    print(f"[Server] Serving on {host}:{port}")

    metadata = dict(getattr(adapter, "metadata", {}))
    metadata.update(
        {
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_file": str(checkpoint_file),
            "server_host": host,
            "server_port": port,
        }
    )
    server = WebsocketPolicyServer(policy=adapter, host=host, port=port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
