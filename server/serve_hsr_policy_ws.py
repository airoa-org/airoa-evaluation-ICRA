import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from runtime_core.websocket_policy_server import WebsocketPolicyServer
from discrete_hybrid_vla.adapter import HSRAdapter

# Fix 3: use POLICY_CHECKPOINT_DIR (matches docker-compose.yml and entrypoint.sh)
CHECKPOINT_DIR = os.environ.get("POLICY_CHECKPOINT_DIR", "/policy_checkpoint")
print(f"[Server] Loading checkpoint from directory: {CHECKPOINT_DIR}")

# Fix 4: load specific file from directory
import os
checkpoint_file = os.path.join(CHECKPOINT_DIR, "model.pt")
print(f"[Server] Checkpoint file: {checkpoint_file}")

adapter = HSRAdapter(checkpoint_path=checkpoint_file)
print("[Server] Ready.")

if __name__ == "__main__":
    server = WebsocketPolicyServer(policy=adapter)
    server.serve()
