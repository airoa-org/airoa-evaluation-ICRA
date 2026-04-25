
import asyncio, json, os, sys
import numpy as np
import websockets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from discrete_hybrid_vla.adapter import HSRAdapter

CHECKPOINT_PATH = os.environ.get("POLICY_CHECKPOINT_PATH", "./checkpoint")
print(f"[Server] Loading from: {CHECKPOINT_PATH}")
adapter = HSRAdapter(checkpoint_path=CHECKPOINT_PATH)
print("[Server] Ready.")

async def handle_client(websocket):
    print(f"[Server] Client connected")
    async for raw in websocket:
        try:
            data     = json.loads(raw)
            head_rgb = np.array(data["head_rgb"], dtype=np.uint8)
            hand_rgb = np.array(data["hand_rgb"], dtype=np.uint8)
            state    = np.array(data["state"],    dtype=np.float32)
            prompt   = str(data["prompt"])
            actions  = adapter(head_rgb, hand_rgb, state, prompt)
            await websocket.send(json.dumps({"actions": actions.tolist()}))
            print(f"[Server] Action executed. shape={actions.shape}")
        except Exception as e:
            print(f"[Server] Error: {e}")
            fallback = np.zeros((1, 11), dtype=np.float32)
            await websocket.send(json.dumps({"actions": fallback.tolist()}))

async def main():
    port = int(os.environ.get("POLICY_SERVER_PORT", 8765))
    print(f"[Server] Listening on ws://0.0.0.0:{port}")
    async with websockets.serve(handle_client, "0.0.0.0", port):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
