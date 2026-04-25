import numpy as np
from .model import DiscreteHybridVLA

class HSRAdapter:
    """
    Adapter for DiscreteHybridVLA — bridges WebSocket contract to our model.
    WebsocketPolicyServer calls: policy.infer(obs: dict) -> dict
    obs keys: head_rgb, hand_rgb, state, prompt
    returns:  {"actions": np.ndarray shape (T, 11)}
    """
    def __init__(self, checkpoint_path: str):
        self.model = DiscreteHybridVLA.load(checkpoint_path)

    def infer(self, obs: dict) -> dict:
        """Called by WebsocketPolicyServer for every timestep."""
        head_rgb = np.array(obs["head_rgb"], dtype=np.uint8)
        hand_rgb = np.array(obs["hand_rgb"], dtype=np.uint8)
        state    = np.array(obs["state"],    dtype=np.float32)
        prompt   = str(obs.get("prompt", "relocate object"))

        actions  = self.model.infer(head_rgb, hand_rgb, state, prompt)
        assert actions.shape[1] == 11, f"Expected (T,11) got {actions.shape}"
        assert np.all(np.isfinite(actions)), "Non-finite actions detected"
        return {"actions": actions}
