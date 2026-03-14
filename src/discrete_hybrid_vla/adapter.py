
import numpy as np
from .model import DiscreteHybridVLA

class HSRAdapter:
    def __init__(self, checkpoint_path: str):
        self.model = DiscreteHybridVLA.load(checkpoint_path)

    def __call__(self, head_rgb, hand_rgb, state, prompt):
        actions = self.model.infer(head_rgb, hand_rgb, state, prompt)
        assert actions.shape[1] == 11
        assert np.all(np.isfinite(actions))
        return actions
