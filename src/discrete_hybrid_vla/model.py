import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class HSRChunkPolicy(nn.Module):
    """
    Transformer chunk policy: state (8-dim) -> action.relative (T, 11).
    Trained on 500 episodes task6911, 121,843 samples, 30 epochs, loss=0.0047.
    """
    def __init__(self, state_dim=8, action_dim=11, action_horizon=16,
                 d_model=256, n_heads=8, n_layers=4, n_tasks=20):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim     = action_dim
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, d_model), nn.GELU(), nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),   nn.GELU(), nn.LayerNorm(d_model),
        )
        self.task_emb       = nn.Embedding(n_tasks + 1, d_model)
        self.action_queries = nn.Parameter(torch.randn(action_horizon, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=0.0,
            batch_first=True)
        self.decoder     = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, action_dim)
        )

    def forward(self, state, task):
        B      = state.shape[0]
        memory = torch.cat([
            self.state_enc(state).unsqueeze(1),
            self.task_emb(task.clamp(0, 19)).unsqueeze(1)
        ], dim=1)
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        return self.action_head(self.decoder(queries, memory))


class DiscreteHybridVLA:
    def __init__(self, checkpoint_path: str = None, device: str = "cuda"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = HSRChunkPolicy().to(self.device)
        if checkpoint_path:
            p = Path(checkpoint_path)
            if p.is_dir():
                p = p / "model.pt"
            if p.exists():
                sd    = torch.load(p, map_location=self.device)
                state = sd.get("model_state_dict", sd)
                self.model.load_state_dict(state, strict=True)
                print(f"[Model] Loaded from {p}")
            else:
                print(f"[Model] WARNING: not found at {p}, random init")
        self.model.eval()
        print(f"[Model] Ready on {self.device}")

    @classmethod
    def load(cls, checkpoint_path: str):
        return cls(checkpoint_path)

    def infer(self, head_rgb, hand_rgb, state, prompt, T=16):
        with torch.no_grad():
            st  = torch.tensor(state[:8], dtype=torch.float32).unsqueeze(0).to(self.device)
            tsk = torch.zeros(1, dtype=torch.long).to(self.device)
            out = self.model(st, tsk).squeeze(0).cpu().numpy()
        return np.nan_to_num(out[:T], nan=0.0, posinf=0.0, neginf=0.0)
