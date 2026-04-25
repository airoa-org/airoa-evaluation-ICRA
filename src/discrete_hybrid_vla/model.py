
import torch
import torch.nn as nn
import numpy as np
import clip
from pathlib import Path
from PIL import Image


class CLIPVLAPolicy(nn.Module):
    def __init__(self, state_dim=8, action_dim=11, action_horizon=10):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim     = action_dim
        self.state_encoder  = nn.Sequential(
            nn.Linear(state_dim, 128), nn.LayerNorm(128), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(1664, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),  nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, action_dim * action_horizon),
            nn.Tanh(),
        )

    def forward(self, head_feat, hand_feat, lang_feat, state):
        B = state.shape[0]
        return self.policy_head(
            torch.cat([head_feat, hand_feat, lang_feat,
                       self.state_encoder(state)], dim=-1)
        ).view(B, self.action_horizon, self.action_dim)


class DiscreteHybridVLA:
    def __init__(self, checkpoint_path: str = None, device: str = "cuda"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.model = CLIPVLAPolicy().to(self.device)
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[Model] Loading from {checkpoint_path}")
            sd = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(sd, strict=False)
        else:
            print("[Model] No checkpoint found — random init")

        self.model.eval()
        print(f"[Model] Ready on {self.device}")

    @classmethod
    def load(cls, checkpoint_path: str):
        return cls(checkpoint_path)

    def infer(self, head_rgb, hand_rgb, state, prompt, T=10):
        with torch.no_grad():
            head_t    = self.clip_preprocess(Image.fromarray(head_rgb)).unsqueeze(0).to(self.device)
            hand_t    = self.clip_preprocess(Image.fromarray(hand_rgb)).unsqueeze(0).to(self.device)
            head_feat = self.clip_model.encode_image(head_t).float()
            hand_feat = self.clip_model.encode_image(hand_t).float()
            tokens    = clip.tokenize([prompt], truncate=True).to(self.device)
            lang_feat = self.clip_model.encode_text(tokens).float()
            st        = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            actions   = self.model(head_feat, hand_feat, lang_feat, st)
        out = actions.squeeze(0).cpu().numpy().astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
