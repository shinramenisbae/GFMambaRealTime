import torch
import torch.nn as nn
from core.utils import RMSNorm

# Lightweight fallback of TGMamba without external kernels.
# It normalizes each modality and computes a simple fused text stream as the mean of (t,a,v).
class TGMamba(nn.Module):
    def __init__(self, num_layers, d_model, dropout=0.1, causal=False, mamba_config=None):
        super().__init__()
        self.a_norm = RMSNorm(d_model)
        self.v_norm = RMSNorm(d_model)
        self.t_norm = RMSNorm(d_model)
        self.text_fuse_norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, a_x, v_x, t_x, a_inference_params=None, v_inference_params=None, t_inference_params=None):
        # a_x, v_x, t_x: [B, T, D]
        a_out = self.a_norm(a_x)
        v_out = self.v_norm(v_x)
        # Fuse text with averaged cues from a and v (very lightweight stand-in for Mamba)
        t_mix = (t_x + a_x + v_x) / 3.0
        t_out = self.text_fuse_norm(t_mix)
        return a_out, v_out, t_out
