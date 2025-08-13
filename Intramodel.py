import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import RMSNorm

class IntraModalEnhancer(nn.Module):
    def __init__(self, feature_dim, dropout=0.5):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim, bias=True)
        self.key = nn.Linear(feature_dim, feature_dim, bias=True)
        self.value = nn.Linear(feature_dim, feature_dim, bias=True)

        self.scale = nn.Parameter(torch.tensor(1.0))  # 可学习 scale
        self.norm = RMSNorm(feature_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, D]
        residual = x
        Q = self.query(x).unsqueeze(2)
        K = self.key(x).unsqueeze(1)
        V = self.value(x).unsqueeze(2)

        attn_scores = torch.bmm(Q, K) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.bmm(attn_weights, V).squeeze(2)
        attended = self.drop(attended)

        out = self.norm(attended + residual)
        return out

