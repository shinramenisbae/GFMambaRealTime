import torch
from torch import nn
import torch.nn.functional as F
from core.utils import RMSNorm

class ModalityProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)



class ContextExtractor(nn.Module):
    def __init__(self, dim, conv_kernel_size=3, hidden_ratio=4, dropout=0.4):
        super(ContextExtractor, self).__init__()
        hidden_dim = dim * hidden_ratio

        self.pre_norm = RMSNorm(dim) #前置归一化
        self.post_norm = RMSNorm(dim) #后置归一化

        # 全局路径 - MLP（Feedforward）
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

        # 局部路径 - Conv1D（维持输入输出维度）
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_raw = x
        # x: [B, T, D]
        x= self.pre_norm(x) # 前置归一化
        # MLP 分支（全局）
        mlp_out = self.mlp(x)  # [B, T, D]

        # Conv 分支（局部）
        # 先变换成 [B, D, T] 以应用 Conv1D
        conv_input = x.transpose(1, 2)  # [B, D, T]
        conv_out = self.conv(conv_input).transpose(1, 2)  # [B, T, D]
        
        # 残差融合
        fusion = mlp_out + conv_out
        fusion = self.post_norm(fusion)
        out = x_raw + fusion

        return out


