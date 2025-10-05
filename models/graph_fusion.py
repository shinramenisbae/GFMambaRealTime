from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
import torch.nn.init as init
import numpy as np
from core.utils import RMSNorm


class graph_fusion(nn.Module):
    """Graph-style fusion over unimodal/bimodal/trimodal features.
    Input tensors t, v, a are [B, D]. Returns logits [B, out] and attention weights [B, 8].
    """
    def __init__(self, in_size, output_dim, hidden=50, dropout=0.5, fusion_hidden=16):
        super(graph_fusion, self).__init__()
        self.norm1 = RMSNorm(in_size * 3)
        self.drop = nn.Dropout(p=dropout)

        self.graph_fusion = nn.Sequential(
            nn.Linear(in_size * 2, fusion_hidden),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.Linear(fusion_hidden, in_size),
            nn.SiLU(),
        )

        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size * 3, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)
        self.hidden = hidden
        self.in_size = in_size

    def forward(self, t, v, a):
        B = t.size(0)
        D = self.in_size
        a1, v1, l1 = a, v, t  # naming consistent with original

        # Unimodal attention weights [B,1] each
        sa = F.silu(self.attention(a1))
        sv = F.silu(self.attention(v1))
        sl = F.silu(self.attention(l1))
        norm_uni = torch.cat([sa, sv, sl], dim=1)                 # [B,3]
        norm_uni = F.softmax(norm_uni, dim=1)
        sa_w = norm_uni[:, 0:1]
        sv_w = norm_uni[:, 1:1+1]
        sl_w = norm_uni[:, 2:3]
        total_weights = norm_uni                                   # [B,3]

        # Weighted unimodal fusion [B, D]
        unimodal = sa_w.expand(B, D) * a1 + sv_w.expand(B, D) * v1 + sl_w.expand(B, D) * l1

        # Bimodal interactions
        a_soft = F.softmax(a1, dim=1).unsqueeze(2)                 # [B, D, 1]
        v_soft = F.softmax(v1, dim=1).unsqueeze(2)                 # [B, D, 1]
        l_soft = F.softmax(l1, dim=1).unsqueeze(1)                 # [B, 1, D]

        mat_la = torch.bmm(l_soft, a_soft).view(B, 1)              # [B,1]
        mat_lv = torch.bmm(l_soft, v_soft).view(B, 1)              # [B,1]
        sal = (1.0 / (mat_la + 0.5)) * (sl_w + sa_w)               # [B,1]
        svl = (1.0 / (mat_lv + 0.5)) * (sl_w + sv_w)               # [B,1]
        norm_bi = torch.cat([sal, svl], dim=1)                     # [B,2]
        norm_bi = F.softmax(norm_bi, dim=1)
        total_weights = torch.cat([total_weights, norm_bi], dim=1) # [B,5]

        a_l = norm_bi[:, 0:1].expand(B, D) * self.graph_fusion(torch.cat([a1, l1], dim=1))
        v_l = norm_bi[:, 1:2].expand(B, D) * self.graph_fusion(torch.cat([v1, l1], dim=1))
        bimodal = a_l + v_l                                        # [B, D]

        # Trimodal interactions
        a_l2 = F.softmax(a_l, dim=1).unsqueeze(2)                  # [B, D, 1]
        v_l2 = F.softmax(v_l, dim=1).unsqueeze(2)                  # [B, D, 1]
        savll = (1.0 / (torch.bmm(a_l2.transpose(1, 2), v_l2).view(B, 1) + 0.5)) * (sal + svl)
        salv = (1.0 / (torch.bmm(a_l2.transpose(1, 2), v_soft).view(B, 1) + 0.5)) * (sal + sv)
        svla = (1.0 / (torch.bmm(v_l2.transpose(1, 2), a_soft).view(B, 1) + 0.5)) * (sa + svl)

        norm_tri = torch.cat([savll, salv, svla], dim=1)           # [B,3]
        norm_tri = F.softmax(norm_tri, dim=1)
        total_weights = torch.cat([total_weights, norm_tri], dim=1)  # [B,8]

        avll = norm_tri[:, 0:1].expand(B, D) * self.graph_fusion(torch.cat([v_l, a_l], dim=1))
        alv  = norm_tri[:, 1:2].expand(B, D) * self.graph_fusion(torch.cat([a_l, v1], dim=1))
        vla  = norm_tri[:, 2:3].expand(B, D) * self.graph_fusion(torch.cat([v_l, a1], dim=1))
        trimodal = avll + alv + vla                                 # [B, D]

        fusion = torch.cat([unimodal, bimodal, trimodal], dim=1)    # [B, 3D]
        fusion = self.norm1(fusion)
        fusion = self.drop(fusion)

        y_1 = F.silu(self.linear_1(fusion))
        y_2 = F.silu(self.linear_2(y_1))
        y_3 = self.linear_3(y_2)
        return y_3, total_weights
