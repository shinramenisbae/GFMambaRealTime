import warnings
from dataclasses import dataclass
from typing import List, Optional
import copy
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
# Mamba
from mamba_ssm import Mamba
from models.mamba.bimamba import Mamba as BiMamba
from models.mamba.mm_bimamba import Mamba as MMBiMamba
from core.utils import RMSNorm

class MMMambaEncoderLayer(nn.Module): #可以加ffn和激活函数
    def __init__(
            self,
            d_model,
            dropout=0.5,
            causal=False,
            mamba_config=None
    ):
        super().__init__()
        assert mamba_config is not None

        bidirectional = mamba_config.pop('bidirectional')

        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = MMBiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )

        mamba_config['bidirectional'] = bidirectional

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
            self,
            m_x, n_x,
            m_inference_params=None,
            n_inference_params=None
    ):
        m_out1, n_out1 = self.mamba(m_x, n_x, m_inference_params, n_inference_params)

        m_out = m_x + self.drop(self.norm1(m_out1))
        n_out = n_x + self.drop(self.norm2(n_out1))

        return m_out, n_out



class TGMamba(nn.Module): #执行以文本为主导的双模态交互


    def __init__(
            self,
            num_layers,
            d_model,
            dropout=0.1,
            causal=False,
            mamba_config=None
    ):
        super().__init__()

        at_mamba_list = [] #文本mamba层
        vt_mamba_list = []
        self.a_norm = RMSNorm(d_model)
        self.v_norm = RMSNorm(d_model)
        self.t_norm = RMSNorm(d_model)
        self.text_fuse_norm=RMSNorm(d_model)
        for i in range(num_layers): #每层都构建一对交互模块
            at_mamba_list.append(MMMambaEncoderLayer( #
                d_model=d_model,
                dropout=dropout,
                causal=causal,
                mamba_config=copy.deepcopy(mamba_config),
            ))
            vt_mamba_list.append(MMMambaEncoderLayer(
                d_model=d_model,
                dropout=dropout,
                causal=causal,
                mamba_config=copy.deepcopy(mamba_config),
            ))

        self.at_mamba_layers = torch.nn.ModuleList(at_mamba_list)
        self.vt_mamba_layers = torch.nn.ModuleList(vt_mamba_list)
        


    def forward(
            self,
            a_x, v_x, t_x,
            a_inference_params=None,
            v_inference_params=None,
            t_inference_params=None
    ):

        a_out = self.a_norm(a_x)
        v_out = self.v_norm(v_x)
        t_out = self.t_norm(t_x)

        for at_mamba_layer, vt_mamba_layer in zip(self.at_mamba_layers, self.vt_mamba_layers):
            a_out, t_out_at = at_mamba_layer( #音频输出+文本输出
                a_out, t_out,
                a_inference_params,
                t_inference_params
            )
            v_out, t_out_vt = vt_mamba_layer(
                v_out, t_out,
                v_inference_params,
                t_inference_params
            )
            t_out = self.text_fuse_norm((t_out_at + t_out_vt) / 2)


            
        return a_out, v_out,t_out #增强后的输出
