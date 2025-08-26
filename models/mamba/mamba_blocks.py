'''
Copied and modified from
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
'''

import math
import torch
import torch.nn as nn

from functools import partial

from mamba_ssm import Mamba
from models.mamba.bimamba import Mamba as BiMamba 
from models.mamba.bimamba import Block as PreNormBlock

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_block( #创建一个mamba块
    d_model,
    ssm_cls=None, #双向or单向Mamba
    ssm_cfg=None,#传给 ssm_cls 的配置字典（超参）
    norm_epsilon=1e-5, #传给 ssm_cls 的配置字典（超参）
    rms_norm=False, #是否使用 RMSNorm
    residual_in_fp32=False,
    fused_add_norm=True,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    assert callable(ssm_cls), "ssm_cls must be callable"
    mixer_cls = partial(ssm_cls, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs  # type: ignore[attr-defined]
    )
    block = PreNormBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,# type: ignore[attr-defined]
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx# type: ignore[attr-defined]
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights( #初始化权重
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class LnMambaAdd(nn.Module): # LayerNorm + Mamba Add block

    def __init__(self, 
        d_model, 
        ssm_cls, 
        ssm_cfg,
        rms_norm=False,
        layer_idx=None
    ):
        super().__init__()
        if rms_norm:
            self.norm = RMSNorm(d_model)# type: ignore[attr-defined]
        else:
            self.norm = nn.LayerNorm(d_model)
        self.mamba = ssm_cls(d_model=d_model, **ssm_cfg)

        print(type(self.mamba))

        print('Created LnMambaAdd.')

    def forward(self, x, residual=None, inference_params=None):
        if residual != None:
            x = x + residual
        return self.mamba(self.norm(x)), x


class MambaBlocksSequential(nn.Module): #构建一组mamba块序列
    """
    A wrapper for the Mamba block to replicate it

    Arguments
    ---------
    n_mamba : int
        Number of Mamba blocks
    d_model : int
        Input dimension to Mamba (bottleneck dimension).
    d_state : int
        Mamba state dimension
    expand: int
        First linear projection d_model -> d_model * expand
    d_conv: int
        Kernel size of Mamba conv
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    ---------
    """

    def __init__(self,  #mamba一些配置
        n_mamba: int,
        bidirectional: bool,
        d_model: int, # bottleneck dimension (B)
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4, # kernel_size of 'Conv' in Mamba
        dt_rank: str="auto",
        conv_bias: bool = True,
        bias: bool = False,
        fused_add_norm: bool = True,
        rms_norm: bool = False,
        norm_epsilon: float = 1e-5,
        initializer_cfg=None,
        residual_in_fp32=False,
        use_simple_block=False
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.bidirectional = bidirectional

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.use_simple_block = use_simple_block

        ssm_cfg = {
            "d_state": d_state,
            "expand": expand,
            "d_conv": d_conv,
            "dt_rank": dt_rank,
            "conv_bias": conv_bias,
            "bias": bias
        }
        if bidirectional:
            ssm_cfg["bimamba_type"] = "v2"

        if use_simple_block: #构建mamba块
            self.layers = nn.Sequential( #简单mamba序列
                *[
                    LnMambaAdd(
                        d_model=d_model,
                        ssm_cls=BiMamba if bidirectional else Mamba,
                        ssm_cfg=ssm_cfg,
                        rms_norm=rms_norm,
                        layer_idx=i
                    )
                    for i in range(n_mamba)
                ]
            )
        else:
            self.layers = nn.Sequential( #标准mamba序列
                *[
                    create_block(
                        d_model=d_model,
                        ssm_cls=BiMamba if bidirectional else Mamba,
                        ssm_cfg=ssm_cfg,
                        norm_epsilon=norm_epsilon,
                        rms_norm=rms_norm,
                        residual_in_fp32=residual_in_fp32,
                        fused_add_norm=fused_add_norm,
                        layer_idx=i,
                    )
                    for i in range(n_mamba)
                ]
            )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(# type: ignore[attr-defined]
            d_model, eps=norm_epsilon
        )

        self.apply( #应用初始权重
            partial(
                _init_weights,
                n_layer=n_mamba,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs): #分配推理缓存
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs) #block.allocate_inference_cache变为layer.allocate_inference_cache
            for i, layer in enumerate(self.layers)
        }
    
    def forward(self, x, inference_params=None): #

        hidden_states = x
        residual = None #初始状态无残差
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer( #前向传递
                hidden_states, residual, inference_params=inference_params
            )
            
        if not self.fused_add_norm: #无融合层归一化慢速
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else: #快速路径
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn# type: ignore[attr-defined]

            hidden_states = fused_add_norm_fn(# type: ignore[attr-defined]
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states
        