import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gl_feature import ModalityProjector, ContextExtractor
from models.enhance import TGMamba
from models.Intramodel import IntraModalEnhancer
from models.graph_fusion import graph_fusion

class GFMamba(nn.Module):
    def __init__(self, args):
        super().__init__()

        text_dim, video_dim, audio_dim = args['model']['input_dim']
        self.proj_text= ModalityProjector(text_dim, args['model']['dim'])
        self.proj_video= ModalityProjector(video_dim, args['model']['dim'])
        self.proj_audio= ModalityProjector(audio_dim, args['model']['dim'])

        self.ctx_text = ContextExtractor(args['model']['dim'],args['model']['ContextExtractor']['conv_kernel_size'],
                                          args['model']['ContextExtractor']['hidden_ratio'],args['model']['ContextExtractor']['dropout'])
        self.ctx_video = ContextExtractor(args['model']['dim'],args['model']['ContextExtractor']['conv_kernel_size'],
                                           args['model']['ContextExtractor']['hidden_ratio'],args['model']['ContextExtractor']['dropout'])
        self.ctx_audio = ContextExtractor(args['model']['dim'],args['model']['ContextExtractor']['conv_kernel_size'],
                                          args['model']['ContextExtractor']['hidden_ratio'],args['model']['ContextExtractor']['dropout'])

        self.tgmamba = TGMamba(args['model']['TGMamba']['num_layers'],args['model']['dim'], args['model']['TGMamba']['dropout'],
                               args['model']['TGMamba']['causal'], args['model']['TGMamba']['mamba_config'])

        self.intra_text = IntraModalEnhancer(args['model']['dim'], dropout=args['model']['IntraModalEnhancer']['dropout'])
        self.intra_video = IntraModalEnhancer(args['model']['dim'], dropout=args['model']['IntraModalEnhancer']['dropout'])
        self.intra_audio = IntraModalEnhancer(args['model']['dim'], dropout=args['model']['IntraModalEnhancer']['dropout'])

        self.graph_fusion = graph_fusion(args['model']['dim'], args['model']['graph_fusion']['num_classes'],
                                         args['model']['graph_fusion']['hidden'], args['model']['graph_fusion']['dropout'],
                                        )
        
        
    def forward(self, text_x, video_x, audio_x):
        """
        输入:
            text_x: [B, T_text, D_text]
            video_x: [B, T_video, D_video]
            audio_x: [B, T_audio, D_audio]
        输出:
            logits: 分类预测 [B, num_classes]
            att_weights: 融合权重解释
            (t, v, a): 增强后的三模态 token 表示
        """
        # 模态投影
        t = self.proj_text(text_x)  # [B, T_text, D_text]
        v = self.proj_video(video_x)  # [B, T_video, D_video]
        a = self.proj_audio(audio_x)  # [B, T_audio, D_audio]

        t_ctx = self.ctx_text(t)  # [B, T_text, D_text]
        v_ctx = self.ctx_video(v)  # [B, T_video, D_video]
        a_ctx = self.ctx_audio(a)  # [B, T_audio, D_audio]

        # TGMamba 融合
        a_fused, v_fused, t_fused = self.tgmamba(a_ctx, v_ctx, t_ctx)

        t_feat = torch.mean(t_fused,dim=1) # [B, D]池化操作需要提升下,原文是平均池化，先池化后注意力
        v_feat = torch.mean(v_fused,dim=1) # [B, D]
        a_feat = torch.mean(a_fused,dim=1) # [B, D]

        # 模态内增强
        t_enhanced = self.intra_text(t_feat)  # [B, T_text, D_text]
        v_enhanced = self.intra_video(v_feat)  # [B, T_video, D_video]
        a_enhanced = self.intra_audio(a_feat)  # [B, T_audio, D_audio]

        # 图融合
        logits, att_weights = self.graph_fusion(t_enhanced, v_enhanced, a_enhanced)


        return {
            'sentiment_preds': logits,
            'att_weights': att_weights,
            'text_pred': t_enhanced,   # 假设 t_enhanced shape 为 [B, D]，可加一层线性映射到输出
            'video_pred': v_enhanced,
            'audio_pred': a_enhanced,
        }

            
        