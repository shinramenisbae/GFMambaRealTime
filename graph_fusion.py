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

class graph_fusion(nn.Module): #top-k k<<n, 复杂度 nlogn

    def __init__(self, in_size, output_dim, hidden = 50, dropout=0.5,fusion_hidden=16):

        super(graph_fusion, self).__init__()       
        self.norm1 = RMSNorm(in_size*3) 
        self.drop = nn.Dropout(p=dropout) 

        self.graph_fusion = nn.Sequential(
        nn.Linear(in_size*2, fusion_hidden),
        nn.PReLU(num_parameters=1, init=0.1),
        nn.Linear(fusion_hidden, in_size),
        nn.SiLU(),
    )

        self.attention = nn.Linear(in_size, 1) #生成权重，使用线性层
        self.linear_1 = nn.Linear(in_size*3, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)
        self.hidden = hidden
        self.in_size = in_size

    def forward(self, t,v,a):
        a1 = a
        v1 = v 
        l1 = t #l1是文本模态，a1是音频模态，v1是视频模态
        ###################### unimodal layer  ########################## 
        sa = F.silu(self.attention(a1)) #对每个模态进行attention计算权重是标量
        sv = F.silu(self.attention(v1)) #原文是sigmoid，改为tahn 消融
        sl = F.silu(self.attention(l1))
        
        normalize=  torch.cat([sa,sv],1) #合并他们
        normalize = torch.cat([normalize,sl],1)
        normalize = F.softmax(normalize,1) #去除softmax效果更好？合并他们计算softmax得到每个部分的权重
        
        sa = normalize[:,0].unsqueeze(1) #取出归一化后的权重
        sv = normalize[:,1].unsqueeze(1)
        sl = normalize[:,2].unsqueeze(1)

        total_weights = torch.cat([sa, sv],1) #合并他们
        total_weights = torch.cat([total_weights, sl],1)

        unimodal_a = (sa.expand(a1.size(0),self.in_size))#扩展至全维度
        unimodal_v = (sv.expand(a1.size(0),self.in_size))
        unimodal_l = (sl.expand(a1.size(0),self.in_size))
        sa = sa.squeeze() #为sal做准备
        sl = sl.squeeze()
        sv = sv.squeeze()
        unimodal = unimodal_a * a1 + unimodal_v * v1 + unimodal_l * l1 #单模态权重

        ##################### bimodal layer ############################
               
        a = F.softmax(a1, 1).unsqueeze(2)  # [B, D, 1] 
        v = F.softmax(v1, 1).unsqueeze(2)  # [B, D, 1]
        l = F.softmax(l1, 1).unsqueeze(1)  # [B, 1, D]
                
        sal = (1/(torch.matmul(l,a).squeeze()+0.5)*(sl+sa)) #计算Smn
        svl = (1/(torch.matmul(l,v).squeeze()+0.5)*(sl+sv))
        
        normalize = torch.cat([sal.unsqueeze(1), svl.unsqueeze(1)],1)
        
        normalize = F.softmax(normalize,1)
        total_weights = torch.cat([total_weights,normalize],1)
        #激活函数可以替换
        a_l = (normalize[:,0].unsqueeze(1).expand(a.size(0), self.in_size)) * self.graph_fusion(torch.cat([a1,l1],1))
        v_l = (normalize[:,1].unsqueeze(1).expand(a.size(0), self.in_size)) * self.graph_fusion(torch.cat([v1,l1],1))
        bimodal = a_l + v_l
    
        ###################### trimodal layer ####################################
        a_l2 = F.softmax(a_l,1).unsqueeze(2) #[B, D, 1]
        v_l2 = F.softmax(v_l,1).unsqueeze(2)
        savll = (1/(torch.matmul(a_l2.squeeze().unsqueeze(1),v_l2).squeeze()+0.5)*(sal+svl))
        salv = (1/(torch.matmul(a_l2.squeeze().unsqueeze(1),v).squeeze()+0.5)*(sal+sv))
        svla = (1/(torch.matmul(v_l2.squeeze().unsqueeze(1),a).squeeze()+0.5)*(sa+svl))

        normalize2 = torch.cat([ savll.unsqueeze(1), salv.unsqueeze(1), svla.unsqueeze(1)],1)
        normalize2 = F.softmax(normalize2,1)
        total_weights = torch.cat([total_weights,normalize2],1)

        avll = (normalize2[:,0].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion(torch.cat([v_l,a_l],1))
        alv = (normalize2[:,1].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion(torch.cat([a_l,v1],1))
        vla = (normalize2[:,2].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion(torch.cat([v_l,a1],1))
        trimodal = avll + alv + vla


        fusion = torch.cat([unimodal,bimodal],1)
        fusion = torch.cat([fusion,trimodal],1)        
        fusion = self.norm1(fusion)
        fusion = self.drop(fusion)
      
        y_1 = F.silu(self.linear_1(fusion))   # (B, 3D) → (B, H)
        y_2 = F.silu(self.linear_2(y_1))      # (B, H)  → (B, H)
        y_3 = self.linear_3(y_2)              # (B, H)  → (B, out)
        return y_3, total_weights
      
      
