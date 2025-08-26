import torch
from models.Intramodel import IntraModalEnhancer

# 加载预训练权重路径
ckpt_path = 'path/to/intra_enhancer.pth'  # 修改为你的权重文件路径

# 假设特征维度
feature_dim = 128  # 修改为你的特征维度

# 构建模型并加载权重
model = IntraModalEnhancer(feature_dim)
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
model.eval()

# 示例输入
x = torch.randn(4, feature_dim)  # 4个样本，特征维度为feature_dim

# 推理
with torch.no_grad():
    enhanced = model(x)
print('增强后特征:', enhanced.shape)
print(enhanced)