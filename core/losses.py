import torch
import torch.nn as nn

class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 推荐替代 MSE 的损失
        self.loss_fn = nn.SmoothL1Loss()  # 更鲁棒
        # 你也可以保留参数控制，例如 args['base']['loss_type']

    def forward(self, out, label, mask=None):
        preds = out['sentiment_preds']
        targets = label['sentiment_labels']
        loss = self.loss_fn(preds, targets)

        return {'loss': loss, 'l_sp': loss}
