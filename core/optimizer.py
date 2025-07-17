import torch

def get_optimizer(model, opt):
    # 可扩展到更多 backbone
    param_groups = []
    if hasattr(model, 'visual_backbone'):
        param_groups.append({'params': filter(lambda p: p.requires_grad, model.visual_backbone.parameters()), 'lr': 0.1 * opt.lr})
    if hasattr(model, 'audio_backbone'):
        param_groups.append({'params': filter(lambda p: p.requires_grad, model.audio_backbone.parameters()), 'lr': 0.1 * opt.lr})
    # 其它参数
    backbone_ids = []
    if hasattr(model, 'visual_backbone'):
        backbone_ids += list(map(id, model.visual_backbone.parameters()))
    if hasattr(model, 'audio_backbone'):
        backbone_ids += list(map(id, model.audio_backbone.parameters()))
    params_fusion = filter(lambda p: id(p) not in backbone_ids and p.requires_grad, model.parameters())
    param_groups.append({'params': params_fusion, 'lr': opt.lr})

    optimizer = torch.optim.Adam(param_groups, lr=opt.lr, weight_decay=opt.weight_decay)
    return optimizer