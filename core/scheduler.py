import torch
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer. """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

def get_scheduler(optimizer, args):
    # CosineAnnealingWarmRestarts 配合 warmup
    scheduler_cosine = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(2, int(0.2 * args['base']['n_epochs'])),  # 首次重启周期
        T_mult=2,
        eta_min=1e-6
    )

    scheduler_warmup = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=int(0.1 * args['base']['n_epochs']),
        after_scheduler=scheduler_cosine
    )

    return scheduler_warmup