from _bisect import bisect_right

from torch.optim import lr_scheduler as lr_scheduler

__all__ = ['WarmupMultiStepLR']


class WarmupMultiStepLR(lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones: list,
            gamma=0.1,
            warmup_factor=0.01,
            warmup_iters=10,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted",
                " but got {}".format(warmup_method)
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        return [base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]
