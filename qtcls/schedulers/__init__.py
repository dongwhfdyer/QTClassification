# Copyright (c) QIU, Tian. All rights reserved.

from torch.optim.lr_scheduler import *

from .cosine import CosineLR


def build_scheduler(args, optimizer):
    scheduler_name = args.scheduler.lower()

    if scheduler_name in ['cosine', 'default']:
        return CosineLR(optimizer, args.epochs, args.lrf)

    if scheduler_name == 'step':
        return StepLR(optimizer, args.step_size, args.gamma)

    if scheduler_name == 'multistep':
        return MultiStepLR(optimizer, args.milestones, args.gamma)

    if scheduler_name == 'warmup...':
        raise NotImplementedError('coming soon ...')

    raise ValueError(f"scheduler '{scheduler_name}' is not found.")
