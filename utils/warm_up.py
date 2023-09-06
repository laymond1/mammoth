# Taken from Co2L Paper code: https://github.com/chaht01/Co2L

import math
import numpy as np


def adjust_learning_rate(args, optimizer, epoch):
    """
    Adjust the learning rate.
    :param args: arguments
    :param epoch: the current epoch
    :param optimizer: the optimizer
    """
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    """
    Set the learning rate within warmup epochs.
    :param args: arguments
    :param epoch: the current epoch
    :param batch_id: the current batch id
    :param total_batches: the total batch size
    :param optimizer: the optimizer
    """
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr