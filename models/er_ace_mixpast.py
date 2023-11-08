# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
from datasets import get_dataset

from models.utils.continual_model import ContinualModel, save_model
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # network
    parser.add_argument('--gamma', type=float, default=0.1)
    return parser


def MixUp(input, target, alpha=1.0):
    if alpha > 0:
        lambda_ = np.random.beta(alpha, alpha)
    else:
        lambda_ = 1
 
    batch_size = input.size(0)
    index = torch.randperm(batch_size)
    
    mixed_input = lambda_ * input + (1 - lambda_) * input[index, :]    
    labels_a, labels_b = target, target[index]
 
    return mixed_input, labels_a, labels_b, lambda_

def MixUpLoss(criterion, pred, labels_a, labels_b, lambda_):
    return lambda_ * criterion(pred, labels_a) + (1 - lambda_) * criterion(pred, labels_b)


class ErACEMixPast(ContinualModel):
    NAME = 'er_ace_mixpast'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ErACEMixPast, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0

    def process_mixup(self, mixed_input, labels_a, labels_b, lambda_):
        output = self.net(mixed_input)
        loss = MixUpLoss(self.loss, output, labels_a, labels_b, lambda_)
        
        return loss

    def observe(self, inputs, labels, not_aug_inputs):

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        loss_re = torch.tensor(0.)

        if self.task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            loss_re = self.loss(self.net(buf_inputs), buf_labels)
            
            # mixup loss
            index = ~torch.isin(buf_labels, present) # self.inc_classes 제거
            if index.sum() > 2:
                mixed_input, labels_a, labels_b, lambda_ = MixUp(buf_inputs[index], buf_labels[index], alpha=self.args.gamma)
                loss += self.process_mixup(mixed_input, labels_a, labels_b, lambda_)

        loss += loss_re

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)

        return loss.item()

    def end_task(self, dataset):
        """
        Save the model
        """
        if self.args.save_store:
            # save the last model
            self.args.model_path = './save_models/{}'.format(self.args.dataset)
            self.args.save_folder = os.path.join(self.args.model_path, self.args.notes) 
            if not os.path.isdir(self.args.save_folder):
                os.makedirs(self.args.save_folder)
            save_file = os.path.join(
                self.args.save_folder, 'task_{task_id}.pth'.format(task_id=self.task))
            save_model(self.net, self.opt, self.args, self.task, save_file)

        self.task += 1