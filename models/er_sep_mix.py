# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np

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


def PairMixUp(input_a, input_b, labels_a, labels_b, alpha=1.0):
    if alpha > 0:
        lambda_ = np.random.beta(alpha, alpha)
    else:
        lambda_ = 1
 
    mixed_input = lambda_ * input_a + (1 - lambda_) * input_b

    return mixed_input, labels_a, labels_b, lambda_

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



class ErSepMix(ContinualModel):
    NAME = 'er_sep_mix'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErSepMix, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0

    def process_mixup(self, mixed_input, labels_a, labels_b, lambda_):
        output = self.net(mixed_input)
        loss = MixUpLoss(self.loss, output, labels_a, labels_b, lambda_)
        
        return loss

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()

        # stream data
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            # buffer loss
            outputs = self.net(buf_inputs)
            loss += self.loss(outputs, buf_labels)
            
            # mixup loss
            mixed_input, labels_a, labels_b, lambda_ = PairMixUp(inputs, buf_inputs, labels, buf_labels, alpha=self.args.gamma)
            loss += self.process_mixup(mixed_input, labels_a, labels_b, lambda_)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset) -> None:
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
