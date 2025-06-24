"""
This module implements the simplest form of incremental training, i.e., finetuning.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from models.utils.continual_model import ContinualModel
from models.vit_utils.model import ViT
from utils.schedulers import CosineSchedule
from utils.args import ArgumentParser
from utils import binary_to_boolean_type


class Sgd(ContinualModel):
    """
    Finetuning baseline - simple incremental training.
    """

    NAME = 'sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the model.
        """
        parser.add_argument('--vit_type', type=str, default='tiny', choices=['tiny', 'small', 'base'], help='ViT type')
        parser.add_argument('--use_scheduler', type=binary_to_boolean_type, default=True, help='Use scheduler')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):

        num_classes = dataset.N_CLASSES
        args.n_tasks = dataset.N_TASKS
        backbone = ViT(args, num_classes=num_classes, pretrained=True)
        super(Sgd, self).__init__(backbone, loss, args, transform, dataset=dataset)

    def begin_task(self, dataset):
        if self.args.use_scheduler:
            self.scheduler = CosineSchedule(self.opt, K=self.args.n_epochs)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """
        SGD trains on the current task using the data provided, with no countermeasures to avoid forgetting.
        """
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
