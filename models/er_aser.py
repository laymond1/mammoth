# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

from models.utils.continual_model import ContinualModel, save_model
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.buffer_retrieve import BufferRetrieve


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # mir subsample size
    parser.add_argument('--k', type=int, default=5,
                        help='Number of nearest neighbors (K) to perform ASER (default: %(default)s).')
    parser.add_argument('--aser_type', dest='aser_type', default="asvm", type=str, choices=['neg_sv', 'asv', 'asvm'],
                        help='Type of ASER: '
                             '"neg_sv" - Use negative SV only,'
                             ' "asv" - Use extremal values of Adversarial SV and Cooperative SV,'
                             ' "asvm" - Use mean values of Adversarial SV and Cooperative SV')
    parser.add_argument('--n_smp_cls', dest='n_smp_cls', default=1.0,
                        type=float,
                        help='Maximum number of samples per class for random sampling (default: %(default)s)')
    return parser


class ErASER(ContinualModel):
    NAME = 'er_aser'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErASER, self).__init__(backbone, loss, args, transform)
        self.buffer = BufferRetrieve(args, self.args.buffer_size, self.device, self.net, retrieve='aser')
        self.task = 0

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            # buf_inputs, buf_labels = self.buffer.get_data(
            #     self.args.minibatch_size, transform=self.transform)
            buf_inputs, buf_labels = self.buffer.get_aser_data(
                inputs, labels, self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
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
