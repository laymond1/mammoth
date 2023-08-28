# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import get_dataset
from datasets.transforms.twocrop import TwoCropTransform

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer
from utils.simclrloss import SupConLoss, AsymSupConLoss
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--head_output_size', type=int, default=128,
                        help='Output size of the Head.')
    parser.add_argument('--simclr_temp', type=float, default=0.1)
    parser.add_argument('--current_temp', type=float, default=0.2,)
    parser.add_argument('--past_temp', type=float, default=0.01,)
    parser.add_argument('--distill_power', type=float, default=1.0,)
    return parser


class SupConMLP(nn.Module):
    """
    Supervised Contrastive MLP
    """
    def __init__(self, input_size, output_size=128, **kwargs):
        super(SupConMLP, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(inplace=True),
                nn.Linear(input_size, output_size)
            )
        self.output_size = output_size

    def forward(self, x):
        feats = F.normalize(self.fc(x), dim=1)
        return feats


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, input_size, output_size, **kwargs):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        return self.fc(x)


class CO2L(ContinualModel):
    NAME = 'co2l'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(CO2L, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.asysimclr_lss = AsymSupConLoss(temperature=self.args.simclr_temp, base_temperature=self.args.simclr_temp, reduction='mean')

        self.class_means = None
        
        # TODO: change it for CO2L (set new transform)
        self.dataset_shape = get_dataset(args).get_data_loaders()[0].dataset.data.shape[2]
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(self.dataset_shape, self.dataset_shape), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2))
        self.two_transform = TwoCropTransform(self.transform)
        
        # head
        self.net.head = SupConMLP(input_size=self.net.linear.in_features, output_size=128)
        # linear head
        self.linear_heads = {}

        self.old_net = None
        self.task = 0
    
    def forward(self, x):
        """
        Forward pass with encoder and NCM Classifier for evluation.
        """
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means()
                self.class_means = self.class_means.squeeze()

        feats = self.net(x, returnt='features')
        feats = feats.view(feats.size(0), -1)
        feats = feats.unsqueeze(1)

        pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return -pred       
        
    def proj_forward(self, x):
        """
        Forward pass with encoder and projection head for training.
        """
        feats = self.net(x, returnt='features')
        feats = self.net.head(feats)
        return feats
        
    def observe(self, inputs, labels, not_aug_inputs):
        # set classes_so_far attribute referenced from icarl.py
        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            # buffer do not transform inputs
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=None)
            # combine the current data with the buffer data
            inputs = torch.cat((not_aug_inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
            # aug(transform) the combined data
            inputs = torch.cat(self.two_transform(inputs), dim=0)
        else:
            inputs = torch.cat(self.two_transform(not_aug_inputs), dim=0)

        bsz = labels.shape[0]
                 
        with torch.no_grad():
            prev_task_mask = labels < self.task * self.dataset.N_CLASSES_PER_TASK
            prev_task_mask = prev_task_mask.repeat(2)
                    
        # compute features
        features = self.proj_forward(inputs)
        
        # IRD (current)
        if self.task > 0:
            features1_prev_task = features

            features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), self.args.current_temp)
            logits_mask = torch.scatter(
                torch.ones_like(features1_sim),
                1,
                torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                0
            )
            logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
            features1_sim = features1_sim - logits_max1.detach()
            row_size = features1_sim.size(0)
            logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
        
        # Asym SupCon Loss
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.asysimclr_lss(features, labels, target_labels=list(range(self.task * self.dataset.N_CLASSES_PER_TASK, (self.task+1) * self.dataset.N_CLASSES_PER_TASK))) 
        
        # IRD (past)
        if self.task > 0:
            with torch.no_grad():
                features2_prev_task = self.old_net(inputs, returnt='features')
                features2_prev_task = self.old_net.head(features2_prev_task)

                features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), self.args.past_temp)
                logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                features2_sim = features2_sim - logits_max2.detach()
                logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)


            loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
            loss += self.args.distill_power * loss_distill
        
        # compute loss
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()
        
    def end_task(self, dataset) -> None:
        self.old_net = deepcopy(self.net.eval())
        
        self.task += 1
        self.net.train()
        self.class_means = None
            
    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        transform = self.dataset.get_normalization_transform()
        class_means = []
        examples, labels = self.buffer.get_all_data(transform)
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)
            with bn_track_stats(self, False):
                allt = None
                while len(x_buf):
                    batch = x_buf[:self.args.batch_size]
                    x_buf = x_buf[self.args.batch_size:]
                    feats = self.net(batch, returnt='features').mean(0)
                    if allt is None:
                        allt = feats
                    else:
                        allt += feats
                        allt /= 2
                class_means.append(allt.flatten())
        self.class_means = torch.stack(class_means)