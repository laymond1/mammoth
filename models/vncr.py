# Copyright 2023-present, Wonseon Lim, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

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
from utils.status import ProgressBar
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--classifier', type=str, default='linear')
    parser.add_argument('--head_output_size', type=int, default=128,
                        help='Output size of the Head.')
    parser.add_argument('--simclr_temp', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.5)
    return parser


def partial_mixup(input: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


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


class VNCR(ContinualModel):
    NAME = 'vncr'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(VNCR, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.simclr_lss = AsymSupConLoss(temperature=self.args.simclr_temp, base_temperature=self.args.simclr_temp, reduction='mean')

        self.class_means = None
        
        # set new transform
        self.dataset_shape = get_dataset(args).get_data_loaders()[0].dataset.data.shape[2]
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(self.dataset_shape, self.dataset_shape), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2))
        self.two_transform = TwoCropTransform(self.transform)
        
        # head
        self.net.head = SupConMLP(input_size=self.net.linear.in_features, output_size=self.args.head_output_size)
        # task id
        self.task = 0

    def forward(self, x, returnt='linear'):
        """
        Forward pass with encoder and NCM Classifier for evluation.
        """
        returnt = self.args.classifier
        
        if returnt == 'linear':
            return self.linear_forward(x)
            
        elif returnt == 'ncm':
            return self.ncm_forward(x)
        else:
            raise ValueError(f'{returnt} is not implimented.')  
        
    def proj_forward(self, x):
        """
        Forward pass with encoder and projection head for training.
        """
        feats = self.net(x, returnt='features')
        feats = self.net.head(feats)
        return feats
    
    def linear_forward(self, x):
        """
        Forward pass with encoder and Linear Classifier for evluation.
        """
        with torch.no_grad():
            feats = self.net(x, returnt='features')
        return self.net.classifier(feats)
    
    def ncm_forward(self, x):
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
            # virtual(mixup) negative sample
            neg_not_aug_inputs = partial_mixup(input=inputs, gamma=self.args.gamma, indices=torch.randperm(inputs.size(0)))
            # merge with all samples
            inputs = torch.cat([inputs, neg_not_aug_inputs], dim=0)
        else:
            # virtual(mixup) negative sample
            neg_not_aug_inputs = partial_mixup(input=not_aug_inputs, gamma=self.args.gamma, indices=torch.randperm(not_aug_inputs.size(0)))
            # merge with all samples
            inputs = torch.cat([not_aug_inputs, neg_not_aug_inputs], dim=0)
        
        # aug(transform) the combined data
        inputs = torch.cat(self.two_transform(inputs), dim=0)
        # create neg labels and merge
        neg_labels = torch.ones(labels.size(0), dtype=torch.long).fill_(
            (self.task+1)*self.dataset.N_CLASSES_PER_TASK
            ).to(self.device)
        labels = torch.cat([labels, neg_labels], dim=0)
        
        bsz = labels.shape[0]
        # Asym SupCon Loss
        features = self.proj_forward(inputs)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.simclr_lss(features, labels, target_labels=list(range((self.task+1)*self.dataset.N_CLASSES_PER_TASK)))
       
        # compute loss
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()
        
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

    def end_task(self, dataset) -> None:
        """
        Reset the class means
        """
        self.net.train()
        
        if self.args.classifier == 'linear':
            train_loader = dataset.train_loader
            self.train_linear_classifier(train_loader)
        
        self.task += 1
        self.class_means = None
        
    def linear_observe(self, inputs, labels, not_aug_inputs, opt):
        """
        Train linear classifier
        """
        
        opt.zero_grad()

        if not self.buffer.is_empty():
            # buffer do not transform inputs
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.dataset.get_transform())
            # combine the current data with the buffer data
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
            
        # freeze encoder and pass linear classifier
        outputs = self.linear_forward(inputs)
        # compute loss
        loss = self.loss(outputs, labels)
        loss.backward()
        opt.step()
            
        return loss.item()
    
    def train_linear_classifier(self, train_loader):
        # linear optimizer
        opt = torch.optim.SGD(self.net.classifier.parameters(), lr=self.args.lr)
        scheduler = self.dataset.get_scheduler(self.net, self.args)
        # start train linear classifier
        progress_bar = ProgressBar(verbose=not self.args.non_verbose)
        for epoch in range(self.args.n_epochs):
            for i, data in enumerate(train_loader):
                if self.args.debug_mode and i > 3:
                    break
                # data
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # compute loss                    
                loss = self.linear_observe(inputs, labels, not_aug_inputs, opt)
                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, self.task+1, loss)

            if scheduler is not None:
                scheduler.step()
