# Copyright 2023-present, Wonseon Lim, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import get_dataset
from datasets.transforms.twocrop import TwoCropTransform
from torchvision.transforms import transforms

from models.utils.continual_model import ContinualModel, save_model, load_model
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer
from utils.loggers import *
from utils.scr_buffer import SCR_Buffer
from utils.simclrloss import SupConLoss
from utils.status import ProgressBar, AverageMeter
from utils.augmentations import strong_aug
from utils.training import evaluate

with suppress(ImportError):
    import wandb


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--save_store', default=1, choices=[0, 1], type=int)
    # learning rate
    parser.add_argument('--lr_decay_epochs', type=str, default='30,40',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--cosine', default=0, choices=[0, 1], type=int,
                        help='using cosine annealing')
    parser.add_argument('--warm', default=0, choices=[0, 1], type=int, 
                        help='warm-up for large batch training')
    # network
    parser.add_argument('--linear_lr', type=float, default=0.1)
    parser.add_argument('--linear_epochs', type=int, default=1)
    parser.add_argument('--linear_lr_decay_epochs', type=str, default='30,40')
    parser.add_argument('--linear_lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--classifier', type=str, default='ncm')
    parser.add_argument('--head_output_size', type=int, default=128,
                        help='Output size of the Head.')
    # loss
    parser.add_argument('--simclr_temp', type=float, default=0.1)
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


class SCR(ContinualModel):
    NAME = 'scr'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(SCR, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)
        # self.buffer = Buffer(self.args.buffer_size, self.device)
        self.buffer = SCR_Buffer(args, self.args.buffer_size, self.device)
        self.simclr_lss = SupConLoss(temperature=self.args.simclr_temp, base_temperature=self.args.simclr_temp, reduction='mean')

        self.class_means = None
        
        # set new transform
        self.dataset_shape = get_dataset(args).get_data_loaders()[0].dataset.data.shape[2]
        normalize = self.dataset.get_normalization_transform()
        args.size = self.dataset.get_image_size()
        self.transform = transforms.Compose([
            # ToPILImage(),
            transforms.Resize(size=(args.size, args.size)),
            transforms.RandomResizedCrop(size=args.size, scale=(0.1 if args.dataset=='seq-tinyimg' else 0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=args.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if args.size>32 else 0.0),
            # ToTensor(),
            normalize,
        ])
        self.two_transform = TwoCropTransform(self.transform)
        
        # head
        self.net.head = SupConMLP(input_size=self.net.linear.in_features, output_size=self.args.head_output_size)
        backbone_weights = [p for n, p in self.net._features.named_parameters()] + [p for n, p in self.net.head.named_parameters()]
        params = [{'params': backbone_weights, 'lr': self.args.lr, 
                   'weight_decay': self.args.optim_wd, 'momentum': self.args.optim_mom}]
        self.opt = torch.optim.SGD(params)
        self.classifier_opt = torch.optim.SGD(self.net.classifier.parameters(), lr=self.args.linear_lr, 
                                              momentum=self.args.optim_mom)
        self.task = 0
        
        # warm-up for large-batch training,
        if args.batch_size >= 256:
            args.warm = True
        if args.warm:
            args.warmup_from = 0.01
            args.warm_epochs = 10
            if args.cosine:
                eta_min = args.lr * (args.lr_decay_rate ** 3)
                args.warmup_to = eta_min + (args.lr - eta_min) * (
                        1 + math.cos(math.pi * args.warm_epochs / args.n_epochs)) / 2
            else:
                args.warmup_to = args.lr
        
    def forward(self, x, returnt='ncm'):
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
    
    # def ncm_forward(self, x):
    #     """
    #     Forward pass with encoder and NCM Classifier for evluation.
    #     """
    #     if self.class_means is None:
    #         with torch.no_grad():
    #             self.compute_class_means()
    #             self.class_means = self.class_means.squeeze()

    #     feats = self.net(x, returnt='features')
    #     feats = feats.view(feats.size(0), -1)
    #     feats = feats.unsqueeze(1)

    #     pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
    #     return -pred
    
    def ncm_forward(self, x):
        """
        Forward pass with encoder and NCM Classifier for evluation.
        """
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means()

        feats = self.net(x, returnt='features')

        for j in range(feats.size(0)):  # Normalize
            feats.data[j] = feats.data[j] / feats.data[j].norm()
        feats = feats.unsqueeze(2)  # (batch_size, feature_size, 1)
        means = torch.stack([self.class_means[cls.item()] for cls in self.classes_so_far])  # (n_classes, feature_size)

        #old ncm
        means = torch.stack([means] * x.size(0))  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)
        feats = feats.expand_as(means)  # (batch_size, feature_size, n_classes)
        dists = (feats - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
        return -dists  

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
            aug_batch_size = inputs.shape[0] # batch * 2
            # aug(transform) the combined data
            aug_inputs = self.transform(inputs)
            inputs = torch.cat([inputs, aug_inputs], dim=0)
            # compute features
            features = self.proj_forward(inputs)
            f1, f2 = torch.split(features, [aug_batch_size, aug_batch_size], dim=0)
        else:
            # generate two crop aug images
            inputs = torch.cat(self.two_transform(not_aug_inputs), dim=0)
           # compute features
            features = self.proj_forward(inputs)
            f1, f2 = torch.split(features, [real_batch_size, real_batch_size], dim=0)
        
        # compute loss
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.simclr_lss(features, labels)        
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()
    
    # def compute_class_means(self) -> None:
    #     """
    #     Computes a vector representing mean features for each class.
    #     """
    #     # This function caches class means
    #     transform = self.dataset.get_normalization_transform()
    #     class_means = []
    #     examples, labels = self.buffer.get_all_data(transform)
    #     for _y in self.classes_so_far:
    #         x_buf = torch.stack(
    #             [examples[i]
    #              for i in range(0, len(examples))
    #              if labels[i].cpu() == _y]
    #         ).to(self.device)
    #         with bn_track_stats(self, False):
    #             allt = None
    #             while len(x_buf):
    #                 batch = x_buf[:self.args.batch_size]
    #                 x_buf = x_buf[self.args.batch_size:]
    #                 feats = self.net(batch, returnt='features').mean(0)
    #                 if allt is None:
    #                     allt = feats
    #                 else:
    #                     allt += feats
    #                     allt /= 2
    #             class_means.append(allt.flatten())
    #     self.class_means = torch.stack(class_means)
    
    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        This function is taken from SCR paper code.
        """
        exemplar_means = {}
        cls_exemplar = {cls.item(): [] for cls in self.classes_so_far}
        buffer_filled = self.buffer.current_index
        for x, y in zip(self.buffer.examples[:buffer_filled], self.buffer.labels[:buffer_filled]):
            cls_exemplar[y.item()].append(x)
            
        for cls, exemplar in cls_exemplar.items():
            features = []
            # Extract feature for each exemplar in p_y
            for ex in exemplar:
                # feature = model.features(ex.unsqueeze(0)).detach().clone()
                feature = self.net(ex.unsqueeze(0), returnt='features').detach().clone()
                feature = feature.squeeze()
                feature.data = feature.data / feature.data.norm()  # Normalize
                features.append(feature)
            if len(features) == 0:
                mu_y = torch.normal(0, 1, size=tuple(self.net(x.unsqueeze(0), returnt='features').detach().size()))
                mu_y = mu_y.squeeze().to(self.device)
            else:
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
                
            mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
            exemplar_means[cls] = mu_y
        self.class_means = exemplar_means
    
    def begin_task(self, dataset) -> None:
        """
        Reset the learning rate
        """
        for param_group in self.opt.param_groups:
            param_group['lr'] = self.args.lr
            
        for param_group in self.classifier_opt.param_groups:
            param_group['lr'] = self.args.linear_lr
        
        if self.task > 0 and self.args.classifier == 'linear':
            save_file = os.path.join(
                        self.args.save_folder, 'task_{task_id}_{classifier}_best.pth'.format(task_id=self.task-1, classifier=self.args.classifier))
            self.net, _ = load_model(self.net, self.opt, save_file)
    
    def end_task(self, dataset) -> None:
        """
        Reset the class means
        """
        self.net.eval()
        self.net.classifier.train()
        
        best_acc = 0.0
        
        if self.args.classifier == 'linear':
            train_loader = dataset.train_loader
            progress_bar = ProgressBar(verbose=not self.args.non_verbose)
            for epoch in range(self.args.linear_epochs):
                adjust_classifier_learning_rate(self.args.linear_lr,
                                                self.args.linear_lr_decay_epochs,
                                                self.args.linear_lr_decay_rate,
                                                self.classifier_opt, epoch)
                # train
                train_loss = self.train_classifier(train_loader, progress_bar, epoch)
                # evaluate
                accs = evaluate(self, dataset)
                val_acc, val_task_acc = np.mean(accs, axis=1)
                # classifier Class-IL Acc logging
                if not self.args.nowand:
                    wandb.log({f'Classifier_Task_{self.task}_class_mean_accs': val_acc})
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    
                    if self.args.save_store:
                        # save the best model
                        self.args.model_path = './save_models/{}'.format(self.args.dataset)
                        self.args.save_folder = os.path.join(self.args.model_path, self.args.notes) 
                        if not os.path.isdir(self.args.save_folder):
                            os.makedirs(self.args.save_folder)
                        save_file = os.path.join(
                            self.args.save_folder, 'task_{task_id}_{classifier}_best.pth'.format(task_id=self.task, classifier=self.args.classifier))
                        save_model(self.net, self.opt, self.args, self.task, save_file)
                
        if self.args.save_store:
            # save the last model
            self.args.model_path = './save_models/{}'.format(self.args.dataset)
            self.args.save_folder = os.path.join(self.args.model_path, self.args.notes) 
            if not os.path.isdir(self.args.save_folder):
                os.makedirs(self.args.save_folder)
            save_file = os.path.join(
                self.args.save_folder, 'task_{task_id}_{classifier}_last.pth'.format(task_id=self.task, classifier=self.args.classifier))
            save_model(self.net, self.opt, self.args, self.task, save_file)
        
        self.task += 1
        self.class_means = None
        self.net.train()
        
    def linear_observe(self, inputs, labels, not_aug_inputs, opt):
        """
        Train linear classifier
        """
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
        
        opt.zero_grad()
        loss.backward()
        opt.step()
            
        return loss.item()
    
    def train_classifier(self, train_loader, progress_bar, epoch):
        losses = AverageMeter()
        
        for i, data in enumerate(train_loader):
            if self.args.debug_mode and i > 3:
                break
            # data
            inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # compute loss                    
            loss = self.linear_observe(inputs, labels, not_aug_inputs, self.classifier_opt)
            assert not math.isnan(loss)
            progress_bar.prog(i, len(train_loader), epoch, self.task, loss)
            losses.update(loss, inputs.size(0))
            
        return losses.avg

            
def adjust_classifier_learning_rate(lr, lr_decay_epochs, lr_decay_rate, optimizer, epoch):
    """
    Adjust the classifier's learning rate.
    :param lr: learning rate
    :param lr_decay_epochs: where to decay lr, can be a list
    :param lr_decay_rate: decay rate for learning rate
    :param epoch: the current epoch
    :param optimizer: the optimizer
    """
    steps = np.sum(epoch > np.fromstring(lr_decay_epochs, dtype=int, sep=','))
    if steps > 0:
        lr = lr * (lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr