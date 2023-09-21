# Copyright 2023-present, Wonseon Lim, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import math
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import get_dataset
from datasets.transforms.twocrop import TwoCropTransform
from torchvision.transforms import transforms
from torch.utils.data import WeightedRandomSampler

from models.utils.continual_model import ContinualModel, save_model, load_model
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer, icarl_replay
from utils.loggers import *
from utils.simclrloss import SupConLoss, AsymSupConLoss
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
    parser.add_argument('--classifier', type=str, default='linear')
    parser.add_argument('--head_output_size', type=int, default=128,
                        help='Output size of the Head.')
    # loss
    parser.add_argument('--simclr_temp', type=float, default=0.1)
    parser.add_argument('--current_temp', type=float, default=0.2,)
    parser.add_argument('--past_temp', type=float, default=0.01,)
    parser.add_argument('--distill_power', type=float, default=1.0,)
    
    return parser


def fill_buffer(self, mem_buffer: Buffer, dataset, t_idx: int) -> None:
    """
    Adds examples from the current task to the memory buffer
    by means of the herding strategy.
    :param mem_buffer: the memory buffer
    :param dataset: the dataset from which take the examples
    :param t_idx: the task index
    """

    mode = self.net.training
    self.net.eval()
    samples_per_class = mem_buffer.buffer_size // len(self.classes_so_far)

    if t_idx > 0:
        # 1) First, subsample prior classes
        buf_x, buf_y, buf_l = self.buffer.get_all_data()

        mem_buffer.empty()
        for _y in buf_y.unique():
            idx = (buf_y == _y)
            _y_x, _y_y, _y_l = buf_x[idx], buf_y[idx], buf_l[idx]
            mem_buffer.add_data(
                examples=_y_x[:samples_per_class],
                labels=_y_y[:samples_per_class],
                logits=_y_l[:samples_per_class]
            )

    # 2) Then, fill with current tasks
    loader = dataset.train_loader
    norm_trans = dataset.get_normalization_transform()
    if norm_trans is None:
        def norm_trans(x): return x
    classes_start, classes_end = t_idx * dataset.N_CLASSES_PER_TASK, (t_idx + 1) * dataset.N_CLASSES_PER_TASK

    # 2.1 Extract all data
    a_x, a_y, a_f, a_l = [], [], [], []
    for x, y, not_norm_x in loader:
        mask = (y >= classes_start) & (y < classes_end)
        x, y, not_norm_x = x[mask], y[mask], not_norm_x[mask]
        if not x.size(0):
            continue
        x, y, not_norm_x = (a.to(self.device) for a in (x, y, not_norm_x))
        a_x.append(not_norm_x.to('cpu'))
        a_y.append(y.to('cpu'))
        # -- this part is not used but just keep it
        feats = self.net(norm_trans(not_norm_x), returnt='features')
        outs = self.net.classifier(feats)
        a_f.append(feats.cpu())
        a_l.append(torch.sigmoid(outs).cpu())
        # -- 
    a_x, a_y, a_f, a_l = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f), torch.cat(a_l)
    
    # 2.2 Randomly fill buffer
    for _y in a_y.unique():
        idx = (a_y == _y)
        _x, _y, _l = a_x[idx], a_y[idx], a_l[idx]
        feats = a_f[idx]
        # mean_feat = feats.mean(0, keepdim=True)

        # running_sum = torch.zeros_like(mean_feat)
        # Need random sample
        i, permuted_indices = 0, torch.randperm(_x.size(0))
        while i < samples_per_class and i < feats.shape[0]:
            # cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)

            # idx_min = cost.argmin().item()
            idx_min = permuted_indices[i].item()

            mem_buffer.add_data(
                examples=_x[idx_min:idx_min + 1].to(self.device),
                labels=_y[idx_min:idx_min + 1].to(self.device),
                logits=_l[idx_min:idx_min + 1].to(self.device)
            )
            i += 1
            
    assert len(mem_buffer.examples) <= mem_buffer.buffer_size
    assert mem_buffer.num_seen_examples <= mem_buffer.buffer_size

    self.net.train(mode)


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


class CO2L(ContinualModel):
    """
    Contrastive Continual Learning(Co2L) published in ICCV 2021.
    [https://github.com/chaht01/Co2L]
    """
    NAME = 'co2l'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(CO2L, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.asysimclr_lss = AsymSupConLoss(temperature=self.args.simclr_temp, base_temperature=self.args.simclr_temp, reduction='mean')

        self.class_means = None
        
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
        return self.net.classifier(feats.detach())
    
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
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

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

    def begin_task(self, dataset):
        """
        Reset the learning rate
        """
        for param_group in self.opt.param_groups:
            param_group['lr'] = self.args.lr
            
        for param_group in self.classifier_opt.param_groups:
            param_group['lr'] = self.args.linear_lr
        
        icarl_replay(self, dataset)
        return True

    def end_task(self, dataset) -> None:
        """
        Reset the class means
        """
        self.net.eval()
        self.net.classifier.train()
        
        best_acc = 0.0
        
        if self.args.classifier == 'linear':
            # train_loader = dataset.train_loader
            linear_train_loader = make_linear_train_loader(self, dataset)
            progress_bar = ProgressBar(verbose=not self.args.non_verbose)
            for epoch in range(self.args.linear_epochs):
                adjust_classifier_learning_rate(self.args.linear_lr,
                                                self.args.linear_lr_decay_epochs,
                                                self.args.linear_lr_decay_rate,
                                                self.classifier_opt, epoch)
                # train
                train_loss = self.train_classifier(linear_train_loader, progress_bar, epoch)
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
        
        self.net.train()
        with torch.no_grad():
            fill_buffer(self, self.buffer, dataset, self.task)
        
        self.task += 1
        self.class_means = None
        self.old_net = deepcopy(self.net.eval())
        self.net.train()
        
    def linear_observe(self, inputs, labels, not_aug_inputs, opt):
        """
        Train linear classifier
        """
        inputs = self.transform(not_aug_inputs)
            
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
            not_aug_inputs = not_aug_inputs.to(self.device)
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
        
        
def make_linear_train_loader(self, dataset):
    train_dataset = dataset.train_loader.dataset
    targets = train_dataset.targets
    
    ut, uc = np.unique(targets, return_counts=True)
    weights = np.array([0.] * len(targets))
    for t, c in zip(ut, uc):
        weights[targets == t] = 1./c
        
    train_sampler = WeightedRandomSampler(torch.Tensor(weights), len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=self.args.batch_size, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler)
    
    return train_loader