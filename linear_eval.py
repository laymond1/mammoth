'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/main_linear.py
'''

from __future__ import print_function

import os
import re
import sys
import argparse
import time
import math
import numpy as np

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')

from datasets import NAMES as DATASET_NAMES
from datasets import ContinualDataset, get_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset

# from datasets import TinyImagenet
from datasets.seq_tinyimagenet import base_path
from utils.status import AverageMeter
from utils.warm_up import adjust_learning_rate, warmup_learning_rate
from utils.conf import set_random_seed

import wandb

# try:
#     import apex
#     from apex import amp, optimizers
# except ImportError:
#     pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument('--notes', type=str, default='')
    
    parser.add_argument('--buffer_size', type=int, default=500)
    
    parser.add_argument('--mode', type=str, default='linear_all', choices=['train', 'linear_eval', 'linear_all'])

    parser.add_argument('--target_task', type=int, default=0, help='Use all classes if None else learned tasks so far')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='seq-cifar10',
                        choices=['seq-cifar10', 'seq-cifar100', 'seq-tinyimg'], help='dataset')
    parser.add_argument('--size', type=int, default=32)

    # other setting
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    # wandb
    parser.add_argument('--wandb_project', type=str, default='Linear-Eval')
    parser.add_argument('--wandb_entity', type=str, default='laymond1')
    parser.add_argument('--nowand', default=0, choices=[0, 1], type=int, help='Inhibit wandb logging')
    parser.add_argument('--debug_mode', type=int, default=0, help='Run only a few forward steps per epoch')         

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = '~/data/'
    # detail setting
    opt.trial = opt.seed
    opt.method = re.search(r'^(.*?)\d+_\d', opt.notes).group(1)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.lr * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.lr - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.lr

    if opt.dataset == 'seq-cifar10':
        opt.n_cls = 10
        opt.cls_per_task = 2
        opt.size = 32
    elif opt.dataset == 'seq-cifar100':
        opt.n_cls = 100
        opt.cls_per_task = 10
        opt.size = 32
    elif opt.dataset == 'seq-tinyimg':
        opt.n_cls = 200
        opt.cls_per_task = 20
        opt.size = 64
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    opt.origin_ckpt = opt.ckpt
    opt.ckpt = os.path.join(opt.ckpt, 'task_{target_task}.pth'.format(target_task=opt.target_task))
    return opt


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
    def __init__(self, feats_dim=512, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feats_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


def set_model(opt):
    model = get_dataset(opt).get_backbone()
    criterion = torch.nn.CrossEntropyLoss()
    # SCR and Co2L 일때만
    if opt.method in ['scr', 'co2l']:
        model.head = SupConMLP(input_size=model.linear.in_features, output_size=128)
    classifier = LinearClassifier(model.linear.in_features, model.linear.out_features)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         model.encoder = torch.nn.DataParallel(model.encoder)
    #     else:
    #         new_state_dict = {}
    #         for k, v in state_dict.items():
    #             k = k.replace("module.", "")
    #             new_state_dict[k] = v
    #         state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    acc = 0.0
    cnt = 0.0
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model(images, returnt='features')
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc += (output.argmax(1) == labels).float().sum().item()
        cnt += bsz

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1:.3f}'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=acc/cnt*100.))
            # wandb log
            if not opt.nowand:
                wandb.log({'running_train_loss': losses.avg, 'running_train_acc': acc/cnt*100.})
            sys.stdout.flush()

    return losses.avg, acc/cnt*100.


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    corr = [0.] * (opt.target_task + 1) * opt.cls_per_task
    cnt  = [0.] * (opt.target_task + 1) * opt.cls_per_task
    correct_task = 0.0


    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model(images, returnt='features'))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #
            cls_list = np.unique(labels.cpu())
            correct_all = (output.argmax(1) == labels)

            for tc in cls_list:
                mask = labels == tc
                correct_task += (output[mask, (tc // opt.cls_per_task) * opt.cls_per_task : ((tc // opt.cls_per_task)+1) * opt.cls_per_task].argmax(1) == (tc % opt.cls_per_task)).float().sum()

            for c in cls_list:
                mask = labels == c
                corr[c] += correct_all[mask].float().sum().item()
                cnt[c] += mask.float().sum().item()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1:.3f} {task_il:.3f}'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.))
                # wandb log
                if not opt.nowand:
                    wandb.log({'running_valid_loss': losses.avg, 
                               'running_class_Acc': np.sum(corr)/np.sum(cnt)*100.,
                               'running_task_Acc': correct_task/np.sum(cnt)*100.})

    print(' * Acc@1 {top1:.3f} {task_il:.3f}'.format(top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.))
    return losses.avg, top1.avg, corr, cnt, correct_task/np.sum(cnt)*100.


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'seq-cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'seq-cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'seq-tinyimg':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))


    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='seq-tinyimg' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size,opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task)) # tasks learned so far.

    if opt.dataset == 'seq-cifar10':
        subset_indices = []
        _train_dataset = datasets.CIFAR10(root=base_path() + 'CIFAR10',
                                         transform=train_transform,
                                         download=True)

        _train_targets = np.array(_train_dataset.targets)
        for tc in target_classes:
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

        ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
        print(ut)
        print(uc)

        train_dataset =  Subset(_train_dataset, subset_indices)

        subset_indices = []
        _val_dataset = datasets.CIFAR10(root=base_path() + 'CIFAR10',
                                       train=False,
                                       transform=val_transform)
        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)

    elif opt.dataset == 'seq-cifar100':
        subset_indices = []
        _train_dataset = datasets.CIFAR100(root=base_path() + 'CIFAR100',
                                           transform=train_transform,
                                           download=True)
        
        _train_targets = np.array(_train_dataset.targets)
        for tc in target_classes:
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

        ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
        print(ut)
        print(uc)
        
        train_dataset =  Subset(_train_dataset, subset_indices)

        subset_indices = []
        _val_dataset = datasets.CIFAR100(root=base_path() + 'CIFAR100',
                                       train=False,
                                       transform=val_transform)
        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)

    elif opt.dataset == 'seq-tinyimg':
        from datasets.seq_tinyimagenet import TinyImagenet
        subset_indices = []
        _train_dataset = TinyImagenet(root=base_path() + 'TINYIMG',
                                         transform=train_transform,
                                         download=True)

        _train_targets = np.array(_train_dataset.targets)
        for tc in target_classes:
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

        ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
        print(ut)
        print(uc)

        train_dataset =  Subset(_train_dataset, subset_indices)

        subset_indices = []
        _val_dataset = TinyImagenet(root=base_path() + 'TINYIMG',
                                       train=False,
                                       transform=val_transform)
        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)

    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader


def main():
    best_acc = 0
    task_acc = 0
    opt = parse_option()
    
    set_random_seed(opt.seed)
    
    if not opt.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=opt.wandb_project, entity=opt.wandb_entity, config=vars(opt))
        opt.wandb_url = wandb.run.get_url()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = torch.optim.SGD(classifier.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
    print( optimizer.param_groups[0]['lr'])

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f} {:.3f}'.format(
            epoch, time2 - time1, acc, optimizer.param_groups[0]['lr']))

        # eval for one epoch
        loss, val_acc, val_corr, val_cnt, task_acc = validate(val_loader, model, classifier, criterion, opt)
        val_acc = np.sum(val_corr)/np.sum(val_cnt)*100.
        if val_acc > best_acc:
            best_acc = val_acc

        val_acc_stats = {}
        for cls, (cr, c) in enumerate(zip(val_corr, val_cnt)):
            if c > 0:
                val_acc_stats[str(cls)] = cr / c * 100.
        if not opt.nowand:
            wandb.log({'val_acc_mean': val_acc, 'task_acc_mean': task_acc})
            wandb.log({f'val_acc_{k}': v for k, v in val_acc_stats.items() if k != '_timestamp'})

    with open(os.path.join(opt.origin_ckpt, 'acc_all_{}.txt'.format(opt.target_task)), 'w') as f:
        out = 'best accuracy: {:.2f}\n'.format(best_acc)
        out += '{:.2f} {:.2f}'.format(val_acc, task_acc)
        print(out)
        out += '\n'
        for k, v in val_acc_stats.items():
            print(v)
            out += '{}\n'.format(v)
        f.write(out)

    save_file = os.path.join(
        opt.origin_ckpt, 'linear_all_{target_task}.pth'.format(target_task=opt.target_task))
    print('==> Saving...'+save_file)
    torch.save({
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, save_file)

if __name__ == '__main__':
    main()
