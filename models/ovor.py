"""
OVOR: OnePrompt with Virtual Outlier Regularization for Rehearsal-Free Class-Incremental Learning

Note:
    OnePrompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import sys
import gc
import time
import datetime
import torch
import torch.nn.functional as F

from datasets import get_dataset
from utils.args import add_rehearsal_args, ArgumentParser

from models.utils.continual_model import ContinualModel
from models.prompt_utils.model import PromptModel
from utils.schedulers import CosineSchedule
from models.ovor_utils.ood import NPOS
from utils.buffer import Buffer
from utils import parse_str_to_int, binary_to_boolean_type

import wandb


class OVOR(ContinualModel):
    """OnePrompt with Virtual Outlier Regularization for Rehearsal-Free Class-Incremental Learning."""
    NAME = 'ovor'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--vit_type', type=str, default='tiny', choices=['tiny', 'small', 'base'], help='ViT type')

        # G-Prompt parameters
        parser.add_argument('--g_prompt_layer_idx', type=int, default=[0, 1], nargs="+", help='the layer index of the G-Prompt')
        parser.add_argument('--g_prompt_length', type=int, default=10, help='length of G-Prompt')

        # E-Prompt parameters
        parser.add_argument('--e_prompt_layer_idx', type=int, default=[2, 3, 4], nargs="+", help='the layer index of the E-Prompt')
        parser.add_argument('--e_prompt_pool_size', type=int, default=1, help='number of prompts (fixed: 1)')
        parser.add_argument('--e_prompt_length', type=int, default=40, help='length of E-Prompt')
        
        # OOD parameters
        parser.add_argument('--cov', type=float, default=1.0) # 0.1 for CUB200
        parser.add_argument('--thres_id', type=float, default=-24.0) # -15.0 for ImageNet-A
        parser.add_argument('--thres_ood', type=float, default=-3.0)
        parser.add_argument('--num_per_class', type=int, default=40)
        parser.add_argument('--id_bsz', type=int, default=16) # In-dist batch size에 의해 ood batch size 결정됨. 16 -> 1 , 128 -> 12
        parser.add_argument('--sample_from', type=int, default=600)
        parser.add_argument('--select', type=int, default=50)
        parser.add_argument('--pick_nums', type=int, default=30)
        parser.add_argument('--K', type=int, default=100)
        parser.add_argument('--lmda', type=float, default=0.1)
        parser.add_argument('--huber',type=binary_to_boolean_type, default=False, help='Use Huber loss instead of MSE loss')

        # ETC
        parser.add_argument('--clip_grad', type=float, default=1, help='Clip gradient norm')
        parser.add_argument('--use_amp', type=bool, default=True, help='Use automatic mixed precision')
        parser.add_argument('--use_scheduler', type=binary_to_boolean_type, default=True, help='Use scheduler')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        del backbone
        print("-" * 20)
        print(f"WARNING: OnePrompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        tmp_dataset = get_dataset(args) if dataset is None else dataset
        num_classes = tmp_dataset.N_CLASSES
        backbone = PromptModel(args, 
                               num_classes=num_classes,
                               pretrained=True, prompt_flag='oneprompt',
                               prompt_param=[args.e_prompt_pool_size, args.e_prompt_length, args.g_prompt_length])
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp)
        self.ood = NPOS(args)
    
    def begin_task(self, dataset):
        if self.current_task > 0:
            self.net.prompt.process_task_count()
        if hasattr(self, 'opt'):
            self.opt.zero_grad(set_to_none=True)
            del self.opt
        self.opt = self.get_optimizer()
        if self.args.use_scheduler:
            self.scheduler = CosineSchedule(self.opt, K=self.args.n_epochs)

    def begin_epoch(self, epoch, dataset):
        self.count = 0
        self.running_loss = 0.0
        self.running_accuracy = 0.0
        # prepare the OOD dataset
        if epoch >= int(self.args.n_epochs * 0.8):
            id_loader, ood_loader = self._get_ood_samples(dataset)
            self.id_iter, self.ood_iter = iter(id_loader), iter(ood_loader)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if isinstance(self.device, str):
            device = torch.device(self.device)
        else:
            device = self.device

        # with torch.amp.autocast(device_type=device.type, enabled=self.args.use_amp):
        if epoch >= int(self.args.n_epochs * 0.8):
            id_data, ood_data = next(self.id_iter), next(self.ood_iter)
            logits, loss = self.model_ood_forward(id_data, ood_data)
        else:
            logits, loss = self.model_forward(inputs, labels) 
        
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        self.opt.step()
        # self.scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        # self.scaler.step(self.opt)
        # self.scaler.update()

        # Calculate accuracy
        preds = torch.argmax(logits[:, :self.n_seen_classes], dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total

        # Update running loss와 accuracy
        self.count += 1
        self.running_loss += loss.item()
        self.running_accuracy += accuracy

        return loss.item()
    
    def _get_ood_samples(self, dataset):
        subset_size = self.n_seen_classes - self.n_past_classes # number of unseen classes
        id_feats = [torch.empty(0, self.net.embed_dim) for _ in range(subset_size)]
        with torch.no_grad():
            for data in dataset.train_loader:
                x, y = data[0], data[1]
                x, y = x.to(self.device), y.to(self.device)
                feats = self.net(x, feat=True).detach().cpu()
                for i, idx in enumerate(y):
                    key = (idx % subset_size).item() #  0 ~ subset_size -1 까지는 모두 in-distribution
                    id_feats[key] = torch.cat((id_feats[key], feats[i].view(1, -1)), 0)
        return self.ood.generate(id_feats, self.n_past_classes)
    
    def model_forward(self, x, y):
        logits, _ = self.net(x, train=True)
        # here is the trick to mask out classes of non-current classes
        logits[:, :self.n_past_classes] = -float('inf')

        loss = self.loss(logits[:, :self.n_seen_classes], y)

        return logits, loss

    def model_ood_forward(self, id_data, ood_data):
        id_x, y = id_data[0].to(self.device), id_data[1].to(self.device)
        ood_x = ood_data[0].to(self.device)

        id_logits = self.net(id_x, last=True)[:, :self.n_seen_classes]
        # here is the trick to mask out classes of non-current classes
        id_logits[:, :self.n_past_classes] = -float('inf')
        ood_logits = self.net(ood_x, last=True)[:, self.n_past_classes:self.n_seen_classes]

        loss = self.loss(id_logits, y)
        ood_loss, id_score, ood_score = self.ood.loss(id_logits[:, self.n_past_classes:], ood_logits)
        loss += ood_loss

        return id_logits, loss

    def get_parameters(self):
        return [p for n, p in self.net.named_parameters() if 'prompt' in n or 'head' in n]
    
    def forward(self, x):
        return self.net(x)[:, :self.n_seen_classes]