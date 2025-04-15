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
from models.ovor_utils.ood import NPOS
from utils.buffer import Buffer

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
        parser.add_argument('--sample_from', type=int, default=600)
        parser.add_argument('--select', type=int, default=50)
        parser.add_argument('--pick_nums', type=int, default=30)
        parser.add_argument('--K', type=int, default=50)
        parser.add_argument('--lmda', type=float, default=0.1)
        parser.add_argument('--huber', action='store_false')

        # ETC
        parser.add_argument('--clip_grad', type=float, default=1, help='Clip gradient norm')
        parser.add_argument('--use_amp', type=bool, default=True, help='Use automatic mixed precision')

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

    # def observe(self, inputs, labels, not_aug_inputs, epoch=None):
    #     if isinstance(self.device, str):
    #         device = torch.device(self.device)
    #     else:
    #         device = self.device

    #     # with torch.amp.autocast(device_type=device.type, enabled=self.args.use_amp):
    #     logits, _ = self.net(inputs, train=True)
    #     # here is the trick to mask out classes of non-current tasks
    #     logits[:, :self.n_past_classes] = -float('inf')

    #     loss = self.loss(logits[:, :self.n_seen_classes], labels)
        
    #     self.opt.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
    #     self.opt.step()
    #     # self.scaler.scale(loss).backward()
    #     # torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
    #     # self.scaler.step(self.opt)
    #     # self.scaler.update()

    #     return loss.item()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if isinstance(self.device, str):
            device = torch.device(self.device)
        else:
            device = self.device

        # with torch.amp.autocast(device_type=device.type, enabled=self.args.use_amp):
        if self.epoch_iteration < self.args.n_epochs * 0.8:
            logits, loss = self.single_train(inputs, labels)
        else:
            logits, loss = self.single_train_ood(inputs, labels)
        
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        self.opt.step()
        # self.scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        # self.scaler.step(self.opt)
        # self.scaler.update()

        return loss.item()
    
    def single_train(self, inputs, labels):
        logits, loss = self.model_forward(inputs, labels) 
        return logits, loss
    
    def _get_ood_samples(self, x, y):
        subset_size = self.n_seen_classes - self.n_past_classes # number of unseen classes
        id_feats = [torch.empty(0, self.net.embed_dim) for _ in range(subset_size)]
        with torch.no_grad():
            feats = self.net(x, train=False, feat=True)
            feats = feats.detach().cpu() 
            # feats = self.model(x=imgs, feat=True).detach().cpu()
            for i, idx in enumerate(y):
                key = (idx % subset_size).item() #  0 ~ subset_size -1 까지는 모두 in-distribution
                id_feats[key] = torch.cat((id_feats[key], feats[i].view(1, -1)), 0)
        return self.ood.generate(id_feats, self.n_past_classes)

    # def _get_ood_samples(self, x, y):
    #     sorted_y, sorted_idx = torch.sort(y.detach().cpu())
    #     with torch.no_grad():
    #         _, feats = self.net(x, train=False, feat=True)
    #         feats = feats.detach().cpu() 
    #         id_feats = feats[sorted_idx]

    #     return self.ood.generate(id_feats, sorted_y)

    def single_train_ood(self, inputs, labels):
        # get ood samples (features)
        id_loader, ood_loader = self._get_ood_samples(inputs, labels)
        # print("length of id_loader: ", len(id_loader))
        # print("length of ood_loader: ", len(ood_loader))
        loss = 0.0
        for ((ids_x, targets), oods) in zip(id_loader, ood_loader):
            oods_x = oods[0]

            ids_x = ids_x.to(self.device)
            oods_x = oods_x.to(self.device)
            targets = targets.to(self.device)

            logits, ood_loss = self.model_ood_forward(ids_x, oods_x, targets)
            loss += ood_loss

        return logits, loss
    
    def model_forward(self, x, y):
        logits, _ = self.net(x, train=True)
        # here is the trick to mask out classes of non-current classes
        logits[:, :self.n_past_classes] = -float('inf')

        loss = self.loss(logits[:, :self.n_seen_classes], y)

        return logits, loss

    def model_ood_forward(self, id_x, ood_x, y):
        id_logits = self.net(id_x, last=True)[:, :self.n_seen_classes]
        # here is the trick to mask out classes of non-current classes
        id_logits[:, :self.n_past_classes] = -float('inf')
        ood_logits = self.net(ood_x, last=True)[:, self.n_past_classes:self.n_seen_classes]
        
        loss = self.loss(id_logits, y)
        ood_loss, id_score, ood_score = self.ood.loss(id_logits[:, self.n_past_classes:], ood_logits)
        loss = loss + ood_loss
            
        return id_logits, loss

    def get_parameters(self):
        return [p for n, p in self.net.named_parameters() if 'prompt' in n or 'head' in n]
    
    def forward(self, x):
        return self.net(x)[:, :self.n_seen_classes]