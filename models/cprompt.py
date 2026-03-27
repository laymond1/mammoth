"""
CPrompt: Consistent Prompting for Rehearsal-Free Continual Learning

Note:
    CPrompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import get_dataset
from utils.args import add_rehearsal_args, ArgumentParser

from models.utils.continual_model import ContinualModel
from models.cprompt_utils.model import PromptModel
from utils.schedulers import CosineSchedule
from utils.buffer import Buffer
from utils import parse_str_to_int, binary_to_boolean_type

import wandb


class CPrompt(ContinualModel):
    """Consistent Prompting for Rehearsal-Free Continual Learning."""
    NAME = 'cprompt'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Parameters
        parser.add_argument('--vit_type', type=str, default='tiny', choices=['tiny', 'small', 'base'], help='ViT type')
        parser.add_argument('--alpha', type=float, default=1.0, help='weight for ccl loss')
        parser.add_argument('--tau', type=float, default=1.2, help='temperature for ccl loss')
        parser.add_argument('--margin', type=float, default=0.0, help='margin for ccl loss')

        # ETC
        parser.add_argument('--clip_grad', type=float, default=1.0, help='Clip gradient norm')
        parser.add_argument('--use_amp_opt', type=binary_to_boolean_type, default=False, help='Use automatic mixed precision')
        parser.add_argument('--use_grad_checkpoint', type=binary_to_boolean_type, default=False,
                            help='Use activation checkpointing per block in ViT')
        parser.add_argument('--use_scheduler', type=binary_to_boolean_type, default=True, help='Use scheduler')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        del backbone
        print("-" * 20)
        print(f"WARNING: CPrompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        tmp_dataset = get_dataset(args) if dataset is None else dataset
        num_classes = tmp_dataset.N_CLASSES
        args.n_tasks = tmp_dataset.N_TASKS
        backbone = PromptModel(args, 
                               num_classes=num_classes,
                               pretrained=True)

        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.scaler = torch.amp.GradScaler(enabled=self.args.use_amp_opt)
        if getattr(self.args, 'use_grad_checkpoint', False):
            self.net.feat.set_grad_checkpointing(True)
    
    def begin_task(self, dataset):
        # update the classifier
        self.net.update_fc(self.n_seen_classes, self.n_seen_classes - self.n_past_classes)
        self.net.to(self.device)
        self.increment = self.n_seen_classes - self.n_past_classes
        
        if hasattr(self, 'opt'):
            self.opt.zero_grad(set_to_none=True)
            del self.opt
        self.opt = self.get_optimizer()
        if self.args.use_scheduler:
            self.scheduler = CosineSchedule(self.opt, K=self.args.n_epochs)
    
    def end_task(self, dataset):
        self.net.fix_branch_layer()

    def begin_epoch(self, epoch, dataset):
        self.count = 0
        self.running_loss = 0.0
        self.running_accuracy = 0.0

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if isinstance(self.device, str):
            device = torch.device(self.device)
        else:
            device = self.device

        def _forward_loss():
            new_labels = labels - self.n_past_classes
            logits, features = self.net.aux_forward(inputs)
            loss_aux = F.cross_entropy(logits, new_labels)
            loss = loss_aux

            if self.current_task > 0:
                for k in range(self.current_task):
                    old_logit = self.net.clas_w[k](features)['logits']
                    c1_logits = self.net.clas_w[self.current_task](features)['logits']
                    bool_ = torch.max(c1_logits, dim=1)[0] > torch.max(old_logit, dim=1)[0] + self.args.margin
                    t = torch.ones((bool_.shape)).to(self.device)
                    t[bool_ == False] = self.args.tau
                    t = t.unsqueeze(1).repeat(1, self.increment)
                    ground = F.softmax(old_logit / t, dim=1).detach().clone()
                    loss_ccl = -torch.sum(ground * torch.log(F.softmax(old_logit, dim=1)), dim=1).mean()
                    loss += self.args.alpha * loss_ccl / self.current_task

            gen_p = []
            x_querry = self.net.feat(inputs, returnbeforepool=True)[:, 0, :]
            K = self.net.keys

            s = self.current_task * self.increment
            f = (self.current_task + 1) * self.increment
            if self.current_task == 0:
                K = K[s:f]
            else:
                K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1)
            mk = torch.einsum('bd,kd->bk', q, n_K)
            loss_mk = F.cross_entropy(mk, labels)
            loss += loss_mk

            m = torch.randint(0, self.current_task + 1, (len(mk), 1))
            ts_prompts_1 = self.net.ts_prompts_1
            P1 = torch.cat([ts_prompts_1[j].weight.unsqueeze(0) for j in m], dim=0)
            gen_p.append(P1)
            ts_prompts_2 = self.net.ts_prompts_2
            P2 = torch.cat([ts_prompts_2[j].weight.unsqueeze(0) for j in m], dim=0)
            gen_p.append(P2)
            out_gen = self.net(inputs, gen_p, train=True)
            loss_ce = F.cross_entropy(out_gen, new_labels)
            loss += loss_ce

            return loss, logits, new_labels

        if self.args.use_amp_opt:
            with torch.amp.autocast(device_type=device.type, enabled=True):
                loss, logits, new_labels = _forward_loss()
        else:
            loss, logits, new_labels = _forward_loss()
        
        self.opt.zero_grad(set_to_none=True)
        if self.args.use_amp_opt:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
            self.opt.step()

        # Calculate accuracy
        preds = torch.argmax(logits[:, :self.n_seen_classes], dim=1)
        correct = (preds == new_labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total

        # Update running loss와 accuracy
        self.count += 1
        self.running_loss += loss.item()
        self.running_accuracy += accuracy

        return loss.item()
    
    def get_parameters(self):
        return [p for n, p in self.net.named_parameters() if p.requires_grad]
    
    def forward(self, x):
        gen_p = []
        with torch.no_grad():
            x_querry = self.net.feat(x, returnbeforepool=True)[:,0,:]
        
        K = self.net.keys
        
        f = self.current_task * self.increment # prediction after task
        K = K[:f]
        n_K = nn.functional.normalize(K, dim=1)
        q = nn.functional.normalize(x_querry, dim=1)
        mk = torch.einsum('bd,kd->bk', q, n_K)
        
        m = torch.max(mk, dim=1, keepdim=True)[1] // self.increment
        
        ts_prompts_1 = self.net.ts_prompts_1
        P1 = torch.cat([ts_prompts_1[j].weight.detach().clone().unsqueeze(0) for j in m],dim=0)
        gen_p.append(P1)
        ts_prompts_2 = self.net.ts_prompts_2
        P2 = torch.cat([ts_prompts_2[j].weight.detach().clone().unsqueeze(0) for j in m],dim=0)
        gen_p.append(P2)
        
        with torch.no_grad():
            out_logits = self.net(x, gen_p, train=False)
        return out_logits
