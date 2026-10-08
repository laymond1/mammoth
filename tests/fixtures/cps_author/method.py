# Copyright (c) 2026 Wonseon Lim, Jaesung Lee, Dae-Won Kim
# Licensed under the MIT License (see LICENSE in the project root).
"""
CPS-Prompt: Critical Patch-Aware Sparse Prompting with Decoupled Training
for Continual Learning on the Edge.

This module defines the top-level continual-learning model `CPSPrompt`, which
plugs into the Mammoth framework. The model wraps a frozen ViT (loaded from
``timm``) with a CODA-style prompt pool and a linear classifier head, and
implements *decoupled training*:

* For the first ``phase_ratio`` fraction of epochs in each task, only the
  prompt is trained, using a sparse forward pass over a critical subset of
  image patches (see ``CriticalPatchSampling``).
* For the remaining epochs, only the linear classifier head is trained on top
  of frozen features computed with a full (dense) forward pass.

The backbone passed in by Mammoth is intentionally discarded; ``CPSPrompt``
constructs its own ``PromptModel`` instead. The ViT size is selected via
``--vit_type`` (``tiny`` / ``small`` / ``base``; default ``tiny``).
"""

import logging

import torch
import torch.nn.functional as F

from datasets import get_dataset
from utils.args import ArgumentParser

from models.utils.continual_model import ContinualModel
from models.cps_prompt_utils.model import PromptModel
from utils.schedulers import CosineSchedule
from utils import binary_to_boolean_type


class CPSPrompt(ContinualModel):
    """CPS-Prompt: Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge."""
    NAME = 'cps-prompt'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Parameters
        parser.add_argument('--vit_type', type=str, default='tiny', choices=['tiny', 'small', 'base'], help='ViT type')
        parser.add_argument('--e_prompt_layer_idx', type=int, default=[0, 1, 2, 3, 4], nargs="+", help='the layer index of the E-Prompt')
        parser.add_argument('--e_prompt_pool_size', type=int, default=100, help='pool size')
        parser.add_argument('--e_prompt_length', type=int, default=8, help='prompt length')
        parser.add_argument('--ortho_mu', type=float, default=0.0, help='orthogonal penalty weight')  # default 0.0 following CODA-Prompt issue #12: https://github.com/GT-RIPL/CODA-Prompt/issues/12
        parser.add_argument('--pull_constraint_coeff', type=float, default=1.0, help='Coefficient(mu) for the pull constraint term, \
                            controlling the weight of the prompt loss in the total loss calculation')

        # Critical Patch Sampling
        parser.add_argument('--reduction_ratio', type=float, default=0.5, help='the ratio of patches to reduce')
        parser.add_argument('--sampling', type=str, default='critical_score', choices=['uniform', 'critical_score'], help='sampling method for patch merging')
        parser.add_argument('--temperature', type=float, default=1.0, help='temperature for the attention scaling')
        # Decoupled Prompt and Classifier Training
        parser.add_argument('--phase_ratio', type=float, default=0.8, help='the ratio of the epochs to start training the classifier')

        # ETC
        parser.add_argument('--clip_grad', type=float, default=1.0, help='Clip gradient norm')
        parser.add_argument('--use_scheduler', type=binary_to_boolean_type, default=True, help='Use scheduler')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        # CPS-Prompt ignores the backbone supplied by Mammoth and builds its
        # own frozen ViT (loaded from timm) inside PromptModel.
        del backbone
        logging.info("-" * 20)
        logging.info(
            "CPS-Prompt uses a custom backbone: vit_%s_patch16_224 "
            "(pretrained on ImageNet-21k and fine-tuned on ImageNet-1k).",
            getattr(args, 'vit_type', 'tiny'),
        )
        logging.info("-" * 20)

        tmp_dataset = get_dataset(args) if dataset is None else dataset
        num_classes = tmp_dataset.N_CLASSES
        args.n_tasks = tmp_dataset.N_TASKS
        backbone = PromptModel(args, 
                               num_classes=num_classes,
                               pretrained=True, prompt_flag='coda',
                               prompt_param=[args.e_prompt_pool_size, args.e_prompt_length, args.ortho_mu])

        super().__init__(backbone, loss, args, transform, dataset=dataset)
    
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
        
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        # Decoupled training: the first ``phase_ratio`` fraction of epochs in
        # each task trains the prompt with a sparse forward pass; the rest of
        # the epochs train only the linear classifier on top of frozen
        # full-patch features.
        if epoch < int(self.args.n_epochs * self.args.phase_ratio):
            logits, loss_prompt = self.net(inputs, train=True)
        else:
            with torch.no_grad():
                feats = self.net(inputs, feat=True, train=False).detach()
            logits = self.net(feats, last=True)
            loss_prompt = None
        # Mask out classes from previous tasks so that the cross-entropy loss
        # is computed only over classes belonging to the current task split
        # (standard class-incremental learning trick).
        logits[:, :self.n_past_classes] = -float('inf')

        loss = self.loss(logits[:, :self.n_seen_classes], labels)
        if self.args.pull_constraint_coeff > 0.0 and loss_prompt is not None:
            # The .mean() reduction is required under DataParallel, where the
            # per-replica losses are concatenated rather than averaged.
            loss = loss + self.args.pull_constraint_coeff * loss_prompt.mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        self.opt.step()

        # Running training accuracy (for logging only).
        preds = torch.argmax(logits[:, :self.n_seen_classes], dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total

        self.count += 1
        self.running_loss += loss.item()
        self.running_accuracy += accuracy

        return loss.item()

    def get_parameters(self):
        # Only the prompt pool and the linear classifier head are trainable;
        # the ViT backbone is kept frozen. The selection is done by name
        # because the underlying PromptModel registers the prompt pool as a
        # submodule named ``prompt`` and the classifier as ``head``.
        return [p for n, p in self.net.named_parameters() if 'prompt' in n or 'head' in n]
    
    def forward(self, x):
        return self.net(x)[:, :self.n_seen_classes]