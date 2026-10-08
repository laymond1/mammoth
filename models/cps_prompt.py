"""CPS-Prompt: author behavior, adapted to Mammoth and pinned local AugReg."""

import math
from typing import ClassVar

import torch

from datasets import get_dataset
from models.cps_prompt_utils.model import PromptModel
from models.resource_prompt_utils.state import PromptStateMixin
from models.utils.continual_model import ContinualModel
from utils.schedulers import CosineSchedule


class CPSPrompt(PromptStateMixin, ContinualModel):
    NAME = 'cps-prompt'
    COMPATIBILITY: ClassVar[list[str]] = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser):
        parser.set_defaults(optimizer='adam', lr=1e-3, n_epochs=20, batch_size=16)
        parser.add_argument('--vit_type', '--cps_vit_size', dest='vit_type', choices=('tiny', 'small', 'base'), default='tiny')
        parser.add_argument('--cps_checkpoint', default=None, help='Existing pinned local AugReg NPZ; automatic downloads are disabled')
        parser.add_argument('--e_prompt_layer_idx', type=int, nargs='+', default=[0, 1, 2, 3, 4])
        parser.add_argument('--e_prompt_pool_size', '--pool_size', dest='e_prompt_pool_size', type=int, default=100)
        parser.add_argument('--e_prompt_length', '--prompt_len', dest='e_prompt_length', type=int, default=8)
        parser.add_argument('--ortho_mu', type=float, default=0.)
        parser.add_argument('--pull_constraint_coeff', type=float, default=1.)
        parser.add_argument('--reduction_ratio', type=float, default=.4)
        parser.add_argument('--temperature', type=float, default=.1)
        parser.add_argument('--sampling', choices=('critical_score', 'uniform'), default='critical_score')
        parser.add_argument('--phase_ratio', type=float, default=.4)
        parser.add_argument('--clip_grad', type=float, default=1.)
        parser.add_argument('--use_scheduler', type=int, choices=(0, 1), default=1)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        del backbone
        if not math.isfinite(args.phase_ratio) or not 0 < args.phase_ratio < 1:
            raise ValueError('CPS phase_ratio must be strictly between zero and one')
        self.joint_epochs = int(args.n_epochs * args.phase_ratio)
        if not 1 <= self.joint_epochs < args.n_epochs:
            raise ValueError('CPS must exercise both phases; phase_ratio=.4 requires at least3 epochs')
        if not math.isfinite(args.clip_grad) or args.clip_grad <= 0:
            raise ValueError('CPS clip_grad must be positive and finite')
        if args.lr_scheduler is not None:
            raise ValueError('CPS owns its cosine scheduler; use lr_scheduler=None')
        dataset = dataset if dataset is not None else get_dataset(args)
        net = PromptModel(num_classes=dataset.N_CLASSES, n_tasks=dataset.N_TASKS,
                          vit_size=args.vit_type, checkpoint=args.cps_checkpoint,
                          pool_size=args.e_prompt_pool_size, prompt_length=args.e_prompt_length,
                          ortho_mu=args.ortho_mu, layers=args.e_prompt_layer_idx,
                          reduction_ratio=args.reduction_ratio, temperature=args.temperature,
                          sampling=args.sampling, seed=getattr(args, 'seed', None) or 0)
        super().__init__(net, loss, args, transform, dataset)
        self._last_epoch = 0

    def get_parameters(self):
        return list(self.net.prompt.parameters()) + list(self.net.head.parameters())

    def begin_task(self, dataset):
        self.net.set_task(self.current_task)
        self.opt = self.get_optimizer()
        if self.args.use_scheduler:
            self.custom_scheduler = CosineSchedule(self.opt, K=self.args.n_epochs)
            self.scheduler = self.custom_scheduler
        self._last_epoch = 0

    def begin_epoch(self, epoch, dataset):
        self._last_epoch = epoch

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        epoch = self._last_epoch if epoch is None else epoch
        if type(epoch) is not int or not 0 <= epoch < self.args.n_epochs:
            raise ValueError('CPS observe requires a valid epoch index')
        self._last_epoch = epoch
        if epoch < self.joint_epochs:
            logits, penalty = self.net(inputs, train=True)
            phase = 0
        else:
            with torch.no_grad():
                features = self.net(inputs, feat=True).detach()
            logits, penalty, phase = self.net(features, last=True), None, 1
        logits = logits[:, :self.n_seen_classes]
        logits[:, :self.n_past_classes] = -float('inf')
        value = self.loss(logits, labels.long())
        if penalty is not None and self.args.pull_constraint_coeff > 0:
            value = value + self.args.pull_constraint_coeff * penalty.mean()
        self.opt.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        self.opt.step()
        return dict(loss=value.item(), cps_phase=phase, **self.net.diagnostics)

    def forward(self, x):
        return self.net(x)[:, :self.n_seen_classes]
