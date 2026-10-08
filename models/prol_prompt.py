"""PROL official active path (bfff8418), with common local AugReg weights.

Preserves source trainable transformer norms, class-key prefix routing, affine
clamps, its zero redundancy term, and both independent cosine schedulers. This
port is an implementation comparison under the shared protocol, not a paper
performance reproduction.
"""

import copy
import math

import torch

from models.prol_prompt_utils.losses import compute_prol_loss
from models.prol_prompt_utils.network import ProlNetwork
from models.utils.continual_model import ContinualModel


class ProlPrompt(ContinualModel):
    NAME = 'prol-prompt'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser):
        parser.set_defaults(backbone='prol-vit', dataset_config='ease', optimizer='adam',
                            lr=.01, optim_wd=0., batch_size=64, n_epochs=1,
                            lr_scheduler=None, enable_other_metrics=False)
        parser.add_argument('--prol_checkpoint', type=str, default=None)
        parser.add_argument('--prol_vit_size', choices=['base', 'small', 'tiny'], default='base')
        parser.add_argument('--prol_num_tasks', type=int, default=10)
        parser.add_argument('--prol_prompt_length', type=int, default=5)
        parser.add_argument('--prol_prompt_layers', type=int, nargs='+', default=list(range(7)))
        parser.add_argument('--prol_top_k_l', type=int, default=1)
        parser.add_argument('--prol_clip_grad', type=float, default=1.)
        parser.add_argument('--prol_intertask_coeff', type=float, default=.001)
        parser.add_argument('--prol_pull_coeff', type=float, default=1.)
        parser.add_argument('--prol_pull_coeff2', type=float, default=1.)
        parser.add_argument('--prol_loss_threshold', type=float, default=.3)
        parser.add_argument('--prol_norm_lr_ratio', type=float, default=.0005)
        parser.add_argument('--prol_cos_tmax', type=int, default=20)
        parser.add_argument('--prol_cos_eta_min', type=float, default=.0005)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if not isinstance(backbone, ProlNetwork):
            raise ValueError('PROL requires --backbone prol-vit')
        if (dataset is None or dataset.SETTING != 'class-il' or dataset.N_TASKS != backbone.total_tasks
                or dataset.N_CLASSES != backbone.num_classes):
            raise ValueError('PROL requires a matching equal-task class-incremental dataset')
        if not getattr(backbone, 'pretrained_checkpoint', None):
            raise ValueError('PROL requires an explicit pretrained local --prol_checkpoint')
        self._validate_protocol(args)
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.inner_scheduler = self.outer_scheduler = None
        self.reduce_lr = False
        self._active_epoch = None
        self.last_loss_components = None
        self._old_epoch_key = None

    @staticmethod
    def _validate_protocol(args):
        if (args.fitting_mode != 'epochs' or args.n_epochs != 1 or args.eval_epochs is not None
                or args.enable_other_metrics or args.lr_scheduler is not None
                or args.optimizer.lower() != 'adam' or args.optim_wd != 0
                or getattr(args, 'label_perc', 1) != 1 or getattr(args, 'label_perc_by_class', 1) != 1):
            raise ValueError('PROL active online recipe requires one epoch, source Adam/schedulers, no other metrics')
        if getattr(args, 'start_from', None) not in (None, 0):
            raise ValueError('PROL cannot use start_from to create untrained task state')
        if getattr(args, 'loadcheck', None) is not None or getattr(args, 'inference_only', False):
            raise ValueError('PROL framework checkpoint stream resume/inference is not supported in this stage')
        positive = (args.lr, args.prol_clip_grad, args.prol_norm_lr_ratio)
        nonnegative = (args.prol_intertask_coeff, args.prol_pull_coeff, args.prol_pull_coeff2,
                       args.prol_loss_threshold, args.prol_cos_eta_min)
        if (any(not math.isfinite(value) or value <= 0 for value in positive)
                or any(not math.isfinite(value) or value < 0 for value in nonnegative)
                or args.batch_size < 1 or type(args.prol_cos_tmax) is not int or args.prol_cos_tmax < 1):
            raise ValueError('PROL rates, coefficients, clipping and scheduler length must be valid')

    def get_parameters(self):
        return [parameter for parameter in self.net.prompt.parameters() if parameter.requires_grad]

    def get_optimizer(self, params=None, lr=None):
        rate = self.args.lr * self.args.batch_size / 256. if lr is None else lr
        if params is not None:
            groups = [{'params': list(params), 'lr': rate, 'weight_decay': self.args.optim_wd}]
        elif int(self.net.active_task) < 1:
            # The source includes frozen tensors in optimizer groups; they never
            # acquire gradients or optimizer state, but group order is retained.
            groups = [{'params': list(self.net.prompt.parameters()), 'lr': rate, 'weight_decay': self.args.optim_wd}]
        else:
            normal, remaining = [], []
            for name, parameter in self.net.prompt.named_parameters():
                is_normal = 'norm1' in name or 'norm2' in name or name.startswith('norm') or 'fc_norm' in name
                (normal if is_normal else remaining).append(parameter)
            groups = [{'params': remaining, 'lr': rate, 'weight_decay': self.args.optim_wd},
                      {'params': normal, 'lr': self.args.prol_norm_lr_ratio * rate, 'weight_decay': self.args.optim_wd}]
        return torch.optim.Adam(groups, weight_decay=self.args.optim_wd, eps=1e-8, betas=(.9, .999))

    def begin_task(self, dataset):
        task = self.net.begin_task(self.n_past_classes, self.n_seen_classes, process_prompt=False)
        if task != self.current_task:
            raise RuntimeError('PROL task state disagrees with the Mammoth task counter')
        # Preserve source order: grow head, build Adam and outer scheduler, then
        # process_new_task (which freezes the shared generator on task two).
        self.opt = self.get_optimizer()
        self.outer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.args.n_epochs, eta_min=0.)
        self.inner_scheduler = None
        self.net.process_task_prompt()
        self._active_epoch = None

    def begin_epoch(self, epoch, dataset):
        if type(epoch) is not int or not 0 <= epoch < self.args.n_epochs:
            raise ValueError('PROL requires a valid local epoch index')
        self.inner_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=self.args.prol_cos_tmax, eta_min=self.args.prol_cos_eta_min)
        self.reduce_lr = False
        self._active_epoch = epoch
        self.net.prompt.train(True)
        # The source retains this unused deepcopy throughout train_one_epoch.
        # Its small but real memory cost is part of the faithful active path.
        # This source-local copy must not register as a new model parameter;
        # otherwise parameter telemetry/checkpoints would include a fake key.
        object.__setattr__(self, '_old_epoch_key', copy.deepcopy(self.net.prompt.e_prompt.prompt_key))

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if type(epoch) is not int or epoch != self._active_epoch or self.inner_scheduler is None:
            raise ValueError('PROL observe requires a started local epoch')
        if bool(((labels < self.n_past_classes) | (labels >= self.n_seen_classes)).any()):
            raise ValueError('PROL training accepts current-task class labels only')
        output = self.net.outputs(inputs, labels=labels, train=True)
        loss, components = compute_prol_loss(output, labels, self.n_past_classes, criterion=self.loss,
                                            intertask_coeff=self.args.prol_intertask_coeff,
                                            pull_constraint_coeff=self.args.prol_pull_coeff,
                                            pull_constraint_coeff2=self.args.prol_pull_coeff2)
        # This threshold uses intra/intertask CE, before GM and pull losses.
        if components['loss_ce'] < self.args.prol_loss_threshold:
            self.reduce_lr = True
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite PROL loss')
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.prompt.parameters(), self.args.prol_clip_grad)
        self.opt.step()
        # The original engine synchronizes after every update, before its inner
        # scheduler. Preserve this GPU execution boundary for efficiency runs.
        if inputs.is_cuda:
            torch.cuda.synchronize(inputs.device)
        if self.reduce_lr:
            self.inner_scheduler.step()
        self.net.record_update()
        self.last_loss_components = {name: value.detach() for name, value in components.items() if name != 'generalization_matrix'}
        return loss.item()

    def end_epoch(self, epoch, dataset):
        if epoch != self._active_epoch or self.outer_scheduler is None:
            raise ValueError('PROL cannot end an epoch that has not started')
        self.outer_scheduler.step()
        self.net.prompt.eval()
        self._active_epoch = None
        self._old_epoch_key = None

    def end_task(self, dataset):
        self.net.finalize_task()

    def forward(self, inputs):
        return self.net.outputs(inputs)['logits'][:, :self.net.seen_classes]

    def get_extra_state(self):
        names = ('_n_classes_current_task', '_n_past_classes', '_n_seen_classes', '_n_remaining_classes')
        return {'version': 1, 'current_task': self.current_task,
                'class_counters': {name: getattr(self, name, None) for name in names}}

    def set_extra_state(self, state):
        names = ('_n_classes_current_task', '_n_past_classes', '_n_seen_classes', '_n_remaining_classes')
        if (not isinstance(state, dict) or state.get('version') != 1 or type(state.get('current_task')) is not int
                or not 0 <= state['current_task'] <= self.n_tasks
                or not isinstance(state.get('class_counters'), dict) or set(state['class_counters']) != set(names)
                or any(value is not None and (type(value) is not int or not 0 <= value <= self.num_classes)
                       for value in state['class_counters'].values())):
            raise ValueError('Invalid PROL checkpoint Mammoth lifecycle state')
        self._current_task = state['current_task']
        for name, value in state['class_counters'].items():
            if value is not None:
                setattr(self, name, value)
            elif hasattr(self, name):
                delattr(self, name)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        state = state_dict.get(prefix + '_extra_state')
        active = state_dict.get(prefix + 'net.active_task')
        finalized = state_dict.get(prefix + 'net.task_finalized')
        if (not isinstance(state, dict) or not isinstance(active, torch.Tensor)
                or active.shape != () or active.device.type == 'meta'
                or not isinstance(finalized, torch.Tensor) or finalized.shape != ()
                or finalized.device.type == 'meta'):
            raise ValueError('PROL checkpoint is missing wrapper/network lifecycle state')
        task = int(active)
        expected_current = 0 if task < 0 else task + int(bool(finalized))
        if state.get('current_task') != expected_current:
            raise ValueError('PROL checkpoint Mammoth and network task counters disagree')
        counters = state.get('class_counters')
        if not isinstance(counters, dict):
            raise ValueError('PROL checkpoint class counters are missing')
        if task < 0 and any(value is not None for value in counters.values()):
            raise ValueError('PROL uninitialized checkpoint must have empty class counters')
        if task >= 0:
            width = self.num_classes // self.n_tasks
            expected = {'_n_classes_current_task': width, '_n_past_classes': task * width,
                        '_n_seen_classes': (task + 1) * width,
                        '_n_remaining_classes': self.num_classes - (task + 1) * width}
            if counters != expected:
                raise ValueError('PROL checkpoint Mammoth and network class intervals disagree')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
