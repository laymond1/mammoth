"""HiDe-Prompt: task inference, task prompts, and distribution-based alignment.

Port of thu-ml/HiDe-Prompt revision 0ed433334ff2cfb206c5e2899651c3044ec85f92.
Each native task trains TII, records uninstructed statistics and aligns TII,
then trains its prompt and records/aligns instructed statistics. No later-task
TII checkpoint or test task identity is supplied to the inference path.
"""

import copy
from contextlib import contextmanager
import math
import time

import torch
from torch.utils.data import DataLoader, Subset

from datasets.utils.continual_dataset import MammothDatasetWrapper
from models.hide_prompt_utils.network import HideNetwork
from models.hide_prompt_utils.statistics import ClassFeatureStatistics, align_classifier, orthogonality_loss
from models.utils.continual_model import ContinualModel
from utils import binary_to_boolean_type


class HidePrompt(ContinualModel):
    NAME = 'hide-prompt'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser):
        # Keep dataset_config out of the model YAML so an explicit common-pilot
        # preprocessing override retains normal CLI precedence.
        parser.set_defaults(backbone='hide-vit', dataset_config='hide_prompt',
                            optimizer='adam', lr=0.03, optim_wd=0., batch_size=48,
                            n_epochs=70, lr_scheduler=None, enable_other_metrics=False)
        parser.add_argument('--hide_checkpoint', type=str, default=None)
        parser.add_argument('--hide_vit_size', choices=['base'], default='base')
        parser.add_argument('--hide_pool_size', type=int, default=10)
        parser.add_argument('--hide_prompt_length', type=int, default=5)
        parser.add_argument('--hide_prompt_layers', type=int, nargs='+', default=[0, 1, 2, 3, 4])
        parser.add_argument('--hide_mlp_structure', type=int, nargs='+', default=[2])
        parser.add_argument('--hide_prompt_momentum', type=float, default=0.01)
        parser.add_argument('--hide_drop', type=float, default=0.)
        parser.add_argument('--hide_drop_path', type=float, default=0.)
        parser.add_argument('--hide_use_permute_fix', type=binary_to_boolean_type, default=False)
        parser.add_argument('--hide_tii_epochs', type=int, default=20)
        parser.add_argument('--hide_prompt_epochs', type=int, default=50)
        parser.add_argument('--hide_ca_epochs', type=int, default=30)
        parser.add_argument('--hide_tii_lr', type=float, default=0.0005)
        parser.add_argument('--hide_ca_lr', type=float, default=0.005)
        parser.add_argument('--hide_head_lr_ratio', type=float, default=0.1)
        parser.add_argument('--hide_decay_epochs', type=int, default=30)
        parser.add_argument('--hide_decay_rate', type=float, default=0.1)
        parser.add_argument('--hide_reg', type=float, default=0.1)
        parser.add_argument('--hide_clip_grad', type=float, default=1.)
        parser.add_argument('--hide_n_centroids', type=int, default=10)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if not isinstance(backbone, HideNetwork):
            raise ValueError('HiDe-Prompt requires --backbone hide-vit')
        self._validate_protocol(args)
        if dataset is None or dataset.SETTING != 'class-il' or dataset.N_TASKS > backbone.total_tasks:
            raise ValueError('HiDe-Prompt requires a class-incremental dataset within its task pool capacity')
        if not getattr(backbone, 'pretrained_checkpoint', None):
            raise ValueError('HiDe-Prompt requires an explicit pretrained local --hide_checkpoint')
        backbone.set_phase('tii')
        self._phase = 'tii'
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        seed = args.seed if args.seed is not None else 0
        self.tii_statistics = ClassFeatureStatistics(self.num_classes, self.net.feature_dim,
                                                    n_centroids=args.hide_n_centroids, seed=seed)
        self.prompt_statistics = ClassFeatureStatistics(self.num_classes, self.net.feature_dim,
                                                       n_centroids=args.hide_n_centroids, seed=seed)
        self.register_buffer('tii_ready', torch.tensor(False))
        self.register_buffer('prompt_ready', torch.tensor(False))
        self._stage_epoch = -1
        self._epoch_started = None
        self.phase_history = []
        self.alignment_history = []
        self._updates = {'tii': 0, 'prompt': 0}

    @staticmethod
    def _validate_protocol(args):
        if args.fitting_mode != 'epochs' or args.eval_epochs is not None or args.enable_other_metrics:
            raise ValueError('HiDe-Prompt requires epoch fitting and task-end evaluation without other metrics')
        if args.lr_scheduler is not None:
            raise ValueError('HiDe-Prompt owns separate TII/prompt schedules; set lr_scheduler=None')
        if args.label_perc != 1 or args.label_perc_by_class != 1:
            raise ValueError('HiDe-Prompt distribution fitting requires fully labeled training data')
        if min(args.hide_tii_epochs, args.hide_prompt_epochs, args.hide_ca_epochs, args.hide_n_centroids) < 1:
            raise ValueError('HiDe-Prompt phase epoch counts and centroid count must be positive')
        if args.n_epochs != args.hide_tii_epochs + args.hide_prompt_epochs:
            raise ValueError('n_epochs must equal hide_tii_epochs + hide_prompt_epochs; a pilot with 1+1 needs n_epochs=2')
        positive = (args.lr, args.hide_tii_lr, args.hide_ca_lr, args.hide_head_lr_ratio,
                    args.hide_clip_grad, args.hide_decay_epochs, args.hide_decay_rate)
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError('HiDe-Prompt learning rates, gradient clipping and decay parameters must be positive')
        if not math.isfinite(args.hide_reg) or args.hide_reg < 0:
            raise ValueError('hide_reg must be finite and nonnegative')

    def get_parameters(self):
        return self.net.phase_parameters(self._phase)

    def get_optimizer(self, params=None, lr=None):
        if params is not None:
            return super().get_optimizer(params, lr)
        phase = getattr(self, '_phase', 'tii')
        if phase not in ('tii', 'prompt'):
            return None
        scale = self.args.batch_size / 256.
        if phase == 'tii':
            parameters = self.net.phase_parameters('tii')
            rate = self.args.hide_tii_lr * scale if lr is None else lr
            groups = [{'params': parameters, 'lr': rate}]
        else:
            rate = self.args.lr * scale if lr is None else lr
            prompts = self.net.prompt.e_prompt.parameters()
            groups = [{'params': list(prompts), 'lr': rate},
                      {'params': list(self.net.prompt.head.parameters()), 'lr': rate * self.args.hide_head_lr_ratio}]
        # Source default Adam has no weight decay. Alignment uses its own SGD.
        kind = self.args.optimizer.lower()
        if kind == 'sgd':
            return torch.optim.SGD(groups, lr=rate, momentum=self.args.optim_mom,
                                   weight_decay=self.args.optim_wd, nesterov=self.args.optim_nesterov)
        if kind in ('adam', 'adamw'):
            return getattr(torch.optim, 'Adam' if kind == 'adam' else 'AdamW')(groups, lr=rate,
                                                                           weight_decay=self.args.optim_wd)
        raise ValueError(f'Unknown optimizer: {self.args.optimizer}')

    def begin_task(self, dataset):
        # Mammoth moves model.net, rather than the whole ContinualModel, before
        # training. Move the separately registered statistics to the same device.
        self.tii_statistics.to(self.device)
        self.prompt_statistics.to(self.device)
        task = self.net.begin_task(self.n_past_classes, self.n_seen_classes)
        if task != self.current_task:
            raise RuntimeError('HiDe-Prompt task mapping disagrees with the native task counter')
        self.tii_ready.fill_(False)
        self.prompt_ready.fill_(False)
        self._stage_epoch = -1
        self._updates = {'tii': 0, 'prompt': 0}
        self._phase = 'tii'
        self.net.set_phase('tii')
        self.opt = self.get_optimizer()

    def begin_epoch(self, epoch, dataset):
        if not 0 <= epoch < self.args.n_epochs:
            raise ValueError('HiDe-Prompt epoch is outside its two-stage schedule')
        phase = 'tii' if epoch < self.args.hide_tii_epochs else 'prompt'
        if phase == 'prompt' and not bool(self.tii_ready):
            raise RuntimeError('Finish current-task TII statistics/alignment before prompt training')
        if phase != self._phase:
            self._phase = phase
            self.net.set_phase(phase)
            self.opt = self.get_optimizer()
        self._stage_epoch = epoch if phase == 'tii' else epoch - self.args.hide_tii_epochs
        if phase == 'prompt':
            rate = self.args.lr * self.args.batch_size / 256.
            # Source calls StepLRScheduler.step(local_epoch) AFTER training.
            # With decay_epochs=30, its first reduced rate is used at epoch31.
            rate *= self.args.hide_decay_rate ** (max(self._stage_epoch - 1, 0) // self.args.hide_decay_epochs)
            self.opt.param_groups[0]['lr'] = rate
            self.opt.param_groups[1]['lr'] = rate * self.args.hide_head_lr_ratio
        self._epoch_started = time.perf_counter()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if self._stage_epoch < 0 or self._phase not in ('tii', 'prompt'):
            raise RuntimeError('Call meta_begin_epoch before observing HiDe-Prompt training samples')
        if epoch is not None and epoch != self._stage_epoch + (self.args.hide_tii_epochs if self._phase == 'prompt' else 0):
            raise ValueError('observe epoch disagrees with the current HiDe-Prompt phase')
        if bool(((labels < self.n_past_classes) | (labels >= self.n_seen_classes)).any()):
            raise ValueError('HiDe-Prompt training may only consume current-task classes')
        preserved = None
        if self._phase == 'tii':
            output = self.net.tii_outputs(inputs)
            regularizer = None
        else:
            with torch.no_grad():
                self.net.tii.eval()
                tii = self.net.tii_outputs(inputs)
                ids = self.net.infer_prompt_ids(tii['logits'], self.n_seen_classes)
            output = self.net.prompt_outputs(inputs, task_id=self.current_task, prompt_id=ids, train=True)
            regularizer = orthogonality_loss(output['pre_logits'], self.prompt_statistics,
                                            task_id=self.current_task, reg=self.args.hide_reg)
            preserved = self.net.snapshot_inactive_prompts()
        # WTP masks all noncurrent classes; global labels retain their mapping.
        logits = output['logits'].clone()
        logits[:, :self.n_past_classes] = -float('inf')
        loss = self.loss(logits[:, :self.n_seen_classes], labels)
        if regularizer is not None:
            loss = loss + regularizer
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('HiDe-Prompt encountered a nonfinite loss')
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.hide_clip_grad, error_if_nonfinite=True)
        self.opt.step()
        if preserved is not None:
            self.net.restore_inactive_prompts(preserved)
        self._updates[self._phase] += 1
        return loss.item()

    def _statistics_loader(self, dataset):
        source = dataset.train_loader
        view = copy.copy(source.dataset)
        if isinstance(view, MammothDatasetWrapper):
            object.__setattr__(view, 'dataset', copy.copy(source.dataset.dataset))
        view.transform = dataset.TEST_TRANSFORM
        return DataLoader(view, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers, drop_last=False,
                          pin_memory=source.pin_memory)

    @contextmanager
    def _record_phase(self, phase):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.phase_history.append({'task_id': self.current_task, 'phase': phase,
                                       'wall_s': time.perf_counter() - start})

    @torch.no_grad()
    def _collect_statistics(self, dataset, branch):
        module = self.net.tii if branch == 'tii' else self.net.prompt
        module.eval()
        stats = self.tii_statistics if branch == 'tii' else self.prompt_statistics
        records = {class_id: [] for class_id in range(self.n_past_classes, self.n_seen_classes)}
        loader = self._statistics_loader(dataset)
        targets = torch.as_tensor(loader.dataset.targets, dtype=torch.long)
        if len(targets) != len(loader.dataset) or bool(((targets < self.n_past_classes) | (targets >= self.n_seen_classes)).any()):
            raise ValueError('HiDe-Prompt statistics may only access current-task training classes')
        indices = {class_id: torch.where(targets == class_id)[0].tolist() for class_id in records}
        if any(not values for values in indices.values()):
            raise ValueError('Every current class needs training samples for HiDe-Prompt statistics')
        # Match the author's per-class batches. In the retained prefix reshape,
        # batch position affects key/value selection, so extracting mixed-class
        # batches and grouping features afterwards changes the algorithm.
        # Within each class use stable native index order, pinned for reference.
        for class_id, positions in indices.items():
            class_loader = DataLoader(Subset(loader.dataset, positions), batch_size=loader.batch_size,
                                      shuffle=False, num_workers=loader.num_workers,
                                      drop_last=False, pin_memory=loader.pin_memory)
            for batch in class_loader:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                if not bool((labels == class_id).all()):
                    raise ValueError('HiDe-Prompt statistics class labels disagree with the loader targets')
                output = (self.net.tii_outputs(inputs) if branch == 'tii'
                          else self.net.prompt_outputs(inputs, task_id=self.current_task, train=True, prompt_momentum=0.))
                features = output['pre_logits'].detach().cpu()
                if not bool(torch.isfinite(features).all()):
                    raise FloatingPointError('HiDe-Prompt distribution features must be finite')
                records[class_id].append(features)
        # Validate every class before writing statistics, preserving prior tasks.
        if any(not chunks for chunks in records.values()):
            raise ValueError('Every current class needs training samples for HiDe-Prompt statistics')
        for class_id, chunks in records.items():
            stats.update_class(class_id, torch.cat(chunks), task_id=self.current_task)

    def _align(self, branch):
        self.net.set_phase(branch + '_alignment')
        module = self.net.tii if branch == 'tii' else self.net.prompt
        stats = self.tii_statistics if branch == 'tii' else self.prompt_statistics
        class_masks = [list(range(*self.dataset.get_offsets(task))) for task in range(self.current_task + 1)]
        lr = self.args.hide_ca_lr * (self.args.batch_size / 256. if branch == 'tii' else 1.)
        with self._record_phase(branch + '_alignment'):
            metrics = align_classifier(module, stats, class_masks, self.current_task, branch,
                                       batch_size=self.args.batch_size, epochs=self.args.hide_ca_epochs,
                                       lr=lr, weight_decay=self.args.optim_wd)
        self.alignment_history.append(metrics)

    def end_epoch(self, epoch, dataset):
        expected = self._stage_epoch + (self.args.hide_tii_epochs if self._phase == 'prompt' else 0)
        if epoch != expected or self._epoch_started is None:
            raise ValueError('HiDe-Prompt end_epoch disagrees with its active phase')
        self.phase_history.append({'task_id': self.current_task, 'phase': self._phase + '_train',
                                   'epoch': self._stage_epoch, 'wall_s': time.perf_counter() - self._epoch_started})
        self._epoch_started = None
        if epoch == self.args.hide_tii_epochs - 1:
            if self._updates['tii'] == 0:
                raise RuntimeError('HiDe-Prompt TII phase needs at least one optimizer update')
            with self._record_phase('tii_statistics'):
                self._collect_statistics(dataset, 'tii')
            self._align('tii')
            self.tii_ready.fill_(True)
            # Keep the native phase label TII until begin_epoch enters prompt.
            self.net.set_phase('eval')

    def end_task(self, dataset):
        if not bool(self.tii_ready) or self._phase != 'prompt' or self._stage_epoch != self.args.hide_prompt_epochs - 1:
            raise RuntimeError('HiDe-Prompt task completion requires both full training phases')
        if self._epoch_started is not None:
            raise RuntimeError('Finish the last prompt epoch before ending the task')
        if self._updates['prompt'] == 0:
            raise RuntimeError('HiDe-Prompt prompt phase needs at least one optimizer update')
        with self._record_phase('prompt_finalize'):
            self.net.finalize_task()
        with self._record_phase('prompt_statistics'):
            self._collect_statistics(dataset, 'prompt')
        self._align('prompt')
        self.prompt_ready.fill_(True)
        self._phase = 'eval'
        self.net.set_phase('eval')

    def forward(self, inputs):
        if not bool(self.prompt_ready):
            raise RuntimeError('HiDe-Prompt evaluates after current-task prompt statistics/alignment')
        return self.net.routed_outputs(inputs, self.n_seen_classes)['logits'][:, :self.n_seen_classes]

    def get_extra_state(self):
        return {'version': 1, 'phase': self._phase, 'stage_epoch': self._stage_epoch,
                'current_task': self._current_task,
                'class_counters': {name: getattr(self, name, None) for name in
                                   ('_n_classes_current_task', '_n_past_classes', '_n_seen_classes', '_n_remaining_classes')},
                'updates': self._updates, 'phase_history': self.phase_history,
                'alignment_history': self.alignment_history}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get('version') != 1 or state.get('phase') not in ('tii', 'prompt', 'eval'):
            raise ValueError('Unsupported HiDe-Prompt checkpoint lifecycle state')
        self._phase = state['phase']
        self._stage_epoch = state['stage_epoch']
        self._current_task = state['current_task']
        for name, value in state['class_counters'].items():
            if value is not None:
                setattr(self, name, value)
        self._updates = state['updates']
        self.phase_history = state['phase_history']
        self.alignment_history = state['alignment_history']
        self._epoch_started = None
