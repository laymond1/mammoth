"""RainbowPrompt official active path (0b84d3e): no learned gate or sparsity loss.

Uses cached pre-update prompts and batch-mean key routing. The paper's gate is
a separate, deferred extension; this port does not claim paper equivalence.
"""

import math

import torch

from models.rainbow_prompt_utils.network import RainbowNetwork
from models.utils.continual_model import ContinualModel


class RainbowPrompt(ContinualModel):
    NAME = 'rainbow-prompt'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser):
        parser.set_defaults(backbone='rainbow-vit', dataset_config='ease', optimizer='adam',
                            lr=0.03, optim_wd=0., batch_size=64, n_epochs=20,
                            lr_scheduler=None, enable_other_metrics=False)
        parser.add_argument('--rainbow_checkpoint', type=str, default=None)
        parser.add_argument('--rainbow_vit_size', choices=['base', 'small', 'tiny'], default='base')
        parser.add_argument('--rainbow_pool_size', type=int, default=10)
        parser.add_argument('--rainbow_prompt_length', type=int, default=20,
                            help='Total prefix length: half K, half V')
        parser.add_argument('--rainbow_prompt_layers', type=int, nargs='+', default=list(range(12)))
        parser.add_argument('--rainbow_self_attn_layers', type=int, nargs='+', default=list(range(6)))
        parser.add_argument('--rainbow_d1', type=int, default=56)
        parser.add_argument('--rainbow_d2', type=int, default=96)
        parser.add_argument('--rainbow_ki_iter', type=int, default=10)
        parser.add_argument('--rainbow_warmup_epochs', type=int, default=5)
        parser.add_argument('--rainbow_balance', type=float, default=0.01)
        parser.add_argument('--rainbow_clip_grad', type=float, default=1.)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if not isinstance(backbone, RainbowNetwork):
            raise ValueError('RainbowPrompt requires --backbone rainbow-vit')
        if dataset is None or dataset.SETTING != 'class-il' or dataset.N_TASKS > backbone.total_tasks:
            raise ValueError('RainbowPrompt requires a class-incremental dataset within its task capacity')
        if not getattr(backbone, 'pretrained_checkpoint', None):
            raise ValueError('RainbowPrompt requires an explicit pretrained local --rainbow_checkpoint')
        self._validate_protocol(args)
        if args.rainbow_warmup_epochs != backbone.prompt.warmup:
            raise ValueError('RainbowPrompt backbone and method warmup settings disagree')
        super().__init__(backbone, loss, args, transform, dataset=dataset)

    @staticmethod
    def _validate_protocol(args):
        if args.fitting_mode != 'epochs' or args.eval_epochs is not None or args.enable_other_metrics:
            raise ValueError('RainbowPrompt requires epoch fitting and task-end evaluation without other metrics')
        if args.lr_scheduler is not None or args.optimizer.lower() != 'adam' or args.optim_wd != 0:
            raise ValueError('RainbowPrompt official active path requires constant Adam with weight decay zero')
        if getattr(args, 'start_from', None) not in (None, 0):
            raise ValueError('RainbowPrompt cannot use start_from to finalize untrained zero-cache tasks')
        if getattr(args, 'loadcheck', None) is not None or getattr(args, 'inference_only', False):
            raise ValueError('RainbowPrompt framework checkpoint stream resume/inference is not supported in this stage')
        if args.rainbow_warmup_epochs < 0 or args.n_epochs <= args.rainbow_warmup_epochs:
            raise ValueError('n_epochs must exceed rainbow_warmup_epochs to populate the Rainbow evaluation cache')
        if (not math.isfinite(args.lr) or args.lr <= 0 or args.batch_size < 1
                or not math.isfinite(args.rainbow_clip_grad) or args.rainbow_clip_grad <= 0
                or not math.isfinite(args.rainbow_balance) or args.rainbow_balance < 0):
            raise ValueError('RainbowPrompt LR/batch/clipping must be positive and balance finite/nonnegative')

    def get_parameters(self):
        return [parameter for parameter in self.net.prompt.parameters() if parameter.requires_grad]

    def get_optimizer(self, params=None, lr=None):
        # Keep args.lr nominal. Scale once whenever Mammoth recreates the optimizer.
        rate = self.args.lr * self.args.batch_size / 256. if lr is None else lr
        return torch.optim.Adam(self.get_parameters() if params is None else params,
                                lr=rate, weight_decay=0., eps=1e-8, betas=(0.9, 0.999))

    def begin_task(self, dataset):
        task = self.net.begin_task(self.n_past_classes, self.n_seen_classes)
        if task != self.current_task:
            raise RuntimeError('RainbowPrompt task state disagrees with Mammoth task counter')
        self.net.query.eval()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if type(epoch) is not int or not 0 <= epoch < self.args.n_epochs:
            raise ValueError('RainbowPrompt observe requires a valid local epoch index')
        if bool(((labels < self.n_past_classes) | (labels >= self.n_seen_classes)).any()):
            raise ValueError('RainbowPrompt training accepts current-task classes only')
        output = self.net.outputs(inputs, train=True, epoch=epoch)
        logits = output['logits'].clone()
        logits[:, :self.n_past_classes] = -float('inf')
        loss_ce = self.loss(logits[:, :self.n_seen_classes], labels)
        loss = loss_ce - self.args.rainbow_balance * output['sim_loss']
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite RainbowPrompt loss')
        self.opt.zero_grad(set_to_none=True)
        # Keep the author's active backward setting for resource comparisons.
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.net.prompt.parameters(), self.args.rainbow_clip_grad)
        self.opt.step()
        return loss.item()

    def end_task(self, dataset):
        self.net.finalize_task()

    def forward(self, x):
        # Explicit Mammoth adaptation: seen classes only, including in the future
        # independent author comparison. Routing never receives a test task label.
        return self.net.outputs(x)['logits'][:, :self.net.seen_classes]

    def get_extra_state(self):
        names = ('_n_classes_current_task', '_n_past_classes', '_n_seen_classes', '_n_remaining_classes')
        return {'version': 1, 'current_task': self.current_task,
                'class_counters': {name: getattr(self, name, None) for name in names}}

    def set_extra_state(self, state):
        names = ('_n_classes_current_task', '_n_past_classes', '_n_seen_classes', '_n_remaining_classes')
        if (not isinstance(state, dict) or state.get('version') != 1
                or type(state.get('current_task')) is not int or not 0 <= state['current_task'] <= self.n_tasks
                or not isinstance(state.get('class_counters'), dict) or set(state['class_counters']) != set(names)
                or any(value is not None and (type(value) is not int or not 0 <= value <= self.num_classes)
                       for value in state['class_counters'].values())):
            raise ValueError('Invalid RainbowPrompt checkpoint lifecycle state')
        self._current_task = state['current_task']
        for name, value in state['class_counters'].items():
            if value is not None:
                setattr(self, name, value)
