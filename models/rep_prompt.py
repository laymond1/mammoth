"""REP-CODA: paper-based resource-efficient prompting with explicit assumptions."""

from typing import ClassVar

from datasets import get_dataset
from models.rep_prompt_utils.model import REPModel
from models.resource_prompt_utils.state import PromptStateMixin
from models.utils.continual_model import ContinualModel
from utils.schedulers import CosineSchedule


class REPPrompt(PromptStateMixin, ContinualModel):
    NAME = 'rep-prompt'
    COMPATIBILITY: ClassVar[list[str]] = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser):
        parser.set_defaults(optimizer='adam', lr=1e-3, n_epochs=20, batch_size=16)
        parser.add_argument('--rep_vit_size', choices=('tiny', 'small', 'base'), default='tiny')
        parser.add_argument('--rep_checkpoint', default=None)
        parser.add_argument('--rep_query_checkpoint', default=None, help='Pinned Tiny AugReg encoder used by swift query')
        parser.add_argument('--pool_size', type=int, default=100)
        parser.add_argument('--prompt_len', type=int, default=8)
        parser.add_argument('--ortho_mu', type=float, default=0.)
        parser.add_argument('--mu', type=float, default=0.)
        parser.add_argument('--rep_swift', type=int, choices=(0, 1), default=1)
        parser.add_argument('--rep_atom', type=int, choices=(0, 1), default=1)
        parser.add_argument('--rep_ald', type=int, choices=(0, 1), default=1)
        parser.add_argument('--rep_merge_max', type=int, default=16, help='Progressive per-block merge maximum; configurable reconstruction default')
        parser.add_argument('--rep_ald_minimum', type=float, default=.5)
        parser.add_argument('--rep_ald_gamma', type=float, default=.001, help='Decay per optimizer update within a task')
        parser.add_argument('--rep_ald_alpha', type=float, default=.9)
        parser.add_argument('--rep_ald_threshold', type=int, default=None, help='Defaults: Tiny8, Small10 adaptation, Base12')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        del backbone
        if args.lr_scheduler is not None:
            raise ValueError('REP-CODA owns its host cosine scheduler')
        dataset = dataset if dataset is not None else get_dataset(args)
        net = REPModel(num_classes=dataset.N_CLASSES, n_tasks=dataset.N_TASKS,
                       vit_size=args.rep_vit_size, checkpoint=args.rep_checkpoint,
                       query_checkpoint=args.rep_query_checkpoint, pool_size=args.pool_size,
                       prompt_length=args.prompt_len, ortho_mu=args.ortho_mu,
                       swift=args.rep_swift, atom=args.rep_atom, ald=args.rep_ald,
                       maximum=args.rep_merge_max, minimum=args.rep_ald_minimum,
                       gamma=args.rep_ald_gamma, alpha=args.rep_ald_alpha,
                       threshold=args.rep_ald_threshold, seed=getattr(args, 'seed', None) or 0)
        super().__init__(net, loss, args, transform, dataset)
        self._last_epoch = 0

    def get_parameters(self):
        return list(self.net.prompt.parameters()) + list(self.net.head.parameters())

    def begin_task(self, dataset):
        self.net.set_task(self.current_task)
        self.opt = self.get_optimizer()
        self.custom_scheduler = CosineSchedule(self.opt, K=self.args.n_epochs)
        self.scheduler = self.custom_scheduler

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self._last_epoch = 0 if epoch is None else epoch
        logits, penalty = self.net(inputs, train=True)
        logits = logits[:, :self.n_seen_classes]
        logits[:, :self.n_past_classes] = -float('inf')
        value = self.loss(logits, labels.long()) + self.args.mu * penalty.sum()
        self.opt.zero_grad(set_to_none=True)
        value.backward()
        self.opt.step()
        # Exactly one advance per optimizer update, never per layer/query/eval.
        self.net.step += 1
        trace = self.net.diagnostics
        return {'loss': value.item(), 'rep_step': trace['step'], 'rep_executed_blocks': sum(trace['executed']),
                    'rep_merged_tokens': sum(trace['merged_tokens']), 'rep_final_tokens': trace['output_tokens'][-1],
                    'query_image_visits': trace['query_image_visits'], 'prompted_image_visits': trace['prompted_image_visits']}

    def forward(self, x):
        return self.net(x)[:, :self.n_seen_classes]
