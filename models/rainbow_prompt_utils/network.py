"""Frozen query/CLS transformer and task state for the active RainbowPrompt path."""

from pathlib import Path

import torch

from backbone import MammothBackbone
from models.rainbow_prompt_utils.vit import RainbowVisionTransformer


RAINBOW_VIT_SIZES = {'base': (768, 12), 'small': (384, 6), 'tiny': (192, 3)}
SOURCE_COMMIT = '0b84d3e3738669fe4bb516ccc1c526a535576ad3'


class RainbowNetwork(MammothBackbone):
    def __init__(self, query, prompt, n_tasks):
        super().__init__()
        if (query.num_features != prompt.num_features or query.num_classes != prompt.num_classes
                or not hasattr(prompt, 'rainbow_prompt') or type(n_tasks) is not int
                or not 1 <= n_tasks <= prompt.rainbow_prompt.n_tasks):
            raise ValueError('RainbowPrompt requires matching branches and sufficient task capacity')
        self.query, self.prompt = query, prompt
        self.num_features = self.feature_dim = self.out_dim = query.num_features
        self.num_classes, self.total_tasks = query.num_classes, n_tasks
        self.query.requires_grad_(False)
        # The author's freeze_components helper is never called by its trainer.
        # Shared prompt transformations stay trainable across all tasks.
        for name, parameter in self.prompt.named_parameters():
            parameter.requires_grad_(name.startswith(('rainbow_prompt.', 'head.')))
        self.register_buffer('active_task', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('task_offsets', torch.full((n_tasks, 2), -1, dtype=torch.long))
        self.register_buffer('task_finalized', torch.tensor(True))
        self.register_buffer('rainbow_ready', torch.zeros(n_tasks, dtype=torch.bool))
        self.query.eval()

    @property
    def seen_classes(self):
        task = int(self.active_task)
        return int(self.task_offsets[task, 1]) if task >= 0 else 0

    def train(self, mode=True):
        super().train(mode)
        self.query.eval()
        return self

    @torch.no_grad()
    def begin_task(self, past_classes, seen_classes):
        if not bool(self.task_finalized):
            raise RuntimeError('Finalize the active RainbowPrompt task before beginning the next')
        task = int(self.active_task) + 1
        if (task >= self.total_tasks or past_classes != self.seen_classes
                or not past_classes < seen_classes <= self.num_classes):
            raise ValueError('RainbowPrompt tasks must add a nonempty contiguous class interval')
        self.task_offsets[task] = torch.tensor((past_classes, seen_classes), device=self.task_offsets.device)
        self.active_task.fill_(task)
        self.task_finalized.fill_(False)
        return task

    def outputs(self, inputs, train=False, epoch=None):
        task = int(self.active_task)
        if task < 0:
            raise RuntimeError('Begin a RainbowPrompt task before forwarding samples')
        if train and bool(self.task_finalized):
            raise RuntimeError('Cannot train a finalized RainbowPrompt task')
        if not train and not bool(self.rainbow_ready[:task + 1].all()):
            raise RuntimeError('RainbowPrompt evaluation requires a cached Rainbow forward for every learned task')
        with torch.no_grad():
            self.query.eval()
            features = self.query(inputs)['pre_logits']
        result = self.prompt(inputs, task_id=task if train else -1, learned_id=task,
                             cls_features=features, train=train, epoch_info=epoch)
        if train and epoch >= self.prompt.warmup:
            self.rainbow_ready[task] = True
        return result

    def forward(self, x, returnt='out'):
        result = self.outputs(x)
        if returnt == 'out':
            return result['logits']
        if returnt == 'features':
            return result['pre_logits']
        if returnt == 'both':
            return result['logits'], result['pre_logits']
        if returnt in ('full', 'all'):
            return result
        raise ValueError(f'Unsupported RainbowPrompt return type: {returnt}')

    @torch.no_grad()
    def finalize_task(self):
        task = int(self.active_task)
        if task < 0 or bool(self.task_finalized) or not bool(self.rainbow_ready[task]):
            raise RuntimeError('RainbowPrompt finalization requires one active task with a stored Rainbow prompt')
        # Preserve the source cache written before the last optimizer step.
        self.task_finalized.fill_(True)

    def get_extra_state(self):
        prompt = self.prompt.rainbow_prompt
        return {'version': 1, 'source_commit': SOURCE_COMMIT, 'source_mode': 'official-active-path',
                'learned_gate_active': False, 'paper_equivalent': False,
                'feature_dim': self.feature_dim, 'num_classes': self.num_classes,
                'n_tasks': self.total_tasks, 'warmup': self.prompt.warmup,
                'length': prompt.length, 'D1': prompt.D1, 'D2': prompt.D2,
                'KI_iter': prompt.KI_iter, 'layers': list(prompt.prompt_tune_idx),
                'self_attn_layers': list(prompt.self_attn_idx)}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('RainbowPrompt checkpoint protocol differs from the current network')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        active = state_dict.get(prefix + 'active_task')
        offsets = state_dict.get(prefix + 'task_offsets')
        finalized = state_dict.get(prefix + 'task_finalized')
        ready = state_dict.get(prefix + 'rainbow_ready')
        expected = ((active, (), torch.long), (offsets, (self.total_tasks, 2), torch.long),
                    (finalized, (), torch.bool), (ready, (self.total_tasks,), torch.bool))
        if any(not isinstance(value, torch.Tensor) or value.shape != torch.Size(shape)
               or value.dtype != dtype or value.device.type == 'meta' for value, shape, dtype in expected):
            raise ValueError('Invalid RainbowPrompt checkpoint task tensors')
        task = int(active)
        if not -1 <= task < self.total_tasks or bool(ready[task + 1:].any()):
            raise ValueError('RainbowPrompt checkpoint exposes future tasks')
        stop = 0
        for index in range(task + 1):
            start, end = offsets[index].tolist()
            if start != stop or not start < end <= self.num_classes:
                raise ValueError('RainbowPrompt checkpoint class intervals are inconsistent')
            stop = end
        if (bool((offsets[task + 1:] != -1).any()) or task == -1 and not bool(finalized)
                or task >= 0 and bool(finalized) and not bool(ready[:task + 1].all())):
            raise ValueError('RainbowPrompt checkpoint cache/task status is inconsistent')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


def build_rainbow_network(args, num_classes, n_tasks):
    checkpoint = getattr(args, 'rainbow_checkpoint', None)
    if not checkpoint or not Path(checkpoint).is_file():
        raise ValueError('RainbowPrompt requires an existing local --rainbow_checkpoint NPZ; no download is attempted')
    size = getattr(args, 'rainbow_vit_size', 'base')
    if size not in RAINBOW_VIT_SIZES:
        raise ValueError('rainbow_vit_size must be base, small or tiny')
    width, heads = RAINBOW_VIT_SIZES[size]
    common = dict(num_classes=num_classes, embed_dim=width, num_heads=heads)
    query = RainbowVisionTransformer(**common)
    prompt = RainbowVisionTransformer(**common, pool_size=args.rainbow_pool_size, n_tasks=n_tasks,
                                      prompt_length=args.rainbow_prompt_length,
                                      prompt_layers=args.rainbow_prompt_layers,
                                      D1=args.rainbow_d1, D2=args.rainbow_d2,
                                      warmup=args.rainbow_warmup_epochs, KI_iter=args.rainbow_ki_iter,
                                      self_attn_idx=args.rainbow_self_attn_layers)
    query.load_pretrained(checkpoint)
    prompt.load_pretrained(checkpoint)
    network = RainbowNetwork(query, prompt, n_tasks)
    network.pretrained_checkpoint = str(Path(checkpoint).resolve())
    return network
