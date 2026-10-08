"""PROL's frozen query, prompted transformer and sequential task lifecycle."""

from pathlib import Path

import torch

from backbone import MammothBackbone
from models.prol_prompt_utils.vit import ProlVisionTransformer


SOURCE_COMMIT = 'bfff8418a4f603a24ae578f1e108bfac89af1e18'
PROL_VIT_SIZES = {'base': (768, 12), 'small': (384, 6), 'tiny': (192, 3)}


class ProlNetwork(MammothBackbone):
    def __init__(self, query, prompt, n_tasks):
        super().__init__()
        if (query.num_features != prompt.num_features or query.num_classes != prompt.num_classes
                or query.use_e_prompt or not prompt.use_e_prompt or n_tasks != prompt.num_tasks):
            raise ValueError('PROL requires matching frozen-query and prompted branches')
        self.query, self.prompt = query, prompt
        self.num_features = self.feature_dim = self.out_dim = query.num_features
        self.num_classes, self.total_tasks = query.num_classes, n_tasks
        self.query.requires_grad_(False)
        for name, parameter in self.prompt.named_parameters():
            frozen_prefix = name.startswith(('blocks', 'patch_embed', 'cls_token', 'pos_embed'))
            parameter.requires_grad_(not frozen_prefix or 'norm1' in name or 'norm2' in name)
        self.register_buffer('active_task', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('task_offsets', torch.full((n_tasks, 2), -1, dtype=torch.long))
        self.register_buffer('task_finalized', torch.tensor(True))
        self.register_buffer('task_updates', torch.zeros(n_tasks, dtype=torch.long))

    @property
    def prompted(self):
        return self.prompt

    @property
    def seen_classes(self):
        task = int(self.active_task)
        return int(self.task_offsets[task, 1]) if task >= 0 else 0

    def train(self, mode=True):
        super().train(mode)
        # The author never calls original_model.eval(), even during evaluation.
        # All approved dropout and stochastic-depth probabilities are zero.
        self.query.train(True)
        return self

    @torch.no_grad()
    def begin_task(self, past_classes, seen_classes, process_prompt=True):
        if not bool(self.task_finalized):
            raise RuntimeError('Finalize the active PROL task before beginning another')
        task = int(self.active_task) + 1
        width = self.num_classes // self.total_tasks
        if (task >= self.total_tasks or past_classes != self.seen_classes
                or seen_classes - past_classes != width or seen_classes > self.num_classes):
            raise ValueError('PROL requires sequential equal-width contiguous class tasks')
        if task > 0:
            self.prompt.head.update(width)
        self.task_offsets[task] = torch.tensor((past_classes, seen_classes), device=self.task_offsets.device)
        self.active_task.fill_(task)
        self.task_finalized.fill_(False)
        if process_prompt:
            self.process_task_prompt()
        return task

    def process_task_prompt(self):
        task = int(self.active_task)
        if task < 0 or len(self.prompt.e_prompt.lmax_list) != task:
            raise RuntimeError('PROL prompt transition is missing or has already been processed')
        self.prompt.e_prompt.process_new_task(task, task + 1)

    def outputs(self, inputs, labels=None, train=False):
        task = int(self.active_task)
        if task < 0 or len(self.prompt.e_prompt.lmax_list) != task + 1:
            raise RuntimeError('Begin and initialize a PROL task before forwarding samples')
        if train and (bool(self.task_finalized) or labels is None):
            raise RuntimeError('PROL training requires labels and an unfinalized active task')
        self.prompt.train(train)
        self.query.train(True)
        with torch.no_grad():
            query_features = self.query(inputs)['pre_logits']
        result = self.prompt.forwardA1(inputs, labels if train else None, task_id=task if train else -1,
                                      cls_features=query_features, train=train)
        result['query_features'] = query_features
        return result

    @torch.no_grad()
    def record_update(self):
        task = int(self.active_task)
        if task < 0 or bool(self.task_finalized):
            raise RuntimeError('PROL update accounting requires an active task')
        self.task_updates[task] += 1

    @torch.no_grad()
    def finalize_task(self):
        task = int(self.active_task)
        if task < 0 or bool(self.task_finalized) or self.task_updates[task] < 1:
            raise RuntimeError('PROL task finalization requires at least one training update')
        self.task_finalized.fill_(True)
        self.prompt.eval()

    def forward(self, inputs, returnt='out'):
        output = self.outputs(inputs)
        if returnt == 'out':
            return output['logits']
        if returnt == 'features':
            return output['pre_logits']
        if returnt == 'both':
            return output['logits'], output['pre_logits']
        if returnt in ('full', 'all'):
            return output
        raise ValueError(f'Unsupported PROL return type: {returnt}')

    def get_extra_state(self):
        return {'version': 1, 'source_commit': SOURCE_COMMIT, 'source_mode': 'official-active-path',
                'paper_equivalent': False, 'feature_dim': self.feature_dim, 'num_classes': self.num_classes,
                'n_tasks': self.total_tasks, 'query_n_tasks': self.query.num_tasks,
                'length': self.prompt.e_prompt.length, 'layers': list(self.prompt.e_prompt_layer_idx),
                'top_k_l': self.prompt.e_prompt.top_k_l}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('PROL checkpoint protocol differs from the current network')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        active, offsets = state_dict.get(prefix + 'active_task'), state_dict.get(prefix + 'task_offsets')
        finalized, updates = state_dict.get(prefix + 'task_finalized'), state_dict.get(prefix + 'task_updates')
        expected = ((active, (), torch.long), (offsets, (self.total_tasks, 2), torch.long),
                    (finalized, (), torch.bool), (updates, (self.total_tasks,), torch.long))
        if any(not isinstance(value, torch.Tensor) or value.shape != torch.Size(shape)
               or value.dtype != dtype or value.device.type == 'meta' for value, shape, dtype in expected):
            raise ValueError('Invalid PROL checkpoint task tensors')
        task = int(active)
        width = self.num_classes // self.total_tasks
        if (not -1 <= task < self.total_tasks or bool((updates < 0).any())
                or bool((updates[task + 1:] != 0).any())
                or bool((offsets[task + 1:] != -1).any())
                or task == -1 and not bool(finalized)):
            raise ValueError('PROL checkpoint exposes inconsistent or future task state')
        for index in range(task + 1):
            if offsets[index].tolist() != [index * width, (index + 1) * width]:
                raise ValueError('PROL checkpoint task class intervals are inconsistent')
            if (index < task or bool(finalized)) and updates[index] < 1:
                raise ValueError('PROL checkpoint finalizes an untrained task')
        head_indices = {int(name[len(prefix + 'prompt.head.heads.'):].split('.')[0]) for name in state_dict
                        if name.startswith(prefix + 'prompt.head.heads.') and name.endswith('.1.weight')}
        query_head_indices = {int(name[len(prefix + 'query.head.heads.'):].split('.')[0]) for name in state_dict
                              if name.startswith(prefix + 'query.head.heads.') and name.endswith('.1.weight')}
        prompt_state = state_dict.get(prefix + 'prompt.e_prompt._extra_state', {})
        if (head_indices != set(range(max(1, task + 1))) or query_head_indices != {0}
                or not isinstance(prompt_state, dict) or prompt_state.get('new_num_k') != task + 1):
            raise ValueError('PROL checkpoint classifier/prompt task counts disagree')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


def build_prol_network(args, num_classes, n_tasks):
    checkpoint = getattr(args, 'prol_checkpoint', None)
    if not checkpoint or not Path(checkpoint).is_file() or Path(checkpoint).suffix != '.npz':
        raise ValueError('PROL requires an existing local --prol_checkpoint NPZ; no download is attempted')
    size = getattr(args, 'prol_vit_size', 'base')
    if size not in PROL_VIT_SIZES:
        raise ValueError('prol_vit_size must be base, small or tiny')
    if num_classes % 5:
        raise ValueError('PROL source query classifier requires a class count divisible by its default five tasks')
    width, heads = PROL_VIT_SIZES[size]
    common = dict(num_classes=num_classes, embed_dim=width, num_heads=heads)
    query = ProlVisionTransformer(**common, n_tasks=5, prompted=False)
    prompt = ProlVisionTransformer(**common, n_tasks=n_tasks, prompt_length=args.prol_prompt_length,
                                  prompt_layers=args.prol_prompt_layers, top_k_l=args.prol_top_k_l)
    query.load_pretrained(checkpoint)
    prompt.load_pretrained(checkpoint)
    network = ProlNetwork(query, prompt, n_tasks)
    network.pretrained_checkpoint = str(Path(checkpoint).resolve())
    return network
