"""HiDe-Prompt's two independent branches and task-indexed prompt lifecycle."""

import math
from pathlib import Path

import torch
from torch import nn

from backbone import MammothBackbone

from models.hide_prompt_utils.vit import HideVisionTransformer


HIDE_VIT_SIZES = {'base': (768, 12), 'small': (384, 6), 'tiny': (192, 3)}


class HideNetwork(MammothBackbone):
    def __init__(self, tii, prompt, n_tasks, prompt_momentum=0.01):
        super().__init__()
        if tii.num_features != prompt.num_features or tii.num_classes != prompt.num_classes:
            raise ValueError('HiDe-Prompt TII and prompt branches must have matching dimensions')
        if prompt.e_prompt is None or prompt.e_prompt.pool_size < n_tasks or n_tasks < 1:
            raise ValueError('HiDe-Prompt requires at least one distinct prefix per task')
        if not math.isfinite(prompt_momentum) or not 0 <= prompt_momentum <= 1:
            raise ValueError('HiDe-Prompt prompt momentum must be in [0, 1]')
        self.tii = tii
        self.prompt = prompt
        self.feature_dim = self.num_features = self.out_dim = tii.num_features
        self.num_classes = tii.num_classes
        self.n_tasks = self.total_tasks = n_tasks
        self.pool_size = prompt.e_prompt.pool_size
        self.prompt_momentum = prompt_momentum
        self.register_buffer('class_to_task', torch.full((self.num_classes,), -1, dtype=torch.long))
        self.register_buffer('task_offsets', torch.full((n_tasks, 2), -1, dtype=torch.long))
        self.register_buffer('active_task', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('task_finalized', torch.tensor(True))
        self.phase = 'eval'
        self.set_phase('eval')

    @property
    def seen_classes(self):
        task = int(self.active_task)
        return int(self.task_offsets[task, 1]) if task >= 0 else 0

    @torch.no_grad()
    def begin_task(self, past_classes, seen_classes):
        if not bool(self.task_finalized):
            raise RuntimeError('Finalize the active HiDe-Prompt task before beginning the next')
        task = int(self.active_task) + 1
        if task >= self.n_tasks or past_classes != self.seen_classes or not past_classes < seen_classes <= self.num_classes:
            raise ValueError('HiDe-Prompt tasks must add a nonempty contiguous class interval')
        if task > 0:
            self.prompt.e_prompt.prompt[:, :, task].copy_(self.prompt.e_prompt.prompt[:, :, task - 1])
        self.task_offsets[task] = torch.tensor((past_classes, seen_classes), device=self.task_offsets.device)
        self.class_to_task[past_classes:seen_classes] = task
        self.active_task.fill_(task)
        self.task_finalized.fill_(False)
        return task

    def phase_parameters(self, phase):
        if phase in ('tii', 'tii_alignment'):
            return list(self.tii.mlp.parameters()) + list(self.tii.head.parameters())
        if phase == 'prompt':
            return [self.prompt.e_prompt.prompt, *self.prompt.head.parameters()]
        if phase == 'prompt_alignment':
            return list(self.prompt.head.parameters())
        if phase == 'eval':
            return []
        raise ValueError(f'Unknown HiDe-Prompt phase: {phase}')

    def set_phase(self, phase):
        parameters = self.phase_parameters(phase)
        self.requires_grad_(False)
        for parameter in parameters:
            parameter.requires_grad_(True)
        self.tii.train(phase == 'tii')
        self.prompt.train(phase == 'prompt')
        self.phase = phase
        return parameters

    def tii_outputs(self, x, fc_only=False):
        return self.tii(x, fc_only=fc_only)

    def prompt_outputs(self, x, task_id=None, prompt_id=None, train=False,
                       fc_only=False, prompt_momentum=None):
        if task_id is None:
            task_id = int(self.active_task)
        momentum = self.prompt_momentum if prompt_momentum is None else prompt_momentum
        return self.prompt(x, task_id=task_id, prompt_id=prompt_id, train=train,
                           fc_only=fc_only, prompt_momentum=momentum)

    @torch.no_grad()
    def infer_prompt_ids(self, tii_logits, seen_classes=None):
        seen = self.seen_classes if seen_classes is None else seen_classes
        if tii_logits.ndim != 2 or tii_logits.shape[1] != self.num_classes:
            raise ValueError('HiDe-Prompt task routing requires full TII class logits')
        if seen < 1 or seen > self.seen_classes:
            raise ValueError('HiDe-Prompt cannot route through unseen classes')
        predictions = tii_logits[:, :seen].argmax(dim=1)
        tasks = self.class_to_task[predictions]
        if bool((tasks < 0).any()):
            raise RuntimeError('HiDe-Prompt predicted classes have no registered task')
        return tasks.unsqueeze(1)

    def routed_outputs(self, x, seen_classes=None):
        with torch.no_grad():
            self.tii.eval()
            tii = self.tii_outputs(x)
            prompt_ids = self.infer_prompt_ids(tii['logits'], seen_classes)
        result = self.prompt_outputs(x, prompt_id=prompt_ids, train=False)
        result['tii_logits'] = tii['logits']
        return result

    def forward(self, x, returnt='out'):
        result = self.routed_outputs(x)
        if returnt == 'out':
            return result['logits']
        if returnt == 'features':
            return result['pre_logits']
        if returnt == 'both':
            return result['logits'], result['pre_logits']
        if returnt in ('full', 'all'):
            return result
        raise ValueError(f'Unsupported HiDe-Prompt return type: {returnt}')

    def snapshot_inactive_prompts(self):
        if int(self.active_task) < 0:
            raise RuntimeError('Begin a HiDe-Prompt task before taking a prompt snapshot')
        return self.prompt.e_prompt.prompt.detach().clone()

    @torch.no_grad()
    def restore_inactive_prompts(self, snapshot):
        bank = self.prompt.e_prompt.prompt
        if snapshot.shape != bank.shape:
            raise ValueError('HiDe-Prompt preserved prompt snapshot has invalid shape')
        task = int(self.active_task)
        bank[:, :, :task].copy_(snapshot[:, :, :task])
        bank[:, :, task + 1:].copy_(snapshot[:, :, task + 1:])

    @torch.no_grad()
    def finalize_task(self):
        if int(self.active_task) < 0 or bool(self.task_finalized):
            raise RuntimeError('HiDe-Prompt finalization requires one active unfinalized task')
        task = int(self.active_task)
        bank = self.prompt.e_prompt.prompt
        if task > 0 and self.prompt_momentum > 0:
            previous = bank[:, :, :task].detach().clone().mean(dim=2)
            bank[:, :, task].copy_((1 - self.prompt_momentum) * bank[:, :, task].detach().clone()
                                  + self.prompt_momentum * previous)
        self.task_finalized.fill_(True)
        self.set_phase('eval')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Task identities are checkpoint state, not an evaluation-time oracle.
        mapping = state_dict.get(prefix + 'class_to_task')
        offsets = state_dict.get(prefix + 'task_offsets')
        active = state_dict.get(prefix + 'active_task')
        finalized = state_dict.get(prefix + 'task_finalized')
        if (not isinstance(mapping, torch.Tensor) or mapping.shape != self.class_to_task.shape
                or mapping.dtype != torch.long or mapping.device.type == 'meta'
                or not isinstance(offsets, torch.Tensor) or offsets.shape != self.task_offsets.shape
                or offsets.dtype != torch.long or offsets.device.type == 'meta'
                or not isinstance(active, torch.Tensor) or active.shape != torch.Size([])
                or active.dtype != torch.long or active.device.type == 'meta'
                or not isinstance(finalized, torch.Tensor) or finalized.shape != torch.Size([])
                or finalized.dtype != torch.bool or finalized.device.type == 'meta'):
            raise ValueError('Invalid HiDe-Prompt checkpoint task routing tensors')
        task = int(active)
        if not -1 <= task < self.n_tasks:
            raise ValueError('HiDe-Prompt checkpoint active task is outside the prompt bank')
        expected_map = torch.full_like(mapping, -1)
        stop = 0
        for index in range(task + 1):
            start, end = offsets[index].detach().cpu().tolist()
            if start != stop or not start < end <= self.num_classes:
                raise ValueError('HiDe-Prompt checkpoint class intervals are inconsistent')
            expected_map[start:end] = index
            stop = end
        if not torch.equal(mapping, expected_map) or bool((offsets[task + 1:] != -1).any()):
            raise ValueError('HiDe-Prompt checkpoint routing exposes future or inconsistent tasks')
        if task == -1 and not bool(finalized):
            raise ValueError('An empty HiDe-Prompt checkpoint cannot have an active task')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def get_extra_state(self):
        return {'version': 1, 'feature_dim': self.feature_dim, 'num_classes': self.num_classes,
                'n_tasks': self.n_tasks, 'prompt_momentum': self.prompt_momentum,
                'prompt_length': self.prompt.e_prompt.length,
                'prompt_layers': self.prompt.prompt_layers,
                'use_permute_fix': self.prompt.e_prompt.use_permute_fix,
                'phase': self.phase}

    def set_extra_state(self, state):
        expected = self.get_extra_state()
        if not isinstance(state, dict) or any(state.get(key) != value for key, value in expected.items() if key != 'phase'):
            raise ValueError('HiDe-Prompt checkpoint configuration does not match the constructed network')
        self.set_phase(state.get('phase'))


def build_hide_network(args, num_classes, n_tasks):
    """Build matched ViT/16 branches from explicitly supplied local weights."""
    size = getattr(args, 'hide_vit_size', 'base')
    if size not in HIDE_VIT_SIZES:
        raise ValueError('HiDe-Prompt supports only base, small and tiny ViT/16 sizes')
    embed_dim, num_heads = HIDE_VIT_SIZES[size]
    checkpoint = getattr(args, 'hide_checkpoint', None)
    if checkpoint is None or not Path(checkpoint).expanduser().is_file():
        raise FileNotFoundError('Set hide_checkpoint to the shared local AugReg IN21K->IN1K .npz file')
    get = lambda name, alias, default: getattr(args, name, getattr(args, alias, default))
    length = get('hide_prompt_length', 'length', 5)
    layers = tuple(get('hide_prompt_layers', 'e_prompt_layer_idx', (0, 1, 2, 3, 4)))
    hidden = tuple(get('hide_mlp_structure', 'original_model_mlp_structure', (2,)))
    drop = get('hide_drop', 'drop', 0.)
    drop_path = get('hide_drop_path', 'drop_path', 0.)
    geometry = dict(embed_dim=embed_dim, num_heads=num_heads, depth=12, img_size=224, patch_size=16)
    tii = HideVisionTransformer(num_classes, **geometry, mlp_structure=hidden, drop=drop, drop_path=drop_path)
    prompt = HideVisionTransformer(num_classes, **geometry, pool_size=n_tasks, prompt_length=length,
                                   prompt_layers=layers, drop=drop, drop_path=drop_path,
                                   use_permute_fix=get('hide_use_permute_fix', 'use_permute_fix', False))
    tii.load_pretrained(checkpoint)
    prompt.load_pretrained(checkpoint)
    network = HideNetwork(tii, prompt, n_tasks, get('hide_prompt_momentum', 'prompt_momentum', 0.01))
    network.pretrained_checkpoint = str(Path(checkpoint).expanduser().resolve())
    network.pretrained_model = f'vit_{size}_patch16_224.augreg_in21k_ft_in1k'
    return network
