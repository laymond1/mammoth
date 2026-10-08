"""REP-CODA with a frozen Tiny query, fixed projection and persistent ALD state."""

import math

import torch
from torch import nn

from models.resource_prompt_utils import CheckpointRNG, LayeredCodaPrompt, build_encoder

from .vit import REPVisionTransformer


class REPModel(nn.Module):
    def __init__(self, num_classes=100, n_tasks=10, vit_size='tiny', checkpoint=None,
                 query_checkpoint=None, pool_size=100, prompt_length=8, ortho_mu=0.,
                 swift=True, atom=True, ald=True, maximum=16, minimum=.5, gamma=.001,
                 alpha=.9, threshold=None, seed=1993, pretrained=True, geometry=None, query_geometry=None):
        super().__init__()
        if vit_size not in ('tiny', 'small', 'base'):
            raise ValueError('REP vit_size must be tiny, small or base')
        if pool_size < n_tasks or pool_size % n_tasks or prompt_length <= 0 or prompt_length % 2:
            raise ValueError('REP-CODA requires a task-divisible pool and a positive even prefix length')
        if type(maximum) is not int or maximum < 0:
            raise ValueError('rep_merge_max must be a nonnegative integer')
        if not all(math.isfinite(value) for value in (minimum, gamma, alpha)) or not 0 < minimum <= 1 or gamma < 0 or not 0 < alpha <= 1:
            raise ValueError('Invalid REP ALD probability or decay parameters')
        threshold = {'tiny': 8, 'small': 10, 'base': 12}[vit_size] if threshold is None else threshold
        if type(threshold) is not int or threshold <= 0:
            raise ValueError('REP spatial threshold must be a positive integer')
        self.feat = build_encoder(vit_size, checkpoint, encoder_class=REPVisionTransformer,
                                  pretrained=pretrained, geometry=geometry)
        self.head = nn.Linear(self.feat.embed_dim, num_classes)
        self.prompt = LayeredCodaPrompt(self.feat.embed_dim, n_tasks, [pool_size, prompt_length, ortho_mu])
        self.swift, self.atom, self.ald = bool(swift), bool(atom), bool(ald)
        self.maximum, self.minimum, self.gamma, self.alpha, self.threshold = maximum, minimum, gamma, alpha, threshold
        self.rng = CheckpointRNG(seed)
        self.query_encoder = None
        self.register_buffer('projection', None)
        if self.swift:
            self.query_encoder = build_encoder('tiny', query_checkpoint, pretrained=pretrained, geometry=query_geometry)
            generator = torch.Generator(device='cpu').manual_seed(int(seed) + 1)
            projection = torch.randn(self.query_encoder.embed_dim, self.feat.embed_dim, generator=generator)
            self.projection = projection / math.sqrt(self.query_encoder.embed_dim)
        self.step = 0
        self.active_task = None
        self.diagnostics = {}

    def set_task(self, task):
        if not 0 <= task < self.prompt.n_tasks:
            raise ValueError('REP task is outside the original task allocation')
        if task == self.prompt.task_count + 1:
            self.prompt.process_task_count()
        elif task != self.prompt.task_count:
            raise ValueError('REP task transitions must be sequential')
        if self.active_task != task:
            self.step = 0
            self.active_task = task

    def get_extra_state(self):
        return {'task_count': self.prompt.task_count, 'step': self.step, 'active_task': self.active_task}

    def set_extra_state(self, state):
        if type(state['task_count']) is not int or not 0 <= state['task_count'] < self.prompt.n_tasks:
            raise ValueError('Invalid saved REP task')
        if type(state['step']) is not int or state['step'] < 0:
            raise ValueError('Invalid saved REP update step')
        active = state['active_task']
        if active is not None and (type(active) is not int or active != state['task_count']):
            raise ValueError('Saved REP active task must match its prompt allocation')
        if active is None and (state['step'] != 0 or state['task_count'] != 0):
            raise ValueError('Unstarted REP checkpoint cannot contain learned task state')
        self.prompt.task_count = state['task_count']
        self.step, self.active_task = state['step'], active

    def forward(self, images, train=False):
        with torch.no_grad():
            if self.swift:
                query_tokens, _ = self.query_encoder(images)
                query = query_tokens[:, 0] @ self.projection
            else:
                # The original host query has no merging or stochastic dropping.
                query_tokens = self._host_query(images)
                query = query_tokens[:, 0]
        tokens, penalty, trace = self.feat(images, self.prompt, query, train=train,
                                         atom=self.atom, ald=self.ald, maximum=self.maximum,
                                         step=self.step, rng=self.rng, minimum=self.minimum,
                                         gamma=self.gamma, alpha=self.alpha, threshold=self.threshold)
        trace.update(query_tokens=query_tokens.shape[1], query_dimension=query.shape[-1],
                     query_image_visits=images.shape[0], prompted_image_visits=images.shape[0])
        self.diagnostics = trace
        logits = self.head(tokens[:, 0])
        return (logits, penalty) if train else logits

    def _host_query(self, images):
        x = self.feat.patch_embed(images)
        x = torch.cat((self.feat.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.feat.pos_drop(x + self.feat.pos_embed[:, :x.shape[1]])
        for block in self.feat.blocks:
            attention, _ = block.attn(block.norm1(x))
            x = x + block.drop_path(attention)
            x = x + block.drop_path(block.mlp(block.norm2(x)))
        return self.feat.norm(x)
