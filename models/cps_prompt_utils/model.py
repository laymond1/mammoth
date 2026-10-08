"""Frozen full-query encoder, sparse joint phase and dense classifier phase."""

import torch
from torch import nn

from models.resource_prompt_utils import LayeredCodaPrompt, build_encoder

from .cps import CriticalPatchSampling
from .vit import CPSVisionTransformer


class PromptModel(nn.Module):
    def __init__(self, num_classes=100, n_tasks=10, vit_size='tiny', checkpoint=None,
                 pool_size=100, prompt_length=8, ortho_mu=0., layers=(0, 1, 2, 3, 4),
                 reduction_ratio=.4, temperature=.1, sampling='critical_score', seed=1993,
                 pretrained=True, geometry=None):
        super().__init__()
        if pool_size < n_tasks or pool_size % n_tasks or prompt_length <= 0 or prompt_length % 2:
            raise ValueError('CPS requires a task-divisible pool and a positive even prefix length')
        self.feat = build_encoder(vit_size, checkpoint, encoder_class=CPSVisionTransformer,
                                  pretrained=pretrained, geometry=geometry)
        if not layers or len(set(layers)) != len(layers) or any(not 0 <= layer < len(self.feat.blocks) for layer in layers):
            raise ValueError('Prompt layers must be unique valid block indices')
        self.head = nn.Linear(self.feat.embed_dim, num_classes)
        self.prompt = LayeredCodaPrompt(self.feat.embed_dim, n_tasks, [pool_size, prompt_length, ortho_mu], layers)
        self.sampler = CriticalPatchSampling(reduction_ratio, sampling, temperature, seed)
        self.diagnostics = {}

    def set_task(self, task):
        if not 0 <= task < self.prompt.n_tasks:
            raise ValueError('CPS task is outside the original task allocation')
        if task == self.prompt.task_count + 1:
            self.prompt.process_task_count()
        elif task != self.prompt.task_count:
            raise ValueError('CPS task transitions must be sequential')

    def get_extra_state(self):
        return {'task_count': self.prompt.task_count}

    def set_extra_state(self, state):
        task = state['task_count']
        if type(task) is not int or not 0 <= task < self.prompt.n_tasks:
            raise ValueError('Invalid saved CPS task')
        self.prompt.task_count = task

    def forward(self, x, train=False, feat=False, last=False):
        if last:
            return self.head(x)
        with torch.no_grad():
            query, _, scores, query_count = self.feat(x)
            query = query[:, 0]
        tokens, penalty, _, prompted_count = self.feat(x, prompt=self.prompt, q=query, train=train,
                                                     sampler=self.sampler, scores=scores)
        self.diagnostics = {'query_tokens': query_count, 'prompted_tokens': prompted_count,
                            'query_image_visits': x.shape[0], 'prompted_image_visits': x.shape[0]}
        features = tokens[:, 0]
        if feat:
            return features
        logits = self.head(features)
        return (logits, penalty) if train else logits
