"""PROL's independent LayerNorm + linear head for each learned task."""

# Adapted from anwarmaxsum/PROL, linears.py, bfff8418 (MIT, Copyright (c) 2025).
# See accompanying LICENSE. Source cites hshustc/CVPR19_Incremental_Learning.

import torch
from torch import nn
from timm.layers import trunc_normal_


class SimpleContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes, feat_expand=False, with_norm=True):
        super().__init__()
        if embed_dim < 1 or nb_classes < 1 or feat_expand or not with_norm:
            raise ValueError('PROL requires a shared feature and normalized task heads')
        self.embed_dim, self.feat_expand, self.with_norm = embed_dim, feat_expand, with_norm
        self.classes_per_head = nb_classes
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, nb_classes, bias=True))])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=.02)
                nn.init.constant_(module.bias, 0)

    def update(self, nb_classes, freeze_old=True):
        if nb_classes != self.classes_per_head or not freeze_old:
            raise ValueError('PROL requires equal task widths and frozen previous heads')
        device = self.heads[0][1].weight.device
        dtype = self.heads[0][1].weight.dtype
        # The author creates on CPU then calls .cuda(), then trunc_normal_ on
        # the target device. Preserve that RNG order while allowing CPU tests.
        norm = nn.LayerNorm(self.embed_dim).to(device=device, dtype=dtype)
        linear = nn.Linear(self.embed_dim, nb_classes, bias=True).to(device=device, dtype=dtype)
        trunc_normal_(linear.weight, std=.02)
        nn.init.constant_(linear.bias, 0)
        self.heads.requires_grad_(False)
        self.heads.append(nn.Sequential(norm, linear))

    def forward(self, x):
        return {'logits': torch.cat([head(x) for head in self.heads], dim=1)}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        fully_frozen = not any(parameter.requires_grad for parameter in self.parameters())
        indices = {int(name[len(prefix + 'heads.'):].split('.')[0]) for name in state_dict
                   if name.startswith(prefix + 'heads.') and name.endswith('.1.weight')}
        if indices != set(range(len(indices))) or not indices:
            raise ValueError('Invalid PROL checkpoint task head indices')
        if len(indices) < len(self.heads):
            self.heads = nn.ModuleList(list(self.heads)[:len(indices)])
        # Restoring structure must not advance CPU/GPU experiment RNG streams.
        devices = [self.heads[0][1].weight.device.index] if self.heads[0][1].weight.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            while len(self.heads) < len(indices):
                self.update(self.classes_per_head)
        for index, head in enumerate(self.heads):
            head.requires_grad_(not fully_frozen and index == len(self.heads) - 1)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
