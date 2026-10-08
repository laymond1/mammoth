"""Source-shaped PROL transformer, with local-only AugReg encoder loading.

Active architecture and initialization follow author commit bfff8418. The
unprompted branch keeps its unused continual classifier, as it affects parameter
count, memory and initialization RNG. Neither branch downloads weights.
"""

# Adapted from anwarmaxsum/PROL, vision_transformerL.py and attention.py,
# bfff8418 (MIT, Copyright (c) 2025; see accompanying LICENSE).
# Original transformer implementation: Copyright 2020 Ross Wightman.
# Source attention modifications: Copyright (c) 2015-present Facebook, Inc.
# All rights reserved. DualPrompt modifications by Jaeho Lee, dlwogh9344@khu.ac.kr.

from pathlib import Path

import numpy as np
import torch
from torch import nn
from timm.layers import DropPath, Mlp, PatchEmbed, trunc_normal_

from models.prol_prompt_utils.head import SimpleContinualLinear
from models.prol_prompt_utils.prompt import ProlPrompt


class PrefixAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        if dim % num_heads:
            raise ValueError('PROL attention width must be divisible by head count')
        self.num_heads, self.scale = num_heads, (dim // num_heads) ** -.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt=None):
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, width // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        if prompt is not None:
            prefix = prompt.permute(1, 0, 3, 2, 4).contiguous()
            if prefix.shape != (2, batch, self.num_heads, prompt.shape[2], width // self.num_heads):
                raise ValueError('PROL prefix dimensions do not match its attention')
            key = torch.cat((prefix[0], key), dim=2)
            value = torch.cat((prefix[1], value), dim=2)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, width)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = PrefixAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.ls1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        self.ls2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, prompt=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), prompt)))
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))


class ProlVisionTransformer(nn.Module):
    def __init__(self, num_classes, img_size=224, patch_size=16, embed_dim=768, depth=12,
                 num_heads=12, n_tasks=10, prompted=True, prompt_length=5,
                 prompt_layers=tuple(range(7)), top_k_l=1, prompt_key_init='uniform',
                 drop=0., attn_drop=0., drop_path=0., mlp_ratio=4., qkv_bias=True):
        super().__init__()
        if (any(type(value) is not int or value < 1 for value in
                (num_classes, img_size, patch_size, embed_dim, depth, num_heads, n_tasks))
                or num_classes % n_tasks or embed_dim % num_heads or top_k_l != 1):
            raise ValueError('Invalid PROL transformer or equal-task class dimensions')
        layers = tuple(prompt_layers) if prompted else ()
        if prompted and (not layers or tuple(sorted(set(layers))) != layers
                         or any(type(index) is not int or not 0 <= index < depth for index in layers)):
            raise ValueError('PROL prefix layers must be unique increasing valid block indices')
        self.img_size, self.num_classes, self.num_tasks = img_size, num_classes, n_tasks
        self.num_features = self.embed_dim = embed_dim
        self.global_pool, self.head_type = 'token', 'token'
        self.class_token, self.num_prefix_tokens, self.no_embed_class = True, 1, False
        self.grad_checkpointing, self.use_multihead = False, True
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, embed_dim) * .02)
        self.pos_drop = nn.Dropout(drop)
        self.prompt_pool = self.use_e_prompt = prompted
        self.use_prompt_mask = prompted
        self.use_g_prompt, self.g_prompt = False, None
        self.g_prompt_layer_idx, self.e_prompt_layer_idx = (), layers
        self.use_prefix_tune_for_g_prompt = self.use_prefix_tune_for_e_prompt = prompted
        if prompted:
            self.e_prompt = ProlPrompt(length=prompt_length, embed_dim=embed_dim, num_tasks=n_tasks,
                                      num_classes=num_classes, num_layers=len(layers), num_heads=num_heads,
                                      top_k_l=top_k_l, prompt_key_init=prompt_key_init)
        self.total_prompt_len = 0
        rates = [rate.item() for rate in torch.linspace(0, drop_path, depth, device='cpu')]
        self.blocks = nn.Sequential(*(Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                                           drop, attn_drop, rate) for rate in rates))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.fc_norm = nn.Identity()
        self.head = SimpleContinualLinear(embed_dim, num_classes // n_tasks)
        # Head initializes its own linear once, followed by the global source
        # init traversal that initializes it again after all backbone linears.
        trunc_normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_featuresA1(self, x, y, task_id=-1, cls_features=None, train=False):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        output = {}
        last_prompt, counter = None, -1
        for index, block in enumerate(self.blocks):
            if index in self.e_prompt_layer_idx:
                counter += 1
                output = self.e_prompt(x, y, task_id=task_id, layer_num=counter, cls_features=cls_features)
                prefix = output['batched_prompt']
                # Adjacent raw prefixes are added, rather than a cumulative sum.
                x = block(x, prompt=prefix + last_prompt if counter > 0 else prefix)
                last_prompt = prefix
            else:
                x = block(x)
        output['x'] = self.norm(x)
        return output

    def forward_headA1(self, output, task_id=-1, device=None, pre_logits=False):
        features = output['x'][:, self.total_prompt_len]
        output['pre_logits'] = features
        output['logits'] = self.head(features)['logits']
        return output

    def forwardA1(self, x, y, task_id=-1, cls_features=None, train=False, kl_idx=None):
        return self.forward_headA1(self.forward_featuresA1(x, y, task_id, cls_features, train), task_id, x.device)

    def forward(self, x, task_id=-1, cls_features=None, train=False):
        if self.use_e_prompt:
            raise ValueError('Use forwardA1 for PROL prompted inference with an explicit frozen query')
        result = self.forward_featuresA1(x, None)
        # Source original_model.forward_head only returns pre_logits; its
        # classifier parameters remain allocated but are never evaluated.
        result['pre_logits'] = result['x'][:, 0]
        return result

    @torch.no_grad()
    def load_pretrained(self, checkpoint):
        """Load all encoder tensors with exact shapes from a local AugReg NPZ."""
        path = Path(checkpoint).expanduser()
        if not path.is_file() or path.suffix != '.npz':
            raise FileNotFoundError('PROL requires an existing local AugReg .npz checkpoint')
        with np.load(path, allow_pickle=False) as archive:
            prefix = 'opt/target/' if 'opt/target/embedding/kernel' in archive else ''

            def copy(target, key, transform=lambda value: value):
                value = torch.from_numpy(np.array(transform(archive[prefix + key]), copy=True))
                if value.shape != target.shape or not torch.isfinite(value).all():
                    raise ValueError(f'PROL checkpoint shape or values invalid for {key}')
                target.copy_(value)

            copy(self.patch_embed.proj.weight, 'embedding/kernel', lambda value: value.transpose(3, 2, 0, 1))
            copy(self.patch_embed.proj.bias, 'embedding/bias')
            copy(self.cls_token, 'cls')
            copy(self.pos_embed, 'Transformer/posembed_input/pos_embedding')
            copy(self.norm.weight, 'Transformer/encoder_norm/scale')
            copy(self.norm.bias, 'Transformer/encoder_norm/bias')
            for index, block in enumerate(self.blocks):
                base = f'Transformer/encoderblock_{index}/'
                copy(block.norm1.weight, base + 'LayerNorm_0/scale')
                copy(block.norm1.bias, base + 'LayerNorm_0/bias')
                attention = base + 'MultiHeadDotProductAttention_1/'
                weight = np.concatenate([archive[prefix + attention + name + '/kernel'].reshape(self.embed_dim, -1).T
                                         for name in ('query', 'key', 'value')], axis=0)
                bias = np.concatenate([archive[prefix + attention + name + '/bias'].reshape(-1)
                                       for name in ('query', 'key', 'value')])
                if (weight.shape != block.attn.qkv.weight.shape or bias.shape != block.attn.qkv.bias.shape
                        or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
                    raise ValueError('PROL checkpoint attention dimensions or values invalid')
                block.attn.qkv.weight.copy_(torch.from_numpy(weight))
                block.attn.qkv.bias.copy_(torch.from_numpy(bias))
                copy(block.attn.proj.weight, attention + 'out/kernel', lambda value: value.transpose(2, 0, 1).reshape(self.embed_dim, -1))
                copy(block.attn.proj.bias, attention + 'out/bias')
                copy(block.norm2.weight, base + 'LayerNorm_2/scale')
                copy(block.norm2.bias, base + 'LayerNorm_2/bias')
                for layer in (0, 1):
                    dense = base + f'MlpBlock_3/Dense_{layer}/'
                    copy(getattr(block.mlp, f'fc{layer + 1}').weight, dense + 'kernel', lambda value: value.T)
                    copy(getattr(block.mlp, f'fc{layer + 1}').bias, dense + 'bias')
