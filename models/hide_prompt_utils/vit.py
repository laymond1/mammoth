"""The frozen transformer and residual TII mapping used by HiDe-Prompt.

Relevant architecture/math ported from the pinned author's
vits/hide_prompt_vision_transformer.py and attention.py. Only the approved
CLS-head, prefix E-prompt recipe is supported, avoiding inactive options.
"""

from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_

from models.hide_prompt_utils.prompt import EPrompt


class PrefixAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt=None):
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, width // self.num_heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if prompt is not None:
            prefix = prompt.permute(1, 0, 3, 2, 4).contiguous()
            expected = (2, batch, self.num_heads, prompt.shape[2], width // self.num_heads)
            if tuple(prefix.shape) != expected:
                raise ValueError('HiDe-Prompt prefix does not match its attention dimensions')
            key = torch.cat((prefix[0], key), dim=2)
            value = torch.cat((prefix[1], value), dim=2)
        attention = self.attn_drop(((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, width)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm(dim)
        self.attn = PrefixAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path else nn.Identity()
        self.norm2 = norm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x, prompt=None):
        x = x + self.drop_path1(self.attn(self.norm1(x), prompt))
        return x + self.drop_path2(self.mlp(self.norm2(x)))


class MlpMapping(nn.Module):
    """Author residual mapping: hidden ratios, GELU, final residual + LN."""

    def __init__(self, dim=768, hidden=(2,), drop=0.):
        super().__init__()
        if not hidden or any(type(ratio) is not int or ratio < 1 for ratio in hidden):
            raise ValueError('HiDe-Prompt TII mapping requires positive integer hidden ratios')
        layers = []
        widths = (dim, *(dim * ratio for ratio in hidden), dim)
        for index, (incoming, outgoing) in enumerate(zip(widths, widths[1:])):
            linear = nn.Linear(incoming, outgoing)
            dropout = nn.Dropout(drop)
            if index == len(hidden) // 2 and index not in (0, len(hidden)):
                layer = (linear, dropout, nn.LayerNorm(outgoing))
            elif index == len(hidden):
                layer = (linear, dropout)
            else:
                layer = (linear, nn.GELU(), dropout)
            layers.append(nn.Sequential(*layer))
        self.norm = nn.LayerNorm(dim)
        self.net = nn.ModuleList(layers)

    def forward(self, x):
        output = x
        for layer in self.net:
            output = layer(output)
        return self.norm(output + x)


class HideVisionTransformer(nn.Module):
    def __init__(self, num_classes, img_size=224, patch_size=16, embed_dim=768,
                 depth=12, num_heads=12, prompt_length=5, pool_size=None,
                 prompt_layers=(0, 1, 2, 3, 4), mlp_structure=(), drop=0.,
                 drop_path=0., use_permute_fix=False):
        super().__init__()
        if embed_dim % num_heads or num_classes < 1:
            raise ValueError('Invalid HiDe-Prompt classifier or attention dimensions')
        if pool_size is not None and (any(index < 0 or index >= depth for index in prompt_layers)
                                      or len(set(prompt_layers)) != len(prompt_layers)):
            raise ValueError('HiDe-Prompt layers must be unique valid block indices')
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_prefix_tokens = 1
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, embed_dim) * .02)
        self.pos_drop = nn.Dropout(drop)
        self.prompt_layers = tuple(prompt_layers) if pool_size is not None else ()
        self.e_prompt = EPrompt(prompt_length, embed_dim, pool_size, len(prompt_layers),
                                num_heads, use_permute_fix=use_permute_fix) if pool_size is not None else None
        rates = [index * drop_path / max(depth - 1, 1) for index in range(depth)]
        self.blocks = nn.Sequential(*(Block(embed_dim, num_heads, drop=drop, drop_path=rate) for rate in rates))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MlpMapping(embed_dim, mlp_structure) if mlp_structure else nn.Identity()
        self.fc_norm = nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)
        trunc_normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x, task_id=-1, prompt_id=None, train=False, prompt_momentum=0):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        result = {}
        prefixes = None
        if self.e_prompt is not None:
            mask = torch.full((x.shape[0], 1), task_id, device=x.device, dtype=torch.long) if train else None
            if task_id == 0:
                prompt_momentum = 0
            result = self.e_prompt(x, prompt_mask=mask, prompt_idx=prompt_id, prompt_momentum=prompt_momentum)
            prefixes = result['batched_prompt']
        counter = 0
        for index, block in enumerate(self.blocks):
            if index in self.prompt_layers:
                x = block(x, prefixes[counter])
                counter += 1
            else:
                x = block(x)
        result['x'] = self.norm(x)
        return result

    def forward(self, x, task_id=-1, prompt_id=None, train=False, fc_only=False, prompt_momentum=0):
        if fc_only:
            return {'logits': self.head(self.fc_norm(self.mlp(x)))}
        result = self.forward_features(x, task_id, prompt_id, train, prompt_momentum)
        features = result['x'][:, 0]
        result['pre_logits'] = result['features'] = features
        result['logits'] = self.head(self.fc_norm(self.mlp(features)))
        return result

    @torch.no_grad()
    def load_pretrained(self, checkpoint):
        """Load only complete frozen-transformer weights from a local AugReg NPZ.

        The task classifier and TII mapping retain their random initialization.
        Exact dimensions are required; no interpolation or downloads occur.
        """
        path = Path(checkpoint).expanduser()
        if not path.is_file() or path.suffix != '.npz':
            raise FileNotFoundError('HiDe-Prompt requires an existing local AugReg .npz checkpoint')
        with np.load(path, allow_pickle=False) as archive:
            prefix = 'opt/target/' if 'opt/target/embedding/kernel' in archive else ''

            def copy(target, key, transform=lambda value: value):
                value = torch.from_numpy(np.array(transform(archive[prefix + key]), copy=True))
                if value.shape != target.shape or not torch.isfinite(value).all():
                    raise ValueError(f'HiDe-Prompt checkpoint shape or values invalid for {key}')
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
                attn = base + 'MultiHeadDotProductAttention_1/'
                qkv_weight = np.concatenate([archive[prefix + attn + name + '/kernel'].reshape(self.embed_dim, -1).T
                                             for name in ('query', 'key', 'value')], axis=0)
                qkv_bias = np.concatenate([archive[prefix + attn + name + '/bias'].reshape(-1)
                                           for name in ('query', 'key', 'value')])
                if qkv_weight.shape != block.attn.qkv.weight.shape or qkv_bias.shape != block.attn.qkv.bias.shape:
                    raise ValueError('HiDe-Prompt checkpoint attention dimensions do not match')
                if not np.isfinite(qkv_weight).all() or not np.isfinite(qkv_bias).all():
                    raise ValueError('HiDe-Prompt checkpoint attention contains nonfinite values')
                block.attn.qkv.weight.copy_(torch.from_numpy(qkv_weight))
                block.attn.qkv.bias.copy_(torch.from_numpy(qkv_bias))
                copy(block.attn.proj.weight, attn + 'out/kernel', lambda value: value.transpose(2, 0, 1).reshape(self.embed_dim, -1))
                copy(block.attn.proj.bias, attn + 'out/bias')
                copy(block.norm2.weight, base + 'LayerNorm_2/scale')
                copy(block.norm2.bias, base + 'LayerNorm_2/bias')
                for layer in (0, 1):
                    dense = base + f'MlpBlock_3/Dense_{layer}/'
                    copy(getattr(block.mlp, f'fc{layer + 1}').weight, dense + 'kernel', lambda value: value.T)
                    copy(getattr(block.mlp, f'fc{layer + 1}').bias, dense + 'bias')
