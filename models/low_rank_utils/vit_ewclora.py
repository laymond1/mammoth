# Ported from CL-LoRA d007ef5c13a6d8a7d7db3eedf93f5e796e2ce429.
# Original ViT implementation Copyright 2020 Ross Wightman.
import math
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_
from timm.models._manipulate import named_apply, checkpoint_seq

from .vit_common import get_init_weights_vit, init_weights_vit_timm
from .pretrained import load_pretrained_encoder as _load_weights

class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lora_A = []
        self.lora_B = []
        self.lora_new_A = []
        self.lora_new_B = []

        # Accumulated LoRA
        self.lora_A_k = nn.Linear(dim, r, bias=False)
        self.lora_B_k = nn.Linear(r, dim, bias=False)
        self.lora_A_v = nn.Linear(dim, r, bias=False)
        self.lora_B_v = nn.Linear(r, dim, bias=False)

        # LoRA for new task
        self.lora_new_A_k = nn.Linear(dim, r, bias=False)
        self.lora_new_B_k = nn.Linear(r, dim, bias=False)
        self.lora_new_A_v = nn.Linear(dim, r, bias=False)
        self.lora_new_B_v = nn.Linear(r, dim, bias=False)

        # Assign custom attributes
        setattr(self.lora_A_k.weight, '_is_a', True)
        setattr(self.lora_B_k.weight, '_is_b', True)
        setattr(self.lora_A_v.weight, '_is_a', True)
        setattr(self.lora_B_v.weight, '_is_b', True)
        setattr(self.lora_new_A_k.weight, '_is_new_a', True)
        setattr(self.lora_new_B_k.weight, '_is_new_b', True)
        setattr(self.lora_new_A_v.weight, '_is_new_a', True)
        setattr(self.lora_new_B_v.weight, '_is_new_b', True)

        # Group
        self.lora_A.append(self.lora_A_k)
        self.lora_B.append(self.lora_B_k)
        self.lora_A.append(self.lora_A_v)
        self.lora_B.append(self.lora_B_v)
        self.lora_new_A.append(self.lora_new_A_k)
        self.lora_new_B.append(self.lora_new_B_k)
        self.lora_new_A.append(self.lora_new_A_v)
        self.lora_new_B.append(self.lora_new_B_v)

        self.rank = r

    def init_param(self):
        for A in self.lora_A:
            nn.init.zeros_(A.weight)
        for B in self.lora_B:
            nn.init.zeros_(B.weight)
        for new_A in self.lora_new_A:
            nn.init.kaiming_uniform_(new_A.weight, a=math.sqrt(5))
        for new_B in self.lora_new_B:
            nn.init.zeros_(new_B.weight)

    def accumulate_and_reset_lora(self):
        for i in range(len(self.lora_A)):
            self.lora_A[i].weight.data += self.lora_new_A[i].weight.data
            self.lora_B[i].weight.data += self.lora_new_B[i].weight.data
            self.reset_new_lora(i)

    def reset_new_lora(self, idx):
        nn.init.kaiming_uniform_(self.lora_new_A[idx].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_new_B[idx].weight)

    def save_grad(self, name):
        def hook(grad):
            setattr(self, f"{name}_grad", grad)
        return hook

    def forward(self, x, use_new=True, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x)
        new_k = self.lora_B_k(self.lora_A_k(x))
        new_v = self.lora_B_v(self.lora_A_v(x))
        qkv[:, :, self.dim : 2*self.dim] += new_k
        qkv[:, :, 2*self.dim :] += new_v

        if use_new:
            # explicitly compute delta_W = B @ A
            delta_w_k_new = self.lora_new_B_k.weight @ self.lora_new_A_k.weight
            delta_w_v_new = self.lora_new_B_v.weight @ self.lora_new_A_v.weight
            new_k = x @ delta_w_k_new.t()
            new_v = x @ delta_w_v_new.t()
            qkv[:, :, self.dim : 2*self.dim] += new_k
            qkv[:, :, 2*self.dim :] += new_v

            if register_hook:
                delta_w_k_new.register_hook(self.save_grad("delta_w_k_new"))
                delta_w_v_new.register_hook(self.save_grad("delta_w_v_new"))

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_tasks=10, r=64):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_LoRA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, n_tasks=n_tasks, r=r)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, use_new=True, register_hook=False):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), use_new=use_new, register_hook=register_hook)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ParallelBlock(nn.Module):

    def __init__(
            self, dim, num_heads, num_parallel=2, mlp_ratio=4., qkv_bias=False, init_values=None,
            drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('attn', Attention_LoRA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))
            self.ffns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('mlp', Mlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))

    def _forward_jit(self, x):
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x):
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def forward(self, x):
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x)
        else:
            return self._forward(x)


class VisionTransformer(nn.Module):
    """ Vision Transformer for continual learning
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64):

        super().__init__()
        assert global_pool in ('', 'avg', 'token')

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.grad_checkpointing = False

        # patch embedding
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # transformer blocks * depth
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                init_values=init_values, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, n_tasks=n_tasks, r=rank)
            for i in range(depth)
        ])

        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Representation layer. Used for original ViT models w/ in21k pretraining.
        self.representation_size = representation_size
        self.pre_logits = nn.Identity()
        if representation_size:
            self._reset_representation(representation_size)

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()
        self.out_dim = final_chs

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def _reset_representation(self, representation_size):
        self.representation_size = representation_size
        if self.representation_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, self.representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None, representation_size=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        if representation_size is not None:
            self._reset_representation(representation_size)
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        x = self.norm(x)

        return x

    def forward_head(self, x, pre_logits: bool = False):
        # pooling and final linear layer
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.pre_logits(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)

        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)

        return {
            'fmaps': [x],
            'features': x
        }
