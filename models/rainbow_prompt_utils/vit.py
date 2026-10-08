"""RainbowPrompt CLS-head ViT with the author's independent K/V prefixes.

Architecture and active forward path follow author commit
0b84d3e3738669fe4bb516ccc1c526a535576ad3. Local AugReg loading uses the
existing complete frozen-transformer loader without downloads or head loading.
"""
from functools import partial

import torch
from torch import nn
from timm.layers import DropPath, Mlp, PatchEmbed, trunc_normal_

from models.rainbow_prompt_utils.prompt import RainbowPrompt
from models.hide_prompt_utils.vit import HideVisionTransformer


class PrefixAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., store_query=True):
        super().__init__()
        if type(dim) is not int or type(num_heads) is not int or dim < 1 or num_heads < 1 or dim % num_heads:
            raise ValueError('RainbowPrompt attention dimensions must be divisible by the head count')
        self.num_heads = num_heads
        self.store_query = store_query
        self.scale = (dim // num_heads) ** -.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt=None):
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, width // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        # The author caches q only in PreT_Attention; the unprompted query
        # transformer's plain Attention releases these potentially large views.
        if self.store_query:
            self.key_ = query
        if prompt is not None:
            if (not isinstance(prompt, (tuple, list)) or len(prompt) != 2
                    or any(prefix.ndim != 3 or prefix.shape[0] != batch or prefix.shape[2] != width for prefix in prompt)
                    or prompt[0].shape[1] != prompt[1].shape[1]):
                raise ValueError('RainbowPrompt prefixes must be independent matching [batch, length, width] K/V tensors')
            prefix_key, prefix_value = (prefix.reshape(batch, -1, self.num_heads, width // self.num_heads).permute(0, 2, 1, 3)
                                        for prefix in prompt)
            key = torch.cat((prefix_key, key), dim=2)
            value = torch.cat((prefix_value, value), dim=2)
        attention = self.attn_drop(((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        output = (attention @ value).transpose(1, 2).reshape(batch, tokens, width)
        return self.proj_drop(self.proj(output))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., store_query=True):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm(dim)
        self.attn = PrefixAttention(dim, num_heads, qkv_bias, attn_drop, drop, store_query)
        self.ls1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        self.ls2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, prompt=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), prompt)))
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))


class RainbowVisionTransformer(nn.Module):
    def __init__(self, num_classes, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                 prompt_length=20, pool_size=None, n_tasks=10, prompt_layers=tuple(range(12)), D1=56, D2=96,
                 warmup=5, KI_iter=10, self_attn_idx=tuple(range(6)), drop=0., drop_path=0., attn_drop=0.,
                 mlp_ratio=4., qkv_bias=True):
        super().__init__()
        if (any(type(value) is not int or value < 1 for value in (num_classes, embed_dim, depth, num_heads))
                or embed_dim % num_heads or type(warmup) is not int or warmup < 0):
            raise ValueError('Invalid RainbowPrompt classifier, attention or warmup dimensions')
        prompt_layers = tuple(prompt_layers)
        if pool_size is not None and (prompt_layers != tuple(range(len(prompt_layers)))
                                     or not prompt_layers or len(prompt_layers) > depth):
            raise ValueError('RainbowPrompt requires contiguous prefix layers within the transformer depth')
        self.num_classes, self.num_features, self.embed_dim = num_classes, embed_dim, embed_dim
        self.img_size, self.warmup = img_size, warmup
        self.class_token, self.num_prefix_tokens = True, 1
        self.global_pool, self.head_type = 'token', 'token'
        self.no_embed_class, self.grad_checkpointing = False, False
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, embed_dim) * .02)
        self.pos_drop = nn.Dropout(drop)
        self.prompt_pool = self.use_e_prompt = pool_size is not None
        self.e_prompt_layer_idx = prompt_layers if pool_size is not None else ()
        self.prompt_layers = self.e_prompt_layer_idx
        self.use_prefix_tune_for_e_prompt = pool_size is not None
        self.use_g_prompt, self.g_prompt = False, None
        self.g_prompt_layer_idx = ()
        self.use_prefix_tune_for_g_prompt = self.use_prompt_mask = False
        if pool_size is not None:
            self.rainbow_prompt = RainbowPrompt(length=prompt_length, embed_dim=embed_dim, pool_size=pool_size,
                num_layers=len(prompt_layers), num_heads=num_heads, prompt_tune_idx=prompt_layers, n_tasks=n_tasks,
                D1=D1, D2=D2, KI_iter=KI_iter, self_attn_idx=self_attn_idx)
        self.total_prompt_len = prompt_length * len(prompt_layers) if pool_size is not None else 0
        rates = [rate.item() for rate in torch.linspace(0, drop_path, depth, device='cpu')]
        self.blocks = nn.Sequential(*(Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, rate,
                                           store_query=self.use_e_prompt) for rate in rates))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.fc_norm = nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)
        trunc_normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x, task_id=-1, learned_id=-1, cls_features=None, train=False, epoch_info=None):
        if self.use_e_prompt and (type(train) is not bool or train and (type(epoch_info) is not int or epoch_info < 0)):
            raise ValueError('RainbowPrompt training requires an explicit nonnegative epoch index')
        prompt_type = ('Rainbow' if not train or epoch_info >= self.warmup else 'Unique') if self.use_e_prompt else None
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        result = {}
        for index, block in enumerate(self.blocks):
            if index in self.e_prompt_layer_idx:
                result = self.rainbow_prompt(x, layer=index, cls_features=cls_features, task_id=task_id,
                                             cur_id=learned_id, train=train, p_type=prompt_type)
                x = block(x, prompt=result['batched_prompt'])
            else:
                x = block(x)
        result['x'] = self.norm(x)
        return result

    def forward_head(self, result, pre_logits=False):
        features = result['x'][:, 0]
        result['pre_logits'] = result['features'] = features
        result['logits'] = self.head(self.fc_norm(features))
        return result

    def forward(self, x, task_id=-1, learned_id=-1, cls_features=None, train=False, epoch_info=None):
        return self.forward_head(self.forward_features(x, task_id, learned_id, cls_features, train, epoch_info))

    def load_pretrained(self, checkpoint):
        return HideVisionTransformer.load_pretrained(self, checkpoint)
