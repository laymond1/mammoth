# This code is a reimplementation based on the Adaptive Token Sampling methodology.
# This code has been modified for online continual learning by Wonseon Lim.
'''
 * Based on vit from blip code base
 * https://github.com/salesforce/BLIP
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models.helpers import named_apply, adapt_input_conv

# from models.ats_prompt_utils.transformers.transformer_block import ALinear
# from models.ats_prompt_utils.ats import ATSBlock


class ALinear(nn.Linear):
    def forward(self, input: torch.Tensor, mask: torch.Tensor, _):
        return super().forward(input)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ALinear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = ALinear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, policy=None, sampler=None):
        x = self.fc1(x, policy, sampler)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x, policy, sampler)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., enable_softmax_policy=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = ALinear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = ALinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.enable_softmax_policy = enable_softmax_policy
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    @staticmethod
    def softmax_with_policy(attn, policy, eps=1e-6):
        # attn: (B, H, N_q, N_k)
        # policy: (B, N_orig, 1) or (B, N_orig), N_orig: original number of tokens without prompt
        B, H, N_q, N_k = attn.size()
        policy_flat = policy.view(B, -1)
        N_orig = policy_flat.size(1)

        # If number of keys (N_k) differs from policy length (N_orig), assume
        # prompt tokens were concatenated to the FRONT of k/v. In that case
        # prepend ones for prompt positions so they are not suppressed by policy.
        if N_k != N_orig:
            num_prompt = N_k - N_orig
            if num_prompt < 0:
                # fewer keys than policy: truncate policy
                full_policy = policy_flat[:, :N_k]
            else:
                prompt_ones = attn.new_ones((B, num_prompt))
                full_policy = torch.cat([prompt_ones, policy_flat], dim=1)
        else:
            full_policy = policy_flat

        # Reshape policy to broadcast over attention: (B, 1, 1, N_k) to match (B, H, N_q, N_k)
        attn_policy = full_policy.view(B, 1, 1, N_k)
        
        # Create diagonal matrix to ensure self-attention is preserved
        # eye shape: (1, 1, N_q, N_k) where diagonal along last two dims
        # For rectangular attention (N_q != N_k), we only mask the key dimension
        eye = torch.eye(N_q, N_k, dtype=attn_policy.dtype, device=attn_policy.device).view(
            1, 1, N_q, N_k
        )
        
        # Apply policy: keep diagonal (self-attention), suppress based on policy for others
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N_k) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)
    
    def forward(self, x, register_hook=False, prompt=None, policy=None, sampler=None):
        B, N, C = x.shape
        qkv = self.qkv(x, policy, sampler)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            # import ipdb; ipdb.set_trace()
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat((pk,k), dim=2)
            v = torch.cat((pv,v), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if policy is None or not self.enable_softmax_policy:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)
        
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x, policy, sampler)
        x = self.proj_drop(x)
        return x
    

class AdaptiveTokenSampler(Attention):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        drop_tokens=False,
    ):
        super(AdaptiveTokenSampler, self).__init__(
            dim,
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.out_zero_mask = nn.Parameter(torch.zeros(1, dim), requires_grad=False)
        self.drop_tokens = drop_tokens

    @staticmethod
    def get_unique_indices(indices: torch.Tensor, max_value: int):
        """
        :param indices: indices of the tokens to be sampled
        :param max_value: maximum number of the tokens to be sampled
        :return: unique indices of the tokens to be sampled
        """
        sorted_indices = torch.sort(indices, dim=1)[0]

        shift_left = F.pad(sorted_indices[:, 1:], (0, 1), value=1.0)
        unique_indices = torch.where(
            (shift_left - sorted_indices) == 0,
            max_value * torch.ones_like(indices),
            sorted_indices,
        )
        unique_indices = torch.sort(unique_indices, dim=1)[0]
        return unique_indices

    @staticmethod
    def create_ys(normalized_cdf: torch.Tensor, n_tokens: int):
        """
        Sample uniformly from y-axis.
        """

        B = normalized_cdf.shape[0]
        # epsilon = (1 / (n_tokens - 1)) / 2
        ys = (
            torch.linspace(
                start=0,
                end=1.0,
                steps=n_tokens - 1,
                device=normalized_cdf.device,
            )
            .unsqueeze(0)
            .repeat(B, 1)
        )
        ys_start = (
            torch.min(normalized_cdf + (normalized_cdf == 0).float() * 1e8, dim=1)[0]
            .unsqueeze(-1)
            .expand_as(ys)
        )
        steps = (
            torch.arange(0, n_tokens - 1, device=normalized_cdf.device)
            .unsqueeze(0)
            .expand_as(ys_start)
        )
        ys = ys_start + (((ys * (n_tokens - 2)) - ys_start * steps) / (n_tokens - 2))

        return ys

    @staticmethod
    def score_assignment_step(attn: torch.Tensor, v: torch.Tensor, num_prompt: int = 0):
        """
        Token Score Assignment Step.
        :param attn: attention matrix (B, H, N_q, N_k)
        :param v: values (B, H, N_v, head_dim)
        :param num_prompt: number of prompt tokens prepended to k and v
        :return: sorted significance scores and their corresponding indices
        """

        B, H, N_q, N_k = attn.shape
        _, _, N_v, head_dim = v.shape
        C = head_dim * H
        
        # If prompt tokens exist, exclude them from v_norm calculation
        # Prompt tokens are at positions [0:num_prompt], CLS at position [num_prompt]
        if num_prompt > 0:
            v_no_prompt = v[:, :, num_prompt:, :]  # [B, H, N_v - num_prompt, head_dim]
        else:
            v_no_prompt = v
        
        # Calculate v_norm only for non-prompt tokens (includes CLS + patches)
        v_norm = torch.linalg.norm(
            v_no_prompt.transpose(1, 2).reshape(B, v_no_prompt.shape[2], C), ord=2, dim=2
        )  # value norm of size [B x (N_v - num_prompt)]
        
        # Exclude CLS token from v_norm since we only score patch tokens
        v_norm = v_norm[:, 1:]  # [B x (N_v - num_prompt - 1)]
        
        # Attention weights of CLS token: shape [B x N_k]
        # CLS token attends to all keys including prompt tokens
        significance_score = attn[:, :, 0].sum(dim=1)  # [B x N_k]
        
        # Exclude prompt tokens from significance score (they're at front: [0:num_prompt])
        # Also exclude CLS token itself (at position num_prompt)
        if num_prompt > 0:
            significance_score = significance_score[:, num_prompt+1:]  # [B x (N_k - num_prompt - 1)]
        else:
            significance_score = significance_score[:, 1:]  # [B x (N_k - 1)], exclude CLS token
        
        significance_score = significance_score * v_norm  # [B x (N_orig - 1)]

        significance_score = significance_score / significance_score.sum(
            dim=1, keepdim=True
        )  # [B x T-1] or [B x (T - 1 - num_prompt)]
        sorted_scores, sorted_indices = torch.sort(
            significance_score, descending=False, dim=1
        )

        return sorted_scores, sorted_indices

    def inverse_transform_sampling(
        self,
        sorted_scores: torch.Tensor,
        sorted_indices: torch.Tensor,
        attn: torch.Tensor,
        n_tokens: int,
        raw_x: torch.Tensor,
        n_ref_tokens: int,
        num_prompt: int = 0,
    ):
        """
        Sample tokens based on their significance scores.
        :param num_prompt: number of prompt tokens in attn (keys dimension)
        """
        B, N, C = raw_x.shape
        
        # If prompt tokens exist in attention keys, extract only the non-prompt part
        # attn shape: [B, H, N_q, N_k] where N_k may include prompt tokens
        # We need attention over [CLS + patches] only, excluding prompt tokens
        if num_prompt > 0:
            # Slice out prompt tokens from key dimension
            # attn structure: [prompt_attn, cls_attn, patch_attn]
            attn_no_prompt = attn[:, :, :, num_prompt:]  # [B, H, N_q, N_orig] where N_orig = CLS + patches
        else:
            attn_no_prompt = attn
        
        # Now attn_no_prompt has shape [B, H, N_q, N] matching raw_x's token dimension

        cdf = torch.cumsum(sorted_scores, dim=1)  # [B x T-1]

        normalized_cdf = (  # normalized cdf
            cdf - cdf.min(dim=1)[0].unsqueeze(dim=1)
        ) / ((cdf.max(dim=1)[0] - cdf.min(dim=1)[0]) / 1.0).unsqueeze(dim=1)

        ys = self.create_ys(normalized_cdf, n_ref_tokens).unsqueeze(
            dim=2
        )  # sampled values from y-axis of size [B, n-1, 1]
        normalized_cdf = normalized_cdf.unsqueeze(dim=1)  # [B, 1, N - 1]

        # expanded_ys = torch.Tensor.expand(ys, (B, n_tokens - 1, N - 1))
        expanded_ys = torch.Tensor.expand(ys, (B, ys.shape[1], ys.shape[1]))
        diff_tokens = ys.shape[1] - (N - 1)
        tokens_to_pick_ind = torch.min(
            torch.abs(expanded_ys - F.pad(normalized_cdf, (diff_tokens, 0))),
            dim=2,
        )[
            1
        ]  # [B x n-1]

        # Offsetting token indices
        tokens_to_pick_ind = tokens_to_pick_ind - diff_tokens

        # sorted_indices has shape [B, num_patches] where num_patches may be less than N-1
        # if this is not the first ATS block (tokens already pruned in previous blocks)
        num_patches = sorted_indices.shape[1]  # actual number of patch tokens to sort

        # Sort attention matrix and add CLS weights.
        attn_sorted = torch.gather(
            attn_no_prompt[:, :, 1:],
            2,
            sorted_indices.unsqueeze(1)
            .unsqueeze(-1)
            .expand(B, self.num_heads, num_patches, N),
        )  # [B x h x num_patches x N]

        attn_tmp = F.pad(attn_sorted, (0, 0, 0, 1), value=0.0)  # [B x h x (num_patches+1) x N]

        # # Sort tokens and add CLS token.
        raw_x_tmp = torch.gather(
            raw_x[:, 1:], 1, sorted_indices.unsqueeze(-1).expand(B, num_patches, C)
        )
        raw_x_tmp = F.pad(raw_x_tmp, (0, 0, 0, 1), value=0.0)  # [B x (num_patches+1) x C]

        unique_indices = self.get_unique_indices(
            indices=tokens_to_pick_ind, max_value=num_patches
        )[:, : n_tokens - 1]  # Select only n_tokens - 1 tokens (CLS will be added separately)

        # Prune the attention matrix and input tokens.
        attn_tmp = torch.gather(
            attn_tmp,
            2,
            unique_indices.unsqueeze(1)
            .unsqueeze(3)
            .expand(B, self.num_heads, n_tokens - 1, N),
        )
        raw_x_tmp = torch.gather(
            raw_x_tmp, 1, unique_indices.unsqueeze(2).expand(B, n_tokens - 1, C)
        )

        attn_tmp = torch.cat([attn_no_prompt[:, :, 0:1], attn_tmp], dim=2)
        raw_x_tmp = torch.cat([raw_x[:, 0:1], raw_x_tmp], dim=1)

        policy = (unique_indices != num_patches).unsqueeze(-1).float()
        policy = F.pad(policy, (0, 0, 1, 0), value=1.0)
        selected_x = raw_x_tmp
        attn_out = attn_tmp

        sampler = torch.nonzero(policy)

        return selected_x, attn_out, policy, sampler

    def forward(
        self,
        x: torch.Tensor,
        policy: torch.Tensor,
        sampler: torch.Tensor,
        n_tokens: float,
        raw_x: torch.Tensor,
        n_ref_tokens: int,
        prompt=None,
    ):
        B, N, C = x.shape

        if isinstance(N, torch.Tensor):
            N = N.cpu().item()

        if n_tokens > N:  # Number of tokens to be sampled can't be larger than N.
            n_tokens = N
        if n_tokens <= 1.0:  # When n_tokens is a ratio.
            n_tokens = n_tokens * N
        if n_tokens < 8:  # Number of tokens to be sampled can't be less than 8.
            n_tokens = 8

        n_tokens = round(n_tokens)
        if N < n_tokens:
            n_tokens = N

        qkv = self.qkv(x, policy, sampler)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        qkv = qkv * policy.unsqueeze(0).unsqueeze(
            2
        )  # Get rid of previously removed tokens.
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        # Add prompt tokens to k and v if provided
        num_prompt = 0
        if prompt is not None:
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            num_prompt = pk.shape[2]  # Number of prompt tokens
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)

        attn_no_softmax = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax_with_policy(attn_no_softmax, policy)  # [B x H x T x T]

        # --------------------------
        # Token Score Assignment
        # --------------------------

        sorted_scores, sorted_indices = self.score_assignment_step(attn, v, num_prompt=num_prompt)

        # --------------------------
        # Inverse Transform Sampling
        # --------------------------

        selected_x, attn, policy, sampler = self.inverse_transform_sampling(
            sorted_scores, sorted_indices, attn, n_tokens, raw_x, n_ref_tokens, num_prompt=num_prompt
        )

        # If prompt tokens exist, remove them from v for the final output computation
        # since attn now only covers [CLS + selected_patches]
        if num_prompt > 0:
            v_no_prompt = v[:, :, num_prompt:, :]  # [B, H, N_orig, head_dim]
        else:
            v_no_prompt = v
        
        x = (attn @ v_no_prompt).transpose(1, 2).reshape(B, attn.shape[2], C)

        # Pruning
        if self.drop_tokens:
            out_mask_size = policy.sum(1).max().int()

            sampler_out = sampler[:, 0] * out_mask_size + sampler[:, 1]
            sampler = sampler[:, 0] * n_tokens + sampler[:, 1]
            sampler_input = sampler.unsqueeze(-1).expand(-1, C)
            sampler_output = sampler_out.unsqueeze(-1).expand(-1, C)
            flatten_x = x.reshape(-1, C)
            flatten_selected_x = selected_x.reshape(-1, C)

            x_prunned = torch.gather(flatten_x, 0, sampler_input)
            selected_x_prunned = torch.gather(flatten_selected_x, 0, sampler_input)

            out_zero_mask = self.out_zero_mask.expand(B * out_mask_size, -1)
            out_zero = out_zero_mask.new_zeros(out_zero_mask.shape)

            x = out_zero.scatter_add(
                0, sampler_output, x_prunned
            ).reshape((B, out_mask_size, C))
            selected_x = out_zero.scatter_add(
                0, sampler_output, selected_x_prunned
            ).reshape((B, out_mask_size, C))

            policy_zero = out_zero_mask[:, 0].new_zeros(out_zero_mask.shape[0])
            policy = (
                policy_zero
                .scatter_add(0, sampler_out, torch.ones_like(sampler_out, dtype=policy_zero.dtype))
                .reshape(B, out_mask_size, 1)
            )

        x = self.proj(x, policy, sampler)
        x = x * policy
        x = self.proj_drop(x)
        return x, selected_x, policy, sampler


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, enable_softmax_policy=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            enable_softmax_policy=enable_softmax_policy)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, register_hook=False, prompt=None, policy=None, sampler=None):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook, prompt=prompt, policy=policy, sampler=sampler))
        x = x * policy if policy is not None else x
        x = x + self.drop_path(self.mlp(self.norm2(x), policy=policy, sampler=sampler))
        x = x * policy if policy is not None else x
        return x


class ATSBlock(nn.Module):
    """
    Transformer Block + ATS
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, 
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop_tokens=False):
        super().__init__()
        # self.insert_control_point = insert_control_point
        self.norm1 = norm_layer(dim)

        self.attn = AdaptiveTokenSampler(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop_path=drop_path,
            drop_tokens=drop_tokens,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, register_hook=False, prompt=None, n_tokens=None, policy: torch.Tensor = None, sampler: torch.Tensor = None, n_ref_tokens: int = 197):
        x_out, selected_x, policy, sampler = self.attn(
            x=self.norm1(x),
            policy=policy,
            sampler=sampler,
            n_tokens=n_tokens,
            raw_x=x,
            n_ref_tokens=n_ref_tokens,
            prompt=prompt,
        )
        x = selected_x + self.drop_path(x_out)
        x = x * policy
        out = self.mlp(x=self.norm2(x), policy=policy, sampler=sampler)
        x = x + self.drop_path(out)
        x = x * policy
        return x, policy


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, ckpt_layer=0, 
                 ats_blocks=[3, 4, 5, 6, 7, 8, 9, 10, 11],
                 num_tokens=[197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197],
                 drop_tokens=False, enable_softmax_policy=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            enable_softmax_policy (bool): enable softmax with policy in attention
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        control_flags = [True for _ in range(depth)]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.ats_blocks = ats_blocks
        self.num_tokens = num_tokens
        self.drop_tokens = drop_tokens
        
        self.blocks = []
        for i in range(depth):
            if i in self.ats_blocks:
                self.blocks.append(
                    ATSBlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        drop_tokens=drop_tokens,
                    )
                )
            else:
                self.blocks.append(
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        enable_softmax_policy=enable_softmax_policy,
                    )
                )
        self.blocks = nn.ModuleList(self.blocks)
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, register_blk=-1, prompt=None, q=None, train=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
  
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)
        # ATS variables
        init_n = x.shape[1]
        policies = []
        policy = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
        sampler = torch.nonzero(policy)

        prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)

        for i, blk in enumerate(self.blocks):

            if prompt is not None:
                if train:
                    p_list, loss, x = prompt.forward(q, i, x, train=True)
                    prompt_loss += loss
                else:
                    p_list, _, x = prompt.forward(q, i, x, train=False)
            else:
                p_list = None

            if i in self.ats_blocks:
                x, policy = blk(
                    x=x,
                    register_hook=register_blk==i, 
                    prompt=p_list,
                    n_tokens=self.num_tokens[i],
                    policy=policy,
                    sampler=sampler,
                    n_ref_tokens=init_n,
                )
            else:
                x = blk(x=x, 
                        register_hook=register_blk==i, 
                        prompt=p_list,
                        policy=policy, 
                        sampler=sampler)

        x = self.norm(x)

        if prompt is not None:
            prompt_loss /= len(prompt.e_layers)
        
        return x, prompt_loss

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)
        

@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
#     if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
#         model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
#         model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
#     if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
#         model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
#         model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))

            
def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint
