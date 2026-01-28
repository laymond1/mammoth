import torch
import torch.nn as nn
import torch.utils.checkpoint
from models.cprompt_utils.vit import VisionTransformer, PatchEmbed, Block


class ViT_KPrompts(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn)

    def _run_blocks(self, x, start, end):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            for blk in self.blocks[start:end]:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.blocks[start:end]:
                x = blk(x)
        return x

    def forward(self, x, instance_tokens=None, second_pro=None, returnbeforepool=False,gen_pro=None, **kwargs):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if gen_pro is None:
            if instance_tokens is not None and instance_tokens.shape[0]!=16:
                instance_tokens = instance_tokens.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

            x = x + self.pos_embed.to(x.dtype)
            if instance_tokens is not None:
                x = torch.cat([x[:,:1,:], instance_tokens, x[:,1:,:]], dim=1)
            x = self.pos_drop(x)

            x = self._run_blocks(x, 0, 5)
            if second_pro is not None:
                second_pro=second_pro.to(x.dtype)+torch.zeros(x.shape[0],1,x.shape[-1],dtype=x.dtype,device=x.device)
                x = torch.cat([x[:,:1+instance_tokens.shape[1],:], second_pro, x[:,1+instance_tokens.shape[1]:,:]], dim=1)
            x = self._run_blocks(x, 5, len(self.blocks))
        else:
            for i in range(len(instance_tokens)):
                if instance_tokens[i].shape[1]==768:
                    instance_tokens[i]=instance_tokens[i].to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                else:
                    instance_tokens[i]=instance_tokens[i].to(x.dtype)
            
            x = x + self.pos_embed.to(x.dtype)
            x = torch.cat([x[:,:1,:], instance_tokens[0], x[:,1:,:]], dim=1)
            x = self.pos_drop(x)
            x = self._run_blocks(x, 0, 1)
            for i in range(len(instance_tokens)-1):
                x = torch.cat([x[:,:1+instance_tokens[0].shape[1]*(i+1),:], instance_tokens[i+1], x[:,1+instance_tokens[0].shape[1]*(i+1):,:]], dim=1)
                x = self._run_blocks(x, i + 1, i + 2)
            x = self._run_blocks(x, len(instance_tokens), len(self.blocks))

        if returnbeforepool == True:
            return x
        x = self.norm(x)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x


def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p 
