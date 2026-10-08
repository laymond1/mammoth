"""Registered local ViT branches for RainbowPrompt's official active path."""

from types import SimpleNamespace

from backbone import register_backbone


@register_backbone('rainbow-vit')
def rainbow_vit(num_classes: int, rainbow_pool_size: int = 10,
                rainbow_checkpoint: str = None, rainbow_vit_size: str = 'base',
                rainbow_prompt_length: int = 20, rainbow_prompt_layers: tuple = tuple(range(12)),
                rainbow_d1: int = 56, rainbow_d2: int = 96,
                rainbow_warmup_epochs: int = 5, rainbow_ki_iter: int = 10,
                rainbow_self_attn_layers: tuple = tuple(range(6))):
    from models.rainbow_prompt_utils.network import build_rainbow_network

    args = SimpleNamespace(rainbow_pool_size=rainbow_pool_size, rainbow_checkpoint=rainbow_checkpoint,
                           rainbow_vit_size=rainbow_vit_size, rainbow_prompt_length=rainbow_prompt_length,
                           rainbow_prompt_layers=rainbow_prompt_layers, rainbow_d1=rainbow_d1,
                           rainbow_d2=rainbow_d2, rainbow_warmup_epochs=rainbow_warmup_epochs,
                           rainbow_ki_iter=rainbow_ki_iter, rainbow_self_attn_layers=rainbow_self_attn_layers)
    return build_rainbow_network(args, num_classes, rainbow_pool_size)
