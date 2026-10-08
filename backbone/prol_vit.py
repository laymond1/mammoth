"""Registered local ViT branches for PROL's pinned official active path."""

from types import SimpleNamespace

from backbone import register_backbone


@register_backbone('prol-vit')
def prol_vit(num_classes: int, prol_num_tasks: int = 10,
             prol_checkpoint: str = None, prol_vit_size: str = 'base',
             prol_prompt_length: int = 5, prol_prompt_layers: tuple = tuple(range(7)),
             prol_top_k_l: int = 1):
    from models.prol_prompt_utils.network import build_prol_network

    args = SimpleNamespace(prol_num_tasks=prol_num_tasks, prol_checkpoint=prol_checkpoint,
                           prol_vit_size=prol_vit_size, prol_prompt_length=prol_prompt_length,
                           prol_prompt_layers=prol_prompt_layers, prol_top_k_l=prol_top_k_l)
    return build_prol_network(args, num_classes, prol_num_tasks)
