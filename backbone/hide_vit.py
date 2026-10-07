"""Registered Base ViT branches for the native HiDe-Prompt model."""

from types import SimpleNamespace

from backbone import register_backbone


@register_backbone('hide-vit')
def hide_vit(num_classes: int, hide_pool_size: int = 10,
             hide_checkpoint: str = None, hide_vit_size: str = 'base',
             hide_prompt_length: int = 5,
             hide_prompt_layers: tuple = (0, 1, 2, 3, 4),
             hide_mlp_structure: tuple = (2,), hide_prompt_momentum: float = 0.01,
             hide_drop: float = 0., hide_drop_path: float = 0.,
             hide_use_permute_fix: bool = False):
    """Load two branches from one explicit local NPZ without a third backbone."""
    # Backbone discovery happens while model modules can still be importing.
    from models.hide_prompt_utils.network import build_hide_network

    args = SimpleNamespace(hide_checkpoint=hide_checkpoint, hide_vit_size=hide_vit_size,
                           hide_prompt_length=hide_prompt_length,
                           hide_prompt_layers=hide_prompt_layers,
                           hide_mlp_structure=hide_mlp_structure,
                           hide_prompt_momentum=hide_prompt_momentum,
                           hide_drop=hide_drop, hide_drop_path=hide_drop_path,
                           hide_use_permute_fix=hide_use_permute_fix)
    return build_hide_network(args, num_classes, hide_pool_size)
