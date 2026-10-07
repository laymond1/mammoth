"""Registered local ViT for the four pinned continual LoRA methods."""

from backbone import register_backbone


@register_backbone('low-rank-vit')
def low_rank_vit(num_classes: int, low_rank_method: str = 'ewclora',
                 lora_vit_size: str = 'base', lora_checkpoint: str = None,
                 lora_rank: int = 10, lora_classes_per_task: int = 10,
                 lora_msa: tuple = (1, 0, 1),
                 lora_shared_pos: tuple = (0, 1, 2, 3, 4, 5),
                 lora_specific_pos: tuple = (6, 7, 8, 9, 10, 11)):
    """Construct the actual source-compatible network, without a dummy ViT.

    The classifier pool needs equal class increments. Its capacity follows the
    complete dataset, independently of ``stop_after`` in a validation run.
    """
    from models.low_rank_utils.factory import build_network

    if num_classes < 1 or lora_classes_per_task < 1 or num_classes % lora_classes_per_task:
        raise ValueError('Low-rank classifier pools require equal class increments')
    return build_network(low_rank_method, {
        'init_cls': lora_classes_per_task, 'increment': lora_classes_per_task,
        'sessions': num_classes // lora_classes_per_task, 'rank': lora_rank,
        'device': ['cpu'], 'msa': list(lora_msa),
        'shared_pos': list(lora_shared_pos), 'specific_pos': list(lora_specific_pos),
    }, vit_size=lora_vit_size, checkpoint=lora_checkpoint)
