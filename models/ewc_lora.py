"""EWC-LoRA basic variant from the pinned common low-rank CL source."""

from models.low_rank_utils import continual


class EWCLoRA(continual.LowRankContinualModel):
    NAME = 'ewc-lora'
    METHOD = 'ewclora'
    COMPATIBILITY = ['class-il', 'task-il']
