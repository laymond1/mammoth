"""inf-lora from the pinned common low-rank continual-learning source."""

from models.low_rank_utils import continual


class InfLoRA(continual.LowRankContinualModel):
    NAME = 'inf-lora'
    METHOD = 'inflora'
    COMPATIBILITY = ["class-il", "task-il"]
