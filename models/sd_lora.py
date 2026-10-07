"""sd-lora from the pinned common low-rank continual-learning source."""

from models.low_rank_utils import continual


class SDLoRA(continual.LowRankContinualModel):
    NAME = 'sd-lora'
    METHOD = 'sdlora'
    COMPATIBILITY = ["class-il", "task-il"]
