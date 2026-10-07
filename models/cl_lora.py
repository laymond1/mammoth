"""cl-lora from the pinned common low-rank continual-learning source."""

from models.low_rank_utils import continual


class CLLoRA(continual.LowRankContinualModel):
    NAME = 'cl-lora'
    METHOD = 'cllora'
    COMPATIBILITY = ["class-il", "task-il"]
