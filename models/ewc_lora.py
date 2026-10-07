"""EWC-LoRA basic variant from the pinned common low-rank CL source."""

from models.low_rank_utils.continual import LowRankContinualModel


class EWCLoRA(LowRankContinualModel):
    NAME = 'ewc-lora'
    METHOD = 'ewclora'
