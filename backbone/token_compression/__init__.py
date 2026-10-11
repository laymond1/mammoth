"""Plug-in token reduction and ToCom compensation for existing ViT encoders."""

from .adapters import attach, compensate, detach, encoders, find_encoder
from .compensator import FrozenToCom, TrainableToCom
from .controller import CompressionController
from .reducers import MAX_DEGREE, OPERATORS, planned, schedule, valid_degree

__all__ = ['attach', 'compensate', 'detach', 'encoders', 'find_encoder', 'CompressionController', 'FrozenToCom', 'TrainableToCom',
           'MAX_DEGREE', 'OPERATORS', 'planned', 'schedule', 'valid_degree']
