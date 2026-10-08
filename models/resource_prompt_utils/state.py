"""Algorithm RNG is separate from data augmentation and travels in state_dict."""

import torch
from torch import nn


class CheckpointRNG(nn.Module):
    def __init__(self, seed=1993):
        super().__init__()
        self.seed = int(seed)
        self._generators = {}
        self._pending = {}

    def generator(self, device):
        key = str(torch.device(device))
        if key not in self._generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            if key in self._pending:
                generator.set_state(self._pending.pop(key).cpu())
            self._generators[key] = generator
        return self._generators[key]

    def get_extra_state(self):
        states = {key: value.clone() for key, value in self._pending.items()}
        states.update({key: generator.get_state().cpu().clone() for key, generator in self._generators.items()})
        return {'seed': self.seed, 'states': states}

    def set_extra_state(self, state):
        self.seed = int(state['seed'])
        self._generators = {}
        self._pending = {key: value.cpu().clone() for key, value in state['states'].items()}


class PromptStateMixin:
    """Restore CL counters without reinitializing already learned prompt rows."""
    _state_fields = ('_current_task', '_n_past_classes', '_n_seen_classes', '_n_classes_current_task',
                     '_n_remaining_classes', '_task_iteration', '_epoch_iteration', '_past_epoch', '_last_epoch')

    def get_extra_state(self):
        return {name: int(getattr(self, name)) for name in self._state_fields if hasattr(self, name)}

    def set_extra_state(self, state):
        for name, value in state.items():
            if name not in self._state_fields or type(value) is not int:
                raise ValueError('Invalid resource-prompt lifecycle state')
            setattr(self, name, value)
