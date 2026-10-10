"""Ambient token-compression state shared by the patched blocks of one encoder.

The controller is a plain object referenced from the blocks. It is never a
submodule, parameter or buffer, so optimizers, state dictionaries and weight
digests of the host network are unchanged.
"""

from contextlib import contextmanager

from .reducers import schedule, valid_degree


class CompressionController:
    def __init__(self, operator, depth, degree=0):
        self.operator = operator
        self.depth = depth
        self.default_degree = valid_degree(degree)
        self.bank = None
        self._override = None
        self.size = None
        self.trace = []

    def __deepcopy__(self, memo):
        # A copied network (e.g. a distillation teacher) follows the same degree.
        return self

    @property
    def current(self):
        return self.default_degree if self._override is None else self._override

    @contextmanager
    def degree(self, value):
        """Run the enclosed forward passes at ``value`` instead of the default."""
        previous, self._override = self._override, valid_degree(value)
        try:
            yield self
        finally:
            self._override = previous

    def begin_pass(self):
        self.size, self.trace = None, []

    def verify_pass(self):
        """Raise unless the last encoder pass followed the current schedule."""
        if not self.trace or self.trace != schedule(self.current, self.trace[0], self.depth):
            raise RuntimeError(f'Token compression did not run as scheduled: {self.trace}')
