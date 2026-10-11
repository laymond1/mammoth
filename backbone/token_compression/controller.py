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
        # Set by compensate(): the training degree of each encoder pass of one forward call, in order.
        self.sources = None
        self.passes = 0

    def __deepcopy__(self, memo):
        # A copied network (e.g. a distillation teacher) follows the same degree.
        return self

    @property
    def current(self):
        return self.default_degree if self._override is None else self._override

    @property
    def overridden(self):
        """Whether an explicit degree context is active."""
        return self._override is not None

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
        self.passes += 1

    def source(self):
        """Training degree of the encoder pass being run, as the bank needs it."""
        if len(self.sources) == 1:
            return self.sources[0]
        if not 1 <= self.passes <= len(self.sources):
            raise RuntimeError(f'Encoder pass {self.passes} has no training degree among {self.sources}')
        return self.sources[self.passes - 1]

    def verify_pass(self):
        """Raise unless the last encoder pass followed the current schedule."""
        if not self.trace or self.trace != schedule(self.current, self.trace[0], self.depth):
            raise RuntimeError(f'Token compression did not run as scheduled: {self.trace}')
