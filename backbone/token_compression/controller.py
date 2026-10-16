"""Ambient token-compression state shared by the patched blocks of one encoder.

The controller is a plain object referenced from the blocks. It is never a
submodule, parameter or buffer, so optimizers, state dictionaries and weight
digests of the host network are unchanged.
"""

from contextlib import contextmanager

from .reducers import schedule, valid_degree
from .shapes import valid_resolution


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
        # Input resolution. attach() sets the native side length and the patch size, and marks the
        # families whose encoders can take another resolution as resizable; None is always the native one.
        self.native, self.patch, self.resizable = None, None, False
        self.default_resolution = None
        self._resolution = None

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

    def check_resolution(self, value):
        """``value`` as the encoders take it: None for the native resolution, else a valid side length."""
        if value is None or value == self.native:
            return None
        if not self.resizable:
            raise ValueError('The encoders of this family keep their native input resolution')
        return valid_resolution(value, self.patch, self.native)

    @property
    def resolution(self):
        """Side length the encoders resize their input to, or None for the native resolution."""
        value = self.default_resolution if self._resolution is None else self._resolution
        return None if value is None or value == self.native else value

    @contextmanager
    def shape(self, resolution, degree):
        """Run the enclosed forward passes at this input resolution (None: native) and this degree."""
        value = self.native if resolution is None else resolution
        self.check_resolution(value)
        previous, self._resolution = self._resolution, value
        try:
            with self.degree(degree):
                yield self
        finally:
            self._resolution = previous

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
