"""Input shapes: a resolution and a reduction degree, and the block cost they lead to.

A shape ``(resolution, degree)`` feeds the encoder an image of ``resolution`` pixels per side and removes
``degree`` tokens per block on the schedule of ``reducers``. The cost counts the multiply-adds of the patch
embedding and of every block (attention on the tokens a block receives, MLP on those it keeps), relative
to the native resolution without reduction. Head and normalization layers are left out.
"""

from .reducers import planned, valid_degree


def valid_resolution(value, patch=16, native=224):
    """A side length the patch embedding tiles exactly, of at least two patches and at most ``native``."""
    if type(value) is not int or value % patch or not 2 * patch <= value <= native:
        raise ValueError(f'Resolution must be a multiple of {patch} between {2 * patch} and {native} pixels')
    return value


def parse_shape(text, patch=16, native=224):
    """``"176:8"`` -> ``(176, 8)``."""
    resolution, _, degree = str(text).partition(':')
    try:
        shape = int(resolution), int(degree)
    except ValueError:
        raise ValueError(f'A shape is RESOLUTION:DEGREE, not {text!r}') from None
    return valid_resolution(shape[0], patch, native), valid_degree(shape[1])


def shape_label(resolution, degree):
    return f'{resolution}:{degree}'


def block_macs(resolution, degree, dim=192, depth=12, patch=16, mlp_ratio=4, prefix=1):
    tokens = (resolution // patch) ** 2 + prefix
    total = 3 * patch * patch * dim * (tokens - prefix)
    for _ in range(depth):
        attention = 4 * tokens * dim * dim + 2 * tokens * tokens * dim
        tokens -= planned(degree, tokens)
        total += attention + 2 * mlp_ratio * tokens * dim * dim
    return total


def shape_cost(resolution, degree, dim=192, depth=12, patch=16, native=224):
    """Cost of a shape as a fraction of the native resolution without reduction."""
    return block_macs(resolution, degree, dim, depth, patch) / block_macs(native, 0, dim, depth, patch)
