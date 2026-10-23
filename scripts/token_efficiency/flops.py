"""Analytic FLOPs per image (one multiply-add = 2 FLOPs) from the token count each block sees.

Counted: patch embedding, the matrix products of every block (qkv, QK^T, AV, projection, MLP),
the classifier, and the matrix products an operator adds to choose tokens (ToMe and PiToMe
similarity, the class-token attention EViT and CPS read). Not counted: norms, softmax,
activations, sorting and gathers; they cost time, which the device measurements show, but
almost no FLOPs.
"""
from .reducers import planned
from .vit import DEPTH, PATCHES, SIZES

HEAD_DIM = 64


def width_of(size):
    return SIZES[size][0]


def attention(tokens, width, keys=None):
    keys = tokens if keys is None else keys
    return 6 * tokens * width ** 2 + 4 * tokens * keys * width + 2 * tokens * width ** 2


def mlp(tokens, width):
    return 16 * tokens * width ** 2


def block(tokens, width):
    return attention(tokens, width) + mlp(tokens, width)


def embed(width, patches=PATCHES):
    return 2 * patches * 3 * 16 * 16 * width


def head(width, classes=1000):
    return 2 * width * classes


def cls_attention(tokens, width):
    """Class-token attention row over all tokens, all heads."""
    return 2 * tokens * width


def dense(size, tokens=PATCHES + 1):
    width = width_of(size)
    return embed(width, tokens - 1) + DEPTH * block(tokens, width) + head(width)


def inblock(size, operator, degree):
    """Returns (total, operator overhead) for an in-block operator at a per-block degree."""
    width = width_of(size)
    total, overhead, tokens = embed(width) + head(width), 0, PATCHES + 1
    for _ in range(DEPTH):
        total += attention(tokens, width)
        count = planned(degree, tokens)
        if count:
            if operator == 'tome':
                overhead += 2 * (tokens // 2 + tokens % 2) * (tokens // 2) * HEAD_DIM
            elif operator == 'pitome':
                overhead += 2 * (tokens - 1) ** 2 * HEAD_DIM
            elif operator in ('evit', 'cls-prune'):
                overhead += cls_attention(tokens, width)
        tokens -= count
        total += mlp(tokens, width)
    return total + overhead, overhead


def scheduled(size, operator, requests):
    """(total, overhead) for a per-block schedule; Representation Shift prunes after the MLP, PPT before it."""
    width = width_of(size)
    total, overhead, tokens = embed(width) + head(width), 0, PATCHES + 1
    after = operator == 'repshift'
    for value in requests:
        count = planned(int(tokens * value) if 0 < value < 1 else int(value), tokens)
        total += attention(tokens, width)
        if count and operator == 'ppt':
            # significance row, and ToMe matching as if every image pooled (an upper bound)
            overhead += cls_attention(tokens, width) + 2 * (tokens // 2 + tokens % 2) * (tokens // 2) * HEAD_DIM
        if after:
            total += mlp(tokens, width)
            tokens -= count
        else:
            tokens -= count
            total += mlp(tokens, width)
    return total + overhead, overhead


def score_at(size, layer):
    """Reading the critical score at ``layer``: blocks before it, then qkv and the class-token row of it."""
    width = width_of(size)
    tokens = PATCHES + 1
    return (layer - 1) * block(tokens, width) + 6 * tokens * width ** 2 + cls_attention(tokens, width)


def kept(size, keep, selector='random', layer=None):
    """Restart path: (total, selection overhead); the patch embedding is shared with the selection."""
    width = width_of(size)
    overhead = score_at(size, layer) if selector == 'cps' else 0
    return embed(width) + DEPTH * block(keep + 1, width) + head(width) + overhead, overhead


def continued(size, layer, keep):
    """Continue path: blocks before ``layer`` and its attention on all tokens, the rest on the kept ones."""
    width = width_of(size)
    tokens = PATCHES + 1
    total = embed(width) + (layer - 1) * block(tokens, width) + attention(tokens, width) + cls_attention(tokens, width)
    total += mlp(keep + 1, width) + (DEPTH - layer) * block(keep + 1, width) + head(width)
    return total, cls_attention(tokens, width)


def resolution(size, pixels):
    return dense(size, (pixels // 16) ** 2 + 1), 0


def keep_for(size, target, selector='random', layer=None):
    """Largest number of kept patches whose restart path stays within ``target`` FLOPs, or None if none does."""
    fitting = [keep for keep in range(1, PATCHES + 1) if kept(size, keep, selector, layer)[0] <= target]
    return max(fitting) if fitting else None


def keep_for_continue(size, target, layer):
    fitting = [keep for keep in range(1, PATCHES + 1) if continued(size, layer, keep)[0] <= target]
    return max(fitting) if fitting else None


def ats(size, kept, layers=tuple(range(3, 12))):
    """Per-image FLOPs of ATS from the kept counts (class token included) of each ATS block, batch of one."""
    width = width_of(size)
    total, tokens, step = embed(width) + head(width), PATCHES + 1, 0
    for index in range(DEPTH):
        if index in layers:
            out = int(kept[step])
            step += 1
            # qkv on every input row, the class-token scoring row, attention of the kept queries, projection and MLP
            total += 6 * tokens * width ** 2 + cls_attention(tokens, width) + 4 * out * tokens * width + 2 * out * width ** 2
            total += mlp(out, width)
            tokens = out
        else:
            total += block(tokens, width)
    return total
