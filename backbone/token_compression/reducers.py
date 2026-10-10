"""Training-free token reduction operators indexed by tokens removed per block.

Every operator keeps CLS at index 0 and removes at most ``(tokens - 1) // 2``
tokens, the ToMe clamp, so all operators share one token schedule per degree.
Sizes are ``None`` until an operator merges tokens; pruning leaves them as is.
"""

import torch
from torch.nn import functional as F

MAX_DEGREE = 16


def valid_degree(value):
    if type(value) is not int or not 0 <= value <= MAX_DEGREE:
        raise ValueError(f'Compression degree must be an integer in [0, {MAX_DEGREE}]')
    return value


def planned(count, tokens):
    """Tokens removed from a sequence of ``tokens`` at a requested count."""
    return min(count, (tokens - 1) // 2)


def schedule(count, tokens, depth):
    """Token counts entering each block when every block removes ``count``."""
    trace = []
    for _ in range(depth):
        trace.append(tokens)
        tokens -= planned(count, tokens)
    return trace


class Signals:
    """Attention quantities of one block that an operator may read.

    ``q`` and ``k`` are the per-head tensors the attention really uses. Keys
    may start with ``prefix`` non-image entries, which never enter the metric.
    """

    def __init__(self, q, k, scale, bias=None, prefix=0, block=0, depth=12):
        self.q, self.k, self.scale, self.bias = q, k, scale, bias
        self.prefix, self.block, self.depth = prefix, block, depth

    @property
    def metric(self):
        return self.k[:, :, self.prefix:].mean(1)

    def cls_attention(self):
        """CLS-to-patch attention averaged over heads: (batch, tokens - 1)."""
        scores = (self.q[:, :, :1] @ self.k.transpose(-2, -1)) * self.scale
        if self.bias is not None:
            scores = scores + self.bias
        return scores.softmax(dim=-1).mean(dim=1)[:, 0, self.prefix + 1:]


def merge_tokens(tokens, metric, size, count):
    """ToMe bipartite soft matching with size-weighted averaging.

    Matches are selected without gradients; averages stay differentiable.
    Same operator as ``models.rep_prompt_utils.merge.weighted_merge`` and
    ``tome.merge`` with a class token, pinned here for compensator banks.
    """
    if metric.shape[:2] != tokens.shape[:2]:
        raise ValueError('Image tokens and key metric must align')
    count = planned(count, tokens.shape[1])
    if count == 0:
        return tokens, size
    if size is None:
        size = tokens.new_ones(*tokens.shape[:2], 1)
    with torch.no_grad():
        direction = F.normalize(metric.detach().float(), dim=-1)
        affinity = direction[:, ::2] @ direction[:, 1::2].transpose(-2, -1)
        # CLS is even index 0 and must stay in the unmatched group.
        affinity[:, 0] = -torch.inf
        strength, destination = affinity.max(dim=-1)
        order = strength.argsort(dim=-1, descending=True)
        selected = order[:, :count]
        retained = order[:, count:].sort(dim=1).values
        matched = destination.gather(1, selected)
    left, right = tokens[:, ::2], tokens[:, 1::2]
    left_size, right_size = size[:, ::2], size[:, 1::2]
    width = tokens.shape[-1]
    weighted = right * right_size
    moved = (left * left_size).gather(1, selected.unsqueeze(-1).expand(-1, -1, width))
    weighted = weighted.scatter_add(1, matched.unsqueeze(-1).expand(-1, -1, width), moved)
    right_size = right_size.scatter_add(1, matched.unsqueeze(-1), left_size.gather(1, selected.unsqueeze(-1)))
    kept = left.gather(1, retained.unsqueeze(-1).expand(-1, -1, width))
    kept_size = left_size.gather(1, retained.unsqueeze(-1))
    return torch.cat((kept, weighted / right_size), dim=1), torch.cat((kept_size, right_size), dim=1)


def prune_tokens(tokens, scores, size, count):
    """Drop the ``count`` patch tokens with the lowest score, keeping order."""
    if scores.shape != (tokens.shape[0], tokens.shape[1] - 1):
        raise ValueError('Pruning needs one score per non-CLS token')
    count = planned(count, tokens.shape[1])
    if count == 0:
        return tokens, size
    with torch.no_grad():
        kept = scores.detach().float().topk(scores.shape[1] - count, dim=1).indices.sort(dim=1).values + 1
        kept = torch.cat((kept.new_zeros(kept.shape[0], 1), kept), dim=1)
    tokens = tokens.gather(1, kept.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
    if size is not None:
        size = size.gather(1, kept.unsqueeze(-1))
    return tokens, size


def fuse_prune_tokens(tokens, scores, size, count):
    """EViT token reorganization with a per-block budget.

    As in the official block (youweiliang/evit, evit.py): the most attended
    patch tokens are kept in descending-attention order, and one appended
    token sums the others weighted by their CLS attention, without
    normalization. ``count`` is the net number of tokens removed, so one more
    token is pruned than by plain pruning and the fused token takes its place.
    """
    if scores.shape != (tokens.shape[0], tokens.shape[1] - 1):
        raise ValueError('Pruning needs one score per non-CLS token')
    if size is not None:
        raise ValueError('EViT has no token sizes and cannot follow a merging operator')
    count = planned(count, tokens.shape[1])
    if count == 0:
        return tokens, size
    patches = tokens[:, 1:]
    left = patches.shape[1] - count - 1
    if left < 1:
        raise ValueError('Too few tokens to keep one and fuse the rest')
    with torch.no_grad():
        kept = scores.detach().topk(left, dim=1, largest=True, sorted=True).indices
        pruned = torch.ones_like(scores, dtype=torch.bool).scatter_(1, kept, False)
    # The fusion weights stay in the graph, as in the official code.
    fused = (patches * (scores * pruned).unsqueeze(-1).to(patches.dtype)).sum(dim=1, keepdim=True)
    kept = patches.gather(1, kept.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
    return torch.cat((tokens[:, :1], kept, fused), dim=1), size


class ToMe:
    """Similarity merging on attention keys (Bolya et al., ICLR 2023)."""
    name = 'tome'

    def __call__(self, tokens, size, count, signals):
        return merge_tokens(tokens, signals.metric, size, count)


class ClsPrune:
    """EViT-style pruning by CLS attention, ``count`` tokens per block, no fused token."""
    name = 'cls-prune'

    def __call__(self, tokens, size, count, signals):
        return prune_tokens(tokens, signals.cls_attention(), size, count)


class EViT:
    """Official EViT reorganization with the fused token (Liang et al., ICLR 2022), ``count`` net tokens per block."""
    name = 'evit'

    def __call__(self, tokens, size, count, signals):
        return fuse_prune_tokens(tokens, signals.cls_attention(), size, count)


OPERATORS = {operator.name: operator for operator in (ToMe, ClsPrune, EViT)}
