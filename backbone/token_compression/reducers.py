"""Training-free token reduction operators indexed by tokens removed per block.

Every operator keeps CLS at index 0 and removes at most ``(tokens - 1) // 2``
tokens, the ToMe clamp, so all operators share one token schedule per degree.
Sizes are ``None`` until an operator merges tokens; pruning leaves them as is.
"""

import math

import torch
from torch.nn import functional as F

MAX_DEGREE = 24


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
    With two patch tokens left the shared schedule removes one, and the fused
    token then stands for both; degree 24 reaches this in the last block of a
    224-pixel ViT with 12 blocks, where it cannot change the class token.
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
    if left < 0:
        raise ValueError('Too few tokens to fuse')
    with torch.no_grad():
        kept = scores.detach().topk(left, dim=1, largest=True, sorted=True).indices
        pruned = torch.ones_like(scores, dtype=torch.bool).scatter_(1, kept, False)
    # The fusion weights stay in the graph, as in the official code.
    fused = (patches * (scores * pruned).unsqueeze(-1).to(patches.dtype)).sum(dim=1, keepdim=True)
    kept = patches.gather(1, kept.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
    return torch.cat((tokens[:, :1], kept, fused), dim=1), size


def energy_merge_tokens(tokens, metric, size, count, margin, protect):
    """PiToMe merging with a per-block budget.

    As in the official code (hchautran/PiToMe, algo/pitome/merge.py): the
    energy of a patch token is the mean over all patch tokens of
    ``ELU(cosine similarity - margin)``, high inside large groups of similar
    tokens. With ``protect`` only the ``2 * count`` tokens of highest energy
    may merge: in energy order they alternate between sources and
    destinations, and every source joins its most similar destination
    (``pitome``). Without it all patch tokens alternate between both sets and
    the ``count`` sources closest to a destination are merged
    (``pitome_bsm``). Averages are weighted by size. The official code derives
    the count from a keep ratio; here it is the shared per-block budget. The
    result is ordered as there: CLS, the untouched tokens, the destinations.
    """
    if metric.shape[:2] != tokens.shape[:2]:
        raise ValueError('Image tokens and key metric must align')
    count = planned(count, tokens.shape[1])
    if count == 0:
        return tokens, size
    if size is None:
        size = tokens.new_ones(*tokens.shape[:2], 1)
    with torch.no_grad():
        direction = F.normalize(metric[:, 1:].detach().float(), dim=-1)
        affinity = direction @ direction.transpose(-2, -1)
        order = F.elu(affinity - margin).mean(dim=-1).argsort(dim=-1, descending=True)
        if protect:
            sources, targets, rest = order[:, :2 * count:2], order[:, 1:2 * count:2], order[:, 2 * count:]
        else:
            sources, targets = order[:, ::2], order[:, 1::2]
        pairs = affinity.gather(1, sources.unsqueeze(-1).expand(-1, -1, affinity.shape[-1]))
        pairs = pairs.gather(2, targets.unsqueeze(1).expand(-1, sources.shape[1], -1))
        strength, destination = pairs.max(dim=-1)
        if not protect:
            ranked = strength.argsort(dim=-1, descending=True)
            rest = sources.gather(1, ranked[:, count:])
            sources, destination = sources.gather(1, ranked[:, :count]), destination.gather(1, ranked[:, :count])
    patches, patch_size = tokens[:, 1:], size[:, 1:]

    def take(values, index):
        return values.gather(1, index.unsqueeze(-1).expand(-1, -1, values.shape[-1]))

    weighted = take(patches * patch_size, targets).scatter_add(
        1, destination.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]), take(patches * patch_size, sources))
    merged_size = take(patch_size, targets).scatter_add(1, destination.unsqueeze(-1), take(patch_size, sources))
    return (torch.cat((tokens[:, :1], take(patches, rest), weighted / merged_size), dim=1),
            torch.cat((size[:, :1], take(patch_size, rest), merged_size), dim=1))


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


class PiToMe:
    """Energy-based merging that protects isolated tokens (Tran et al., NeurIPS 2024), ``count`` tokens per block.

    Margin and block split follow the official DeiT patch (algo/pitome/patch/deit.py): the margin falls
    linearly from 0.75 over the blocks, and only the second half of the blocks protects tokens. The paper
    states 0.9 and protection in every block. Proportional attention is the plug-in's, as the paper says;
    the official block does not pass sizes to its attention.
    """
    name = 'pitome'
    margin = .75

    def __call__(self, tokens, size, count, signals):
        margin = self.margin - self.margin * signals.block / signals.depth
        protect = signals.block >= math.ceil(signals.depth / 2)
        return energy_merge_tokens(tokens, signals.metric, size, count, margin, protect)


OPERATORS = {operator.name: operator for operator in (ToMe, ClsPrune, EViT, PiToMe)}
