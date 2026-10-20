"""Training-free in-block methods added for question 2, written from their official code.

PPT (Wu et al., arXiv 2310.01812; github.com/xjwu1024/PPT @706bcdd): at blocks 4, 7 and 10 every image
either merges (ToMe, size-weighted, when the variance of its normalized significance scores is below
7e-5) or prunes the lowest-significance tokens; both remove the same number of tokens. Significance is
the head-summed class-token attention (with the size bias) times the value norm.

Representation Shift (Choi et al., ICCV 2025; github.com/mlvlab/Representation-Shift @a528e02): after
blocks 1, 4 and 7 the block output keeps the tokens whose MLP branch output lies farthest from the MLP
input, ||MLP(LN(x')) - x'||, with the class token always kept. No attention map is read.

Both run in this package's ``Block.reduced``; the token-count rule ``planned`` (at most half) applies
to every operator.
"""
import torch

from .reducers import OPERATORS, merge_tokens

PPT_LAYERS = (3, 6, 9)
REPSHIFT_LAYERS = (0, 3, 6)


class PPT:
    name = 'ppt'
    threshold = 7e-5

    def __call__(self, tokens, size, count, signals):
        if size is None:
            size = tokens.new_ones(*tokens.shape[:2], 1)
        with torch.no_grad():
            score = signals.significance().float()
            score = score / score.sum(dim=1, keepdim=True)
            pool = score.var(dim=1) < self.threshold
        batch, count_in, width = tokens.shape
        out = tokens.new_empty(batch, count_in - count, width)
        out_size = size.new_empty(batch, count_in - count, 1)
        # As in the official block: split the batch by policy (a host synchronisation), reduce, write back.
        pooled, pruned = pool.nonzero().squeeze(-1), (~pool).nonzero().squeeze(-1)
        if len(pooled):
            merged, merged_size = merge_tokens(tokens[pooled], signals.metric[pooled], size[pooled], count)
            out[pooled], out_size[pooled] = merged, merged_size
        if len(pruned):
            with torch.no_grad():
                kept = score[pruned].topk(count_in - 1 - count, dim=1).indices.sort(dim=1).values + 1
                kept = torch.cat((kept.new_zeros(len(kept), 1), kept), dim=1)
            out[pruned] = tokens[pruned].gather(1, kept.unsqueeze(-1).expand(-1, -1, width))
            out_size[pruned] = size[pruned].gather(1, kept.unsqueeze(-1))
        return out, out_size


class RepresentationShift:
    name = 'repshift'
    after_mlp = True

    def __call__(self, tokens, size, count, signals):
        raise RuntimeError('Representation Shift reduces after the MLP; Block.reduced calls after()')

    @staticmethod
    def after(x, update, state, count):
        """Prune the block output ``x + update``, keeping tokens with the largest ||update - x||."""
        with torch.no_grad():
            score = torch.linalg.vector_norm((update - x).float(), dim=-1)
            score[:, 0] = torch.inf
            kept = score.topk(x.shape[1] - count, dim=1).indices.sort(dim=1).values
        out = (x + update).gather(1, kept.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        if state.size is not None:
            state.size = state.size.gather(1, kept.unsqueeze(-1))
        return out


ADDED = {operator.name: operator for operator in (PPT, RepresentationShift)}
ALL_OPERATORS = {**OPERATORS, **ADDED}


def schedule(operator, value, depth=12):
    """Per-block requests: PPT removes ``value`` tokens at its three layers; Representation Shift a fraction."""
    layers = PPT_LAYERS if operator == 'ppt' else REPSHIFT_LAYERS
    return [value if index in layers else 0 for index in range(depth)]
