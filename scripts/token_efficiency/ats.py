"""ATS, Adaptive Token Sampling (Fayyaz et al., ECCV 2022), from the official code
(github.com/adaptivetokensampling/ATS @9271257, ``libs/models/transformers/ats_block.py``).

In blocks 4-12 the attention scores every patch by its head-summed class-token attention times its
value norm, samples 196 points evenly along the normalized CDF of the sorted scores, keeps the nearest
token for each point and drops duplicates. The class token and the unique picks continue; how many
survive differs per image. A batch is padded to its largest count and a 0/1 policy masks padded keys
in every later attention and zeroes padded rows after attention and MLP.

Implemented as in the official block, with two equivalent rewrites: the attention of the selected rows
is one scaled-dot-product call on the selected queries over all keys (rows of a row-wise softmax), and
padded keys are masked with -inf instead of the official ``softmax_with_policy`` (identical for real
tokens up to its 1e-6 smoothing). ``rho`` below 1 samples ``round(rho * N) - 1`` points, the paper's
reduced-budget variant, which the released sampler does not implement; ``rho = 1`` is the release.

As in the release, an image's picks can depend on its batch: from the second ATS block on, padded
tokens (score 0) enter the normalized CDF, which shifts the sampling grid. Accuracy is therefore
measured at a fixed batch size (``accuracy.py``), and a batch of one never pads.
"""
import torch
from torch.nn import functional as F

LAYERS = tuple(range(3, 12))
REFERENCE_TOKENS = 197


def unique_indices(indices, max_value):
    """Official ``get_unique_indices``: sorted unique picks first, duplicates replaced by ``max_value``."""
    ordered = indices.sort(dim=1).values
    shifted = F.pad(ordered[:, 1:], (0, 1), value=1.0)
    return torch.where(shifted - ordered == 0, torch.full_like(indices, max_value), ordered).sort(dim=1).values


def sample_points(normalized_cdf, count):
    """Official ``create_ys``: ``count`` points from the smallest non-zero CDF value to 1."""
    batch = normalized_cdf.shape[0]
    ys = torch.linspace(0, 1.0, count, device=normalized_cdf.device).unsqueeze(0).repeat(batch, 1)
    start = torch.min(normalized_cdf + (normalized_cdf == 0).float() * 1e8, dim=1)[0].unsqueeze(-1).expand_as(ys)
    steps = torch.arange(0, count, device=normalized_cdf.device, dtype=ys.dtype).unsqueeze(0).expand_as(start)
    return start + ((ys * (count - 1)) - start * steps) / (count - 1)


def key_bias(policy, dtype):
    return None if policy is None else ((1 - policy) * torch.finfo(dtype).min).to(dtype)[:, None, None, :, 0]


def masked_block(block, x, policy):
    """A block after an ATS block: padded keys masked, padded rows zeroed after attention and MLP."""
    if policy is None:
        return block(x)
    q, k, v = block.attn.heads(block.norm1(x))
    x = (x + block.attn.mix(q, k, v, key_bias(policy, q.dtype))) * policy
    return (x + block.mlp(block.norm2(x))) * policy


def ats_block(block, x, policy, rho=1.0):
    """One ATS block; returns the reduced tokens, the new policy and each image's kept count."""
    batch, count, width = x.shape
    attention = block.attn
    q, k, v = attention.heads(block.norm1(x))
    if policy is not None:
        mask = policy[:, None]  # (batch, 1, tokens, 1)
        q, k, v = q * mask, k * mask, v * mask
    bias = key_bias(policy, q.dtype)
    with torch.no_grad():
        logits = (q[:, :, :1] @ k.transpose(-2, -1)) * attention.scale
        if bias is not None:
            logits = logits + bias
        cls = logits.float().softmax(dim=-1)[:, :, 0].sum(dim=1)  # (batch, tokens)
        heads, dim = v.shape[1], v.shape[-1]
        norm = torch.linalg.vector_norm(v.float().transpose(1, 2).reshape(batch, count, heads * dim), dim=2)
        score = (cls * norm)[:, 1:]
        score = score / score.sum(dim=1, keepdim=True)
        ordered, order = score.sort(dim=1)
        cdf = ordered.cumsum(dim=1)
        low, high = cdf.min(dim=1, keepdim=True)[0], cdf.max(dim=1, keepdim=True)[0]
        normalized = (cdf - low) / (high - low)
        points = REFERENCE_TOKENS - 1 if rho >= 1 else max(round(rho * count), 8) - 1
        ys = sample_points(normalized, points).unsqueeze(2)
        pad = ys.shape[1] - (count - 1)
        reference = F.pad(normalized, (pad, 0)) if pad > 0 else normalized
        picks = (ys - reference.unsqueeze(1)).abs().min(dim=2)[1] - max(pad, 0)
        picks = unique_indices(picks, count - 1)[:, :count - 1]
        valid = picks != count - 1
        positions = order.gather(1, picks.clamp(max=count - 2)) + 1  # back to token positions
        kept = valid.sum(dim=1) + 1
        size = int(kept.max())
        # Compact as the official nonzero/scatter does: class token, then the unique picks in their order.
        slots = torch.cumsum(valid.long(), dim=1) * valid  # 1-based slot of each valid pick
        index = torch.zeros(batch, size, dtype=torch.long, device=x.device)
        index.scatter_(1, slots, torch.where(valid, positions, torch.zeros_like(positions)))
        index[:, 0] = 0
        new_policy = (torch.arange(size, device=x.device)[None] < kept[:, None]).to(x.dtype).unsqueeze(-1)
    selected = q.gather(2, index[:, None, :, None].expand(-1, q.shape[1], -1, q.shape[-1]))
    out = attention.mix(selected, k, v, bias) * new_policy
    x = (x.gather(1, index.unsqueeze(-1).expand(-1, -1, width)) * new_policy + out) * new_policy
    x = (x + block.mlp(block.norm2(x))) * new_policy
    return x, new_policy, kept


def forward(model, images, rho=1.0, layers=LAYERS, trace=None):
    """ATS forward of ``model``; ``trace`` (a list) receives each ATS block's per-image kept counts."""
    x = model.embed(images)
    policy = None
    for index, block in enumerate(model.blocks):
        if index in layers:
            x, policy, kept = ats_block(block, x, policy, rho)
            if trace is not None:
                trace.append(kept)
        else:
            x = masked_block(block, x, policy)
    return model.classify(x)
