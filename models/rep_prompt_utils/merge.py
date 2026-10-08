"""Weighted bipartite token merging with an untouched CLS token.

This is a native implementation of ToMe-style matching for REP AToM. CODA
prefix K/V do not live in the image sequence and never enter this matcher.
See SOURCE.json for the paper and reconstruction choices.
"""

import torch
from torch.nn import functional as F


def progressive_merges(maximum, depth):
    if type(maximum) is not int or maximum < 0 or depth < 1:
        raise ValueError('AToM requires nonnegative integer maximum and positive depth')
    return [int(maximum * index / (depth - 1)) if depth > 1 else 0 for index in range(depth)]


def weighted_merge(tokens, metric, mass, count):
    """Select matches without gradients; propagate gradients through averages."""
    if type(count) is not int or count < 0:
        raise ValueError('Merge count must be a nonnegative integer')
    if metric.shape[:2] != tokens.shape[:2] or mass.shape != (*tokens.shape[:2], 1):
        raise ValueError('Image token, key metric and attention mass must align')
    count = min(count, (tokens.shape[1] - 1) // 2)
    if count == 0:
        return tokens, mass, 0
    with torch.no_grad():
        direction = F.normalize(metric.detach().float(), dim=-1)
        affinity = direction[:, ::2] @ direction[:, 1::2].transpose(-2, -1)
        # CLS is even index0 and must stay in the unmatched group.
        affinity[:, 0] = -torch.inf
        strength, destination = affinity.max(dim=-1)
        order = strength.argsort(dim=-1, descending=True)
        selected = order[:, :count]
        retained = order[:, count:].sort(dim=1).values
        matched_destination = destination.gather(1, selected)
    left, right = tokens[:, ::2], tokens[:, 1::2]
    left_mass, right_mass = mass[:, ::2], mass[:, 1::2]
    width = tokens.shape[-1]
    weighted_right = right * right_mass
    contributions = (left * left_mass).gather(1, selected.unsqueeze(-1).expand(-1, -1, width))
    weighted_right = weighted_right.scatter_add(1, matched_destination.unsqueeze(-1).expand(-1, -1, width), contributions)
    right_mass = right_mass.scatter_add(1, matched_destination.unsqueeze(-1), left_mass.gather(1, selected.unsqueeze(-1)))
    kept_left = left.gather(1, retained.unsqueeze(-1).expand(-1, -1, width))
    kept_mass = left_mass.gather(1, retained.unsqueeze(-1))
    return (torch.cat((kept_left, weighted_right / right_mass), dim=1),
            torch.cat((kept_mass, right_mass), dim=1), count)
