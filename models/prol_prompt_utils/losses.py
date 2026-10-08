"""The pinned author's active PROL objective, including its zero NonD term."""

# Adapted from anwarmaxsum/PROL, engine_prol.py, bfff8418 (MIT, Copyright (c) 2025).
# See the accompanying LICENSE.

import torch
from torch.nn import functional as F


def compute_prol_loss(output, labels, past_classes, criterion=F.cross_entropy,
                      intertask_coeff=.001, pull_constraint_coeff=1., pull_constraint_coeff2=1.):
    logits, features = output['logits'], output['pre_logits']
    query_features = output['query_features']
    loss_ce = intertask_coeff * criterion(logits, labels)
    current_targets = torch.where(labels - past_classes >= 0, labels - past_classes, -100)
    loss_ce += criterion(logits[:, past_classes:], current_targets)
    query_normalized, features_normalized = F.normalize(query_features), F.normalize(features)
    generalization = torch.matmul(query_normalized.t(), features_normalized) / logits.shape[0]
    diagonal = torch.diagonal(generalization)
    loss_diagonal = torch.square(torch.ones_like(diagonal) - diagonal).mean()
    # GM @ I is GM; retaining this exact expression preserves active source
    # operations and numerical gradients. It is not an off-diagonal mask.
    non_diagonal = generalization - torch.matmul(generalization,
                                                torch.eye(generalization.shape[0]).to(logits.device, non_blocking=True))
    loss_redundancy = torch.sum(torch.square(non_diagonal)) / (generalization.shape[0] * (generalization.shape[0] - 1))
    loss = loss_ce + (loss_diagonal + loss_redundancy) * .1
    if 'reduce_sim' in output:
        loss = loss - pull_constraint_coeff * output['reduce_sim']
    if 'reduce_sim2' in output:
        loss = loss + pull_constraint_coeff2 * output['reduce_sim2']
    return loss, {'loss_ce': loss_ce, 'loss_diagonal': loss_diagonal, 'loss_redundancy': loss_redundancy,
                  'generalization_matrix': generalization}
