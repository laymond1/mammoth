"""HiDe-Prompt feature distributions, orthogonality loss and classifier alignment.

Ported from ``hide_tii_engine.py`` and ``hide_promtp_wtp_and_tap_engine.py``
at author commit 0ed433334ff2cfb206c5e2899651c3044ec85f92. Statistics contain
aggregates only; temporary features are never retained as replay samples.

The source leaves sklearn's initialization count and random state implicit.
This implementation records an explicit Lloyd/k-means++/10-initialization
policy and a per-class seed (seed + class ID). An independent reference must
use that same policy. Neither temporal mean scaling nor logit normalization
is present in the pinned source, so neither is applied here.
"""

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ClassFeatureStatistics(nn.Module):
    """Reloadable, immutable per-class distributions for one feature branch.

    Make separate instances for TII and prompted features. ``class_tasks``
    records the task where each class was observed and prevents reading a
    future distribution. Buffer sizes are fixed to make strict checkpoint
    round trips independent of how many tasks have already completed.
    """

    METHODS = ('multi-centroid', 'variance', 'covariance')

    def __init__(self, num_classes: int, feature_dim: int, n_centroids: int = 10,
                 method: str = 'multi-centroid', seed: int = 0, kmeans_n_init: int = 10):
        super().__init__()
        if num_classes < 1 or feature_dim < 1 or n_centroids < 1 or kmeans_n_init < 1:
            raise ValueError('Statistics dimensions and initialization count must be positive.')
        if method not in self.METHODS:
            raise ValueError(f'Unsupported HiDe statistics method: {method}')
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.n_centroids = n_centroids
        self.method = method
        self.seed = int(seed)
        self.kmeans_n_init = kmeans_n_init
        count = n_centroids if method == 'multi-centroid' else 1
        self.register_buffer('means', torch.zeros(num_classes, count, feature_dim, dtype=torch.float64))
        self.register_buffer('variances', torch.zeros_like(self.means))
        self.register_buffer('valid', torch.zeros(num_classes, count, dtype=torch.bool))
        self.register_buffer('class_tasks', torch.full((num_classes,), -1, dtype=torch.long))
        # Dense covariance storage is allocated only when explicitly requested.
        shape = (num_classes, feature_dim, feature_dim) if method == 'covariance' else (0,)
        self.register_buffer('covariances', torch.zeros(shape, dtype=torch.float64))

    def get_extra_state(self):
        return {'version': 1, 'num_classes': self.num_classes, 'feature_dim': self.feature_dim,
                'n_centroids': self.n_centroids, 'method': self.method, 'seed': self.seed,
                'kmeans_n_init': self.kmeans_n_init, 'kmeans_init': 'k-means++',
                'kmeans_algorithm': 'lloyd', 'kmeans_random_state': 'seed + class_id'}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('HiDe statistics checkpoint configuration or clustering policy differs.')

    def _class_index(self, class_id):
        class_id = int(class_id)
        if not 0 <= class_id < self.num_classes:
            raise ValueError(f'Class ID {class_id} is outside the configured class set.')
        return class_id

    @torch.no_grad()
    def update_class(self, class_id: int, features: torch.Tensor, task_id: int):
        """Store the source distribution of a newly observed class, once.

        Multi-centroid variances use NumPy population variance (ddof=0).
        The other modes use the source's unbiased torch.cov plus 1e-4.
        Empty clusters fail explicitly rather than silently changing K.
        """
        class_id = self._class_index(class_id)
        if task_id < 0:
            raise ValueError('Task IDs must be nonnegative.')
        if self.class_tasks[class_id].item() >= 0:
            raise ValueError(f'Class {class_id} already has immutable statistics.')
        if features.ndim != 2 or features.shape[1] != self.feature_dim or not features.is_floating_point():
            raise ValueError('Expected floating features with shape [samples, feature_dim].')
        if not torch.isfinite(features).all():
            raise ValueError('Features must be finite.')
        features = features.detach()
        if self.method == 'multi-centroid':
            if features.shape[0] < self.n_centroids:
                raise ValueError('There must be at least n_centroids observations per class.')
            from sklearn.cluster import KMeans
            values = features.cpu().numpy()
            labels = KMeans(n_clusters=self.n_centroids, init='k-means++', n_init=self.kmeans_n_init,
                            algorithm='lloyd', random_state=(self.seed + class_id) % (2 ** 32)).fit_predict(values)
            means, variances = [], []
            for cluster in range(self.n_centroids):
                group = values[labels == cluster]
                if group.shape[0] == 0:
                    raise ValueError('KMeans produced an empty centroid; class statistics were not stored.')
                means.append(torch.tensor(np.mean(group, axis=0), dtype=torch.float64))
                variances.append(torch.tensor(np.var(group, axis=0), dtype=torch.float64))
            means, variances = torch.stack(means), torch.stack(variances)
            covariance = None
        else:
            if features.shape[0] < 2:
                raise ValueError('Covariance and variance statistics need at least two observations.')
            means = features.mean(dim=0, keepdim=True)
            covariance = torch.cov(features.T).reshape(self.feature_dim, self.feature_dim)
            covariance = covariance + torch.eye(self.feature_dim, device=features.device, dtype=features.dtype) * 1e-4
            variances = torch.diagonal(covariance).unsqueeze(0)
        self.means[class_id].copy_(means)
        self.variances[class_id].copy_(variances)
        if self.method == 'covariance':
            self.covariances[class_id].copy_(covariance)
        self.valid[class_id].fill_(True)
        self.class_tasks[class_id] = task_id

    def old_means(self, task_id: int):
        """Return old centroids only, never current or future task means."""
        if task_id < 0:
            raise ValueError('Task IDs must be nonnegative.')
        classes = (self.class_tasks >= 0) & (self.class_tasks < task_id)
        return self.means[classes][self.valid[classes]].detach()

    def sample_alignment(self, class_masks: Sequence[Sequence[int]], task_id: int,
                         batch_size: int, branch: str, generator=None):
        """Draw and shuffle the pinned source's synthetic alignment population.

        Every nonzero-variance centroid contributes batch_size samples for
        TII, or batch_size * 5 for TAP. Zero-variance centroids are skipped
        before adding sampling jitter, exactly as in both author engines.
        """
        masks = _seen_masks(class_masks, task_id)
        samples_per_centroid = _samples_per_centroid(batch_size, branch)
        sampled_features, sampled_labels = [], []
        for observed_task, classes in enumerate(masks):
            for class_id in classes:
                class_id = self._class_index(class_id)
                if self.class_tasks[class_id].item() != observed_task or not self.valid[class_id].all():
                    raise ValueError(f'Missing statistics for class {class_id} in observed task {observed_task}.')
                for centroid in range(self.means.shape[1]):
                    mean = self.means[class_id, centroid].float()
                    variance = self.variances[class_id, centroid]
                    if self.method == 'multi-centroid' and variance.mean().item() == 0:
                        continue
                    if self.method == 'covariance':
                        covariance = self.covariances[class_id].float()
                    else:
                        covariance = torch.diag(variance)
                        if self.method == 'multi-centroid':
                            covariance = covariance + torch.eye(self.feature_dim, device=mean.device) * 1e-4
                        covariance = covariance.float()
                    factor = torch.linalg.cholesky(covariance)
                    noise = torch.randn(samples_per_centroid, self.feature_dim, device=mean.device,
                                        dtype=torch.float32, generator=generator)
                    sampled_features.append(mean + noise @ factor.T)
                    sampled_labels.append(torch.full((samples_per_centroid,), class_id, device=mean.device, dtype=torch.long))
        if not sampled_features:
            raise ValueError('No nondegenerate centroids are available for classifier alignment.')
        features, labels = torch.cat(sampled_features), torch.cat(sampled_labels)
        # The author's permutation is on CPU, even when feature draws use CUDA.
        permutation_device = 'cpu' if generator is None else generator.device
        order = torch.randperm(features.shape[0], device=permutation_device, generator=generator).to(features.device)
        return features[order], labels[order]


def orthogonality_loss(features: torch.Tensor, statistics=None, task_id=None, reg: float = 0.1,
                       temperature: float = 0.8):
    """Source ``orth_loss``: unnormalized Gram-matrix identity-target CE.

    Actual class labels are intentionally absent: every centroid or sample
    has its own identity target. Stored centroids are detached aggregates.
    """
    if features.ndim != 2 or not features.shape[0] or not torch.isfinite(features).all():
        raise ValueError('Orthogonality loss needs a nonempty matrix of finite features.')
    if temperature <= 0 or reg < 0:
        raise ValueError('Temperature must be positive and regularization nonnegative.')
    if statistics is not None:
        if task_id is None:
            raise ValueError('A task ID is required to exclude future statistics.')
        means = statistics.old_means(task_id).to(device=features.device)
        # NumPy centroid means are float64 in the author code. The other
        # methods retain torch feature precision (FP32 for this protocol).
        if statistics.method != 'multi-centroid':
            means = means.to(dtype=features.dtype)
        if means.shape[0]:
            features = torch.cat((means, features), dim=0)
    similarity = features @ features.T / temperature
    targets = torch.arange(similarity.shape[0], device=features.device)
    return reg * F.cross_entropy(similarity, targets)


def _seen_masks(class_masks, task_id):
    if not 0 <= task_id < len(class_masks):
        raise ValueError('Task ID must identify an observed class mask.')
    masks = [tuple(int(value) for value in classes) for classes in class_masks[:task_id + 1]]
    flat = [value for classes in masks for value in classes]
    if any(not classes for classes in masks) or len(set(flat)) != len(flat):
        raise ValueError('Observed task class masks must be nonempty and disjoint.')
    return masks


def _samples_per_centroid(batch_size, branch):
    if branch not in ('tii', 'prompt'):
        raise ValueError('Alignment branch must be tii or prompt.')
    if batch_size < 1:
        raise ValueError('Alignment batch size must be positive.')
    return batch_size * (5 if branch == 'prompt' else 1)


def mask_seen_logits(logits: torch.Tensor, seen_classes: Sequence[int]):
    """Preserve raw source logits and mask all classes outside seen tasks."""
    if logits.ndim != 2:
        raise ValueError('Logits must have shape [samples, classes].')
    indices = torch.as_tensor(seen_classes, device=logits.device, dtype=torch.long)
    if not indices.numel() or (indices < 0).any() or (indices >= logits.shape[1]).any():
        raise ValueError('Seen classes must be valid classifier column indices.')
    unavailable = torch.ones(logits.shape[1], device=logits.device, dtype=torch.bool)
    unavailable[indices] = False
    return logits.masked_fill(unavailable.unsqueeze(0), float('-inf'))


def align_classifier(model: nn.Module, statistics: ClassFeatureStatistics,
                     class_masks: Sequence[Sequence[int]], task_id: int, branch: str,
                     batch_size: int, epochs: int = 30, lr: float = 0.005,
                     weight_decay: float = 0.0, generator=None):
    """Run source classifier alignment using aggregate-generated features.

    ``model(features, fc_only=True)`` must return ``{'logits': tensor}``.
    The TII model selects trainable MLP/head parameters; the prompted model
    excludes prompt parameters. Optimizer steps are bounded by the number
    of *old classes*, not the full synthetic population or centroid count.
    This unusual author loop is intentionally preserved.
    The effective author CIFAR weight decay is zero: the optimizer group
    overrides the SGD constructor's 5e-4 with the parsed weight_decay value.
    """
    masks = _seen_masks(class_masks, task_id)
    samples_per_centroid = _samples_per_centroid(batch_size, branch)
    if epochs < 0 or lr <= 0 or weight_decay < 0:
        raise ValueError('Alignment epochs, learning rate and weight decay are invalid.')
    seen_classes = [value for classes in masks for value in classes]
    old_classes = sum(len(classes) for classes in masks[:-1])
    result = {'branch': branch, 'task_id': task_id, 'epochs': epochs, 'optimizer_steps': 0,
              'old_classes': old_classes, 'seen_classes': len(seen_classes),
              'samples_per_centroid': samples_per_centroid, 'sampled_examples_per_epoch': [],
              'losses': [], 'learning_rates': [], 'skipped': task_id == 0 or epochs == 0}
    if result['skipped']:
        return result
    params = [parameter for name, parameter in model.named_parameters()
              if parameter.requires_grad and (branch == 'tii' or 'prompt' not in name)]
    if not params:
        raise ValueError('The alignment model has no trainable classifier parameters.')
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    training = model.training
    model.train()
    try:
        for _ in range(epochs):
            inputs, targets = statistics.sample_alignment(masks, task_id, batch_size, branch, generator=generator)
            result['sampled_examples_per_epoch'].append(inputs.shape[0])
            result['learning_rates'].append(optimizer.param_groups[0]['lr'])
            losses = []
            for iteration in range(old_classes):
                start, end = iteration * samples_per_centroid, (iteration + 1) * samples_per_centroid
                inp, target = inputs[start:end], targets[start:end]
                if not inp.shape[0]:
                    raise ValueError('Synthetic features are insufficient for the source old-class alignment loop.')
                logits = model(inp, fc_only=True)['logits']
                loss = F.cross_entropy(mask_seen_logits(logits, seen_classes), target)
                if not torch.isfinite(loss):
                    raise ValueError('Classifier alignment produced nonfinite loss.')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                result['optimizer_steps'] += 1
                losses.append(float(loss.detach()))
            result['losses'].append(sum(losses) / len(losses))
            scheduler.step()
    finally:
        model.train(training)
    return result
