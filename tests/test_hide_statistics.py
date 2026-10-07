"""CPU arithmetic checks on HiDe aggregates and synthetic feature alignment.

No dataset, image training, pretrained checkpoint, GPU or service is used.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F


_PATH = Path(__file__).resolve().parents[1] / 'models/hide_prompt_utils/statistics.py'
_SPEC = importlib.util.spec_from_file_location('_hide_statistics_arithmetic', _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ClassFeatureStatistics = _MODULE.ClassFeatureStatistics
orthogonality_loss = _MODULE.orthogonality_loss
align_classifier = _MODULE.align_classifier
mask_seen_logits = _MODULE.mask_seen_logits


def _variance_statistics():
    stats = ClassFeatureStatistics(6, 2, method='variance', seed=1993)
    for task, classes in enumerate(([1, 3], [0, 4], [2, 5])):
        for class_id in classes:
            values = torch.tensor([[class_id + 1., 0.], [class_id + 3., 2.], [class_id + 5., 1.]])
            stats.update_class(class_id, values, task)
    return stats


def test_multi_centroid_means_population_variances_and_explicit_rng_policy():
    from sklearn.cluster import KMeans
    features = torch.tensor([[-8., 2.], [-6., 4.], [-4., 0.], [8., -2.], [6., -4.], [4., 0.]])
    stats = ClassFeatureStatistics(3, 2, n_centroids=2, seed=1993)
    stats.update_class(1, features, task_id=0)
    labels = KMeans(n_clusters=2, init='k-means++', n_init=10, algorithm='lloyd', random_state=1994).fit_predict(features.numpy())
    expected_mean = torch.from_numpy(np.stack([features.numpy()[labels == k].mean(0) for k in range(2)])).double()
    expected_variance = torch.from_numpy(np.stack([features.numpy()[labels == k].var(0) for k in range(2)])).double()
    torch.testing.assert_close(stats.means[1], expected_mean)
    torch.testing.assert_close(stats.variances[1], expected_variance)
    # Per-feature variances are population variances, not unbiased estimates.
    torch.testing.assert_close(stats.variances[1], torch.full((2, 2), 8. / 3.).double())
    assert stats.means.dtype == stats.variances.dtype == torch.float64
    assert stats.valid[1].all() and not stats.valid[0].any()
    assert stats.get_extra_state()['kmeans_random_state'] == 'seed + class_id'
    assert not any('features' in key for key in stats.state_dict())


@pytest.mark.parametrize('method', ['variance', 'covariance'])
def test_noncentroid_covariance_is_unbiased_and_contains_source_jitter(method):
    features = torch.tensor([[0., 1.], [2., 3.], [4., 2.]])
    stats = ClassFeatureStatistics(2, 2, method=method)
    stats.update_class(0, features, task_id=0)
    # Explicit sample covariance: deviations [-2,0], [0,2], [2,1]
    # around mean [2,2] yield [[4,1],[1,1]], denominator n-1.
    covariance = torch.tensor([[4.0001, 1.], [1., 1.0001]])
    torch.testing.assert_close(stats.means[0, 0], torch.tensor([2., 2.]).double())
    torch.testing.assert_close(stats.variances[0, 0], covariance.diag().double())
    if method == 'covariance':
        torch.testing.assert_close(stats.covariances[0], covariance.double())
    else:
        assert stats.covariances.numel() == 0


def test_new_class_statistics_preserve_old_buffers_and_reject_overwrites():
    stats = ClassFeatureStatistics(3, 2, method='variance')
    first = torch.tensor([[2., 3.], [4., 5.]])
    stats.update_class(0, first, task_id=0)
    snapshot = {key: value.clone() for key, value in stats.named_buffers()}
    stats.update_class(1, first + 10, task_id=1)
    torch.testing.assert_close(stats.means[0], snapshot['means'][0], rtol=0, atol=0)
    torch.testing.assert_close(stats.variances[0], snapshot['variances'][0], rtol=0, atol=0)
    with pytest.raises(ValueError, match='immutable'):
        stats.update_class(0, first + 1, task_id=1)


@pytest.mark.parametrize('method', ['multi-centroid', 'variance', 'covariance'])
def test_registered_statistics_strict_round_trip_and_clustering_policy_validation(method):
    stats = ClassFeatureStatistics(3, 2, n_centroids=2, method=method, seed=1993)
    stats.update_class(1, torch.tensor([[0., 1.], [1., 3.], [5., 0.], [7., 2.]]), task_id=0)
    state = stats.state_dict()
    restored = ClassFeatureStatistics(3, 2, n_centroids=2, method=method, seed=1993)
    restored.load_state_dict(state, strict=True)
    for key, buffer in stats.named_buffers():
        torch.testing.assert_close(dict(restored.named_buffers())[key], buffer, rtol=0, atol=0)
    with pytest.raises(ValueError, match='configuration'):
        ClassFeatureStatistics(3, 2, n_centroids=2, method=method, seed=0).load_state_dict(state)


def test_old_means_exclude_current_and_future_task_statistics():
    stats = _variance_statistics()
    assert stats.old_means(0).shape == (0, 2)
    torch.testing.assert_close(stats.old_means(1), stats.means[[1, 3], 0])
    torch.testing.assert_close(stats.old_means(2), stats.means[[0, 1, 3, 4], 0])
    assert not stats.old_means(2).requires_grad


@pytest.mark.parametrize('with_old', [False, True])
@pytest.mark.parametrize('method', ['variance', 'multi-centroid'])
def test_orthogonality_loss_matches_explicit_unnormalized_gram_ce_and_gradients(with_old, method):
    features = torch.tensor([[1., 2.], [3., -.5]], requires_grad=True)
    independent = features.detach().clone().requires_grad_()
    stats = ClassFeatureStatistics(3, 2, n_centroids=1, method=method)
    stats.update_class(0, torch.tensor([[.1, .2], [.3, .4]]), task_id=0)
    stats.update_class(1, torch.tensor([[50., 60.], [70., 80.]]), task_id=1)
    task = 1 if with_old else 0
    mean = stats.means[0, 0].unsqueeze(0)
    if method != 'multi-centroid':
        mean = mean.float()
    matrix = torch.cat((mean, independent)) if with_old else independent
    expected = F.cross_entropy(matrix @ matrix.T / .8, torch.arange(matrix.shape[0])) * .1
    actual = orthogonality_loss(features, stats, task_id=task, reg=.1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(features.grad, independent.grad, rtol=0, atol=0)
    normalized = F.normalize(matrix, dim=1)
    assert not torch.isclose(actual, F.cross_entropy(normalized @ normalized.T / .8, torch.arange(matrix.shape[0])) * .1)


def test_orthogonality_loss_without_statistics_keeps_first_task_return_and_zero_weight():
    features = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    expected = .1 * math.log1p(math.exp(-1. / .8))
    assert orthogonality_loss(features).item() == pytest.approx(expected)
    zero = orthogonality_loss(features, reg=0.)
    assert zero.item() == 0. and zero.requires_grad


@pytest.mark.parametrize('branch, multiplier', [('tii', 1), ('prompt', 5)])
def test_sampling_uses_seen_class_labels_and_source_centroid_sample_multiplier(branch, multiplier):
    stats = _variance_statistics()
    features, labels = stats.sample_alignment([[1, 3], [0, 4], [2, 5]], 1, 4, branch,
                                               generator=torch.Generator().manual_seed(20))
    assert features.dtype == torch.float32 and labels.dtype == torch.long
    assert features.shape == (4 * 4 * multiplier, 2)
    assert set(labels.tolist()) == {0, 1, 3, 4}
    for class_id in (0, 1, 3, 4):
        assert (labels == class_id).sum().item() == 4 * multiplier
    again = stats.sample_alignment([[1, 3], [0, 4], [2, 5]], 1, 4, branch,
                                    generator=torch.Generator().manual_seed(20))
    torch.testing.assert_close(features, again[0], rtol=0, atol=0)
    torch.testing.assert_close(labels, again[1], rtol=0, atol=0)


def test_variance_sampling_matches_independent_multivariate_normal_with_same_rng():
    stats = ClassFeatureStatistics(1, 2, method='variance')
    stats.update_class(0, torch.tensor([[0., 1.], [2., 3.], [4., 2.]]), task_id=0)
    torch.manual_seed(17)
    actual, labels = stats.sample_alignment([[0]], 0, 7, 'tii')
    torch.manual_seed(17)
    distribution = torch.distributions.MultivariateNormal(stats.means[0, 0].float(),
                                                          torch.diag(stats.variances[0, 0]).float())
    expected = distribution.sample((7,))
    expected = expected[torch.randperm(7)]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert labels.tolist() == [0] * 7


def test_multicentroid_sampler_skips_zero_variance_before_jitter():
    stats = ClassFeatureStatistics(1, 2, n_centroids=2)
    stats.means[0] = torch.tensor([[99., 99.], [1., 2.]])
    stats.variances[0] = torch.tensor([[0., 0.], [.1, .2]])
    stats.valid[0] = True
    stats.class_tasks[0] = 0
    values, labels = stats.sample_alignment([[0]], 0, 3, 'tii', torch.Generator().manual_seed(10))
    assert values.shape == (3, 2) and labels.tolist() == [0] * 3
    assert values.max() < 10.
    stats.variances.zero_()
    with pytest.raises(ValueError, match='nondegenerate'):
        stats.sample_alignment([[0]], 0, 3, 'tii')


def test_raw_logit_mask_does_not_normalize_rescale_or_mutate_input():
    logits = torch.tensor([[10., -20., 999., 3.], [1., 2., 88., 7.]])
    original = logits.clone()
    masked = mask_seen_logits(logits, [0, 1, 3])
    torch.testing.assert_close(masked[:, [0, 1, 3]], original[:, [0, 1, 3]], rtol=0, atol=0)
    assert torch.isneginf(masked[:, 2]).all()
    torch.testing.assert_close(logits, original, rtol=0, atol=0)


class _SyntheticClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(2, 2)
        self.head = nn.Linear(2, 6)
        self.prompt = nn.Parameter(torch.ones(1))
        self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)
        self.batches = []

    def forward(self, inputs, fc_only=False):
        assert fc_only
        self.batches.append(inputs.shape[0])
        return {'logits': self.head(self.mlp(inputs))}


@pytest.mark.parametrize('branch, minibatch', [('tii', 4), ('prompt', 20)])
def test_alignment_preserves_author_old_class_step_bound_cosine_lr_and_frozen_state(branch, minibatch):
    stats = _variance_statistics()
    snapshot = {key: buffer.clone() for key, buffer in stats.named_buffers()}
    torch.manual_seed(25)
    model = _SyntheticClassifier().eval()
    old_head, old_mlp = model.head.weight.detach().clone(), model.mlp.weight.detach().clone()
    report = align_classifier(model, stats, [[1, 3], [0, 4], [2, 5]], 1, branch,
                              batch_size=4, epochs=2, generator=torch.Generator().manual_seed(19))
    assert report['optimizer_steps'] == 2 * 2  # Two old classes, not four seen classes.
    assert model.batches == [minibatch] * 4
    assert report['sampled_examples_per_epoch'] == [minibatch * 4] * 2
    assert report['learning_rates'] == pytest.approx([.005, .0025])
    assert not model.training and not torch.equal(old_head, model.head.weight)
    assert not torch.equal(old_mlp, model.mlp.weight)
    assert model.prompt.item() == model.frozen.item() == 1.
    for key, buffer in stats.named_buffers():
        torch.testing.assert_close(buffer, snapshot[key], rtol=0, atol=0)


def test_alignment_task_zero_does_not_sample_or_touch_model():
    model = _SyntheticClassifier().eval()
    stats = ClassFeatureStatistics(6, 2)
    report = align_classifier(model, stats, [[1, 3]], 0, 'tii', 4)
    assert report['skipped'] and report['optimizer_steps'] == 0
    assert not model.training and not model.batches


@pytest.mark.parametrize('invalid', ['too_few', 'nonfinite', 'dimension'])
def test_failed_update_does_not_partially_mark_class_as_observed(invalid):
    stats = ClassFeatureStatistics(2, 2, n_centroids=2)
    values = {'too_few': torch.ones(1, 2), 'nonfinite': torch.tensor([[1., float('nan')], [2., 3.]]),
              'dimension': torch.ones(2, 3)}[invalid]
    with pytest.raises(ValueError):
        stats.update_class(0, values, task_id=0)
    assert stats.class_tasks[0].item() == -1 and not stats.valid[0].any()


def test_alignment_rejects_missing_seen_distribution_and_duplicate_masks():
    stats = ClassFeatureStatistics(3, 2, method='variance')
    stats.update_class(0, torch.tensor([[0., 1.], [1., 2.]]), task_id=0)
    with pytest.raises(ValueError, match='Missing'):
        stats.sample_alignment([[0], [1]], 1, 4, 'tii')
    with pytest.raises(ValueError, match='disjoint'):
        stats.sample_alignment([[0], [0]], 1, 4, 'tii')
