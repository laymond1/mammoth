"""EASE arithmetic checks using only small, precomputed feature vectors.

No image backbone, training loop, optimizer, model download, or external
service is used. Expected values are explicit cosine and softmax calculations.
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch


# Import this standalone helper without triggering models package discovery.
_CLASSIFIER_FILE = Path(__file__).resolve().parents[1] / "models/ease_utils/classifier.py"
_SPEC = importlib.util.spec_from_file_location("_ease_classifier_arithmetic", _CLASSIFIER_FILE)
_CLASSIFIER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CLASSIFIER)
PrototypeClassifier = _CLASSIFIER.PrototypeClassifier
CosineHead = _CLASSIFIER.CosineHead


def test_single_task_cosine_is_finite_without_off_diagonal_division():
    classifier = PrototypeClassifier(2)
    classifier.expand(2, 1, [(0, 2)])
    classifier.update_current(torch.tensor([[2.0, 0.0], [0.0, 4.0]]), past_classes=0)
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    expected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    logits = classifier(features, alpha=100.0)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(classifier(features, use_reweight=False), expected)


def test_two_task_reweighting_retains_own_space_and_scales_other_space():
    classifier = PrototypeClassifier(2)
    classifier.expand(2, 2, [(0, 1), (1, 2)])
    classifier.weight.copy_(torch.tensor([[2.0, 0.0, 0.0, 3.0],
                                         [0.0, 5.0, 7.0, 0.0]]))
    features = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    # First sample aligns with each class's own space; the second only with
    # its other space. There is one other space, so its coefficient is 0.1.
    torch.testing.assert_close(classifier(features, alpha=0.1),
                               torch.tensor([[1.0, 1.0], [0.1, 0.1]]))


def test_three_task_reweighting_divides_alpha_between_two_other_spaces():
    classifier = PrototypeClassifier(2)
    classifier.expand(3, 3, [(0, 1), (1, 2), (2, 3)])
    classifier.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                                         [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                                         [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]]))
    features = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
    # Two off-diagonal contributions of 0.05 add to 0.1, rather than 0.2.
    torch.testing.assert_close(classifier(features, alpha=0.1),
                               torch.tensor([[1.0, 1.0, 1.0], [0.1, 0.1, 0.1]]))


def test_nonuniform_task_boundaries_select_the_correct_own_adapter():
    classifier = PrototypeClassifier(2)
    classifier.expand(5, 3, [(0, 2), (2, 3), (3, 5)])
    classifier.weight.copy_(torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]] * 5))
    # Per-space similarities are +1, 0, -1. Off-diagonal weight is 0.2/2.
    features = torch.tensor([[1.0, 0.0, 0.0, 1.0, -1.0, 0.0]])
    torch.testing.assert_close(classifier(features, alpha=0.2),
                               torch.tensor([[0.9, 0.9, 0.0, -0.9, -0.9]]))
    assert classifier.task_offsets == [(0, 2), (2, 3), (3, 5)]


def test_unweighted_cosine_normalizes_the_whole_concatenated_vector():
    classifier = PrototypeClassifier(2)
    classifier.expand(2, 2, [(0, 1), (1, 2)])
    classifier.weight.copy_(torch.tensor([[3.0, 0.0, 0.0, 4.0],
                                         [0.0, 2.0, 0.0, 0.0]]))
    features = torch.tensor([[3.0, 0.0, 0.0, 4.0]])
    torch.testing.assert_close(classifier(features, use_reweight=False), torch.tensor([[1.0, 0.0]]))


def test_expansion_keeps_observed_blocks_and_zeros_new_rows_and_columns():
    classifier = PrototypeClassifier(2).double()
    classifier.expand(2, 1, [(0, 2)])
    classifier.update_current(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64), 0)
    classifier.expand(3, 2, [(0, 2), (2, 3)])
    expected = torch.tensor([[1.0, 2.0, 0.0, 0.0],
                             [3.0, 4.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(classifier.weight, expected)
    assert classifier.weight.dtype == torch.float64
    assert classifier.in_features == 4 and classifier.out_features == 3


def test_current_prototypes_preserve_raw_means_and_do_not_replace_old_rows():
    classifier = PrototypeClassifier(2)
    classifier.expand(1, 1, [(0, 1)])
    # Raw mean of [10, 0] and [0, 2] is [5, 1]. Normalizing samples before
    # averaging would produce [0.5, 0.5], and normalizing the mean differs too.
    classifier.update_current(torch.tensor([[5.0, 1.0]]), 0)
    torch.testing.assert_close(classifier.weight, torch.tensor([[5.0, 1.0]]))
    classifier.expand(2, 2, [(0, 1), (1, 2)])
    classifier.update_current(torch.tensor([[8.0, 2.0, 6.0, 3.0]]), 1)
    torch.testing.assert_close(classifier.weight,
                               torch.tensor([[5.0, 1.0, 0.0, 0.0], [8.0, 2.0, 6.0, 3.0]]))


@pytest.mark.parametrize("recalc_sim", [True, False])
def test_two_task_completion_uses_row_softmax_cosine_and_raw_reference_means(recalc_sim):
    classifier = PrototypeClassifier(2).double()
    classifier.expand(1, 1, [(0, 1)])
    classifier.update_current(torch.tensor([[1.0, 0.0]], dtype=torch.float64), 0)
    classifier.expand(3, 2, [(0, 1), (1, 3)])
    current = torch.tensor([[7.0, 0.0, 2.0, 0.0],
                            [0.0, 5.0, 0.0, 4.0]], dtype=torch.float64)
    classifier.update_current(current, 1)
    classifier.complete_old_prototypes(recalc_sim=recalc_sim)
    # The old class has cosine similarities [1, 0] to the two reference
    # classes. Row softmax produces [e/(e+1), 1/(e+1)].
    denominator = math.e + 1.0
    expected = torch.tensor([[1.0, 0.0, 2.0 * math.e / denominator, 4.0 / denominator]],
                            dtype=torch.float64)
    torch.testing.assert_close(classifier.weight[:1], expected)
    torch.testing.assert_close(classifier.weight[1:], current)


@pytest.mark.parametrize("recalc_sim, expected_old_block", [(True, 4.0), (False, 2.0)])
def test_third_task_recalculation_updates_previously_completed_missing_blocks(recalc_sim, expected_old_block):
    classifier = PrototypeClassifier(1)
    classifier.expand(1, 1, [(0, 1)])
    classifier.update_current(torch.tensor([[1.0]]), 0)
    classifier.expand(2, 2, [(0, 1), (1, 2)])
    classifier.update_current(torch.tensor([[1.0, 2.0]]), 1)
    classifier.complete_old_prototypes()
    torch.testing.assert_close(classifier.weight, torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
    classifier.expand(3, 3, [(0, 1), (1, 2), (2, 3)])
    classifier.update_current(torch.tensor([[1.0, 6.0, 10.0]]), 2)
    classifier.complete_old_prototypes(recalc_sim=recalc_sim)
    # Reset completion uses both later classes for the old task-1 block:
    # cosine similarities [1, 1] give equal weights, and (2 + 6)/2 = 4.
    # Incremental-only completion leaves the earlier completed value 2 intact.
    expected = torch.tensor([[1.0, expected_old_block, 10.0],
                             [1.0, 2.0, 10.0],
                             [1.0, 6.0, 10.0]])
    torch.testing.assert_close(classifier.weight, expected)


def test_semantic_completion_matches_official_cosine_epsilon_for_small_vectors():
    classifier = PrototypeClassifier(2).double()
    classifier.expand(1, 1, [(0, 1)])
    classifier.update_current(torch.tensor([[1e-10, 0.0]], dtype=torch.float64), 0)
    classifier.expand(3, 2, [(0, 1), (1, 3)])
    classifier.update_current(torch.tensor([[1e-10, 0.0, 2.0, 0.0],
                                            [0.0, 1e-10, 0.0, 4.0]], dtype=torch.float64), 1)
    classifier.complete_old_prototypes()
    # Both norms are clamped to 1e-8, hence aligned cosine is 1e-4.
    ratio = math.exp(1e-4)
    expected = torch.tensor([[2.0 * ratio / (ratio + 1.0), 4.0 / (ratio + 1.0)]], dtype=torch.float64)
    torch.testing.assert_close(classifier.weight[:1, 2:], expected)


def test_zero_source_prototype_has_finite_uniform_semantic_completion():
    classifier = PrototypeClassifier(2)
    classifier.expand(1, 1, [(0, 1)])
    classifier.update_current(torch.tensor([[0.0, 0.0]]), 0)
    classifier.expand(3, 2, [(0, 1), (1, 3)])
    classifier.update_current(torch.tensor([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 4.0]]), 1)
    classifier.complete_old_prototypes()
    torch.testing.assert_close(classifier.weight[:1, 2:], torch.tensor([[1.0, 2.0]]))
    assert torch.isfinite(classifier.weight).all()


def test_registered_prototype_state_reloads_into_empty_classifier_with_dynamic_shapes():
    original = PrototypeClassifier(2)
    original.expand(5, 3, [(0, 2), (2, 3), (3, 5)])
    original.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                       [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                                       [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                                       [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                                       [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]))
    state = {name: value.clone() for name, value in original.state_dict().items()}
    restored = PrototypeClassifier(2)
    restored.load_state_dict(state)
    assert restored.task_offsets == original.task_offsets
    assert restored.n_adapters == 3 and restored.in_features == 6 and restored.out_features == 5
    torch.testing.assert_close(restored.weight, original.weight)
    torch.testing.assert_close(restored.sigma, torch.ones(1))
    assert set(dict(restored.named_buffers())) == {"weight", "sigma", "_task_offsets"}
    assert list(restored.parameters()) == []
    assert not restored.weight.requires_grad and not restored.sigma.requires_grad


def test_prototype_checkpoint_rejects_incompatible_feature_dimension():
    source = PrototypeClassifier(2)
    source.expand(2, 2, [(0, 1), (1, 2)])
    with pytest.raises(RuntimeError, match="incompatible feature dimensions"):
        PrototypeClassifier(3).load_state_dict(source.state_dict())


def test_proxy_head_scale_and_cosine_match_official_training_head():
    head = CosineHead(2, 2)
    assert head.weight.requires_grad and head.sigma.requires_grad
    assert torch.all(head.weight.abs() <= 1.0 / math.sqrt(2))
    torch.testing.assert_close(head.sigma, torch.ones(1))
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[3.0, 0.0], [0.0, 7.0]]))
        head.sigma.fill_(2.0)
    torch.testing.assert_close(head(torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
                               torch.tensor([[2.0, 0.0], [0.0, 2.0]]))


@pytest.mark.parametrize("alpha", [-0.1, float("nan"), float("inf")])
def test_invalid_reweighting_coefficient_is_rejected(alpha):
    classifier = PrototypeClassifier(2)
    classifier.expand(1, 1, [(0, 1)])
    with pytest.raises(ValueError, match="alpha"):
        classifier(torch.tensor([[1.0, 0.0]]), alpha=alpha)


@pytest.mark.parametrize("n_classes, n_adapters, offsets", [
    (2, 2, [(0, 2)]),
    (3, 2, [(0, 1), (2, 3)]),
    (3, 2, [(0, 1), (1, 1)]),
    (4, 2, [(0, 1), (1, 3)]),
])
def test_invalid_task_intervals_are_rejected(n_classes, n_adapters, offsets):
    with pytest.raises(ValueError):
        PrototypeClassifier(2).expand(n_classes, n_adapters, offsets)


def test_expansion_cannot_redefine_or_remove_observed_task_boundaries():
    classifier = PrototypeClassifier(2)
    classifier.expand(2, 1, [(0, 2)])
    with pytest.raises(ValueError, match="cannot remove or change"):
        classifier.expand(3, 2, [(0, 1), (1, 3)])
    classifier.expand(3, 2, [(0, 2), (2, 3)])
    with pytest.raises(ValueError, match="cannot remove or change"):
        classifier.expand(2, 1, [(0, 2)])


def test_current_update_rejects_old_rows_wrong_shapes_and_nonfinite_means():
    classifier = PrototypeClassifier(2)
    classifier.expand(3, 2, [(0, 1), (1, 3)])
    with pytest.raises(ValueError, match="boundary"):
        classifier.update_current(torch.zeros(2, 4), past_classes=0)
    with pytest.raises(ValueError, match="shape"):
        classifier.update_current(torch.zeros(3, 4), past_classes=1)
    with pytest.raises(ValueError, match="finite floating-point"):
        classifier.update_current(torch.tensor([[0.0, 0.0, float("nan"), 0.0], [0.0] * 4]), 1)
    with pytest.raises(ValueError, match="finite floating-point"):
        classifier.update_current(torch.zeros(2, 4, dtype=torch.long), 1)


def test_uninitialized_classifier_and_incomplete_feature_concatenation_are_rejected():
    classifier = PrototypeClassifier(2)
    with pytest.raises(ValueError, match="Expand"):
        classifier.complete_old_prototypes()
    with pytest.raises(ValueError, match="Expand"):
        classifier(torch.zeros(1, 2))
    classifier.expand(2, 2, [(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="concatenate"):
        classifier(torch.zeros(1, 2))


@pytest.mark.parametrize("feature_dim", [0, -1, True, 1.5])
def test_feature_dimension_must_be_a_positive_integer(feature_dim):
    with pytest.raises(ValueError, match="feature_dim"):
        PrototypeClassifier(feature_dim)
    with pytest.raises(ValueError, match="feature_dim"):
        CosineHead(feature_dim, 2)


def test_unsupported_init_ptm_option_is_not_silently_approximated():
    with pytest.raises(TypeError):
        PrototypeClassifier(2, use_init_ptm=True)
