"""Full fine-tuning size and gradient contracts for the shared CL backbone."""

import pytest
import torch

from backbone.vit_augreg import vit_augreg


@pytest.fixture(autouse=True)
def cpu_threads():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize('size,dimension,heads', [('tiny', 192, 3), ('small', 384, 6), ('base', 768, 12)])
def test_augreg_geometry_and_full_trainability(size, dimension, heads):
    model = vit_augreg(100, vit_size=size, pretrained=False)
    assert model.embed_dim == dimension
    assert len(model.blocks) == 12
    assert model.blocks[0].attn.num_heads == heads
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.head.heads[0].weight.shape == (100, dimension)


def test_augreg_rejects_checkpoint_in_random_mode():
    with pytest.raises(ValueError, match='pretrained=True'):
        vit_augreg(100, pretrained=False, vit_checkpoint='/not/a/checkpoint.npz')


def test_pinned_tiny_encoder_and_full_gradient_contract():
    # Actual shared weights, real inputs and all encoder/head gradients. EWC/SI
    # require a stable flat vector, unlike frozen prompt/adapter backbones.
    from models.low_rank_utils.factory import CHECKPOINTS
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(1993)
        model = vit_augreg(100, vit_size='tiny')
        assert model.canonical_pretrained_sha256 == CHECKPOINTS['tiny']['canonical_sha256']
        images = torch.randn(2, 3, 224, 224)
        logits, features = model(images, returnt='both')
        assert logits.shape == (2, 100)
        assert features.shape == (2, 192)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        gradients = model.get_grads()
        assert gradients.shape == model.get_params().shape
        assert torch.isfinite(gradients).all()
        model.set_grads(gradients.clone())
        assert all(parameter.grad is not None for parameter in model.parameters())
        with torch.no_grad():
            assert model(images, returnt='features').shape == (2, 192)
            full_logits, full_features = model(images, returnt='full')
        assert full_logits.shape == (2, 100)
        # Mammoth includes the embedded input followed by every block output.
        assert len(full_features) == 13
    finally:
        torch.set_num_threads(previous_threads)
