"""CPU checks for the ToMe latency tools: the three implementations compute the same function.

Run: JE_ARROW_MALLOC_CONF=background_thread:false <venv with timm and tome> -m pytest tests/test_tome_latency.py
"""
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path('/mnt/sdab1/userHome/wslim/.cache/token_efficiency')
pytestmark = pytest.mark.skipif(not (ROOT / 'weights' / 'augreg-tiny.npz').exists(), reason='pinned weights are not on this machine')


def logits(arm, operator, degree, images, patched=False):
    from scripts.tome_latency import arms
    forward, info = arms.build(arm, 'tiny', 'fp32', 'cpu', operator, degree, ROOT / 'weights', patched=patched)
    with torch.inference_mode():
        return forward(images), info


@pytest.fixture(scope='module')
def images():
    from scripts.token_efficiency import data
    array = np.array(np.load(ROOT / 'bench_images.npy', mmap_mode='r')[:4])
    return data.normalize(array, torch.device('cpu'), torch.float32)


@pytest.mark.parametrize('operator,degree', [('dense', 0), ('tome', 16), ('tome', 24)])
def test_official_and_plugin_match_study(images, operator, degree):
    pytest.importorskip('tome')
    reference, _ = logits('study', operator, degree, images)
    for arm in ('plugin', 'official'):
        other, _ = logits(arm, operator, degree, images)
        assert (reference - other).abs().max() < 1e-4
        assert torch.equal(reference.argmax(1), other.argmax(1))


@pytest.mark.parametrize('operator', ['cls-prune', 'evit', 'pitome'])
def test_plugin_matches_study_for_every_operator(images, operator):
    reference, _ = logits('study', operator, 16, images)
    other, _ = logits('plugin', operator, 16, images)
    close = ((reference - other).abs().amax(1) < 1e-4).float().mean()
    # PiToMe ranks near-equal energies: one image of sixteen can pick another token in one implementation
    # after a rounding difference, so it only has to agree on nearly every image.
    assert close >= (0.7 if operator == 'pitome' else 1.0)
    assert torch.equal(reference.argmax(1), other.argmax(1))


def test_token_schedule_is_untouched(images):
    from backbone.token_compression import schedule
    for arm in ('study', 'plugin'):
        _, info = logits(arm, 'tome', 16, images)
        assert list(info['state'].trace) == schedule(16, 197, 12)
    assert schedule(16, 197, 12)[-1] == 21


def test_unit_ids_name_the_arm_and_the_patch():
    from scripts.tome_latency.run_unit import unit_id
    unit = {'arm': 'official', 'size': 'tiny', 'dtype': 'fp16', 'operator': 'tome', 'degree': 0, 'patched': True}
    assert unit_id(unit) == 'official__tiny__fp16__tome-r0-patched'
    assert unit_id(dict(unit, arm='study', operator='dense', patched=False)) == 'study__tiny__fp16__dense'
