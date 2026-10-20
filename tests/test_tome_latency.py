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
    assert unit_id(dict(unit, arm='study', degree=16, patched=False, variant='shrink-masked')) == 'study__tiny__fp16__tome-r16__shrink-masked'


@pytest.mark.parametrize('variant', ['shrink-masked', 'shrink-unmasked', 'real-unmasked'])
def test_timing_controls_keep_the_token_schedule(images, variant, monkeypatch):
    from backbone.token_compression import schedule
    from scripts.token_efficiency import vit
    from scripts.tome_latency import arms
    monkeypatch.setattr(vit.Attention, 'mix', vit.Attention.mix)  # the unmasked variants patch it for the process
    forward, info = arms.build('study', 'tiny', 'fp32', 'cpu', 'tome', 16, ROOT / 'weights', variant=variant)
    with torch.inference_mode():
        out = forward(images)
    assert list(info['state'].trace) == schedule(16, 197, 12)
    assert torch.isfinite(out).all()


def test_kernel_attribution_uses_the_innermost_range_of_the_launch():
    from scripts.tome_latency.profile_blocks import attribute
    base = {'ph': 'X', 'pid': 1, 'tid': 1}
    events = [
        dict(base, cat='user_annotation', name='blk00', ts=0, dur=100),
        dict(base, cat='user_annotation', name='merge', ts=40, dur=20),
        dict(base, cat='cuda_runtime', name='cudaLaunchKernel', ts=45, dur=3, args={'correlation': 7}),
        dict(base, cat='cuda_runtime', name='cudaLaunchKernel', ts=10, dur=3, args={'correlation': 8}),
        dict(base, cat='cuda_runtime', name='cudaLaunchKernel', ts=500, dur=3, args={'correlation': 9}),
        dict(base, cat='kernel', name='a', pid=0, tid=7, ts=200, dur=5, args={'correlation': 7}),
        dict(base, cat='kernel', name='b', pid=0, tid=7, ts=210, dur=3, args={'correlation': 8}),
        dict(base, cat='kernel', name='c', pid=0, tid=7, ts=220, dur=2, args={'correlation': 9}),
        # a GEMM launched through the driver API and a memset of a sort, both inside the merge range
        dict(base, cat='cuda_driver', name='cuLaunchKernel', ts=50, dur=2, args={'correlation': 10}),
        dict(base, cat='cuda_runtime', name='cudaMemsetAsync', ts=52, dur=1, args={'correlation': 11}),
        dict(base, cat='kernel', name='gemm', pid=0, tid=7, ts=230, dur=4, args={'correlation': 10}),
        dict(base, cat='gpu_memset', name='Memset', pid=0, tid=7, ts=240, dur=1, args={'correlation': 11}),
    ]
    table = attribute(events)
    assert (table[('blk00', 'merge')]['kernels'], table[('blk00', 'merge')]['gpu_us']) == (3, 10)
    assert table[('blk00', 'rest')]['kernels'] == 1
    assert table[('outside', 'embed-classify')]['kernels'] == 1
    assert table[('blk00', 'merge')]['cpu_us'] == 20
