"""Explicit CPU requests must not query, allocate, or select any GPU."""

import pytest
import torch

from utils.conf import get_device


@pytest.mark.parametrize('cached', (False, True))
def test_explicit_cpu_overrides_cached_device_without_cuda_calls(monkeypatch, cached):
    if cached:
        monkeypatch.setattr(get_device, 'device', torch.device('cuda:0'), raising=False)
    else:
        monkeypatch.delattr(get_device, 'device', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CPU selection queried CUDA'))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: pytest.fail('CPU selection enumerated CUDA'))
    assert get_device('cpu') == torch.device('cpu')
    assert get_device() == torch.device('cpu')
