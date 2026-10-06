import pytest


def test_tak_parser_smoke():
    pytest.importorskip("clip")

    from mammoth import get_avail_args

    required_args, optional_args = get_avail_args(dataset="seq-cifar10", model="tak")

    assert isinstance(required_args, dict)
    assert "clip_backbone" in optional_args
    assert "reg_lambda" in optional_args
    assert "kfac_quantization" in optional_args
