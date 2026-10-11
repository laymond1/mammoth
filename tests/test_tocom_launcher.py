"""The launcher's plan for CPS-Prompt and REP, and the FLOPs their own settings are matched on."""

from pathlib import Path

import pytest
import torch

from scripts.tocom import flops, run_cl


@pytest.fixture(autouse=True)
def cpu_policy(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('MAMMOTH_DEVICE', 'cpu')
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(old)


def test_inference_degrees_reduce_flops_as_stated():
    stated = {0: 0, 4: 13, 8: 26, 10: 33, 13: 41, 16: 50}
    for operator in ('tome', 'evit'):
        measured = flops.plain_reduction(stated, operator)
        assert all(abs(measured[degree] * 100 - percent) < 1 for degree, percent in stated.items())


def test_own_settings_are_the_closest_to_tome_in_flops():
    targets, sampled = flops.cps_reductions((8, 16), [*range(147, 152), *range(100, 105)])
    for degree, keep in flops.closest(targets, {keep: value[1] for keep, value in sampled.items()}).items():
        assert sampled[keep][0] == run_cl.OWN['cps-prompt'][degree]
        assert abs(sampled[keep][1] - targets[degree]) < .003
    targets, own = flops.rep_reductions((8, 16), [*range(25, 30), *range(69, 74)])
    chosen = flops.closest(targets, own)
    assert chosen == {degree: run_cl.OWN['rep-prompt'][degree] for degree in (8, 16)}
    assert all(abs(own[chosen[degree]] - targets[degree]) < .005 for degree in chosen)
    assert run_cl.OWN['cps-prompt'][0] == run_cl.OWN['rep-prompt'][0] == 0


def test_plan_gives_each_pass_its_training_degree():
    banks = ['tome', 'evit']
    assert run_cl.grid_plan('coda-prompt', 8, None, banks) == (8, None)
    source, compensations = run_cl.grid_plan('cps-prompt', 0, 16, banks)
    # The query pass was trained on every token, the prompted pass at the matched degree.
    assert source == 0 and compensations['evit'] == ('evit', (0, 16), None)
    assert compensations['tome@0'] == ('tome', (0, 0), ['tome']) and len(compensations) == 4
    source, compensations = run_cl.grid_plan('rep-prompt', 0, 0, banks)
    assert source is None and compensations == {name: (name, (0, 0), None) for name in banks}


def test_arguments_of_the_methods_that_compress_on_their_own():
    common = ['--seed', '1', '--output', str(Path('unused'))]
    cps = run_cl.method_args('cps-prompt', 'tiny', 'weights.npz', 64, 8)
    assert cps['reduction_ratio'] == .237 and int(196 * (1 - cps['reduction_ratio'])) == 149
    rep = run_cl.method_args('rep-prompt', 'tiny', 'weights.npz', 64, 16)
    assert (rep['rep_atom'], rep['rep_ald'], rep['rep_merge_max'], rep['rep_swift']) == (1, 0, 71, 1)
    assert run_cl.method_args('rep-prompt', 'tiny', 'weights.npz', 64, 0)['rep_atom'] == 0
    args = run_cl.parse_args(['--method', 'rep-prompt', '--matched-degree', '8', *common])
    assert args.targets == [0, 4, 8, 10, 13, 16] and args.degree == 0
    for wrong in (['--method', 'cps-prompt'], ['--method', 'er', '--matched-degree', '8'],
                  ['--method', 'cps-prompt', '--matched-degree', '8', '--degree', '8'],
                  ['--method', 'rep-prompt', '--matched-degree', '8', '--vit-size', 'base']):
        with pytest.raises(SystemExit):
            run_cl.parse_args([*wrong, *common])
