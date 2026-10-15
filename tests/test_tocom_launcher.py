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
    # Values set for this study where the unified recipe fails at batch 16 on Tiny.
    ewc, sd = (run_cl.method_args(method, 'tiny', 'weights.npz', 16) for method in ('ewc-lora', 'sd-lora'))
    assert (ewc['lora_ewc_merge'], ewc['lora_ewc_lambda'], ewc['lr'], ewc['lora_fc_lr']) == ('product', 1000., .0005, .005)
    assert (sd['lr'], sd['lora_fc_lr']) == (.00025, .002)
    for method in ('er', 'lwf-mc'):
        full = run_cl.method_args(method, 'tiny', 'weights.npz', 16)
        assert (full['optimizer'], full['lr'], full['optim_mom'], full['optim_wd']) == ('sgd', .001, .9, 0.)
    assert run_cl.method_args('er', 'tiny', 'weights.npz', 16)['minibatch_size'] == 16
    args = run_cl.parse_args(['--method', 'rep-prompt', '--matched-degree', '8', *common])
    assert args.targets == [0, 4, 8, 10, 13, 16] and args.degree == 0
    assert (args.batch_size, args.eval_batch_size) == (16, 256)
    for wrong in (['--method', 'cps-prompt'], ['--method', 'er', '--matched-degree', '8'],
                  ['--method', 'cps-prompt', '--matched-degree', '8', '--degree', '8'],
                  ['--method', 'rep-prompt', '--matched-degree', '8', '--vit-size', 'base']):
        with pytest.raises(SystemExit):
            run_cl.parse_args([*wrong, *common])


def test_evaluation_batch_is_fixed_without_touching_the_training():
    from argparse import Namespace
    from torch.utils.data import DataLoader, TensorDataset

    class Tasks:
        """Stands in for a Mammoth dataset: one more task, hence one more test loader, at every call."""

        def __init__(self):
            self.args, self.test_loaders = Namespace(seed=7, num_workers=0), []

        def get_data_loaders(self):
            samples = torch.arange(600.).unsqueeze(1) + 1000 * len(self.test_loaders)
            train = DataLoader(TensorDataset(samples), batch_size=16, shuffle=True)
            self.test_loaders.append(DataLoader(TensorDataset(samples), batch_size=16, shuffle=False))
            return train, self.test_loaders[-1]

    tasks = Tasks()
    run_cl.fixed_evaluation_batch(tasks, 256)
    torch.manual_seed(3)
    expected = torch.rand(4)
    torch.manual_seed(3)
    for task in range(2):
        train, test = tasks.get_data_loaders()
        assert train.batch_size == 16 and test.batch_size == 256 and test is tasks.test_loaders[task]
        batches = [batch[0] for batch in test]
        assert [len(batch) for batch in batches] == [256, 256, 88]
        # Every test sample once, in the order Mammoth would give them.
        assert torch.equal(torch.cat(batches).squeeze(1), torch.arange(600.) + 1000 * task)
    assert torch.equal(torch.rand(4), expected)
    same = Tasks()
    run_cl.fixed_evaluation_batch(same, 16)
    assert same.get_data_loaders()[1] is same.test_loaders[0] and same.test_loaders[0].batch_size == 16


def test_a_budget_path_replaces_the_training_degree():
    common = ['--method', 'coda-prompt', '--operator', 'tome', '--seed', '1', '--output', 'unused',
              '--targets', '0', '4', '8', '12', '16', '20', '24']
    path = ['--path', 'P5', '--schedule', *map(str, [0] * 5 + [24] * 5)]
    args = run_cl.parse_args(common + path)
    assert args.schedule == [0] * 5 + [24] * 5 and args.degree == 0 and args.eval_operators == ['tome']
    assert args.sources == [0, 8, 16, 24]
    args = run_cl.parse_args(common + ['--path', 'P2', '--schedule', *map(str, [24, 24, 16, 8, 0, 0, 0, 0, 0, 0])])
    assert args.degree == 24
    plan = run_cl.switch_plan('tome', [24, 0, 8, 16, 8])
    assert plan == {f'src{d}': ('tome', (d,), None) for d in (0, 8, 16, 24)} and list(plan) == ['src0', 'src8', 'src16', 'src24']
    for broken in (common + ['--schedule', '0', '24'],                                       # no path name
                   common + path + ['--degree', '8'],                                        # both
                   common + ['--path', 'P-1', '--schedule', '0'],                            # name
                   common + ['--path', 'P1', '--schedule', '0', '25'],                       # degree out of range
                   common[:-7] + ['--targets', '0', '16'] + path,                            # 24 not inferred
                   common + path + ['--eval-operators', 'tome', 'evit'],                     # another operator
                   common + ['--path', 'P1'],                                                # path without schedule
                   ['--method', 'cps-prompt', '--matched-degree', '0', '--seed', '1', '--output', 'unused'] + path):
        with pytest.raises(SystemExit):
            run_cl.parse_args(broken)


def test_joint_and_mixed_degrees_are_references_for_budget_paths():
    common = ['--method', 'coda-prompt', '--operator', 'tome', '--seed', '1', '--output', 'unused',
              '--targets', '0', '4', '8', '12', '16', '20', '24']
    mix = ['--mix', '0', '8', '16', '24']
    args = run_cl.parse_args(common + mix)
    assert args.mix == [0, 8, 16, 24] and args.degree == 0 and args.eval_operators == ['tome'] and not args.joint
    args = run_cl.parse_args(common + ['--joint', '--degree', '8'])
    assert args.joint and args.degree == 8 and args.eval_operators == ['tome'] and args.mix is None
    args = run_cl.parse_args(common + mix + ['--path', 'P5', '--schedule', *['0'] * 5, *['24'] * 5])
    assert args.mix == [0, 8, 16, 24] and args.schedule[-1] == 24 and args.degree == 0 and args.eval_operators == ['tome']
    assert run_cl.joint_args('coda-prompt') == {'joint': 1}
    assert run_cl.joint_args('inf-lora') == {'joint': 1, 'lora_classes_per_task': 100}
    for broken in (common + mix + ['--joint'],                                                 # both
                   common + ['--joint', '--path', 'P1', '--schedule', *['0'] * 10],              # joint path
                   common + ['--joint', '--tasks', '2'],                                         # joint stops early
                   common + ['--mix', '0', '25'],                                                # degree out of range
                   common + ['--mix', '8', '16'],                                                # --degree 0 not drawn
                   common[:-7] + ['--targets', '0', '8'] + mix,                                  # 24 not inferred
                   common + mix + ['--eval-operators', 'tome', 'evit'],                          # another operator
                   common + ['--mix', '0', '8', '--path', 'P5', '--schedule', *['0'] * 5, *['24'] * 5],  # 24 not drawn
                   ['--method', 'cps-prompt', '--matched-degree', '0', '--seed', '1', '--output', 'unused'] + mix):
        with pytest.raises(SystemExit):
            run_cl.parse_args(broken)


def test_predictions_keep_class_and_task_il_and_check_the_grid(tmp_path):
    import numpy as np
    from scripts.tocom.reevaluate import Predictions
    keep = Predictions(2)
    logits = torch.tensor([[0., 3., 1., 2.], [5., 0., 1., 6.], [0., 1., 4., 2.]])
    labels = torch.tensor([1, 0, 3])
    keep(0, None, logits, labels)
    keep(0, ('tome', None, 8), logits, labels)
    keep(0, ('tome', 'src0', 8), -logits, labels)
    one, none = 1 / 3 * 100, 0 / 3 * 100
    keep.check({'source': [one], 'operators': {'tome': {'off': {'8': [one]}, 'on': {'src0': {'8': [none]}}}}})
    with pytest.raises(RuntimeError):
        keep.check({'source': [one], 'operators': {'tome': {'off': {'8': [one]}, 'on': {'src0': {'8': [one]}}}}})
    keep.save(tmp_path / 'p.npz')
    saved = np.load(tmp_path / 'p.npz')
    assert saved['label'].tolist() == [1, 0, 3] and saved['loader'].tolist() == [0, 0, 0]
    assert saved['source|pred'].tolist() == [1, 3, 2] and saved['source|within'].tolist() == [1, 0, 2]
    assert saved['tome/off/8|task_max'].tolist() == [[3, 2], [5, 6], [1, 4]]
    with pytest.raises(ValueError):
        keep(0, None, logits[:, :3], labels)
