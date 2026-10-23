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


def test_degree_24_own_settings_match_tome_24_in_flops():
    targets, sampled = flops.cps_reductions((24,), range(69, 74))
    keep = flops.closest(targets, {keep: value[1] for keep, value in sampled.items()})[24]
    assert keep == 71 and sampled[keep][0] == run_cl.OWN['cps-prompt'][24] == .635
    assert int(196 * (1 - run_cl.OWN['cps-prompt'][24])) == 71 and abs(sampled[keep][1] - targets[24]) < .003
    targets, own = flops.rep_reductions((24,), range(184, 191))
    chosen = flops.closest(targets, own)[24]
    assert chosen == run_cl.OWN['rep-prompt'][24] == 187 and abs(own[chosen] - targets[24]) < .005
    # REP's merges are clamped to half of the tokens in every block, so smaller maxima fall short.
    assert own[186] < targets[24] - .003 and own[chosen] > own[186]
    arguments = run_cl.method_args('rep-prompt', 'tiny', 'weights.npz', 16, 24)
    assert (arguments['rep_atom'], arguments['rep_merge_max']) == (1, 187)
    assert run_cl.method_args('cps-prompt', 'tiny', 'weights.npz', 16, 24)['reduction_ratio'] == .635


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
    args = run_cl.parse_args(common + mix + ['--mix-own', '.5', '--mix-no-return', '--path', 'P3', '--schedule', *'0 0 24 24 24 24 24 0 0 0'.split()])
    assert args.mix_own == .5 and args.mix_no_return and run_cl.mix_suffix(args) == 'MRW'
    assert run_cl.mix_suffix({'mix_own': None, 'mix_no_return': False}) == 'M'
    for broken in (common + mix + ['--mix-own', '1.5'], common + mix + ['--mix-no-return'], common + ['--mix-own', '.5']):
        with pytest.raises(SystemExit):
            run_cl.parse_args(broken)
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


def test_cps_and_rep_take_a_budget_path():
    common = ['--seed', '1', '--output', 'unused', '--targets', '0', '4', '8', '12', '16', '20', '24']
    path = ['--path', 'P5', '--schedule', *map(str, [0] * 5 + [24] * 5)]
    for method in ('cps-prompt', 'rep-prompt'):
        args = run_cl.parse_args(['--method', method, *common, *path, '--eval-operators', 'tome', 'evit'])
        # No training degree: the plug-in stays at 0 and every task takes the own setting of its matched degree.
        assert args.schedule == [0] * 5 + [24] * 5 and args.degree == 0 and args.matched_degree is None
        assert args.eval_operators == ['tome', 'evit'] and args.sources == [0, 8, 16, 24]
        assert run_cl.parse_args(['--method', method, *common, *path]).eval_operators == sorted(run_cl.OPERATORS)
        args = run_cl.parse_args(['--method', method, '--matched-degree', '24', '--seed', '1', '--output', 'unused'])
        assert args.matched_degree == 24 and args.schedule is None
        for broken in (['--method', method, *common, *path, '--matched-degree', '24'],          # a path gives every task its own
                       ['--method', method, *common, '--path', 'P1', '--schedule', *map(str, [0] * 9 + [12])],  # no setting at 12
                       ['--method', method, *common, '--schedule', *map(str, [0] * 10)],        # no path name
                       ['--method', method, *common, *path, '--degree', '8'],
                       ['--method', method, *common, *path, '--vit-size', 'base'],
                       ['--method', method, *common, *path, '--mix', '0', '24'],
                       ['--method', method, *common, *path, '--joint'],
                       ['--method', method, '--seed', '1', '--output', 'unused'],                # neither a budget nor a path
                       ['--method', method, '--seed', '1', '--output', 'unused', '--targets', '0', '8', '--path', 'P5',
                        '--schedule', *map(str, [0] * 5 + [24] * 5)]):                          # 24 not inferred
            with pytest.raises(SystemExit):
                run_cl.parse_args(broken)
    # The plug-in methods keep their rules: the operator of the path is the only one inferred.
    args = run_cl.parse_args(['--method', 'inf-lora', '--operator', 'evit', *common, *path])
    assert args.eval_operators == ['evit'] and args.degree == 0
    with pytest.raises(SystemExit):
        run_cl.parse_args(['--method', 'inf-lora', *common, *path, '--matched-degree', '24'])


def test_a_path_changes_the_own_setting_between_tasks():
    from types import SimpleNamespace
    from models.cps_prompt_utils.model import PromptModel
    from models.rep_prompt_utils.model import REPModel
    torch.manual_seed(0)
    images = torch.randn(2, 3, 224, 224)
    cps = SimpleNamespace(net=PromptModel(num_classes=100, n_tasks=10, vit_size='tiny', pretrained=False).train())
    for degree, kept in ((0, 196), (24, 71), (8, 149), (16, 102), (0, 196)):
        run_cl.apply_own_setting(cps, 'cps-prompt', run_cl.OWN['cps-prompt'][degree])
        cps.net(images, train=True)
        assert run_cl.own_setting(cps, 'cps-prompt') == run_cl.OWN['cps-prompt'][degree]
        assert cps.net.diagnostics['prompted_tokens'] == 1 + kept
        cps.net.eval()
        cps.net(images)
        # Inference keeps every token whatever the task was trained with.
        assert cps.net.diagnostics['prompted_tokens'] == 197
        cps.net.train()
    rep = SimpleNamespace(net=REPModel(num_classes=100, n_tasks=10, vit_size='tiny', pretrained=False, atom=False, ald=False).train())
    for degree in (0, 24, 8, 16, 0):
        setting = run_cl.OWN['rep-prompt'][degree]
        run_cl.apply_own_setting(rep, 'rep-prompt', setting)
        assert run_cl.own_setting(rep, 'rep-prompt') == setting and rep.net.atom == bool(setting)
        # The same model built with that setting from the start merges the same tokens.
        built = REPModel(num_classes=100, n_tasks=10, vit_size='tiny', pretrained=False, atom=bool(setting), ald=False,
                         maximum=setting).train()
        built.load_state_dict(rep.net.state_dict())
        rep.net(images, train=True)
        built(images, train=True)
        assert rep.net.diagnostics['merged_tokens'] == built.diagnostics['merged_tokens']
        assert (sum(rep.net.diagnostics['merged_tokens']) > 0) == (setting > 0)
    # More budget removed, fewer tokens left at the end of the encoder.
    ends = {}
    for degree in (8, 16, 24):
        run_cl.apply_own_setting(rep, 'rep-prompt', run_cl.OWN['rep-prompt'][degree])
        rep.net(images, train=True)
        ends[degree] = rep.net.diagnostics['output_tokens'][-1]
    assert ends[8] > ends[16] > ends[24] >= 2
    for method, wrong in (('cps-prompt', 1.), ('cps-prompt', -.1), ('rep-prompt', -1), ('rep-prompt', .5), ('er', 0)):
        with pytest.raises(ValueError):
            run_cl.apply_own_setting(cps if method == 'cps-prompt' else rep, method, wrong)


def test_the_grid_of_a_path_gives_each_operator_the_bank_from_every_source():
    plan = run_cl.own_switch_plan([24, 0, 8, 16, 8])
    # The query pass saw every token; the prompted pass was trained at the source degree. No bank is named:
    # each inference operator takes its own.
    assert plan == {f'src{d}': (None, (0, d), None) for d in (0, 8, 16, 24)} and list(plan) == ['src0', 'src8', 'src16', 'src24']
    assert run_cl.own_source('cps-prompt') == 0 and run_cl.own_source('rep-prompt') is None
    assert run_cl.grid_plan('cps-prompt', 0, 24, ['tome'])[0] == 0
    # A constant path is the fixed run: its conditions are the fixed run's, under the labels of a path.
    _, fixed = run_cl.grid_plan('rep-prompt', 0, 24, ['tome', 'evit'])
    assert fixed['tome'] == ('tome', (0, 24), None) and fixed['evit@0'] == ('evit', (0, 0), ['evit'])
    assert plan['src24'][1] == fixed['tome'][1] and plan['src0'][1] == fixed['tome@0'][1]


def test_a_run_at_another_resolution_trains_and_is_inferred_at_one_shape():
    common = ['--method', 'coda-prompt', '--operator', 'tome', '--seed', '1', '--output', 'unused',
              '--targets', '0', '4', '8', '12', '16', '20', '24']
    args = run_cl.parse_args(common + ['--degree', '8', '--resolution', '176', '--eval-shapes', '224:0', '176:8', '128:0'])
    assert args.resolution == 176 and args.eval_shapes == [(224, 0), (176, 8), (128, 0)] and args.eval_operators == ['tome']
    # 224 pixels is an ordinary run; ER evaluates other shapes of a model trained at 224.
    assert run_cl.parse_args(common + ['--resolution', '224']).resolution is None
    assert run_cl.parse_args(['--method', 'er', '--seed', '1', '--output', 'unused', '--eval-shapes', '128:0']).eval_shapes == [(128, 0)]
    for broken in (common + ['--resolution', '100'],                                       # not whole patches
                   common + ['--resolution', '240'],                                       # above 224
                   common + ['--eval-shapes', '176'],                                      # no degree
                   common + ['--eval-shapes', '176:8', '176:8'],                           # twice
                   common + ['--resolution', '176', '--banks', 'tome=unused'],             # banks are trained at 224
                   common + ['--mix', '0', '8', '16', '24', '--resolution', '176'],        # mixed degrees
                   common + ['--joint', '--degree', '8', '--eval-shapes', '224:0'],        # joint training
                   common + ['--path', 'P5', '--schedule', *['0'] * 5, *['24'] * 5, '--resolution', '176'],
                   ['--method', 'sd-lora', '--seed', '1', '--output', 'unused', '--resolution', '176']):
        with pytest.raises(SystemExit):
            run_cl.parse_args(broken)


def test_imagenet_r_takes_the_unified_recipe_of_cifar_with_its_own_classes():
    import yaml
    configs = Path(run_cl.ROOT) / 'datasets' / 'configs'
    cifar, imagenet_r = (yaml.safe_load((configs / name / 'unified.yaml').read_text()) for name in run_cl.DATASETS)
    assert (imagenet_r['N_TASKS'], imagenet_r['N_CLASSES_PER_TASK'], imagenet_r['N_CLASSES']) == (10, 20, 200)
    assert (cifar['N_CLASSES_PER_TASK'], imagenet_r['N_CLASSES_PER_TASK']) == tuple(run_cl.DATASETS.values())
    assert {key: value for key, value in imagenet_r.items() if not key.startswith('N_')} == \
        {key: value for key, value in cifar.items() if not key.startswith('N_')}
    common = ['--method', 'inf-lora', '--seed', '1', '--output', 'unused']
    assert run_cl.parse_args(common).dataset == 'seq-cifar100-224'
    # The low-rank heads take the classes of a task; on CIFAR-100 the arguments stay as they were.
    assert run_cl.dataset_args('inf-lora', 'seq-imagenet-r') == {'dataset': 'seq-imagenet-r', 'lora_classes_per_task': 20}
    assert run_cl.dataset_args('inf-lora', 'seq-cifar100-224') == {'dataset': 'seq-cifar100-224'}
    assert run_cl.dataset_args('coda-prompt', 'seq-imagenet-r') == {'dataset': 'seq-imagenet-r'}
    args = run_cl.parse_args(common + ['--dataset', 'seq-imagenet-r', '--resolution', '176', '--degree', '8', '--targets', '8'])
    assert (args.dataset, args.resolution) == ('seq-imagenet-r', 176)
    for broken in (common + ['--dataset', 'seq-cifar10'], common + ['--dataset', 'seq-imagenet-r', '--joint']):
        with pytest.raises(SystemExit):
            run_cl.parse_args(broken)


def test_a_path_runs_at_the_shapes_of_its_budgets_or_mixes_them():
    from backbone.token_compression import shape_cost
    # Every shape of a budget level costs no more than the level at 224 pixels, within the tolerance.
    for level, shapes in run_cl.SHAPE_LEVELS.items():
        assert (224, level) in shapes and all(shape_cost(*shape) <= shape_cost(224, level) * run_cl.TOLERANCE for shape in shapes)
    common = ['--method', 'coda-prompt', '--operator', 'tome', '--seed', '1', '--output', 'unused', '--no-grid',
              '--targets', '0', '4', '8', '12', '16', '20', '24', '--path', 'P5', '--schedule', *['0'] * 5, *['24'] * 5]
    args = run_cl.parse_args(common + ['--budget-shapes', '24=176:8', '--eval-shapes', '224:0', '176:8'])
    assert args.budget_shapes == {24: (176, 8)} and run_cl.path_suffix(args) == 'S'
    assert run_cl.budget_shape(args.budget_shapes, 24) == (176, 8) and run_cl.budget_shape(args.budget_shapes, 0) == (224, 0)
    mixed = run_cl.parse_args(common + ['--budget-shapes', '24=176:8', '--mix', '0', '8', '16', '24', '--mix-shapes'])
    assert mixed.mix_shapes and run_cl.path_suffix(mixed) == 'MS'
    assert run_cl.path_suffix(run_cl.parse_args(common)) == '' and run_cl.path_suffix(run_cl.parse_args(common + ['--mix', '0', '8', '16', '24'])) == 'M'
    assert run_cl.mix_suffix({'mix': [0, 24], 'mix_shapes': True, 'mix_own': .5}) == 'MSW'
    for broken in (common + ['--budget-shapes', '24=208:16'],                          # costs more than budget 24
                   common + ['--budget-shapes', '24=176:8', '24=192:12'],              # twice
                   common + ['--budget-shapes', '24=176'],                             # no degree
                   [item for item in common if item != '--no-grid'] + ['--budget-shapes', '24=176:8'],   # the grid
                   common + ['--budget-shapes', '24=176:8', '--mix', '0', '8', '16', '24'],    # degrees mixed, shapes fixed
                   common + ['--mix', '0', '4', '24', '--mix-shapes'],                 # 4 is no budget level
                   common + ['--mix-shapes'],                                          # nothing to mix
                   ['--method', 'sd-lora', *common[2:], '--budget-shapes', '24=176:8'],
                   common[:17] + ['--budget-shapes', '24=176:8']):                     # no path
        with pytest.raises(SystemExit):
            run_cl.parse_args(broken)


def test_shapes_may_also_be_drawn_from_everything_within_the_budget():
    from backbone.token_compression import shape_cost
    # The 13 shapes of the levels, and the grid of resolutions 128-224 by 16 and degrees 0-24 by 4.
    assert [len(run_cl.shape_pool('menu', b)) for b in (0, 8, 16, 24)] == [13, 12, 9, 4]
    assert [len(run_cl.shape_pool('grid', b)) for b in (0, 8, 16, 24)] == [49, 46, 39, 30]
    assert set(run_cl.shape_pool('menu', 24)) == set(run_cl.SHAPE_LEVELS[24])
    assert all(shape_cost(*s) <= shape_cost(224, 24) * run_cl.TOLERANCE for s in run_cl.shape_pool('grid', 24))
    common = ['--method', 'coda-prompt', '--operator', 'tome', '--seed', '1', '--output', 'unused', '--no-grid',
              '--targets', '0', '4', '8', '12', '16', '20', '24', '--path', 'P5', '--schedule', *['0'] * 5, *['24'] * 5,
              '--budget-shapes', '24=176:8', '--mix', '0', '8', '16', '24', '--mix-shapes']
    assert run_cl.path_suffix(run_cl.parse_args(common)) == 'MS'
    assert run_cl.path_suffix(run_cl.parse_args(common + ['--shape-pool', 'menu'])) == 'MSU'
    assert run_cl.path_suffix(run_cl.parse_args(common + ['--shape-pool', 'grid'])) == 'MSG'
    for broken in (common[:-1] + ['--shape-pool', 'grid'],                   # no --mix-shapes
                   common + ['--shape-pool', 'grid', '--mix-own', '.5'],
                   common + ['--shape-pool', 'other']):
        with pytest.raises(SystemExit):
            run_cl.parse_args(broken)


def test_a_level_may_also_take_the_grid_shapes_of_its_cost():
    from backbone.token_compression import shape_cost
    levels = {level: run_cl.level_shapes('grid-levels', level) for level in (0, 8, 16, 24)}
    assert all(run_cl.level_shapes('levels', level) == run_cl.SHAPE_LEVELS[level] for level in levels)
    # Each level takes the grid shapes between its cost and the next cheaper level's, the cheapest down to FLOOR of its
    # own: 24 shapes in all, each at one level, with every shape of SHAPE_LEVELS at its own level.
    assert [len(shapes) for shapes in levels.values()] == [3, 7, 9, 5]
    union = {s for shapes in levels.values() for s in shapes}
    assert len(union) == 24 and set(levels[0]) == {(224, 0), (224, 4), (208, 0)}
    assert all(set(run_cl.SHAPE_LEVELS[level]) <= set(shapes) for level, shapes in levels.items())
    for upper, lower in ((0, 8), (8, 16), (16, 24)):
        assert all(shape_cost(224, lower) * run_cl.TOLERANCE < shape_cost(*s) <= shape_cost(224, upper) * run_cl.TOLERANCE
                   for s in levels[upper])
    # The grid shapes left out are those cheaper than the floor.
    floor = shape_cost(224, 24) * run_cl.FLOOR
    assert all(floor <= shape_cost(*s) <= shape_cost(224, 24) * run_cl.TOLERANCE for s in levels[24])
    assert {s for s in run_cl.SHAPE_GRID if shape_cost(*s) >= floor} == union
    common = ['--method', 'coda-prompt', '--operator', 'tome', '--seed', '1', '--output', 'unused', '--no-grid',
              '--targets', '0', '4', '8', '12', '16', '20', '24', '--path', 'P5', '--schedule', *['0'] * 5, *['24'] * 5,
              '--budget-shapes', '24=176:8', '--mix', '0', '8', '16', '24', '--mix-shapes', '--shape-pool', 'grid-levels']
    assert run_cl.path_suffix(run_cl.parse_args(common)) == 'MSL'
    # The level is drawn as with the levels pool, so a share or no mixing after a return still applies.
    assert run_cl.path_suffix(run_cl.parse_args(common + ['--mix-own', '.5'])) == 'MSLW'
    assert run_cl.path_suffix(run_cl.parse_args(common + ['--mix-no-return'])) == 'MSLR'
    with pytest.raises(SystemExit):
        run_cl.parse_args([a for a in common if a != '--mix-shapes'])
