"""CPU-only authorization, recipe and argv contracts for the fresh benchmark."""
from collections import Counter
import json

import pytest

from scripts import unified_cl_protocol as protocol


def _spec(method='sgd', size='tiny', batch=16, mode='measurement', repeat=1):
    return next(row for row in protocol.build_run_specs(modes=[mode])
                if (row['method'], row['vit_size'], row['batch_size'], row['repeat']) == (method, size, batch, repeat))


def test_saved_plan_and_cartesian_scope_are_complete_without_extra_jobs():
    saved = json.loads(protocol.PLAN.read_text())
    assert protocol.load_protocol() == protocol.approved_protocol() == saved
    specs = protocol.build_run_specs()
    assert len(specs) == len({spec['run_id'] for spec in specs}) == 510
    assert Counter(row['mode'] for row in specs) == {'validation': 102, 'measurement': 306, 'flops': 102}
    assert Counter((row['method'], row['vit_size'], row['batch_size']) for row in specs) == {
        (method, size, batch): 5 for method in protocol.METHODS for size in protocol.SIZES for batch in protocol.BATCHES}
    assert specs[0]['vit_size'] == 'tiny'
    assert all(row['tasks'] == 2 and row['dataset_total_tasks'] == 10 and row['num_classes'] == 100 for row in specs)
    assert all(row['minibatch_size'] == row['batch_size'] for row in specs)
    assert all(row['runtime_ready'] is False and row['executed'] is False for row in specs)


@pytest.mark.parametrize('path,value', [
    (('training_authorized',), False), (('experiments_executed',), True), (('runtime_ready',), True),
    (('methods',), list(protocol.METHODS) + ['prol']), (('sizes',), ['base']),
    (('batch_sizes',), [True, 64]), (('dataset', 'total_tasks'), 2), (('dataset', 'stop_after'), 10),
    (('dataset', 'config'), 'ease'), (('dataset', 'drop_last'), True), (('environment', 'gpu'), '0'),
    (('environment', 'torch_cpu_threads'), 4), (('precision', 'autocast'), True),
    (('replay', 'minibatch_policy'), 'fixed64'), (('replay', 'buffer_size'), 2000),
    (('schedule', 'measurement_repeats'), 4), (('schedule', 'measurement_epochs_per_task'), 30),
    (('schedule', 'failure_policy'), 'automatically retry'),
    (('method_recipes', 'sd-lora', 'lr'), 0.008), (('method_recipes', 'derpp', 'alpha'), 0.1),
    (('checkpoints', 'tiny', 'canonical_sha256'), protocol.CHECKPOINTS['base']['canonical_sha256']),
])
def test_scope_recipe_and_pins_drift_are_rejected(path, value):
    plan = protocol.approved_protocol()
    parent = plan
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match='approved plan'):
        protocol.validate_protocol(plan)


def test_unknown_or_credential_fields_are_not_echoed():
    plan = protocol.approved_protocol()
    plan['private'] = 'must-not-appear-in-error'
    with pytest.raises(ValueError) as error:
        protocol.validate_protocol(plan)
    assert 'must-not-appear' not in str(error.value)
    assert 'private' not in str(error.value)


@pytest.mark.parametrize('modes', [[], ['measurement', 'measurement'], ['measurement', 'retry'], 'measurement', [['measurement']]])
def test_mode_subsets_cannot_expand_or_duplicate_execution(modes):
    with pytest.raises(ValueError):
        protocol.build_run_specs(modes=modes)


def test_plan_and_run_specs_do_not_share_mutable_state():
    plan = protocol.approved_protocol()
    plan['checkpoints']['tiny']['sha256'] = 'changed'
    assert protocol.approved_protocol()['checkpoints']['tiny']['sha256'] != 'changed'
    spec = _spec('cl-lora')
    other = _spec('cl-lora')
    spec['method_recipe']['lora_msa'][0] = 0
    assert other['method_recipe']['lora_msa'] == [1, 0, 1]
    assert protocol.METHOD_RECIPES['cl-lora']['lora_msa'] == [1, 0, 1]


@pytest.mark.parametrize('key,value', [('repeat', 4), ('repeat', True), ('batch_size', True), ('minibatch_size', 64),
                                      ('runtime_ready', True), ('tasks', 10), ('nominal_epochs_per_task', 1)])
def test_run_spec_mutation_cannot_bypass_scope_guard(key, value):
    spec = _spec()
    spec[key] = value
    with pytest.raises(ValueError):
        protocol.build_method_args(spec)


def test_epoch_interpretation_and_nontraining_source_phases_are_explicit():
    hide = _spec('hide-prompt')
    args = protocol.build_method_args(hide)
    assert (hide['nominal_epochs_per_task'], args['n_epochs'], args['hide_tii_epochs'], args['hide_prompt_epochs']) == (20, 40, 20, 20)
    assert args['hide_ca_epochs'] == 30
    assert protocol.build_method_args(_spec('hide-prompt', mode='validation'))['n_epochs'] == 4
    assert protocol.build_method_args(_spec('rainbow-prompt', mode='validation'))['rainbow_warmup_epochs'] == 0
    assert protocol.build_method_args(_spec('rainbow-prompt'))['rainbow_warmup_epochs'] == 5
    ranpac = _spec('ranpac')
    assert ranpac['expected_training']['main_optimizer_steps_per_task'] == [6260, 0]
    assert ranpac['expected_training']['observe_calls_per_task'] == [6260, 6260]
    cl = _spec('cl-lora', batch=64)
    assert cl['expected_training']['main_optimizer_steps_per_task'] == [1580, 3160]
    ewc = _spec('ewc_on', batch=64)
    assert ewc['expected_training']['main_optimizer_steps_per_task'] == [1580, 1580]
    assert ewc['expected_training']['fisher_backward_calls_per_task'] == [5000, 5000]


@pytest.mark.parametrize('method', protocol.METHODS)
@pytest.mark.parametrize('size', protocol.SIZES)
def test_every_selected_method_has_explicit_same_geometry_and_shared_checkpoint(method, size):
    spec = _spec(method, size)
    args = protocol.build_method_args(spec)
    assert args['dataset_config'] == 'unified'
    assert args['batch_size'] == 16
    if method in protocol.GENERIC_METHODS:
        assert args['backbone'] == 'vit-augreg'
        assert args['vit_size'] == size and args['vit_checkpoint'] == protocol.CHECKPOINTS[size]['file']
        assert args['optimizer'] == 'adamw' and args['lr'] == 0.0001 and args['pretrained'] == 1
    elif method in ('l2p', 'dualprompt', 'coda-prompt'):
        assert args['pretrained'] == 0 and 'vit_checkpoint' not in args
        assert args['vit_size'] == size
        prefix = {'l2p': 'l2p', 'dualprompt': 'dualprompt', 'coda-prompt': 'coda'}[method]
        assert args[prefix + '_vit_size'] == size
        assert args['coda_checkpoint' if method == 'coda-prompt' else 'pretrained_checkpoint'] == protocol.CHECKPOINTS[size]['file']
    elif method == 'ranpac':
        assert args['ranpac_vit_size'] == args['vit_size'] == size
        assert args['pretrained_checkpoint'] == protocol.CHECKPOINTS[size]['file'] and args['pretrained'] == 0
        assert 'adapter_size' not in args
    elif method == 'ease':
        assert args['vit_size'] == size and args['ease_checkpoint'] == protocol.CHECKPOINTS[size]['file']
        assert args['adapter_size'] == 64
    else:
        prefix = 'lora' if method.endswith('-lora') else {'hide-prompt': 'hide', 'rainbow-prompt': 'rainbow'}[method]
        assert args[prefix + '_vit_size'] == size
        assert args[prefix + '_checkpoint'] == protocol.CHECKPOINTS[size]['file']


def test_commands_are_argv_and_separate_profiler_runs_do_not_enter_cost_means():
    for mode in protocol.MODES:
        spec = _spec('derpp', mode=mode)
        args = protocol.build_method_args(spec)
        command = protocol.build_command(spec)
        assert command[:4] == ['uv', 'run', 'python', 'main.py']
        assert command[command.index('--minibatch_size') + 1] == '16'
        assert command[command.index('--buffer_size') + 1] == '500'
        assert args['alpha'] == 0.3 and args['beta'] == 0.5
        assert args['results_path'].startswith(str(protocol.ROOT / 'results/unified_comparison'))
        assert args['wandb_project'] == 'cl-efficiency' and args['wandb_entity'] == 'laymond1'
        assert '--lr_scheduler' not in command and '--nowand' not in command
        assert '--disable_log' not in command
        assert spec['resource_statistics_eligible'] is (mode == 'measurement')
        assert not any('run_low_rank_efficiency' in token or 'run_prol' in token for token in command)
    command = protocol.build_command(_spec('cl-lora'))
    offset = command.index('--lora_msa')
    assert command[offset + 1:offset + 4] == ['1', '0', '1']


def test_planned_learning_rates_preserve_method_scaling_and_new_generic_recipe():
    assert protocol.expected_initial_learning_rates(_spec('sgd')) == {'main': 0.0001}
    assert protocol.expected_initial_learning_rates(_spec('sgd', batch=64)) == {'main': 0.0001}
    assert protocol.expected_initial_learning_rates(_spec('l2p')) == {'main': 0.001875}
    assert protocol.expected_initial_learning_rates(_spec('rainbow-prompt', batch=64)) == {'main': 0.0075}
    assert protocol.expected_initial_learning_rates(_spec('sd-lora')) == {'main': 0.001, 'head': 0.008}
    assert protocol.expected_initial_learning_rates(_spec('hide-prompt')) == {
        'tii': 0.00003125, 'prompt': 0.001875, 'prompt_head': 0.0001875,
        'tii_alignment': 0.0003125, 'prompt_alignment': 0.005}


def test_environment_overlay_is_public_same_gpu_and_shared_roots():
    env = protocol.build_environment(_spec())
    assert env['CUDA_VISIBLE_DEVICES'] == '1'
    assert env['OMP_NUM_THREADS'] == env['MKL_NUM_THREADS'] == '2'
    assert env['CL_DATA_ROOT'] == protocol.DATA_ROOT
    assert env['TORCH_HOME'] == protocol.CACHE_ROOT + '/torch'
    assert env['WANDB_ENTITY'] == 'laymond1'
    assert not {'WANDB_API_KEY', 'SLACK_WEBHOOK_URL', 'HOME', 'CODEX_HOME'} & env.keys()


def test_checkpoint_hashing_is_read_only_and_shape_pins_stay_separate(monkeypatch, tmp_path):
    plan = protocol.approved_protocol()
    path = tmp_path / 'weights.npz'
    path.write_bytes(b'not-the-approved-checkpoint')
    # Preserve the approved in-memory plan while replacing only opened files.
    original_open = protocol.Path.open
    def redirected_open(self, *args, **kwargs):
        if str(self) in {pin['file'] for pin in protocol.CHECKPOINTS.values()}:
            return original_open(path, *args, **kwargs)
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(protocol.Path, 'open', redirected_open)
    with pytest.raises(ValueError, match='checkpoint file changed'):
        protocol.verify_checkpoints(plan)
    assert path.read_bytes() == b'not-the-approved-checkpoint'
