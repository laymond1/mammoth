"""Size approval rejects expanded workloads and changed comparison settings."""
import copy
import json

import pytest

from scripts import hide_protocol, hide_size_protocol as sizes


@pytest.mark.parametrize('name,dimension,heads', [('small', 384, 6), ('tiny', 192, 3)])
def test_variant_geometry_and_native_command(name, dimension, heads):
    variant = sizes.load_variant(name, verify_checkpoint=False)
    assert (variant['feature_dim'], variant['num_heads']) == (dimension, heads)
    settings = hide_protocol.protocol('validation', variant=name)
    assert settings['tasks'] == 2
    assert settings['n_epochs'] == 70
    command = hide_protocol.native_command('efficiency', checkpoint=variant['checkpoint_file'], variant=name)
    assert command[command.index('--hide_vit_size') + 1] == name
    assert command[command.index('--stop_after') + 1] == '1'
    assert command[command.index('--batch_size') + 1] == '64'
    assert set(sizes.SOURCE_FILES).issubset(hide_protocol.source_hashes(variant=name))
    assert 'benchmarks/cl_efficiency/hide_sizes/execution_plan.json' not in hide_protocol.source_hashes(variant=name)


def test_size_scope_rejects_full_cl_and_other_sizes():
    with pytest.raises(ValueError, match='first-task pilot only'):
        hide_protocol.protocol('full_cl', variant='small')
    with pytest.raises(ValueError, match='Unapproved'):
        sizes.load_variant('base', verify_checkpoint=False)
    assert hide_protocol.protocol('validation')['vit_size'] == 'base'


@pytest.mark.parametrize('section,key,value', [
    (None, 'full_cl_authorized', True), (None, 'variant_order', ['tiny', 'small']),
    ('common', 'batch_size', 32), ('common', 'seed', 1994),
    ('common', 'cache_root', '/tmp/cache'), ('common', 'gpu', '0'),
    ('common', 'prompt_layers', [0, 1]), ('common', 'mlp_structure', [1]),
    ('validation', 'tasks', 10), ('validation', 'prompt_epochs', 1),
    ('efficiency', 'measurement_repeats', 1), ('efficiency', 'ca_runs_task1', True),
])
def test_plan_mutations_fail_closed(tmp_path, monkeypatch, section, key, value):
    plan = copy.deepcopy(sizes.load_plan())
    (plan if section is None else plan[section])[key] = value
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(sizes, 'PLAN', path)
    with pytest.raises(ValueError, match='authorized protocol'):
        sizes.load_plan()


def test_changed_weight_geometry_and_precision_fail_closed(tmp_path, monkeypatch):
    plan = sizes.load_plan()
    plan['variants'][0]['feature_dim'] = 768
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(sizes, 'PLAN', path)
    with pytest.raises(ValueError, match='geometry'):
        sizes.load_variant('small', verify_checkpoint=False)
    plan['variants'][0]['feature_dim'] = 384
    plan['common']['precision']['tf32_allowed'] = True
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='precision'):
        sizes.load_plan()
