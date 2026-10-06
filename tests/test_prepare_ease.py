"""Variant preparation must preserve common telemetry and never execute."""

import json
from pathlib import Path

import pytest

from scripts import benchmark_cl, prepare_ease


def test_all_variants_are_dry_runs_with_shared_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_ease, 'configure_environment', lambda: {
        'gpu': '1', 'data_root': '/mnt/sdab1/share/data',
        'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency'})
    monkeypatch.setattr(benchmark_cl, 'readiness', lambda *args: {})
    monkeypatch.setattr(benchmark_cl, 'execute', lambda *args: pytest.fail('Preparation must never execute'))
    monkeypatch.setattr(benchmark_cl, 'notify_slack', lambda *args, **kwargs: pytest.fail('Preparation must never notify'))
    commands = prepare_ease.prepare(tmp_path)
    observed = set()
    for command in commands:
        assert command['experiment_executed'] is False
        assert command['execute_option_present'] is False
        assert '--execute' not in command['instrumented_runner_argv']
        plan = json.loads(Path(command['plan']).read_text())
        assert plan['experiment_executed'] is False
        assert len(plan['jobs']) == 1 and plan['jobs'][0]['model'] == 'ease'
        assert plan['protocol']['dataset'] == 'seq-cifar100-224'
        assert plan['protocol']['tasks'] == 1 and plan['protocol']['outer_epochs'] == 1
        assert plan['protocol']['batch_size'] == 64
        assert plan['jobs'][0]['passes'] == ['measurement', 'flops']
        assert plan['runtime']['gpu'] == '1'
        assert plan['logging'] == {'wandb_project': 'cl-efficiency', 'wandb_entity': 'laymond1',
                                   'wandb_mode': 'online', 'slack': 'on'}
        arguments = plan['jobs'][0]['args']
        assert arguments['ease_pretrain_type'] == 'in21k_ft_in1k'
        observed.add((arguments['vit_size'], arguments['adapter_size']))
    assert observed == {('base', 64), ('small', 64), ('tiny', 64), ('small', 32), ('tiny', 16)}


def test_preparer_rejects_execution_option():
    with pytest.raises(SystemExit):
        prepare_ease.main(['--execute'])
