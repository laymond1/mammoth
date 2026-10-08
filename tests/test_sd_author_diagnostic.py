"""CPU-only regressions for the isolated immutable-author SD diagnostic."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / 'results/peft_pilot/sd_author_diagnostic.py'


@pytest.fixture
def diagnostic():
    spec = importlib.util.spec_from_file_location('_sd_author_diagnostic_test', DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_diagnostic_is_stdlib_only_and_ignores_missing_execution_inputs(tmp_path):
    code = '''
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
class RejectHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'numpy', 'wandb', 'PIL', 'psutil', 'timm', 'models', 'methods'}:
            raise RuntimeError('Preparation attempted a numerical or sensor import: ' + fullname)
sys.meta_path.insert(0, RejectHeavyImports())
spec = importlib.util.spec_from_file_location('_sd_author_diagnostic_dry', DRIVER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def forbidden(*args, **kwargs):
    raise RuntimeError('Dry invocation attempted credential access or worker launch')
module.configure_environment = forbidden
assert module.main(['--output', 'absent-output', '--native-result', 'absent-native.json', '--reference-source', 'absent-source']) == 0
assert not list(Path.cwd().iterdir())
assert not {'torch', 'numpy', 'timm', 'wandb', 'models', 'methods'}.intersection(sys.modules)
'''.replace('DRIVER', repr(str(DRIVER)))
    result = subprocess.run([sys.executable, '-S', '-c', code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document.get('execution', document.get('training_executed')) is False


def test_preflight_is_stdlib_only_and_cannot_start_worker_or_read_credentials(tmp_path):
    code = '''
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
class RejectHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'numpy', 'wandb', 'PIL', 'psutil', 'timm', 'models', 'methods'}:
            raise RuntimeError('Preflight attempted a numerical or sensor import: ' + fullname)
sys.meta_path.insert(0, RejectHeavyImports())
spec = importlib.util.spec_from_file_location('_sd_author_diagnostic_preflight', DRIVER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def forbidden(*args, **kwargs):
    raise RuntimeError('Preflight attempted credential access or worker launch')
module.configure_environment = module.execute = module.worker = forbidden
module.prepare_spec = lambda options: {'checkpoint_sha256': 'checkpoint-pin', 'initial_state_sha256': 'initial-pin', 'native_result_sha256': 'native-pin'}
assert module.main(['--preflight', '--output', 'absent-output']) == 0
assert not list(Path.cwd().iterdir())
assert not {'torch', 'numpy', 'timm', 'wandb', 'models', 'methods'}.intersection(sys.modules)
'''.replace('DRIVER', repr(str(DRIVER)))
    result = subprocess.run([sys.executable, '-S', '-c', code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document['ready'] is True and document['training_executed'] is False
    assert document['output_created'] is False and document['native_result_sha256'] == 'native-pin'


def test_diagnostic_calls_original_author_training_routine_without_native_lifecycle():
    tree = ast.parse(DRIVER.read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'worker')
    calls = [node.func for node in ast.walk(worker) if isinstance(node, ast.Call)]
    assert sum(isinstance(call, ast.Attribute) and call.attr == 'incremental_train' for call in calls) == 1
    assert sum(isinstance(call, ast.Attribute) and call.attr == 'before_task' for call in calls) == 1
    assert sum(isinstance(call, ast.Attribute) and call.attr == 'incremental_test' for call in calls) == 1
    assert not any(isinstance(call, ast.Attribute) and call.attr in ('meta_observe', 'observe', 'train_single_epoch')
                   for call in calls)
    imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    imports.extend(node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    assert not any(name.startswith(('models.low_rank_utils.sd', 'models.sd_lora')) for name in imports)


def valid_spec(diagnostic):
    shared = diagnostic.shared
    return {'method': 'sd-lora', 'mode': 'diagnostic', 'tasks': 1, 'gpu': '1',
            'protocol': diagnostic.diagnostic_protocol(), 'precision': deepcopy(shared.PRECISION),
            'checkpoint_sha256': shared.CHECKPOINT_SHA256, 'reference_commit': shared.REFERENCE_COMMIT,
            'canonical_pretrained_sha256': shared.CANONICAL_PRETRAINED_SHA256,
            'wandb_mode': 'online', 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency',
            'measurement_aggregate_eligible': False, 'paper_reproduction': False}


def completed(diagnostic):
    return {'counts': {'observe_calls': 79, 'optimizer_updates': 79,
                       'task_begin_extra_samples': 0, 'task_end_extra_samples': 0},
            'epoch_results': [{'task': 1, 'epoch': 1, 'steps': 79, 'samples': 5000, 'loss_mean': 5.321488868963869}],
            'observed_totals': {'training_examples': 5000, 'evaluation_examples': 1000, 'steps': 79, 'tasks_completed': 1},
            'task_results': [{'task': 1, 'seen_correct': 100, 'seen_samples': 1000,
                              'old_correct': 0, 'old_samples': 0, 'new_correct': 100, 'new_samples': 1000}],
            'first_batch_audits': [{'task': 1, 'epoch': 1, 'seed': 2993, 'samples': 64,
                                   'optimizer_update_index': 1, 'input_sha256': 'a' * 64,
                                   'labels_sha256': 'b' * 64, 'gradient_sha256': 'c' * 64}],
            'state_audits': {'network_sha256': 'd' * 64}, 'accuracy': {'mean_accuracy_pct': 10.},
            'structural_checks': {name: True for name in diagnostic.REQUIRED_CHECKS}}


def test_diagnostic_is_separate_from_efficiency_and_preserves_sd_source_settings(diagnostic):
    settings = diagnostic.diagnostic_protocol()
    assert settings['method'] == 'sd-lora' and settings['mode'] == 'diagnostic'
    assert settings['tasks'] == settings['n_epochs'] == 1
    assert settings['dataset_total_tasks'] == 10 and settings['class_order'] == list(range(100))
    assert settings['rank'] == 10 and settings['batch_size'] == 64 and settings['num_workers'] == 4
    assert settings['dataset_config'] == 'ease' and settings['seed'] == 1993
    assert settings['measurement_repeats'] == settings['flops_runs'] == 0
    assert settings['efficiency_comparison'] is False
    assert settings['hyperparameters'] == diagnostic.shared.protocol('sd-lora')['hyperparameters']
    diagnostic.validate_spec(valid_spec(diagnostic))


@pytest.mark.parametrize('key,value', [('method', 'inf-lora'), ('mode', 'efficiency'), ('tasks', 2),
    ('tasks', True), ('gpu', '0'), ('wandb_mode', 'offline'), ('measurement_aggregate_eligible', True),
    ('checkpoint_sha256', 'changed'), ('reference_commit', 'changed')])
def test_scope_guard_rejects_other_training_or_aggregate_eligibility(diagnostic, key, value):
    spec = valid_spec(diagnostic)
    spec[key] = value
    with pytest.raises(ValueError):
        diagnostic.validate_spec(spec)


@pytest.mark.parametrize('key,value', [('n_epochs', 20), ('n_epochs', True), ('seed', 0),
    ('batch_size', 128), ('rank', 4), ('dataset_config', 'default')])
def test_source_protocol_cannot_expand_or_change(diagnostic, key, value):
    spec = valid_spec(diagnostic)
    spec['protocol'][key] = value
    with pytest.raises(ValueError):
        diagnostic.validate_spec(spec)


def test_complete_source_epoch_can_report_collapse_without_claiming_efficiency(diagnostic):
    assert diagnostic.check_completion(completed(diagnostic)) is True


@pytest.mark.parametrize('corruption', ['tail', 'second_epoch', 'missing_update', 'hidden_statistics',
    'boolean_statistics', 'extra_evaluation', 'boolean_tasks', 'missing_predictions', 'missing_first_update', 'failed_state_audit'])
def test_incomplete_or_unverifiable_source_work_cannot_be_called_complete(diagnostic, corruption):
    record = completed(diagnostic)
    if corruption == 'tail':
        record['epoch_results'][0]['samples'] = 4992
    elif corruption == 'second_epoch':
        record['epoch_results'].append(deepcopy(record['epoch_results'][0]))
    elif corruption == 'missing_update':
        record['counts']['optimizer_updates'] = 78
    elif corruption == 'hidden_statistics':
        record['counts']['task_end_extra_samples'] = 5000
    elif corruption == 'boolean_statistics':
        record['counts']['task_end_extra_samples'] = False
    elif corruption == 'extra_evaluation':
        record['observed_totals']['evaluation_examples'] = 2000
    elif corruption == 'boolean_tasks':
        record['observed_totals']['tasks_completed'] = True
    elif corruption == 'missing_predictions':
        record['task_results'].clear()
    elif corruption == 'missing_first_update':
        record['first_batch_audits'].clear()
    else:
        record['structural_checks']['method_state_valid'] = False
    with pytest.raises(RuntimeError):
        diagnostic.check_completion(record)


def test_comparison_preserves_first_divergence_and_does_not_mutate_evidence(diagnostic):
    source, native = completed(diagnostic), completed(diagnostic)
    native['first_batch_audits'][0]['gradient_sha256'] = 'different-gradient'
    native['state_audits']['network_sha256'] = 'different-network'
    native['counts']['optimizer_updates'] = 78
    native['epoch_results'][0]['loss_mean'] += .25
    native['accuracy']['mean_accuracy_pct'] = 12.
    original = deepcopy((source, native))
    compared = diagnostic.compare_records(source, native)
    assert compared['input_sha256_matches'] is True and compared['labels_sha256_matches'] is True
    assert compared['gradient_sha256_matches'] is False and compared['network_sha256_matches'] is False
    assert compared['counts_match'] is False
    assert compared['loss_mean_abs_difference'] == .25
    assert compared['seen_accuracy_pp_difference'] == -2.
    assert (source, native) == original


def test_cross_entropy_observer_returns_same_tensor_with_exact_cpu_gradient(diagnostic):
    import torch
    from torch.nn import functional as functional
    base = torch.tensor([[.1, .5, -.4], [.2, -.1, .3]], requires_grad=True)
    actual = base.detach().clone().requires_grad_(True)
    untouched = actual.detach().clone()
    labels = torch.tensor([1, 2])
    expected = functional.cross_entropy(base, labels, label_smoothing=.1)
    expected.backward()
    returns = []
    def original(*args, **kwargs):
        loss = functional.cross_entropy(*args, **kwargs)
        returns.append(loss)
        return loss
    current = {'stage': 'train', 'loss_sum': 0., 'loss_calls': 0}
    record = {}
    observed = diagnostic.cross_entropy_observer(original, record, current)(actual, labels, label_smoothing=.1)
    assert len(returns) == 1 and observed is returns[0]
    observed.backward()
    assert torch.equal(actual.grad, base.grad) and torch.equal(actual.detach(), untouched)
    assert current['loss_calls'] == 1 and current['loss_sum'] == float(expected.detach())
    assert record == {}


def test_cross_entropy_observer_does_not_count_other_phases_and_rejects_nonfinite_training(diagnostic):
    import torch
    returned = torch.tensor(2., requires_grad=True)
    current = {'stage': 'evaluation'}
    assert diagnostic.cross_entropy_observer(lambda: returned, {}, current)() is returned
    assert current == {'stage': 'evaluation'}
    current = {'stage': 'train', 'loss_sum': 0., 'loss_calls': 0}
    with pytest.raises(FloatingPointError):
        diagnostic.cross_entropy_observer(lambda: torch.tensor(float('inf')), {}, current)()
    assert current['loss_sum'] == 0. and current['loss_calls'] == 0


@pytest.fixture
def prepared_inputs(diagnostic, monkeypatch, tmp_path):
    paths = {name: tmp_path / name for name in ('initial.pt', 'encoder.npz', 'predictions.npz',
                                              'task1_state.pt', 'raw-data', 'protected.py')}
    for name, path in paths.items():
        path.write_bytes(name.encode())
    file_hash = diagnostic.shared.file_hash
    monkeypatch.setattr(diagnostic.shared, 'CHECKPOINT_SHA256', file_hash(paths['encoder.npz']))
    monkeypatch.setattr(diagnostic.shared, 'verify_reference_source', lambda source: {'methods/sdlora.py': 'fixed-source'})
    native = {**completed(diagnostic), 'method': 'sd-lora', 'protocol': diagnostic.efficiency_protocol('sd-lora'),
        'status': 'completed', 'computation_status': 'completed', 'wandb_logging_status': 'uploaded',
        'precision': diagnostic.shared.PRECISION, 'gpu': '1', 'data_root': str(tmp_path / 'data'),
        'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency',
        'initial_state': str(paths['initial.pt']), 'initial_state_sha256': file_hash(paths['initial.pt']),
        'initial_learned_sha256': 'same-learned-state', 'checkpoint_file': str(paths['encoder.npz']),
        'prediction_files': [{'task': 1, 'path': str(paths['predictions.npz']), 'sha256': file_hash(paths['predictions.npz'])}],
        'data_files': {'files': {'raw': {'path': str(paths['raw-data']), 'sha256': file_hash(paths['raw-data'])}}},
        'execution_source_sha256': {str(paths['protected.py']): file_hash(paths['protected.py'])},
        'common_reference_data_snapshots': {}, 'common_transform_snapshots': {}}
    native['state_audits']['checkpoint_sha256'] = file_hash(paths['task1_state.pt'])
    native_path = tmp_path / 'native-result.json'
    native_path.write_text(json.dumps(native))
    options = SimpleNamespace(native_result=native_path, reference_source=tmp_path / 'source', output=tmp_path / 'diagnostic-output')
    return options, paths, native


def test_prepare_spec_pins_existing_evidence_without_creating_output_or_loading_credentials(diagnostic, prepared_inputs, monkeypatch):
    options, paths, native = prepared_inputs
    monkeypatch.setattr(diagnostic, 'configure_environment', lambda: pytest.fail('preparation read credentials'))
    spec = diagnostic.prepare_spec(options)
    diagnostic.validate_spec(spec)
    assert spec['native_predictions_sha256'] == native['prediction_files'][0]['sha256']
    assert spec['native_checkpoint_sha256'] == native['state_audits']['checkpoint_sha256']
    assert spec['measurement_aggregate_eligible'] is False
    assert not options.output.exists()


@pytest.mark.parametrize('changed', ['predictions.npz', 'task1_state.pt', 'initial.pt', 'encoder.npz', 'raw-data', 'protected.py'])
def test_preparation_cannot_adopt_artifacts_changed_after_native_completion(diagnostic, prepared_inputs, changed):
    options, paths, _ = prepared_inputs
    paths[changed].write_bytes(b'changed-after-completion')
    with pytest.raises(ValueError):
        diagnostic.prepare_spec(options)
    assert not options.output.exists()
