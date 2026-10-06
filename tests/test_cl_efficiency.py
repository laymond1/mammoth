"""Benchmark preparation/telemetry tests. These do not train any CL algorithm."""

import ast
import csv
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import benchmark_cl
from scripts.cl_efficiency import Recorder, capture_accuracy, instrument, integrate_power, inventory, notify_slack, prepare_loader_metadata, tensor_bytes, valid_slack_webhook


ROOT = benchmark_cl.ROOT


def test_manifest_covers_source_without_importing_models():
    entries = json.loads((ROOT / 'benchmarks/cl_efficiency/manifest.json').read_text())['models']
    benchmark_cl.validate_manifest(entries)
    sources = inventory(ROOT)
    assert len(sources) == len(entries)
    assert len([entry for entry in entries if entry['kind'] == 'inference']) == 3
    assert {entry['model'] for entry in entries if entry.get('deferred_reason')} == {'joint', 'joint-gcl', 'lws', 'cnll', 'puridiver'}
    assert {entry['dataset'] for entry in entries} == {'seq-cifar100-224'}


def test_explicit_required_model_args_are_provided():
    entries = json.loads((ROOT / 'benchmarks/cl_efficiency/manifest.json').read_text())['models']
    for entry in entries:
        tree = ast.parse((ROOT / entry['source']).read_text())
        required = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                if any(k.arg == 'required' and isinstance(k.value, ast.Constant) and k.value.value for k in node.keywords):
                    required.append(ast.literal_eval(node.args[0])[2:])
        assert set(required) <= entry['args'].keys(), entry['model']
        if 'add_rehearsal_args' in (ROOT / entry['source']).read_text():
            assert entry['args'].get('buffer_size') == 500, entry['model']
        if entry['model'] in {'spr', 'cnll', 'puridiver', 'er-ace-aer-abs'}:
            assert entry['clean_true_labels']  # add metadata via native wrapper without corrupting labels


def test_default_cli_never_executes_or_notifies(tmp_path, monkeypatch):
    execute = Mock(side_effect=AssertionError('No experiments allowed'))
    notify = Mock(side_effect=AssertionError('No messages allowed'))
    monkeypatch.setattr(benchmark_cl, 'execute', execute)
    monkeypatch.setattr(benchmark_cl, 'notify_slack', notify)
    monkeypatch.setattr(benchmark_cl, 'readiness', lambda *args: {'experiment_executed': False})
    monkeypatch.setattr(benchmark_cl, 'configure_environment', lambda: {})
    assert benchmark_cl.main(['--output', str(tmp_path), '--models', 'l2p,second_order']) == 0
    plan = json.loads((tmp_path / 'plan.json').read_text())
    assert plan['experiment_executed'] is False
    assert [job['model'] for job in plan['jobs']] == ['l2p', 'second-order']
    for job in plan['jobs']:
        arguments = dict(zip(job['main_args'][::2], job['main_args'][1::2]))
        assert arguments['--stop_after'] == '1'
        assert arguments['--n_epochs'] == '1'
        assert arguments['--num_workers'] == '4'
        assert arguments['--permute_classes'] == '0'
        assert arguments['--debug_mode'] == '0'
        assert arguments['--eval_future'] == '0'
        assert arguments['--dataset'] == 'seq-cifar100-224'
        assert arguments['--noise_rate'] == '0'
        assert arguments['--validation'] == '0'
        assert len(job['main_args'][::2]) == len(arguments)
    execute.assert_not_called()
    notify.assert_not_called()


def test_wandb_disabled_initialization_with_current_sdk(monkeypatch):
    """The removed get_url() API must not prevent disabled/test runs from starting."""
    import wandb
    from utils import training

    monkeypatch.setattr(training, 'wandb', wandb)
    monkeypatch.setenv('MAMMOTH_TEST', '1')
    args = SimpleNamespace(wandb_name=None, model='sgd', conf_jobnum='test-connection',
                           wandb_project='cl-efficiency', wandb_entity='test-user')
    try:
        training.initialize_wandb(args)
        assert wandb.run.disabled is True
        assert args.wandb_url is None
    finally:
        wandb.finish()


def test_shared_data_root_is_explicit_in_plan(tmp_path, monkeypatch):
    monkeypatch.setenv('CL_DATA_ROOT', str(tmp_path / 'shared-data'))
    monkeypatch.setenv('CL_GPU', '1')
    entries = json.loads((ROOT / 'benchmarks/cl_efficiency/manifest.json').read_text())['models']
    args = benchmark_cl.parser().parse_args(['--models', 'l2p'])
    job = benchmark_cl.make_plan(args, entries)[0]
    arguments = dict(zip(job['main_args'][::2], job['main_args'][1::2]))
    assert arguments['--base_path'] == str(tmp_path / 'shared-data')
    assert arguments['--device'] == 'cuda:0'
    assert args.gpu == '1'
    overridden = benchmark_cl.parser().parse_args(['--models', 'l2p', '--data-root', str(tmp_path / 'explicit'), '--gpu', 'cpu'])
    command = benchmark_cl.make_plan(overridden, entries)[0]['main_args']
    assert command[command.index('--base_path') + 1] == str(tmp_path / 'explicit')
    assert command[command.index('--device') + 1] == 'cpu'


def test_workers_cannot_start_from_default_environment(tmp_path, monkeypatch):
    monkeypatch.delenv('CL_EFFICIENCY_EXECUTE', raising=False)
    with pytest.raises(ValueError, match='explicit'):
        benchmark_cl.worker(tmp_path / 'does-not-exist.json')


def test_unknown_or_reserved_args_are_rejected(tmp_path):
    entries = json.loads((ROOT / 'benchmarks/cl_efficiency/manifest.json').read_text())['models']
    args = benchmark_cl.parser().parse_args(['--models', 'not-an-algorithm', '--output', str(tmp_path)])
    with pytest.raises(ValueError, match='Unknown'):
        benchmark_cl.make_plan(args, entries)
    entries[0]['args']['stop_after'] = 10
    with pytest.raises(ValueError, match='reserved'):
        benchmark_cl.validate_manifest(entries)


def test_mixed_datasets_are_rejected():
    entries = json.loads((ROOT / 'benchmarks/cl_efficiency/manifest.json').read_text())['models']
    entries[0]['dataset'] = 'seq-cifar10'
    with pytest.raises(ValueError, match='all models'):
        benchmark_cl.validate_manifest(entries)


def test_dualprompt_manifest_uses_real_parser_options():
    entries = json.loads((ROOT / 'benchmarks/cl_efficiency/manifest.json').read_text())['models']
    entry = next(entry for entry in entries if entry['model'] == 'dualprompt')
    assert entry['args']['use_permute_fix'] == 0
    assert entry['args']['batchwise_prompt'] == 0
    tree = ast.parse((ROOT / entry['source']).read_text())
    options = {ast.literal_eval(node.args[0])[2:] for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == 'add_argument' and node.args and isinstance(node.args[0], ast.Constant)}
    assert set(entry['args']) - {'dataset_config', 'lr'} <= options


def test_static_models_have_no_training_epochs():
    entries = json.loads((ROOT / 'benchmarks/cl_efficiency/manifest.json').read_text())['models']
    args = benchmark_cl.parser().parse_args(['--models', 'clip,llava,idefics'])
    for job in benchmark_cl.make_plan(args, entries):
        arguments = dict(zip(job['main_args'][::2], job['main_args'][1::2]))
        assert arguments['--n_epochs'] == '0'
        assert arguments['--inference_only'] == '1'


def test_energy_integral_and_phase_boundaries():
    samples = [{'elapsed_s': 0., 'gpu_power_w': 100.}, {'elapsed_s': 2., 'gpu_power_w': 200.}]
    assert integrate_power(samples, [(0., 2.)], 3.) == (300., 2.)
    assert integrate_power(samples, [(.5, 1.5)], 3.) == (150., 1.)
    first = integrate_power(samples, [(0., 1.)], 3.)[0]
    second = integrate_power(samples, [(1., 2.)], 3.)[0]
    assert first + second == 300.


@pytest.mark.parametrize('power', [None, float('nan'), float('inf'), -1.])
def test_missing_or_invalid_energy_is_not_zero(power):
    samples = [{'elapsed_s': 0., 'gpu_power_w': power}, {'elapsed_s': 1., 'gpu_power_w': 100.}]
    assert integrate_power(samples, [(0., 1.)], 3.) == (None, 0.)


def test_long_power_gaps_are_not_extrapolated():
    samples = [{'elapsed_s': 0., 'gpu_power_w': 100.}, {'elapsed_s': 10., 'gpu_power_w': 200.}]
    assert integrate_power(samples, [(0., 10.)], 3.) == (None, 0.)


def test_slack_secret_is_not_returned_on_failure(monkeypatch):
    webhook = 'https://hooks.slack.com/services/test/secret/value'
    monkeypatch.setenv('SLACK_WEBHOOK_URL', webhook)
    monkeypatch.setattr('scripts.cl_efficiency.urlopen', Mock(side_effect=OSError(webhook)))
    status = notify_slack('mock notification')
    assert status == 'failed:OSError'
    assert webhook not in status


def test_disabled_slack_never_opens_a_connection(monkeypatch):
    network = Mock(side_effect=AssertionError('No network writes'))
    monkeypatch.setattr('scripts.cl_efficiency.urlopen', network)
    assert notify_slack('message', enabled=False) == 'disabled'
    monkeypatch.setenv('SLACK_WEBHOOK_URL', 'https://example.com/secret')
    assert notify_slack('message') == 'missing_or_invalid_webhook'
    network.assert_not_called()


def test_slack_webhook_readiness_is_read_only(monkeypatch):
    network = Mock(side_effect=AssertionError('No network writes'))
    monkeypatch.setattr('scripts.cl_efficiency.urlopen', network)
    monkeypatch.setenv('SLACK_WEBHOOK_URL', 'https://hooks.slack.com/services/test/secret/value')
    assert valid_slack_webhook()
    monkeypatch.setenv('SLACK_WEBHOOK_URL', 'https://hooks.slack.com/other')
    assert not valid_slack_webhook()
    network.assert_not_called()


def test_clean_metadata_keeps_original_samples_and_labels():
    import numpy as np
    labels = np.array([3, 4, 3])
    dataset = SimpleNamespace(targets=labels, indexes=np.array([9, 7, 5]), extra_return_fields=(), transform='same transform')

    def add_field(name, values):
        setattr(dataset, name, values)
        dataset.extra_return_fields += (name,)

    dataset.add_extra_return_field = add_field
    loader = SimpleNamespace(dataset=dataset)
    baseline = prepare_loader_metadata(loader)
    result = prepare_loader_metadata(loader, clean_true_labels=True)
    assert result['indices_labels_sha256'] == baseline['indices_labels_sha256']
    assert result['extra_return_fields'] == ['true_labels']
    assert result['samples'] == 3
    assert np.array_equal(dataset.targets, labels)
    assert np.array_equal(dataset.true_labels, labels)
    assert dataset.targets is not dataset.true_labels
    with pytest.raises(ValueError, match='Clean-label'):
        prepare_loader_metadata(loader, clean_true_labels=True)


def test_tensor_views_count_allocated_storage_once():
    # Test bookkeeping only, without forward/backward, optimizer, data or GPU.
    import torch
    storage = torch.empty(12)
    assert tensor_bytes([storage, storage[2:4]]) == 12 * storage.element_size()


def test_accuracy_uses_native_numbers_without_repr_eval():
    import numpy as np
    logger = SimpleNamespace(setting='class-il', accs=[np.float64(72.5)], accs_mask_classes=[np.float64(81.)],
                             fullaccs=[[72.5]], fullaccs_mask_classes=[[81.]])
    result = capture_accuracy(logger)
    assert result['mean_accuracy_pct'] == 72.5
    assert result['task_il_mean_accuracy_pct'] == 81.
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_task_only_accuracy_placeholder_is_not_a_measured_zero():
    logger = SimpleNamespace(setting='class-il', accs=[0.], accs_mask_classes=[81.],
                             fullaccs=[[0.]], fullaccs_mask_classes=[[81.]])
    result = capture_accuracy(logger, ['task-il'])
    assert result['mean_accuracy_pct'] is None
    assert result['fullaccs'] is None
    assert result['class_il_status'] == 'unsupported_by_model'
    assert result['task_il_mean_accuracy_pct'] == 81.


class FakeModel:
    """Hook contract stub, with no neural network or learning."""
    def __init__(self):
        self.calls = []

    def parameters(self):
        return iter(())

    def buffers(self):
        return iter(())

    def meta_begin_task(self, dataset):
        self.calls.append('begin')
        self.meta_observe([1, 2])  # hidden work must not count as outer-loop examples

    def meta_end_task(self, dataset):
        self.calls.append('end')

    def meta_begin_epoch(self, *args):
        pass

    def meta_end_epoch(self, *args):
        pass

    def meta_observe(self, inputs, **kwargs):
        self.calls.append('observe')
        return 3.25

    def forward(self, inputs):
        return inputs

    def meta_end_eval(self, *args):
        pass


@pytest.mark.parametrize('raise_inside', [True, False])
def test_instrumentation_calls_and_restores_meta_hooks(tmp_path, raise_inside):
    model = FakeModel()
    recorder = Recorder(tmp_path, gpu='cpu')
    dataset = SimpleNamespace(get_data_loaders=lambda: (None, None), evaluate=lambda: model.forward([1, 2, 3]))
    training = SimpleNamespace(train_single_epoch=lambda: model.meta_observe([1, 2, 3, 4]))
    original = model.meta_observe
    try:
        with instrument(recorder, model, dataset, training):
            assert dataset.get_data_loaders() == (None, None)
            model.meta_begin_task(dataset)
            assert recorder.training_examples == 0
            assert training.train_single_epoch() == 3.25
            assert dataset.evaluate() == [1, 2, 3]
            model.meta_end_eval(dataset, ([], []))
            model.meta_end_task(dataset)
            if raise_inside:
                raise RuntimeError('test failure')
    except RuntimeError:
        assert raise_inside
    assert recorder.training_examples == 4
    assert recorder.evaluation_examples == 3
    assert recorder.tasks_completed == 1
    assert model.meta_observe == original
    assert 'meta_observe' not in vars(model)
    assert model.calls == ['begin', 'observe', 'observe', 'end']
    assert 'after_end_task' in recorder.snapshots
    assert recorder.data_snapshots['first_task'] == {'train': {}, 'test': {}}


def test_failed_run_can_preserve_local_metrics_without_gpu(tmp_path):
    recorder = Recorder(tmp_path, gpu='cpu', interval=.1)
    recorder.start()
    with recorder.phase('initialize'):
        pass
    record = recorder.save({'status': 'failed', 'error_type': 'TestFailure'})
    assert record['status'] == 'failed'
    assert record['metrics']['gpu_board_energy_j'] is None
    assert record['metrics']['cuda_peak_allocated_mib'] is None
    assert (tmp_path / 'telemetry.csv').exists()


def test_profiler_timings_never_enter_comparison_csv(tmp_path):
    results = [{'model': 'fake', 'group': 'test', 'status': 'completed', 'flops_status': 'completed',
                'measurement': {'metrics': {'process_wall_s': 2.}, 'accuracy': {'mean_accuracy_pct': 75.}},
                'flops': {'metrics': {'process_wall_s': 999.}, 'flops': {'supported_ops_total': 12345}}}]
    benchmark_cl.write_results(tmp_path, results)
    row = next(csv.DictReader((tmp_path / 'results.csv').open()))
    assert row['efficiency/process_wall_s'] == '2.0'
    assert row['flops_partial'] == '12345'
    assert row['accuracy/mean_accuracy_pct'] == '75.0'


@pytest.mark.parametrize('fails', [False, True])
def test_worker_preserves_native_hooks_and_saves_before_wandb_finish(tmp_path, monkeypatch, fails):
    """Mock the framework and SDK completely; no dataset, network, or model training."""
    import numpy as np

    class Logger:
        setting = 'class-il'
        accs, accs_mask_classes = [75.], [80.]
        fullaccs, fullaccs_mask_classes = [[75.]], [[80.]]

        def write(self, *args, **kwargs):
            return 'native log written'

    class DatasetWrapper:
        targets = np.array([0, 1])
        indexes = np.array([2, 7])
        extra_return_fields = ()
        transform = 'mock transform'

        def add_extra_return_field(self, name, values):
            setattr(self, name, values)
            self.extra_return_fields += (name,)

    loader = SimpleNamespace(dataset=DatasetWrapper())
    test_loader = SimpleNamespace(dataset=DatasetWrapper())
    model = FakeModel()
    dataset = SimpleNamespace(NAME=benchmark_cl.COMMON_DATASET, N_CLASSES=100, N_TASKS=10, N_CLASSES_PER_TASK=10,
                              train_loader=loader, test_loaders=[test_loader], get_data_loaders=lambda: (loader, test_loader),
                              evaluate=lambda *args: model.forward([1, 2]))
    arguments = SimpleNamespace(dataset=benchmark_cl.COMMON_DATASET, nowand=0, stop_after=1, debug_mode=0,
                                eval_future=0, noise_rate=0, validation=0, label_perc=1, label_perc_by_class=1)
    main_module, training_module, utils_module, wandb_module = (ModuleType(name) for name in ('main', 'utils.training', 'utils', 'wandb'))
    main_module.parse_args = lambda **kwargs: arguments
    original_initialize = lambda *args: (model, dataset, arguments)
    main_module.initialize = original_initialize
    training_module.Logger = Logger
    training_module.train_single_epoch = lambda: model.meta_observe([1, 2, 3])
    utils_module.training = training_module
    artifact = Mock()
    wandb_module.Artifact = Mock(return_value=artifact)
    wandb_module.log = Mock()
    wandb_module.run = SimpleNamespace(url='https://wandb.ai/mock/run', summary={}, log_artifact=Mock())

    def finish(**kwargs):
        assert (tmp_path / 'metrics.json').exists()
        assert (tmp_path / 'telemetry.csv').exists()

    native_finish = Mock(side_effect=finish)
    wandb_module.finish = native_finish

    def native_main(args):
        native_model, native_dataset, _ = main_module.initialize(args)
        native_dataset.get_data_loaders()
        native_model.meta_begin_task(native_dataset)
        training_module.train_single_epoch()
        if fails:
            raise RuntimeError('controlled mock failure')
        native_model.meta_end_task(native_dataset)
        native_dataset.evaluate()
        native_model.meta_end_eval(native_dataset, ([], []))
        assert Logger().write() == 'native log written'
        wandb_module.finish()

    main_module.main = native_main
    for name, module in (('main', main_module), ('utils', utils_module), ('utils.training', training_module), ('wandb', wandb_module)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, 'argv', sys.argv.copy())
    monkeypatch.setattr(benchmark_cl.importlib.metadata, 'version', lambda package: 'mock-version')
    monkeypatch.setenv('CL_EFFICIENCY_EXECUTE', '1')
    spec = {'model': 'mock', 'dataset': benchmark_cl.COMMON_DATASET, 'group': 'mock', 'kind': 'training',
            'pass': 'measurement', 'suite': 'mock-suite', 'output': str(tmp_path), 'gpu': 'cpu', 'sample_interval': .1,
            'main_args': [], 'wandb_mode': 'online', 'wandb_entity': 'mock', 'wandb_project': 'mock', 'clean_true_labels': True}
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(spec))
    assert benchmark_cl.worker(spec_path) == int(fails)
    record = json.loads((tmp_path / 'metrics.json').read_text())
    assert record['status'] == ('failed' if fails else 'completed')
    assert record['metrics']['outer_training_stream_examples'] == 3
    assert record['metrics']['tasks_completed'] == (0 if fails else 1)
    assert record['data_snapshots']['first_task']['train']['clean_true_labels_added']
    assert np.array_equal(loader.dataset.true_labels, loader.dataset.targets)
    assert main_module.initialize is original_initialize
    assert 'meta_observe' not in vars(model)
    assert wandb_module.finish is native_finish
    native_finish.assert_called_once_with(exit_code=int(fails))
    wandb_module.run.log_artifact.assert_called_once_with(artifact)
    wandb_module.run.log_artifact.return_value.wait.assert_called_once_with()
    wandb_module.log.assert_called_once()
    if not fails:
        assert record['accuracy']['mean_accuracy_pct'] == 75.


def test_ease_dry_run_does_not_hash_weights_or_initialize_a_model(tmp_path, monkeypatch):
    pin = Mock(side_effect=AssertionError('Weight access requires execution'))
    monkeypatch.setattr(benchmark_cl, 'ease_execution_pin', pin)
    monkeypatch.setattr(benchmark_cl, 'readiness', lambda *args: {'experiment_executed': False})
    monkeypatch.setattr(benchmark_cl, 'configure_environment', lambda: {})
    assert benchmark_cl.main(['--models', 'ease', '--output', str(tmp_path)]) == 0
    assert json.loads((tmp_path / 'plan.json').read_text())['experiment_executed'] is False
    pin.assert_not_called()


@pytest.fixture
def ease_pin_files(tmp_path, monkeypatch):
    """Small public stand-ins, without accessing the shared weight or dataset cache."""
    monkeypatch.setattr(benchmark_cl, 'ROOT', tmp_path)
    for name in ('scripts/benchmark_cl.py', 'scripts/cl_efficiency.py', 'main.py', 'utils/training.py',
                 'models/ease.py', 'backbone/ease_vit.py', 'models/ease_utils/vit.py',
                 'models/ease_utils/network.py', 'models/ease_utils/classifier.py',
                 'datasets/configs/seq-cifar100-224/ease.yaml', 'uv.lock'):
        source = tmp_path / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('public test source')
    directory = tmp_path / 'benchmarks/cl_efficiency/ease'
    directory.mkdir(parents=True)
    checkpoint = tmp_path / 'pinned.npz'
    checkpoint.write_bytes(b'public mock checkpoint bytes; never loaded')
    digest = benchmark_cl.file_hash(checkpoint)
    official = {'official_reference': {'data_snapshots': {'task1_train': {'n_samples': 5000}, 'task1_test': {'n_samples': 1000}}}}
    native = {'native_validation': {'status': 'completed', 'acceptance_status': 'pass'},
              'acceptance': {'status': 'pass'}, 'checkpoint_sha256': digest}
    (directory / 'official.json').write_text(json.dumps(official))
    (directory / 'native.json').write_text(json.dumps(native))
    plan = {'efficiency': {'training_authorized': True},
            'reference': {'checkpoint_file': str(checkpoint), 'checkpoint_sha256': digest, 'result_summary': 'official.json'},
            'validation': {'result_summary': 'native.json'}}
    (directory / 'validation_plan.json').write_text(json.dumps(plan))
    return SimpleNamespace(directory=directory, checkpoint=checkpoint, plan=plan, native=native)


def test_ease_pin_requires_accepted_validation_and_unchanged_local_weights(ease_pin_files):
    files = ease_pin_files
    pin = benchmark_cl.ease_execution_pin()
    assert pin['checkpoint_sha256'] == benchmark_cl.file_hash(files.checkpoint)
    assert len(pin['execution_source_sha256']) == 11
    assert pin['reference_data_snapshots']['task1_train']['n_samples'] == 5000
    files.native['acceptance']['status'] = 'fail'
    (files.directory / 'native.json').write_text(json.dumps(files.native))
    with pytest.raises(ValueError, match='accepted native validation'):
        benchmark_cl.ease_execution_pin()
    files.native['acceptance']['status'] = 'pass'
    (files.directory / 'native.json').write_text(json.dumps(files.native))
    files.checkpoint.write_bytes(b'changed public mock checkpoint')
    with pytest.raises(ValueError, match='SHA256'):
        benchmark_cl.ease_execution_pin()


def test_ease_pin_rejects_unauthorized_execution(ease_pin_files):
    files = ease_pin_files
    files.plan['efficiency']['training_authorized'] = False
    (files.directory / 'validation_plan.json').write_text(json.dumps(files.plan))
    with pytest.raises(ValueError, match='not been authorized'):
        benchmark_cl.ease_execution_pin()


@pytest.mark.parametrize('command', [[], ['--code_optimization', '0'], ['--code_optimization=0']])
def test_ease_pinned_cli_accepts_only_the_explicit_value(command):
    benchmark_cl._pinned_argument(command, '--code_optimization', '0')
    assert command in (['--code_optimization', '0'], ['--code_optimization=0'])


@pytest.mark.parametrize('command', [['--code_optimization', '1'], ['--code_optimization=2'], ['--code_optimization']])
def test_ease_pinned_cli_rejects_conflicts(command):
    with pytest.raises(ValueError, match='requires'):
        benchmark_cl._pinned_argument(command, '--code_optimization', '0')


def test_ease_precision_matches_native_validation_without_initializing_cuda():
    torch = SimpleNamespace(float32='float32', set_default_dtype=Mock(), set_float32_matmul_precision=Mock(),
                            get_float32_matmul_precision=lambda: 'highest',
                            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                                     cudnn=SimpleNamespace(allow_tf32=False, deterministic=False, benchmark=True)))
    result = benchmark_cl._configure_ease_precision(torch)
    torch.set_default_dtype.assert_called_once_with('float32')
    torch.set_float32_matmul_precision.assert_called_once_with('highest')
    assert result['tf32_allowed'] is False
    assert result['cudnn_tf32_allowed'] is True
    assert result['cudnn_deterministic'] is True and result['cudnn_benchmark'] is False


@pytest.fixture
def ease_audit(tmp_path):
    """Static CPU tensors only: no model forward/backward, optimizer, or real data."""
    import numpy as np
    import torch
    from torch.utils.data import SequentialSampler
    from scripts.ease_reference_utils import dataset_fingerprint

    backbone = torch.nn.Module()
    backbone.backbone = torch.nn.Module()
    backbone.backbone.register_parameter('weight', torch.nn.Parameter(torch.ones(2), requires_grad=False))
    backbone.register_parameter('adapter', torch.nn.Parameter(torch.ones(1)))
    backbone.pretrained_loaded, backbone.feature_dim, backbone.num_adapters = True, 768, 1
    backbone.extract_all = Mock(side_effect=lambda inputs: inputs)
    net = torch.nn.Module()
    net.backbone = backbone
    net.classifier = torch.nn.Module()
    net.classifier.register_parameter('weight', torch.nn.Parameter(torch.zeros(10, 768), requires_grad=False))
    net.register_buffer('task_finalized', torch.tensor(True))
    model = FakeModel()
    model.net = net
    model.forward = Mock(side_effect=lambda inputs: inputs)
    train = SimpleNamespace(data=np.zeros((5000, 1, 1, 3), dtype=np.uint8), targets=np.arange(5000) % 10,
                            indexes=np.arange(5000), transform='static transform')
    test = SimpleNamespace(data=np.zeros((1000, 1, 1, 3), dtype=np.uint8), targets=np.arange(1000) % 10,
                           indexes=np.arange(1000), transform='static transform')
    dataset = SimpleNamespace(train_loader=SimpleNamespace(dataset=train),
                              test_loaders=[SimpleNamespace(dataset=test, sampler=SequentialSampler(range(1000)), drop_last=False)])
    args = SimpleNamespace(n_epochs=1, batch_size=64, num_workers=4, seed=1993, vit_size='base', adapter_size=64,
                           ease_pretrain_type='in21k_ft_in1k', dataset_config='ease', code_optimization=0, drop_last=False,
                           fitting_mode='epochs', permute_classes=False, joint=False, lr=.025, optim_mom=.9, optim_wd=.0005,
                           optimizer='sgd', lr_scheduler='cosine', scheduler_mode='epoch', eval_epochs=None,
                           start_from=None, loadcheck=None, ease_checkpoint=str(tmp_path / 'pinned.npz'))
    metadata = {'checkpoint_file': args.ease_checkpoint, 'accuracy': {'mean_accuracy_pct': 100.},
                'reference_data_snapshots': {f'task1_{split}': dataset_fingerprint(source.data, source.targets, source.indexes)
                                            for split, source in (('train', train), ('test', test))}}
    recorder = Recorder(tmp_path, gpu='cpu')
    recorder.data_snapshots['first_task'] = {'train': {}, 'test': {}}
    audit = benchmark_cl.EaseAudit(model, dataset, args, recorder, metadata, torch)
    audit.args = args
    return audit


def finalize_static_ease_audit(audit):
    audit.losses = [1.] * 79
    audit.end_epoch()
    audit.model.net.backbone.adapter.requires_grad_(False)
    audit.prototype_examples = 5000
    audit.end_task()
    audit.predictions = [audit.dataset.test_loaders[0].dataset.targets.copy()]
    audit.recorder.steps, audit.recorder.training_examples = 79, 5000
    audit.recorder.evaluation_examples, audit.recorder.tasks_completed = 1000, 1


def test_ease_audit_saves_predictions_and_proves_the_complete_protocol(ease_audit):
    import numpy as np
    audit = ease_audit
    finalize_static_ease_audit(audit)
    audit.complete()
    assert len(audit.checks) == 8 and all(audit.checks.values())
    assert audit.metadata['initial_frozen_backbone_sha256'] == audit.metadata['final_frozen_backbone_sha256']
    with np.load(audit.recorder.output / 'task1_predictions.npz') as saved:
        assert saved['predictions'].shape == (1000, 1)
        assert np.array_equal(saved['predictions'][:, 0], saved['labels'])
        assert np.array_equal(saved['global_ids'], np.arange(1000))


@pytest.mark.parametrize('field,value', [('steps', 78), ('training_examples', 4992), ('evaluation_examples', 999), ('tasks_completed', 0)])
def test_ease_audit_cannot_accept_a_partial_native_return(ease_audit, field, value):
    audit = ease_audit
    finalize_static_ease_audit(audit)
    setattr(audit.recorder, field, value)
    with pytest.raises(RuntimeError, match='complete 79-step'):
        audit.complete()
    assert audit.checks['training_protocol_complete'] is False


@pytest.mark.parametrize('changed', ['raw_image', 'backbone', 'prototype', 'adapter_count', 'finalized'])
def test_ease_end_task_rejects_corrupted_evidence(ease_audit, changed):
    import torch
    audit = ease_audit
    audit.model.net.backbone.adapter.requires_grad_(False)
    if changed == 'raw_image':
        audit.dataset.train_loader.dataset.data[0, 0, 0, 0] = 1
    elif changed == 'backbone':
        audit.ptm.weight.data[0] = 2.
    elif changed == 'prototype':
        audit.model.net.classifier.weight.data[0, 0] = float('nan')
    elif changed == 'adapter_count':
        audit.model.net.backbone.num_adapters = 2
    else:
        audit.model.net.task_finalized.fill_(False)
    with pytest.raises(RuntimeError):
        audit.end_task()
    assert audit.checks['training_protocol_complete'] is False


@pytest.mark.parametrize('frozen', [False, True])
def test_ease_epoch_rejects_nonfinite_adapter_or_any_frozen_gradient(ease_audit, frozen):
    import torch
    audit = ease_audit
    if frozen:
        audit.ptm.weight.grad = torch.zeros_like(audit.ptm.weight)
    else:
        audit.model.net.backbone.adapter.grad = torch.full_like(audit.model.net.backbone.adapter, float('nan'))
    with pytest.raises(RuntimeError, match='Frozen|Non-finite'):
        audit.end_epoch()


def test_ease_audit_checks_effective_args_after_native_configuration(ease_audit):
    import torch
    audit = ease_audit
    audit.args.n_epochs = 20
    with pytest.raises(ValueError, match='Effective EASE arguments'):
        benchmark_cl.EaseAudit(audit.model, audit.dataset, audit.args, audit.recorder, audit.metadata, torch)


def test_ease_wrapper_captures_existing_calls_and_restores_them(ease_audit):
    import torch
    audit = ease_audit
    original_forward, original_extract = audit.model.forward, audit.model.net.backbone.extract_all
    end_epoch, end_task = Mock(), Mock()
    audit.end_epoch, audit.end_task = end_epoch, end_task
    with benchmark_cl.instrument_ease(audit):
        with audit.recorder.phase('train_epoch'):
            assert audit.model.meta_observe([0, 1]) == 3.25
        with audit.recorder.phase('end_task'):
            assert audit.model.net.backbone.extract_all([0, 1, 2]) == [0, 1, 2]
        logits = torch.tensor([[2., 2.], [0., 1.]])
        with audit.recorder.phase('evaluation'):
            returned = audit.model.forward(logits)
            returned.fill_(float('-inf'))  # native task-IL mask must not corrupt captured predictions
        audit.model.meta_end_epoch()
        audit.model.meta_end_task(audit.dataset)
    assert audit.losses == [3.25] and audit.prototype_examples == 3
    assert audit.predictions[0].tolist() == [0, 1]
    assert audit.model.forward is original_forward and audit.model.net.backbone.extract_all is original_extract
    original_forward.assert_called_once()
    original_extract.assert_called_once()
    end_epoch.assert_called_once_with()
    end_task.assert_called_once_with()


@pytest.mark.parametrize('failure', ['upload', 'finish', 'missing_run', None])
def test_wandb_failure_keeps_successful_computation_and_returns_failure(tmp_path, failure):
    artifact, uploaded, finish = Mock(), Mock(), Mock()
    if failure == 'upload':
        uploaded.wait.side_effect = RuntimeError('mock upload failure')
    if failure == 'finish':
        finish.side_effect = RuntimeError('mock SDK shutdown failure')
    run = None if failure == 'missing_run' else SimpleNamespace(summary={}, log_artifact=Mock(return_value=uploaded))
    wandb = SimpleNamespace(run=run, log=Mock(), Artifact=Mock(return_value=artifact), finish=Mock())
    record = {'status': 'completed', 'computation_status': 'completed', 'metrics': {}, 'phases': {},
              'model_snapshots': {}, 'accuracy': {}}
    (tmp_path / 'task1_predictions.npz').write_bytes(b'public evidence stand-in')
    spec = {'suite': 'mock-suite', 'model': 'ease', 'pass': 'measurement', 'wandb_mode': 'online'}
    code = benchmark_cl.finish_worker_logging(tmp_path, spec, record, wandb, finish, 0)
    assert code == int(failure is not None)
    assert record['computation_status'] == 'completed'
    assert record['status'] == ('logging_failed' if failure else 'completed')
    assert wandb.finish is finish
    if failure != 'missing_run':
        uploaded.wait.assert_called_once_with()
        finish.assert_called_once_with(exit_code=int(failure == 'upload'))
        artifact.add_file.assert_called_once_with(str(tmp_path / 'task1_predictions.npz'), name='task1_predictions.npz')
    else:
        assert record['wandb_logging_status'] == 'run_not_initialized'


def test_offline_artifact_never_waits_for_an_online_upload(tmp_path):
    saved, finish = Mock(), Mock()
    wandb = SimpleNamespace(run=SimpleNamespace(summary={}, log_artifact=Mock(return_value=saved)), log=Mock(),
                            Artifact=Mock(return_value=Mock()), finish=Mock())
    record = {'status': 'completed', 'computation_status': 'completed', 'metrics': {}, 'phases': {},
              'model_snapshots': {}, 'accuracy': {}}
    spec = {'suite': 'mock-suite', 'model': 'ease', 'pass': 'measurement', 'wandb_mode': 'offline'}
    assert benchmark_cl.finish_worker_logging(tmp_path, spec, record, wandb, finish, 0) == 0
    assert record['wandb_logging_status'] == 'offline_saved'
    saved.wait.assert_not_called()
