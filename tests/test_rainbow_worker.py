"""CPU-only worker contract, live audits and immutable-author epoch checks."""
from argparse import ArgumentParser
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts import rainbow_reference_utils as reference
from scripts import run_rainbow_worker as worker
from scripts import rainbow_protocol as shared


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def code():
    if not shared.REFERENCE_SOURCE.is_dir():
        pytest.skip('Optional pinned RainbowPrompt author checkout is absent')
    return reference.load_author_code(shared.REFERENCE_SOURCE)


def small_native(warmup=1):
    from models.rainbow_prompt_utils.vit import RainbowVisionTransformer
    from models.rainbow_prompt_utils.network import RainbowNetwork
    common = dict(num_classes=20, img_size=8, patch_size=4, embed_dim=12, depth=2, num_heads=3)
    query = RainbowVisionTransformer(**common)
    prompt = RainbowVisionTransformer(**common, pool_size=2, n_tasks=2, prompt_length=4,
        prompt_layers=(0, 1), self_attn_idx=(0,), D1=6, D2=5, KI_iter=2, warmup=warmup)
    net = RainbowNetwork(query, prompt, 2)
    return {'query': query, 'prompt': prompt}, net


def small_author(code, warmup=1):
    return reference.build_author_branches(code, num_classes=20, img_size=8, patch_size=4,
        embed_dim=12, depth=2, num_heads=3, pool_size=2, n_tasks=2, prompt_length=4,
        prompt_layers=(0, 1), self_attn_layers=(0,), D1=6, D2=5, KI_iter=2, warmup=warmup)


def record_for_test(mode='short_validation', side='native', kind='validation'):
    p = shared.protocol(mode)
    updates = p['tasks'] * p['n_epochs'] * 79
    result = {'mode': mode, 'side': side, 'pass': kind, 'method': 'rainbow-prompt', 'model': 'rainbow-prompt',
        'protocol': p, 'gpu': '1', 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency',
        'checkpoint_sha256': shared.CHECKPOINT_SHA256, 'canonical_pretrained_sha256': shared.CANONICAL_PRETRAINED_SHA256,
        'reference_commit': shared.REFERENCE_COMMIT, 'training_authorized': True,
        'source_mode': 'official-active-path', 'learned_gate_active': False, 'paper_equivalent': False,
        'measurement_aggregate_eligible': kind == 'measurement', 'status': 'completed', 'computation_status': 'completed',
        'precision': dict(shared.PRECISION), 'structural_checks': {key: True for key in shared.REQUIRED_CHECKS},
        'counts': {'observe_calls': updates, 'optimizer_updates': updates, 'task_begin_extra_samples': 0, 'task_end_extra_samples': 0},
        'epoch_results': [], 'task_results': [], 'first_batch_audits': [], 'prediction_files': [],
        'phases': {}, 'metrics': {'outer_observe_steps': updates, 'outer_training_stream_examples': p['tasks'] * p['n_epochs'] * 5000,
                    'evaluation_examples': 1000 if p['tasks'] == 1 else 3000, 'tasks_completed': p['tasks']}}
    for task in range(1, p['tasks'] + 1):
        for epoch in range(1, p['n_epochs'] + 1):
            result['epoch_results'].append({'task': task, 'epoch': epoch, 'steps': 79, 'samples': 5000,
                'seed': shared.phase_seed(1993, task - 1, 'train', epoch - 1), 'loss_mean': .3})
            result['first_batch_audits'].append({'task': task, 'epoch': epoch})
        result['task_results'].append(worker.exact_counts(np.zeros(task * 1000, dtype=int),
                                                        np.repeat(np.arange(task * 10), 100), task))
        result['prediction_files'].append({'task': task})
    result['accuracy'] = {'mean_accuracy_pct': result['task_results'][-1]['seen_accuracy_pct']}
    return result


def audit_for_test(branches, tmp_path, warmup=1):
    record = {'protocol': {'prompt_layers': [0, 1], 'n_epochs': 2, 'warmup_epochs': warmup, 'tasks': 2},
        'counts': {'observe_calls': 0, 'optimizer_updates': 0, 'task_begin_extra_samples': 0, 'task_end_extra_samples': 0},
        'first_batch_audits': [], 'cache_audits': [], 'state_audits': [], 'epoch_results': [],
        'sample_stream_audits': [], 'structural_checks': {key: False for key in shared.REQUIRED_CHECKS}}
    images, labels = torch.randn(3, 3, 8, 8), torch.tensor([0, 1, 2])
    raw = TensorDataset(images, labels, images)
    raw.indexes = np.arange(3)
    dataset = SimpleNamespace(train_loader=DataLoader(raw, batch_size=3, shuffle=True))
    recorder = SimpleNamespace(output=tmp_path, snapshots={})
    audit = worker.RainbowAudit(branches, dataset, recorder, record, torch, np)
    audit.task = 0
    audit.task_old = audit.old_state()
    optimizer = torch.optim.Adam([parameter for parameter in branches['prompt'].parameters() if parameter.requires_grad], lr=.0075)
    audit.attach_optimizer(optimizer)
    return audit, optimizer, dataset, record


@pytest.mark.parametrize('mode,kind', [('full_validation', 'validation'), ('short_validation', 'validation'),
                                      ('efficiency', 'measurement'), ('efficiency', 'flops')])
def test_completed_records_require_the_full_exact_workload(mode, kind):
    assert worker.validate_saved_record(record_for_test(mode, kind=kind))


@pytest.mark.parametrize('mutation', [lambda r: r['counts'].update(optimizer_updates=78),
    lambda r: r['counts'].update(task_begin_extra_samples=1), lambda r: r['metrics'].update(evaluation_examples=1040),
    lambda r: r['epoch_results'][0].update(seed=1993), lambda r: r['epoch_results'][0].update(samples=4999),
    lambda r: r['structural_checks'].update(stored_prompt_cache_timing_matches=False),
    lambda r: r['task_results'][0].update(seen_correct=999), lambda r: r.update(paper_equivalent=True),
    lambda r: r['first_batch_audits'].clear(), lambda r: r['protocol'].update(warmup_epochs=5)])
def test_completed_record_rejects_partial_or_changed_evidence(mutation):
    value = record_for_test()
    mutation(value)
    with pytest.raises(ValueError):
        worker.validate_saved_record(value)


def test_exact_old_new_counts_use_integer_predictions():
    result = worker.exact_counts(np.array([0, 0, 10, 10]), np.array([0, 1, 10, 11]), 2)
    assert result == {'task': 2, 'seen_samples': 4, 'seen_correct': 2, 'seen_accuracy_pct': 50.,
        'old_samples': 2, 'old_correct': 1, 'old_accuracy_pct': 50., 'new_samples': 2,
        'new_correct': 1, 'new_accuracy_pct': 50.}


def test_catalog_roundtrip_includes_both_heads_every_prompt_weight_and_cache(tmp_path):
    branches, _ = small_native()
    original = reference.learned_state(branches)
    path = tmp_path / 'initial.pt'
    result = reference.save_initial_state(branches, path, 'a' * 64)
    assert 'query.head.weight' in original and 'query.head.bias' in original
    assert 'prompt.rainbow_prompt.stored_rainbow_prompts' in original
    for module in branches.values():
        with torch.no_grad():
            module.head.weight.add_(2)
    assert reference.load_initial_state(branches, path, 'a' * 64) == result['learned_sha256']
    assert all(torch.equal(value, reference.learned_state(branches)[name]) for name, value in original.items())
    with pytest.raises(FileExistsError):
        reference.save_initial_state(branches, path, 'a' * 64)
    with pytest.raises(ValueError):
        reference.load_initial_state(branches, path, 'b' * 64)


def test_catalog_rejects_missing_query_head_or_cache(tmp_path):
    branches, _ = small_native()
    path = tmp_path / 'initial.pt'
    reference.save_initial_state(branches, path, 'a' * 64)
    payload = torch.load(path, weights_only=True)
    del payload['state']['query.head.weight']
    payload['learned_sha256'] = reference.state_digest(payload['state'])
    torch.save(payload, path)
    with pytest.raises(ValueError):
        reference.load_initial_state(branches, path, 'a' * 64)


def test_seen_criterion_excludes_future_classes_and_preserves_current_mask():
    logits = torch.randn(3, 20, requires_grad=True)
    masked = logits.index_fill(1, torch.arange(10), -float('inf'))
    loss = reference.SeenClassCriterion(20)(masked, torch.tensor([10, 11, 12]))
    assert torch.equal(loss, nn.functional.cross_entropy(logits[:, 10:], torch.tensor([0, 1, 2])))
    loss.backward()
    assert bool((logits.grad[:, :10] == 0).all())


def test_worker_requires_execution_gpu_and_online_environment_before_imports(monkeypatch, tmp_path):
    for key in ('CL_RAINBOW_EXECUTE', 'CUDA_VISIBLE_DEVICES', 'WANDB_MODE'):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match='explicit execution'):
        worker.worker(tmp_path / 'absent.json')


def test_routing_audit_verifies_batch_mean_and_rejects_oracle(tmp_path):
    branches, _ = small_native(warmup=0)
    audit, _, _, _ = audit_for_test(branches, tmp_path, 0)
    audit.task, audit.stage, audit.eval_test_task = 1, 'evaluation', 0
    features = torch.randn(3, 12)
    prompt = branches['prompt'].rainbow_prompt
    with torch.no_grad():
        prompt.stored_rainbow_prompts.normal_()
    kwargs = dict(layer=0, cls_features=features, cur_id=1, task_id=-1, train=False, p_type='Rainbow')
    prompt(torch.randn(3, 5, 12), **kwargs)
    assert len(audit.eval_routes) == 1
    assert audit.eval_routes[0]['samples'] == 3 and audit.eval_routes[0]['selected_task'] in (0, 1)
    kwargs['task_id'] = 0
    with pytest.raises(RuntimeError, match='oracle'):
        prompt(torch.randn(3, 5, 12), **kwargs)
    audit.close()


def test_optimizer_rejects_wrong_effective_lr_and_frozen_parameters(tmp_path):
    branches, _ = small_native()
    audit, _, _, _ = audit_for_test(branches, tmp_path)
    for params, lr in ((branches['prompt'].parameters(), .03),
                       (list(branches['prompt'].parameters()) + list(branches['query'].parameters()), .0075)):
        with pytest.raises(RuntimeError, match='task-local Adam'):
            audit.attach_optimizer(torch.optim.Adam(params, lr=lr))
    audit.close()


def test_actual_immutable_author_epoch_matches_native_meta_observe(code, monkeypatch, tmp_path):
    from models.rainbow_prompt import RainbowPrompt
    monkeypatch.setenv('MAMMOTH_DEVICE', 'cpu')
    torch.manual_seed(1993)
    author = small_author(code)
    torch.manual_seed(1993)
    native, net = small_native()
    for branch in author:
        native[branch].load_state_dict(author[branch].state_dict(), strict=True)
    net.pretrained_checkpoint = '/tmp/synthetic-only.npz'
    args = RainbowPrompt.get_parser(ArgumentParser()).parse_args([])
    for key, value in dict(n_epochs=2, rainbow_warmup_epochs=1, fitting_mode='epochs', eval_epochs=None,
        label_perc=1., nowand=1, device='cpu', optim_mom=.9, optim_nesterov=False).items():
        setattr(args, key, value)
    dataset = SimpleNamespace(N_CLASSES=20, N_TASKS=2, N_CLASSES_PER_TASK=10, SETTING='class-il',
                              get_offsets=lambda task: (task * 10, (task + 1) * 10))
    model = RainbowPrompt(net, nn.CrossEntropyLoss(), args, None, dataset)
    model.meta_begin_task(dataset)
    author_optimizer = torch.optim.Adam([parameter for parameter in author['prompt'].parameters() if parameter.requires_grad], lr=.0075)
    images, labels = torch.randn(3, 3, 8, 8), torch.tensor([0, 1, 2])
    source_args = SimpleNamespace(distributed=False, epochs=2, print_freq=10000, train_mask=True, nb_classes=20, balancing=.01)
    loader = [(images, labels)]
    # Exercise both Unique warmup and active Rainbow cache/dropout, with identical RNG.
    for epoch in (0, 1):
        torch.manual_seed(2993 + epoch)
        source = code.train_one_epoch(author['prompt'], author['query'], reference.SeenClassCriterion(10),
            loader, author_optimizer, torch.device('cpu'), epoch, max_norm=1., task_id=0,
            class_mask=[list(range(10)), list(range(10, 20))], args=source_args)
        torch.manual_seed(2993 + epoch)
        actual = model.meta_observe(images, labels, images, epoch=epoch)
        assert actual == source['Loss']
        assert all(torch.equal(value, native['prompt'].state_dict()[name]) for name, value in author['prompt'].state_dict().items())
        assert all(value.grad is None for value in author['query'].parameters())
    assert bool(author['prompt'].rainbow_prompt.stored_rainbow_prompts[0].abs().sum() > 0)
    assert hasattr(author['prompt'].blocks[0].attn, 'key_')
    assert not hasattr(author['query'].blocks[0].attn, 'key_')


def test_source_engine_live_audits_capture_preclip_and_preupdate_cache(code, tmp_path):
    branches = small_author(code, warmup=0)
    audit, optimizer, dataset, record = audit_for_test(branches, tmp_path, 0)
    audit.begin_epoch(0)
    with ExitStack() as stack:
        audit.install_common(stack)
        source_args = SimpleNamespace(distributed=False, epochs=2, print_freq=10000, train_mask=True, nb_classes=20, balancing=.01)
        stats = code.train_one_epoch(branches['prompt'], branches['query'], reference.SeenClassCriterion(10),
            reference.PairLoader(dataset.train_loader, audit.observed), optimizer, torch.device('cpu'), 0,
            max_norm=1., task_id=0, class_mask=[list(range(10)), list(range(10, 20))], args=source_args)
    assert np.isfinite(stats['Loss'])
    assert record['counts']['observe_calls'] == record['counts']['optimizer_updates'] == 1
    assert record['first_batch_audits'][0]['gradient_timing'] == 'before_clip'
    assert record['cache_audits'][0]['prefix_matches_pre_update_cache'] is True
    assert record['cache_audits'][0]['layers'] == [0, 1]
    audit.close()


def test_effective_argument_check_uses_real_cli_names(monkeypatch):
    import main
    command = shared.native_command({'mode': 'short_validation'})
    monkeypatch.setattr('sys.argv', ['main.py', *command])
    args = main.parse_args(cmd=command)
    dataset = SimpleNamespace(N_TASKS=10, N_CLASSES_PER_TASK=10, N_CLASSES=100)
    assert worker.effective_args(args, dataset, {'protocol': shared.protocol('short_validation')})['rainbow_d1'] == 56
    args.rainbow_balance = .02
    with pytest.raises(ValueError):
        worker.effective_args(args, dataset, {'protocol': shared.protocol('short_validation')})


def test_successful_uploaded_record_bytes_do_not_change_and_checkpoints_are_local(tmp_path):
    record = record_for_test()
    record.update(name='synthetic', wandb_url='https://wandb.ai/laymond1/cl-efficiency/runs/unit')
    uploaded = SimpleNamespace(id='artifact', state='COMMITTED', wait=Mock())
    captured = {}
    artifact = SimpleNamespace(add_file=lambda path: captured.setdefault(Path(path).name, Path(path).read_bytes()))
    wandb = SimpleNamespace(run=SimpleNamespace(id='unit', summary={}, log_artifact=lambda item: uploaded),
                           Artifact=lambda *a, **k: artifact, finish=Mock())
    (tmp_path / 'task1_state.pt').write_bytes(b'checkpoint never uploaded')
    worker.upload_record(wandb, tmp_path, record)
    assert captured['result.json'] == (tmp_path / 'result.json').read_bytes()
    assert captured['result.json'] == captured['artifact_metrics.json']
    assert 'task1_state.pt' not in captured
    assert wandb.run.summary['benchmark_status'] == 'completed'
    assert wandb.run.summary['efficiency/outer_observe_steps'] == 79
    assert json.loads((tmp_path / 'wandb_receipt.json').read_text())['artifact_state'] == 'COMMITTED'
    uploaded.wait.assert_called_once()
    wandb.finish.assert_called_once_with(exit_code=0)


def test_error_context_retains_locations_and_coordinates_without_exception_secrets():
    secret = 'example-private-webhook-token'
    recorder = SimpleNamespace(active_phase='train_epoch')
    audit = SimpleNamespace(task=1, epoch={'epoch': 6})
    try:
        raise RuntimeError(secret)
    except RuntimeError as error:
        context = worker.error_context(error, recorder, audit)
    assert context['task'] == 2 and context['epoch'] == 6 and context['phase'] == 'train_epoch'
    assert context['frames'] and secret not in json.dumps(context)
    assert all(set(frame) == {'file', 'line', 'function'} for frame in context['frames'])


def test_last_optimizer_cache_proof_rejects_refresh_after_step(tmp_path):
    branches, _ = small_native(warmup=0)
    audit, optimizer, dataset, record = audit_for_test(branches, tmp_path, 0)
    audit.begin_epoch(0)
    images, labels, _ = next(iter(dataset.train_loader))
    audit.observed(images, labels)
    features = branches['query'](images)['pre_logits']
    output = branches['prompt'](images, task_id=0, learned_id=0, cls_features=features, train=True, epoch_info=0)
    loss = reference.SeenClassCriterion(10)(output['logits'], labels) - .01 * output['sim_loss']
    loss.backward(retain_graph=True)
    audit.epoch['steps'] = 78
    optimizer.step()
    proof = record['cache_audits'][-1]
    assert proof['kind'] == 'last_batch_cache_timing' and proof['pre_post_optimizer_bitexact']
    assert torch.equal(audit.last_batch_cache, audit.prompt.stored_rainbow_prompts[0])
    audit.prompt.stored_rainbow_prompts[0].add_(1)
    audit.epoch['steps'] = 79
    audit.epoch['samples'] = 5000
    audit.stream_ids = list(range(5000))
    audit.stream_labels = [0] * 5000
    dataset.train_loader.dataset.indexes = np.arange(5000)
    with pytest.raises(RuntimeError, match='last pre-optimizer'):
        audit.end_epoch()
    audit.close()


def test_native_actual_trainer_and_meta_wrappers_complete_cpu_synthetic_workload(monkeypatch, tmp_path):
    """Exercise instrumentation ordering, ordinary counts, routing and checkpoint probes."""
    import main
    from utils import training
    from datasets.utils.continual_dataset import ContinualDataset, MammothDatasetWrapper
    from scripts.cl_efficiency import Recorder
    from scripts.ease_reference_utils import dataset_fingerprint
    from models.rainbow_prompt import RainbowPrompt
    from utils.evaluate import evaluate
    monkeypatch.setenv('MAMMOTH_DEVICE', 'cpu')
    command = shared.native_command({'mode': 'short_validation'})
    monkeypatch.setattr('sys.argv', ['main.py', *command])
    args = main.parse_args(cmd=command)
    args.device, args.num_workers, args.nowand = 'cpu', 0, 1
    args.results_path = str(tmp_path / 'mammoth')
    class Raw(torch.utils.data.Dataset):
        def __init__(self, train):
            self.data = np.zeros((5000 if train else 1000, 8, 8, 3), dtype=np.uint8)
            self.targets = np.zeros(len(self.data), dtype=np.int64)
            self.transform = None
            self.train = train
        def __len__(self):
            return len(self.data)
        def __getitem__(self, index):
            image = torch.from_numpy(self.data[index].copy()).permute(2, 0, 1).float()
            return (image, int(self.targets[index]), image) if self.train else (image, int(self.targets[index]))
    class Synthetic(ContinualDataset):
        NAME, SETTING = 'synthetic-rainbow-worker', 'class-il'
        N_CLASSES, N_TASKS, N_CLASSES_PER_TASK, SIZE = 20, 2, 10, (8, 8)
        def __init__(self):
            self.args, self.train_loader, self.test_loaders, self.c_task = args, None, [], -1
            self.eval_fn = evaluate
        def get_data_loaders(self):
            self.train_loader = DataLoader(MammothDatasetWrapper(Raw(True), train=True), batch_size=64, shuffle=True)
            test = DataLoader(MammothDatasetWrapper(Raw(False)), batch_size=64)
            self.test_loaders.append(test)
            self.c_task = 0
            return self.train_loader, test
        def get_loss(self):
            return nn.CrossEntropyLoss()
        def log(self, *arguments, **keywords):
            return None
    dataset = Synthetic()
    branches, net = small_native(warmup=0)
    net.pretrained_checkpoint = '/tmp/synthetic-only.npz'
    model = RainbowPrompt(net, nn.CrossEntropyLoss(), args, None, dataset)
    recorder = Recorder(tmp_path, gpu='cpu')
    recorder.torch, recorder.device = torch, torch.device('cpu')
    p = {'prompt_layers': [0, 1], 'n_epochs': 1, 'warmup_epochs': 0, 'tasks': 1}
    record = {'protocol': p, 'side': 'native', 'mode': 'short_validation', 'pass': 'validation', 'wandb_url': 'unit',
        'counts': {'observe_calls': 0, 'optimizer_updates': 0, 'task_begin_extra_samples': 0, 'task_end_extra_samples': 0},
        'first_batch_audits': [], 'cache_audits': [], 'state_audits': [], 'epoch_results': [], 'prediction_files': [],
        'task_results': [], 'routing_audits': [], 'sample_stream_audits': [], 'structural_checks': {key: False for key in shared.REQUIRED_CHECKS},
        'common_reference_data_snapshots': {}, 'common_transform_snapshots': {}}
    for split, train in (('train', True), ('test', False)):
        raw = Raw(train)
        proof = dataset_fingerprint(raw.data, raw.targets, np.arange(len(raw)), raw.transform)
        record['common_reference_data_snapshots']['task1_' + split] = proof
        record['common_transform_snapshots']['task1_' + split] = proof
    monkeypatch.setattr(worker, 'canonical_hashes', lambda branches: {'query': shared.CANONICAL_PRETRAINED_SHA256,
                                                                    'prompt': shared.CANONICAL_PRETRAINED_SHA256})
    monkeypatch.setattr(worker, 'notify_slack', lambda *args, **kwargs: 'sent')
    audit = worker.RainbowAudit(branches, dataset, recorder, record, torch, np, model)
    with ExitStack() as stack:
        audit.install_native(stack, training)
        training.train(model, dataset, args)
    audit.close()
    assert record['counts']['observe_calls'] == record['counts']['optimizer_updates'] == 79
    assert recorder.training_examples == 5000 and recorder.evaluation_examples == 1000 and recorder.tasks_completed == 1
    assert model.current_task == 1 and model.task_iteration == 79
    assert record['task_results'][0]['seen_accuracy_pct'] == 100.
    assert record['first_batch_audits'][0]['seed'] == 2993
    assert record['state_audits'][0]['checkpoint_outputs_bitexact']
    assert record['state_audits'][0]['checkpoint_probe_examples'] == 40
    assert record['routing_audits'][0]['batches'][-1]['samples'] == 40
    assert (tmp_path / 'task1_state.pt').is_file()
    assert model.net.prompt.rainbow_prompt.stored_rainbow_prompts[0].abs().sum() > 0
    with np.load(tmp_path / 'task1_predictions.npz', allow_pickle=False) as archive:
        assert np.array_equal(archive['batch_sizes'], np.array([64] * 15 + [40]))
        assert archive['evaluation_global_ids'].shape == (1000,)
