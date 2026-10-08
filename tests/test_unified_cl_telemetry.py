import copy
from contextlib import contextmanager
import json
import time
from types import SimpleNamespace
import weakref
import gc

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.unified_cl_telemetry import (UnifiedTelemetry, persistent_state_snapshot,
                                         registered_state_digest, phase_cost, union_intervals)


class FakeRecorder:
    def __init__(self):
        self.started = time.perf_counter()
        self.samples, self.phases = [], []
        self.snapshots, self.data_snapshots = {}, {}
        self.active_phase, self.interval = 'framework', 1.
        self.training_examples = self.evaluation_examples = self.steps = 0

    @contextmanager
    def phase(self, name):
        before, self.active_phase = self.active_phase, name
        start = time.perf_counter() - self.started
        try:
            yield
        finally:
            end = time.perf_counter() - self.started
            self.phases.append(dict(phase=name, start_s=start, end_s=end, wall_s=end-start))
            self.active_phase = before


class Rows(Dataset):
    def __init__(self, task, train=False):
        self.targets = np.arange(5, dtype=np.int64) % 3 + task * 10
        self.indexes = np.arange(5, dtype=np.int64) + task * 5
        self.extra_return_fields = ['original_ids']
        self.transform = None
        self.train = train

    def __len__(self):
        return 5

    def __getitem__(self, index):
        image = torch.tensor([index / 5, .25, .5])
        row = (image, int(self.targets[index]))
        return (*row, image.clone(), int(self.indexes[index])) if self.train else (*row, int(self.indexes[index]))


class FakeDataset:
    def __init__(self):
        self.test_loaders = []

    def get_data_loaders(self):
        task = len(self.test_loaders)
        self.train_loader = DataLoader(Rows(task, True), batch_size=2)
        test = DataLoader(Rows(task), batch_size=2)
        self.test_loaders.append(test)
        return self.train_loader, test

    def get_offsets(self):
        return 0, len(self.test_loaders) * 10

    def evaluate(self, model, dataset):
        results = []
        with torch.no_grad():
            for task, loader in enumerate(dataset.test_loaders):
                correct, total = 0, 0
                for image, label, _ in loader:
                    output = model(image)
                    pred = output[:, :dataset.get_offsets()[1]].argmax(1)
                    correct += (pred == label).sum().item()
                    total += len(label)
                    # The native evaluator mutates logits AFTER class-IL pred.
                    output[:, :task*10] = -float('inf')
                results.append(100*correct/total)
        return results, list(results)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(3, 20)
        self.opt = torch.optim.SGD(self.net.parameters(), lr=.03)
        self.current_task = 0

    def forward(self, inputs):
        return self.net(inputs)

    def meta_begin_task(self, dataset):
        self.opt = torch.optim.SGD(self.net.parameters(), lr=.03)

    def meta_begin_epoch(self, epoch, dataset):
        pass

    def meta_end_epoch(self, epoch, dataset):
        pass

    def meta_observe(self, inputs, labels, not_aug_inputs, **kwargs):
        self.opt.zero_grad()
        output = self.net(inputs)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        self.opt.step()
        if self.current_task:
            self.opt.zero_grad()
            torch.nn.functional.cross_entropy(self.net(inputs), labels).backward()
            self.opt.step()
        return loss.item()

    def meta_end_task(self, dataset):
        if self.current_task:
            # An internal local optimizer, not model.opt, must be observed.
            alignment = torch.optim.SGD(self.net.parameters(), lr=.01)
            alignment.zero_grad()
            self.net(torch.ones(2, 3)).sum().backward()
            alignment.step()
        self.current_task += 1

    def meta_end_eval(self, dataset, accs):
        pass


def train_single_epoch(model, loader, args=None, epoch=0):
    for image, label, raw, ids in loader:
        model.meta_observe(image, label, raw, original_ids=ids)


def run_two_tasks(model, dataset, training):
    for _ in range(2):
        train, _ = dataset.get_data_loaders()
        model.meta_begin_task(dataset)
        for epoch in range(2):
            model.meta_begin_epoch(epoch, dataset)
            training.train_single_epoch(model, train, epoch=epoch)
            model.meta_end_epoch(epoch, dataset)
        model.meta_end_task(dataset)
        accs = dataset.evaluate(model, dataset)
        model.meta_end_eval(dataset, accs)


def test_task_preservation_actual_steps_predictions_and_no_sensor_work():
    torch.set_num_threads(2)
    model, dataset, recorder = FakeModel(), FakeDataset(), FakeRecorder()
    training = SimpleNamespace(train_single_epoch=train_single_epoch)
    observer = UnifiedTelemetry(recorder, model, dataset, training)
    with observer.context():
        run_two_tasks(model, dataset, training)
    record = observer.finalize()
    first, second = record['tasks']['task1'], record['tasks']['task2']
    assert set(record['tasks']) == {'task1', 'task2'}
    assert first['counters']['outer_observe_calls'] == second['counters']['outer_observe_calls'] == 6
    assert first['counters']['outer_current_image_visits'] == second['counters']['outer_current_image_visits'] == 10
    assert first['counters']['optimizer_steps'] == 6
    assert second['counters']['optimizer_steps'] == 13
    assert second['optimizer_events']['end_task']['completed'] == 1
    assert second['optimizer_events']['train_epoch']['completed'] == 12
    assert first['counters']['feature_image_visits'] is None
    assert second['counters']['replay_image_visits'] is None
    assert first['model_snapshots']['after_end_task']['registered_state_sha256'] != second['model_snapshots']['after_end_task']['registered_state_sha256']
    assert first['model_snapshots']['after_end_task']['parameters_receiving_gradient_so_far'] == 80
    assert second['parameters_stepped_with_gradient'] == 80
    assert first['data_snapshots']['loaders']['train']['classes'] == [0, 1, 2]
    assert second['data_snapshots']['loaders']['train']['classes'] == [10, 11, 12]
    assert set(recorder.data_snapshots) == {'task1/loaders', 'task2/loaders'}
    assert len(first['evaluations'][0]['predictions']['0']['labels']) == 5
    assert second['evaluations'][0]['predictions']['1']['original_ids'] == list(range(5, 10))
    assert second['evaluations'][0]['metrics']['seen_accuracy_pct'] == sum(second['evaluations'][0]['accuracy'][0])/2
    assert first['evaluations'][0]['metrics']['old_accuracy_pct'] is None
    assert record['cost']['gpu_board_energy_j'] is None
    assert record['cost']['gpu_energy_coverage_s'] == 0
    json.dumps(record, allow_nan=False)


def test_observation_preserves_algorithm_state_rng_and_native_methods():
    model = FakeModel()
    reference = copy.deepcopy(model)
    torch.default_generator.manual_seed(1993)
    training = SimpleNamespace(train_single_epoch=train_single_epoch)
    run_two_tasks(reference, FakeDataset(), training)
    rng = torch.default_generator.get_state().clone()
    torch.default_generator.manual_seed(1993)
    dataset = FakeDataset()
    observer = UnifiedTelemetry(FakeRecorder(), model, dataset, training)
    with observer.context():
        run_two_tasks(model, dataset, training)
    assert torch.equal(torch.default_generator.get_state(), rng)
    assert registered_state_digest(model) == registered_state_digest(reference)
    assert training.train_single_epoch is train_single_epoch
    assert 'meta_observe' not in vars(model)
    assert 'evaluate' not in vars(dataset)


def test_context_exception_restores_methods_hooks_and_does_not_retain_old_optimizer():
    model, dataset = FakeModel(), FakeDataset()
    training = SimpleNamespace(train_single_epoch=train_single_epoch)
    observer = UnifiedTelemetry(FakeRecorder(), model, dataset, training)
    previous = weakref.ref(model.opt)
    with pytest.raises(RuntimeError, match='deliberate'):
        with observer.context():
            dataset.get_data_loaders()
            model.meta_begin_task(dataset)
            gc.collect()
            assert previous() is None
            raise RuntimeError('deliberate')
    assert 'meta_begin_task' not in vars(model)
    assert not model.net.weight._backward_hooks
    assert not observer._active


def test_parameter_hooks_do_not_keep_replaced_prompt_parameter_alive():
    model, dataset = FakeModel(), FakeDataset()
    model.prompt = torch.nn.Parameter(torch.ones(3))
    observer = UnifiedTelemetry(FakeRecorder(), model, dataset, SimpleNamespace(train_single_epoch=train_single_epoch))
    with observer.context():
        previous = weakref.ref(model.prompt)
        model.prompt = torch.nn.Parameter(torch.ones(3))
        # Immediate refcount release, without relying on the GC scheduling.
        assert previous() is None
        observer.snapshot()


def test_gradient_hook_identity_handles_id_reuse_and_context_reentry():
    model, dataset = FakeModel(), FakeDataset()
    observer = UnifiedTelemetry(FakeRecorder(), model, dataset, SimpleNamespace(train_single_epoch=train_single_epoch))
    with observer.context():
        new_parameter = torch.nn.Parameter(torch.ones(3))
        model.prompt = new_parameter
        foreign = torch.nn.Parameter(torch.zeros(3))
        # Simulate a reused Python id entry from a different Parameter object.
        observer._gradient_hook_ids[id(new_parameter)] = foreign
        observer.snapshot()
        assert len(new_parameter._backward_hooks) == 1
        assert observer._gradient_hook_ids[id(new_parameter)] is new_parameter
    assert not model.prompt._backward_hooks
    with observer.context():
        assert len(model.prompt._backward_hooks) == 1
    assert not model.prompt._backward_hooks


def test_storage_dedup_unregistered_numpy_teacher_fisher_optimizer_and_gradients():
    model = torch.nn.Linear(2, 2)
    model.fisher = torch.zeros(6)
    model.fisher_view = model.fisher[:3]
    model.numpy_shared = model.fisher.numpy()
    model.cache = np.zeros(10, dtype=np.float32)
    model.cache_view = model.cache[2:5]
    model.old_teacher = [copy.deepcopy(model)]
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    snapshot = persistent_state_snapshot(model, optimizers=[optimizer])
    # NumPy deepcopy creates separate teacher Fisher-array and cache-view
    # allocations, while Tensor deepcopy preserves Fisher/view shared storage.
    # These extra allocations must be counted, rather than grouped by shape.
    assert snapshot['persistent_storage_bytes'] == (6+6+6+10+6+10+6+3+12+2)*4
    assert snapshot['additional_gradient_storage_bytes'] == 6*4
    assert snapshot['persistent_storage_bytes'] == sum(snapshot['persistent_bytes_by_category'].values())
    assert snapshot['unsupported_storage_types'] == []


def test_storage_zero_scalar_and_disjoint_inclusive_phase_cost():
    model = torch.nn.Module()
    model.register_buffer('scalar', torch.tensor(3.))
    model.register_buffer('empty', torch.empty(0))
    assert persistent_state_snapshot(model)['persistent_storage_bytes'] == 4
    assert len(registered_state_digest(model)) == 64
    phases = [dict(start_s=0., end_s=3.), dict(start_s=1., end_s=2.), dict(start_s=2., end_s=5.)]
    samples = [dict(elapsed_s=0., gpu_power_w=10.), dict(elapsed_s=5., gpu_power_w=10.)]
    assert phase_cost(phases, samples)['wall_s'] == 5.
    assert phase_cost(phases, samples)['gpu_board_energy_j'] == 50.
    assert union_intervals([(0, 1), (1, 2), (4, 5)]) == [[0., 2.], [4., 5.]]
    with pytest.raises(ValueError):
        union_intervals([(0, float('nan'))])


def test_explicit_replay_and_module_counters_zero_available_new_task():
    class Buffer:
        def get_data(self, count, transform=None, return_index=False):
            result = (torch.ones(count, 3), torch.zeros(count, dtype=torch.long))
            return (torch.arange(count), *result) if return_index else result
    model, dataset, recorder = FakeModel(), FakeDataset(), FakeRecorder()
    training = SimpleNamespace(train_single_epoch=train_single_epoch)
    observer, buffer = UnifiedTelemetry(recorder, model, dataset, training), Buffer()
    with observer.context():
        observer.watch_replay_buffer(buffer)
        observer.watch_module(model.net, counter='feature_vector_visits', role='test-feature-head')
        dataset.get_data_loaders()
        model.meta_begin_task(dataset)
        buffer.get_data(3)
        buffer.get_data(2, None, True)
        model(torch.ones(4, 3))
        model.meta_end_task(dataset)
        dataset.get_data_loaders()
        model.meta_begin_task(dataset)
    assert observer.task_record(0)['counters']['replay_image_visits'] == 5
    assert observer.task_record(0)['counters']['feature_vector_visits'] == 4
    assert observer.task_record(1)['counters']['replay_image_visits'] == 0
    assert observer.task_record(1)['counters']['feature_vector_visits'] == 0
    assert 'get_data' not in vars(buffer)
    with pytest.raises(ValueError):
        observer.add_counter('unknown', 3, source='bad')


def test_optimizer_membership_gradient_receipt_and_step_participation_distinct():
    class ExtraGradient(FakeModel):
        def __init__(self):
            super().__init__()
            self.unoptimized = torch.nn.Parameter(torch.ones(2))
        def meta_begin_task(self, dataset):
            self.opt = torch.optim.SGD(self.net.parameters(), lr=.03)
        def meta_observe(self, inputs, labels, not_aug_inputs, **kwargs):
            self.opt.zero_grad()
            (self.net(inputs).sum() + self.unoptimized.sum()).backward()
            self.opt.step()
    model, dataset = ExtraGradient(), FakeDataset()
    observer = UnifiedTelemetry(FakeRecorder(), model, dataset, SimpleNamespace(train_single_epoch=train_single_epoch))
    with observer.context():
        dataset.get_data_loaders()
        model.meta_begin_task(dataset)
        model.meta_observe(torch.ones(2, 3), torch.zeros(2, dtype=torch.long), torch.ones(2, 3))
        snapshot = observer.snapshot()
    assert snapshot['parameters_requires_grad'] == 82
    assert snapshot['parameters_in_observed_optimizers'] == 80
    assert snapshot['parameters_receiving_gradient_so_far'] == 82
    assert snapshot['parameters_stepped_with_gradient_so_far'] == 80


def test_failed_optimizer_step_is_attempt_not_success_and_unrelated_steps_ignored():
    class Fails(torch.optim.SGD):
        def step(self, closure=None):
            raise RuntimeError('step failed')
    model, dataset = FakeModel(), FakeDataset()
    observer = UnifiedTelemetry(FakeRecorder(), model, dataset, SimpleNamespace(train_single_epoch=train_single_epoch))
    with observer.context():
        dataset.get_data_loaders()
        model.meta_begin_task(dataset)
        optimizer = Fails(model.net.parameters(), lr=.1)
        model.net(torch.ones(1, 3)).sum().backward()
        with pytest.raises(RuntimeError, match='step failed'):
            optimizer.step()
        foreign = torch.nn.Parameter(torch.ones(2))
        external_optimizer = torch.optim.SGD([foreign], lr=.1)
        foreign.sum().backward()
        external_optimizer.step()
        row = observer.task_record()
        assert row['counters']['optimizer_step_attempts'] == 1
        assert row['counters']['optimizer_steps'] == 0
        assert row['stepped_gradient_parameters'] == {}


def test_predictions_missing_original_ids_remain_unavailable_and_partial_retained():
    class UnknownIds(Dataset):
        def __len__(self):
            return 3
        def __getitem__(self, index):
            return torch.ones(3), index
    model, dataset = FakeModel(), FakeDataset()
    observer = UnifiedTelemetry(FakeRecorder(), model, dataset, SimpleNamespace(train_single_epoch=train_single_epoch))
    with observer.context():
        dataset.get_data_loaders()
        model.meta_begin_task(dataset)
        dataset.test_loaders = [DataLoader(UnknownIds(), batch_size=2)]
        # This alternative ordinary evaluator does not require ID tuple fields.
        def ordinary(model, dataset):
            for batch in dataset.test_loaders[0]:
                model(batch[0])
            raise RuntimeError('evaluation failed')
        # Use the same decorator as the native instance context would install.
        with pytest.raises(RuntimeError, match='evaluation failed'):
            observer._evaluated(ordinary)(model, dataset)
        row = observer.task_record()['evaluations'][0]
        assert row['complete'] is False
        assert row['predictions']['0']['original_ids'] is None
        assert len(row['predictions']['0']['predictions']) == 3
        assert type(dataset.test_loaders[0]) is DataLoader
