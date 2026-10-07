"""Completion audits for the explicitly approved three-method cost pilot.

These hooks observe the native Mammoth wrappers and evaluation, without extra
image forwards or replacing any method's training or task-end algorithm.
"""

from contextlib import ExitStack, contextmanager
import functools
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('ranpac', 'l2p', 'dualprompt')
PTM_SHA = 'cd8acbba3d33807125c2e30d2d0159f90825208e63a95034ab57ef0af837a300'
CHECKS = ('finite_loss_and_gradients', 'frozen_backbone_gradients_absent',
          'frozen_backbone_unchanged', 'pretrained_backbones_match_checkpoint',
          'training_protocol_complete', 'evaluation_predictions_match_accuracy',
          'dataset_matches_reference', 'minimum_accuracy_not_collapsed',
          'method_specific_final_state')


def execution_pin(spec):
    from scripts.check_ease_runtime import file_hash
    plan_path = ROOT / 'benchmarks/cl_efficiency/baselines/validation_plan.json'
    plan = json.loads(plan_path.read_text())
    expected = {'tasks': 1, 'n_epochs': 1, 'batch_size': 64, 'num_workers': 4,
                'seed': 1993, 'measurement_repeats': 3, 'flops_runs': 1}
    if (plan.get('training_authorized') is not True or tuple(plan['models']) != MODELS
            or spec['model'] not in MODELS or spec['gpu'] != '1'
            or any(type(plan['efficiency'].get(k)) is not type(v) or plan['efficiency'].get(k) != v
                   for k, v in expected.items())):
        raise ValueError('Baseline pilot is outside the authorized three-method scope')
    checkpoint = Path(plan['checkpoint']['file']).resolve()
    if not checkpoint.is_file() or file_hash(checkpoint) != plan['checkpoint']['sha256']:
        raise ValueError('Baseline pretrained checkpoint changed')
    ease = json.loads((ROOT / 'benchmarks/cl_efficiency/ease/validation_plan.json').read_text())['reference']
    if (str(checkpoint) != str(Path(ease['checkpoint_file']).resolve())
            or plan['checkpoint']['sha256'] != ease['checkpoint_sha256']):
        raise ValueError('Baseline checkpoint must be the exact accepted EASE file')
    reference = json.loads((ROOT / 'benchmarks/cl_efficiency/ease/execution_results.json').read_text())
    reference = reference['official_reference']['data_snapshots']
    files = ['scripts/benchmark_cl.py', 'scripts/baseline_audit.py', 'scripts/run_baseline_efficiency.py',
             'scripts/cl_efficiency.py', 'scripts/cl_runtime.py', 'main.py', 'utils/training.py',
             'backbone/vit.py', 'datasets/configs/seq-cifar100-224/ease.yaml', 'uv.lock']
    for method in MODELS:
        files.append(f'models/{method}.py')
        files.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / f'models/{method}_utils').glob('*.py')))
    return {'checkpoint_file': str(checkpoint), 'checkpoint_sha256': plan['checkpoint']['sha256'],
            'checkpoint_tag': plan['checkpoint']['tag'], 'validation_plan_sha256': file_hash(plan_path),
            'execution_source_sha256': {name: file_hash(ROOT / name) for name in files},
            'reference_data_snapshots': {name: reference[name] for name in ('task1_train', 'task1_test')},
            'minimum_accuracy_pct': 20.0,
            'accuracy_policy': 'Conservative gross-collapse guard (>20% on balanced ten classes); not paper reproduction or the twenty-epoch 1.5/3pp acceptance rule.'}


def digest(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        cpu = tensor.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(tuple(cpu.shape)).encode())
        result.update(str(cpu.dtype).encode())
        result.update(cpu.numpy().tobytes())
    return result.hexdigest()


def canonical_state(module, method):
    """Normalize RanPAC's split QKV/MLP and L2P's extra position prefixes on CPU."""
    import torch
    raw = module.state_dict()
    state = {}
    for name, tensor in raw.items():
        if name.startswith('head.') or 'prompt' in name or 'adaptmlp' in name:
            continue
        if method == 'ranpac':
            if any(f'.attn.{part}_proj.' in name for part in ('q', 'k', 'v')):
                continue
            name = name.replace('.fc1.', '.mlp.fc1.').replace('.fc2.', '.mlp.fc2.')
        if name == 'pos_embed' and tensor.shape[1] != 197:
            # L2P inserts extra class-position copies before the patch positions.
            tensor = torch.cat((tensor[:, :1].cpu(), tensor[:, -196:].cpu()), dim=1)
        state[name] = tensor
    if method == 'ranpac':
        for index in range(12):
            for kind in ('weight', 'bias'):
                prefix = f'blocks.{index}.attn.'
                state[prefix + 'qkv.' + kind] = torch.cat(
                    [raw[prefix + part + '_proj.' + kind].detach().cpu() for part in ('q', 'k', 'v')], dim=0)
    if len(state) != 150:
        raise RuntimeError('Pilot backbone must contain all 150 canonical pretrained tensors')
    return state


def require_initialization_args(args, metadata):
    """Check parsed arguments before any backbone can load default weights."""
    expected = {'n_epochs': 1, 'batch_size': 64, 'num_workers': 4, 'seed': 1993,
                'dataset_config': 'ease', 'code_optimization': 0, 'drop_last': False,
                'fitting_mode': 'epochs', 'permute_classes': False, 'joint': False}
    checkpoint = getattr(args, 'pretrained_checkpoint', None)
    if (any(getattr(args, k, None) != v for k, v in expected.items())
            or getattr(args, 'pretrained', None) is not False
            or args.eval_epochs is not None or args.loadcheck is not None or args.start_from is not None
            or not checkpoint or str(Path(checkpoint).resolve()) != metadata['checkpoint_file']):
        raise ValueError('Effective baseline arguments changed the shared pilot protocol')


class BaselineAudit:
    def __init__(self, model, dataset, args, recorder, metadata, torch):
        require_initialization_args(args, metadata)
        self.model, self.dataset, self.recorder, self.metadata, self.torch = model, dataset, recorder, metadata, torch
        self.method = model.NAME
        self.roles = ({'adapter_backbone': model.net._network.convnet} if self.method == 'ranpac'
                      else {'query': model.net.original_model, 'prompted': model.net.model})
        self.checks = metadata['structural_checks'] = dict.fromkeys(CHECKS, False)
        self.losses, self.predictions, self.epochs, self.prototype_examples = [], [], 0, 0
        self.initial = {}
        with recorder.phase('state_audit'):
            canonical = {name: digest(canonical_state(module, self.method)) for name, module in self.roles.items()}
            if any(value != PTM_SHA for value in canonical.values()):
                raise RuntimeError('Baseline loaded pretrained tensors differ from the accepted EASE weights')
            for name, module in self.roles.items():
                for key, parameter in module.named_parameters():
                    if not self.adaptation_parameter(key) and (parameter.requires_grad or parameter.dtype != torch.float32):
                        raise RuntimeError('Pretrained baseline backbone must be frozen FP32')
                self.initial[name] = digest(self.frozen_state(module))
        metadata['canonical_pretrained_sha256'] = canonical
        metadata['initial_frozen_backbone_sha256'] = self.initial
        self.checks['pretrained_backbones_match_checkpoint'] = True

    @staticmethod
    def adaptation_parameter(name):
        return name.startswith('head.') or 'prompt' in name or 'adaptmlp' in name

    def frozen_state(self, module):
        return {name: value for name, value in module.state_dict().items() if not self.adaptation_parameter(name)}

    def end_epoch(self):
        with self.recorder.phase('state_audit'):
            if any(parameter.grad is not None for module in self.roles.values()
                   for name, parameter in module.named_parameters() if not self.adaptation_parameter(name)):
                raise RuntimeError('Frozen baseline backbone received gradients')
            for parameter in self.model.parameters():
                if parameter.requires_grad and (not self.torch.isfinite(parameter).all().item()
                        or parameter.grad is not None and not self.torch.isfinite(parameter.grad).all().item()):
                    raise RuntimeError('Baseline adaptation has non-finite parameters or gradients')
        self.epochs += 1
        self.checks.update(finite_loss_and_gradients=bool(self.losses), frozen_backbone_gradients_absent=True)

    def end_task(self):
        from scripts.ease_reference_utils import dataset_fingerprint
        with self.recorder.phase('state_audit'):
            final = {name: digest(self.frozen_state(module)) for name, module in self.roles.items()}
            if final != self.initial:
                raise RuntimeError('Frozen baseline backbone changed')
            for split, loader in (('train', self.dataset.train_loader), ('test', self.dataset.test_loaders[0])):
                source = loader.dataset
                actual = dataset_fingerprint(source.data, source.targets, source.indexes, source.transform)
                self.recorder.data_snapshots['first_task'][split].update(actual)
                reference = self.metadata['reference_data_snapshots'][f'task1_{split}']
                keys = ('n_samples', 'class_counts', 'samples_ordered_sha256',
                        'samples_unordered_sha256', 'global_ids_ordered_sha256')
                if any(actual[key] != reference[key] for key in keys):
                    raise RuntimeError('Baseline raw samples differ from EASE first task')
            if self.method == 'ranpac':
                net, width = self.model.net._network, self.model.args.rp_size
                expected = {'G': (width, width), 'Q': (width, 100), 'W_rand': (768, width)}
                for name, shape in expected.items():
                    value = getattr(self.model, name)
                    if tuple(value.shape) != shape or not self.torch.isfinite(value).all().item():
                        raise RuntimeError('RanPAC RP/ridge state is incomplete or non-finite')
                    if value.dtype != self.torch.float32 or name in ('G', 'Q') and value.device.type != 'cpu':
                        raise RuntimeError('RanPAC CPU ridge state dtype/device changed')
                if (net.fc.W_rand.device != self.model.W_rand.device
                        or net.fc.W_rand.dtype != self.torch.float32
                        or net.fc.W_rand.untyped_storage().data_ptr() != self.model.W_rand.untyped_storage().data_ptr()):
                    raise RuntimeError('RanPAC classifier lost its registered shared projection')
                if (tuple(net.fc.weight.shape) != (10, width) or not net.fc.use_RP
                        or not self.torch.isfinite(net.fc.weight).all().item()
                        or self.prototype_examples != 5000):
                    raise RuntimeError('RanPAC task-end classifier completion failed')
                self.metadata['task_end_training_image_presentations'] = self.prototype_examples
                self.metadata['ridge_candidate_count'] = 17
                self.metadata['ridge_solve_count'] = 18
                self.metadata['rp_size'] = width
                from scripts.cl_efficiency import tensor_bytes
                self.metadata['method_state_storage'] = {
                    'cpu_gram_G_bytes': tensor_bytes(self.model.G),
                    'cpu_cross_product_Q_bytes': tensor_bytes(self.model.Q),
                    'cpu_GQ_total_bytes': tensor_bytes((self.model.G, self.model.Q)),
                    'registered_projection_W_rand_bytes': tensor_bytes(self.model.W_rand),
                    'GQ_device': 'cpu', 'GQ_dtype': 'float32',
                    'projection_device': str(self.model.W_rand.device), 'projection_shared_storage': True,
                    'categories_overlap': 'G/Q are included in generic auxiliary bytes; W_rand may overlap registered/auxiliary categories. Do not add categories.',
                    'solve_scope': 'Seventeen validation candidates plus one final CPU solve, unchanged rp_size=10000. Linalg may be absent from partial FLOPs; GPU energy excludes CPU power.'}
            else:
                net = self.model.net
                if net.model.head.out_features != 100 or any(p.requires_grad for p in net.original_model.parameters()):
                    raise RuntimeError('Prompt/query classifier or freeze contract changed')
        self.metadata['final_frozen_backbone_sha256'] = final
        self.checks.update(frozen_backbone_unchanged=True, dataset_matches_reference=True,
                           method_specific_final_state=True)

    def complete(self):
        import numpy as np
        from torch.utils.data import SequentialSampler
        with self.recorder.phase('state_audit'):
            loader = self.dataset.test_loaders[0]
            if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
                raise RuntimeError('Saved baseline predictions require sequential evaluation')
            labels, ids = np.asarray(loader.dataset.targets), np.asarray(loader.dataset.indexes)
            predicted = np.concatenate(self.predictions) if self.predictions else np.empty(0, dtype=np.int64)
            if len(predicted) != len(labels) or len(labels) != 1000:
                raise RuntimeError('Baseline evaluation is incomplete')
            accuracy = float((predicted == labels).mean() * 100)
            if abs(accuracy - self.metadata['accuracy'].get('mean_accuracy_pct', -1)) > 1e-9:
                raise RuntimeError('Saved baseline predictions disagree with native class-IL accuracy')
            if accuracy <= self.metadata['minimum_accuracy_pct'] or np.unique(predicted).size < 2:
                raise RuntimeError('Baseline accuracy collapsed; investigate implementation before continuing')
            np.savez(self.recorder.output / 'task1_predictions.npz', predictions=predicted[:, None], labels=labels, global_ids=ids)
        self.checks.update(evaluation_predictions_match_accuracy=True, minimum_accuracy_not_collapsed=True)
        valid = (self.epochs == 1 and len(self.losses) == 79 and self.recorder.steps == 79
                 and self.recorder.training_examples == 5000 and self.recorder.evaluation_examples == 1000
                 and self.recorder.tasks_completed == 1
                 and all(v for k, v in self.checks.items() if k != 'training_protocol_complete'))
        if not valid:
            raise RuntimeError('Baseline did not complete the full first-task pilot')
        self.checks['training_protocol_complete'] = True
        self.metadata['training_loss'] = {'batches': len(self.losses), 'mean': sum(self.losses) / len(self.losses),
                                          'minimum': min(self.losses), 'maximum': max(self.losses)}
        self.metadata['torch_cpu_threads'] = self.torch.get_num_threads()


@contextmanager
def instrument_baseline(audit):
    from unittest.mock import patch
    model, recorder = audit.model, audit.recorder

    def observed(original):
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            if recorder.active_phase == 'train_epoch':
                loss = float(result['loss'] if isinstance(result, dict) else result)
                if not math.isfinite(loss):
                    raise RuntimeError('Baseline loss is non-finite')
                audit.losses.append(loss)
            return result
        return wrapper

    def audited(original, callback):
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            callback()
            return result
        return wrapper

    original_forward = model.forward

    @functools.wraps(original_forward)
    def forwarded(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        if recorder.active_phase == 'evaluation':
            if not audit.torch.isfinite(result).all().item():
                raise RuntimeError('Baseline evaluation logits are non-finite')
            audit.predictions.append(result.detach().argmax(1).cpu().numpy())
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(model, 'meta_observe', observed(model.meta_observe)))
        stack.enter_context(patch.object(model, 'meta_end_epoch', audited(model.meta_end_epoch, audit.end_epoch)))
        stack.enter_context(patch.object(model, 'meta_end_task', audited(model.meta_end_task, audit.end_task)))
        stack.enter_context(patch.object(model, 'forward', forwarded))
        if audit.method == 'ranpac':
            def count_features(module, inputs, result):
                if recorder.active_phase == 'end_task':
                    audit.prototype_examples += len(inputs[0])
            handle = model.net._network.convnet.register_forward_hook(count_features)
            stack.callback(handle.remove)
        yield
