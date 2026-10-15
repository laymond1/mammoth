"""Independent author size variants for the fresh two-task LoRA comparison.

The reviewed checkout is never edited. Only import namespaces, factory geometry,
classifier input dimensions, local NPZ loading and runtime compatibility are
adapted in memory. Source losses, forwards, freezing and optimizer loops execute
from that checkout. Small/Tiny are size variants, not published reproductions.
The CLI defaults to a stdlib-only plan; training is owned by the serial launcher.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import importlib
import importlib.abc
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = Path('/tmp/cl-low-rank-official-20261008')
METHODS = {'ewc-lora': 'ewclora', 'sd-lora': 'sdlora', 'inf-lora': 'inflora', 'cl-lora': 'cllora'}
_COUNTER = itertools.count()


def reference_spec(method, vit_size='tiny', batch_size=16):
    """Return an approved cell without constructing models or reading secrets."""
    from scripts.unified_cl_protocol import approved_protocol, build_run_specs
    if method not in METHODS:
        raise ValueError('Independent LoRA reference requires one of the four approved methods')
    candidates = build_run_specs(approved_protocol(), modes=('validation',))
    for spec in candidates:
        if (spec['method'], spec['vit_size'], spec['batch_size']) == (method, vit_size, batch_size):
            return spec
    raise ValueError('Independent LoRA reference requires a supported size and batch16/64')


def reference_description(spec):
    from scripts.low_rank_protocol import REFERENCE_COMMIT
    from scripts.unified_cl_protocol import validate_run_spec
    spec = validate_run_spec(spec)
    if spec['method'] not in METHODS or spec['mode'] != 'validation' or spec['n_epochs'] != 2:
        raise ValueError('Independent LoRA reference supports the approved two-epoch validation cells')
    return {'implementation': 'pinned_author_size_variant_common_input_reference',
            'reference_commit': REFERENCE_COMMIT, 'method': spec['method'],
            'vit_size': spec['vit_size'], 'feature_dim': spec['feature_dim'], 'num_heads': spec['num_heads'],
            'tasks': 2, 'dataset_total_tasks': 10, 'n_epochs': 2, 'batch_size': spec['batch_size'],
            'checkpoint': deepcopy(spec['checkpoint']), 'recipe': deepcopy(spec['method_recipe']),
            'normalization_mean': [0.5] * 3, 'normalization_std': [0.5] * 3,
            'paper_reproduction': False, 'resource_statistics_eligible': False,
            'adaptations': ['private imports; no native package replacement',
                            'in-memory Net constructor geometry and classifier input dimensions',
                            'disable unused timm registration decorators in isolated source imports',
                            'strict local canonical150 encoder loading',
                            'Inf NumPy covariance boundary', 'CL saved-tensor mutation compatibility',
                            'caller-supplied common inputs and phase RNG; unchanged source training loop']}


def common_transforms():
    """Explicit shared mean/std.5 pipeline; no dataset creation or downloads."""
    from torchvision import transforms
    interpolation = transforms.InterpolationMode.BICUBIC
    normalize = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    training = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.), ratio=(0.75, 4 / 3), interpolation=interpolation, antialias=True),
        transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), normalize])
    testing = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation, antialias=True), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize])
    return training, testing


class _SourceAdapter(ast.NodeTransformer):
    def __init__(self, namespace, geometry, is_network):
        self.namespace, self.geometry, self.is_network = namespace, geometry, is_network
        self.geometry_patches = self.classifier_patches = self.registration_patches = 0
        self.in_net_init = False

    def visit_ImportFrom(self, node):
        if node.module and node.module.split('.')[0] in ('models', 'methods', 'utils', 'dataloaders'):
            node.module = self.namespace + '.' + node.module
        return node

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split('.')[0] in ('models', 'methods', 'utils', 'dataloaders'):
                # Pinned numerical dependencies use from-imports; fail on new
                # ambiguous top-level bindings rather than changing semantics.
                raise ValueError('Unexpected absolute source import requires review')
        return node

    def visit_ClassDef(self, node):
        if self.is_network and node.name == 'Net':
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == '__init__':
                    self.in_net_init = True
                    self.visit(child)
                    self.in_net_init = False
                else:
                    self.visit(child)
            return node
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        retained = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'register_model':
                self.registration_patches += 1
            else:
                retained.append(decorator)
        node.decorator_list = retained
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if (self.in_net_init and any(isinstance(target, ast.Name) and target.id == 'model_kwargs' for target in node.targets)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'dict'):
            existing = {keyword.arg: keyword for keyword in node.value.keywords}
            for key, value in self.geometry.items():
                if key in existing:
                    existing[key].value = ast.Constant(value)
                else:
                    node.value.keywords.append(ast.keyword(arg=key, value=ast.Constant(value)))
            self.geometry_patches += 1
        return self.generic_visit(node)

    def visit_Call(self, node):
        if (self.in_net_init and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'nn' and node.func.attr == 'Linear' and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == 768):
            node.args[0] = ast.Attribute(value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                                          attr='image_encoder', ctx=ast.Load()), attr='out_dim', ctx=ast.Load())
            self.classifier_patches += 1
        return self.generic_visit(node)


def _load_source(source, method, geometry):
    """Compile immutable numerical source into a private package namespace."""
    from scripts.low_rank_protocol import verify_reference_source
    pins = verify_reference_source(source)
    source = Path(source).resolve()
    namespace = '_unified_lora_author_' + str(next(_COUNTER))
    catalog = {}

    class Loader(importlib.abc.Loader):
        def __init__(self, path):
            self.path = path

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            relative = self.path.relative_to(source).as_posix()
            is_network = relative == f'models/net_{METHODS[method]}.py'
            tree = ast.parse(self.path.read_text(), filename=str(self.path))
            adapter = _SourceAdapter(namespace, geometry, is_network)
            tree = ast.fix_missing_locations(adapter.visit(tree))
            if is_network and (adapter.geometry_patches != 1 or adapter.classifier_patches != (0 if method == 'cl-lora' else 1)):
                raise ValueError('Pinned Net geometry/classifier structure requires review')
            catalog[relative] = {'source_sha256': pins.get(relative), 'geometry_patches': adapter.geometry_patches,
                                 'classifier_patches': adapter.classifier_patches, 'registration_patches': adapter.registration_patches}
            exec(compile(tree, str(self.path), 'exec'), module.__dict__)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if not fullname.startswith(namespace + '.'):
                return None
            relative = fullname[len(namespace) + 1:].replace('.', '/')
            candidate = source / (relative + '.py')
            if candidate.is_file():
                return importlib.util.spec_from_file_location(fullname, candidate, loader=Loader(candidate))
            return None

    for suffix in ('', '.models', '.methods', '.utils', '.dataloaders', '.models.modules'):
        name = namespace + suffix
        package = types.ModuleType(name)
        package.__path__ = [str(source / suffix.lstrip('.').replace('.', '/'))]
        sys.modules[name] = package
    finder = Finder()
    sys.meta_path.insert(0, finder)
    try:
        module = importlib.import_module(namespace + '.methods.' + METHODS[method])
        network = importlib.import_module(namespace + '.models.net_' + METHODS[method])
    finally:
        sys.meta_path.remove(finder)
    return module, network, {'reference_source_sha256': pins, 'loaded_source_adaptations': catalog, 'namespace': namespace}


def author_args(spec, source=DEFAULT_SOURCE, device='cpu'):
    """Map the fresh approved recipe onto the source's own argument names."""
    from scripts.cl_runtime import public_setting
    reference_description(spec)
    settings = json.loads((Path(source) / f'configs/{METHODS[spec["method"]]}/cifar100.json').read_text())
    recipe = spec['method_recipe']
    settings.update(seed=[spec['seed']], epochs=spec['n_epochs'], batch_size=spec['batch_size'], num_workers=spec['num_workers'],
                    device=[device], sessions=10, init_cls=10, increment=10, data_path=public_setting('data_root'),
                    optimizer=recipe['optimizer'], scheduler='cosine', lrate=recipe['lr'], weight_decay=recipe['optim_wd'],
                    fc_lrate=recipe['lora_fc_lr'], rank=recipe['lora_rank'])
    names = {'lora_gamma': 'gamma', 'lora_ewc_lambda': 'lambda', 'lora_lamb': 'lamb', 'lora_lame': 'lame',
             'lora_kd_ratio': 'kd_ratio', 'lora_temperature': 'temperature', 'lora_msa': 'msa',
             'lora_shared_pos': 'shared_pos', 'lora_specific_pos': 'specific_pos'}
    settings.update({destination: deepcopy(recipe[name]) for name, destination in names.items() if name in recipe})
    return settings


def build_reference(spec, source=DEFAULT_SOURCE, device='cpu', *, synthetic_geometry=None, synthetic_args=None):
    """Construct the author learner; synthetic overrides are CPU test fixtures.

    Production construction verifies the file, 150 canonical tensors and source
    pins. No inputs are loaded here, and no forward/optimizer step is performed.
    """
    from scripts.cl_runtime import configure_native_environment
    configure_native_environment()
    import torch
    from models.low_rank_utils.factory import checkpoint_spec
    from models.low_rank_utils.pretrained import canonical_pretrained_digest, load_pretrained_encoder

    description = reference_description(spec)
    synthetic = synthetic_geometry is not None
    if synthetic_args is not None and not synthetic:
        raise ValueError('Argument overrides are confined to explicit synthetic CPU fixtures')
    if synthetic and str(device) != 'cpu':
        raise ValueError('Synthetic source references are CPU-only')
    geometry = dict(patch_size=16, embed_dim=spec['feature_dim'], depth=12, num_heads=spec['num_heads'])
    geometry.update(synthetic_geometry or {})
    pin = None if synthetic else checkpoint_spec(spec['vit_size'], spec['checkpoint']['file'])
    source_module, network_module, evidence = _load_source(source, spec['method'], geometry)

    def local_encoder(variant, pretrained=False, **kwargs):
        kwargs['num_classes'] = 0 if spec['method'] == 'cl-lora' else 1000
        encoder = network_module.ViT(**kwargs)
        if pin is not None:
            load_pretrained_encoder(encoder, pin['file'])
        return encoder

    network_module._create_vision_transformer = local_encoder
    settings = author_args(spec, source, torch.device(device))
    settings.update(synthetic_args or {})
    name = {'ewc-lora': 'EWCLoRA', 'sd-lora': 'SDLoRA', 'inf-lora': 'InfLoRA', 'cl-lora': 'CLLoRA'}[spec['method']]
    learner = getattr(source_module, name)(settings)
    canonical = canonical_pretrained_digest(learner.network.image_encoder)
    if pin is not None and canonical != pin['canonical_sha256']:
        raise RuntimeError('Independent author size encoder differs from the pinned canonical150 digest')
    public_settings = {**settings, 'device': [str(value) for value in settings['device']]}
    return AuthorSession(learner, source_module, spec, {**description, **evidence,
                         'canonical_pretrained_sha256': canonical, 'synthetic_fixture': synthetic,
                         'geometry': geometry, 'effective_author_args': public_settings})


def initial_catalog(session):
    """Export only source learnables/unused heads, retaining pretrained pins."""
    from models.low_rank_utils.pretrained import learned_initial_state
    from scripts.low_rank_protocol import tensor_digest
    state = learned_initial_state(session.learner.network)
    return {'metadata': {name: deepcopy(session.evidence[name]) for name in
                         ('method', 'vit_size', 'reference_commit', 'checkpoint', 'geometry', 'synthetic_fixture')},
            'learned_state': state, 'initial_learned_sha256': tensor_digest(state)}


def restore_initial_catalog(network, catalog, session):
    """Use the same persisted initialization in either independent network."""
    from models.low_rank_utils.pretrained import canonical_pretrained_digest, restore_learned_initial_state
    from scripts.low_rank_protocol import tensor_digest
    expected = {name: deepcopy(session.evidence[name]) for name in
                ('method', 'vit_size', 'reference_commit', 'checkpoint', 'geometry', 'synthetic_fixture')}
    if catalog.get('metadata') != expected or tensor_digest(catalog['learned_state']) != catalog.get('initial_learned_sha256'):
        raise ValueError('Independent LoRA learned initialization identity/hash differs')
    if canonical_pretrained_digest(network) != session.evidence['canonical_pretrained_sha256']:
        raise ValueError('Independent LoRA initialization requires identical frozen encoder tensors')
    restore_learned_initial_state(network, catalog['learned_state'])
    return catalog['initial_learned_sha256']


class AuthorSession:
    """Feed actual source phases with the caller's already-transformed inputs.

    ``inputs(task_id, phase, epoch)`` returns a native-layout iterable of
    (augmented,label,non_augmented) batches. ``phase_rng`` owns model RNG and
    loader resets; the adapter never invents augmentation or shuffles itself.
    Evaluation labels include all seen classes, and never select oracle heads.
    """
    def __init__(self, learner, module, spec, evidence):
        self.learner, self.module, self.spec, self.evidence = learner, module, deepcopy(spec), evidence
        self.results = []

    def run_task(self, inputs, *, phase_rng):
        import torch
        from scripts.low_rank_protocol import tensor_digest
        from scripts.run_low_rank_reference import numpy_covariances

        learner = self.learner
        task = learner.cur_task + 1
        if task >= 2:
            raise ValueError('The independent reference stops after two tasks')
        current = {'phase': 'task_begin', 'epoch': 0, 'batch': None}
        row = {'task': task + 1, 'observe_calls': 0, 'optimizer_updates': 0,
               'first_batch_audits': [], 'first_gradients': [], 'epoch_lrs': []}
        self.current_task_record = row

        def phase(name, epoch=0):
            current.update(phase=name, epoch=epoch)
            phase_rng(task, name, epoch)

        class SourceInputs:
            def __iter__(self):
                for batch in inputs(task, current['phase'], current['epoch']):
                    images, labels = batch[:2]
                    if current['phase'] != 'evaluation' and bool(((labels < task * learner.increment) | (labels >= (task + 1) * learner.increment)).any()):
                        raise ValueError('Independent source read old/future training samples')
                    current['batch'] = (images, labels)
                    if current['phase'] == 'train':
                        row['observe_calls'] += 1
                    yield None, images, labels

            def __len__(self):
                return len(inputs(task, current['phase'], current['epoch']))

        class Progress:
            def __init__(self, values):
                self.values = values

            def __iter__(self):
                for epoch in self.values:
                    phase('train', epoch)
                    yield epoch

            def set_description(self, text):
                pass

        class DataManager:
            def get_task_size(self, selected):
                return learner.increment

        loader = SourceInputs()
        original_progress, original_optimizer = self.module.tqdm, learner.build_optimizer
        handles, originals = [], {}

        def build_optimizer(parameters):
            optimizer, scheduler = original_optimizer(parameters)

            def before_step(subject, args, kwargs):
                row['optimizer_updates'] += 1
                # The author leaves stale gradients on newly frozen old heads.
                # Audit the parameters eligible for this optimizer's update.
                gradients = {name: parameter.grad for name, parameter in learner.network.named_parameters()
                             if parameter.requires_grad and parameter.grad is not None}
                if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients.values()):
                    raise FloatingPointError('Independent source optimizer received missing/nonfinite gradients')
                if current['epoch'] == 0 and row['observe_calls'] == 1:
                    images, labels = current['batch']
                    row['first_batch_audits'].append({'input_sha256': tensor_digest({'inputs': images}),
                                                    'labels_sha256': tensor_digest({'labels': labels}),
                                                    'gradient_sha256': tensor_digest(gradients),
                                                    'lrs': [group['lr'] for group in subject.param_groups]})
                    row['first_gradients'].append({name: value.detach().cpu().clone() for name, value in gradients.items()})

            handles.append(optimizer.register_step_pre_hook(before_step))
            original_step = scheduler.step

            def scheduled(*args, **kwargs):
                result = original_step(*args, **kwargs)
                row['epoch_lrs'].append([group['lr'] for group in optimizer.param_groups])
                return result

            scheduler.step = scheduled
            return optimizer, scheduler

        learner.build_optimizer = build_optimizer
        learner.build_train_loader = lambda manager: setattr(learner, 'train_loader', loader)
        self.module.tqdm = lambda values, **kwargs: Progress(values) if isinstance(values, range) else original_progress(values, **kwargs)
        if self.spec['method'] == 'inf-lora':
            originals['update_DualGPM'] = learner.update_DualGPM
            learner.update_DualGPM = lambda matrices: originals['update_DualGPM'](numpy_covariances(matrices))
            for name, selected_phase in (('init_drm', 'task_begin'), ('psv_info', 'task_end')):
                originals[name] = getattr(learner, name)

                def wrapper(supplied, _name=name, _phase=selected_phase):
                    phase(_phase)
                    return originals[_name](supplied)

                setattr(learner, name, wrapper)
        if self.spec['method'] == 'cl-lora':
            def prototypes(manager):
                phase('task_end')
                learner.train_loader_for_protonet = loader
            learner.build_train_loader_for_protonet = prototypes
        try:
            phase('task_begin')
            learner.before_task(DataManager())
            if self.spec['method'] == 'cl-lora':
                with torch.autograd.graph.allow_mutation_on_saved_tensors():
                    learner.incremental_train(DataManager())
            else:
                learner.incremental_train(DataManager())
            phase('evaluation')
            learner.network.eval()
            outputs, labels = [], []
            with torch.no_grad():
                for batch in inputs(task, 'evaluation', 0):
                    images, targets = batch[:2]
                    if bool(((targets < 0) | (targets >= learner.total_classes)).any()):
                        raise ValueError('Independent source evaluation labels are outside the seen classes')
                    logits = learner.network.interface(images.to(learner.device))
                    if logits.shape != (len(targets), learner.total_classes) or not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError('Independent source seen-class inference is invalid')
                    outputs.append(logits.detach().cpu())
                    labels.append(targets.detach().cpu())
            row['logits'], row['labels'] = torch.cat(outputs), torch.cat(labels)
            phase('task_end')
            learner.after_task()
            row['network_task'], row['known_classes'] = learner.network._cur_task, learner.known_classes
            row['state_sha256'] = tensor_digest(learner.network.state_dict())
            self.results.append(row)
            return row
        finally:
            self.module.tqdm = original_progress
            learner.build_optimizer = original_optimizer
            for name, function in originals.items():
                setattr(learner, name, function)
            for handle in handles:
                handle.remove()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=tuple(METHODS), required=True)
    parser.add_argument('--vit-size', choices=('tiny', 'small', 'base'), default='tiny')
    parser.add_argument('--batch-size', type=int, choices=(16, 64), default=16)
    parser.add_argument('--reference-source', type=Path, default=DEFAULT_SOURCE)
    options = parser.parse_args(argv)
    spec = reference_spec(options.method, options.vit_size, options.batch_size)
    print(json.dumps({'experiment_executed': False, 'model_constructed': False,
                      'reference_source': str(options.reference_source), 'reference': reference_description(spec),
                      'training_entrypoint': 'Serial launcher calls build_reference and AuthorSession.run_task with audited common inputs'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
