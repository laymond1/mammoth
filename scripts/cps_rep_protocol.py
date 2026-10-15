"""Bounded protocol and acceptance gates for CPS/REP implementation checks."""
import hashlib
from pathlib import Path

from scripts.cl_runtime import public_setting
from scripts.cps_author_reference import SOURCE_COMMIT, file_hash, verify_source

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'cps-author': ('cps-prompt', '1'),
    'cps-native': ('cps-prompt', '1'),
    'coda-frozen': ('coda-frozen', '0'),
    'rep-off': ('rep-prompt', '0'),
    'rep-on': ('rep-prompt', '0'),
}


def protocol():
    return {'dataset': 'seq-cifar100-224', 'dataset_config': 'unified', 'original_tasks': 10,
            'tasks': 2, 'epochs_per_task': 3, 'batch_size': 64, 'seed': 1993,
            'vit_size': 'tiny', 'precision': 'fp32', 'amp': False, 'tf32': False,
            'optimizer': 'adam', 'lr': .001, 'weight_decay': 0., 'scheduler': 'CosineSchedule(K=3)',
            'class_order': list(range(100)), 'num_workers': 2, 'cpu_threads': 2,
            'pool_size': 100, 'prompt_length': 8, 'prompt_layers': [0, 1, 2, 3, 4], 'ortho_mu': 0.,
            'cps': {'reduction_ratio': .4, 'temperature': .1, 'phase_ratio': .4,
                    'joint_epochs': 1, 'head_epochs': 2, 'clip_grad': 1., 'pull_constraint_coeff': 1.},
            'rep': {'merge_max': 16, 'minimum': .5, 'gamma': .001, 'alpha': .9, 'threshold': 8},
            'comparison_scope': 'implementation verification; not paper accuracy reproduction or efficiency pilot'}


def native_arguments(run, output, *, cpu_smoke=False):
    if run not in RUNS:
        raise ValueError('Unknown bounded CPS/REP validation run')
    args = ['--model', RUNS[run][0], '--model_config', 'default', '--dataset', 'seq-cifar100-224',
            '--dataset_config', 'unified', '--backbone', 'resnet18', '--device', 'cpu' if cpu_smoke else 'cuda:0',
            '--base_path', public_setting('data_root'), '--results_path', str(Path(output) / 'mammoth'),
            '--n_epochs', '3', '--batch_size', '2' if cpu_smoke else '64', '--seed', '1993',
            '--num_workers', '0' if cpu_smoke else '2', '--non_verbose', '1',
            '--permute_classes', '0', '--enable_other_metrics', '0', '--stop_after', '2',
            '--optimizer', 'adam', '--lr', '0.001', '--optim_wd', '0', '--code_optimization', '0']
    if run.startswith('rep-'):
        enabled = '1' if run == 'rep-on' else '0'
        args += ['--rep_swift', enabled, '--rep_atom', enabled, '--rep_ald', enabled]
    if cpu_smoke:
        args += ['--debug_mode', '1']
    return args


def learned_state(network):
    import torch
    head = network.head if hasattr(network, 'head') else network.last
    return {**{'head.' + name: value.detach().cpu().clone() for name, value in head.state_dict().items()},
            **{'prompt.' + name: value.detach().cpu().clone() for name, value in network.prompt.state_dict().items()
               if torch.is_tensor(value)}}


def tensor_digest(state):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not torch.is_tensor(value):
            continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode() + str(tuple(value.shape)).encode() + str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def restore_learned(network, state):
    import torch
    expected = learned_state(network)
    if set(state) != set(expected) or any(state[name].shape != value.shape or state[name].dtype != value.dtype
                                         or not bool(torch.isfinite(state[name]).all()) for name, value in expected.items()):
        raise ValueError('Matched CPS/REP prompt/head initialization has invalid tensors')
    head = network.head if hasattr(network, 'head') else network.last
    head.load_state_dict({name.removeprefix('head.'): value for name, value in state.items() if name.startswith('head.')})
    network.prompt.load_state_dict({name.removeprefix('prompt.'): value for name, value in state.items() if name.startswith('prompt.')})


def prepare_initial(output):
    """CPU-only source initialization, shared by all five verification runs."""
    from types import SimpleNamespace

    import torch

    from models.low_rank_utils.factory import CHECKPOINTS
    from models.low_rank_utils.pretrained import canonical_pretrained_digest
    from scripts.cl_efficiency import write_json
    from scripts.cps_author_reference import author_classes

    output = Path(output)
    path = output / 'initial_learned.pt'
    if path.exists():
        return path
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.default_generator.manual_seed(1993)
    args = SimpleNamespace(n_tasks=10, vit_type='tiny', e_prompt_pool_size=100, e_prompt_length=8,
                           e_prompt_layer_idx=[0, 1, 2, 3, 4], ortho_mu=0., reduction_ratio=.4,
                           sampling='critical_score', temperature=.1)
    network = author_classes().network(args, num_classes=100, pretrained=True, prompt_flag='coda', prompt_param=[100, 8, 0.])
    if canonical_pretrained_digest(network.feat) != CHECKPOINTS['tiny']['canonical_sha256']:
        raise ValueError('CPS author encoder differs from canonical150 pin')
    state = learned_state(network)
    torch.save(state, path)
    write_json(output / 'initial_manifest.json', {'file_sha256': file_hash(path), 'tensor_sha256': tensor_digest(state),
               'encoder_pin': CHECKPOINTS['tiny'], 'source_commit': SOURCE_COMMIT, 'source_sha256': verify_source(),
               'construction': 'CPU pinned author network, seed1993; shared prompt/head initialization only',
               'task2_initialization': 'Independent forked CPU/CUDA RNG seeded by common task coordinate; source Gram-Schmidt preserved'})
    return path


def accuracy_status(delta, *, decline_only=False):
    difference = max(0., -delta) if decline_only else abs(delta)
    if difference <= 1.5:
        return 'pass'
    if difference <= 3.:
        return 'review'
    return 'hold' if decline_only else 'fail'


def compare_records(actual, expected, *, decline_only=False):
    """Structural evidence is mandatory before interpreting accuracy gaps."""
    if any(record.get('status') != 'completed' or not record.get('structural_checks_passed') for record in (actual, expected)):
        return {'status': 'fail', 'reason': 'incomplete execution or structural checks'}
    keys = ('protocol', 'physical_gpu_uuid', 'initial_learned_sha256', 'epoch_inputs', 'data_fingerprints')
    if any(actual.get(key) != expected.get(key) for key in keys):
        return {'status': 'fail', 'reason': 'comparison protocol, GPU, initialization or actual input stream mismatch'}
    rows = []
    for current, reference in zip(actual['task_results'], expected['task_results'], strict=True):
        for metric in ('seen_accuracy_pct', 'old_accuracy_pct', 'new_accuracy_pct'):
            if reference[metric] is None:
                continue
            delta = current[metric] - reference[metric]
            # REP full has only the agreed gross-decline stop gate, not an
            # author reproduction tolerance or an ablation equivalence claim.
            status = ('hold' if delta < -3. else 'pass') if decline_only else accuracy_status(delta)
            rows.append({'task': current['task'], 'metric': metric, 'actual_pct': current[metric],
                         'reference_pct': reference[metric], 'delta_pp': delta, 'status': status})
    order = {'pass': 0, 'review': 1, 'hold': 2, 'fail': 3}
    return {'status': max((row['status'] for row in rows), key=order.get), 'metrics': rows,
            'scope': 'gross implementation/configuration decline check' if decline_only else 'independent implementation parity'}
