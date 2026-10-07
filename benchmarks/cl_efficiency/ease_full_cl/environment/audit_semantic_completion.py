"""Replay the CPU-only EASE T3..T10 arithmetic audit against pinned author code.

This evidence script executes the author's unchanged solve_sim_reset AST and
CosineLinear.forward_reweight on precomputed tensors. It performs no image
forwards, learning, GPU work, credential reads or network calls. It changes no
runtime source or checkpoint. Run with CUDA_VISIBLE_DEVICES='' explicitly.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def isolated_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_method(path, class_name, method_name):
    source = path.read_text()
    tree = ast.parse(source)
    subject = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in subject.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    return method, {'file': str(path), 'file_sha256': file_hash(path),
                    'class': class_name, 'function': method_name,
                    'first_line': method.lineno, 'last_line': method.end_lineno,
                    'function_source_sha256': hashlib.sha256(ast.get_source_segment(source, method).encode()).hexdigest(),
                    'function_ast_sha256': hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest(),
                    'modification': 'None; original AST/function body executed unchanged'}


def audit(author_root, native_run):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError("CPU evidence audit requires explicit CUDA_VISIBLE_DEVICES=''")
    import torch
    from torch.nn import functional as F
    from scripts.ease_fullcl_protocol import REFERENCE_COMMIT, source_hashes
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise RuntimeError('CPU evidence audit must not expose or initialize CUDA')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=author_root,
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=author_root,
                           capture_output=True, text=True, check=True).stdout.strip()
    if commit != REFERENCE_COMMIT or dirty:
        raise ValueError('Author arithmetic source must remain clean at its pinned commit')
    native_spec = json.loads((native_run / 'spec.json').read_text())
    source_pins = source_hashes()
    if source_pins != native_spec['execution_source_sha256']:
        raise ValueError('Protected native execution sources changed')
    native_path = ROOT / 'models/ease_utils/classifier.py'
    native = isolated_module('_audit_ease_native_classifier', native_path)
    author_linear_path = author_root / 'backbone/linears.py'
    author_linear = isolated_module('_audit_ease_author_linears', author_linear_path)
    method, completion_provenance = extract_method(author_root / 'models/ease.py', 'Learner', 'solve_sim_reset')
    _, inference_provenance = extract_method(author_linear_path, 'CosineLinear', 'forward_reweight')
    namespace = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=[method], type_ignores=[]), completion_provenance['file'], 'exec'), namespace)
    solve = namespace['solve_sim_reset']
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(1993)

    def author_complete(weight, dimension, tasks):
        subject = SimpleNamespace(_cur_task=tasks - 1, _total_classes=10 * tasks,
            _device=torch.device('cpu'), init_cls=10, inc=10, use_init_ptm=False,
            moni_adam=False, adapter_num=-1,
            _network=SimpleNamespace(out_dim=dimension, fc=SimpleNamespace(weight=weight)))
        subject.get_cls_range = lambda task_id: (10 * task_id, 10 * (task_id + 1))
        solve(subject)
        return weight

    synthetic = []
    prototype = native.PrototypeClassifier(768)
    author_weight = torch.empty(0, 0)
    with torch.no_grad():
        # Tasks1/2 establish the actual incremental history; the recorded audit
        # scope is T3..T10, where multiple old task spaces must be aggregated.
        for tasks in range(1, 11):
            dimension = 768
            prototype.expand(10 * tasks, tasks, [(10 * i, 10 * (i + 1)) for i in range(tasks)])
            expanded = torch.zeros(10 * tasks, dimension * tasks)
            expanded[:author_weight.shape[0], :author_weight.shape[1]] = author_weight
            measured = torch.randn(10, dimension * tasks)
            prototype.update_current(measured, 10 * (tasks - 1))
            expanded[10 * (tasks - 1):] = measured
            prototype.complete_old_prototypes(True)
            author_weight = author_complete(expanded, dimension, tasks)
            delta = float((prototype.weight - author_weight).abs().max())
            torch.testing.assert_close(prototype.weight, author_weight, atol=2e-6, rtol=2e-5)
            if tasks < 3:
                continue
            diagonal_exact = all(torch.equal(
                prototype.weight[10 * i:10 * (i + 1), dimension * i:dimension * (i + 1)],
                author_weight[10 * i:10 * (i + 1), dimension * i:dimension * (i + 1)]) for i in range(tasks))
            assert diagonal_exact
            features = torch.randn(7, dimension * tasks)
            inference = []
            for alpha in (0., .1, 1.):
                cosine = author_linear.CosineLinear(dimension * tasks, 10 * tasks)
                cosine.weight.copy_(prototype.weight)
                expected = cosine.forward_reweight(features, cur_task=tasks - 1, alpha=alpha,
                    init_cls=10, inc=10, out_dim=dimension, use_init_ptm=False)['logits']
                actual = prototype(features, use_reweight=True, alpha=alpha)
                difference = float((actual - expected).abs().max())
                torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
                same_top1 = bool(torch.equal(actual.argmax(1), expected.argmax(1)))
                assert same_top1
                inference.append({'alpha': alpha, 'max_abs_logit_difference': difference,
                                  'top1_identical': same_top1})
            synthetic.append({'tasks': tasks, 'semantic_max_abs_difference': delta,
                              'old_diagonal_exact': diagonal_exact, 'inference': inference})

    actual_tables = []
    with torch.no_grad():
        for tasks in (3, 4):
            path = native_run / f'task{tasks}_state.pt'
            if not path.is_file():
                raise FileNotFoundError('The original audited task3/task4 states must exist')
            state_pin = file_hash(path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            weight = checkpoint['model']['classifier.weight'].clone()
            offsets = checkpoint['model']['classifier._task_offsets'].clone()
            del checkpoint
            assert tuple(weight.shape) == (10 * tasks, 768 * tasks)
            assert offsets.tolist() == [[10 * i, 10 * (i + 1)] for i in range(tasks)]
            expected = author_complete(weight.clone(), 768, tasks)
            delta = float((weight - expected).abs().max())
            torch.testing.assert_close(weight, expected, atol=1e-5, rtol=2e-5)
            actual_tables.append({'tasks': tasks, 'state_file': str(path), 'state_file_sha256': state_pin,
                'state_file_bytes': path.stat().st_size, 'prototype_tensor_sha256': hashlib.sha256(weight.numpy().tobytes()).hexdigest(),
                'shape': list(weight.shape), 'task_offsets': offsets.tolist(),
                'max_abs_semantic_difference': delta, 'finite': bool(torch.isfinite(weight).all())})
    assert not torch.cuda.is_initialized()
    assert source_hashes() == source_pins
    return {'schema_version': 1, 'status': 'pass',
        'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'audit_scope': 'Pinned-author semantic completion and inference reweight on precomputed CPU tensors; actual saved native task3/task4 prototypes',
        'gpu_used': False, 'cuda_initialized': False, 'image_forwards': 0,
        'training_executed': False, 'optimizer_steps': 0, 'network_calls': 0,
        'cpu_conditions': {'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
            'torch_cpu_threads': torch.get_num_threads(), 'parameter_dtype': 'float32',
            'seed': 1993, 'device': 'cpu', 'torch_version': importlib.metadata.version('torch')},
        'author_commit': commit, 'author_clean': True,
        'author_functions': [completion_provenance, inference_provenance],
        'native_classifier': {'file': str(native_path), 'sha256': file_hash(native_path)},
        'native_run': str(native_run), 'native_run_spec_sha256': file_hash(native_run / 'spec.json'),
        'native_execution_commit': native_spec['git_commit'],
        'native_source_count': len(source_pins), 'all_execution_sources_frozen': True,
        'native_execution_source_sha256': source_pins,
        'comparison_conditions': {'synthetic_tasks': list(range(3, 11)), 'feature_dim': 768,
            'classes_per_task': 10, 'synthetic_inference_samples': 7, 'alpha_values': [0., .1, 1.],
            'recalc_sim': True, 'use_init_ptm': False, 'moni_adam': False,
            'completion_assertion': {'atol': 2e-6, 'rtol': 2e-5},
            'inference_assertion': {'atol': 2e-7, 'rtol': 2e-5},
            'actual_state_assertion': {'atol': 1e-5, 'rtol': 2e-5}},
        'synthetic_T3_to_T10': synthetic, 'actual_saved_prototype_tables': actual_tables,
        'interpretation': 'No semantic range/index or alpha aggregation mismatch. Small nonzero differences reflect FP32 loop versus vectorized reduction order; this audit is not an independent full10 image performance reproduction.',
        'audit_script': {'file': str(Path(__file__).resolve()), 'sha256': file_hash(__file__)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--author-source', type=Path, default=Path('/tmp/cl-ease-official-20261007'))
    parser.add_argument('--native-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.author_source.resolve(), args.native_run.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'status': result['status'], 'output': str(args.output.resolve()),
                      'native_sources_frozen': result['all_execution_sources_frozen'],
                      'synthetic_task_count': len(result['synthetic_T3_to_T10']),
                      'actual_audited_tasks': [row['tasks'] for row in result['actual_saved_prototype_tables']]}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
