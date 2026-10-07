"""Independent, reversible integration of the pinned HiDe-Prompt author code.

Model, prefix, training loss, class statistics and alignment functions are
compiled directly from immutable upstream AST nodes. Native algorithm modules
are never used here. Changes adapt one-process execution and explicit RNG pins.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from functools import partial
import hashlib
import json
import logging
import math
from pathlib import Path
import subprocess
import types

REFERENCE_COMMIT = '0ed433334ff2cfb206c5e2899651c3044ec85f92'
ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def verify_author_source(source):
    source = Path(source).resolve()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, capture_output=True,
                            text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=source, capture_output=True,
                           text=True, check=True).stdout.strip()
    if commit != REFERENCE_COMMIT or dirty:
        raise ValueError('Author reference must be the clean pinned commit')
    expected = json.loads((ROOT / 'benchmarks/cl_efficiency/hide_prompt/source_review.json').read_text())['source_files_sha256']
    actual = {name: file_hash(source / name) for name in expected}
    if actual != expected:
        raise ValueError('Author source file hashes differ from the reviewed pin')
    actual['utils.py'] = file_hash(source / 'utils.py')
    return actual


def _definitions(path, names, namespace, transformer=None):
    """Compile selected source definitions with their original filename/lines."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    chosen = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in chosen} != set(names):
        raise ValueError('Pinned author source is missing required definitions')
    selected = ast.Module(body=chosen, type_ignores=[])
    if transformer is not None:
        selected = transformer.visit(selected)
    ast.fix_missing_locations(selected)
    exec(compile(selected, str(path), 'exec'), namespace)
    return namespace


class _SingleProcessEngine(ast.NodeTransformer):
    """Keep source arithmetic, pin KMeans and replace distributed-only I/O."""
    function = None

    def visit_FunctionDef(self, node):
        previous = self.function
        self.function = node.name
        result = self.generic_visit(node)
        self.function = previous
        return result

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            call = node.value
            if ast.unparse(call.func) == 'dist.barrier':
                return ast.copy_location(ast.Pass(), node)
            if ast.unparse(call.func) == 'dist.all_gather':
                return ast.copy_location(ast.parse('features_per_cls_list[0].copy_(features_per_cls)').body[0], node)
            if ast.unparse(call.func) == 'optimizer.step' and self.function == 'train_one_epoch':
                audit = ast.parse('_audit_batch(input, target, loss, model, optimizer)').body[0]
                return [ast.copy_location(audit, node), self.generic_visit(node)]
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        name = ast.unparse(node.func)
        if name == 'KMeans':
            node.keywords.extend([
                ast.keyword(arg='init', value=ast.Constant('k-means++')),
                ast.keyword(arg='algorithm', value=ast.Constant('lloyd')),
                ast.keyword(arg='n_init', value=ast.Constant(10)),
                ast.keyword(arg='random_state', value=ast.parse('(args.seed + cls_id) % 2**32', mode='eval').body),
            ])
        elif name == 'torch.range':
            node.func = ast.parse('torch.arange', mode='eval').body
            node.args[1] = ast.BinOp(left=node.args[1], op=ast.Add(), right=ast.Constant(1))
        elif name == 'torch.cuda.synchronize':
            node.func = ast.Name(id='_synchronize', ctx=ast.Load())
        return node


def load_author_code(source):
    """Return source-derived constructors and two independent engine namespaces."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
    from timm.models._manipulate import named_apply, adapt_input_conv, checkpoint_seq
    from timm.utils import accuracy
    source = Path(source).resolve()
    basic = {'__name__': 'hide_pinned_author', 'torch': torch, 'nn': nn, 'F': F,
             'math': math, 'partial': partial, 'OrderedDict': OrderedDict,
             'Optional': __import__('typing').Optional, '_logger': logging.getLogger(__name__),
             'PatchEmbed': PatchEmbed, 'Mlp': Mlp, 'DropPath': DropPath,
             'trunc_normal_': trunc_normal_, 'lecun_normal_': lecun_normal_,
             'named_apply': named_apply, 'adapt_input_conv': adapt_input_conv,
             'checkpoint_seq': checkpoint_seq}
    _definitions(source / 'peft/prompt/hide_prompt.py', {'EPrompt'}, basic)
    _definitions(source / 'attention.py', {'PreT_Attention'}, basic)
    _definitions(source / 'vits/hide_prompt_vision_transformer.py', {
        'Attention', 'LayerScale', 'Block', 'MlpMapping', 'VisionTransformer',
        'init_weights_vit_timm', 'init_weights_vit_jax', 'init_weights_vit_moco',
        'get_init_weights_vit', '_load_weights', 'resize_pos_embed'}, basic)
    utilities = types.ModuleType('hide_pinned_author_utils')
    exec(compile((source / 'utils.py').read_text(), str(source / 'utils.py'), 'exec'), utilities.__dict__)

    def synchronize():
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    engines = {}
    for branch, file in [('tii', 'hide_tii_engine.py'), ('prompt', 'hide_promtp_wtp_and_tap_engine.py')]:
        namespace = {'__name__': f'hide_pinned_author_{branch}', 'torch': torch, 'np': np,
                     'math': math, 'sys': __import__('sys'), 'Iterable': __import__('typing').Iterable,
                     'optim': torch.optim, 'accuracy': accuracy, 'utils': utilities,
                     'MultivariateNormal': torch.distributions.MultivariateNormal,
                     '_synchronize': synchronize, '_audit_batch': lambda *items: None, 'cls_mean': {}, 'cls_cov': {}}
        names = {'train_one_epoch', '_compute_mean', 'train_task_adaptive_prediction'}
        if branch == 'prompt':
            names.add('orth_loss')
        _definitions(source / 'engines' / file, names, namespace, _SingleProcessEngine())
        engines[branch] = namespace
    return types.SimpleNamespace(VisionTransformer=basic['VisionTransformer'], EPrompt=basic['EPrompt'],
                                 engines=engines, source=source,
                                 adaptations=['Compile immutable author AST, avoid global timm registrations and package collisions',
                                              'Single-process feature gather copy; no distributed barriers',
                                              'Explicit KMeans init=k-means++, Lloyd, n_init10, seed+class',
                                              'Deprecated inclusive torch.range becomes equivalent torch.arange',
                                              'Guard CUDA synchronize for synthetic CPU tests',
                                              'Read-only first-batch input/loss/gradient callback before actual optimizer.step',
                                              'Matched interleaved task stages, per-class sequential statistics and shared dataset adapter'])


def build_author_branches(code, checkpoint=None, *, embed_dim=768, depth=12, num_heads=12, img_size=224, patch_size=16):
    """Build actual author ViTs with explicit geometry and an optional local NPZ."""
    common = dict(num_classes=100, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                  img_size=img_size, patch_size=patch_size, drop_rate=0., drop_path_rate=0.,
                  head_type='token', global_pool='token')
    tii = code.VisionTransformer(**common, mlp_structure=[2])
    prompt = code.VisionTransformer(**common, mlp_structure=[], prompt_length=5,
                                   prompt_pool=True, prompt_key=False, pool_size=10, top_k=1,
                                   batchwise_prompt=False, use_prompt_mask=True, use_g_prompt=False,
                                   g_prompt_length=0, g_prompt_layer_idx=[], use_prefix_tune_for_g_prompt=True,
                                   use_e_prompt=True, e_prompt_layer_idx=list(range(min(5, depth))),
                                   use_prefix_tune_for_e_prompt=True, same_key_value=False)
    branches = {'tii': tii, 'prompt': prompt}
    if checkpoint is not None:
        if not Path(checkpoint).is_file() or Path(checkpoint).suffix != '.npz':
            raise ValueError('Reference requires an existing local NPZ')
        for module in branches.values():
            module.load_pretrained(str(checkpoint))
    set_author_phase(branches, 'eval')
    return branches


def learned_state(module, branch):
    if branch not in ('tii', 'prompt'):
        raise ValueError('Unknown HiDe-Prompt branch')
    prefixes = ('mlp.', 'head.') if branch == 'tii' else ('e_prompt.prompt', 'head.')
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items() if name.startswith(prefixes)}


def state_digest(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        cpu = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(tuple(cpu.shape)).encode())
        result.update(str(cpu.dtype).encode())
        result.update(cpu.numpy().tobytes())
    return result.hexdigest()


def learned_state_digest(module, branch):
    return state_digest(learned_state(module, branch))


def save_initial_state(branches, path, checkpoint_sha256, seed=1993):
    import torch
    path = Path(path)
    if path.exists():
        raise FileExistsError('Matched initialization artifact already exists')
    state = {branch: learned_state(module, branch) for branch, module in branches.items()}
    saved = {'schema_version': 1, 'seed': seed, 'reference_commit': REFERENCE_COMMIT,
             'checkpoint_sha256': checkpoint_sha256, 'branches': state,
             'learned_sha256': {branch: state_digest(values) for branch, values in state.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, path)
    return {key: saved[key] for key in ('schema_version', 'seed', 'reference_commit', 'checkpoint_sha256', 'learned_sha256')}


def load_initial_state(branches, path):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('schema_version') != 1 or saved.get('seed') != 1993 or saved.get('reference_commit') != REFERENCE_COMMIT:
        raise ValueError('Matched initial state metadata differ from the protocol')
    if set(saved.get('branches', {})) != {'tii', 'prompt'}:
        raise ValueError('Matched initialization needs both complete learned branches')
    for branch, module in branches.items():
        expected = learned_state(module, branch)
        state = saved['branches'][branch]
        if set(state) != set(expected) or state_digest(state) != saved['learned_sha256'][branch]:
            raise ValueError('Learned initialization tensor names or hashes differ')
        if any(value.shape != expected[name].shape or value.dtype != expected[name].dtype or
               not bool(torch.isfinite(value).all()) for name, value in state.items()):
            raise ValueError('Learned initialization contains invalid tensors')
        result = module.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(key in expected for key in result.missing_keys):
            raise ValueError('Matched initial state did not load every learned tensor')
    return {key: saved[key] for key in ('schema_version', 'seed', 'reference_commit', 'checkpoint_sha256', 'learned_sha256')}


def set_author_phase(branches, phase):
    for module in branches.values():
        module.requires_grad_(False)
        for parameter in module.parameters():
            parameter.grad = None
        module.eval()
    if phase in ('tii', 'tii_alignment'):
        for name, parameter in branches['tii'].named_parameters():
            parameter.requires_grad_(name.startswith(('mlp.', 'head.')))
        branches['tii'].train(phase == 'tii')
    elif phase == 'prompt':
        for name, parameter in branches['prompt'].named_parameters():
            parameter.requires_grad_(name.startswith(('e_prompt.prompt', 'head.')))
        branches['prompt'].train()
    elif phase == 'prompt_alignment':
        for parameter in branches['prompt'].head.parameters():
            parameter.requires_grad_(True)
    elif phase != 'eval':
        raise ValueError('Unknown author HiDe-Prompt phase')


class PairLoader:
    """Expose the author's pair contract while preserving native sample batching."""
    def __init__(self, loader, recorder=None, max_batches=None):
        self.loader, self.recorder, self.max_batches = loader, recorder, max_batches
        self.sampler = loader.sampler

    def __len__(self):
        return min(len(self.loader), self.max_batches) if self.max_batches is not None else len(self.loader)

    def __iter__(self):
        import itertools
        rows = iter(self.loader)
        if self.max_batches is not None:
            rows = itertools.islice(rows, self.max_batches)
        for batch in rows:
            if self.recorder is not None:
                self.recorder.steps += 1
                self.recorder.training_examples += len(batch[1])
            yield batch[0], batch[1]
