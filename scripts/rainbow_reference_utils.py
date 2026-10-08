"""Independent active RainbowPrompt classes and engine from pinned author ASTs.

No native RainbowPrompt implementation is imported here. The only training
engine adaptation guards CUDA synchronization for synthetic CPU checks; matched
seen-class CE is supplied through a criterion wrapper at the call boundary.
"""
from __future__ import annotations

import ast
from collections import defaultdict, deque
import datetime
from functools import partial
import hashlib
import logging
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Iterable

REFERENCE_COMMIT = '0b84d3e3738669fe4bb516ccc1c526a535576ad3'
SOURCE_HASHES = {
    'prompt.py': 'f0411c0ae36af03baeeafc10971db7fc109494ee82e15baede3d7de118d92303',
    'attention.py': 'bed8a5f1dbc13717e7c0ae29ae9bab16d15592428d91e9547a4a25cf22f67e5c',
    'vision_transformer.py': '59c16c8929654768bdaf109024d710cbb7c306dbfc2f865e2f617041a6c09df8',
    'engine.py': '269c612f134fb705e633abb6cee7b29e96b5315652c7c3df0cf10aca0cf3b4d7',
    'utils.py': '7f4c402e074590068c5440c501f6f8b571d1e9ed564a1559dbd8737264ef4e6d',
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def verify_author_source(source, expected=None):
    source = Path(source).resolve()
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=source,
                            capture_output=True, text=True, check=True).stdout.splitlines()
    dirty = [line for line in status if not (line.startswith('?? ') and '__pycache__/' in line and line.endswith('.pyc'))]
    expected = dict(SOURCE_HASHES if expected is None else expected)
    if any(expected.get(name) != digest for name, digest in SOURCE_HASHES.items()):
        raise ValueError('RainbowPrompt author class pins differ from the reviewed source')
    actual = {name: file_hash(source / name) for name in expected}
    if head != REFERENCE_COMMIT or dirty or actual != expected:
        raise ValueError('RainbowPrompt author source must remain clean at the pinned commit')
    return actual


class _GuardSynchronize(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node.func) == 'torch.cuda.synchronize':
            node.func = ast.copy_location(ast.Name(id='_synchronize', ctx=ast.Load()), node.func)
        return node


def _definitions(path, names, namespace, transformer=None):
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError('Pinned RainbowPrompt author source is missing required definitions')
    tree = ast.Module(body=nodes, type_ignores=[])
    if transformer is not None:
        tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), 'exec'), namespace)


def load_author_code(source, expected=None):
    verify_author_source(source, expected)
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
    from timm.models._manipulate import named_apply, adapt_input_conv, checkpoint_seq
    from timm.utils import accuracy
    source = Path(source).resolve()
    logger = logging.getLogger('rainbow_pinned_author')
    logger.setLevel(logging.ERROR)
    namespace = {'__name__': 'rainbow_pinned_author', 'torch': torch, 'nn': nn, 'F': F,
        'np': np, 'math': math, 'partial': partial, 'logger': logger, '_logger': logger,
        'PatchEmbed': PatchEmbed, 'Mlp': Mlp, 'DropPath': DropPath, 'trunc_normal_': trunc_normal_,
        'lecun_normal_': lecun_normal_, 'named_apply': named_apply, 'adapt_input_conv': adapt_input_conv,
        'checkpoint_seq': checkpoint_seq}
    _definitions(source / 'prompt.py', {'RainbowPrompt'}, namespace)
    _definitions(source / 'attention.py', {'PreT_Attention'}, namespace)
    _definitions(source / 'vision_transformer.py', {'Attention', 'LayerScale', 'Block', 'VisionTransformer',
        'init_weights_vit_timm', 'init_weights_vit_jax', 'init_weights_vit_moco', 'get_init_weights_vit',
        '_load_weights', 'resize_pos_embed'}, namespace)
    utilities = {'__name__': 'rainbow_pinned_author_utils', 'torch': torch, 'dist': torch.distributed,
                 'defaultdict': defaultdict, 'deque': deque, 'datetime': datetime, 'time': time}
    _definitions(source / 'utils.py', {'SmoothedValue', 'MetricLogger', 'is_dist_avail_and_initialized',
                                      'get_world_size'}, utilities)

    def synchronize():
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    engine = {'__name__': 'rainbow_pinned_author_engine', 'torch': torch, 'np': np, 'math': math,
              'sys': sys, 'Iterable': Iterable, 'logger': logger, 'accuracy': accuracy,
              'utils': SimpleNamespace(**utilities), '_synchronize': synchronize}
    _definitions(source / 'engine.py', {'train_one_epoch'}, engine, _GuardSynchronize())
    return SimpleNamespace(**namespace, engine=engine, train_one_epoch=engine['train_one_epoch'], source=source,
        adaptations=['Compile immutable source AST without timm registration or package collisions',
                     'Guard CUDA synchronize only for synthetic CPU execution',
                     'Use criterion wrapper for matched seen-class CE after original current-task mask',
                     'Source loss.backward(retain_graph=True), clipping and task-local Adam are unchanged',
                     'Source batchwise_prompt/use_prompt_mask=True versus native False are inert in the active prompt forward',
                     'Evaluation clips seen logits and preserves separate test-task batches and batch-mean routing'])


def build_author_branches(code, checkpoint=None, *, num_classes=100, embed_dim=768, depth=12, num_heads=12,
                          img_size=224, patch_size=16, prompt_length=20, pool_size=10, n_tasks=10,
                          prompt_layers=None, D1=56, D2=96, warmup=5, KI_iter=10, self_attn_layers=None):
    layers = list(range(depth)) if prompt_layers is None else list(prompt_layers)
    self_layers = list(range(min(6, depth))) if self_attn_layers is None else list(self_attn_layers)
    common = dict(num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                  img_size=img_size, patch_size=patch_size, drop_rate=0., drop_path_rate=0.,
                  head_type='token', global_pool='token')
    query = code.VisionTransformer(**common)
    prompt = code.VisionTransformer(**common, prompt_length=prompt_length, pool_size=pool_size,
        prompt_pool=True, prompt_key=True, embedding_key='cls', prompt_init='uniform',
        prompt_key_init='uniform', top_k=1, batchwise_prompt=True,
        use_prompt_mask=True, use_e_prompt=True, e_prompt_layer_idx=layers,
        use_prefix_tune_for_e_prompt=True, n_tasks=n_tasks, D1=D1, D2=D2,
        warm_up=warmup, KI_iter=KI_iter, self_attn_idx=self_layers, relation_type='attention', use_linear=True)
    branches = {'query': query, 'prompt': prompt}
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file() or checkpoint.suffix != '.npz':
            raise ValueError('Independent RainbowPrompt requires an existing local NPZ')
        for module in branches.values():
            module.load_pretrained(str(checkpoint))
    set_trainable(branches)
    return branches


def set_trainable(branches):
    branches['query'].requires_grad_(False).eval()
    prompt = branches['prompt']
    for name, parameter in prompt.named_parameters():
        parameter.requires_grad_(name.startswith(('rainbow_prompt.', 'head.')))


def learned_state(branches):
    return {f'{branch}.{name}': value.detach().cpu().clone()
            for branch, module in branches.items() for name, value in module.state_dict().items()
            if name.startswith(('rainbow_prompt.', 'head.'))}


def save_initial_state(branches, path, checkpoint_sha256, seed=1993):
    import torch
    path = Path(path)
    if path.exists():
        raise FileExistsError('RainbowPrompt initialization already exists; no overwrite')
    state = learned_state(branches)
    record = {'schema_version': 1, 'seed': seed, 'reference_commit': REFERENCE_COMMIT,
              'checkpoint_sha256': checkpoint_sha256, 'state': state, 'learned_sha256': state_digest(state)}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, path)
    return {key: record[key] for key in ('schema_version', 'seed', 'reference_commit', 'checkpoint_sha256', 'learned_sha256')}


def load_initial_state(branches, path, checkpoint_sha256):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=True)
    expected = learned_state(branches)
    state = saved.get('state', {})
    if (saved.get('schema_version') != 1 or saved.get('seed') != 1993
            or saved.get('reference_commit') != REFERENCE_COMMIT or saved.get('checkpoint_sha256') != checkpoint_sha256
            or set(state) != set(expected) or state_digest(state) != saved.get('learned_sha256')
            or any(value.shape != expected[name].shape or value.dtype != expected[name].dtype
                   or not bool(torch.isfinite(value).all()) for name, value in state.items())):
        raise ValueError('RainbowPrompt matched initialization differs from the complete pinned catalogue')
    for branch, module in branches.items():
        selected = {key[len(branch) + 1:]: value for key, value in state.items() if key.startswith(branch + '.')}
        result = module.load_state_dict(selected, strict=False)
        if result.unexpected_keys or any(branch + '.' + key in expected for key in result.missing_keys):
            raise ValueError('RainbowPrompt did not load every matched learned or query-head tensor')
    return saved['learned_sha256']


class PairLoader:
    def __init__(self, loader, callback=None):
        self.loader, self.callback = loader, callback

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            if self.callback is not None:
                self.callback(batch[0], batch[1])
            yield batch[0], batch[1]


class SeenClassCriterion:
    """Explicit Mammoth adaptation; source masking and auxiliary loss stay intact."""
    def __init__(self, seen_classes):
        self.seen_classes = seen_classes

    def __call__(self, logits, labels):
        from torch.nn import functional as F
        return F.cross_entropy(logits[:, :self.seen_classes], labels)
