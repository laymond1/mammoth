"""Independent PROL classes and active engine from immutable author ASTs.

This module imports no native PROL math. Compilation omits registration and
unrelated optional packages, while preserving the selected source definitions.
The engine's CUDA synchronization is guarded on CPU; source classifier updates
use an explicitly scoped CPU placement guard when required by synthetic checks.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict, deque
from contextlib import contextmanager
from copy import deepcopy
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

REFERENCE_COMMIT = 'bfff8418a4f603a24ae578f1e108bfac89af1e18'
SOURCE_HASHES = {
    'promptL.py': '7d7ef40a76f32abd8b2ed86c5eb75fe1790e125208ffada605f88f87e3be965d',
    'attention.py': '25a138c9fe27785301442f0ce35245ca785a810e8dbffdca1b08f2591840d8c7',
    'linears.py': '8217380c2e4513b5cfbd9f7f7bbad38d6166d9f605561dfcf04d1d0796cb0ddb',
    'vision_transformerL.py': 'e67f54d8bb729453ec79baa8759e05f2dde29c60d46fd363669c6e4038d14578',
    'engine_prol.py': 'bd7dea9709eb585db320a5844393fa98f863bf5b3a9e10157c9afec98c0c838c',
    'utils.py': 'fc291f42f4a418812694bac8e334ef57286d08e5491435457f36b89733997ed6',
    'main_prol.py': '928954560a9f4f378e47ed25a1eaf22ef0128f271e894c62000c4daf9013ca7b',
    'configs/cifar100_prol.py': 'cee2085030cade383957e4a4106964fdf99b7c9d1517c75785e22f8c40f48f7b',
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_author_source(source, expected=None):
    source = Path(source).resolve()
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, capture_output=True,
                          text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=source,
                           capture_output=True, text=True, check=True).stdout.splitlines()
    dirty = [line for line in dirty if not (line.startswith('?? ') and '__pycache__/' in line and line.endswith('.pyc'))]
    expected = dict(SOURCE_HASHES if expected is None else expected)
    if any(expected.get(name) != digest for name, digest in SOURCE_HASHES.items()):
        raise ValueError('PROL author definition pins differ from the reviewed source')
    actual = {name: file_hash(source / name) for name in expected}
    if head != REFERENCE_COMMIT or dirty or actual != expected:
        raise ValueError('PROL author source must remain clean at the pinned commit')
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
        raise ValueError('Pinned PROL source is missing required definitions')
    tree = ast.Module(body=nodes, type_ignores=[])
    if transformer is not None:
        tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), 'exec'), namespace)


def _boundary_functions(source, namespace):
    """Use actual main freeze/scaling and actual task prelude, without trainers."""
    main_tree = ast.parse((source / 'main_prol.py').read_text())
    main = next(node for node in main_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    freeze = next(node for node in main.body if isinstance(node, ast.If) and ast.unparse(node.test) == 'args.freeze')
    scale = next(node for node in main.body if isinstance(node, ast.If) and ast.unparse(node.test) == 'args.unscale_lr')
    scale_index = main.body.index(scale)
    engine_tree = ast.parse((source / 'engine_prol.py').read_text())
    train = next(node for node in engine_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'train_and_evaluate')
    task_loop = next(node for node in train.body if isinstance(node, ast.For) and ast.unparse(node.target) == 'task_id')
    epoch_index = next(i for i, node in enumerate(task_loop.body)
                       if isinstance(node, ast.For) and ast.unparse(node.target) == 'epoch')
    functions = [
        ('apply_author_freeze', 'args, original_model, model', [deepcopy(freeze)]),
        ('scale_author_learning_rate', 'args', [deepcopy(scale), deepcopy(main.body[scale_index + 1]),
                                              ast.parse('return args.lr').body[0]]),
        ('_author_task_prelude', 'args, model, task_id, class_mask, old_num_k, aux_desc_emb=None',
         deepcopy(task_loop.body[:epoch_index]) + ast.parse('return optimizer, lr_scheduler').body),
    ]
    for name, signature, body in functions:
        function = ast.parse(f'def {name}({signature}):\n    pass').body[0]
        function.body = body
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(module, str(source / ('engine_prol.py' if name.startswith('_author') else 'main_prol.py')), 'exec'), namespace)


def load_author_code(source, expected=None):
    verify_author_source(source, expected)
    from scripts.cl_runtime import configure_native_environment
    configure_native_environment()
    import numpy as np
    import torch
    from torch import nn, optim
    from torch.nn import functional as F
    from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
    from timm.models._manipulate import named_apply, adapt_input_conv, checkpoint_seq
    from timm.utils import accuracy
    source = Path(source).resolve()
    logger = logging.getLogger('prol_pinned_author')
    logger.setLevel(logging.ERROR)
    namespace = {'__name__': 'prol_pinned_author', 'torch': torch, 'nn': nn, 'F': F, 'optim': optim,
                 'np': np, 'math': math, 'partial': partial, 'copy': SimpleNamespace(deepcopy=deepcopy),
                 'deepcopy': deepcopy, 'logging': logging, '_logger': logger, 'argparse': argparse,
                 'PatchEmbed': PatchEmbed, 'Mlp': Mlp, 'DropPath': DropPath, 'trunc_normal_': trunc_normal_,
                 'lecun_normal_': lecun_normal_, 'named_apply': named_apply, 'adapt_input_conv': adapt_input_conv,
                 'checkpoint_seq': checkpoint_seq, 'time': time, 'Iterable': Iterable, 'sys': sys}
    _definitions(source / 'attention.py', {'PreT_Attention', 'Gen_Attention2'}, namespace)
    _definitions(source / 'promptL.py', {'LPrompt'}, namespace)
    _definitions(source / 'linears.py', {'SimpleContinualLinear'}, namespace)
    _definitions(source / 'vision_transformerL.py', {'Attention', 'LayerScale', 'Block', 'VisionTransformerL',
        'init_weights_vit_timm', 'init_weights_vit_jax', 'init_weights_vit_moco', 'get_init_weights_vit',
        '_load_weights', 'resize_pos_embed'}, namespace)
    _definitions(source / 'configs/cifar100_prol.py', {'get_args_parser'}, namespace)
    utilities = {'__name__': 'prol_pinned_author_utils', 'torch': torch, 'dist': torch.distributed,
                 'defaultdict': defaultdict, 'deque': deque, 'datetime': datetime, 'time': time}
    _definitions(source / 'utils.py', {'SmoothedValue', 'MetricLogger', 'is_dist_avail_and_initialized', 'get_world_size'}, utilities)

    def synchronize():
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    namespace.update(utils=SimpleNamespace(**utilities), accuracy=accuracy, _synchronize=synchronize)
    _definitions(source / 'engine_prol.py', {'train_one_epoch_with_aux'}, namespace, _GuardSynchronize())
    _boundary_functions(source, namespace)
    return SimpleNamespace(**namespace, source=source, adaptations=[
        'Compile immutable author ASTs without timm registration or optional-package imports',
        'Guard CUDA synchronization for synthetic CPU execution only',
        'Scoped nn.Module.cuda identity only for source classifier placement on CPU',
        'Task prelude and backbone freeze are selected unchanged from author function bodies',
        'Use existing local NPZ when requested; no checkpoint download or dataset access'])


def author_args(code, *, scale_lr=False, **overrides):
    parser = argparse.ArgumentParser(add_help=False)
    code.get_args_parser(parser)
    args = parser.parse_args([])
    args.distributed = False
    for key, value in overrides.items():
        setattr(args, key, value)
    if scale_lr:
        code.scale_author_learning_rate(args)
    return args


def build_author_branches(code, checkpoint=None, *, num_classes=100, embed_dim=768, depth=12, num_heads=12,
                          img_size=224, patch_size=16, prompt_length=5, n_tasks=10, top_k_l=1,
                          prompt_layers=None):
    layers = list(range(min(depth, 7))) if prompt_layers is None else list(prompt_layers)
    common = dict(num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                  img_size=img_size, patch_size=patch_size, drop_rate=0., drop_path_rate=0.,
                  head_type='token', global_pool='token')
    # main_prol.py intentionally leaves the query's num_tasks at the source default (5).
    query = code.VisionTransformerL(**common)
    prompt = code.VisionTransformerL(**common, num_tasks=n_tasks, prompt_length=prompt_length,
        prompt_pool=True, prompt_key=True, embedding_key='cls', prompt_init='uniform',
        prompt_key_init='uniform', top_k=1, top_k_l=top_k_l, batchwise_prompt=False,
        use_prompt_mask=True, use_e_prompt=True, e_prompt_layer_idx=layers,
        use_prefix_tune_for_e_prompt=True, use_prefix_tune_for_g_prompt=True,
        g_prompt_length=5, g_prompt_layer_idx=[], prompts_per_task=5)
    branches = {'query': query, 'prompt': prompt}
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file() or checkpoint.suffix != '.npz':
            raise ValueError('Independent PROL requires an existing local NPZ')
        for model in branches.values():
            model.load_pretrained(str(checkpoint))
    code.apply_author_freeze(author_args(code), query, prompt)
    return branches


@contextmanager
def author_cpu_device():
    """Preserve source head.update math; adapt its hardcoded CUDA placement."""
    import torch
    original = torch.nn.Module.cuda

    def cpu_placement(module, device=None):
        if any(parameter.device.type != 'cpu' for parameter in module.parameters()):
            raise ValueError('CPU author placement guard cannot relocate a non-CPU module')
        if device not in (None, 'cpu', torch.device('cpu')):
            raise ValueError('CPU author placement guard received an explicit CUDA device')
        return module

    torch.nn.Module.cuda = cpu_placement
    try:
        yield
    finally:
        torch.nn.Module.cuda = original


def begin_author_task(code, branches, args, task_id, class_mask, old_num_k):
    model = branches['prompt']
    if next(model.parameters()).device.type == 'cpu':
        with author_cpu_device():
            return code._author_task_prelude(args, model, task_id, class_mask, old_num_k)
    return code._author_task_prelude(args, model, task_id, class_mask, old_num_k)


def run_author_epoch(code, branches, batches, optimizer, args, task_id=0, *, class_mask=None, old_num_k=0, epoch=0, criterion=None):
    import torch
    if class_mask is None:
        width = branches['prompt'].num_classes // args.num_tasks
        class_mask = [list(range(task * width, (task + 1) * width)) for task in range(args.num_tasks)]
    return code.train_one_epoch_with_aux(model=branches['prompt'], original_model=branches['query'],
        criterion=torch.nn.CrossEntropyLoss() if criterion is None else criterion, data_loader=batches, aux_desc_emb=None,
        device=next(branches['prompt'].parameters()).device, epoch=epoch, max_norm=args.clip_grad,
        optimizer=optimizer, set_training_mode=True, task_id=task_id, class_mask=class_mask,
        args=args, old_num_k=old_num_k)
