"""Pinned CPS author definitions, with offline weights and framework adapters.

No native CPS attention, sampling, prompt or observe implementation is imported.
The six upstream files remain byte-for-byte fixtures; only their imports are
rebound. RNG scoping, checkpoint counters and custom_scheduler are adapters.
"""
import ast
import copy
import hashlib
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

SOURCE_COMMIT = '7ec2be108f4f032f2bb10a6647304076f406ef6b'
SOURCE = Path(__file__).resolve().parents[1] / 'tests/fixtures/cps_author'
PINS = {
    'method.py': 'b634b7f9be58bda5d836c53949c318ceb2f92950d415224d4218021ed9875919',
    'model.py': 'c278fb69bf0b592883b75d30653fad3ec8601f7228a254bbb4598f40819cec06',
    'prompt.py': '13b8f68d9d5cfbed63d066e106672f7c34204b646db10cb9dd21b43294ced00f',
    'vit.py': 'cce514180e36646c95c924a789f7633ef87c87a408991113ea5c637c4392c20e',
    'cps.py': 'e9116b2a0b013d5f66487c98bc49656de50d5681bc18f83cce6f512c46ca906b',
    'cps_vit.py': 'df62a3b2fecf5921f3715ab6dbca708281975954f1c960839a61bcf87fc7c62b',
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source():
    for name, expected in PINS.items():
        if file_hash(SOURCE / name) != expected:
            raise ValueError(f'CPS author source pin mismatch: {name}')
    return dict(PINS)


def _definitions(name, namespace):
    tree = ast.parse((SOURCE / name).read_text(), filename=str(SOURCE / name))
    # Every function/class body is unchanged. Imports are supplied explicitly,
    # avoiding a second Mammoth installation or shadowing native models.*.
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(SOURCE / name), 'exec'), namespace)  # noqa: S102 - verified immutable upstream pins
    return namespace


def author_classes(checkpoint=None, *, geometry=None):
    """Return independent author network/model classes; no training/downloads."""
    verify_source()
    import logging

    import torch
    from timm.layers import DropPath, trunc_normal_
    from timm.models.vision_transformer import PatchEmbed
    from torch import nn

    from datasets import get_dataset
    from models.low_rank_utils.factory import checkpoint_spec
    from models.low_rank_utils.pretrained import (
        canonical_pretrained_state,
        load_pretrained_encoder,
    )
    from models.resource_prompt_utils.state import CheckpointRNG, PromptStateMixin
    from models.utils.continual_model import ContinualModel
    from utils import binary_to_boolean_type
    from utils.args import ArgumentParser
    from utils.schedulers import CosineSchedule

    spec = checkpoint_spec('tiny', checkpoint) if geometry is None else None
    common = {'torch': torch, 'nn': nn, 'F': nn.functional, 'copy': copy, 'partial': partial,
              '__name__': 'scripts.cps_author_reference', 'Tuple': tuple}
    vit = _definitions('vit.py', dict(common, PatchEmbed=PatchEmbed, DropPath=DropPath, trunc_normal_=trunc_normal_))
    prompt = _definitions('prompt.py', dict(common))
    sampling = _definitions('cps.py', dict(common))
    original_sampler = sampling['CriticalPatchSampling']

    class ScopedAuthorSampler(original_sampler):
        """Run original global-RNG sampling with a checkpointed private stream."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.rng = CheckpointRNG(1993)

        def forward(self, inputs, *args, **kwargs):
            generator = self.rng.generator(inputs.device)
            devices = [inputs.device] if inputs.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                if inputs.is_cuda:
                    torch.cuda.set_rng_state(generator.get_state(), inputs.device)
                else:
                    torch.default_generator.set_state(generator.get_state())
                result = super().forward(inputs, *args, **kwargs)
                state = torch.cuda.get_rng_state(inputs.device) if inputs.is_cuda else torch.default_generator.get_state()
                generator.set_state(state)
            return result

    patched = _definitions('cps_vit.py', dict(common, CriticalPatchSampling=ScopedAuthorSampler,
                           **{name: vit[name] for name in ('Attention', 'Block', 'VisionTransformer')}))

    def encoder_constructor(**kwargs):
        kwargs.update(geometry or {})
        return vit['VisionTransformer'](**kwargs)

    def local_pretrained(*args, **kwargs):
        if spec is None:
            raise ValueError('Synthetic reference must set pretrained=False')
        donor = vit['VisionTransformer'](img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3)
        load_pretrained_encoder(donor, spec['file'])
        return SimpleNamespace(state_dict=lambda: canonical_pretrained_state(donor))

    model = _definitions('model.py', dict(common, VisionTransformer=encoder_constructor,
                         CodaPrompt=prompt['CodaPrompt'], cps=SimpleNamespace(apply_patch=patched['apply_patch']),
                         timm=SimpleNamespace(create_model=local_pretrained)))
    original_net = model['PromptModel']

    class ReferenceNet(original_net):
        def get_extra_state(self):
            return {'task_count': self.prompt.task_count}

        def set_extra_state(self, state):
            self.prompt.task_count = state['task_count']

    method = _definitions('method.py', dict(common, logging=logging, get_dataset=get_dataset,
                          ArgumentParser=ArgumentParser, ContinualModel=ContinualModel,
                          PromptModel=ReferenceNet, CosineSchedule=CosineSchedule,
                          binary_to_boolean_type=binary_to_boolean_type))
    original_method = method['CPSPrompt']

    class ReferenceCPS(PromptStateMixin, original_method):
        COMPATIBILITY: ClassVar[list[str]] = ['class-il', 'task-il']

        def begin_task(self, dataset):
            super().begin_task(dataset)
            # Current Mammoth looks up custom_scheduler, upstream used scheduler.
            if self.args.use_scheduler:
                self.custom_scheduler = self.scheduler

    return SimpleNamespace(model=ReferenceCPS, network=ReferenceNet, vit=vit,
                           prompt=prompt, sampler=original_sampler, patched=patched)
