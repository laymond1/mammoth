"""Construct the pinned native EWC-, SD-, Inf- and CL-LoRA networks locally."""
import hashlib
from importlib import import_module
import os
from pathlib import Path

from .pretrained import canonical_pretrained_digest


SOURCE_COMMIT = 'd007ef5c13a6d8a7d7db3eedf93f5e796e2ce429'
SIZES = {'base': (768, 12), 'small': (384, 6), 'tiny': (192, 3)}


def _checkpoint_dir():
    """The torch hub checkpoint cache: TORCH_HOME, else CL_CACHE_ROOT/torch, else ~/.cache/torch."""
    home = os.environ.get('TORCH_HOME')
    if not home:
        cache = os.environ.get('CL_CACHE_ROOT') or os.environ.get('XDG_CACHE_HOME') or str(Path.home() / '.cache')
        home = str(Path(cache) / 'torch')
    return Path(home).expanduser() / 'hub' / 'checkpoints'


_CACHE = _checkpoint_dir()
CHECKPOINTS = {
    'base': {
        'file': str(_CACHE / 'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
        'sha256': 'ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124',
        'canonical_sha256': 'cd8acbba3d33807125c2e30d2d0159f90825208e63a95034ab57ef0af837a300'},
    'small': {
        'file': str(_CACHE / 'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
        'sha256': '545815b4e770d2fa6ca4b3ccba7c16b035e474354e52d17ca197ea4efecbf4d3',
        'canonical_sha256': '1bc2129fcd9c44c0104385f3db58a422595d205fcfff592392124b5f2bf7b1a5'},
    'tiny': {
        'file': str(_CACHE / 'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
        'sha256': '0c5c4818ecf1049ae34194223c908172a2967334bab6d86670351b59d608b5dd',
        'canonical_sha256': '3258efdf29eb69379b559f8884b31134e5f299a2e6cd682f08eb9483e9c8e1cb'},
}
METHODS = {'ewc': 'ewclora', 'sd': 'sdlora', 'inf': 'inflora', 'cl': 'cllora'}


def method_name(method):
    """Accept native CLI names, short method names and original module names."""
    name = str(method).lower().replace('_', '-').removesuffix('-lora')
    if name in METHODS:
        return METHODS[name]
    if name in METHODS.values():
        return name
    raise ValueError(f'Unknown native LoRA method: {method}')


def checkpoint_spec(vit_size='base', checkpoint=None):
    """Verify the agreed existing file before allocating network parameters."""
    if vit_size not in SIZES:
        raise ValueError('LoRA vit_size must be base, small or tiny')
    spec = CHECKPOINTS[vit_size].copy()
    path = Path(checkpoint or spec['file']).expanduser().resolve()
    if not path.is_file() or path.suffix != '.npz':
        raise FileNotFoundError('LoRA requires the existing pinned shared AugReg .npz checkpoint; downloads are disabled')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    if digest.hexdigest() != spec['sha256']:
        raise ValueError(f'LoRA {vit_size} checkpoint differs from the pinned AugReg weights')
    spec['file'] = str(path)
    return spec


def build_network(method, args, vit_size='base', checkpoint=None):
    """Return the original-compatible Net with exact B/S/T encoder weights.

    Construction retains original encoder/head/classifier/adapter RNG order.
    Direct ``Net(args, geometry=..., checkpoint=None)`` construction is available
    for synthetic CPU tests; this production entrypoint always loads the pinned
    complete pretrained encoder and never downloads or creates cache artifacts.
    ``_cur_task`` remains the original Python integer, owned by the native
    continual-model lifecycle and restored by its checkpoint buffers.
    """
    method = method_name(method)
    spec = checkpoint_spec(vit_size, checkpoint)
    dimension, heads = SIZES[vit_size]
    config = dict(init_cls=10, increment=10, sessions=10, rank=10, device=['cpu'],
                  msa=[1, 0, 1], shared_pos=list(range(6)), specific_pos=list(range(6, 12)))
    config.update(args)
    for name in ('init_cls', 'increment', 'sessions', 'rank'):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f'LoRA {name} must be a positive integer')
    if config['init_cls'] != config['increment']:
        raise ValueError('The original LoRA classifiers require equal init_cls and increment')
    if config['rank'] > dimension:
        raise ValueError('LoRA rank must not exceed the encoder dimension')
    if method == 'cllora':
        shared, specific = config['shared_pos'], config['specific_pos']
        if (len(config['msa']) != 3 or any(value not in (0, 1) for value in config['msa'])
                or len(set(shared + specific)) != len(shared + specific)
                or any(type(position) is not int or not 0 <= position < 12 for position in shared + specific)):
            raise ValueError('CL-LoRA shared/specific positions must be disjoint valid blocks and msa must contain three binary switches')
    module = import_module(f'.net_{method}', __package__)
    network = module.Net(config, geometry=dict(patch_size=16, embed_dim=dimension, depth=12, num_heads=heads), checkpoint=spec['file'])
    if canonical_pretrained_digest(network) != spec['canonical_sha256']:
        raise ValueError('LoRA canonical pretrained encoder differs from the pinned tensor digest')
    network.pretrained_checkpoint = spec['file']
    network.pretrained_model = f'vit_{vit_size}_patch16_224.augreg_in21k_ft_in1k'
    network.pretrained_sha256 = spec['sha256']
    network.canonical_pretrained_sha256 = spec['canonical_sha256']
    network.vit_size = vit_size
    return network
