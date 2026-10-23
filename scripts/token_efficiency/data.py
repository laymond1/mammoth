"""ImageNet-val images decoded once into a uint8 array, with the token-ranking study's recipe.

Recipe: resize the short side to 256 (bicubic, antialias), center crop 224 (``scripts/token_rank/data.py``).
``build_val`` runs on the server; devices only receive ``bench_images.npy``, a fixed 256-image
subset used as timing input.
"""
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DATA_ROOT = Path('/mnt/sdab1/share/data/imagenet')
CACHE = Path('/mnt/sdab1/userHome/wslim/.cache/token_efficiency/imagenet-val')
RECIPE = 'resize256-bicubic-antialias_centercrop224'
BENCH_IMAGES = 256


def _transform():
    from torchvision import transforms
    bicubic = transforms.InterpolationMode.BICUBIC
    return transforms.Compose([transforms.Resize(256, interpolation=bicubic, antialias=True), transforms.CenterCrop(224)])


def list_val(root=DATA_ROOT / 'val'):
    """(path, label) for every validation image; labels follow the sorted WordNet-id folders."""
    classes = sorted(path.name for path in Path(root).iterdir() if path.is_dir() and path.name.startswith('n'))
    if len(classes) != 1000:
        raise ValueError(f'Expected 1000 class folders under {root}, found {len(classes)}')
    return [(str(Path(root) / name / file), label) for label, name in enumerate(classes)
            for file in sorted(path.name for path in (Path(root) / name).iterdir())]


_TRANSFORM = None


def _decode(item):
    from PIL import Image
    global _TRANSFORM
    if _TRANSFORM is None:
        _TRANSFORM = _transform()
    with Image.open(item[0]) as image:
        return np.asarray(_TRANSFORM(image.convert('RGB')), dtype=np.uint8)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_val(out=CACHE, workers=8):
    out = Path(out)
    if (out / 'manifest.json').exists():
        return json.loads((out / 'manifest.json').read_text())
    items = list_val()
    out.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(out / 'images.npy.partial', mode='w+', dtype=np.uint8, shape=(len(items), 224, 224, 3))
    with Pool(workers) as pool:
        for index, crop in enumerate(pool.imap(_decode, items, chunksize=64)):
            images[index] = crop
    images.flush()
    del images
    (out / 'images.npy.partial').rename(out / 'images.npy')
    np.save(out / 'labels.npy', np.array([label for _, label in items], dtype=np.int16))
    ids = '\n'.join(str(Path(path).relative_to(DATA_ROOT / 'val')) for path, _ in items)
    manifest = {'count': len(items), 'recipe': RECIPE, 'ids_sha256': hashlib.sha256(ids.encode()).hexdigest(),
                'images_sha256': sha256(out / 'images.npy'), 'labels_sha256': sha256(out / 'labels.npy')}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    return manifest


def bench_indices(count=50000, size=BENCH_IMAGES):
    """Evenly spread validation indices, so the timing input covers all classes."""
    return np.linspace(0, count - 1, size).round().astype(np.int64)


def write_bench_images(cache=CACHE, out=None):
    images = np.load(Path(cache) / 'images.npy', mmap_mode='r')
    target = Path(out or Path(cache).parent / 'bench_images.npy')
    np.save(target, np.ascontiguousarray(images[bench_indices(len(images))]))
    return target


def normalize(images, device='cpu', dtype=None):
    """uint8 (batch, 224, 224, 3) -> (batch, 3, 224, 224) with the AugReg mean 0.5 and std 0.5."""
    import torch
    x = torch.as_tensor(np.ascontiguousarray(images), device=device).permute(0, 3, 1, 2).float().div_(255).sub_(0.5).div_(0.5)
    return x if dtype is None else x.to(dtype)
