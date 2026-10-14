"""Pre-train ToCom banks on ImageNet-1K for the plug-in's token reduction operators.

One process reproduces the official 8-GPU recipe in data order and updates:
every optimizer step consumes eight 128-image microbatches, drawn from the
eight repeated-augmentation replicas of the official launcher, each with its
own random (source, target) pair. The student runs at ``target`` with the bank
active and is distilled towards the same frozen model at ``source``. Several
operators train side by side on identical batches, one model pair per
operator, optionally on different devices.

Run from the repository root:
    uv run --no-sync python -m scripts.tocom.train_bank --weights augreg \\
        --operators tome cls-prune --devices cuda:0 cuda:1 --output <dir> --cache-dir <nvme dir>
"""

from scripts.cl_runtime import DEFAULT_PROFILE, configure_native_environment

configure_native_environment()

import argparse  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402
import timm  # noqa: E402
from timm.data import Mixup, create_transform  # noqa: E402
from timm.scheduler.cosine_lr import CosineLRScheduler  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402

from backbone.token_compression import MAX_DEGREE, OPERATORS, TrainableToCom, attach  # noqa: E402
from backbone.token_compression.compensator import OFFICIAL_COMMIT, file_sha256  # noqa: E402
from backbone.token_compression.reference import augreg_reference, deit_reference  # noqa: E402
from scripts.tocom import image_cache as CACHE  # noqa: E402

# Official launcher: 8 processes x 128 images per update, 3 repeated augmentations.
REPLICAS, MICROBATCH, REPEATS = 8, 128, 3
TRAIN_IMAGES = 1281167


class ImageNetSynsets(ImageFolder):
    """Exclude extraction/archive directories without modifying shared data."""

    def find_classes(self, directory):
        classes = sorted(p.name for p in Path(directory).iterdir()
                         if p.is_dir() and re.fullmatch(r'n[0-9]{8}', p.name))
        if len(classes) != 1000:
            raise ValueError(f'Expected 1000 ImageNet synsets, found {len(classes)}')
        return classes, {name: i for i, name in enumerate(classes)}


def distillation_loss(student, teacher, temperature=1.0):
    # The official loss divides by batch * number of classes, not just batch.
    return F.kl_div(F.log_softmax(student.float() / temperature, dim=1),
                    F.log_softmax(teacher.float() / temperature, dim=1),
                    reduction='sum', log_target=True) * temperature**2 / student.numel()


def epoch_microbatches(length, epoch):
    """Per update, the microbatch each official RASampler replica would yield."""
    generator = torch.Generator()
    generator.manual_seed(epoch)
    indices = torch.repeat_interleave(torch.randperm(length, generator=generator), REPEATS).tolist()
    total = math.ceil(length * REPEATS / REPLICAS) * REPLICAS
    indices += indices[:total - len(indices)]
    selected = length // 256 * 256 // REPLICAS
    streams = [indices[rank:total:REPLICAS][:selected] for rank in range(REPLICAS)]
    return [[stream[step * MICROBATCH:(step + 1) * MICROBATCH] for stream in streams]
            for step in range(selected // MICROBATCH)]


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(payload, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(temporary, path)


class Track:
    """Teacher, student and bank of one operator."""

    def __init__(self, operator, device, build, args):
        self.operator, self.device = operator, torch.device(device)
        self.teacher = build(0.).requires_grad_(False).eval().to(self.device)
        self.student = build(args.drop_path).requires_grad_(False).train().to(self.device)
        self.teacher_controller = attach(self.teacher, operator)
        self.student_controller = attach(self.student, operator)
        self.canonical_sha256 = self.teacher.canonical_pretrained_sha256
        self.bank = TrainableToCom(len(self.student.blocks), self.student.embed_dim, rank=2, scale=args.scale,
                                   max_degree=args.max_degree)
        self.degrees = range(args.max_degree + 1)
        self.bank.to(self.device).train().install(self.student)
        self.optimizer = torch.optim.AdamW(self.bank.parameters(), lr=0.001 * REPLICAS * MICROBATCH / 512,
                                           betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)
        self.scheduler = CosineLRScheduler(self.optimizer, t_initial=args.epochs, lr_min=1e-5,
                                           warmup_t=0, warmup_lr_init=1e-6, t_in_epochs=True)
        self.scaler = torch.amp.GradScaler('cuda')
        # Pairs depend on the operator name only, so adding a track leaves the others unchanged.
        self.pairs = random.Random(args.seed * 2**32 + zlib.crc32(operator.encode()))
        self.loss_sum, self.successful = 0., 0

    def micro(self, images):
        source, target = self.pairs.sample(self.degrees, 2)
        images = images.to(self.device, non_blocking=True)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            with self.teacher_controller.degree(source):
                reference = self.teacher(images)
        self.bank.pair = (source, target)
        with torch.autocast('cuda', dtype=torch.float16):
            with self.student_controller.degree(target):
                outputs = self.student(images)
            loss = distillation_loss(outputs, reference)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite distillation loss for {self.operator}')
        self.scaler.scale(loss / REPLICAS).backward()
        self.loss_sum += loss.item()

    def step(self):
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.successful += int(self.scaler.get_scale() >= old_scale)
        self.optimizer.zero_grad(set_to_none=True)
        mean, self.loss_sum = self.loss_sum / REPLICAS, 0.
        return mean

    def weights(self):
        state = {name: value.detach().cpu().clone() for name, value in self.bank.named_parameters()}
        if len(state) != 4 * len(self.student.blocks) or any(not torch.isfinite(v).all() for v in state.values()):
            raise RuntimeError('Unexpected or nonfinite bank weights')
        if not any(torch.count_nonzero(v) for name, v in state.items() if 'adapter_up' in name):
            raise RuntimeError('No trained adapter updates to save')
        return state

    def resume_state(self):
        return {'bank': self.bank.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(), 'scaler': self.scaler.state_dict(),
                'pairs': self.pairs.getstate(), 'successful': self.successful,
                'cuda_rng': torch.cuda.get_rng_state(self.device)}

    def load_resume_state(self, state):
        self.bank.load_state_dict(state['bank'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.scaler.load_state_dict(state['scaler'])
        self.pairs.setstate(state['pairs'])
        self.successful = state['successful']
        torch.cuda.set_rng_state(state['cuda_rng'], self.device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', choices=('augreg', 'deit'), required=True)
    parser.add_argument('--vit-size', choices=('tiny', 'small', 'base'), default='tiny', help='AugReg size')
    parser.add_argument('--pretrained', help='DeiT .pth file, or an AugReg .npz overriding the pinned path')
    parser.add_argument('--operators', nargs='+', choices=tuple(OPERATORS), default=['tome'])
    parser.add_argument('--devices', nargs='+', default=['cuda:0'], help='Assigned to operators in turn')
    parser.add_argument('--data', type=Path, help='ImageNet root containing train/<synset>/')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--drop-path', type=float, default=.1, help='Student stochastic depth (official: 0.1)')
    parser.add_argument('--scale', type=float, default=.1)
    parser.add_argument('--max-degree', type=int, default=MAX_DEGREE,
                        help='Degrees 0 to this value are sampled and covered by the bank (official: 16)')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--max-updates', type=int, default=0, help='Nonzero makes a labelled partial/smoke run')
    parser.add_argument('--resume', action='store_true', help='Continue from <output>/resume.pth')
    parser.add_argument('--cache-dir', type=Path, help='Private directory on NVMe for original image bytes')
    parser.add_argument('--cache-gib', type=float, default=55)
    parser.add_argument('--cache-reserve-gib', type=float, default=15)
    parser.add_argument('--cache-workers', type=int, default=8)
    args = parser.parse_args(argv)
    if args.weights == 'deit' and not args.pretrained:
        parser.error('--weights deit requires --pretrained')
    if len(set(args.operators)) != len(args.operators):
        parser.error('Operators must be distinct')
    if min(args.epochs, args.save_every, args.log_every) < 1 or args.max_updates < 0 or args.workers < 0:
        parser.error('Epochs and intervals must be positive')
    if not 0 <= args.drop_path < 1 or not math.isfinite(args.scale) or args.scale <= 0:
        parser.error('drop-path must be in [0, 1) and scale positive')
    if not 1 <= args.max_degree <= MAX_DEGREE:
        parser.error(f'max-degree must be in [1, {MAX_DEGREE}]')
    return args


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('Bank pre-training requires CUDA')
    args.output.mkdir(parents=True, exist_ok=True)
    resume_path = args.output / 'resume.pth'
    if (args.output / 'run.json').exists() != args.resume:
        raise FileExistsError('Use a new output directory, or --resume for an existing run')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    # Match official autocast precision rather than silently changing to TF32/BF16.
    torch.backends.cuda.matmul.allow_tf32 = False

    def build(drop_path):
        if args.weights == 'augreg':
            return augreg_reference(args.vit_size, args.pretrained, drop_path_rate=drop_path)
        return deit_reference(args.pretrained, drop_path_rate=drop_path)

    tracks = [Track(operator, args.devices[index % len(args.devices)], build, args)
              for index, operator in enumerate(args.operators)]
    first = tracks[0]
    mean, std = first.teacher.normalization
    data = args.data or Path(json.loads(DEFAULT_PROFILE.read_text())['data_root']) / 'imagenet'
    transform = create_transform(input_size=224, is_training=True, color_jitter=0.3,
                                 auto_augment='rand-m9-mstd0.5-inc1', interpolation='bicubic',
                                 re_prob=0.25, re_mode='pixel', re_count=1, mean=mean, std=std)
    dataset = ImageNetSynsets((data / 'train').resolve(), transform=transform)
    if len(dataset) != TRAIN_IMAGES:
        raise ValueError(f'Expected all 1,281,167 ImageNet training images, found {len(dataset)}')
    source_loader = dataset.loader
    mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
                  mode='batch', label_smoothing=0.1, num_classes=1000)
    updates_per_epoch = len(epoch_microbatches(len(dataset), 0))

    contract = {'weights': args.weights, 'vit_size': args.vit_size, 'operators': args.operators,
                'epochs': args.epochs, 'seed': args.seed, 'drop_path': args.drop_path, 'scale': args.scale,
                'max_degree': args.max_degree,
                'backbone_canonical_sha256': first.canonical_sha256}
    start_epoch, start_step, global_step = 0, 0, 0
    if args.resume:
        state = torch.load(resume_path, map_location='cpu', weights_only=False)
        if state['contract'] != contract:
            raise ValueError('Resume must use the same weights, operators, schedule and seed')
        for track in tracks:
            track.load_resume_state(state['tracks'][track.operator])
        start_epoch, start_step, global_step = state['next_epoch'], state['next_step'], state['global_step']
        torch.set_rng_state(state['rng']['torch'])
        np.random.set_state(state['rng']['numpy'])
        random.setstate(state['rng']['python'])
    else:
        atomic_json({'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                     'contract': contract, 'official_commit': OFFICIAL_COMMIT,
                     'trainer_sha256': file_sha256(__file__), 'torch': str(torch.__version__), 'timm': timm.__version__,
                     'devices': {track.operator: torch.cuda.get_device_name(track.device) for track in tracks},
                     'training_images': len(dataset), 'global_batch_size': REPLICAS * MICROBATCH,
                     'updates_per_epoch': updates_per_epoch, 'initial_lr': 0.002, 'rank_per_gap': 2,
                     'degree_range': [0, args.max_degree], 'normalization': [mean, std],
                     'data_access': 'ImageNet-1K train only; no downstream CL data or validation images',
                     'differences_from_official_launcher': [
                         'one process emulating the 8 repeated-augmentation replicas with gradient accumulation',
                         'timm 0.9.8 augmentation and the plug-in ViT instead of timm 0.4.12 with the ToMe patch',
                         'seed Python RNG', 'omit unused model EMA', 'resumable periodic adapter-only checkpoints'],
                     'started_at_unix': time.time()}, args.output / 'run.json')

    cache_config = ({'directory': str(args.cache_dir.resolve()), 'max_gib': args.cache_gib,
                     'reserve_gib': args.cache_reserve_gib, 'workers': args.cache_workers} if args.cache_dir else None)
    started, start_global, cache_seconds, cache = time.monotonic(), global_step, 0., None

    def cache_event(record):
        with (args.output / 'cache.jsonl').open('a') as stream:
            stream.write(json.dumps({**record, 'at_unix': time.time()}) + '\n')

    def save(next_epoch, next_step, complete=False):
        progress = {'global_step': global_step, 'next_epoch': next_epoch, 'next_step': next_step,
                    'complete': complete, 'elapsed_seconds_this_launch': time.monotonic() - started,
                    'official_commit': OFFICIAL_COMMIT, 'cache': cache_config}
        for track in tracks:
            folder = args.output / track.operator
            folder.mkdir(exist_ok=True)
            entry = {**progress, 'successful_steps': track.successful}
            atomic_save({'model': track.weights(), 'max_degree': args.max_degree, 'preparation': entry},
                        folder / 'checkpoint.pth')
            atomic_json({'official_commit': OFFICIAL_COMMIT, 'operator': track.operator,
                         'backbone': {'weights': args.weights, 'vit_size': args.vit_size if args.weights == 'augreg' else None,
                                      'canonical_sha256': track.canonical_sha256},
                         'checkpoint_sha256': file_sha256(folder / 'checkpoint.pth'), 'scale': args.scale,
                         'rank_per_gap': 2, 'degree_range': [0, args.max_degree], 'student_drop_path': args.drop_path,
                         'complete': complete,
                         'preparation_data': f'ImageNet-1K train, {len(dataset)} images, {args.epochs} epochs planned',
                         'progress': entry}, folder / 'provenance.json')
        atomic_save({'tracks': {track.operator: track.resume_state() for track in tracks}, 'contract': contract,
                     'next_epoch': next_epoch, 'next_step': next_step, 'global_step': global_step,
                     'rng': {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
                             'python': random.getstate()}}, resume_path)
        atomic_json(progress, args.output / 'status.json')

    pool = ThreadPoolExecutor(len(tracks)) if len({track.device for track in tracks}) > 1 else None
    for epoch in range(start_epoch, args.epochs):
        first_step = start_step if epoch == start_epoch else 0
        steps = epoch_microbatches(len(dataset), epoch)[first_step:]
        batches = [micro for step in steps for micro in step]
        if args.cache_dir:
            if cache is None:
                cache = CACHE.EpochImageCache(dataset.root, args.cache_dir, int(args.cache_gib * 1024**3),
                                              int(args.cache_reserve_gib * 1024**3), args.cache_workers)
            wanted = {index for micro in batches for index in micro}
            cache_event({'event': 'cache_start', 'epoch': epoch + 1, 'requested_images': len(wanted)})
            result = cache.prepare((dataset.samples[index][0] for index in wanted), cache_event)
            cache_event({**result, 'epoch': epoch + 1})
            cache_seconds += result['seconds']
            dataset.loader = CACHE.CachedImageLoader(dataset.root, args.cache_dir,
                                                     CACHE.read_members(args.cache_dir), source_loader)
        loader = DataLoader(dataset, batch_sampler=batches, num_workers=args.workers, pin_memory=True)
        epoch_loss = {track.operator: 0. for track in tracks}
        for track in tracks:
            track.optimizer.zero_grad(set_to_none=True)
        for micro, (images, labels) in enumerate(loader, start=first_step * REPLICAS):
            images, labels = images.to(first.device, non_blocking=True), labels.to(first.device, non_blocking=True)
            images, _ = mixup(images, labels)
            if pool is None:
                for track in tracks:
                    track.micro(images)
            else:
                list(pool.map(lambda track: track.micro(images), tracks))
            if (micro + 1) % REPLICAS:
                continue
            losses = {track.operator: track.step() for track in tracks}
            global_step += 1
            step = (micro + 1) // REPLICAS
            for name, value in losses.items():
                epoch_loss[name] += value
            if global_step == 1 or global_step % args.log_every == 0:
                elapsed = time.monotonic() - started
                record = {'event': 'train', 'epoch': epoch + 1, 'step_in_epoch': step,
                          'updates_per_epoch': updates_per_epoch, 'step': global_step, 'loss': losses,
                          'lr': first.optimizer.param_groups[0]['lr'],
                          'images_per_second': (global_step - start_global) * REPLICAS * MICROBATCH / max(elapsed, 1e-9),
                          'cache_seconds': cache_seconds, 'elapsed_seconds': elapsed}
                print(json.dumps(record), flush=True)
                with (args.output / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
            stop = bool(args.max_updates and global_step >= args.max_updates)
            end_epoch = step == updates_per_epoch
            if end_epoch:
                # The author's launcher calls step(epoch), including step(0) after epoch 1.
                for track in tracks:
                    track.scheduler.step(epoch)
            saved = global_step == 1 or global_step % args.save_every == 0 or stop or end_epoch
            # A bank is only written once every operator has taken a real update.
            if saved and all(track.successful for track in tracks):
                save(epoch + 1 if end_epoch else epoch, 0 if end_epoch else step,
                     complete=end_epoch and epoch + 1 == args.epochs)
            if stop:
                break
        else:
            print(json.dumps({'event': 'epoch_end', 'epoch': epoch + 1,
                              'mean_loss': {name: value / max(len(steps), 1) for name, value in epoch_loss.items()}}),
                  flush=True)
            continue
        break
    if cache is not None:
        cache.close()
    if pool is not None:
        pool.shutdown()


if __name__ == '__main__':
    main()
