"""Opt-in independent pinned-author HiDe-Prompt reference on shared local data.

The default only describes the run. --prepare-initial-state constructs CPU
models and exports learned initialization, without training or external writes.
--execute owns the authorized GPU run, local telemetry, W&B and Slack alerts.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import copy
import datetime
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
import subprocess
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import Recorder, flatten, model_snapshot, notify_slack, write_json
from scripts.hide_reference_utils import (REFERENCE_COMMIT, PairLoader, build_author_branches,
    file_hash, learned_state, learned_state_digest, load_author_code, load_initial_state,
    save_initial_state, set_author_phase, state_digest, verify_author_source)
from scripts.hide_protocol import CHECKPOINT, CHECKPOINT_SHA256, CANONICAL_PRETRAINED_SHA256, phase_seed, protocol


def source_spec(options):
    """Validate all local pins before any training, credentials or network writes."""
    settings = protocol(options.mode)
    source = options.reference_source.resolve()
    hashes = verify_author_source(source)
    if file_hash(options.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Reference pretrained NPZ differs from the approved local pin')
    if options.initial_state is None or not options.initial_state.is_file():
        raise ValueError('Reference execution requires the exported matched learned initialization')
    return {'schema_version': 1, 'model': 'hide-prompt', 'implementation': 'pinned_author_reference',
            'mode': options.mode, 'protocol': settings, 'reference_commit': REFERENCE_COMMIT,
            'reference_source': str(source), 'reference_source_sha256': hashes,
            'checkpoint_file': str(options.checkpoint.resolve()), 'checkpoint_sha256': CHECKPOINT_SHA256,
            'canonical_pretrained_sha256': CANONICAL_PRETRAINED_SHA256,
            'initial_state': str(options.initial_state.resolve()), 'initial_state_sha256': file_hash(options.initial_state),
            'training_authorized': True, 'accuracy_acceptance_eligible': settings['accuracy_acceptance_eligible'],
            'paper_reproduction': False,
            'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                                         text=True, check=True).stdout.strip(),
            'execution_source_sha256': {
                name: file_hash(ROOT / name) for name in ('scripts/run_hide_reference.py',
                    'scripts/hide_reference_utils.py', 'scripts/hide_protocol.py', 'scripts/cl_efficiency.py')}}


def prepare_initial(options):
    """Actual author CPU initialization; no data reads, forwards or optimizer."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    from scripts.cl_runtime import configure_native_environment
    configure_native_environment()
    verify_author_source(options.reference_source)
    if file_hash(options.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Initialization requires the approved local Base NPZ')
    import torch
    import numpy as np
    import random
    from scripts.ease_size_protocol import canonical_pretrained_digest
    torch.set_num_threads(2)
    torch.manual_seed(1993)
    np.random.seed(1993)
    random.seed(1993)
    code = load_author_code(options.reference_source)
    branches = build_author_branches(code, options.checkpoint)
    canonical = {name: canonical_pretrained_digest(module) for name, module in branches.items()}
    if any(value != CANONICAL_PRETRAINED_SHA256 for value in canonical.values()):
        raise RuntimeError('Author CPU pretrained tensors differ from the canonical pin')
    saved = save_initial_state(branches, options.prepare_initial_state, CHECKPOINT_SHA256)
    write_json(options.prepare_initial_state.with_suffix('.json'), {
        **saved, 'file': str(options.prepare_initial_state.resolve()),
        'file_sha256': file_hash(options.prepare_initial_state), 'canonical_pretrained_sha256': canonical,
        'origin': 'Actual immutable author VisionTransformer constructors + local Base NPZ',
        'training_executed': False, 'image_forwards': 0, 'optimizer_steps': 0,
        'gpu_used': False, 'new_data_or_weights_downloaded': False})
    print(json.dumps({'initial_state': str(options.prepare_initial_state), 'learned_sha256': saved['learned_sha256']}))
    return 0


@contextmanager
def observed_optimizer(optimizer, branches, checks, step_counter=None):
    """Inspect the gradients that the real source optimizer is about to use."""
    import torch
    def before(subject, args, kwargs):
        observed = False
        for module in branches.values():
            for parameter in module.parameters():
                if parameter.grad is None:
                    continue
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError('Author gradient is nonfinite')
                if not parameter.requires_grad:
                    raise RuntimeError('Author frozen backbone acquired a gradient')
                observed = True
        if not observed:
            raise RuntimeError('Author optimizer has no gradients')
        checks['finite_loss_and_gradients'] = True
        checks['frozen_backbone_gradients_absent'] = True
        if step_counter is not None:
            step_counter[0] += 1
    hook = optimizer.register_step_pre_hook(before)
    try:
        yield
    finally:
        hook.remove()


def per_class_loaders(dataset, settings, classes):
    """Match native sequential class-specific batches before ViT forwarding."""
    import torch
    from torch.utils.data import DataLoader, Subset
    source = dataset.train_loader
    view = copy.copy(source.dataset)
    if hasattr(source.dataset, 'dataset'):
        object.__setattr__(view, 'dataset', copy.copy(source.dataset.dataset))
    view.transform = dataset.TEST_TRANSFORM
    targets = torch.as_tensor(view.targets, dtype=torch.long)
    loaders = {}
    for class_id in classes:
        indices = torch.where(targets == class_id)[0].tolist()
        if not indices:
            raise ValueError('Every current class requires statistics samples')
        loader = DataLoader(Subset(view, indices), batch_size=settings['batch_size'], shuffle=False,
                            num_workers=settings['num_workers'], drop_last=False, pin_memory=source.pin_memory)
        loaders[class_id] = {'train': PairLoader(loader)}
    return loaders


def author_args(settings, branch):
    return types.SimpleNamespace(seed=settings['seed'], batch_size=settings['batch_size'],
        nb_classes=100, num_tasks=10, distributed=False, world_size=1, print_freq=1000,
        epochs=settings['tii_epochs'] if branch == 'tii' else settings['prompt_epochs'],
        train_mask=True, task_inc=False, prompt_momentum=.01, reg=.1,
        ca_storage_efficient_method='multi-centroid', n_centroids=10, crct_epochs=settings['ca_epochs'],
        model='vit_base_patch16_224', ca_lr=.005 * (settings['batch_size'] / 256 if branch == 'tii' else 1.),
        weight_decay=0.)


def evaluate_author(branches, dataset, task_id, recorder, output):
    """Author routing arithmetic, with full-seen mask and stable task batching."""
    import numpy as np
    import torch
    from scripts.hide_acceptance import counts_from_predictions
    set_author_phase(branches, 'eval')
    tii, prompt = branches['tii'], branches['prompt']
    predictions, labels, routing, tii_predictions = [], [], [], []
    device = next(tii.parameters()).device
    seen = (task_id + 1) * 10
    with torch.no_grad():
        for loader in dataset.test_loaders:
            for batch in loader:
                images = batch[0].to(device)
                targets = batch[1].to(device)
                tii_logits = tii(images)['logits'][:, :seen]
                prompt_ids = (tii_logits.argmax(1) // 10).unsqueeze(1)
                result = prompt(images, task_id=task_id, prompt_id=prompt_ids)
                if not torch.equal(result['prompt_idx'], prompt_ids):
                    raise RuntimeError('Author routing did not select the inferred prompt IDs')
                logits = result['logits'][:, :seen]
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError('Author seen-class inference logits are nonfinite')
                predictions.append(logits.argmax(1).cpu().numpy())
                labels.append(targets.cpu().numpy())
                routing.append(prompt_ids[:, 0].cpu().numpy())
                tii_predictions.append(tii_logits.argmax(1).cpu().numpy())
                recorder.evaluation_examples += len(targets)
    predictions, labels, routing = np.concatenate(predictions), np.concatenate(labels), np.concatenate(routing)
    path = output / f'task{task_id + 1}_predictions.npz'
    global_ids = np.concatenate([loader.dataset.indexes for loader in dataset.test_loaders])
    order = np.argsort(global_ids, kind='stable')
    np.savez(path, predictions=predictions[order, None], labels=labels[order], inferred_tasks=routing[order],
             global_ids=global_ids[order], evaluation_global_ids=global_ids,
             tii_predictions=np.concatenate(tii_predictions)[order])
    counts = counts_from_predictions(predictions, labels, past_classes=task_id * 10, seen_classes=seen)
    return {'task': task_id + 1, **counts,
            'seen_accuracy_pct': 100 * counts['seen_correct'] / counts['seen_samples'],
            'old_accuracy_pct': 100 * counts['old_correct'] / counts['old_samples'] if counts['old_samples'] else None,
            'new_accuracy_pct': 100 * counts['new_correct'] / counts['new_samples'],
            'predictions_file': str(path), 'predictions_sha256': file_hash(path),
            'routing_correct': int((routing == labels // 10).sum()), 'routing_samples': len(labels)}


def checkpoint_round_trip(branches, code, task_id, output):
    """Serialize sufficient learned/statistical state and restore actual modules."""
    import torch
    state = {name: learned_state(module, name) for name, module in branches.items()}
    hashes = {name: state_digest(values) for name, values in state.items()}
    statistics = {name: {'cls_mean': engine['cls_mean'], 'cls_cov': engine['cls_cov']}
                  for name, engine in code.engines.items()}
    path = output / f'task{task_id + 1}_checkpoint.pt'
    torch.save({'schema_version': 1, 'task_id': task_id, 'class_to_task': torch.arange(100) // 10,
                'checkpoint_sha256': CHECKPOINT_SHA256, 'reference_commit': REFERENCE_COMMIT,
                'branches': state, 'statistics': statistics}, path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['task_id'] != task_id or saved['checkpoint_sha256'] != CHECKPOINT_SHA256:
        raise RuntimeError('Author checkpoint metadata did not survive serialization')
    for name, module in branches.items():
        module.load_state_dict(saved['branches'][name], strict=False)
        if learned_state_digest(module, name) != hashes[name]:
            raise RuntimeError('Author learned checkpoint did not round trip exactly')
        for field in ('cls_mean', 'cls_cov'):
            old = statistics[name][field]
            loaded = saved['statistics'][name][field]
            if set(old) != set(loaded) or any(not torch.equal(a.cpu(), b.cpu())
                    for key in old for a, b in zip(old[key], loaded[key])):
                raise RuntimeError('Author statistics checkpoint did not round trip exactly')
    return {'file': str(path), 'sha256': file_hash(path), 'learned_sha256': hashes,
            'pretrained_storage': 'Shared immutable NPZ, referenced by verified SHA256'}


def run(options, profile):
    spec = source_spec(options)
    if profile['gpu'] != '1':
        raise ValueError('Author HiDe reference requires the approved physical GPU1')
    name = 'hide-author-' + options.mode + '-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (options.output / name).resolve()
    output.mkdir(parents=True, exist_ok=False)
    spec.update(name=name, output=str(output), gpu='1', data_root=profile['data_root'],
                wandb_entity=profile['wandb_entity'], wandb_project=profile['wandb_project'])
    write_json(output / 'spec.json', spec)
    os.environ.update(CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1', WANDB_MODE='online', WANDB_DIR=str(output))
    os.environ.pop('MAMMOTH_DEVICE', None)
    recorder = Recorder(output, '1')
    recorder.start()
    record = {**spec, 'status': 'failed', 'task_results': [], 'epoch_results': [], 'checkpoints': [],
              'structural_checks': {}, 'alignment_results': [], 'first_batch_audits': [], 'wandb_logging_status': 'not_started'}
    notifications = []
    def notify(message):
        notifications.append({'message': message, 'status': notify_slack(message)})
        write_json(output / 'notifications.json', notifications)
    notify(f'HiDe-Prompt author {options.mode} started: Base; 2 tasks; batch{spec["protocol"]["batch_size"]}; GPU1')
    wandb = None
    code_status = 1
    download_guard = ExitStack()
    try:
        import numpy as np
        import torch
        import wandb
        from scripts.hide_protocol import seed_phase, configure_epoch_loader, create_dataset, no_cifar_downloads
        from scripts.ease_reference_utils import dataset_fingerprint, shared_cifar_metadata
        from scripts.ease_size_protocol import canonical_pretrained_digest
        from unittest.mock import patch
        # The shared files must exist before any dataset constructor runs.
        shared_data = shared_cifar_metadata(profile['data_root'])
        download_guard.enter_context(no_cifar_downloads())
        torch.set_num_threads(2)
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available():
            raise RuntimeError('The approved author reference requires CUDA')
        device = torch.device('cuda:0')
        recorder.torch, recorder.device = torch, device
        settings = spec['protocol']
        seed_phase(1993, 0, 'tii_train')
        author = load_author_code(options.reference_source)
        author.adaptations.append('Class-IL evaluation restricts final prompt logits to seen classes, matching Mammoth; pinned author default evaluates all 100 logits')
        with recorder.phase('initialize'):
            branches = build_author_branches(author, options.checkpoint)
            initial = load_initial_state(branches, options.initial_state)
            if initial['checkpoint_sha256'] != CHECKPOINT_SHA256:
                raise ValueError('Matched initialization uses a different pretrained NPZ')
            canonical = {name: canonical_pretrained_digest(module) for name, module in branches.items()}
            if any(value != CANONICAL_PRETRAINED_SHA256 for value in canonical.values()):
                raise RuntimeError('Author pretrained weights differ from the approved canonical tensors')
            for module in branches.values():
                module.to(device)
            dataset = create_dataset(options.mode, options.checkpoint, profile['data_root'])
        record.update(initialization=initial, initial_learned_sha256=initial['learned_sha256'],
                      initial_canonical_pretrained_sha256=canonical,
                      runtime_adaptations=author.adaptations,
                      versions={name: importlib.metadata.version(name) for name in
                        ('torch', 'torchvision', 'timm', 'numpy', 'scikit-learn', 'wandb')},
                      precision={'parameter_dtype': 'float32', 'autocast': False, 'matmul_precision': 'highest',
                                 'tf32_allowed': False, 'cudnn_tf32_allowed': True,
                                 'cudnn_deterministic': True, 'cudnn_benchmark': False})
        write_json(output / 'shared_data.json', shared_data)
        write_json(output / 'runtime_adaptations.json', {'adaptations': author.adaptations,
                                                        'source_files_sha256': spec['reference_source_sha256']})
        wandb.init(entity=profile['wandb_entity'], project=profile['wandb_project'], name=name,
                   group='hide-base-validation', job_type='author-reference-' + options.mode,
                   config=spec, dir=str(output))
        record['wandb_url'] = wandb.run.url
        checks = record['structural_checks']
        checks.update(matched_learned_initialization=True, matched_phase_rng_and_batching=True,
                      no_future_or_replay_samples=True, routing_without_task_oracle=True)
        class_masks = [list(range(i * 10, (i + 1) * 10)) for i in range(10)]
        task_map = {class_id: class_id // 10 for class_id in range(100)}
        criterion = torch.nn.CrossEntropyLoss().to(device)
        snapshot_subject = torch.nn.ModuleDict(branches)
        last_alert = time.monotonic()
        for task_id in range(settings['tasks']):
            with recorder.phase('data_loading'):
                train, test = dataset.get_data_loaders()
            if dataset.N_TASKS != 10 or dataset.N_CLASSES_PER_TASK != 10:
                raise ValueError('Reference data must retain 10 tasks of 10 classes')
            source = train.dataset
            recorder.data_snapshots[f'task{task_id + 1}_train'] = dataset_fingerprint(
                source.data, source.targets, source.indexes, source.transform)
            if recorder.data_snapshots[f'task{task_id + 1}_train']['class_counts'] != {
                    str(class_id): 500 for class_id in class_masks[task_id]}:
                raise RuntimeError('Author reference training includes old, future or missing samples')
            images = np.concatenate([loader.dataset.data for loader in dataset.test_loaders])
            labels = np.concatenate([loader.dataset.targets for loader in dataset.test_loaders])
            ids = np.concatenate([loader.dataset.indexes for loader in dataset.test_loaders])
            order = np.argsort(ids, kind='stable')
            recorder.data_snapshots[f'task{task_id + 1}_test'] = dataset_fingerprint(
                images[order], labels[order], ids[order], dataset.TEST_TRANSFORM)
            if task_id > 0:
                with torch.no_grad():
                    bank = branches['prompt'].e_prompt.prompt
                    bank[:, :, task_id].copy_(bank[:, :, task_id - 1])
            old_bank = branches['prompt'].e_prompt.prompt[:, :, :task_id].detach().clone()
            for branch in ('tii', 'prompt'):
                phase = branch + '_train'
                set_author_phase(branches, branch)
                module = branches[branch]
                args = author_args(settings, branch)
                rate = settings[branch + '_lr'] * settings['batch_size'] / 256.
                if branch == 'tii':
                    groups = [{'params': [p for p in module.parameters() if p.requires_grad], 'lr': rate}]
                else:
                    groups = [{'params': [module.e_prompt.prompt], 'lr': rate},
                              {'params': list(module.head.parameters()), 'lr': rate * .1}]
                optimizer = torch.optim.Adam(groups, lr=rate, weight_decay=0., betas=(.9, .999), eps=1e-8)
                scheduler = None
                if branch == 'prompt':
                    from timm.scheduler import StepLRScheduler
                    scheduler = StepLRScheduler(optimizer, decay_t=30, decay_rate=.1, warmup_t=0)
                for epoch in range(args.epochs):
                    seed_phase(1993, task_id, phase, epoch)
                    configure_epoch_loader(train, 1993, task_id, phase, epoch)
                    audited_batches = [0]
                    def first_batch_audit(images, targets, loss, current_model, current_optimizer):
                        audited_batches[0] += 1
                        if epoch != 0 or audited_batches[0] != 1:
                            return
                        gradients = {branch + '.' + key: parameter.grad for key, parameter in current_model.named_parameters()
                                     if parameter.requires_grad and parameter.grad is not None}
                        record['first_batch_audits'].append({
                            'task': task_id + 1, 'phase': phase, 'epoch': 1,
                            'loss': loss.item(), 'samples': len(targets),
                            'seed': phase_seed(1993, task_id, phase),
                            'input_sha256': state_digest({'inputs': images}),
                            'labels_sha256': state_digest({'labels': targets}),
                            'clipped_gradient_sha256': state_digest(gradients)})
                        write_json(output / 'first_batch_audits.json', record['first_batch_audits'])
                    author.engines[branch]['_audit_batch'] = first_batch_audit
                    pair_loader = PairLoader(train, recorder, max_batches=2 if options.mode == 'smoke' else None)
                    inputs = dict(model=module, criterion=criterion, data_loader=pair_loader, optimizer=optimizer,
                                  device=device, epoch=epoch, max_norm=1., set_training_mode=True,
                                  task_id=task_id, class_mask=class_masks, args=args)
                    if branch == 'prompt':
                        inputs.update(original_model=branches['tii'], target_task_map=task_map)
                    with recorder.phase('train_epoch'), recorder.phase(phase), observed_optimizer(optimizer, branches, checks):
                        stats = author.engines[branch]['train_one_epoch'](**inputs)
                    row = {'task': task_id + 1, 'phase': phase, 'epoch': epoch + 1, 'epochs': args.epochs,
                           'loss': stats['Loss'], 'lr': optimizer.param_groups[0]['lr'], 'training_top1_pct': stats['Acc@1']}
                    record['epoch_results'].append(row)
                    snapshot_subject.optimizer = optimizer
                    wandb.log(flatten({'epoch': row}))
                    write_json(output / 'epoch_results.json', record['epoch_results'])
                    write_json(output / 'progress.json', {'task': task_id + 1, 'phase': phase,
                        'epoch': epoch + 1, 'epochs': args.epochs, 'wandb_url': record['wandb_url']})
                    if scheduler is not None:
                        scheduler.step(epoch)
                    if time.monotonic() - last_alert >= 600:
                        notify(f'HiDe author {options.mode}: task{task_id + 1}/2 {phase} epoch{epoch + 1}/{args.epochs}; W&B {record["wandb_url"]}')
                        last_alert = time.monotonic()
                if branch == 'prompt' and task_id > 0:
                    with recorder.phase('prompt_finalize'), torch.no_grad():
                        bank = module.e_prompt.prompt
                        bank[:, :, task_id].copy_(.99 * bank[:, :, task_id].detach().clone()
                            + .01 * bank[:, :, :task_id].detach().clone().mean(dim=2))
                seed_phase(1993, task_id, branch + '_statistics')
                cls_loaders = per_class_loaders(dataset, settings, class_masks[task_id])
                kwargs = dict(model=module, data_loader=cls_loaders, device=device,
                              class_mask=class_masks[task_id], args=args)
                if branch == 'prompt':
                    kwargs['task_id'] = task_id
                with recorder.phase(branch + '_statistics'):
                    author.engines[branch]['_compute_mean'](**kwargs)
                means, variances = author.engines[branch]['cls_mean'], author.engines[branch]['cls_cov']
                if set(means) != set(range((task_id + 1) * 10)) or any(len(means[key]) != 10 for key in means):
                    raise RuntimeError('Author centroid statistics have incorrect class or centroid counts')
                if any(value.shape != torch.Size([768]) or not bool(torch.isfinite(value).all())
                       for values in (*means.values(), *variances.values()) for value in values):
                    raise RuntimeError('Author centroid statistics have invalid shape or values')
                set_author_phase(branches, branch + '_alignment')
                seed_phase(1993, task_id, branch + '_alignment')
                alignment_steps = [0]
                if task_id > 0:
                    original_sgd = torch.optim.SGD
                    hooks = []
                    def observed_sgd(*positional, **named):
                        subject = original_sgd(*positional, **named)
                        context = observed_optimizer(subject, branches, checks, alignment_steps)
                        context.__enter__()
                        hooks.append(context)
                        return subject
                    try:
                        with patch.object(torch.optim, 'SGD', observed_sgd), recorder.phase(branch + '_alignment'):
                            author.engines[branch]['train_task_adaptive_prediction'](
                                module, args, device, class_masks, task_id)
                    finally:
                        for context in reversed(hooks):
                            context.__exit__(None, None, None)
                expected_steps = settings['ca_epochs'] * task_id * 10
                if alignment_steps[0] != expected_steps:
                    raise RuntimeError('Author classifier alignment optimizer schedule is incomplete')
                record['alignment_results'].append({'task': task_id + 1, 'branch': branch,
                    'status': 'not_applicable' if task_id == 0 else 'completed',
                    'epochs': settings['ca_epochs'] if task_id > 0 else 0,
                    'optimizer_steps': alignment_steps[0], 'lr': args.ca_lr})
                recorder.snapshots[f'task{task_id + 1}_{branch}_completed'] = model_snapshot(snapshot_subject)
            if not torch.equal(old_bank, branches['prompt'].e_prompt.prompt[:, :, :task_id]):
                raise RuntimeError('Author changed an old prompt bank slot')
            if any(canonical_pretrained_digest(module) != canonical[name] for name, module in branches.items()):
                raise RuntimeError('Author modified frozen pretrained weights')
            seed_phase(1993, task_id, 'evaluation')
            with recorder.phase('evaluation'):
                row = evaluate_author(branches, dataset, task_id, recorder, output)
            record['task_results'].append(row)
            with recorder.phase('checkpoint'):
                record['checkpoints'].append(checkpoint_round_trip(branches, author, task_id, output))
            recorder.tasks_completed += 1
            wandb.log(flatten({'task': row, 'alignment': record['alignment_results'][-1]}))
            write_json(output / 'task_results.json', record['task_results'])
            notify(f'HiDe author {options.mode}: task{task_id + 1}/2 completed; seen accuracy {row["seen_accuracy_pct"]:.2f}%')
        expected_steps = settings['tasks'] * (settings['tii_epochs'] + settings['prompt_epochs']) * (2 if options.mode == 'smoke' else 79)
        if recorder.steps != expected_steps:
            raise RuntimeError('Author reference did not consume the complete outer training schedule')
        checks.update(old_prompt_slots_unchanged=True, frozen_backbones_unchanged=True,
                      statistics_shapes_and_seen_classes=True, alignment_schedule_complete=True,
                      checkpoint_round_trip=True, training_protocol_complete=True,
                      dataset_matches_reference=True)
        record['status'], code_status = 'completed', 0
    except BaseException as error:
        record['error_type'] = type(error).__name__
        logging.exception('Author HiDe reference failed')
    finally:
        download_guard.close()
        record = recorder.save(record)
        if wandb is not None and wandb.run is not None:
            try:
                wandb.run.summary.update(flatten({'status': record['status'], 'resources': record['metrics'],
                    'structural_checks': record['structural_checks']}))
                artifact = wandb.Artifact(name, type='cl-hide-author-reference')
                write_json(output / 'artifact_metrics.json', record)
                artifact.add_file(str(output / 'artifact_metrics.json'), name='metrics.json')
                artifact_files = {'metrics.json': {'local_file': 'artifact_metrics.json',
                                                  'sha256': file_hash(output / 'artifact_metrics.json')}}
                for filename in ('spec.json', 'runtime_adaptations.json', 'epoch_results.json',
                    'task_results.json', 'shared_data.json', 'telemetry.csv', 'phases.json', 'first_batch_audits.json',
                    'task1_predictions.npz', 'task2_predictions.npz', 'task1_checkpoint.pt', 'task2_checkpoint.pt'):
                    if (output / filename).is_file():
                        artifact.add_file(str(output / filename), name=filename)
                        artifact_files[filename] = {'local_file': filename, 'sha256': file_hash(output / filename)}
                write_json(output / 'artifact_files.json', artifact_files)
                artifact.add_file(str(output / 'artifact_files.json'), name='artifact_files.json')
                logged = wandb.run.log_artifact(artifact)
                logged.wait()
                record['wandb_artifact'] = {'name': logged.qualified_name, 'state': logged.state}
                record['wandb_logging_status'] = 'uploaded'
            except Exception as error:
                record['wandb_logging_status'] = 'failed:' + type(error).__name__
                record['status'], code_status = 'logging_failed', 1
            finally:
                try:
                    wandb.finish(exit_code=code_status)
                except Exception as error:
                    record['wandb_logging_status'] = 'finish_failed:' + type(error).__name__
                    record['status'], code_status = 'logging_failed', 1
        if record['status'] == 'completed' and record['wandb_logging_status'] != 'uploaded':
            record['status'], code_status = 'logging_failed', 1
        write_json(output / 'result.json', record)
        write_json(output / 'metrics.json', record)
        notify(f'HiDe author {options.mode} {record["status"]}; logging={record["wandb_logging_status"]}; W&B {record.get("wandb_url") or "unavailable"}')
        print(json.dumps({'status': record['status'], 'result': str(output / 'result.json'), 'wandb_url': record.get('wandb_url')}))
    return code_status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--mode', choices=('validation', 'smoke'), default='validation')
    parser.add_argument('--reference-source', type=Path, default=Path('/tmp/cl-hide-official-20261007'))
    parser.add_argument('--checkpoint', type=Path, default=Path(CHECKPOINT))
    parser.add_argument('--initial-state', type=Path)
    parser.add_argument('--prepare-initial-state', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/hide_prompt/runs')
    options = parser.parse_args(argv)
    if options.prepare_initial_state is not None:
        if options.execute:
            raise ValueError('Prepare learned initialization separately from execution')
        return prepare_initial(options)
    if not options.execute:
        print(json.dumps({'prepared': True, 'executed': False, 'protocol': protocol(options.mode)}))
        return 0
    from scripts.cl_runtime import configure_environment
    profile = configure_environment()
    logging.basicConfig(level=logging.INFO)
    return run(options, profile)


if __name__ == '__main__':
    sys.exit(main())
