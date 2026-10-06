"""Opt-in, two-minibatch EASE Base forward/backward check with local/W&B logs.

This is an implementation smoke check, not an accuracy or efficiency benchmark.
It does not evaluate accuracy, construct prototypes, or run a full task.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import Recorder, flatten, model_snapshot, notify_slack, write_json
from scripts.cl_runtime import configure_environment

CHECKPOINT_NAME = 'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'
CHECKPOINT_SHA256 = 'ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/ease/runs')
    options = parser.parse_args(argv)
    if not options.execute:
        print('Prepared EASE Base two-minibatch smoke; execution=False')
        return 0
    profile = configure_environment()
    os.environ['CUDA_VISIBLE_DEVICES'] = profile['gpu']
    os.environ['HF_HUB_OFFLINE'] = '1'
    checkpoint = Path(profile['cache_root']) / 'torch/hub/checkpoints' / CHECKPOINT_NAME
    if file_hash(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('The selected cached Base checkpoint hash does not match')
    name = 'ease-base-smoke-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = options.output.resolve() / name
    output.mkdir(parents=True, exist_ok=False)
    spec = {'stage': 'implementation_smoke', 'experiment_executed': True, 'full_task_executed': False,
            'accuracy_evaluated': False, 'steps': 2, 'batch_size': 2, 'seed': 1993, 'gpu': profile['gpu'],
            'checkpoint_file': str(checkpoint), 'checkpoint_sha256': CHECKPOINT_SHA256,
            'checkpoint_tag': 'vit_base_patch16_224.augreg_in21k_ft_in1k', 'adapter_size': 64,
            'dataset': 'seq-cifar100-224', 'data_root': profile['data_root']}
    write_json(output / 'spec.json', spec)
    notifications = []

    def notify(message):
        notifications.append({'message': message, 'status': notify_slack(message)})
        write_json(output / 'notifications.json', notifications)

    notify(f'EASE Base implementation smoke started: two minibatches; GPU {profile["gpu"]}; no accuracy evaluation')
    recorder = Recorder(output, profile['gpu'])
    recorder.start()
    record, wandb = {**spec, 'status': 'failed'}, None
    code = 1
    try:
        import torch
        import wandb
        import main as mammoth
        recorder.torch, recorder.device = torch, torch.device('cuda:0')
        wandb.init(entity=profile['wandb_entity'], project=profile['wandb_project'],
                   name=name, group='ease-base-validation', job_type='implementation-smoke',
                   config=spec, dir=str(output))
        record['wandb_url'] = wandb.run.url
        command = ['--model', 'ease', '--dataset', 'seq-cifar100-224', '--model_config', 'default',
                   '--ease_checkpoint', str(checkpoint), '--vit_size', 'base', '--adapter_size', '64',
                   '--ease_pretrain_type', 'in21k_ft_in1k', '--batch_size', '2', '--num_workers', '0',
                   '--n_epochs', '20', '--seed', '1993', '--permute_classes', '0', '--enable_other_metrics', '0',
                   '--eval_future', '0', '--base_path', profile['data_root'], '--device', 'cuda:0',
                   '--results_path', str(output / 'mammoth')]
        sys.argv = ['main.py', *command]
        with recorder.phase('initialize'):
            args = mammoth.parse_args(cmd=command)
            model, dataset, args = mammoth.initialize(args)
        # This standalone diagnostic owns its W&B run; keep native autologging
        # disabled while still calling the native lifecycle/observe wrappers.
        args.nowand = 1
        write_json(output / 'effective_args.json', vars(args))
        with recorder.phase('data_loading'):
            train_loader, _ = dataset.get_data_loaders()
        from scripts.ease_reference_utils import dataset_fingerprint, shared_cifar_metadata
        write_json(output / 'shared_data.json', shared_cifar_metadata(profile['data_root']))
        train_data = train_loader.dataset
        recorder.data_snapshots['task1_train'] = dataset_fingerprint(
            train_data.data, train_data.targets, train_data.indexes, train_data.transform)
        recorder.data_snapshots['task1_test'] = dataset_fingerprint(
            dataset.test_loaders[0].dataset.data, dataset.test_loaders[0].dataset.targets,
            dataset.test_loaders[0].dataset.indexes, dataset.test_loaders[0].dataset.transform)
        write_json(output / 'data_fingerprints.json', recorder.data_snapshots)
        with recorder.phase('begin_task'):
            model.meta_begin_task(dataset)
        recorder.snapshots['after_begin_task'] = model_snapshot(model)
        assert model.net.backbone.num_adapters == 1
        assert model.net.classifier.weight.shape == (10, 768)
        assert not any(p.requires_grad for p in model.net.backbone.backbone.parameters())
        optimized = {id(p) for group in model.opt.param_groups for p in group['params']}
        assert all(id(p) not in optimized for p in model.net.backbone.backbone.parameters())
        losses, gradients, changed = [], [], []
        iterator = iter(train_loader)
        for step in range(2):
            inputs, labels, not_augmented = next(iterator)[:3]
            adapter = model.net.backbone.adapters[0][0]
            previous = adapter.up_proj.weight.detach().clone()
            with recorder.phase('smoke_step'):
                loss = model.meta_observe(inputs.to(model.device), labels.to(model.device),
                                          not_augmented.to(model.device), epoch=0)
            assert isinstance(loss, (int, float)) and torch.isfinite(torch.tensor(loss))
            assert all(p.grad is None for p in model.net.backbone.backbone.parameters())
            current_gradients = {f'layer{layer}_{projection}': float(getattr(module, projection).weight.grad.norm())
                                 for layer, module in enumerate(model.net.backbone.adapters[0])
                                 for projection in ('down_proj', 'up_proj')}
            assert all(torch.isfinite(torch.tensor(value)) for value in current_gradients.values())
            assert any(value > 0 for key, value in current_gradients.items() if key.endswith('up_proj'))
            change = float((adapter.up_proj.weight.detach() - previous).norm())
            assert change > 0
            losses.append(float(loss))
            gradients.append(current_gradients)
            changed.append(change)
            recorder.steps += 1
            recorder.training_examples += len(labels)
            wandb.log({'smoke/loss': loss, 'smoke/step': step + 1, 'smoke/adapter_up_change_norm': change})
        assert any(value > 0 for key, value in gradients[-1].items() if key.endswith('down_proj'))
        record.update(status='completed', loss=losses, gradient_norms=gradients,
                      adapter_up_change_norms=changed, frozen_backbone_gradients_absent=True,
                      no_full_task_or_accuracy_evaluation=True, wandb_url=wandb.run.url)
        recorder.snapshots['after_smoke'] = model_snapshot(model)
        code = 0
    except BaseException as error:
        import traceback
        traceback.print_exc()
        record['error_type'] = type(error).__name__
    finally:
        record['computation_status'] = record['status']
        record = recorder.save(record)
        if wandb is not None and wandb.run is not None:
            try:
                wandb.run.summary.update({'smoke_status': record['status'], 'steps': recorder.steps,
                                         'accuracy_evaluated': False, **flatten(record['metrics'], 'smoke_resources')})
                artifact = wandb.Artifact(name, type='cl-implementation-smoke')
                for filename in ('spec.json', 'effective_args.json', 'metrics.json', 'telemetry.csv', 'phases.json',
                                 'shared_data.json', 'data_fingerprints.json'):
                    if (output / filename).is_file():
                        artifact.add_file(str(output / filename), name=filename)
                wandb.run.log_artifact(artifact).wait()
                record['wandb_logging_status'] = 'uploaded'
            except Exception as error:
                record['wandb_logging_status'] = f'failed:{type(error).__name__}'
                record['status'] = 'logging_failed'
                code = 1
            finally:
                try:
                    wandb.finish(exit_code=code)
                except Exception as error:
                    record['wandb_logging_status'] = f'finish_failed:{type(error).__name__}'
                    record['status'], code = 'logging_failed', 1
        notify(f'EASE Base implementation smoke {record["status"]}; logging={record.get("wandb_logging_status", "unavailable")}; W&B {record.get("wandb_url") or "unavailable"}')
        write_json(output / 'metrics.json', record)
        print(json.dumps({'status': record['status'], 'output': str(output), 'wandb_url': record.get('wandb_url')}))
    return code


if __name__ == '__main__':
    sys.exit(main())
