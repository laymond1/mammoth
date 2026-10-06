"""Opt-in native EASE two-task acceptance using the unchanged Mammoth trainer.

The approved seed1993 run is separate from the one-task efficiency pilot. No
training, credentials, W&B or Slack calls occur without explicit --execute.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import datetime
import functools
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_ease_runtime import file_hash
from scripts.cl_efficiency import Recorder, flatten, instrument, model_snapshot, notify_slack, write_json
from scripts.cl_runtime import configure_environment


def validate_spec(spec):
    protocol = spec['protocol']
    if (spec['stage'] != 'native_two_task_acceptance' or spec['tasks'] != 2
            or protocol['seed'] != 1993 or protocol['n_epochs_per_task'] != 20
            or protocol['batch_size'] != 48 or protocol['initial_variant'] != 'base_fixed64'
            or protocol['training_authorized'] is not True):
        raise ValueError('Native validation must remain the authorized Base two-task seed1993 protocol')
    policy = protocol['accuracy_policy']
    if (policy['confirmed_by'] != 'user' or policy['pass_max_drop_pp'] != 1.5
            or policy['review_max_drop_pp'] != 3.0):
        raise ValueError('Native validation requires the confirmed 1.5/3.0pp policy')


def execute(options, profile):
    plan = json.loads((ROOT / 'benchmarks/cl_efficiency/ease/validation_plan.json').read_text())
    reference_path = Path(plan['reference']['result_summary'])
    if not reference_path.is_absolute():
        reference_path = ROOT / 'benchmarks/cl_efficiency/ease' / reference_path
    reference = json.loads(reference_path.read_text())['official_reference']
    checkpoint = Path(plan['reference']['checkpoint_file'])
    if file_hash(checkpoint) != plan['reference']['checkpoint_sha256']:
        raise ValueError('Base checkpoint does not match the pinned reference hash')
    if reference['status'] != 'completed':
        raise ValueError('Matched official reference is incomplete')
    name = 'ease-native-validation-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = options.output.resolve() / name
    spec = {'name': name, 'stage': 'native_two_task_acceptance', 'tasks': 2,
            'output': str(output), 'gpu': profile['gpu'], 'data_root': profile['data_root'],
            'checkpoint_file': str(checkpoint), 'checkpoint_sha256': plan['reference']['checkpoint_sha256'],
            'checkpoint_tag': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
            'protocol': plan['validation'], 'reference_summary': str(reference_path),
            'reference_summary_sha256': file_hash(reference_path),
            'reference_wandb_url': reference['wandb_url'],
            'wandb_entity': profile['wandb_entity'], 'wandb_project': profile['wandb_project'],
            'paper_reproduction': False, 'efficiency_protocol': False, 'flops_measured': False,
            'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                                         text=True, check=True).stdout.strip(),
            'execution_source_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in
                [Path(__file__), ROOT/'scripts/ease_acceptance.py', ROOT/'scripts/cl_efficiency.py',
                 ROOT/'scripts/ease_reference_utils.py', ROOT/'models/ease.py',
                 ROOT/'backbone/ease_vit.py', ROOT/'utils/training.py', ROOT/'uv.lock']}}
    validate_spec(spec)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'spec.json', spec)
    notifications = []

    def notify(message):
        notifications.append({'message': message, 'status': notify_slack(message)})
        write_json(output / 'notifications.json', notifications)

    environment = os.environ.copy()
    environment.pop('MAMMOTH_DEVICE', None)
    environment.update(CUDA_VISIBLE_DEVICES=profile['gpu'], CL_EASE_VALIDATION_EXECUTE='1',
                       HF_HUB_OFFLINE='1', WANDB_MODE='online', WANDB_DIR=str(output))
    notify(f'EASE native validation started: Base/adapter64; seed1993; 2 tasks, 20 epochs/task; GPU {profile["gpu"]}')
    started, heartbeat = time.monotonic(), time.monotonic()
    timed_out = False
    from scripts.benchmark_cl import stop_worker
    with (output / 'console.log').open('w') as console:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(output / 'spec.json')],
                                   cwd=ROOT, env=environment, stdout=console, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started >= options.timeout:
                    timed_out = True
                    stop_worker(process)
                    break
                if time.monotonic() - heartbeat >= 600:
                    file = output / 'progress.json'
                    progress = json.loads(file.read_text()) if file.exists() else {}
                    notify(f'EASE native validation running: task {progress.get("task", "?")}/2, epoch {progress.get("epoch", "?")}/20; W&B {progress.get("wandb_url") or "initializing"}')
                    heartbeat = time.monotonic()
                time.sleep(1)
        except BaseException:
            stop_worker(process)
            notify('EASE native validation interrupted; partial local results preserved')
            raise
    file = output / 'metrics.json'
    record = json.loads(file.read_text()) if file.exists() else {'status': 'failed'}
    if timed_out:
        record['status'] = 'timeout'
    elif process.returncode != 0 and record['status'] == 'completed':
        record['status'] = 'failed'
    write_json(file, record)
    notify(f'EASE native validation {record["status"]}; acceptance={record.get("acceptance_status", "not_evaluated")}; logging={record.get("wandb_logging_status", "unavailable")}; W&B {record.get("wandb_url") or "unavailable"}')
    write_json(output / 'completion.json', {'exit_code': process.returncode, 'status': record['status'],
        'acceptance_status': record.get('acceptance_status'), 'timeout': timed_out,
        'wall_s': time.monotonic()-started, 'wandb_logging_status': record.get('wandb_logging_status')})
    print(json.dumps({'status': record['status'], 'acceptance': record.get('acceptance_status'),
                      'output': str(output), 'wandb_url': record.get('wandb_url')}))
    # A successfully measured run may have failed acceptance; the caller must
    # inspect acceptance_status, rather than interpreting an exit code as a gate.
    return 0 if process.returncode == 0 and record['status'] == 'completed' else 1


def worker(spec_path):
    if os.getenv('CL_EASE_VALIDATION_EXECUTE') != '1':
        raise ValueError('Validation worker requires explicit --execute')
    spec = json.loads(spec_path.read_text())
    validate_spec(spec)
    output = Path(spec['output'])
    recorder = Recorder(output, spec['gpu'])
    recorder.start()
    record = {**spec, 'status': 'failed', 'acceptance_status': 'not_evaluated',
              'tasks_results': [], 'epoch_results': [], 'structural_checks': {}}
    code, wandb, stack = 1, None, ExitStack()
    try:
        import torch
        import numpy as np
        import wandb
        from unittest.mock import patch
        from torch.utils.data import SequentialSampler
        import main as mammoth
        import utils.training as training
        from scripts.ease_acceptance import counts_from_predictions, evaluate_acceptance
        from scripts.ease_reference_utils import dataset_fingerprint, shared_cifar_metadata
        reference = json.loads(Path(spec['reference_summary']).read_text())['official_reference']
        if file_hash(spec['reference_summary']) != spec['reference_summary_sha256']:
            raise ValueError('Reference summary changed after preparation')
        if file_hash(spec['checkpoint_file']) != spec['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after preparation')
        recorder.torch, recorder.device = torch, torch.device('cuda:0')
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        command = ['--model', 'ease', '--dataset', 'seq-cifar100-224', '--model_config', 'default',
            '--ease_checkpoint', spec['checkpoint_file'], '--vit_size', 'base', '--adapter_size', '64',
            '--ease_pretrain_type', 'in21k_ft_in1k', '--stop_after', '2', '--n_epochs', '20',
            '--batch_size', '48', '--num_workers', '4', '--seed', '1993', '--permute_classes', '0',
            '--lr', '0.025', '--optim_mom', '0.9', '--optim_wd', '0.0005', '--optimizer', 'sgd',
            '--lr_scheduler', 'cosine', '--scheduler_mode', 'epoch', '--code_optimization', '0',
            '--noise_rate', '0', '--validation', '0', '--label_perc', '1', '--label_perc_by_class', '1',
            '--eval_future', '0', '--enable_other_metrics', '0', '--debug_mode', '0', '--non_verbose', '1',
            '--base_path', spec['data_root'], '--device', 'cuda:0',
            '--results_path', str(output / 'mammoth'), '--checkpoint_path', str(output / 'checkpoints')]
        sys.argv = ['main.py', *command]
        args = mammoth.parse_args(cmd=command)
        wandb.init(entity=spec['wandb_entity'], project=spec['wandb_project'], name=spec['name'],
                   group='ease-base-validation', job_type='native-acceptance', config=spec, dir=str(output))
        record['wandb_url'] = wandb.run.url
        with recorder.phase('initialize'):
            model, dataset, args = mammoth.initialize(args)
        # extend_args enables native W&B when environment targets are present.
        # This diagnostic owns the run and must retain it through artifact save.
        args.nowand = 1
        expected = {'stop_after':2, 'n_epochs':20, 'batch_size':48, 'num_workers':4, 'seed':1993,
            'vit_size':'base', 'adapter_size':64, 'ease_pretrain_type':'in21k_ft_in1k', 'lr':0.025,
            'optim_mom':0.9, 'optim_wd':0.0005, 'optimizer':'sgd', 'lr_scheduler':'cosine',
            'scheduler_mode':'epoch', 'code_optimization':0, 'permute_classes':0, 'noise_rate':0,
            'validation':0, 'label_perc':1, 'label_perc_by_class':1, 'eval_future':0, 'debug_mode':0}
        if any(getattr(args, key) != value for key, value in expected.items()) or args.eval_epochs is not None:
            raise ValueError('Effective native arguments changed the matched protocol')
        assert dataset.N_TASKS == 10 and dataset.N_CLASSES_PER_TASK == 10 and dataset.N_CLASSES == 100
        write_json(output / 'effective_args.json', vars(args))
        write_json(output / 'shared_data.json', shared_cifar_metadata(spec['data_root']))
        record['versions'] = {name: importlib.metadata.version(name) for name in
                             ('torch','torchvision','timm','numpy','pyarrow','wandb')}
        record['precision'] = {'matmul_precision':torch.get_float32_matmul_precision(),
            'tf32_allowed':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'cudnn_tf32_allowed':torch.backends.cudnn.allow_tf32, 'autocast':False, 'parameter_dtype':'float32'}
        record['protocol_differences'] = ['Native and official initialization consume RNG differently.',
            'Native prototype loader is sequential; official keeps shuffle=True.',
            'Native evaluate visits task loaders separately; saved predictions are reordered by original global ID.',
            'Native trainer moves non_augmented inputs and records native system statistics.',
            'This diagnostic owns W&B epoch/result logs; native autologging is disabled.']
        checks = record['structural_checks']
        checks.update(finite_loss_and_gradients=True, dataset_matches_reference=True,
                      frozen_backbone_unchanged=True, old_adapters_unchanged=True,
                      prototype_and_adapter_shapes=True, training_protocol_complete=False,
                      frozen_backbone_gradients_absent=True)
        ptm = model.net.backbone.backbone

        def digest(module):
            result = hashlib.sha256()
            for name, tensor in sorted(module.state_dict().items()):
                if not torch.is_tensor(tensor):
                    raise TypeError('Frozen state must contain only tensors')
                cpu = tensor.detach().cpu().contiguous()
                result.update(name.encode()); result.update(str(tuple(cpu.shape)).encode())
                result.update(str(cpu.dtype).encode()); result.update(cpu.numpy().tobytes())
            return result.hexdigest()

        with recorder.phase('state_audit'):
            ptm_digest = digest(ptm)
        adapter_digests = {}
        record['state_audits'] = {'initial_frozen_backbone_sha256':ptm_digest, 'tasks':[]}
        predictions, epoch, current_labels = [], {}, None
        record['prototype_adapter_image_pairs'] = 0
        stack.enter_context(instrument(recorder, model, dataset, training))

        def loaders(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                result = original(*arguments, **keywords)
                task = len(dataset.test_loaders)
                with recorder.phase('data_audit'):
                    source = dataset.train_loader.dataset
                    recorder.data_snapshots[f'task{task}_train'] = dataset_fingerprint(
                        source.data, source.targets, source.indexes, source.transform)
                    images=np.concatenate([x.dataset.data for x in dataset.test_loaders])
                    labels=np.concatenate([x.dataset.targets for x in dataset.test_loaders])
                    ids=np.concatenate([x.dataset.indexes for x in dataset.test_loaders])
                    order=np.argsort(ids)
                    recorder.data_snapshots[f'task{task}_test'] = dataset_fingerprint(
                        images[order], labels[order], ids[order], dataset.test_loaders[0].dataset.transform)
                    keys=('n_samples','class_counts','samples_ordered_sha256','samples_unordered_sha256','global_ids_ordered_sha256')
                    for split in ('train','test'):
                        actual=recorder.data_snapshots[f'task{task}_{split}']; expected=reference['data_snapshots'][f'task{task}_{split}']
                        if any(actual[key] != expected[key] for key in keys):
                            checks['dataset_matches_reference']=False
                            raise RuntimeError('Native samples differ from matched official reference')
                return result
            return wrapped

        def begin(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                result=original(*arguments, **keywords)
                task=model.net.backbone.num_adapters
                assert not any(p.requires_grad for p in ptm.parameters()) and not ptm.training
                optimized={id(p) for group in model.opt.param_groups for p in group['params']}
                assert all(id(p) not in optimized for p in ptm.parameters())
                assert all(not p.requires_grad and id(p) not in optimized for a in model.net.backbone.adapters[:-1] for p in a.parameters())
                assert all(not a.training for a in model.net.backbone.adapters[:-1])
                assert all(g['momentum']==0.9 and g['weight_decay']==0.0005 and g['lr']==0.025 and not g['nesterov'] for g in model.opt.param_groups)
                recorder.snapshots[f'task{task}_after_begin_task']=model_snapshot(model)
                return result
            return wrapped

        def observed(original):
            @functools.wraps(original)
            def wrapped(inputs, labels, *arguments, **keywords):
                nonlocal current_labels
                current_labels=labels
                result=original(inputs, labels, *arguments, **keywords)
                loss=float(result['loss'] if isinstance(result,dict) else result)
                if not math.isfinite(loss):
                    checks['finite_loss_and_gradients']=False
                    raise RuntimeError('Non-finite native loss')
                epoch['loss_sum']+=loss; epoch['loss_weighted_sum']+=loss*len(labels)
                epoch['steps']+=1; epoch['samples']+=len(labels)
                current_labels=None
                return result
            return wrapped

        def training_logits(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                result=original(*arguments, **keywords)
                if current_labels is not None and recorder.active_phase=='train_epoch':
                    epoch['correct']+=int((result.detach().argmax(1)==current_labels-model.n_past_classes).sum().item())
                return result
            return wrapped

        def train_epoch(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                epoch.clear(); epoch.update(loss_sum=0.,loss_weighted_sum=0.,steps=0,samples=0,correct=0)
                result=original(*arguments, **keywords)
                with recorder.phase('epoch_audit'):
                    if any(p.grad is not None for p in ptm.parameters()):
                        checks['frozen_backbone_gradients_absent']=False
                        raise RuntimeError('Frozen backbone received gradients')
                    if any(not torch.isfinite(p).all().item() or (p.grad is not None and not torch.isfinite(p.grad).all().item())
                           for p in model.net.parameters() if p.requires_grad):
                        checks['finite_loss_and_gradients']=False
                        raise RuntimeError('Non-finite adapter/proxy parameter or gradient')
                row={'task':model.net.backbone.num_adapters,'epoch':keywords['epoch']+1,
                    'epochs':20,'loss':epoch['loss_sum']/epoch['steps'],
                    'loss_example_weighted':epoch['loss_weighted_sum']/epoch['samples'],
                    'train_accuracy_pct':epoch['correct']/epoch['samples']*100,
                    'steps':epoch['steps'],'samples':epoch['samples'], 'lr_after_scheduler':model.opt.param_groups[0]['lr']}
                record['epoch_results'].append(row)
                write_json(output/'epoch_results.json',record['epoch_results'])
                write_json(output/'progress.json',{**row,'wandb_url':record['wandb_url']})
                wandb.log({f'native/{key}':value for key,value in row.items()})
                print(f'Native task {row["task"]}/2 epoch {row["epoch"]}/20: loss={row["loss"]:.4f}, train={row["train_accuracy_pct"]:.2f}%',flush=True)
                return result
            return wrapped

        def prototype(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                with recorder.phase('prototype_completion'):
                    return original(*arguments, **keywords)
            return wrapped

        def extracted(original):
            @functools.wraps(original)
            def wrapped(inputs,*arguments,**keywords):
                result=original(inputs,*arguments,**keywords)
                if recorder.active_phase=='prototype_completion':
                    record['prototype_adapter_image_pairs']+=len(inputs)*model.net.backbone.num_adapters
                return result
            return wrapped

        def forward(original):
            @functools.wraps(original)
            def wrapped(inputs,*arguments,**keywords):
                result=original(inputs,*arguments,**keywords)
                if recorder.active_phase=='evaluation':
                    # Stable ties match native torch.max's first class index.
                    predictions.append(torch.argsort(result.detach(), dim=1, descending=True, stable=True)[:, :5].cpu().numpy())
                return result
            return wrapped

        def evaluated(original):
            @functools.wraps(original)
            def wrapped(*arguments,**keywords):
                predictions.clear()
                assert not keywords.get('last',False) and not keywords.get('return_loss',False)
                for loader in dataset.test_loaders:
                    assert isinstance(loader.sampler,SequentialSampler) and not loader.drop_last
                result=original(*arguments,**keywords)
                task=model.net.backbone.num_adapters
                with recorder.phase('acceptance_audit'):
                    labels=np.concatenate([x.dataset.targets for x in dataset.test_loaders])
                    ids=np.concatenate([x.dataset.indexes for x in dataset.test_loaders])
                    predicted=np.concatenate(predictions)
                    assert len(predicted)==len(labels)
                    counts=counts_from_predictions(predicted[:,0],labels,past_classes=(task-1)*10,seen_classes=task*10)
                    row={'task':task,'seen_classes':task*10,**counts,
                         'seen_accuracy_pct':counts['seen_correct']/counts['seen_samples']*100,
                         'old_accuracy_pct':counts['old_correct']/counts['old_samples']*100 if counts['old_samples'] else None,
                         'new_accuracy_pct':counts['new_correct']/counts['new_samples']*100,
                         'native_class_il_per_task':result[0],'native_task_il_per_task':result[1],
                         'prototype_shape':list(model.net.classifier.weight.shape),'observed_adapters':task}
                    if row['prototype_shape']!=[task*10,task*768] or not model.net.task_finalized.item():
                        checks['prototype_and_adapter_shapes']=False
                        raise RuntimeError('Native prototype/adapter lifecycle differs from protocol')
                    native_correct=[int((predicted[labels//10==i,0]==labels[labels//10==i]).sum()) for i in range(task)]
                    assert all(abs(result[0][i]-native_correct[i]/1000*100)<1e-9 for i in range(task))
                    current_ptm=digest(ptm)
                    if current_ptm!=ptm_digest:
                        checks['frozen_backbone_unchanged']=False
                        raise RuntimeError('Frozen PTM changed during native training')
                    current={str(i):digest(adapter) for i,adapter in enumerate(model.net.backbone.adapters)}
                    if any(current[key]!=value for key,value in adapter_digests.items()):
                        checks['old_adapters_unchanged']=False
                        raise RuntimeError('Previously finalized adapter changed')
                    adapter_digests.update(current)
                    record['state_audits']['tasks'].append({'task':task,'backbone_sha256':current_ptm,'adapter_sha256':current})
                    order=np.argsort(ids)
                    np.savez(output/f'task{task}_predictions.npz',predictions=predicted[order],labels=labels[order],
                             global_ids=ids[order],evaluation_global_ids=ids)
                    torch.save({'model':model.net.state_dict(),'task':task,'checkpoint_sha256':spec['checkpoint_sha256']},output/f'task{task}_state.pt')
                    recorder.snapshots[f'task{task}_after_end_task']=model_snapshot(model)
                    record['tasks_results'].append(row)
                    write_json(output/'task_results.json',record['tasks_results'])
                    wandb.log({f'native/{key}':value for key,value in row.items() if isinstance(value,(int,float))})
                    status=notify_slack(f'EASE native task {task}/2 complete: seen={row["seen_accuracy_pct"]:.2f}%; W&B {record["wandb_url"]}')
                    record.setdefault('task_notifications',[]).append({'task':task,'status':status})
                return result
            return wrapped

        for target,name,wrapper in [(dataset,'get_data_loaders',loaders),(model,'meta_begin_task',begin),
            (model,'meta_observe',observed),(model.net,'training_logits',training_logits),
            (training,'train_single_epoch',train_epoch),(model,'end_task',prototype),
            (model.net.backbone,'extract_all',extracted),(model,'forward',forward),(dataset,'evaluate',evaluated)]:
            stack.enter_context(patch.object(target,name,wrapper(getattr(target,name))))
        # Use the native trainer and all meta wrappers without replacing its loop.
        training.train(model,dataset,args)
        complete=(recorder.tasks_completed==2 and len(record['epoch_results'])==40
            and {(x['task'],x['epoch']) for x in record['epoch_results']}=={(t,e) for t in (1,2) for e in range(1,21)}
            and recorder.steps==4200 and recorder.training_examples==200000 and recorder.evaluation_examples==3000
            and record['prototype_adapter_image_pairs']==15000 and model.net.backbone.num_adapters==2
            and list(model.net.classifier.weight.shape)==[20,1536] and model.net.task_finalized.item())
        checks['training_protocol_complete']=bool(complete)
        if not complete:
            raise RuntimeError('Native training returned before the complete approved protocol')
        report=evaluate_acceptance(record['tasks_results'],spec['protocol']['accuracy_policy'],structural_checks=checks)
        record.update(status='completed',acceptance_status=report['status'],acceptance=report)
        write_json(output/'acceptance.json',report)
        code=0
    except BaseException as error:
        logging.exception('Native EASE validation failed')
        record['error_type']=type(error).__name__
    finally:
        stack.close()
        record['computation_status']=record['status']
        record=recorder.save(record)
        if wandb is not None and wandb.run is not None:
            try:
                scalars={key:value for key,value in flatten({'native_resources':record['metrics'],
                    'native_accuracy':record['tasks_results'][-1] if record['tasks_results'] else {},
                    'acceptance':record.get('acceptance',{})}).items() if isinstance(value,(int,float,bool))}
                wandb.run.summary.update({**scalars,'native_status':record['status'],
                    'acceptance_status':record['acceptance_status'],'paper_reproduction':False,'efficiency_protocol':False})
                artifact=wandb.Artifact(spec['name'],type='cl-native-acceptance')
                for filename in ('spec.json','metrics.json','acceptance.json','effective_args.json','epoch_results.json',
                    'task_results.json','shared_data.json','telemetry.csv','phases.json','task1_predictions.npz','task2_predictions.npz'):
                    if (output/filename).is_file(): artifact.add_file(str(output/filename),name=filename)
                wandb.run.log_artifact(artifact).wait()
                record['wandb_logging_status']='uploaded'
            except Exception as error:
                record['wandb_logging_status']=f'failed:{type(error).__name__}'
                record['status'],code='logging_failed',1
            finally:
                try: wandb.finish(exit_code=code)
                except Exception as error:
                    record['wandb_logging_status']=f'finish_failed:{type(error).__name__}'
                    record['status'],code='logging_failed',1
        elif record['computation_status'] == 'completed':
            record['wandb_logging_status'] = 'run_missing_before_artifact_save'
            record['status'], code = 'logging_failed', 1
        write_json(output/'metrics.json',record)
    return code


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/cl_efficiency/ease/runs')
    parser.add_argument('--timeout',type=float,default=7200.)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    options=parser.parse_args(argv)
    if not options.execute and options.worker is None:
        print('Prepared native EASE Base two-task validation; execution=False')
        return 0
    profile=configure_environment()
    logging.basicConfig(level=logging.INFO)
    return worker(options.worker) if options.worker else execute(options,profile)


if __name__=='__main__':
    sys.exit(main())
