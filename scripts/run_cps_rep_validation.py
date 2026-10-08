"""Opt-in five-run Tiny CPS/REP GPU validation through Mammoth meta hooks.

Default prints scope only. --prepare exports CPU initialization. --execute
starts one worker per physical GPU, each processing its approved group in order.
--cpu-smoke exercises the complete worker without GPU or service writes.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cl_runtime import configure_environment, configure_native_environment

configure_native_environment()

from scripts.cl_efficiency import (
    Recorder,
    flatten,
    model_snapshot,
    notify_slack,
    query_gpu,
    write_json,
)
from scripts.cps_author_reference import (
    SOURCE_COMMIT,
    author_classes,
    file_hash,
    verify_source,
)
from scripts.cps_rep_protocol import (
    RUNS,
    compare_records,
    learned_state,
    native_arguments,
    prepare_initial,
    protocol,
    restore_learned,
    tensor_digest,
)


def execution_sources():
    paths = ['scripts/cps_author_reference.py', 'scripts/cps_rep_protocol.py',
             'scripts/run_cps_rep_validation.py', 'scripts/unified_cl_inputs.py',
             'scripts/cl_efficiency.py', 'scripts/cl_runtime.py', 'utils/training.py',
             'models/utils/continual_model.py', 'models/cps_prompt.py', 'models/rep_prompt.py',
             'models/coda_frozen.py', 'models/coda_prompt.py', 'models/coda_prompt_utils/model.py',
             'models/coda_prompt_utils/vit.py', 'models/coda_prompt_utils/__init__.py']
    for directory in ('models/cps_prompt_utils', 'models/rep_prompt_utils', 'models/resource_prompt_utils'):
        paths.extend(str(path.relative_to(ROOT)) for path in (ROOT / directory).glob('*.py'))
    paths.extend(('models/config/cps_prompt.yaml', 'models/config/rep_prompt.yaml',
                  'models/config/coda_frozen.yaml', 'datasets/configs/seq-cifar100-224/unified.yaml'))
    return {name: file_hash(ROOT / name) for name in sorted(paths)}


def validate_spec(spec):
    run = spec.get('run')
    if run not in RUNS or spec.get('protocol') != protocol() or spec.get('gpu') != RUNS[run][1]:
        raise ValueError('Run differs from the bounded approved five-run protocol')
    if spec.get('sources') != execution_sources() or spec.get('author_sources') != verify_source():
        raise ValueError('Execution sources changed since this run was prepared')
    if file_hash(spec['initial_state']) != spec.get('initial_file_sha256'):
        raise ValueError('Matched learned initialization file changed')


def worker(spec):
    validate_spec(spec)
    if os.getenv('CL_CPS_REP_VALIDATE') != '1':
        raise ValueError('Worker is opt-in through the validation controller')
    cpu_smoke = spec.get('cpu_smoke', False)
    output = Path(spec['output'])
    profile = configure_environment()
    import numpy as np
    import torch

    import main
    import models
    from models.low_rank_utils.factory import CHECKPOINTS
    from models.low_rank_utils.pretrained import canonical_pretrained_digest
    from scripts.unified_cl_inputs import (
        CommonInputStreams,
        build_loader,
        fingerprint_batch,
        fingerprint_dataset,
        fingerprint_loader,
        stable_seed,
    )
    from utils import training

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = main.parse_args(native_arguments(spec['run'], output, cpu_smoke=cpu_smoke))
    with ExitStack() as setup:
        if spec['run'] == 'cps-author':
            reference = author_classes()
            setup.enter_context(patch.object(models, 'get_model',
                lambda arguments, backbone, loss, transform, dataset=None:
                    reference.model(backbone, loss, arguments, transform, dataset)))
        model, dataset, args = main.initialize(args)
    # extend_args enables native W&B from environment. This worker owns one
    # explicit run, including epoch diagnostics and final artifact publication.
    args.nowand = 1
    model.to(args.device)
    if not cpu_smoke and (not torch.cuda.is_available() or model.device.type != 'cuda'
                         or os.getenv('CUDA_VISIBLE_DEVICES') != spec['gpu']):
        raise ValueError('GPU verification requires its assigned physical CUDA device')
    if dataset.N_TASKS != 10 or dataset.N_CLASSES_PER_TASK != 10 or dataset.N_CLASSES != 100:
        raise ValueError('Original ten-task allocation must be retained')
    if args.batch_size != (2 if cpu_smoke else 64) or args.n_epochs != 3 or args.stop_after != 2 or args.code_optimization != 0:
        raise ValueError('Actual parser settings differ from approved FP32 protocol')
    restore_learned(model.net, torch.load(spec['initial_state'], map_location='cpu', weights_only=True))
    initial_digest = tensor_digest(learned_state(model.net))
    pin = CHECKPOINTS['tiny']['canonical_sha256']
    if canonical_pretrained_digest(model.net.feat) != pin:
        raise ValueError('Host encoder differs from canonical150 pin')
    query = getattr(model.net, 'query_encoder', None)
    if query is not None and canonical_pretrained_digest(query) != pin:
        raise ValueError('Swift query encoder differs from canonical150 pin')

    recorder = Recorder(output, gpu='cpu' if cpu_smoke else spec['gpu'])
    recorder.torch, recorder.device = torch, model.device
    recorder.start()
    record = {**spec, 'status': 'running', 'initial_learned_sha256': initial_digest,
              'physical_gpu_uuid': None if cpu_smoke else query_gpu(spec['gpu'])['gpu_uuid'],
              'task_results': [], 'epoch_results': [], 'epoch_inputs': [], 'data_fingerprints': {},
              'task_initialization': [], 'structural_checks_passed': False, 'notifications': [],
              'packages': {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'wandb')},
              'paper_reproduction': False, 'efficiency_pilot': False, 'wandb_url': None}
    run = None
    streams = CommonInputStreams(1993)
    epoch = {}
    training_logits = None
    eval_logits = []
    old_rows = {}
    task_started = None

    def notify(message):
        status = notify_slack(message, enabled=not cpu_smoke)
        record['notifications'].append({'message': message, 'status': status})
        write_json(output / 'notifications.json', record['notifications'])

    def log(value):
        if run is not None:
            run.log(value)

    def progress():
        write_json(output / 'progress.json', {'run': spec['run'], 'task': model.current_task + 1,
                   'epoch': epoch.get('epoch'), 'steps': epoch.get('steps', 0), 'status': record['status'],
                   'wandb_url': record['wandb_url']})

    def replace_loader(original):
        def wrapped():
            train, test = original()
            task = len(dataset.test_loaders) - 1
            streams.set_phase(task, 0, 'train')
            train_view = streams.wrap_dataset(train.dataset, split='train', include_original_ids=True)
            test_view = streams.wrap_dataset(test.dataset, split='test')
            dataset.train_loader = build_loader(train_view, batch_size=args.batch_size,
                num_workers=args.num_workers, shuffle=True, pin_memory=not cpu_smoke)
            dataset.test_loaders[-1] = build_loader(test_view, batch_size=args.batch_size,
                num_workers=args.num_workers, shuffle=False, pin_memory=not cpu_smoke)
            for split, view, expected in (('train', train_view, 5000), ('test', test_view, 1000)):
                fingerprint = fingerprint_dataset(view)
                if fingerprint['n_samples'] != expected or not np.array_equal(np.unique(view.targets), np.arange(task * 10, (task + 1) * 10)):
                    raise ValueError('Masked CIFAR stream has missing, old or future samples')
                record['data_fingerprints'][f'task{task + 1}_{split}'] = fingerprint
            write_json(output / 'data_fingerprints.json', record['data_fingerprints'])
            return dataset.train_loader, dataset.test_loaders[-1]
        return wrapped

    def begin_task(original):
        def wrapped(current):
            nonlocal task_started
            task_started = time.perf_counter()
            # Source and native task-wise Gram-Schmidt receive the same stream;
            # it cannot advance the CPS sampler or REP ALD generator.
            with torch.random.fork_rng(devices=[model.device] if model.device.type == 'cuda' else []):
                selected = stable_seed(1993, 'task_initialization', model.current_task)
                torch.default_generator.manual_seed(selected)
                if model.device.type == 'cuda':
                    torch.cuda.manual_seed(selected)
                result = original(current)
            task = model.current_task
            record['task_initialization'].append({'task': task + 1, 'seed': selected,
                'learned_sha256': tensor_digest(learned_state(model.net)),
                'current_rows_sha256': tensor_digest({name: value[task * 10:(task + 1) * 10]
                     for name, value in model.net.prompt.state_dict().items()})})
            recorder.snapshots[f'task{task + 1}_begin'] = model_snapshot(model)
            progress()
            return result
        return wrapped

    def begin_epoch(original):
        def wrapped(index, current):
            streams.set_phase(model.current_task, index, 'train')
            epoch.clear()
            epoch.update(task=model.current_task + 1, epoch=index + 1, steps=0, samples=0,
                         loss_sum=0., correct=0, digest=hashlib.sha256(),
                         lrs=[group['lr'] for group in model.opt.param_groups], diagnostics={})
            # Sampler reads streams.phase at __iter__, so the trainer's loader
            # reference remains valid and workers see the new epoch coordinate.
            audit = fingerprint_loader(dataset.train_loader)
            if not audit['isolated'] or audit['drop_last'] or audit['persistent_workers']:
                raise ValueError('Common-input loader is not isolated')
            result = original(index, current)
            progress()
            return result
        return wrapped

    def net_forward(original):
        def wrapped(*arguments, **keywords):
            nonlocal training_logits
            result = original(*arguments, **keywords)
            logits = result[0] if isinstance(result, tuple) else result
            if recorder.active_phase == 'train_epoch' and not keywords.get('feat', False):
                training_logits = logits.detach()
            return result
        return wrapped

    def observed(original):
        def wrapped(inputs, labels, not_aug_inputs, **keywords):
            batch = fingerprint_batch(inputs, labels, keywords['original_ids'])
            for key in ('inputs_sha256', 'labels_sha256', 'original_ids_sha256'):
                epoch['digest'].update(batch[key].encode())
            if epoch['steps'] == 0:
                write_json(output / f'task{epoch["task"]}_epoch{epoch["epoch"]}_first_batch.json', batch)
            result = original(inputs, labels, not_aug_inputs, **keywords)
            if not math.isfinite(float(result)):
                raise FloatingPointError('Nonfinite training loss')
            epoch['loss_sum'] += float(result) * len(labels)
            epoch['steps'] += 1
            epoch['samples'] += len(labels)
            recorder.steps += 1
            recorder.training_examples += len(labels)
            logits = training_logits[:, model.n_past_classes:model.n_seen_classes]
            epoch['correct'] += int((logits.argmax(1) + model.n_past_classes == labels).sum())
            diagnostic = getattr(model.net, 'diagnostics', {})
            for key, value in diagnostic.items():
                if key in ('executed', 'merged_tokens'):
                    value = sum(value)
                if isinstance(value, (int, float)):
                    epoch['diagnostics'][key] = epoch['diagnostics'].get(key, 0.) + value
            if epoch['steps'] % 20 == 0:
                progress()
            return result
        return wrapped

    def train_epoch(original):
        def wrapped(*arguments, **keywords):
            with recorder.phase('train_epoch'):
                result = original(*arguments, **keywords)
            if not cpu_smoke and (epoch['steps'] != 79 or epoch['samples'] != 5000):
                raise ValueError('Epoch did not consume exactly the approved full task stream')
            for encoder in (model.net.feat, query):
                if encoder is not None and any(parameter.requires_grad or parameter.grad is not None for parameter in encoder.parameters()):
                    raise ValueError('Frozen encoder received gradients or was made trainable')
            if any(not bool(torch.isfinite(parameter).all()) or parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                   for parameter in model.get_parameters()):
                raise FloatingPointError('Trainable prompt/head state or gradient is nonfinite')
            past = model.current_task * 10
            for name, value in model.net.prompt.state_dict().items():
                if past and tensor_digest({name: value[:past]}) != old_rows[name]:
                    raise ValueError('Previously learned prompt rows changed')
            row = {key: epoch[key] for key in ('task', 'epoch', 'steps', 'samples', 'lrs')}
            row.update(loss=epoch['loss_sum'] / epoch['samples'], train_accuracy_pct=100 * epoch['correct'] / epoch['samples'],
                       diagnostics={key: value / epoch['steps'] for key, value in epoch['diagnostics'].items()})
            record['epoch_results'].append(row)
            record['epoch_inputs'].append({key: row[key] for key in ('task', 'epoch', 'steps', 'samples')} |
                                           {'batches_sha256': epoch['digest'].hexdigest()})
            write_json(output / 'epoch_results.json', record['epoch_results'])
            write_json(output / 'epoch_inputs.json', record['epoch_inputs'])
            log(flatten({'epoch': row}))
            print(f'{spec["run"]}: task{row["task"]} epoch{row["epoch"]}, loss={row["loss"]:.5f}, train={row["train_accuracy_pct"]:.2f}%', flush=True)
            return result
        return wrapped

    def forward(original):
        def wrapped(*arguments, **keywords):
            result = original(*arguments, **keywords)
            if recorder.active_phase == 'evaluation':
                if not bool(torch.isfinite(result).all()):
                    raise FloatingPointError('Nonfinite seen-class evaluation logits')
                # CPU tensors otherwise alias evaluation's in-place Task-IL
                # masking and silently overwrite the saved Class-IL logits.
                eval_logits.append(result.detach().cpu().numpy().copy())
                recorder.evaluation_examples += len(result)
            return result
        return wrapped

    def evaluate(original):
        def wrapped(*arguments, **keywords):
            eval_logits.clear()
            streams.set_phase(model.net.prompt.task_count, 0, 'evaluation')
            with recorder.phase('evaluation'):
                result = original(*arguments, **keywords)
            task = model.net.prompt.task_count + 1
            labels = np.concatenate([loader.dataset.targets for loader in dataset.test_loaders])
            ids = np.concatenate([loader.dataset.indexes for loader in dataset.test_loaders])
            logits = np.concatenate(eval_logits)
            if cpu_smoke:
                # Debug evaluator truncates each task separately.
                counts = [min(len(loader.dataset), (model.get_debug_iters() + 1) * args.batch_size) for loader in dataset.test_loaders]
                labels = np.concatenate([loader.dataset.targets[:count] for loader, count in zip(dataset.test_loaders, counts)])
                ids = np.concatenate([loader.dataset.indexes[:count] for loader, count in zip(dataset.test_loaders, counts)])
            if len(logits) != len(labels):
                raise ValueError('Captured evaluation predictions have missing samples')
            predictions = logits.argmax(1)
            old = labels < (task - 1) * 10
            row = {'task': task, 'seen_samples': len(labels), 'seen_accuracy_pct': float(100 * (predictions == labels).mean()),
                   'old_accuracy_pct': float(100 * (predictions[old] == labels[old]).mean()) if old.any() else None,
                   'new_accuracy_pct': float(100 * (predictions[~old] == labels[~old]).mean()),
                   'class_il_per_task': result[0], 'task_il_per_task': result[1],
                   'task_wall_s_with_audit': time.perf_counter() - task_started}
            if (not cpu_smoke and len(labels) != task * 1000) or any(abs(value - 100 * (predictions[labels // 10 == index] == labels[labels // 10 == index]).mean()) > 1e-8
                                                                    for index, value in enumerate(result[0])):
                raise ValueError('Saved predictions disagree with Mammoth class-IL evaluation')
            np.savez_compressed(output / f'task{task}_predictions.npz', logits=logits, predictions=predictions, labels=labels, global_ids=ids)
            for encoder in (model.net.feat, query):
                if encoder is not None and canonical_pretrained_digest(encoder) != pin:
                    raise ValueError('Frozen canonical encoder tensors changed')
            old_rows.clear()
            old_rows.update({name: tensor_digest({name: value[:task * 10]}) for name, value in model.net.prompt.state_dict().items()})
            saved = {'model': model.state_dict(), 'optimizer': model.opt.state_dict(), 'scheduler': model.custom_scheduler.state_dict(),
                     'protocol': protocol(), 'task': task, 'epoch_inputs': record['epoch_inputs']}
            state_hash = tensor_digest(saved['model'])
            torch.save(saved, output / f'task{task}_state.pt')
            restored = torch.load(output / f'task{task}_state.pt', map_location=model.device, weights_only=True)
            model.load_state_dict(restored['model'], strict=True)
            model.opt.load_state_dict(restored['optimizer'])
            model.custom_scheduler.load_state_dict(restored['scheduler'])
            if state_hash != tensor_digest(model.state_dict()) or model.net.prompt.task_count != task - 1:
                raise ValueError('Checkpoint model/task round trip failed')
            recorder.tasks_completed += 1
            recorder.snapshots[f'task{task}_end'] = model_snapshot(model)
            record['task_results'].append(row)
            write_json(output / 'task_results.json', record['task_results'])
            log(flatten({'task': row}))
            notify(f'CPS/REP 검증 {spec["run"]}: task{task}/2 완료, seen={row["seen_accuracy_pct"]:.2f}%, W&B {record["wandb_url"] or "CPU smoke"}')
            return result
        return wrapped

    try:
        if not cpu_smoke:
            import wandb
            run = wandb.init(entity=profile['wandb_entity'], project=profile['wandb_project'],
                name=f'cps-rep-{spec["run"]}-{spec["batch_id"]}', group=f'cps-rep-{spec["batch_id"]}',
                job_type='implementation-validation', config={**spec, 'initial_learned_sha256': initial_digest},
                dir=str(output), mode='online')
            record['wandb_url'], record['wandb_id'] = run.url, run.id
        notify(f'CPS/REP 검증 {spec["run"]} 시작: GPU{spec["gpu"]}, Tiny/batch64/2tasks×3epochs; W&B {record["wandb_url"] or "CPU smoke"}')
        with ExitStack() as stack:
            wrappers = [(dataset, 'get_data_loaders', replace_loader), (model, 'meta_begin_task', begin_task),
                        (model, 'meta_begin_epoch', begin_epoch), (model, 'meta_observe', observed),
                        (training, 'train_single_epoch', train_epoch), (model.net, 'forward', net_forward),
                        (model, 'forward', forward), (dataset, 'evaluate', evaluate)]
            for target, name, wrapper in wrappers:
                stack.enter_context(patch.object(target, name, wrapper(getattr(target, name))))
            training.train(model, dataset, args)
        if len(record['task_results']) != 2 or len(record['epoch_results']) != 6 or recorder.tasks_completed != 2:
            raise ValueError('Training returned before all approved tasks and epochs completed')
        if not cpu_smoke and (recorder.steps != 474 or recorder.training_examples != 30000 or recorder.evaluation_examples != 3000):
            raise ValueError('Full stream visit/step counts are incomplete')
        record.update(status='completed', structural_checks_passed=True)
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error)[:500])
        raise
    finally:
        telemetry = recorder.save(record)
        record['metrics'], record['phases'], record['sensor_errors'] = telemetry['metrics'], telemetry['phases'], telemetry['sensor_errors']
        write_json(output / 'result.json', record)
        if run is not None:
            log(flatten({'resources': record['metrics']}))
            for key in ('status', 'structural_checks_passed'):
                run.summary[key] = record[key]
            artifact = wandb.Artifact(f'cps-rep-{spec["run"]}-{spec["batch_id"]}', type='validation-results')
            for name in ('spec.json', 'result.json', 'epoch_results.json', 'epoch_inputs.json', 'task_results.json',
                         'data_fingerprints.json', 'metrics.json', 'telemetry.csv', 'phases.json',
                         'task1_predictions.npz', 'task2_predictions.npz', 'task1_state.pt', 'task2_state.pt'):
                if (output / name).is_file():
                    artifact.add_file(str(output / name))
            run.log_artifact(artifact).wait()
            run.finish(exit_code=0 if record['status'] == 'completed' else 1)
        notify(f'CPS/REP 검증 {spec["run"]}: {record["status"]}; W&B {record["wandb_url"] or "CPU smoke"}')
        write_json(output / 'result.json', record)
        progress()
    return 0


def build_spec(run, output, initial, *, cpu_smoke=False):
    return {'schema': 'cps-rep-gpu-validation-v1', 'run': run, 'gpu': RUNS[run][1], 'protocol': protocol(),
            'output': str(Path(output).resolve()), 'initial_state': str(Path(initial).resolve()),
            'initial_file_sha256': file_hash(initial), 'sources': execution_sources(), 'author_sources': verify_source(),
            'source_commit': SOURCE_COMMIT, 'cpu_smoke': cpu_smoke,
            'batch_id': Path(output).parent.name,
            'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()}


def execute(output):
    output = Path(output).resolve()
    configure_environment()
    initial = prepare_initial(output)
    queues = {'1': ['cps-author', 'cps-native'], '0': ['coda-frozen', 'rep-off', 'rep-on']}
    running, completed = {}, {}
    write_json(output / 'plan.json', {'protocol': protocol(), 'queues': queues, 'execution_authorized': True,
               'held': ['20epoch efficiency pilot', 'Small/Base and batch16/64 expansion', '11method unified batch']})

    def launch(gpu):
        if not queues[gpu]:
            return
        run = queues[gpu].pop(0)
        device = query_gpu(gpu)
        if device['gpu_device_memory_mib'] is None or device['gpu_device_memory_mib'] > 1000 or device['gpu_utilization_percent'] > 10:
            raise RuntimeError(f'GPU{gpu} is occupied; no competing validation process will be launched')
        directory = output / run
        directory.mkdir(parents=True, exist_ok=False)
        spec = build_spec(run, directory, initial)
        write_json(directory / 'spec.json', spec)
        env = os.environ.copy()
        env.pop('MAMMOTH_DEVICE', None)
        env.update(CUDA_VISIBLE_DEVICES=gpu, CL_CPS_REP_VALIDATE='1', WANDB_MODE='online',
                   HF_HUB_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        console = (directory / 'console.log').open('w')
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(directory / 'spec.json')],
                                   cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        running[gpu] = (run, process, console, directory)
        print(f'Launched {run} on GPU{gpu}: {directory}', flush=True)

    try:
        for gpu in queues:
            launch(gpu)
        while running:
            for gpu, (name, process, console, directory) in list(running.items()):
                if process.poll() is None:
                    continue
                console.close()
                del running[gpu]
                completed[name] = json.loads((directory / 'result.json').read_text()) if (directory / 'result.json').exists() else {'status': 'failed'}
                if process.returncode or completed[name]['status'] != 'completed':
                    queues[gpu].clear()
                    print(f'{name} failed; remaining runs on GPU{gpu} held', flush=True)
                else:
                    launch(gpu)
            time.sleep(1)
    except BaseException:
        from scripts.benchmark_cl import stop_worker
        for _, process, console, _ in running.values():
            stop_worker(process)
            console.close()
        raise
    gates = {}
    for name, reference, decline in (('cps-native', 'cps-author', False), ('rep-off', 'coda-frozen', False), ('rep-on', 'coda-frozen', True)):
        if name in completed and reference in completed:
            gates[name] = compare_records(completed[name], completed[reference], decline_only=decline)
    summary = {'protocol': protocol(), 'runs': {name: {'status': row['status'], 'wandb_url': row.get('wandb_url'),
               'task_results': row.get('task_results'), 'result': str(output / name / 'result.json')} for name, row in completed.items()},
               'acceptance': gates, 'completed_runs': sum(row['status'] == 'completed' for row in completed.values()),
               'next_stage_held': True}
    write_json(output / 'summary.json', summary)
    status = notify_slack(f'CPS/REP 구현 검증 종료: {summary["completed_runs"]}/5 완료; 판정 { {key: value["status"] for key, value in gates.items()} }; 효율성 후속 실험은 보류 상태')
    write_json(output / 'completion_notification.json', {'status': status})
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary['completed_runs'] == 5 else 1


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/cps_rep_gpu_validation_20261009')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--execute', action='store_true')
    actions.add_argument('--worker', type=Path)
    actions.add_argument('--cpu-smoke', choices=tuple(RUNS))
    options = parser.parse_args()
    if options.worker:
        return worker(json.loads(options.worker.read_text()))
    if options.execute:
        return execute(options.output)
    if options.prepare or options.cpu_smoke:
        initial = prepare_initial(options.output)
        if options.cpu_smoke:
            directory = options.output / options.cpu_smoke
            directory.mkdir(parents=True, exist_ok=False)
            spec = build_spec(options.cpu_smoke, directory, initial, cpu_smoke=True)
            write_json(directory / 'spec.json', spec)
            os.environ['CL_CPS_REP_VALIDATE'] = '1'
            return worker(spec)
        print(json.dumps({'initial_state': str(initial), 'protocol': protocol()}))
        return 0
    print(json.dumps({'protocol': protocol(), 'runs': RUNS, 'execution_started': False}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main_cli())
