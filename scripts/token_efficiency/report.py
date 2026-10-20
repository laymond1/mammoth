"""Collect device records and accuracy records into the tables the results page shows.

    python -m scripts.token_efficiency.report --results results/token_efficiency --out results/token_efficiency/summary.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import cascade, flops, methods

DEVICES = {  # folder -> (short name, kind)
    'cal07-rtx4090': ('RTX 4090', 'gpu'),
    'cal08-rtx2080ti': ('RTX 2080 Ti', 'gpu'),
    'aiatom-gb10': ('GB10', 'gpu'),
    'jetson-orin-nano-25w': ('Jetson Orin Nano 25W', 'gpu'),
    'rpi5-8gb': ('Raspberry Pi 5', 'cpu'),
}
FAMILIES = [  # (pattern on the spec id, family, question)
    (r'^dense$', 'dense', 0),
    (r'^tome-', 'ToMe', 1), (r'^pitome-', 'PiToMe', 1), (r'^evit-', 'EViT', 1), (r'^cls-prune-', 'CLS-prune', 1),
    (r'^ppt-', 'PPT', 2), (r'^repshift-', 'Rep. Shift', 2), (r'^ats-', 'ATS', 2),
    (r'^kept-random-', 'random patches', 3), (r'^kept-center-', 'centre patches', 3),
    (r'^kept-variance-', 'pixel variance', 3), (r'^res\d+', 'lower resolution', 3),
    (r'^kept-cps-L12-', 'CPS score, layer 12', 4), (r'^kept-cps-L2-', 'score at layer 2', 4),
    (r'^kept-cps-L3-', 'score at layer 3', 4), (r'^continue-L2-', 'continue from layer 2', 4),
    (r'^continue-L3-', 'continue from layer 3', 4),
    (r'^kept-crosstiny-L3-', 'ViT-Ti layer-3 scorer', 5), (r'^kept-crosstiny-L12-', 'ViT-Ti layer-12 scorer', 5),
]


def family(method):
    for pattern, name, question in FAMILIES:
        if re.search(pattern, method):
            return name, question
    return method, None


def load_units(folder):
    units = {}
    for path in sorted(Path(folder).glob('*.json')):
        if path.name == 'idle_power.json' or '.contaminated' in path.name:
            continue
        record = json.loads(path.read_text())
        units[record['unit_id']] = record
    return units


def ok_rows(record):
    return {row['batch']: row for row in record['rows'] if row.get('status') == 'ok'}


def regime_table(units):
    """Per (size, dtype, method): latency and energy relative to dense at each batch, and the first batch that is faster."""
    table = []
    for record in units.values():
        unit = record['unit']
        if unit['spec']['kind'] == 'dense' or unit.get('train'):
            continue
        dense = units.get(f"{unit['size']}__{unit['dtype']}__dense")
        if not dense:
            continue
        rows, base = ok_rows(record), ok_rows(dense)
        entry = {'size': unit['size'], 'dtype': unit['dtype'], 'method': methods.spec_id(unit['spec']),
                 'flops_ratio': record['flops']['total'] / record['dense_flops'], 'batches': {}}
        for batch, row in rows.items():
            if batch not in base:
                continue
            cell = {'eager': row['eager']['median_ms'] / base[batch]['eager']['median_ms'],
                    'eager_ms': row['eager']['median_ms'], 'dense_ms': base[batch]['eager']['median_ms']}
            if 'median_ms' in row.get('graph', {}) and 'median_ms' in base[batch].get('graph', {}):
                cell['graph'] = row['graph']['median_ms'] / base[batch]['graph']['median_ms']
                cell['graph_ms'], cell['dense_graph_ms'] = row['graph']['median_ms'], base[batch]['graph']['median_ms']
            energy, dense_energy = row.get('energy', {}), base[batch].get('energy', {})
            if energy.get('joules_per_image') and dense_energy.get('joules_per_image'):
                cell['energy'] = energy['joules_per_image'] / dense_energy['joules_per_image']
                cell['joules_per_image'] = energy['joules_per_image']
                cell['dense_joules_per_image'] = dense_energy['joules_per_image']
                cell['mean_w'], cell['dense_mean_w'] = energy.get('mean_w'), dense_energy.get('mean_w')
            for key in ('cuda_peak_allocated_mb', 'rss_peak_mb'):
                if key in row['memory'] and key in base[batch]['memory']:
                    cell[f'memory_{key}'] = row['memory'][key] / base[batch]['memory'][key]
            if row.get('profile', {}).get('kernels'):
                cell['kernels'] = row['profile']['kernels']
                cell['gpu_ms'] = row['profile']['gpu_us'] / 1000
                cell['groups'] = {name: value['gpu_us'] / 1000 for name, value in row['profile']['groups'].items()}
                if base[batch].get('profile', {}).get('kernels'):
                    cell['dense_kernels'] = base[batch]['profile']['kernels']
                    cell['dense_gpu_ms'] = base[batch]['profile']['gpu_us'] / 1000
                    cell['dense_groups'] = {n: v['gpu_us'] / 1000 for n, v in base[batch]['profile']['groups'].items()}
            entry['batches'][batch] = cell
        faster = [batch for batch, cell in sorted(entry['batches'].items()) if cell['eager'] < 1]
        entry['first_faster_batch'] = faster[0] if faster else None
        table.append(entry)
    return sorted(table, key=lambda e: (e['size'], e['dtype'], e['method']))


def accuracy_index(folder):
    index = {}
    for path in sorted(Path(folder).glob('*/*.json')):
        record = json.loads(path.read_text())
        index[(record['size'], record['id'])] = record
    return index


def sweep_table(units, accuracy):
    """Every measured (size, dtype, batch, method) with latency, energy, FLOPs and ImageNet accuracy."""
    rows = []
    for record in units.values():
        unit = record['unit']
        method = methods.spec_id(unit['spec'])
        acc = accuracy.get((unit['size'], method))
        name, question = family(method)
        for batch, row in ok_rows(record).items():
            energy = row.get('energy', {})
            rows.append({'size': unit['size'], 'dtype': unit['dtype'], 'batch': batch, 'method': method, 'family': name,
                         'question': question, 'latency_ms': row['eager']['median_ms'], 'p90_ms': row['eager']['p90_ms'],
                         'joules_per_image': energy.get('joules_per_image'), 'mean_w': energy.get('mean_w'),
                         'flops_ratio': (acc or {}).get('flops_ratio', (record['flops']['total'] or 0) / record['dense_flops']),
                         'accuracy': (acc or {}).get('accuracy'),
                         'memory_mb': row['memory'].get('cuda_peak_allocated_mb', row['memory'].get('rss_peak_mb'))})
    return rows


def pareto(points, x, y='accuracy'):
    """Points not beaten by a point with no more ``x`` and no less ``y``."""
    best, front = -1.0, []
    for point in sorted((p for p in points if p.get(x) is not None and p.get(y) is not None), key=lambda p: (p[x], -p[y])):
        if point[y] > best:
            front.append(point)
            best = point[y]
    return front


def train_table(units):
    rows = []
    for record in units.values():
        unit = record['unit']
        method = methods.spec_id(unit['spec'])
        for batch, row in ok_rows(record).items():
            energy = row.get('energy', {})
            rows.append({'size': unit['size'], 'kind': unit['train'], 'dtype': unit['dtype'], 'batch': batch, 'method': method,
                         'family': family(method)[0], 'step_ms': row['step']['median_ms'], 'trainable': row['trainable'],
                         'memory_mb': row['memory'].get('cuda_peak_allocated_mb', row['memory'].get('rss_peak_mb')),
                         'joules_per_step': energy.get('joules') / energy['calls'] if energy.get('joules') and energy.get('calls') else None})
        for row in record['rows']:
            if row.get('status') == 'oom':
                rows.append({'size': unit['size'], 'kind': unit['train'], 'dtype': unit['dtype'], 'batch': row['batch'],
                             'method': method, 'family': family(method)[0], 'oom': True})
    return rows


def accuracy_table(index):
    rows = []
    for (size, method), record in sorted(index.items()):
        name, question = family(method)
        rows.append({'size': size, 'method': method, 'family': name, 'question': question, 'accuracy': record['accuracy'],
                     'flops_ratio': record['flops_ratio'], 'overhead_ratio': (record['overhead_flops'] or 0) / flops.dense(size)})
    return rows


def cascades(folder, devices):
    """Question 5 curves on FLOPs, and S3 expected batch-1 latency from each device's dense latencies."""
    out = {'s1': [], 's2': [], 's3': []}
    for size in ('tiny', 'small', 'base'):
        target = flops.inblock(size, 'tome', 16)[0]
        drafts = [{'kind': 'inblock', 'operator': 'tome', 'degree': 16},
                  {'kind': 'continue', 'layer': 3, 'keep': flops.keep_for_continue(size, target, 3)}]
        for spec in drafts:
            try:
                out['s1'].append(cascade.s1(folder, size, spec))
            except FileNotFoundError:
                pass
        try:
            out['s2'].append(cascade.s2(folder, size, 3))
        except (FileNotFoundError, KeyError):
            pass
    for large in ('small', 'base'):
        try:
            result = cascade.s3(folder, 'tiny', large)
        except FileNotFoundError:
            continue
        result['latency'] = {}
        for device, tables in devices.items():
            dense = {(row['size'], row['dtype'], row['batch']): row['latency_ms'] for row in tables.get('sweep_rows', [])
                     if row['method'] == 'dense'}
            for dtype in ('fp32', 'fp16'):
                small_ms, large_ms = dense.get(('tiny', dtype, 1)), dense.get((large, dtype, 1))
                if small_ms and large_ms:
                    result['latency'][f'{device}/{dtype}'] = {
                        'dense_ms': large_ms, 'points': [{'threshold': row['threshold'], 'accuracy': row['accuracy'],
                                                         'latency_ms': small_ms + (1 - row['acceptance']) * large_ms}
                                                        for row in result['rows']]}
        out['s3'].append(result)
    return out


def build(root):
    root = Path(root)
    index = accuracy_index(root / 'accuracy')
    summary = {'accuracy': accuracy_table(index), 'devices': {}}
    for folder in sorted((root / 'devices').glob('*')):
        if not folder.is_dir():
            continue
        entry = {'name': DEVICES.get(folder.name, (folder.name, None))[0]}
        for grid in ('regime', 'sweep', 'train'):
            path = folder / grid
            if not path.exists():
                continue
            units = load_units(path)
            idle = path / 'idle_power.json'
            entry[f'{grid}_units'] = len(units)
            if idle.exists():
                payload = json.loads(idle.read_text())
                entry[f'{grid}_idle'] = payload['before']
                entry.setdefault('environment', payload.get('environment'))
            if grid == 'regime':
                entry['regime'] = regime_table(units)
            elif grid == 'sweep':
                entry['sweep_rows'] = sweep_table(units, index)
            else:
                entry['train_rows'] = train_table(units)
        summary['devices'][folder.name] = entry
    summary['cascade'] = cascades(root / 'accuracy', summary['devices'])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results/token_efficiency')
    parser.add_argument('--out', default='results/token_efficiency/summary.json')
    args = parser.parse_args()
    summary = build(args.results)
    Path(args.out).write_text(json.dumps(summary, indent=1, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))
    print('wrote', args.out, {name: {key: value for key, value in device.items() if key.endswith('_units')} for name, device in summary['devices'].items()})


if __name__ == '__main__':
    main()
