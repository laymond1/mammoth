"""Tables from the stage 1 and stage 2 result folders: ratios to dense, the cost split of the controls, compile modes.

    python -m scripts.tome_latency.report eager    FOLDER [--old OLD_DEVICE_FOLDER]
    python -m scripts.tome_latency.report ablation FOLDER
    python -m scripts.tome_latency.report compile  FOLDER

Every ratio is the median time of one call of the operator over the median time of the dense model of the
same implementation, dtype and batch. Reads JSON only; needs no GPU.
"""
import argparse
import glob
import json
import os
from pathlib import Path


def load(folder):
    records = {}
    for path in sorted(glob.glob(os.path.join(folder, '*.json'))):
        if '.contaminated' in path:
            continue
        record = json.loads(Path(path).read_text())
        unit = record['unit']
        key = (unit['arm'], unit['dtype'], unit['operator'], unit.get('degree', 0), unit.get('variant', 'real'), bool(unit.get('patched')))
        records[key] = record
    return records


def rows_of(record):
    return {row['batch']: row for row in record['rows'] if row.get('status') == 'ok'}


def metric(record, batch, field='eager'):
    row = rows_of(record).get(batch)
    return row[field]['median_ms'] if row and field in row and 'median_ms' in row[field] else None


def ratio(records, key, dense_key, batch, field='eager'):
    a, b = records.get(key), records.get(dense_key)
    if not a or not b:
        return None
    top, bottom = metric(a, batch, field), metric(b, batch, field)
    return top / bottom if top and bottom else None


def fmt(value, spec='{:.2f}'):
    return spec.format(value) if value is not None else '  -  '


def eager_tables(folder, old=None):
    records = load(folder)
    out = {}
    print('## ratio to dense (eager / CUDA graph), by implementation')
    for dtype in ('fp32', 'fp16'):
        print(f'\n### {dtype}')
        print(f"{'implementation':<26}" + ''.join(f'| b{b:<2} eager  graph ' for b in (1, 16, 64)))
        for arm, operator, degree, patched in [('study', 'tome', 16, False), ('plugin', 'tome', 16, False), ('official', 'tome', 16, False),
                                               ('official', 'tome', 0, True)]:
            key = (arm, dtype, operator, degree, 'real', patched)
            dense = (arm, dtype, 'dense', 0, 'real', False)
            cells = []
            for batch in (1, 16, 64):
                cells.append((ratio(records, key, dense, batch), ratio(records, key, dense, batch, 'graph')))
            label = f"{arm} {operator}-r{degree}{' (patched)' if patched else ''}"
            print(f'{label:<26}' + ''.join(f'| {fmt(e):>8} {fmt(g):>6}   ' for e, g in cells))
            out[f'{dtype}/{label}'] = cells
        print('\nbatch 1 eager ratio by operator and degree (study / plugin):')
        for operator in ('tome', 'cls-prune', 'evit', 'pitome'):
            cells = []
            for degree in (8, 16, 24):
                pair = []
                for arm in ('study', 'plugin'):
                    pair.append(fmt(ratio(records, (arm, dtype, operator, degree, 'real', False), (arm, dtype, 'dense', 0, 'real', False), 1)))
                cells.append('/'.join(pair))
            print(f'  {operator:<10}' + ''.join(f'r{d}: {c:<14}' for d, c in zip((8, 16, 24), cells)))
    if old:
        print('\n## repeatability against the five-device study (tome-r16, ViT-Ti, eager ratio)')
        for dtype in ('fp32', 'fp16'):
            ref = json.loads(Path(old, 'regime', f'tiny__{dtype}__tome-r16.json').read_text())
            dense = json.loads(Path(old, 'regime', f'tiny__{dtype}__dense.json').read_text())
            for batch in (1, 16, 64):
                before = rows_of(ref)[batch]['eager']['median_ms'] / rows_of(dense)[batch]['eager']['median_ms']
                after = ratio(records, ('study', dtype, 'tome', 16, 'real', False), ('study', dtype, 'dense', 0, 'real', False), batch)
                print(f'  {dtype} b{batch:<3} before {before:5.2f}  now {fmt(after)}')
    return out


def ablation_table(folder):
    records = load(folder)
    steps = [('dense', 'dense', 0, 'real'), ('shrink', 'tome', None, 'shrink-unmasked'), ('+merge', 'tome', None, 'real-unmasked'), ('+mask', 'tome', None, 'real')]
    result = {}
    for field, title in (('eager', 'eager (CPU issues every kernel)'), ('graph', 'CUDA graph replay (no launch cost)')):
        print(f'## where the slowdown comes from, {title}: study ViT, ms; each step against the one before')
        for dtype in ('fp32', 'fp16'):
            for batch in (1, 16, 64):
                for degree in (8, 16, 24):
                    times = []
                    for name, operator, deg, variant in steps:
                        record = records.get(('study', dtype, operator, degree if deg is None else deg, variant, False))
                        times.append(metric(record, batch, field) if record else None)
                    masked = records.get(('study', dtype, 'tome', degree, 'shrink-masked', False))
                    masked = metric(masked, batch, field) if masked else None
                    if None in times:
                        continue
                    dense, shrink, merge, real = times
                    result[f'{field}/{dtype}/b{batch}/r{degree}'] = {'dense': dense, 'shrink': shrink, 'merge': merge, 'real': real, 'shrink_masked': masked}
                    print(f"{dtype} b{batch:<3} r{degree:<3} dense {dense:7.3f} -> shrink {shrink:7.3f} ({shrink - dense:+.3f}) -> +merge {merge:7.3f} ({merge - shrink:+.3f}) "
                          f"-> +mask {real:7.3f} ({real - merge:+.3f})   total {real - dense:+.3f} ms = {real / dense:.2f}x"
                          f"{'' if masked is None else f'   [mask on shrink: {masked - shrink:+.3f}]'}")
        print()
    return result


def compile_table(folder):
    print('## compile modes (study ViT, median ms per call; agreement with eager over 256 images)')
    for path in sorted(glob.glob(os.path.join(folder, '*.json'))):
        if '.contaminated' in path:
            continue
        record = json.loads(Path(path).read_text())
        print(f"\n{record['unit_id']}")
        eager = next((m for m in record['modes'] if m['mode'] == 'eager'), None)
        for mode in record['modes']:
            if 'error' in mode:
                print(f"  {mode['mode']:<38} ERROR {mode['error'][:100]}")
                continue
            ms = mode['timing']['median_ms']
            gain = f"{eager['timing']['median_ms'] / ms:5.2f}x vs eager" if eager and 'timing' in eager else ''
            agree = mode['agreement']
            extra = f" compile {mode['compile_seconds']:.0f}s" if 'compile_seconds' in mode else ''
            bit = f" bitwise={mode['bitwise_equal_to_eager']}" if 'bitwise_equal_to_eager' in mode else ''
            print(f"  {mode['mode']:<38} {ms:8.3f} ms {gain:<16} max|dlogit| {agree['max_abs_diff']:.1e} "
                  f"beyond tol {100 * agree['share_beyond_tolerance']:.1f}% top1 {100 * agree['top1_agreement']:.1f}%{extra}{bit}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('table', choices=['eager', 'ablation', 'compile'])
    parser.add_argument('folder')
    parser.add_argument('--old')
    parser.add_argument('--json')
    args = parser.parse_args()
    result = {'eager': lambda: eager_tables(args.folder, args.old), 'ablation': lambda: ablation_table(args.folder),
              'compile': lambda: compile_table(args.folder)}[args.table]()
    if args.json and result is not None:
        Path(args.json).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
