"""Read back uploaded HiDe results and compare immutable artifact file hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(result_path, output):
    from scripts.cl_runtime import configure_environment
    from scripts.cl_efficiency import write_json
    configure_environment()
    import wandb
    record = json.loads(result_path.read_text())
    parsed = urlparse(record['wandb_url'])
    parts = parsed.path.strip('/').split('/')
    if parsed.hostname != 'wandb.ai' or len(parts) != 4 or parts[2] != 'runs' or parts[:2] != ['laymond1', 'cl-efficiency']:
        raise ValueError('HiDe readback requires the configured W&B project')
    api = wandb.Api(timeout=60)
    run = api.run('/'.join([parts[0], parts[1], parts[3]]))
    if run.state != 'finished':
        raise ValueError('Uploaded W&B run is not finished')
    artifact_pin = record.get('wandb_artifact')
    artifact_name = artifact_pin['name'] if isinstance(artifact_pin, dict) else artifact_pin
    artifact = api.artifact(artifact_name)
    if artifact.state != 'COMMITTED':
        raise ValueError('HiDe artifact is not committed')
    local = result_path.parent
    pin_path = local / 'artifact_files.json'
    pins = json.loads(pin_path.read_text()) if pin_path.is_file() else {
        name: {'local_file': 'artifact_metrics.json' if name == 'metrics.json' else name,
               'sha256': digest(local / ('artifact_metrics.json' if name == 'metrics.json' else name))}
        for name in artifact.manifest.entries}
    output.mkdir(parents=True, exist_ok=True)
    downloaded = Path(artifact.download(root=str(output / 'download')))
    checks = {}
    for name, pin in pins.items():
        local_path = Path(pin['local_file'])
        if not local_path.is_absolute():
            local_path = local / local_path
        expected = pin['sha256']
        actual = digest(downloaded / name)
        checks[name] = {'sha256': actual, 'matches_uploaded_pin': actual == expected,
                        'local_snapshot_matches': digest(local_path) == expected}
    if not checks or any(not row['matches_uploaded_pin'] or not row['local_snapshot_matches'] for row in checks.values()):
        raise ValueError('HiDe artifact readback differs from immutable local snapshots')
    summary = dict(run.summary)
    receipt = {'status': 'pass', 'wandb_url': record['wandb_url'], 'run_state': run.state,
               'artifact': artifact.qualified_name, 'artifact_state': artifact.state,
               'result_file': str(result_path), 'result_sha256': digest(result_path),
               'artifact_files': checks,
               'summary': {key: value for key, value in summary.items()
                           if isinstance(value, (int, float, bool, str)) and not key.startswith('_')}}
    write_json(output / 'readback.json', receipt)
    print(json.dumps({'status': 'pass', 'wandb_url': record['wandb_url'],
                      'artifact_files_verified': len(checks), 'receipt': str(output / 'readback.json')}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args(argv)
    if not options.execute:
        print('HiDe artifact readback prepared; no credentials or network used.')
        return 0
    return verify(options.result.resolve(), options.output.resolve())


if __name__ == '__main__':
    sys.exit(main())
