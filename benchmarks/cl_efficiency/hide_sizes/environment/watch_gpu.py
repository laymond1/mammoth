"""Read GPU occupancy without modifying or interrupting any process."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    args = parser.parse_args()
    uuid = 'GPU-1fb8c05f-24e7-75d1-5cf9-0f47df8e46ad'
    with args.output.open('a', buffering=1) as stream:
        while not args.stop_file.exists():
            row = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'apps': []}
            query = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            if query.returncode:
                row['query_error'] = query.returncode
            else:
                for line in query.stdout.splitlines():
                    fields = [field.strip() for field in line.split(',')]
                    if len(fields) != 4 or fields[0] != uuid:
                        continue
                    pid = int(fields[1])
                    try:
                        cwd = Path('/proc') / str(pid) / 'cwd'
                        own = cwd.resolve() == args.workspace.resolve()
                    except OSError:
                        own = None
                    row['apps'].append({'pid': pid, 'process': fields[2], 'memory_mib': fields[3], 'workspace_owned': own})
            stream.write(json.dumps(row) + '\n')
            time.sleep(1)


if __name__ == '__main__':
    main()
