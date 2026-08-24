#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkg', default='CLASSIFIER')
    parser.add_argument('--id', required=True)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    repo = Path('/mnt/e/fyassine/ad-early-detection')
    py = repo / '.venv' / 'bin' / 'python'
    pkg_dir = repo / args.pkg
    out_dir = pkg_dir / 'outputs' / args.id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / 'dispatch.log'

    cmd = [str(py), 'run_experiment.py', '--id', args.id]
    if args.no_wandb:
        cmd.append('--no-wandb')

    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(pkg_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f'Launched {args.id} (pid {proc.pid}) -> {log_file}')

if __name__ == '__main__':
    main()
