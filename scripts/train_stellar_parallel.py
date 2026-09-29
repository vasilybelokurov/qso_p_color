#!/usr/bin/env python
"""Continue the stellar checkpoint with a bounded pool of E-step workers."""
import argparse
import json
import os
from pathlib import Path

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

from qso_pcolor.stellar_parallel_training import train_stellar_parallel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/full_sample_training.json'))
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--task-rows', type=int, default=8192)
    args = parser.parse_args()
    print('Stellar continuation:', train_stellar_parallel(json.loads(args.config.read_text()),
        resume_from=args.resume_from, workers=args.workers, task_rows=args.task_rows), flush=True)


if __name__ == '__main__':
    main()
