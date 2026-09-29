#!/usr/bin/env python
"""Continue a prepared full-sample run with parallel independent QSO slices."""
import argparse
import json
import os
from pathlib import Path

# Set before importing NumPy or spawning processes, including on continuation.
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'

from qso_pcolor.parallel_training import train_parallel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/full_sample_training.json'))
    parser.add_argument('--qso-workers', type=int, default=4)
    args = parser.parse_args()
    print('Continuing with', args.qso_workers, 'QSO slice workers and the stellar fit', flush=True)
    print('Candidate fits:', train_parallel(json.loads(args.config.read_text()), workers=args.qso_workers), flush=True)


if __name__ == '__main__':
    main()
