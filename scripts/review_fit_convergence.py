#!/usr/bin/env python
"""Audit saved fits; --fit explicitly enables selective continuation and diagnostics."""
import argparse
import json
import os
from pathlib import Path
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

from qso_pcolor.convergence_review import run_review, progress_report
from qso_pcolor.full_sample import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/convergence_continuation.json'))
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()
    options = json.loads(args.config.read_text())
    if args.monitor:
        out = Path(options['output'])
        while True:
            report = json.loads((out/'audit.json').read_text())
            write_json(out/'monitor_progress.json', progress_report(out, report, options))
            execution = out/'execution.json'
            if execution.exists():
                state = json.loads(execution.read_text())
                if state['state'] != 'running':
                    return
                try:
                    os.kill(state['pid'], 0)
                except ProcessLookupError:
                    return
            time.sleep(60)
    else:
        print(run_review(options, fit=args.fit), flush=True)


if __name__ == '__main__':
    main()
