#!/usr/bin/env python
"""Preflight full-sample inputs by default; fitting requires explicit --fit."""
import argparse
import json
from pathlib import Path
import resource
import sys

from qso_pcolor.full_sample import write_json
from qso_pcolor.full_training import preflight, train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/full_sample_training.json'))
    p.add_argument('--fit', action='store_true', help='Launch the full density fits; requires the agreed readiness review')
    p.add_argument('--benchmark', action='store_true', help='Time a few E-step batches without updating any mixture')
    p.add_argument('--report', type=Path, default=Path('docs/FULL_TRAINING_PREFLIGHT_2026-09-29.json'))
    a = p.parse_args(); cfg = json.loads(a.config.read_text())
    report = preflight(cfg, benchmark=a.benchmark)
    report['peak_rss_native'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['rss_units'] = 'bytes on macOS; KiB on Linux'
    report['peak_rss_bytes'] = report['peak_rss_native'] * (1 if sys.platform == 'darwin' else 1024)
    report['memory']['measured_preflight_within_budget'] = report['peak_rss_bytes'] <= cfg['memory_budget_gib'] * 1024**3
    report['density_fit_ready'] &= report['memory']['measured_preflight_within_budget']
    write_json(a.report, report)
    print('Full-data preflight saved:', a.report, flush=True)
    if a.fit:
        if not report['density_fit_ready']:
            raise ValueError('preflight did not pass; refusing production fit')
        print('Candidate fits:', train(cfg), flush=True)


if __name__ == '__main__':
    main()
