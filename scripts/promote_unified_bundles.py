#!/usr/bin/env python
"""Promote the two extinction-corrected unified bundles; keep the old one for rollback.

``current`` -> magnitude-independent QSO colours (the PI's baseline),
``current_magdep`` -> magnitude-dependent QSO colours, ``previous`` -> the
bundle that was active before. Each promoted bundle gets its support cut set
at a declared calibration retention, recomputed from its own role-2
calibration percentiles with the same rule as the original calibration.

Usage::

    python scripts/promote_unified_bundles.py --retention 0.995
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path('models/multisurvey_psf')
RUNS = {'current': ('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059', 'magnitude-independent QSO colours (baseline)'),
        'current_magdep': ('models/multisurvey_psf/work/unified_full/20261001/5f4002492dbb849c', 'magnitude-dependent QSO colours')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retention', type=float, required=True)
    args = parser.parse_args()
    old = json.loads((ROOT/'current').read_text())
    if not (ROOT/'previous').exists():
        (ROOT/'previous').write_text(json.dumps(dict(old, note='Active bundle before the 2 October 2026 promotion; rollback target.'), indent=2)+'\n')
    for pointer, (run, label) in RUNS.items():
        run = Path(run); tag = run.name; dest = ROOT/tag
        if not dest.exists():
            shutil.copytree(run/'bundle', dest)
        vals = np.concatenate([np.load(p)['percentile'] for p in sorted(glob.glob(str(run/'support_calibration_*.npz')))])
        vals = vals[np.isfinite(vals)]
        threshold = float(np.sort(vals)[int(np.floor((1-args.retention)*len(vals)))])
        support = json.loads((dest/'support.json').read_text())
        support.update(threshold=threshold, target_retention=args.retention,
                       calibration_retention=float(np.mean(vals >= threshold)),
                       informative_threshold=bool(threshold > 1/(support['draws']+1)),
                       threshold_selection=('Retention chosen from the release-check threshold sweep '
                                            '(docs/SUPPORT_CUT_SWEEP_2026-10-01.md) on the same test rows that report '
                                            'the result; confirm on independent rows before quoting performance.'))
        (dest/'support.json').write_text(json.dumps(support, indent=2)+'\n')
        manifest = json.loads((dest/'manifest.json').read_text())
        manifest['unified_files'] = {name: sha(dest/name) for name in ('latent.json', 'support.json')}
        manifest['status'] = f'promoted 2 October 2026 as {pointer}: {label}; extinction-corrected (SFD98); not full probability calibration'
        (dest/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        (ROOT/pointer).write_text(json.dumps(dict(bundle=tag, qso_colours=label, extinction='SFD98',
            validation='docs/SUPPORT_CUT_SWEEP_2026-10-01.md', release_reports=['docs/UNIFIED_RELEASE_BASELINE_2026-10-01.json',
            'docs/UNIFIED_RELEASE_MAGDEP_2026-10-02.json'], support_retention=args.retention,
            scope='PSF sources; any nonempty subset of 41 bands; extinction-corrected; spatial background; '
                  'Student-t catch-all; calibrated QSO-support cut. Load with UnifiedPSFModel.load.',
            full_probability_calibration=False), indent=2)+'\n')
        print(pointer, tag, 'threshold', threshold, 'calibration retention', support['calibration_retention'])


if __name__ == '__main__':
    main()
