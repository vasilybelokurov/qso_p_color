#!/usr/bin/env python
"""Recalibrate a bundle's QSO-support cut under the faint limit (Legacy r reference).

The support percentile depends on the QSO model and on the reference band. With the faint limit
the reference is Legacy r, so the cut chosen with the old reference no longer applies. Same rule
as validate_unified_pilot.calibration: role-2 (calibration) QSOs per hemisphere, detected at
>=5 sigma with >=2 bands, inside the QSO redshift support; percentiles under the 'all' and
'legacy_optical' masks (SDSS-only or PS1-only objects have no Legacy r and are not scored);
threshold = the (1 - retention) quantile of the pooled finite percentiles. Rows failing the
faint limit are dropped, as the scorer would not score them.

Rewrites support.json and the manifest's unified_files hash in the bundle directory.
Usage: python scripts/method_unified/recalibrate_support.py --bundle DIR [--per-hemisphere 1000 --retention 0.995]
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qso_pcolor.full_sample import write_json
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.unified import UnifiedPSFModel, qso_support
from run_unified_pilot import arrays
from validate_unified_pilot import choose, mask_phot

RUN = Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059')
_s = {}


def _init(bundle):
    _s['model'] = UnifiedPSFModel.load(bundle).base.model; _s['data'] = arrays(RUN, 'qso')


def _support(args):
    rows, mask, draws, seed = args; m, d = _s['model'], _s['data']
    keep = np.array([True]*len(m.transform.bands)) if mask == 'all' else \
        np.array([b.startswith('decals_') and b.split(':')[1] in ('g', 'r', 'z') for b in m.transform.bands])
    phot = mask_phot(d, rows, m.transform.bands, keep)
    has, bright, _ = m.faint_limit_status(phot); use = has & bright & (phot.observed.sum(axis=1) >= 2)
    if not use.any():
        return np.array([])
    s = qso_support(m, phot.subset(use), d['zspec'][rows[use]], draws=draws, seed=seed,
                    l_deg=d['l'][rows[use]], b_deg=d['b'][rows[use]])
    return s['percentile']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bundle', type=Path, required=True); p.add_argument('--per-hemisphere', type=int, default=1000)
    p.add_argument('--retention', type=float, default=.995); p.add_argument('--workers', type=int, default=12); a = p.parse_args()
    support = json.loads((a.bundle/'support.json').read_text()); cfg = json.loads((RUN/'config.json').read_text())
    model = UnifiedPSFModel.load(a.bundle).base.model; data = arrays(RUN, 'qso')
    chosen = choose(data, 2, a.per_hemisphere, model.transform.bands, cfg['seed'] + 7, model.qso.support)
    rows = np.sort(np.concatenate(list(chosen.values())))
    jobs = [(rows[i:i+100], mask, support['draws'], support['seed']) for mask in ('all', 'legacy_optical') for i in range(0, len(rows), 100)]
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context('spawn'), initializer=_init, initargs=(str(a.bundle),)) as pool:
        parts = list(pool.map(_support, jobs))
    values = np.concatenate(parts); values = values[np.isfinite(values)]
    threshold = float(np.sort(values)[int(np.floor((1 - a.retention)*len(values)))])
    previous = dict(threshold=support['threshold'], target_retention=support.get('target_retention'))
    support.update(threshold=threshold, target_retention=a.retention, calibration_retention=float(np.mean(values >= threshold)),
                   calibration_rows=int(len(rows)), calibration_evaluations=int(len(values)), by_mask=['all', 'legacy_optical'],
                   row_indices_sha256=hashlib.sha256(data['source_row'][rows].tobytes()).hexdigest(),
                   informative_threshold=bool(threshold > 1/(support['draws'] + 1)), previous=previous,
                   faint_limit=model.meta.get('faint_limit'),
                   scope='Recalibrated under the faint limit (Legacy r reference) on role-2 QSOs, all and Legacy-optical masks')
    write_json(a.bundle/'support.json', support)
    manifest = json.loads((a.bundle/'manifest.json').read_text())
    manifest['unified_files'] = {n: hashlib.sha256((a.bundle/n).read_bytes()).hexdigest() for n in ('latent.json', 'support.json')}
    write_json(a.bundle/'manifest.json', manifest)
    print('SUPPORT', a.bundle, 'threshold', threshold, 'previous', previous['threshold'], 'evaluations', len(values),
          'retention', support['calibration_retention'], flush=True)


if __name__ == '__main__':
    main()
