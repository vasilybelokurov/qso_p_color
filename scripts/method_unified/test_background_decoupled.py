#!/usr/bin/env python
"""Light test of a background model with fixed colour shapes and magnitude-dependent amplitudes.

Each component has its own magnitude Gaussian (shared Legacy r latent coordinate) and zero
covariance between magnitude and colours (fit_projected decoupled_coordinate), so conditioning
on magnitude changes only component weights. Fitted on a random subsample of fit/select
background rows from the extinction-corrected baseline run, warm-started from the current
background with its magnitude-colour cross terms removed; MAP update; held-out density on the
background stopping panel every 20 updates. Writes nothing into production runs.

Usage::

    python scripts/method_unified/test_background_decoupled.py --rows 150000 --out models/multisurvey_psf/work/background_decoupled/test1
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.projected_parallel import ProjectedBatchFactory, ProjectedParallelAccumulator
from qso_pcolor.projected_xd import fit_projected
from run_unified_pilot import arrays, magnitude_colour_matrix, operator, stopping_density

ROOT = Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=int, default=150000); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-iter', type=int, default=200); p.add_argument('--workers', type=int, default=8)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT/'config.json').read_text()); layout = json.loads((ROOT/'layout.json').read_text())
    d = arrays(ROOT, 'stars'); rng = np.random.default_rng(20261002)
    fit_rows = np.flatnonzero(np.isin(d['role'], cfg['fit_roles']))
    rows = np.sort(rng.choice(fit_rows, a.rows, replace=False))
    t = magnitude_colour_matrix(layout['latent_labels'], 'legacy:r'); k = layout['latent_labels'].index('legacy:r')
    current = GaussianMixture.from_dict(json.loads((ROOT/'bundle'/'latent.json').read_text())['background'])
    means = current.means @ t.T; covs = t @ current.covs @ t.T
    rest = np.arange(len(t)) != k; covs[:, k, rest] = 0.; covs[:, rest, k] = 0.
    init = GaussianMixture(current.weights, means, covs)
    h, b, tt = operator(layout, 'stars'); op = (h @ np.linalg.inv(t), b, tt)
    lay_u = dict(layout, operators=dict(layout['operators'], stars=dict(matrix=op[0].tolist(), offset=b.tolist(), covariance=tt.tolist())))
    panel = dict(np.load(ROOT/'stopping'/'stars_00.npz'))
    base = stopping_density(current, layout, 'stars', d, panel, 128)
    factory = ProjectedBatchFactory(ROOT/'stars', rows, cfg['batch_size']); source = lambda: factory(0, len(rows))
    trace = [dict(iteration=0, heldout=stopping_density(init, lay_u, 'stars', d, panel, 128))]
    print('current background held-out', round(base, 4), '| decoupled warm start', round(trace[0]['heldout'], 4), flush=True)
    history, mix, it, best = [], init, 0, (trace[0]['heldout'], 0, init); start = time.time()
    with ProjectedParallelAccumulator(factory, len(rows), workers=a.workers, task_rows=8192) as acc:
        while it < a.max_iter:
            f = fit_projected(source, init=mix, operators={0: op}, expected_rows=len(rows), max_iter=it + 20, tol=cfg['tol'],
                              regularization=cfg['regularization'], covariance_update='map', initial_history=tuple(history),
                              accumulator=acc, final_evaluation=False, decoupled_coordinate=k)
            history, mix, it = list(f.history), f.mixture, f.n_iter
            v = stopping_density(mix, lay_u, 'stars', d, panel, 128); trace.append(dict(iteration=it, heldout=v))
            if v > best[0]: best = (v, it, mix)
            print(f'update {it}: held-out {v:.4f}  ({time.time() - start:.0f} s)', flush=True)
            if f.converged or v - trace[-2]['heldout'] < cfg['predictive_stopping']['min_gain']:
                break
    (a.out/'result.json').write_text(json.dumps(dict(rows=int(len(rows)), current_heldout=base, trace=trace, best_iteration=best[1],
        mixture_u=best[2].to_dict(), magnitude_index=k, transform=t.tolist(), operator=dict(matrix=op[0].tolist(), offset=b.tolist(),
        covariance=tt.tolist())), indent=1))
    print('best', round(best[0], 4), 'at', best[1], '| current', round(base, 4), flush=True)


if __name__ == '__main__':
    main()
