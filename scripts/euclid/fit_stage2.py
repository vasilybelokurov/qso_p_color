#!/usr/bin/env python
"""Stage 2 of docs/euclid/PLAN.md: joint refit of every mixture in the 40-d latent space (Euclid bands included).

Warm start: the stage-1 bundle's extended latent mixtures. Fitter: qso_pcolor.projected_xd.fit_projected with the
MAP covariance update (regularization 0.001 mag^2, prior strength 1), as the parent. Rows (parent roles, unchanged):
- QSO slice s: parent fit/select rows (roles 0, 1) with |z - z_s| < 0.1 (the parent's z half-width);
- background bin j: the parent binned design's training rows (stellar_binned/20261003/rows.npz 'train') with
  Legacy r inside the bin +- 0.25 mag (the parent's overlap).
Euclid columns are attached (dereddened luptitudes) only for rows that pass the deployment selection AND lie in
Euclid FIT cells; elsewhere they are unobserved. Model B is fitted in (magnitude, colour) coordinates with the
magnitude coordinate fixed (N(20, 100)), as the parent.
Predictive stopping: blocks of 10 updates (max 40); after each block the mean log likelihood of a held-out panel
(QSOs: parent select-role rows of the slice; background: the binned design's 'stop' rows of the bin) is computed
and the best mixture (including the warm start) is kept; stop when a block gains < 0.002 nat per object.

    python scripts/euclid/fit_stage2.py --bundle 20261005_magdep_qlf --workers 10
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import numpy as np

MODELS = Path('models/multisurvey_psf')
RUN = MODELS/'work/unified_full/20261001/5f4002492dbb849c'
BINNED = MODELS/'work/stellar_binned/20261003'
BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1'
EDGES = np.array([15.5, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.5]); OVERLAP = 0.25
Z_HALF = 0.1; BLOCK, MAX_ITER, MIN_GAIN = 10, 40, 0.002


def euclid_columns(kind, settings, roles):
    """(parent_row -> Euclid luptitude/noise) for deployment-selected rows in Euclid fit cells."""
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from fit_stage1 import load_rows
    d = load_rows('qso' if kind == 'qso' else 'background', settings)
    fitcell = np.array([roles.get(int(c)) == 'fit' for c in d['euclid_cell']])
    use = d['clean'] & fitcell
    return d['parent_row'][use], d['y'][use, 10:], d['noise'][use, 10:], d['observed'][use, 10:]


def euclid_colour_map(n_dim, k, n_euclid=4):
    """U: identity except the last n_euclid rows, x_E -> x_E - x_k (Euclid magnitude -> colour vs coordinate k)."""
    u = np.eye(n_dim)
    u[n_dim - n_euclid:, k] = -1.
    return u


def check_fixed(mix, fixed, name, tol=1e-6):
    """The warm start must already satisfy the fixed-coordinate rule; otherwise the refit would start elsewhere."""
    k = fixed['index']; other = np.delete(mix.covs[:, k, :], k, axis=1)
    bad = (np.abs(mix.means[:, k] - fixed['mean']).max(), np.abs(mix.covs[:, k, k] - fixed['variance']).max(),
           np.abs(other).max())
    if max(bad) > tol*max(1., fixed['variance']):
        raise ValueError(f'{name}: warm start violates the fixed coordinate (mean, var, cross) = {bad}')


def batches(y, noise, obs, size=128):
    def source():
        for lo in range(0, len(y), size):
            v = noise[lo:lo+size]; cov = np.zeros((len(v), v.shape[1], v.shape[1]))
            cov[:, np.arange(v.shape[1]), np.arange(v.shape[1])] = v
            yield y[lo:lo+size], cov, obs[lo:lo+size], np.zeros(len(v), int)
    return source


def load_task_rows(kind, rows, euc):
    arr = RUN/('qso' if kind == 'qso' else 'stars')
    y = np.asarray(np.load(arr/'y.npy', mmap_mode='r')[rows]); n = np.asarray(np.load(arr/'noise.npy', mmap_mode='r')[rows])
    o = np.asarray(np.load(arr/'observed.npy', mmap_mode='r')[rows])
    ey = np.zeros((len(rows), 4)); en = np.ones((len(rows), 4)); eo = np.zeros((len(rows), 4), bool)
    prow, yy, nn, oo = euc
    pos = np.searchsorted(prow, rows); hit = (pos < len(prow)) & (prow[np.minimum(pos, len(prow) - 1)] == rows)
    ey[hit] = np.nan_to_num(yy[pos[hit]]); en[hit] = np.where(oo[pos[hit]], nn[pos[hit]], 1.); eo[hit] = oo[pos[hit]]
    y = np.where(o, y, 0.); n = np.where(o, n, 1.)
    return np.c_[y, ey], np.c_[n, en], np.c_[o, eo], int(hit.sum())


def fit_one(task):
    from qso_pcolor.gaussmix import GaussianMixture
    from qso_pcolor.projected_xd import fit_projected, mean_log_likelihood
    out = Path(task['out'])/f"{task['name']}.json"
    if out.exists():
        return task['name']
    euc = tuple(np.load(task['euclid'])[k] for k in ('prow', 'y', 'noise', 'obs'))
    y, n, o, n_euc = load_task_rows(task['kind'], np.asarray(task['rows']), euc)
    hy, hn, ho, _ = load_task_rows(task['kind'], np.asarray(task['held']), euc)
    op = {0: tuple(np.asarray(x) for x in task['operator'])}
    init = GaussianMixture.from_dict(task['init'])
    fixed = task.get('fixed')
    best, best_ll = init, mean_log_likelihood(batches(hy, hn, ho), init, op, len(hy)) if len(hy) else np.nan
    trace = [dict(iteration=0, heldout=float(best_ll))]; mix = init; hist = []; it = 0; t0 = time.monotonic()
    while it < MAX_ITER:
        fit = fit_projected(batches(y, n, o), init=mix, operators=op, expected_rows=len(y), max_iter=it + BLOCK, tol=1e-6,
                            regularization=1e-3, covariance_update='map', prior_strength=1., final_evaluation=False,
                            initial_history=tuple(hist), fixed_coordinate=fixed)
        mix = fit.mixture; hist = list(fit.history)
        if len(hist) <= it and fit.converged:
            break
        if len(hist) <= it:
            raise RuntimeError(f"{task['name']}: no progress in a block (history {len(hist)} after {it})")
        it = len(hist)
        ll = mean_log_likelihood(batches(hy, hn, ho), mix, op, len(hy)) if len(hy) else np.nan
        trace.append(dict(iteration=it, heldout=float(ll)))
        gain = ll - trace[-2]['heldout']
        if ll > best_ll:
            best, best_ll = mix, ll
        if not np.isfinite(gain) or gain < MIN_GAIN or fit.converged:
            break
    Path(out).write_text(json.dumps(dict(task=task['name'], mixture=best.to_dict(), trace=trace, n_rows=len(y),
                                         n_euclid_rows=n_euc, n_heldout=len(hy), iterations=it,
                                         elapsed_seconds=time.monotonic() - t0)))
    return task['name']


def main():
    from qso_pcolor.gaussmix import GaussianMixture
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', required=True); ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--tasks', nargs='*', help='only these task names (timing)')
    a = ap.parse_args()
    s = json.loads(Path('docs/euclid/settings.json').read_text())
    roles = {c['cell']: c['role'] for c in json.loads(Path('docs/euclid/sky_roles.json').read_text())['cells']}
    st1 = MODELS/f'euclid_stage1_{a.bundle}'; lat = json.loads((st1/'latent.json').read_text()); lay = lat['layout']
    out = MODELS/'work/euclid'/f'stage2_{a.bundle}'; (out/'fits').mkdir(parents=True, exist_ok=True)
    for kind in ('qso', 'stars'):
        prow, yy, nn, oo = euclid_columns(kind, s, roles); order = np.argsort(prow)
        np.savez(out/f'euclid_{kind}.npz', prow=prow[order], y=yy[order], noise=nn[order], obs=oo[order])
    model = json.loads((st1/'model.json').read_text())
    zc = np.asarray(model['qso']['z_centres'], float)
    zs = np.load(RUN/'qso/zspec.npy'); role = np.load(RUN/'qso/role.npy')
    tasks = []
    qc = lay.get('qso_coordinates')
    hq = np.asarray(lay['operators']['qso']['matrix']); bq = np.asarray(lay['operators']['qso']['offset'])
    tq = np.asarray(lay['operators']['qso']['covariance'])
    for j, m in enumerate(lat['qso']):
        mix = GaussianMixture.from_dict(m); near = np.abs(zs - zc[j]) < Z_HALF
        rows = np.flatnonzero(near & np.isin(role, [0, 1])); held = np.flatnonzero(near & (role == 1))
        held = held[np.linspace(0, len(held) - 1, min(len(held), 3000)).astype(int)] if len(held) else held
        if qc is not None:
            # Model A: the stage-1 latent is (magnitude, colour) coordinates for the parent block (the layout's QSO
            # operator already contains inv(T)) plus Euclid MAGNITUDES. Only the Euclid rows are moved to colours
            # relative to the magnitude coordinate, so that the fixed-coordinate rule holds for the whole vector.
            U = euclid_colour_map(mix.n_dim, qc['index'])
            means = mix.means @ U.T; covs = U @ mix.covs @ U.T; k = qc['index']
            init = GaussianMixture(mix.weights, means, 0.5*(covs + covs.swapaxes(1, 2)), tuple(qc['labels']))
            opm = hq @ np.linalg.inv(U); fixed = dict(index=k, mean=qc['mean'], variance=qc['variance'])
            check_fixed(init, fixed, f'qso_{j:02d}')
        else:
            init, opm, fixed = mix, hq, None
        tasks.append(dict(name=f'qso_{j:02d}', kind='qso', rows=rows.tolist(), held=held.tolist(), init=init.to_dict(),
                          operator=[opm.tolist(), bq.tolist(), tq.tolist()], fixed=fixed, out=str(out/'fits'),
                          euclid=str(out/'euclid_qso.npz')))
    binned = np.load(BINNED/'rows.npz'); train, stop = binned['train'], binned['stop']
    ys = np.load(RUN/'stars/y.npy', mmap_mode='r'); obs = np.load(RUN/'stars/observed.npy', mmap_mode='r')
    lab = lay['native_labels'][:41]; si, ni = lab.index('decals_dr9_south:r'), lab.index('decals_dr9_north:r')
    def rref(rows):
        o = np.asarray(obs[rows]); y = np.asarray(ys[rows])
        return np.where(o[:, si], y[:, si], np.where(o[:, ni], y[:, ni], np.nan))
    rt, rs = rref(train), rref(stop)
    bmix = GaussianMixture.from_dict(lat['background']); r_index = lay['latent_labels'].index('legacy:r')
    comp_bin = np.clip(np.searchsorted(EDGES, bmix.means[:, r_index]) - 1, 0, len(EDGES) - 2)
    hb = [np.asarray(lay['operators']['stars'][k]) for k in ('matrix', 'offset', 'covariance')]
    for j in range(len(EDGES) - 1):
        comps = np.flatnonzero(comp_bin == j)
        init = GaussianMixture(bmix.weights[comps]/bmix.weights[comps].sum(), bmix.means[comps], bmix.covs[comps], bmix.labels)
        inb = lambda r: (r >= EDGES[j] - OVERLAP) & (r < EDGES[j+1] + OVERLAP)
        tasks.append(dict(name=f'bkg_{j}', kind='stars', rows=np.sort(train[inb(rt)]).tolist(), held=np.sort(stop[inb(rs)]).tolist(),
                          init=init.to_dict(), operator=[x.tolist() for x in hb], fixed=None, components=comps.tolist(),
                          share=float(bmix.weights[comps].sum()), out=str(out/'fits'), euclid=str(out/'euclid_stars.npz')))
    (out/'tasks.json').write_text(json.dumps([{k: v for k, v in t.items() if k not in ('rows', 'held', 'init', 'operator')}
                                              | dict(n_rows=len(t['rows']), n_held=len(t['held'])) for t in tasks], indent=1))
    print('tasks', len(tasks), 'rows', sum(len(t['rows']) for t in tasks), flush=True)
    if a.tasks:
        tasks = [t for t in tasks if t['name'] in a.tasks]
    order = sorted(tasks, key=lambda t: -len(t['rows'])*len(t['init']['weights']))
    with ProcessPoolExecutor(a.workers) as pool:
        futs = {pool.submit(fit_one, t): t['name'] for t in order}
        for f in as_completed(futs):
            print(time.strftime('%H:%M:%S'), 'done', f.result(), flush=True)
    print('STAGE2 COMPLETE', flush=True)


if __name__ == '__main__':
    main()
