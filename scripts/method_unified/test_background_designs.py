#!/usr/bin/env python
"""Three-way light test of the background (non-QSO) colour model across magnitude.

Models, all in the latent (Legacy r, colours) coordinates of the extinction-corrected
baseline run and all with all-sky (global) weights:

  current  the promoted joint K=20 background (not refitted; reference).
  binned   XDQSO-style: an independent K=20 XD fit in each Legacy-r bin (Bovy et al. 2011,
           ApJ 729, 141). Training rows extend 0.25 mag beyond each bin edge; an object is
           scored with the bin that contains its r.
  tied     shared colour shapes: K=20 colour Gaussians, each replicated at fixed magnitude
           nodes (Gaussian in r, centre = node, sigma = NODE_SIGMA), magnitude-colour
           covariance zero, colour mean/covariance tied across replicas; only the replica
           weights are free (fit_projected tied_coordinate).

Data: fit/select (roles 0, 1) eligible background rows, drawn up to ROWS_PER_BIN per bin so
that the bright end is not swamped (the conditional colour density does not depend on the
magnitude distribution of the sample, only on the colour distribution at each magnitude).
Held-out rows: role 3, eligible, outside the production stopping panel; per bin a stopping
set (early stopping, 20-update blocks, min gain 0.02 nat) and a disjoint evaluation set.
Held-out score: ln p(other observed bands | Legacy r), per object, as in the production
stopping panels. These held-out rows must be excluded from later final assessments.

Usage (from the repository root)::

    python scripts/method_unified/test_background_designs.py prepare --out models/multisurvey_psf/work/background_designs/test1
    python scripts/method_unified/test_background_designs.py binned  --out ... --workers 8
    python scripts/method_unified/test_background_designs.py tied    --out ... --workers 10
    python scripts/method_unified/test_background_designs.py score   --out ...
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
from qso_pcolor.multisurvey import conditional_log_prob
from qso_pcolor.projected_parallel import ProjectedBatchFactory, ProjectedParallelAccumulator
from qso_pcolor.projected_xd import fit_projected, native_mixture
from run_unified_pilot import arrays, magnitude_colour_matrix, operator

ROOT = Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059')
EDGES = [15.5, 17., 18., 19., 20., 21., 22., 23., 24.5]       # Legacy r [mag], extinction corrected
OVERLAP = .25                                                   # mag added to each side for binned training
ROWS_PER_BIN, STOP_PER_BIN, EVAL_PER_BIN = 12000, 1000, 2500
NODE_SIGMA = .6                                                 # mag
BLOCK, MIN_GAIN, MAX_ITER = 20, .02, 200
S, N = 'decals_dr9_south:r', 'decals_dr9_north:r'


def setup():
    cfg = json.loads((ROOT/'config.json').read_text()); layout = json.loads((ROOT/'layout.json').read_text())
    d = arrays(ROOT, 'stars'); lab = layout['native_labels']; s, n = lab.index(S), lab.index(N)
    obs = np.asarray(d['observed'][:, [s, n]]); y = np.asarray(d['y'][:, [s, n]])
    r = np.where(obs[:, 0], y[:, 0], np.where(obs[:, 1], y[:, 1], np.nan)); anchor = np.where(obs[:, 0], s, n)
    t = magnitude_colour_matrix(layout['latent_labels'], 'legacy:r'); k = layout['latent_labels'].index('legacy:r')
    h, b, tt = operator(layout, 'stars'); op = (h @ np.linalg.inv(t), b, tt)
    current = GaussianMixture.from_dict(json.loads((ROOT/'bundle'/'latent.json').read_text())['background'])
    current_u = GaussianMixture(current.weights, current.means @ t.T, t @ current.covs @ t.T)
    return cfg, layout, d, r, anchor, op, k, current_u


def bin_of(r):
    return np.clip(np.searchsorted(EDGES, r, side='right') - 1, 0, len(EDGES) - 2)


def prepare(out):
    cfg, layout, d, r, anchor, op, k, cur = setup(); rng = np.random.default_rng(20261003)
    role, ok = np.asarray(d['role']), np.asarray(d['eligible']) & np.isfinite(r)
    panel = np.load(ROOT/'stopping'/'stars_00.npz')['rows']
    pool = ok & np.isin(role, cfg['fit_roles']); held = ok & (role == 3); held[panel] = False
    sets = dict(train=[], stop=[], eval=[]); counts = []
    for j, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        inb = (r >= lo) & (r < hi); a = np.flatnonzero(pool & inb); hh = np.flatnonzero(held & inb)
        tr = rng.choice(a, min(ROWS_PER_BIN, len(a)), replace=False); hh = rng.permutation(hh)
        sets['train'].append(tr); sets['stop'].append(hh[:STOP_PER_BIN]); sets['eval'].append(hh[STOP_PER_BIN:STOP_PER_BIN + EVAL_PER_BIN])
        counts.append(dict(bin=[lo, hi], pool=int(len(a)), train=int(len(tr)), held=int(len(hh)),
                           stop=int(len(sets['stop'][-1])), eval=int(len(sets['eval'][-1]))))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'rows.npz', **{key: np.sort(np.concatenate(v)) for key, v in sets.items()})
    (out/'prepare.json').write_text(json.dumps(dict(root=str(ROOT), edges=EDGES, overlap=OVERLAP, node_sigma=NODE_SIGMA,
        counts=counts, note='stop/eval rows are role 3: exclude from later final assessments'), indent=1))
    for c in counts:
        print(c)


def heldout(native_for_rows, rows, d, anchor, r, batch=256):
    """Per-row ln p(other observed bands | Legacy r). ``native_for_rows(rows)`` yields (mask, native mixture)."""
    out = np.full(len(rows), np.nan)
    for sel, nat in native_for_rows(r[rows]):
        ix = np.flatnonzero(sel)
        for lo in range(0, len(ix), batch):
            ii = ix[lo:lo + batch]; rr = rows[ii]; v = np.asarray(d['noise'][rr])
            cov = np.zeros((len(rr), v.shape[1], v.shape[1])); cov[:, np.arange(v.shape[1]), np.arange(v.shape[1])] = v
            for a in np.unique(anchor[rr]):
                t = anchor[rr] == a
                out[ii[t]] = conditional_log_prob(nat, np.asarray(d['y'][rr[t]]), cov[t], np.asarray(d['observed'][rr[t]]), int(a))
    if not np.isfinite(out).all():
        raise ValueError('nonfinite held-out density')
    return out


def fit_blocks(rows, init, op, cfg, workers, score, label, **constraint):
    """MAP XD in 20-update blocks; keep the best held-out checkpoint (including the warm start)."""
    factory = ProjectedBatchFactory(ROOT/'stars', rows, cfg['batch_size']); source = lambda: factory(0, len(rows))
    trace = [dict(iteration=0, heldout=score(init))]; best = (trace[0]['heldout'], 0, init)
    history, mix, it, start = [], init, 0, time.time()
    print(f'{label}: {len(rows)} rows, {init.n_components} components, warm start {trace[0]["heldout"]:.4f}', flush=True)
    with ProjectedParallelAccumulator(factory, len(rows), workers=workers, task_rows=4096) as acc:
        while it < MAX_ITER:
            f = fit_projected(source, init=mix, operators={0: op}, expected_rows=len(rows), max_iter=it + BLOCK, tol=cfg['tol'],
                              regularization=cfg['regularization'], covariance_update='map', initial_history=tuple(history),
                              accumulator=acc, final_evaluation=False, **constraint)
            history, mix, it = list(f.history), f.mixture, f.n_iter
            v = score(mix); trace.append(dict(iteration=it, heldout=v))
            if v > best[0]:
                best = (v, it, mix)
            print(f'{label} update {it}: held-out {v:.4f} ({time.time() - start:.0f} s)', flush=True)
            if f.converged or v - trace[-2]['heldout'] < MIN_GAIN:
                break
    return best, trace


def fit_binned(out, workers):
    cfg, layout, d, r, anchor, op, k, cur = setup(); sets = dict(np.load(out/'rows.npz')); res = []
    train = sets['train']; stop = sets['stop']
    for j, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        path = out/f'binned_{j}.json'
        if path.exists():
            continue
        rows = train[(r[train] >= lo - OVERLAP) & (r[train] < hi + OVERLAP)]
        srows = stop[bin_of(r[stop]) == j]
        means = cur.means.copy(); covs = cur.covs.copy(); rest = np.arange(cur.n_dim) != k
        means[:, k] = .5*(lo + hi); covs[:, k, rest] = 0.; covs[:, rest, k] = 0.; covs[:, k, k] = (hi - lo + 2*OVERLAP)**2/12
        init = GaussianMixture(np.full(cur.n_components, 1/cur.n_components), means, covs)
        score = lambda m: float(heldout(lambda rr: [(np.ones(len(rr), bool), native_mixture(m, *op))], srows, d, anchor, r).mean())
        best, trace = fit_blocks(rows, init, op, cfg, workers, score, f'binned {lo}-{hi}')
        path.write_text(json.dumps(dict(bin=[lo, hi], rows=int(len(rows)), best_iteration=best[1], trace=trace,
                                        mixture_u=best[2].to_dict()), indent=1))


def tied_init(cur, k):
    nodes = .5*(np.array(EDGES[:-1]) + np.array(EDGES[1:])); kk, m = cur.n_components, len(nodes)
    rest = np.arange(cur.n_dim) != k; means = np.repeat(cur.means, m, axis=0); covs = np.repeat(cur.covs, m, axis=0)
    means[:, k] = np.tile(nodes, kk); covs[:, k, rest] = 0.; covs[:, rest, k] = 0.; covs[:, k, k] = NODE_SIGMA**2
    return GaussianMixture(np.full(kk*m, 1/(kk*m)), means, covs), np.repeat(np.arange(kk), m), nodes


def fit_tied(out, workers):
    cfg, layout, d, r, anchor, op, k, cur = setup(); sets = dict(np.load(out/'rows.npz'))
    init, group, nodes = tied_init(cur, k)
    score = lambda m: float(np.mean([heldout(lambda rr: [(np.ones(len(rr), bool), native_mixture(m, *op))],
                                             sets['stop'][bin_of(r[sets['stop']]) == j], d, anchor, r).mean()
                                     for j in range(len(EDGES) - 1)]))
    best, trace = fit_blocks(sets['train'], init, op, cfg, workers, score, 'tied',
                             tied_coordinate=dict(index=k, group=group))
    (out/'tied.json').write_text(json.dumps(dict(rows=int(len(sets['train'])), nodes=nodes.tolist(), node_sigma=NODE_SIGMA,
        group=group.tolist(), best_iteration=best[1], trace=trace, mixture_u=best[2].to_dict()), indent=1))


def load_models(out):
    """Name -> function(r array) -> list of (row mask, native mixture)."""
    cfg, layout, d, r, anchor, op, k, cur = setup(); nb = len(EDGES) - 1
    models = dict(current=lambda rr: [(np.ones(len(rr), bool), native_mixture(cur, *op))])
    if all((out/f'binned_{j}.json').exists() for j in range(nb)):
        bins = [native_mixture(GaussianMixture.from_dict(json.loads((out/f'binned_{j}.json').read_text())['mixture_u']), *op)
                for j in range(nb)]
        models['binned'] = lambda rr: [(bin_of(rr) == j, bins[j]) for j in range(nb)]
    if (out/'tied.json').exists():
        tied = native_mixture(GaussianMixture.from_dict(json.loads((out/'tied.json').read_text())['mixture_u']), *op)
        models['tied'] = lambda rr: [(np.ones(len(rr), bool), tied)]
    return models, (cfg, layout, d, r, anchor)


def score_all(out):
    models, (cfg, layout, d, r, anchor) = load_models(out); ev = np.load(out/'rows.npz')['eval']; jb = bin_of(r[ev])
    panel = dict(np.load(ROOT/'stopping'/'stars_00.npz'))
    report = dict(edges=EDGES, models={})
    for name, f in models.items():
        v = heldout(f, ev, d, anchor, r)
        per = [float(v[jb == j].mean()) for j in range(len(EDGES) - 1)]
        # production stopping panel (natural magnitude distribution, its own reference bands)
        pr = panel['rows']; out_p = np.empty(len(pr))
        for sel, nat in f(r[pr]):
            for a in np.unique(panel['anchors'][sel]):
                t = sel & (panel['anchors'] == a); rr = pr[t]; vv = np.asarray(d['noise'][rr])
                cov = np.zeros((len(rr), vv.shape[1], vv.shape[1])); cov[:, np.arange(vv.shape[1]), np.arange(vv.shape[1])] = vv
                out_p[t] = conditional_log_prob(nat, np.asarray(d['y'][rr]), cov, np.asarray(d['observed'][rr]), int(a))
        report['models'][name] = dict(per_bin=per, mean_over_bins=float(np.mean(per)), production_panel=float(out_p.mean()))
        print(name, 'per bin', np.round(per, 3), '| mean over bins', round(float(np.mean(per)), 4),
              '| production stopping panel', round(float(out_p.mean()), 4), flush=True)
    (out/'score.json').write_text(json.dumps(report, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=('prepare', 'binned', 'tied', 'score')); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8); a = p.parse_args()
    dict(prepare=lambda: prepare(a.out), binned=lambda: fit_binned(a.out, a.workers), tied=lambda: fit_tied(a.out, a.workers),
         score=lambda: score_all(a.out))[a.step]()


if __name__ == '__main__':
    main()
