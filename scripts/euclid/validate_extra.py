#!/usr/bin/env python
"""Stage-1 diagnostics T-V2, T-V4, T-V8, T-V9 (docs/euclid/PLAN.md) on seeded subsamples of the Euclid test roles.

T-V2  predictive calibration, Euclid only: QSO support percentile at the true z (a PIT of the conditional density of
      the non-reference bands; uniform if calibrated) and background outside-both p_B (uniform if calibrated);
      fractions in [0, .25), [.25, .5), [.5, .75), [.75, 1] with binomial 99% intervals.
T-V4  window sensitivity: p_quasar with the Legacy-r window half-widths x0.5 and x1.5, all / red background
      (VIS - Y > 1) / z > 3 QSOs; median and 90th percentile |dp|.
T-V8  redshift: log R per unit z (proportional to p(z | Euclid)) on a z0 grid; PIT of the true z, 68%/95% HPD coverage.
T-V9  model B: slope of the Euclid-only support percentile and of p_quasar against VIS for test QSOs (should be flat
      apart from the abundance prior; reported, both models).

    python scripts/euclid/validate_extra.py --bundle 20261005_magdep_qlf --workers 6
"""
import argparse
import copy
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from validate_stage1 import EUC, MODELS, score            # noqa: E402

ZGRID = np.round(np.arange(0.2, 4.31, 0.05), 3)


def frac(x, edges=(0, .25, .5, .75, 1.0001)):
    x = x[np.isfinite(x)]; n = len(x); h = np.histogram(x, edges)[0]/max(n, 1)
    return dict(n=n, fractions=h.round(4).tolist(), binomial_99=float(2.58*np.sqrt(.25*.75/max(n, 1))))


def hpd_cover(dens, grid, z, level):
    d = dens/dens.sum(); o = np.argsort(-d); c = np.cumsum(d[o]); keep = o[:np.searchsorted(c, level) + 1]
    return np.min(np.abs(grid[keep] - z)) <= (grid[1] - grid[0])/2


def window_variant(src, dst, factor):
    import shutil, hashlib
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy(f, dst/f.name)
    m = json.loads((src/'model.json').read_text()); w = m['meta']['fallback']['windows']['euclid:vis']
    w['lo'] *= factor; w['hi'] *= factor
    (dst/'model.json').write_text(json.dumps(m))
    man = json.loads((dst/'manifest.json').read_text())
    man['files']['model.json'] = hashlib.sha256((dst/'model.json').read_bytes()).hexdigest()
    (dst/'manifest.json').write_text(json.dumps(man, indent=1))
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', required=True); ap.add_argument('--stage', type=int, choices=(1, 2), default=1); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--n-z', type=int, default=400); ap.add_argument('--n-window', type=int, default=1500)
    a = ap.parse_args(); rng = np.random.default_rng(7)
    ext = MODELS/(f'euclid_stage1_{a.bundle}' if a.stage == 1 else f'euclid_stage2_euclid_point_{a.bundle}'); vdir = MODELS/'work/euclid'/(f'validation_{a.bundle}' if a.stage == 1 else f'validation_stage2_{a.bundle}')
    res = np.load(vdir/'scores.npy', allow_pickle=True).item()
    sys.path.insert(0, str(Path(__file__).parent)); from fit_stage1 import load_rows
    s = json.loads(Path('docs/euclid/settings.json').read_text()); qt = load_rows('qso', s)
    Q = res['qso_test']; idx = Q['idx']
    ef, ev = qt['euclid_flux'][idx], qt['euclid_variance'][idx]; ra, dec = qt['ra'][idx], qt['dec'][idx]
    out = {}
    # T-V2 (QSO side from the saved scores; background p_B needs the score records -> rescored subsample)
    out['T-V2_qso_support_pit'] = frac(Q['euclid']['support'][Q['euclid']['eligible']])
    # T-V8 redshift posterior
    pick = rng.choice(np.flatnonzero(Q['euclid']['eligible']), min(a.n_z, int(Q['euclid']['eligible'].sum())), replace=False)
    rows = np.repeat(pick, len(ZGRID)); zz = np.tile(ZGRID, len(pick))
    sc = score(ext, ef[rows], ev[rows], EUC, ra[rows], dec[rows], zz, a.workers)
    L = sc['logr'].reshape(len(pick), len(ZGRID)); pit = []; c68 = []; c95 = []
    for i, j in enumerate(pick):
        d = np.exp(np.where(np.isfinite(L[i]), L[i] - np.nanmax(L[i]), -np.inf))
        if not np.isfinite(d).any() or d.sum() == 0:
            continue
        zt = Q['z'][j]; cdf = np.cumsum(d)/d.sum(); pit.append(float(np.interp(zt, ZGRID, cdf)))
        c68.append(hpd_cover(d, ZGRID, zt, .68)); c95.append(hpd_cover(d, ZGRID, zt, .95))
    from scipy.stats import kstest
    out['T-V8_redshift'] = dict(n=len(pit), pit=frac(np.array(pit)), ks_p=float(kstest(pit, 'uniform').pvalue),
                                coverage68=float(np.mean(c68)), coverage95=float(np.mean(c95)))
    # T-V4 window sensitivity (+ T-V2 background p_B via the same rescoring)
    B = res['bkg']
    import glob
    from cones import deployment
    bk = []
    for p in sorted(glob.glob(str(Path.home()/'data/qso_p_color/catalogues/euclid_dr1/cones/test_*.npz'))):
        r, _ = deployment(p, s); bk.append({k: v[~r['known_qso']] for k, v in r.items()})
    bk = {k: np.concatenate([o[k] for o in bk]) for k in bk[0]}
    cols = ('vis_to_y_templfit', 'y_templfit', 'j_templfit', 'h_templfit'); ujy = 10**((22.5 - 23.9)/2.5)
    bf = np.column_stack([bk[f'flux_{c}'] for c in cols])*ujy; be = np.column_stack([bk[f'fluxerr_{c}'] for c in cols])*ujy
    ok = np.isfinite(bf) & np.isfinite(be) & (be > 0); bf = np.where(ok, bf, np.nan); bv = np.where(ok, be**2, np.inf)
    vis = 22.5 - 2.5*np.log10(np.clip(bf[:, 0], 1e-9, None)); bright = np.flatnonzero(vis < 23)
    bsel = np.sort(rng.choice(bright, min(a.n_window, len(bright)), replace=False))
    qsel = np.sort(rng.choice(len(idx), min(a.n_window, len(idx)), replace=False))
    zb = rng.choice(Q['z'], len(bsel))
    variants = {f: window_variant(ext, MODELS/'work/euclid'/f'window_{f}_{a.bundle}', f) for f in (0.5, 1.5)}
    base_b = score(ext, bf[bsel], bv[bsel], EUC, bk['e_ra'][bsel], bk['e_dec'][bsel], zb, a.workers)
    base_q = score(ext, ef[qsel], ev[qsel], EUC, ra[qsel], dec[qsel], Q['z'][qsel], a.workers)
    red = (vis[bsel] - (22.5 - 2.5*np.log10(np.clip(bf[bsel, 1], 1e-9, None)))) > 1
    hiz = Q['z'][qsel] > 3
    w = {}
    for f, path in variants.items():
        vb = score(path, bf[bsel], bv[bsel], EUC, bk['e_ra'][bsel], bk['e_dec'][bsel], zb, a.workers)
        vq = score(path, ef[qsel], ev[qsel], EUC, ra[qsel], dec[qsel], Q['z'][qsel], a.workers)
        db = np.abs(vb['p'] - base_b['p']); dq = np.abs(vq['p'] - base_q['p'])
        stat = lambda x: dict(n=int(np.isfinite(x).sum()), median=float(np.nanmedian(x)) if np.isfinite(x).any() else None,
                              p90=float(np.nanpercentile(x, 90)) if np.isfinite(x).any() else None)
        w[str(f)] = dict(background=stat(db), red_background=stat(db[red]), qso=stat(dq), qso_z_gt_3=stat(dq[hiz]),
                         eligibility_changed=float(np.mean(vb['eligible'] != base_b['eligible'])))
    out['T-V4_window'] = w
    out['T-V2_background_bkg_ood_p'] = frac(base_b['bkg_ood_p'][base_b['eligible']])
    out['T-V2_qso_qso_ood_p'] = frac(base_q['qso_ood_p'][base_q['eligible']])
    # T-V9 trend with VIS
    e = Q['euclid']; m = e['eligible'] & np.isfinite(e['support'])
    out['T-V9_trend'] = dict(support_slope_per_mag=float(np.polyfit(Q['vis'][m], e['support'][m], 1)[0]),
                             logit_p_slope_per_mag=float(np.polyfit(Q['vis'][m], np.log(np.clip(e['p'][m], 1e-6, 1-1e-6)/np.clip(1-e['p'][m], 1e-6, 1)), 1)[0]))
    (vdir/'extra.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
