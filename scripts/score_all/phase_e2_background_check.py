#!/usr/bin/env python
"""Phase E step 2: does the existing (stage-1) background model reproduce the Euclid colours of the deployment
selection at bright VIS? If yes, the VIS-referenced background fit (step 3) is needed only at fainter VIS.

Model side, exactly as the scorer evaluates the background for a source of reference magnitude r at (l, b)
(qso_pcolor.multisurvey: spatial component weights w_k(l, b), selection thinning f_k(r) of 'euclid_point',
conditioning on r): per fit cone,
    r ~ Sigma_B(r)                        (bundle priors.json background_density, global_density; spatial counts
                                           not used - ASSUMED adequate for the colour shape)
    k ~ w_k(l, b) f_k(r) N(r; mu_rk, C_rrk)
    (VIS, Y, J, H) ~ N(. | r, k)          (noise-free dereddened luptitudes, Euclid softenings)
plus noise: each draw takes the luptitude variances of a random selected row of the same cone within 0.25 mag in VIS.
Data side: the deployment selection of the 106 fit cones (docs/score_all/new_cones.json; never used for any fit),
Euclid luptitudes dereddened with MER gal_ebv (qso_pcolor.euclid_bundle.dereddened, EXTINCTION_R).

Per VIS bin of 0.5 mag (18.5-24) and colour (VIS-Y, VIS-J, VIS-H): observed and predicted 16/50/84 percentiles, the
two-sample KS distance, and the fraction of observed rows outside the predicted central 95% interval.
Acceptance (pre-stated) for "reuse at this VIS": |d median| <= 0.05 mag, |d half-width (84-16)/2| <= 20%, and the
outside-95% fraction within 3-8%, for all three colours.

Output: docs/score_all/phase_e2_background_check.json and plots/score_all/phase_e2_background_check.png.
Light: one process, cones handled one at a time.

    nice -n 19 python scripts/score_all/phase_e2_background_check.py [--limit N]
"""
import glob
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/euclid'))
BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1/cones'
OUT = ROOT/'docs/score_all/phase_e2_background_check.json'
POINTER = 'current_euclid'
R_BAND = 'decals_dr9_south:r'
EUC = ('euclid:vis', 'euclid:y', 'euclid:j', 'euclid:h')
V_EDGES = np.arange(18.5, 24.01, 0.5)
DRAWS_PER_ROW = 5
SEED = 20261119


def observed_luptitudes(rows, settings):
    from qso_pcolor.euclid_bundle import dereddened, luptitudes, photometry
    p = dereddened(photometry(dict(rows, matched=np.ones(len(rows['e_ra']), bool))), np.asarray(rows['gal_ebv'], float))
    soft = np.asarray(settings['softening'], float)
    y = np.empty_like(p.flux); n = np.empty_like(p.flux)
    for j in range(4):
        y[:, j], n[:, j] = luptitudes(p.flux[:, j], p.variance[:, j], soft[j])
    return y, n


def model_parts():
    from qso_pcolor.unified import UnifiedPSFModel
    root = ROOT/'models/multisurvey_psf'
    mo = UnifiedPSFModel.load(root/POINTER).base.model
    bundle = root/json.loads((root/POINTER).read_text())['bundle']
    bd = json.loads((bundle/'priors.json').read_text())['anchors'][R_BAND]['background_density']
    b = mo.background; ir = b.labels.index(R_BAND); ie = [b.labels.index(x) for x in EUC]
    return mo, b, ir, ie, np.asarray(bd['mag_edges']), np.asarray(bd['global_density'])


def draw(mo, b, ir, ie, medges, gdens, l, bb, n, rng):
    """n background draws (noise-free VIS, Y, J, H luptitudes) at Galactic (l, b)."""
    from qso_pcolor.multisurvey import thinning_log_factor
    w = mo.spatial_background.evaluate([l], [bb])[0][0]
    p = gdens/gdens.sum(); i = rng.choice(len(p), n, p=p)
    r = medges[i] + rng.random(n)*np.diff(medges)[i]
    mu_r, c_rr = b.means[:, ir], b.covs[:, ir, ir]
    lt = thinning_log_factor(mo.meta['selection']['thinning'], r)                     # [n, K]
    lp = np.log(np.maximum(w, 1e-300))[None] + lt - 0.5*(r[:, None] - mu_r[None])**2/c_rr[None] - 0.5*np.log(c_rr)[None]
    pk = np.exp(lp - lp.max(1, keepdims=True)); pk /= pk.sum(1, keepdims=True)
    k = (pk.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    out = np.empty((n, 4))
    for kk in np.unique(k):
        s = np.flatnonzero(k == kk)
        mu = b.means[kk, ie]; c_er = b.covs[kk, ie, ir]; c_ee = b.covs[kk][np.ix_(ie, ie)]
        m = mu[None] + np.outer(r[s] - mu_r[kk], c_er/c_rr[kk])
        C = c_ee - np.outer(c_er, c_er)/c_rr[kk]
        out[s] = m + rng.multivariate_normal(np.zeros(4), C, size=len(s), method='eigh')
    return out


def main():
    import argparse
    from cones import deployment
    from qso_pcolor.unified import galactic_from_equatorial
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--limit', type=int, default=None, help='first N fit cones only (smoke test)')
    a = ap.parse_args()
    settings = json.loads((ROOT/'docs/euclid/settings.json').read_text())
    mo, b, ir, ie, medges, gdens = model_parts()
    rng = np.random.default_rng(SEED)
    paths = sorted(glob.glob(str(BASE/'fit_*.npz')))[:a.limit]
    obs, pred = [], []
    for path in paths:
        rows, _ = deployment(path, settings)
        y, n = observed_luptitudes(rows, settings)
        ok = np.isfinite(y).all(1) & (y[:, 0] >= V_EDGES[0]) & (y[:, 0] < V_EDGES[-1])
        y, n = y[ok], n[ok]
        if len(y) == 0:
            continue
        c = np.load(path); l, bb = galactic_from_equatorial(np.array([c['centre'][0]]), np.array([c['centre'][1]]))
        # draw enough to cover the cone's VIS range at the bright end, then keep those inside it
        x = draw(mo, b, ir, ie, medges, gdens, float(l[0]), float(bb[0]), DRAWS_PER_ROW*len(y)*4, rng)
        x = x[(x[:, 0] >= V_EDGES[0]) & (x[:, 0] < V_EDGES[-1])]
        order = np.argsort(y[:, 0]); vs = y[order, 0]
        j = np.searchsorted(vs, x[:, 0]); j = np.clip(j + rng.integers(-3, 4, len(j)), 0, len(vs) - 1)
        near = np.abs(vs[j] - x[:, 0]) < 0.25
        x, nn = x[near], n[order][j[near]]
        pred.append(x + rng.standard_normal(x.shape)*np.sqrt(nn)); obs.append(y)
        print(Path(path).name, len(y), len(x), flush=True)
    obs, pred = np.concatenate(obs), np.concatenate(pred)
    from scipy.stats import ks_2samp
    rep = dict(pointer=POINTER, n_cones=len(paths), n_obs=int(len(obs)), n_pred=int(len(pred)),
               acceptance='|d median| <= 0.05, |d half-width| <= 20%, outside-95% in [3%, 8%], all colours', bins=[])
    for lo, hi in zip(V_EDGES[:-1], V_EDGES[1:]):
        so = (obs[:, 0] >= lo) & (obs[:, 0] < hi); sp = (pred[:, 0] >= lo) & (pred[:, 0] < hi)
        if so.sum() < 50 or sp.sum() < 50:
            continue
        row = dict(vis=f'{lo:.1f}-{hi:.1f}', n_obs=int(so.sum()), n_pred=int(sp.sum()), colours={}); good = True
        for jc, name in ((1, 'VIS-Y'), (2, 'VIS-J'), (3, 'VIS-H')):
            co = obs[so, 0] - obs[so, jc]; cp = pred[sp, 0] - pred[sp, jc]
            qo, qp = np.percentile(co, [16, 50, 84]), np.percentile(cp, [16, 50, 84])
            lo95, hi95 = np.percentile(cp, [2.5, 97.5]); out95 = float(((co < lo95) | (co > hi95)).mean())
            dmed = float(qo[1] - qp[1]); dw = float((qo[2] - qo[0])/(qp[2] - qp[0]) - 1)
            ok = abs(dmed) <= 0.05 and abs(dw) <= 0.2 and 0.03 <= out95 <= 0.08; good &= ok
            row['colours'][name] = dict(obs_q16_50_84=np.round(qo, 3).tolist(), pred_q16_50_84=np.round(qp, 3).tolist(),
                                        d_median=round(dmed, 3), d_halfwidth_frac=round(dw, 3), outside95=round(out95, 3),
                                        ks=round(float(ks_2samp(co, cp).statistic), 3), pass_=bool(ok))
        row['pass'] = bool(good); rep['bins'].append(row)
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"{'VIS':9s} {'n_obs':>7s}  " + '  '.join(f'{c:>28s}' for c in ('VIS-Y dmed dw out95', 'VIS-J', 'VIS-H')) + '  pass')
    for r in rep['bins']:
        print(f"{r['vis']:9s} {r['n_obs']:7d}  " + '  '.join(
            f"{v['d_median']:+8.3f} {v['d_halfwidth_frac']:+7.2f} {v['outside95']:6.3f} ks{v['ks']:5.2f}" for v in r['colours'].values())
              + f"  {r['pass']}")
    plot(obs, pred)


def plot(obs, pred):
    sys.path.insert(0, str(ROOT/'scripts/method_unified'))
    import matplotlib.pyplot as plt
    from common import save_figure, use_paper_style
    use_paper_style()
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for jc, name, a in ((1, 'VIS$-$Y', ax[0]), (2, 'VIS$-$J', ax[1]), (3, 'VIS$-$H', ax[2])):
        for y, c, lab in ((obs, 'k', 'observed (fit cones)'), (pred, '#b5342c', 'stage-1 background model')):
            cen = 0.5*(V_EDGES[1:] + V_EDGES[:-1]); q = np.full((len(cen), 3), np.nan)
            for i, (lo, hi) in enumerate(zip(V_EDGES[:-1], V_EDGES[1:])):
                s = (y[:, 0] >= lo) & (y[:, 0] < hi)
                if s.sum() >= 50:
                    q[i] = np.percentile(y[s, 0] - y[s, jc], [16, 50, 84])
            a.plot(q[:, 1], cen, '-', color=c, label=lab); a.fill_betweenx(cen, q[:, 0], q[:, 2], color=c, alpha=.15)
        a.set(xlabel=f'{name} [mag]'); a.invert_yaxis()
    ax[0].set_ylabel('VIS [mag]'); ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); print(save_figure(fig, 'score_all/phase_e2_background_check'))


if __name__ == '__main__':
    main()
