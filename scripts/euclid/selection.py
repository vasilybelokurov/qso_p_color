#!/usr/bin/env python
"""Selection factors of the Euclid deployment selection (docs/euclid/PLAN.md 2.3).

Selected ("euclid_point") = MER match within 1.5" AND clean (vis_det, spurious, bad bits, VIS >= vis_bright)
AND isolated AND point_like_prob >= p_min, all from docs/euclid/settings.json.

f_k(r): for every parent background row of the overlapping cones in COVERED area (its nside-128 probe pixel holds
>= 20 MER sources), old-block responsibilities from Legacy g,r,z,W1,W2 only (parent latent mixture through the
parent observation operator, sky-gate weights); per component a weighted logistic in Legacy r, shrunk to its bin's
logistic (Gaussian priors sd 1 on intercept, 0.5 on slope).
c_Q(z, r): eligible QSO targets in Euclid FIT cells and covered area; logit c = s(z) + b1 (r - 20) + b2 (r - 20)^2
with s a cubic B-spline in z (8 knots); evaluated on the Legacy-r prior grids.

    python scripts/euclid/selection.py --bundle 20261005_magdep_qlf --out <dir>
"""
import argparse
import json
from pathlib import Path
import healpy as hp
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from qso_pcolor.euclid_bundle import clean, point_like
from qso_pcolor.gaussmix import GaussianMixture, log_gauss_batch
from qso_pcolor.multisurvey import MultiSurveyModel

BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1'
RUN = Path('models/multisurvey_psf/work/unified_full/20261001/5f4002492dbb849c')
EDGES = np.array([15.5, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.5])
R0 = 20.


def covered_mask(ra, dec):
    pr = np.load(BASE/'footprint/probes_nside128.npz')
    from astropy.coordinates import SkyCoord; import astropy.units as u
    g = SkyCoord(ra=ra*u.deg, dec=dec*u.deg).galactic
    pix = hp.ang2pix(128, g.l.deg, g.b.deg, lonlat=True, nest=True)
    good = set(pr['pix'][pr['count'] >= 20].tolist())
    return np.array([p in good for p in pix])


def selected(m, nb, s):
    mm = dict(m); mm['matched'] = np.asarray(m['matched'], bool)
    return clean(mm, neighbours=nb, bad_bits=s['bad_bits'], vis_bright=s['vis_bright']) & point_like(mm, s['p_min'])


def logistic_fit(x, y, w, prior=None, sd=(1., .5)):
    """Weighted logistic regression logit = a + b x; Gaussian prior on (a, b) if given."""
    def nll(t):
        z = t[0] + t[1]*x
        v = -np.sum(w*(y*(-np.logaddexp(0, -z)) + (1 - y)*(-np.logaddexp(0, z))))
        if prior is not None:
            v += 0.5*((t[0] - prior[0])/sd[0])**2 + 0.5*((t[1] - prior[1])/sd[1])**2
        return v
    t0 = np.zeros(2) if prior is None else np.asarray(prior, float)
    return minimize(nll, t0, method='L-BFGS-B').x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    s = json.loads(Path('docs/euclid/settings.json').read_text())
    roles = {c['cell']: c['role'] for c in json.loads(Path('docs/euclid/sky_roles.json').read_text())['cells']}
    root = Path('models/multisurvey_psf')/a.bundle
    latent = json.loads((root/'latent.json').read_text()); model = MultiSurveyModel.load(root/'model.json')
    lay = latent['layout']; labels = lay['native_labels']
    leg = [labels.index(f'decals_dr9_{h}:{b}') for h in ('south', 'north') for b in ('g', 'r', 'z', 'w1', 'w2')]
    # ---- background: all parent rows of the overlapping cones
    rows_file = np.load(BASE/'background/rows.npz'); prow = rows_file['rows']
    import glob
    pos = dict(np.load(glob.glob(str(BASE/'background/positional/*/match.npz'))[0]))
    pro_path = Path(glob.glob(str(BASE/'background/propagated/*/match.npz'))[0]); pro = dict(np.load(pro_path))
    npos = np.load(Path(glob.glob(str(BASE/'background/positional/*/neighbours.npz'))[0]))['neighbours']
    npro = np.load(pro_path.with_name('neighbours.npz'))['neighbours']
    has = np.isfinite(rows_file['pmra']) & np.isfinite(rows_file['pmdec']); hi = np.flatnonzero(has)
    for k in pos:
        pos[k] = np.asarray(pos[k]).copy(); pos[k][hi] = pro[k]
    npos = npos.copy(); npos[hi] = npro
    sel = selected(pos, npos, s)
    ra = np.load(RUN/'stars/ra.npy', mmap_mode='r')[prow]; dec = np.load(RUN/'stars/dec.npy', mmap_mode='r')[prow]
    cov = covered_mask(np.asarray(ra), np.asarray(dec))
    y = np.asarray(np.load(RUN/'stars/y.npy', mmap_mode='r')[prow][:, leg]); n = np.asarray(np.load(RUN/'stars/noise.npy', mmap_mode='r')[prow][:, leg])
    l = np.asarray(np.load(RUN/'stars/l.npy', mmap_mode='r')[prow]); b = np.asarray(np.load(RUN/'stars/b.npy', mmap_mode='r')[prow])
    use = np.flatnonzero(cov)
    obs = (n > 0) & np.isfinite(y)
    mix = GaussianMixture.from_dict(latent['background'])
    op = lay['operators']['stars']; h = np.asarray(op['matrix'])[leg]; off = np.asarray(op['offset'])[leg]
    t = np.asarray(op['covariance'])[np.ix_(leg, leg)]
    gate, _ = model.spatial_background.evaluate(l[use], b[use])
    with np.errstate(divide='ignore'):
        lw = np.log(gate)
    resp = np.zeros((len(use), mix.n_components))
    keys, inv = np.unique(obs[use], axis=0, return_inverse=True)
    for g, mask in enumerate(keys):
        rr = np.flatnonzero(inv.ravel() == g); cols = np.flatnonzero(mask)
        if not len(cols):
            continue
        ho = h[cols]; mean = mix.means @ ho.T + off[cols]; cv = ho @ mix.covs @ ho.T + t[np.ix_(cols, cols)]
        for lo in range(0, len(rr), 4000):
            r2 = rr[lo:lo+4000]; u_ = use[r2]
            tot = cv[None] + n[np.ix_(u_, cols)][:, None, :, None]*np.eye(len(cols))
            lp = log_gauss_batch(y[np.ix_(u_, cols)][:, None, :], mean[None], tot) + lw[r2]
            resp[r2] = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))
    rref = np.where(obs[use, 1], y[use, 1], y[use, 6]); ok = np.isfinite(rref)
    rr_ = rref[ok] - R0; ys = sel[use][ok].astype(float); R = resp[ok]
    comp_r = mix.means[:, lay['latent_labels'].index('legacy:r')]
    comp_bin = np.clip(np.searchsorted(EDGES, comp_r) - 1, 0, len(EDGES) - 2)
    alpha = np.zeros(mix.n_components); beta = np.zeros(mix.n_components); report = []
    for j in range(len(EDGES) - 1):
        comps = np.flatnonzero(comp_bin == j); wbin = R[:, comps].sum(1)
        pb = logistic_fit(rr_, ys, wbin)
        for k in comps:
            alpha[k], beta[k] = logistic_fit(rr_, ys, R[:, k], prior=pb)
        report.append(dict(bin=EDGES[j:j+2].tolist(), weight=float(wbin.sum()), bin_fit=pb.tolist(),
                           selected_fraction=float(np.average(ys, weights=wbin)) if wbin.sum() else None))
    thinning = dict(alpha=alpha.tolist(), beta=beta.tolist(), r0=R0,
                    rule='f_k(r) = expit(alpha_k + beta_k (r - r0)); scripts/euclid/selection.py')
    # ---- QSOs: c_Q(z, r)
    t_ = np.load('models/multisurvey_psf/work/full_sample_photometry/e2615aaa39862aff/targets.npz')
    qm = dict(np.load(glob.glob(str(BASE/'qso_targets/*/match.npz'))[0]))
    qn = np.load(glob.glob(str(BASE/'qso_targets/*/neighbours.npz'))[0])['neighbours']
    qsel = selected(qm, qn, s)
    ql = np.asarray(np.load(RUN/'qso/l.npy', mmap_mode='r')); qb = np.asarray(np.load(RUN/'qso/b.npy', mmap_mode='r'))
    qcell = hp.ang2pix(16, ql, qb, lonlat=True, nest=True)
    infit = np.array([roles.get(int(c)) == 'fit' for c in qcell])
    cand = np.flatnonzero(infit)
    qcov = np.zeros(len(qcell), bool); qcov[cand] = covered_mask(t_['ra'][cand], t_['dec'][cand])
    qy = np.load(RUN/'qso/y.npy', mmap_mode='r'); qo = np.load(RUN/'qso/observed.npy', mmap_mode='r')
    si, ni = labels.index('decals_dr9_south:r'), labels.index('decals_dr9_north:r')
    idx = np.flatnonzero(qcov)
    rq = np.where(np.asarray(qo[idx, si]), np.asarray(qy[idx, si]), np.asarray(qy[idx, ni])); zq = t_['zspec'][idx]
    good = np.isfinite(rq) & (rq < 24.5) & (rq > 15)
    idx, rq, zq = idx[good], rq[good], zq[good]; yq = qsel[idx].astype(float)
    knots = np.r_[[0.1]*4, np.linspace(0.5, 4.0, 6), [4.4]*4]; nb_ = len(knots) - 4
    basis = lambda z: BSpline.design_matrix(np.clip(z, 0.1, 4.399), knots, 3).toarray()
    X = np.column_stack([basis(zq), rq - R0, (rq - R0)**2])
    def nll(th):
        zz = X @ th
        return -np.sum(yq*(-np.logaddexp(0, -zz)) + (1 - yq)*(-np.logaddexp(0, zz))) + 0.5*np.sum(th**2)/100.
    th = minimize(nll, np.zeros(X.shape[1]), method='L-BFGS-B').x
    pri = json.loads((root/'priors.json').read_text())
    c_q = {}
    for anchor in ('decals_dr9_south:r', 'decals_dr9_north:r'):
        qp = pri['anchors'][anchor]['qso_prior']; zc = np.asarray(qp['z_centres']); mc = np.asarray(qp['mag_centres'])
        Z, M = np.meshgrid(zc, mc, indexing='ij')
        G = np.column_stack([basis(Z.ravel()), M.ravel() - R0, (M.ravel() - R0)**2])
        c_q[anchor] = expit(G @ th).reshape(Z.shape).tolist()
    zbins = [(0, .5), (.5, .8), (.8, 1.5), (1.5, 2.5), (2.5, 4.4)]
    qrep = {f'{a_}-{b_}': dict(n=int(((zq >= a_) & (zq < b_)).sum()), observed=float(yq[(zq >= a_) & (zq < b_)].mean()),
                               model=float(expit(X[(zq >= a_) & (zq < b_)] @ th).mean())) for a_, b_ in zbins}
    out = dict(thinning=thinning, c_q=c_q, background_bins=report, qso_check=qrep, settings=s,
               n_background_covered=int(len(use)), n_background_selected=int(sel[use].sum()), n_qso_used=int(len(idx)),
               c_q_coef=th.tolist(), c_q_knots=knots.tolist())
    (a.out/'selection.json').write_text(json.dumps(out))
    print(json.dumps(dict(background_bins=report, qso_check=qrep, n_bkg=len(use), n_qso=len(idx)), indent=1))


if __name__ == '__main__':
    main()
