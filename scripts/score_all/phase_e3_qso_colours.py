#!/usr/bin/env python
"""Phase E step 3b input: the quasar colour density p_Q(VIS-Y, VIS-J, VIS-H | VIS bin), held fixed in the background fit.

For each 0.25-mag VIS bin of step 3b (21.0-26.5) and each QSO model (B = current_euclid, A = current_euclid_magindep):
draw (z slice j, Legacy r) with weight Sigma_Q(z_j, r) c_Q(z_j, r) dz_j dr P(VIS in bin | r, z_j) (step 1: parent prior,
flat c_Q scenario, phase_e1_qso_vis_prior.py), then a slice component given r, then noise-free (VIS, Y, J, H) from the
component's Gaussian conditional on r, keeping draws inside the bin; fit a Gaussian mixture (K_Q components,
scikit-learn, noise-free) to the colours. Also stored: Sigma_Q(VIS bin) per deg^2 (flat), the same as step 1.
Model A beyond its training range is an extrapolation (flag `qso_colour_extrapolated` in deployment).

Output: ~/data/qso_p_color/score_all/phase_e/qso_colour_{A,B}.npz. One process, < 1 min.

    nice -n 19 python scripts/score_all/phase_e3_qso_colours.py
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/score_all'))
import phase_e1_qso_vis_prior as E1  # noqa: E402

BINS = np.round(np.arange(21.0, 26.5001, 0.25), 2)
N_DRAW = 40_000
K_Q = 4
SEED = 20261119
EUC = ('euclid:vis', 'euclid:y', 'euclid:j', 'euclid:h')


def slice_inputs(name):
    from qso_pcolor.unified import UnifiedPSFModel
    root = ROOT/'models/multisurvey_psf'
    model = UnifiedPSFModel.load(root/name).base.model
    bundle = root/json.loads((root/name).read_text())['bundle']
    parent = ROOT/model.meta['euclid']['parent_bundle']
    pe = json.loads((bundle/'priors.json').read_text())['anchors'][E1.R_BAND]['qso_prior']
    pp = json.loads((parent/'priors.json').read_text())['anchors'][E1.R_BAND]['qso_prior']
    se, sp = np.asarray(pe['sigma']), np.asarray(pp['sigma'])
    with np.errstate(invalid='ignore', divide='ignore'):
        cq_tab = np.where(sp > 0, se/sp, np.nan)
    cq_tab = np.where(np.isfinite(cq_tab), cq_tab, np.nanmedian(cq_tab, axis=1, keepdims=True))
    zc, sig, _ = E1.sigma_r(pp)
    cq = E1.c_q_grid(cq_tab, np.asarray(pp['mag_centres']), 'flat')
    dz = np.diff(np.asarray(pp['z_edges']))[:, None]                   # the prior is per unit z: weight slices by dz
    return model.qso, sig*cq*dz*np.gradient(E1.R_GRID)[None]


def colour_density(q, w_r, lo, hi, rng):
    """Draws of noise-free (VIS-Y, VIS-J, VIS-H) for VIS in [lo, hi); Sigma_Q in the bin (per deg^2)."""
    from scipy.stats import norm
    ir = q.labels.index(E1.R_BAND); ie = [q.labels.index(b) for b in EUC]
    # P(VIS in bin | r, z) per slice from the step-1 conditional (V_EDGES contains the bin edges)
    iv0 = int(np.argmin(np.abs(E1.V_EDGES - lo))); iv1 = int(np.argmin(np.abs(E1.V_EDGES - hi)))
    P = np.stack([E1.p_v_given_r(q.mixtures[j], ir, ie[0])[:, iv0:iv1].sum(1) for j in range(len(q.mixtures))])
    lam = w_r*P; total = float(lam.sum())
    out = []
    while sum(len(o) for o in out) < N_DRAW:
        flat = rng.choice(lam.size, N_DRAW, p=(lam/total).ravel())
        js, irr = np.unravel_index(flat, lam.shape)
        r = E1.R_GRID[irr] + (rng.random(N_DRAW) - 0.5)*0.02
        for j in np.unique(js):
            s = np.flatnonzero(js == j); mix = q.mixtures[j]
            mu_r, crr = mix.means[:, ir], mix.covs[:, ir, ir]
            lp = np.log(np.maximum(mix.weights, 1e-300))[None] + norm.logpdf(r[s, None], mu_r[None], np.sqrt(crr)[None])
            pk = np.exp(lp - lp.max(1, keepdims=True)); pk /= pk.sum(1, keepdims=True)
            k = (pk.cumsum(1) > rng.random(len(s))[:, None]).argmax(1)
            for kk in np.unique(k):
                t = s[k == kk]
                mu = mix.means[kk, ie]; cer = mix.covs[kk, ie, ir]; cee = mix.covs[kk][np.ix_(ie, ie)]
                m = mu[None] + np.outer(r[t] - mu_r[kk], cer/crr[kk]); C = cee - np.outer(cer, cer)/crr[kk]
                x = m + rng.multivariate_normal(np.zeros(4), C, size=len(t), method='eigh')
                x = x[(x[:, 0] >= lo) & (x[:, 0] < hi)]
                out.append(np.column_stack([x[:, 0] - x[:, 1], x[:, 0] - x[:, 2], x[:, 0] - x[:, 3]]))
    return np.concatenate(out)[:N_DRAW], total


def main():
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(SEED)
    for tag, name in E1.MODELS.items():
        q, w_r = slice_inputs(name)
        W, M, C, S, med = [], [], [], [], []
        for lo in BINS[:-1]:
            c, tot = colour_density(q, w_r, lo, lo + 0.25, rng)
            g = GaussianMixture(K_Q, covariance_type='full', random_state=SEED, reg_covar=1e-5).fit(c)
            W.append(g.weights_); M.append(g.means_); C.append(g.covariances_); S.append(tot); med.append(np.median(c, 0))
            print(tag, f'{lo:.2f}', f'Sigma_Q {tot:8.1f}/deg2', 'median colours', np.round(np.median(c, 0), 3), flush=True)
        np.savez(E1.OUT/f'qso_colour_{tag}.npz', bins=BINS, weights=np.array(W), means=np.array(M), covs=np.array(C),
                 sigma_q_bin=np.array(S), median=np.array(med), model=name, k=K_Q, n_draw=N_DRAW)


if __name__ == '__main__':
    main()
