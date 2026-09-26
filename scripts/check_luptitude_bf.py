#!/usr/bin/env python
"""Does the luptitude delta method bias the Bayes factor?

For reserved objects with Legacy south g, r, z, W1, W2, compute ln BF (quasar at
z0 against the dedicated Legacy field fit, both conditioned on u_r) twice with
the same Monte Carlo code:

* noise likelihood Gaussian in luptitude, sigma_u = |g'(f_obs)| sigma_f  (what
  the scorer assumes);
* noise likelihood exact: Gaussian in FLUX, p(u_obs | u) proportional to
  N(g^-1(u_obs); g^-1(u), sigma_f)   (the truth for Gaussian flux errors).

Each component's integral over the true luptitudes is done by importance
sampling from the Gaussian product of the component and the delta-method
likelihood, inflated; the Jacobian of u_obs is common to both hypotheses and
cancels. The difference of the two ln BF is the error the approximation makes.

    python scripts/check_luptitude_bf.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

MAG = 2.5 / np.log(10.0)
LEGACY = tuple(f"decals_dr9_south:{b}" for b in ("g", "r", "z", "w1", "w2"))


def ginv(u, s):
    """Flux from luptitude, u in native magnitudes."""
    return 2 * s * np.sinh(-(u - 22.5) / MAG - np.log(s))


def log_like_comp(mu, V, uobs, fobs, sig, s, idx, rng, n=4000, exact=True):
    """log of integral N(u; mu, V) p(uobs | u) du over the bands idx."""
    su2 = (MAG / np.hypot(fobs, 2 * s)) ** 2 * sig ** 2        # delta-method variance
    m, Vk = mu[idx], V[np.ix_(idx, idx)]
    S = np.diag(su2[idx])
    # Gaussian product as a proposal, inflated by 2 in variance
    P = np.linalg.inv(np.linalg.inv(Vk) + np.linalg.inv(S))
    c = P @ (np.linalg.solve(Vk, m) + np.linalg.solve(S, uobs[idx]))
    L = np.linalg.cholesky(2.0 * P)
    z = rng.standard_normal((n, idx.size))
    u = c + z @ L.T
    lq = (-0.5 * np.sum(z ** 2, 1) - np.log(np.diag(L)).sum() - 0.5 * idx.size * np.log(2 * np.pi))
    d = u - m
    Lv = np.linalg.cholesky(Vk)
    y = np.linalg.solve(Lv, d.T)
    lp = -0.5 * np.sum(y ** 2, 0) - np.log(np.diag(Lv)).sum() - 0.5 * idx.size * np.log(2 * np.pi)
    # Terms constant in u (normalisations, the Jacobian at u_obs) are the same for
    # both hypotheses and cancel in ln BF, so they are dropped from both versions.
    if exact:
        ft = ginv(u, s[idx])
        ln = np.sum(-0.5 * ((fobs[idx] - ft) / sig[idx]) ** 2, 1)
    else:
        ln = np.sum(-0.5 * (uobs[idx] - u) ** 2 / su2[idx], 1)
    return logsumexp(lp + ln - lq) - np.log(n)


def log_cond(mixes, weights, uobs, fobs, sig, s, full, anchor, rng, exact):
    lj, lm = [], []
    for mix, w in zip(mixes, weights):
        for k in range(mix.n_components):
            lw = np.log(w) + np.log(mix.weights[k])
            lj.append(lw + log_like_comp(mix.means[k], mix.covs[k], uobs, fobs, sig, s, full, rng,
                                         exact=exact))
            lm.append(lw + log_like_comp(mix.means[k], mix.covs[k], uobs, fobs, sig, s, anchor, rng,
                                         exact=exact))
    return logsumexp(lj) - logsumexp(lm)


def main() -> None:
    from qso_pcolor.multisurvey import MultiSurveyModel
    from qso_pcolor.multisurvey_data import Photometry

    model = MultiSurveyModel.load("models/multisurvey.json")
    B = model.transform.bands
    ib = np.array([B.index(b) for b in LEGACY])
    soft = model.transform.softening[ib]
    field = [m for m in model.background_marginals if tuple(m.labels) == LEGACY][0]
    root = Path(json.loads(Path("configs/multisurvey_lsw.json").read_text())["data_dir"])
    rng = np.random.default_rng(0)
    zc = model.qso.z_centres
    res = []
    for name, n_obj in (("quasars", 150), ("background", 150)):
        r = dict(np.load(root / f"{name}.npz"))
        p = Photometry(r["flux"], r["variance"], tuple(str(x) for x in r["bands"])).align(B)
        ok = r["held"].astype(bool) & p.observed[:, ib].all(1)
        if name == "quasars":
            ok &= (r["zspec"] > 0.2) & (r["zspec"] < 4.3)
        idx = rng.choice(np.flatnonzero(ok), n_obj, replace=False)
        for i in idx:
            f = p.flux[i, ib]; sig = np.sqrt(p.variance[i, ib])
            u = 22.5 - MAG * (np.arcsinh(f / (2 * soft)) + np.log(soft))
            z0 = float(r["zspec"][i]) if name == "quasars" else float(rng.uniform(0.3, 4.0))
            j = int(np.clip(np.searchsorted(zc, z0) - 1, 0, zc.size - 2)); w = (z0 - zc[j]) / (zc[j + 1] - zc[j])
            qm = [model.qso.mixtures[j].marginal(ib), model.qso.mixtures[j + 1].marginal(ib)]
            full, anchor = np.arange(5), np.array([1])
            out = {}
            for exact in (False, True):
                lq = log_cond(qm, [1 - w, w], u, f, sig, soft, full, anchor, rng, exact)
                lb = log_cond([field], [1.0], u, f, sig, soft, full, anchor, rng, exact)
                out[exact] = lq - lb
            snr = np.min(np.abs(f) / sig)
            res.append(dict(kind=name, z0=z0, bf_delta=out[False], bf_exact=out[True],
                            d=out[True] - out[False], min_snr=float(snr)))
    d = np.array([x["d"] for x in res]); snr = np.array([x["min_snr"] for x in res])
    bfd = np.array([x["bf_delta"] for x in res]); kind = np.array([x["kind"] for x in res])
    print(f"objects: {len(res)}  ln BF_exact - ln BF_delta: median {np.median(d):+.3f}, "
          f"median |.| {np.median(np.abs(d)):.3f}, 90th {np.percentile(np.abs(d), 90):.3f}, "
          f"max {np.abs(d).max():.3f}")
    for lo, hi in ((0, 3), (3, 10), (10, np.inf)):
        m = (snr >= lo) & (snr < hi)
        if m.any():
            print(f"  faintest band S/N {lo:g}-{hi:g}: n={m.sum():3d} median |d| {np.median(np.abs(d[m])):.3f} "
                  f"90th {np.percentile(np.abs(d[m]), 90):.3f}")
    # does it change the sign of the evidence?
    flips = np.sum(np.sign(bfd) != np.sign(bfd + d))
    print(f"  sign changes of ln BF: {flips} of {len(res)}")
    # MC noise floor: repeat the delta computation with a different seed for a few objects
    Path("docs/HEALTH_luptitude_bf.json").write_text(json.dumps(
        {"n": len(res), "median_d": float(np.median(d)), "median_abs_d": float(np.median(np.abs(d))),
         "p90_abs_d": float(np.percentile(np.abs(d), 90)), "max_abs_d": float(np.abs(d).max()),
         "sign_flips": int(flips), "rows": res}, indent=1))


if __name__ == "__main__":
    main()
