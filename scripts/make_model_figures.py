#!/usr/bin/env python
"""The method note's model figures, drawn from the shipped model.

Every panel uses ``models/multisurvey.json`` (with its priors and unmodelled
term where a score is shown) and the model's own cached samples: the
quasar draw (``quasars.npz``) and the field cones (``background.npz``). No
fitting except the deconvolution test, which refits a 1-D colour on purpose.

Colour planes are Legacy DR9 south luptitude colours, g-r and r-z, the model's
native coordinates for those bands. A density drawn in a plane is the model's
joint density marginalised over every other band and conditioned on a fixed
reference luptitude u_r, noiseless -- the intrinsic density the scorer
convolves with each object's errors.

Out-of-sample panels (window, factorisation, separation) use only quasars in
the reserved sky blocks and sources in the reserved field cones.

    python scripts/make_model_figures.py
    python scripts/make_model_figures.py --only locus slices

Writes plots/method/model_*.png and data/model_figures.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

G, R, Z, W1, W2 = (f"decals_dr9_south:{b}" for b in ("g", "r", "z", "w1", "w2"))
LEGACY = (G, R, Z, W1, W2)
Z_LO, Z_HI = 0.15, 4.35


# -- data ------------------------------------------------------------------------

def load(cfg_path: Path):
    from qso_pcolor.multisurvey import MultiSurveyModel, MultiSurveyOutlier, load_priors
    from qso_pcolor.multisurvey_data import Photometry

    cfg = json.loads(cfg_path.read_text())
    root = Path(cfg["data_dir"])
    model = MultiSurveyModel.load("models/multisurvey.json")
    priors = load_priors("models/multisurvey_priors.json", model)
    out = MultiSurveyOutlier.load("models/multisurvey_outlier.json")
    d = {}
    for key, name in (("q", "quasars"), ("b", "background")):
        r = dict(np.load(root / f"{name}.npz"))
        p = Photometry(r["flux"], r["variance"], tuple(str(x) for x in r["bands"])).align(
            model.transform.bands)
        f = model.transform(p)
        d[key] = dict(raw=r, phot=p, feat=f)
    for key in ("q", "b"):
        f = d[key]["feat"]
        idx = {b: model.transform.bands.index(b) for b in LEGACY}
        d[key]["idx"] = idx
        u = {b: f.x[:, i] for b, i in idx.items()}
        d[key]["u"] = u
        d[key]["grz"] = f.observed[:, idx[G]] & f.observed[:, idx[R]] & f.observed[:, idx[Z]]
        d[key]["grzw"] = d[key]["grz"] & f.observed[:, idx[W1]] & f.observed[:, idx[W2]]
    d["q"]["z"] = d["q"]["raw"]["zspec"]
    d["q"]["held"] = d["q"]["raw"]["held"].astype(bool)
    d["b"]["held"] = d["b"]["raw"]["held"].astype(bool)
    return model, priors, out, d


# -- densities in a Legacy colour plane --------------------------------------------

def plane_mixture(mixtures, weights, bands, r0):
    """Components of p(g - r, r - z | u_r = r0) from joint mixtures.

    ``mixtures`` with relative ``weights`` form one density (e.g. two redshift
    slices interpolated in density). Returns log-weights, means (K,2), covs (K,2,2).
    """
    lw, mu, cv = [], [], []
    for mix, wmix in zip(mixtures, weights):
        if wmix <= 0:
            continue
        ig, ir, iz = (mix.labels.index(b) for b in (G, R, Z)) if mix.labels else [bands.index(b) for b in (G, R, Z)]
        m = mix.means[:, [ig, ir, iz]]
        v = mix.covs[:, [ig, ir, iz]][:, :, [ig, ir, iz]]
        vrr = v[:, 1, 1]
        lw.append(np.log(wmix) + np.log(mix.weights) + norm.logpdf(r0, m[:, 1], np.sqrt(vrr)))
        k = np.array([v[:, 0, 1], v[:, 2, 1]]).T / vrr[:, None]           # regression on r
        mg = m[:, 0] + k[:, 0] * (r0 - m[:, 1])
        mz = m[:, 2] + k[:, 1] * (r0 - m[:, 1])
        vgg = v[:, 0, 0] - k[:, 0] * v[:, 0, 1]
        vzz = v[:, 2, 2] - k[:, 1] * v[:, 2, 1]
        vgz = v[:, 0, 2] - k[:, 0] * v[:, 2, 1]
        mu.append(np.column_stack([mg - r0, r0 - mz]))                     # (g-r, r-z)
        cv.append(np.stack([np.stack([vgg, -vgz], -1), np.stack([-vgz, vzz], -1)], -2))
    lw = np.concatenate(lw)
    return lw - logsumexp(lw), np.concatenate(mu), np.concatenate(cv)


def plane_logpdf(comp, xx, yy):
    lw, mu, cv = comp
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    out = np.full((len(lw), len(pts)), -np.inf)
    for k in range(len(lw)):
        L = np.linalg.cholesky(cv[k])
        y = np.linalg.solve(L, (pts - mu[k]).T)
        out[k] = lw[k] - 0.5 * np.sum(y * y, 0) - np.log(np.diag(L)).sum() - np.log(2 * np.pi)
    return logsumexp(out, axis=0).reshape(xx.shape)


def qso_plane(model, z, r0):
    zc = model.qso.z_centres
    j = int(np.clip(np.searchsorted(zc, z) - 1, 0, zc.size - 2))
    w = (z - zc[j]) / (zc[j + 1] - zc[j])
    return plane_mixture([model.qso.mixtures[j], model.qso.mixtures[j + 1]], [1 - w, w],
                         model.transform.bands, r0)


def field_plane(model, r0):
    grz = [m for m in model.background_marginals if tuple(m.labels) == (G, R, Z)]
    return plane_mixture(grz, [1.0], model.transform.bands, r0)


def colours(dd):
    u = dd["u"]
    return u[G] - u[R], u[R] - u[Z]


# -- figures -----------------------------------------------------------------------

def fig_locus(model, d, out):
    import matplotlib.pyplot as plt
    from qso_pcolor.plotting import SEQUENTIAL_CMAP, SERIES, save_figure

    q, b = d["q"], d["b"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax = axes[0]
    gr_b, rz_b = colours(b)
    ext = (-0.8, 2.8, -0.7, 2.3)
    sel = b["grz"] & (b["u"][R] < 22.5)
    ax.hexbin(rz_b[sel], gr_b[sel], gridsize=90, extent=ext, bins="log", cmap="Greys",
              mincnt=1, linewidths=0)
    gr_q, rz_q = colours(q)
    s = q["grz"] & (q["u"][R] < 22.5)
    sc = ax.scatter(rz_q[s], gr_q[s], c=q["z"][s], s=1.2, cmap=SEQUENTIAL_CMAP, vmin=Z_LO,
                    vmax=Z_HI, alpha=0.7, linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("spectroscopic redshift"); cb.outline.set_visible(False)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("Legacy $r - z$ (luptitudes)"); ax.set_ylabel("Legacy $g - r$")
    ax.set_title("Quasars over the field population", loc="left")
    ax.text(0.04, 0.95, "grey: every source in the field cones", transform=ax.transAxes,
            fontsize=7.5, color=SERIES["neutral"], va="top")

    ax = axes[1]
    zc = np.linspace(Z_LO, Z_HI, 43)
    tracks = (("$g - r$", gr_q, q["grz"], SERIES["same_z"]),
              ("$r - z$", rz_q, q["grz"], SERIES["field_q"]),
              ("$z - W1$", q["u"][Z] - q["u"][W1], q["grz"] & q["feat"].observed[:, q["idx"][W1]],
               SERIES["background"]))
    for lab, y, ok, col in tracks:
        stats = np.array([np.nanpercentile(y[ok & (np.abs(q["z"] - zz) < 0.05)], [16, 50, 84])
                          for zz in zc])
        ax.fill_between(zc, stats[:, 0], stats[:, 2], color=col, alpha=0.18, lw=0)
        ax.plot(zc, stats[:, 1], color=col)
        ax.annotate(lab, (zc[-1], stats[-1, 1]), xytext=(4, 0), textcoords="offset points",
                    color=col, fontsize=8, va="center")
    ax.set_xlabel("spectroscopic redshift"); ax.set_ylabel("colour (luptitudes)")
    ax.set_xlim(Z_LO, Z_HI + 0.4)
    ax.set_title("Median colour vs redshift, 16–84th percentile", loc="left")
    fig.tight_layout()
    out["locus"] = {"n_quasars": int(s.sum()), "n_field": int(sel.sum())}
    return save_figure(fig, "method/model_locus")


def fig_slices(model, d, out, r0=20.5):
    import matplotlib.pyplot as plt
    from qso_pcolor.plotting import SERIES, save_figure

    q = d["q"]
    gr, rz = colours(q)
    gx = np.linspace(-0.6, 2.0, 200); gy = np.linspace(-0.6, 2.2, 200)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharex=True, sharey=True)
    for ax, z0 in zip(axes, (0.9, 1.8, 3.0)):
        lp = plane_logpdf(qso_plane(model, z0, r0), xx, yy)
        p = np.exp(lp - lp.max())
        s = q["grz"] & (np.abs(q["z"] - z0) < 0.1) & (np.abs(q["u"][R] - r0) < 0.5)
        ax.scatter(gr[s], rz[s], s=2.0, color=SERIES["neutral"], alpha=0.4, linewidths=0,
                   rasterized=True)
        ax.contour(gx, gy, p.T, levels=[0.05, 0.3, 0.8], colors=SERIES["same_z"],
                   linewidths=1.0)
        ax.set_title(f"$z = {z0}$", loc="left")
        ax.set_xlabel("Legacy $g - r$")
    axes[0].set_ylabel("Legacy $r - z$")
    axes[0].text(0.04, 0.04, f"points: quasars in the slice,\n$|u_r - {r0}| < 0.5$\n"
                 "lines: model at $u_r = $" + f"{r0}\n(deconvolved, so narrower)",
                 transform=axes[0].transAxes, fontsize=7, color=SERIES["neutral"], va="bottom")
    fig.tight_layout()
    return save_figure(fig, "method/model_slices")


def plane_outlier(model, outl, r0, xx, yy):
    """log of the unmodelled density on the plane and its share eta at u_r = r0."""
    B = model.transform.bands
    ia = B.index(R)
    n = xx.size
    x = np.full((n, len(B)), np.nan)
    obs = np.zeros((n, len(B)), bool)
    x[:, B.index(G)] = r0 + xx.ravel(); x[:, ia] = r0; x[:, B.index(Z)] = r0 - yy.ravel()
    obs[:, [B.index(G), ia, B.index(Z)]] = True
    c = outl.conditional(ia, R, model.qso.system)
    return (c.log_prob(x, np.zeros((n, len(B), len(B))), observed=obs).reshape(xx.shape),
            float(c.fraction_at(np.array([r0]))[0]))


def fig_two_densities(model, outl, d, out, z0=1.8, r0=20.8):
    import matplotlib.pyplot as plt
    from qso_pcolor.plotting import SERIES, save_figure

    b = d["b"]
    gx = np.linspace(-0.6, 2.2, 220); gy = np.linspace(-0.6, 2.4, 220)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    lq = plane_logpdf(qso_plane(model, z0, r0), xx, yy)
    lb = plane_logpdf(field_plane(model, r0), xx, yy)
    lu, eta = plane_outlier(model, outl, r0, xx, yy)
    lf = np.logaddexp(np.log1p(-eta) + lb, np.log(eta) + lu)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.18]})
    ax = axes[0]
    for lp, key, lab in ((lq, "same_z", f"quasar at $z={z0}$"), (lb, "background", "field")):
        ax.contour(gx, gy, np.exp(lp - lp.max()).T, levels=[0.05, 0.3, 0.8],
                   colors=SERIES[key], linewidths=1.1)
        ax.plot([], [], color=SERIES[key], label=lab)
    ax.legend(loc="upper right", fontsize=6.5)
    ax.set_xlabel("Legacy $g - r$"); ax.set_ylabel("Legacy $r - z$")
    ax.set_title(f"Densities, $u_r={r0}$", loc="left", fontsize=9)
    gr, rz = colours(b)
    s = np.flatnonzero(b["grz"] & (np.abs(b["u"][R] - r0) < 0.4))
    lim = 12
    for ax, bf, title in ((axes[1], lq - lb, "ln BF, field model only"),
                          (axes[2], lq - lf, "ln BF as scored (with $U$)")):
        im = ax.pcolormesh(gx, gy, np.clip(bf, -lim, lim).T, cmap="RdBu_r", vmin=-lim, vmax=lim,
                           shading="auto", rasterized=True)
        ax.contour(gx, gy, bf.T, levels=[0.0], colors="#2b2b28", linewidths=0.8)
        ax.scatter(gr[s], rz[s], s=0.8, color="k", alpha=0.25, linewidths=0, rasterized=True)
        ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gy[0], gy[-1])
        ax.set_xlabel("Legacy $g - r$")
        ax.set_title(title, loc="left", fontsize=9)
    axes[2].set_yticklabels([])
    cb = fig.colorbar(im, ax=axes[2], pad=0.02)
    cb.set_label(r"$\ln\mathrm{BF}$ (quasar / field)"); cb.outline.set_visible(False)
    fig.tight_layout()
    low = (lq < lq.max() + np.log(0.01)) & (lb < lb.max() + np.log(0.01))
    out["two_densities"] = {"n_field_points": int(s.size), "z0": z0, "r0": r0, "eta": eta,
                            "low_cells": float(low.mean()),
                            "low_bf_gt5_field_only": float(((lq - lb)[low] > 5).mean()),
                            "low_bf_gt5_scored": float(((lq - lf)[low] > 5).mean()),
                            "low_bf_gt0_field_only": float(((lq - lb)[low] > 0).mean()),
                            "low_bf_gt0_scored": float(((lq - lf)[low] > 0).mean()),
                            "max_bf_field_only": float((lq - lb).max()),
                            "max_bf_scored": float((lq - lf).max())}
    return save_figure(fig, "method/model_two_densities")


def fig_deconvolution(model, d, out):
    import matplotlib.pyplot as plt
    from qso_pcolor.plotting import SERIES, save_figure
    from qso_pcolor.xd import fit_xd

    q = d["q"]
    ig, ir = q["idx"][G], q["idx"][R]
    s = q["grz"] & (np.abs(q["z"] - 1.8) < 0.15) & (q["u"][R] < 20.0)
    x0 = (q["u"][G] - q["u"][R])[s]
    c = q["feat"].cov[s]
    v0 = c[:, ig, ig] + c[:, ir, ir] - 2 * c[:, ig, ir]

    def width(mix):
        m = float(mix.weights @ mix.means[:, 0])
        return float(np.sqrt(max(mix.weights @ (mix.covs[:, 0, 0] + mix.means[:, 0] ** 2) - m**2, 0)))

    base = fit_xd(x0[:, None], v0[:, None, None], n_components=2, seed=0, max_iter=400,
                  regularization=1e-9)
    sigma0 = width(base.mixture)
    rng = np.random.default_rng(0)
    fr = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    wx, wg, stored = [], [], {}
    for f in fr:
        sa = f * sigma0
        xn = x0 + rng.normal(0, sa, x0.size) if sa > 0 else x0.copy()
        vn = v0 + sa**2
        xd = fit_xd(xn[:, None], vn[:, None, None], n_components=2, seed=0, max_iter=400,
                    regularization=1e-9)
        gm = fit_xd(xn[:, None], None, n_components=2, seed=0, max_iter=400, regularization=1e-9)
        wx.append(width(xd.mixture)); wg.append(width(gm.mixture))
        if np.isclose(f, 1.0):
            stored = dict(xn=xn, xd=xd, gm=gm)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    lo, hi = np.percentile(x0, [0.5, 99.5])
    grid = np.linspace(lo - 0.3, hi + 0.3, 500)[:, None]
    ax.hist(np.clip(stored["xn"], grid[0, 0], grid[-1, 0]), bins=60, density=True,
            range=(grid[0, 0], grid[-1, 0]), color="#e3e3de", label="after noise injection")
    ax.plot(grid[:, 0], np.exp(stored["gm"].mixture.log_prob(grid)), color=SERIES["field_q"],
            label="ordinary mixture")
    ax.plot(grid[:, 0], np.exp(stored["xd"].mixture.log_prob(grid)), color=SERIES["same_z"],
            label="deconvolved")
    ax.plot(grid[:, 0], np.exp(base.mixture.log_prob(grid)), color="#2b2b28", ls="--", lw=1,
            label="truth (before injection)")
    ax.set_xlabel("Legacy $g - r$"); ax.set_ylabel("density")
    ax.set_title(r"Noise added at $\sigma_{\rm add} = \sigma_0$", loc="left")
    ax.legend(loc="upper right", fontsize=7)
    ax = axes[1]
    ax.axhline(sigma0, color="#2b2b28", ls="--", lw=1)
    ax.plot(fr, np.sqrt(sigma0**2 + (fr * sigma0) ** 2), color=SERIES["neutral"], ls=":",
            label=r"$\sqrt{\sigma_0^2+\sigma_{\rm add}^2}$")
    ax.plot(fr, wg, color=SERIES["field_q"], marker="o", ms=3.5, label="ordinary mixture")
    ax.plot(fr, wx, color=SERIES["same_z"], marker="o", ms=3.5, label="deconvolved")
    ax.set_xlabel(r"injected noise $\sigma_{\rm add}/\sigma_0$")
    ax.set_ylabel("recovered width of $g - r$")
    ax.set_title("Recovery under known added noise", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    out["deconvolution"] = {"n": int(x0.size), "sigma0": sigma0, "gmm_at_2": wg[-1],
                            "xd_at_2": wx[-1]}
    print(f"    deconvolution: N={x0.size} sigma0={sigma0:.3f} gmm@2={wg[-1]:.3f} xd@2={wx[-1]:.3f}")
    return save_figure(fig, "method/model_deconvolution")


def fig_model_across_redshift(model, d, out):
    import matplotlib.pyplot as plt
    from qso_pcolor.plotting import SERIES, save_figure

    q = d["q"]
    bands = model.transform.bands
    pairs = ((G, R, "$g - r$"), (R, Z, "$r - z$"), (Z, W1, "$z - W1$"), (W1, W2, "$W1 - W2$"))
    zs = np.linspace(Z_LO + 0.01, Z_HI - 0.01, 120)
    zc = model.qso.z_centres

    def quantiles(a, b_):
        ia, ib = bands.index(a), bands.index(b_)
        grid = np.linspace(-3, 6, 4000)
        res = np.zeros((zs.size, 3))
        for i, z in enumerate(zs):
            j = int(np.clip(np.searchsorted(zc, z) - 1, 0, zc.size - 2))
            w = (z - zc[j]) / (zc[j + 1] - zc[j])
            cdf = np.zeros_like(grid)
            for mix, wm in ((model.qso.mixtures[j], 1 - w), (model.qso.mixtures[j + 1], w)):
                mu = mix.means[:, ia] - mix.means[:, ib]
                sd = np.sqrt(mix.covs[:, ia, ia] + mix.covs[:, ib, ib] - 2 * mix.covs[:, ia, ib])
                cdf += wm * (mix.weights * norm.cdf((grid[:, None] - mu) / sd)).sum(1)
            res[i] = np.interp([0.16, 0.5, 0.84], cdf, grid)
        return res

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for ax, (a, b_, lab) in zip(axes.ravel(), pairs):
        ok = q["feat"].observed[:, bands.index(a)] & q["feat"].observed[:, bands.index(b_)]
        y = (q["u"][a] - q["u"][b_])[ok]
        zz = q["z"][ok]
        lo, hi = np.percentile(y, [0.5, 99.5]); pad = 0.1 * (hi - lo)
        ax.hexbin(zz, y, gridsize=70, extent=(Z_LO, Z_HI, lo - pad, hi + pad), bins="log",
                  cmap="Greys", mincnt=1, linewidths=0)
        edges = np.linspace(Z_LO, Z_HI, 43)
        cen = 0.5 * (edges[:-1] + edges[1:])
        obs = np.array([np.percentile(y[(zz >= e0) & (zz < e1)], [16, 84])
                        if ((zz >= e0) & (zz < e1)).sum() > 30 else [np.nan, np.nan]
                        for e0, e1 in zip(edges[:-1], edges[1:])])
        ax.plot(cen, obs[:, 0], color=SERIES["field_q"], lw=1, ls=":")
        ax.plot(cen, obs[:, 1], color=SERIES["field_q"], lw=1, ls=":",
                label="data, 16–84%" if a == G else None)
        mq = quantiles(a, b_)
        ax.fill_between(zs, mq[:, 0], mq[:, 2], color=SERIES["same_z"], alpha=0.25, lw=0,
                        label="model, 16–84%" if a == G else None)
        ax.plot(zs, mq[:, 1], color=SERIES["same_z"], lw=1.4,
                label="model median" if a == G else None)
        ax.set_ylabel(f"Legacy {lab}"); ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(Z_LO, Z_HI)
    for ax in axes[1]:
        ax.set_xlabel("redshift")
    axes[0, 0].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return save_figure(fig, "method/model_across_redshift")


def held_quasars(d, z0, near, n, rng, need=("grzw",)):
    q = d["q"]
    ok = q["held"] & q["grzw"] & (q["u"][R] < 22.5)
    ok &= (np.abs(q["z"] - z0) < near[1]) if near[0] == "near" else (np.abs(q["z"] - z0) > near[1])
    idx = np.flatnonzero(ok)
    return np.sort(rng.choice(idx, min(n, idx.size), replace=False))


def legacy_phot(dd, idx):
    from qso_pcolor.multisurvey_data import Photometry
    p = dd["phot"]
    cols = [p.bands.index(b) for b in LEGACY]
    return Photometry(p.flux[idx][:, cols], p.variance[idx][:, cols], LEGACY)


def fig_window(model, d, out, z0=1.8):
    import matplotlib.pyplot as plt
    from qso_pcolor.multisurvey import _ConditionalQSO
    from qso_pcolor.plotting import SERIES, save_figure
    from qso_pcolor.qso_model import RedshiftMatch

    q = d["q"]
    rng = np.random.default_rng(0)
    cq = _ConditionalQSO(model.qso, q["idx"][R])
    zg = np.linspace(Z_LO, Z_HI, 1200)

    def posterior(idx):
        f = model.transform(legacy_phot(q, idx).align(model.transform.bands))
        return cq.redshift_posterior(f.x, f.cov, zg, observed=f.observed)

    three = held_quasars(d, z0, ("near", 0.03), 3, rng)
    many = held_quasars(d, 0.0, ("far", -1), 1500, rng)       # all reserved quasars
    post = posterior(many)
    mean = np.trapezoid(post * zg, zg, axis=1)
    sd = np.sqrt(np.trapezoid(post * (zg - mean[:, None]) ** 2, zg, axis=1))
    out["window"] = {"n": int(many.size), "median_sigma_z": float(np.median(sd)),
                     "sigma_z_16_84": np.percentile(sd, [16, 84]).tolist(),
                     "median_abs_mean_minus_zspec": float(np.median(np.abs(mean - q["z"][many])))}
    print(f"    sigma_z median {np.median(sd):.3f} over {many.size} reserved quasars")
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
    p3 = posterior(three)
    for j, pp in enumerate(p3):
        axes[0].plot(zg, pp, color=SERIES["same_z"], alpha=0.85 - 0.22 * j,
                     label=r"quasar-only $p(z\,|\,\mathbf{u},Q)$" if j == 0 else None)
    axes[0].axvline(z0, color=SERIES["neutral"], lw=0.9, ls="--")
    axes[0].legend(loc="upper right", fontsize=7.5)
    axes[0].set_ylabel(r"$p(z\,|\,\mathbf{u},Q)$")
    axes[0].set_title(r"Three reserved quasars with $z_{\rm spec}\approx1.8$, Legacy $g,r,z,W1,W2$",
                      loc="left")
    ax = axes[1]
    ax.plot(zg, p3[0], color=SERIES["same_z"])
    hw = RedshiftMatch(half_width_kms=2000.0).half_width(z0)
    ax.axvspan(z0 - hw, z0 + hw, color=SERIES["field_q"], alpha=0.85, lw=0)
    ax.annotate(f"$\\pm2000$ km s$^{{-1}}$\n$\\Delta z = {2*hw:.3f}$", (z0 + hw, 0.6 * p3[0].max()),
                xytext=(10, 0), textcoords="offset points", fontsize=8, color=SERIES["field_q"])
    m0 = np.trapezoid(p3[0] * zg, zg)
    s0 = np.sqrt(np.trapezoid(p3[0] * (zg - m0) ** 2, zg))
    ax.annotate(f"this object: $\\sigma_z \\approx {s0:.2f}$", (0.03, 0.88), xycoords="axes fraction",
                fontsize=8, color=SERIES["same_z"])
    ax.set_xlabel("redshift"); ax.set_ylabel(r"$p(z\,|\,\mathbf{u},Q)$")
    ax.set_title("The match window against the measurement", loc="left")
    fig.tight_layout()
    return save_figure(fig, "method/model_window")


def fig_factorisation(model, priors, outl, d, out, z0=1.8):
    import matplotlib.pyplot as plt
    from qso_pcolor.plotting import SERIES, save_figure
    from qso_pcolor.qso_model import RedshiftMatch

    q = d["q"]
    i = held_quasars(d, z0, ("near", 0.03), 1, np.random.default_rng(1))
    ph = legacy_phot(q, i)
    kms = np.geomspace(200.0, 20000.0, 22)
    ps, rs, ws = [], [], []
    for v in kms:
        s = model.score(ph, z_primary=np.array([z0]), l_deg=q["raw"]["l"][i], b_deg=q["raw"]["b"][i],
                        match=RedshiftMatch(half_width_kms=float(v)), min_bands=2, priors=priors,
                        outlier=outl)[0]
        ps.append(s.p_sameq); rs.append(np.exp(s.log_r_per_unit_z)); ws.append(s.dz_match_eff)
    ps, rs, ws = map(np.asarray, (ps, rs, ws))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    ax.plot(ws, ps, color=SERIES["same_z"], marker="o", ms=3.5)
    ax.plot(ws, rs[0] * ws, color=SERIES["neutral"], ls=":", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"window area $\Delta Z_{\rm eff}$"); ax.set_ylabel(r"$p_{\rm same}$")
    ax.set_title("The posterior tracks the window", loc="left")
    ax = axes[1]
    ax.plot(ws, rs, color=SERIES["field_q"], marker="o", ms=3.5)
    ax.set_xscale("log"); ax.set_ylim(0, 1.35 * rs.max())
    ax.set_xlabel(r"window area $\Delta Z_{\rm eff}$"); ax.set_ylabel(r"$R$ [redshift$^{-1}$]")
    ax.set_title("The evidence does not", loc="left")
    fig.tight_layout()
    narrow = ws < 0.05
    out["factorisation"] = {"R_range_narrow": [float(rs[narrow].min()), float(rs[narrow].max())],
                            "rel_change_narrow": float(rs[narrow].max() / rs[narrow].min() - 1),
                            "dz_narrow": [float(ws[narrow].min()), float(ws[narrow].max())]}
    return save_figure(fig, "method/model_factorisation")


def fig_separation(model, priors, outl, d, out, z0=1.8):
    import matplotlib.pyplot as plt
    from qso_pcolor.plotting import SERIES, save_figure
    from qso_pcolor.qso_model import RedshiftMatch

    q, b = d["q"], d["b"]
    rng = np.random.default_rng(0)
    groups = {"same_z": (q, held_quasars(d, z0, ("near", 0.05), 400, rng)),
              "field_q": (q, held_quasars(d, z0, ("far", 0.6), 400, rng))}
    okb = b["held"] & b["grzw"] & (b["u"][R] > 19.5) & (b["u"][R] < 21.5)
    groups["background"] = (b, np.sort(rng.choice(np.flatnonzero(okb), 400, replace=False)))
    res = {}
    for key, (dd, idx) in groups.items():
        rows = model.score(legacy_phot(dd, idx), z_primary=np.full(idx.size, z0),
                           l_deg=dd["raw"]["l"][idx], b_deg=dd["raw"]["b"][idx],
                           match=RedshiftMatch(half_width_kms=2000.0), min_bands=2,
                           priors=priors, outlier=outl)
        res[key] = np.array([r.log_r_per_unit_z for r in rows])
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bins = np.linspace(-45, 5, 60)
    labels = {"same_z": f"quasars at $z\\approx{z0}$", "field_q": "quasars at other $z$",
              "background": "field sources"}
    for key, v in res.items():
        v = v[np.isfinite(v)]
        ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype="step", color=SERIES[key],
                lw=1.4, density=True, label=labels[key])
        ax.axvline(np.median(v), color=SERIES[key], lw=0.7, alpha=0.5)
    ax.set_xlabel(r"$\ln R$ [redshift$^{-1}$]"); ax.set_ylabel("density of objects")
    ax.set_title(f"Ranking statistic against a primary at $z_0={z0}$", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    out["separation"] = {k: {"n": int(np.isfinite(v).sum()), "median_log_r": float(np.nanmedian(v))}
                         for k, v in res.items()}
    for k, v in out["separation"].items():
        print(f"    {k:11s} n={v['n']} median ln R = {v['median_log_r']:.2f}")
    return save_figure(fig, "method/model_separation")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/multisurvey_lsw.json"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--report", type=Path, default=Path("data/model_figures.json"))
    args = ap.parse_args()
    from qso_pcolor.plotting import use_paper_style

    use_paper_style()
    model, priors, outl, d = load(args.config)
    out = json.loads(args.report.read_text()) if args.report.exists() else {}
    figs = {"locus": lambda: fig_locus(model, d, out),
            "slices": lambda: fig_slices(model, d, out),
            "two_densities": lambda: fig_two_densities(model, outl, d, out),
            "deconvolution": lambda: fig_deconvolution(model, d, out),
            "across_redshift": lambda: fig_model_across_redshift(model, d, out),
            "window": lambda: fig_window(model, d, out),
            "factorisation": lambda: fig_factorisation(model, priors, outl, d, out),
            "separation": lambda: fig_separation(model, priors, outl, d, out)}
    for name, fn in figs.items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        print(f"  {fn()}  ({time.time() - t0:.0f} s)")
    args.report.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
