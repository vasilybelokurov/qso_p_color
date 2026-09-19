#!/usr/bin/env python
"""Generate the figures for the method write-up, from real WSDB photometry.

Every panel is built from DESI DR1 quasars and Legacy Surveys DR9 imaging, not
from a cartoon, so the reader sees what the model actually has to cope with.
Queries are cached, so a re-run costs seconds.

    python scripts/make_method_figures.py

Writes PNGs into ``plots/method/``.  Figures are numbered to match the sections
of ``docs/method/method.tex``.

Scope note: the models fitted here are deliberately small -- one sky patch, a
coarse redshift grid -- because the job of these figures is to *illustrate the
method*, not to report performance.  Nothing here is held out, so no accuracy
claim can be read off them.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

BANDS = ("g", "r", "z", "w1", "w2")

QSO_QUERY = """
SELECT z.z AS zspec, p.ra, p.dec,
       p.flux_g, p.flux_r, p.flux_z, p.flux_w1, p.flux_w2,
       p.flux_ivar_g, p.flux_ivar_r, p.flux_ivar_z,
       p.flux_ivar_w1, p.flux_ivar_w2,
       p.mw_transmission_g, p.mw_transmission_r, p.mw_transmission_z,
       p.mw_transmission_w1, p.mw_transmission_w2
FROM desi_dr1.zpix z
JOIN desi_dr1.photometry p ON p.targetid = z.targetid
WHERE z.spectype = 'QSO' AND z.zwarn = 0 AND z.zcat_primary
  AND z.z > {zmin} AND z.z < {zmax}
  AND q3c_radial_query(p.ra, p.dec, {ra}, {dec}, {radius})
"""

BKG_QUERY = """
SELECT ra, dec,
       flux_g, flux_r, flux_z, flux_w1, flux_w2,
       flux_ivar_g, flux_ivar_r, flux_ivar_z, flux_ivar_w1, flux_ivar_w2,
       mw_transmission_g, mw_transmission_r, mw_transmission_z,
       mw_transmission_w1, mw_transmission_w2
FROM decals_dr9.main
WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {radius})
  AND maskbits = 0 AND flux_ivar_r > 0 AND flux_r > 0 AND flux_r < 250
"""

# Release-aware: north (BASS/MzLS) and south (DECam) are different
# photometric systems and their models must not be interchangeable.
SYSTEM = "ls_dr9_south_grzw"

Z_MIN, Z_MAX = 0.4, 3.6
MAG_EDGES = np.array([17.0, 19.5, 20.5, 21.5, 22.5])


# ----------------------------------------------------------------------------
# data


def stack(r, prefix):
    return np.stack([np.asarray(r[f"{prefix}{b}"], float) for b in BANDS], axis=1)


def display_colours(flux, var):
    """(g-r, r-z) for plotting only, from bands with a positive flux.

    The model never needs this: it works in relative fluxes, which stay defined
    when a flux is negative.  Magnitudes are used here purely because a reader
    knows where the stellar locus sits in a colour-colour diagram.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        m = 22.5 - 2.5 * np.log10(np.where(flux > 0, flux, np.nan))
    ok = np.isfinite(m[:, 0]) & np.isfinite(m[:, 1]) & np.isfinite(m[:, 2])
    return m[:, 0] - m[:, 1], m[:, 1] - m[:, 2], ok


def load(args):
    from qso_pcolor.data import cached_query, galactic_from_equatorial
    from qso_pcolor.features import RelativeFluxTransform, deredden

    qso_q = QSO_QUERY.format(
        zmin=Z_MIN, zmax=Z_MAX, ra=args.ra, dec=args.dec, radius=args.qso_radius
    )
    bkg_q = BKG_QUERY.format(ra=args.ra, dec=args.dec, radius=args.bkg_radius)

    t0 = time.time()
    q = cached_query(qso_q, args.cache / "method_qso.npz")
    b = cached_query(bkg_q, args.cache / "method_bkg.npz")
    print(f"  quasars {q['zspec'].size:,}   background {b['ra'].size:,} "
          f"({time.time() - t0:.0f} s)")

    tr = RelativeFluxTransform(reference_band="r", min_ref_snr=5.0)
    out = {}
    for key, r in (("q", q), ("b", b)):
        if "maskbits" in r:
            r = {k: np.asarray(v)[np.asarray(r["maskbits"]) == 0] for k, v in r.items()}
        f, v = deredden(stack(r, "flux_"), stack(r, "flux_ivar_"),
                        stack(r, "mw_transmission_"))
        fs = tr(f, v, BANDS)
        gr, rz, ok_col = display_colours(f, v)
        l, bb = galactic_from_equatorial(r["ra"], r["dec"])
        out[key] = dict(
            raw=r, feat=fs, gr=gr, rz=rz, ok_col=ok_col, l=l, b=bb,
            ok=fs.usable(min_dims=3) & np.isfinite(fs.ref_mag),
        )
    out["q"]["z"] = q["zspec"]
    return out


def fit_models(d, args):
    """Fit the illustration models, caching them so re-running a figure is cheap."""
    from qso_pcolor.background import BackgroundColourModel, fit_background_model
    from qso_pcolor.priors import (
        BackgroundSurfaceDensity, EmpiricalQSOPrior, GridQSOPrior,
    )
    from qso_pcolor.qso_model import SlicedColourRedshiftModel, fit_sliced_model

    q, b = d["q"], d["b"]
    mdir = Path("models")
    mdir.mkdir(exist_ok=True)
    paths = {k: mdir / f"method_{k}.json" for k in ("qso", "bkg", "qp", "bd")}

    z_edges = np.linspace(Z_MIN, Z_MAX, args.n_slices + 1)
    okq = q["ok"]
    okb = b["ok"] & (b["feat"].ref_mag < MAG_EDGES[-1])

    if all(p.exists() for p in paths.values()) and not args.refit:
        print("  reusing cached models (pass --refit to rebuild)")
        return (
            SlicedColourRedshiftModel.load(paths["qso"]),
            BackgroundColourModel.load(paths["bkg"]),
            GridQSOPrior.load(paths["qp"]),
            BackgroundSurfaceDensity.load(paths["bd"]),
            okq, okb,
        )

    t0 = time.time()
    qso = fit_sliced_model(
        q["feat"].x[okq], q["feat"].cov[okq], q["z"][okq], z_edges=z_edges,
        observed=q["feat"].observed[okq], n_components=args.n_components,
        min_per_slice=150, overlap=0.5, system=SYSTEM,
        labels=q["feat"].labels, seed=0, max_iter=200, regularization=1e-6,
    )
    print(f"  quasar model: {args.n_slices} slices, "
          f"{qso.n_train.astype(int).min()}-{qso.n_train.astype(int).max()} "
          f"objects each ({time.time() - t0:.0f} s)")

    # Known quasars must come out of the background, or the background model
    # learns the quasar locus: measured, they are 1% of the sample overall but
    # 72% of it where log BF > 5.
    from qso_pcolor.data import drop_known_quasars, fetch_known_quasars
    qq = fetch_known_quasars(args.ra, args.dec, args.bkg_radius,
                             cache=args.cache / "method_bkg_qso.npz")
    notq = drop_known_quasars(b["raw"]["ra"], b["raw"]["dec"], qq["ra"], qq["dec"])
    n_before = int(okb.sum())
    okb = okb & notq
    print(f"  background: removed {n_before - int(okb.sum()):,} known quasars "
          f"({100 * (n_before - int(okb.sum())) / max(n_before, 1):.2f}%)")

    t0 = time.time()
    bkg = fit_background_model(
        b["feat"].x[okb], b["feat"].cov[okb], b["feat"].ref_mag[okb],
        b["l"][okb], b["b"][okb], mag_edges=MAG_EDGES, nside=8, nside_parent=2,
        observed=b["feat"].observed[okb], n_components=8, min_per_cell=800,
        n0=500.0, system=SYSTEM, labels=b["feat"].labels, seed=0,
        max_iter=200, regularization=1e-6,
    )
    print(f"  background model ({time.time() - t0:.0f} s)")

    qp = EmpiricalQSOPrior.build(
        q["z"][okq], q["feat"].ref_mag[okq],
        area_deg2=np.pi * args.qso_radius**2,
        z_edges=z_edges, mag_edges=MAG_EDGES,
    )
    # The surveyed area is the background CONE, not the HEALPix cells it falls
    # in.  An nside=8 pixel is 53.7 deg^2 against a 1 deg cone of 3.14 deg^2:
    # assuming the pixel understated Sigma_B by 17x and inflated every quasar
    # posterior by the same factor.
    bd = BackgroundSurfaceDensity.from_catalogue(
        b["feat"].ref_mag[okb], b["l"][okb], b["b"][okb],
        mag_edges=MAG_EDGES, nside=8, nside_parent=2,
        total_area_deg2=np.pi * args.bkg_radius**2,
    )
    for obj, key in ((qso, "qso"), (bkg, "bkg"), (qp, "qp"), (bd, "bd")):
        obj.save(paths[key])
    return qso, bkg, qp, bd, okq, okb


# ----------------------------------------------------------------------------
# figures


def fig_locus(d, okq, okb):
    """Fig 1: the quasar colour locus moves with redshift, non-monotonically."""
    import matplotlib.pyplot as plt
    from qso_pcolor.plotting import SEQUENTIAL_CMAP, SERIES, save_figure

    q, b = d["q"], d["b"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    ax = axes[0]
    sel = okb & b["ok_col"]
    ax.hexbin(b["rz"][sel], b["gr"][sel], gridsize=90, extent=(-1, 3, -0.7, 2.5),
              bins="log", cmap="Greys", mincnt=1, linewidths=0)
    selq = okq & q["ok_col"]
    sc = ax.scatter(q["rz"][selq], q["gr"][selq], c=q["z"][selq], s=1.4,
                    cmap=SEQUENTIAL_CMAP, vmin=Z_MIN, vmax=Z_MAX, alpha=0.75,
                    linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("spectroscopic redshift")
    cb.outline.set_visible(False)
    ax.set_xlabel("$r - z$")
    ax.set_ylabel("$g - r$")
    ax.set_xlim(-1, 3)
    ax.set_ylim(-0.7, 2.5)
    ax.set_title("Quasars over the field population", loc="left")
    ax.text(0.04, 0.94, "grey: all catalogue sources", transform=ax.transAxes,
            fontsize=7.5, color=SERIES["neutral"], va="top")

    # Median track, which is what a linear model would have to follow.
    ax = axes[1]
    zc = np.linspace(Z_MIN, Z_MAX, 33)
    for lab, col, colr in (("$g - r$", "gr", SERIES["same_z"]),
                           ("$r - z$", "rz", SERIES["field_q"])):
        med = np.array([
            np.nanmedian(q[col][selq][np.abs(q["z"][selq] - z) < 0.08])
            for z in zc
        ])
        lo = np.array([
            np.nanpercentile(q[col][selq][np.abs(q["z"][selq] - z) < 0.08], 16)
            for z in zc
        ])
        hi = np.array([
            np.nanpercentile(q[col][selq][np.abs(q["z"][selq] - z) < 0.08], 84)
            for z in zc
        ])
        ax.fill_between(zc, lo, hi, color=colr, alpha=0.18, linewidth=0)
        ax.plot(zc, med, color=colr, label=lab)
        ax.annotate(lab, (zc[-1], med[-1]), xytext=(4, 0),
                    textcoords="offset points", color=colr, fontsize=8, va="center")
    ax.set_xlabel("spectroscopic redshift")
    ax.set_ylabel("colour")
    ax.set_xlim(Z_MIN, Z_MAX + 0.25)
    ax.set_title("Median colour vs redshift, 16–84th percentile", loc="left")

    fig.tight_layout()
    return save_figure(fig, "method/fig1_qso_locus")


def fig_slices(d, qso, okq):
    """Fig 2: the fitted conditional density at three redshifts."""
    import matplotlib.pyplot as plt
    from qso_pcolor.plotting import SERIES, save_figure

    q = d["q"]
    zs = [0.9, 1.8, 3.0]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True, sharey=True)

    # Model space is relative flux; show the two optical dimensions.
    ix, iy = 0, 1                      # g/r and z/r
    gx = np.linspace(-0.2, 2.2, 140)
    gy = np.linspace(-0.2, 3.2, 140)
    gxx, gyy = np.meshgrid(gx, gy, indexing="ij")
    npts = gxx.size

    for ax, z0 in zip(axes, zs):
        pts = np.full((npts, qso.n_dim), np.nan)
        pts[:, ix] = gxx.ravel()
        pts[:, iy] = gyy.ravel()
        obs = np.zeros((npts, qso.n_dim), dtype=bool)
        obs[:, [ix, iy]] = True
        lp = qso.log_p_colour_given_z(pts, None, np.array([z0]), observed=obs)[:, 0]
        p = np.exp(lp - lp.max()).reshape(gxx.shape)

        sel = okq & (np.abs(q["z"] - z0) < 0.1)
        ax.scatter(q["feat"].x[sel, ix], q["feat"].x[sel, iy], s=2.0,
                   color=SERIES["neutral"], alpha=0.35, linewidths=0,
                   rasterized=True)
        ax.contour(gx, gy, p.T, levels=[0.05, 0.2, 0.5, 0.85],
                   colors=SERIES["same_z"], linewidths=1.0)
        ax.set_title(f"$z = {z0}$", loc="left")
        ax.set_xlabel("$f_g / f_r$")
        ax.set_xlim(-0.2, 2.2)
        ax.set_ylim(-0.2, 3.2)
    axes[0].set_ylabel("$f_z / f_r$")
    axes[0].text(0.05, 0.05, "points: quasars in the slice\nlines: fitted density\n"
                 "(narrower, because deconvolved)",
                 transform=axes[0].transAxes, fontsize=7.5, va="bottom",
                 color=SERIES["neutral"])
    fig.tight_layout()
    return save_figure(fig, "method/fig2_model_slices")


def fig_two_densities(d, qso, bkg, okb):
    """Fig 3: the quasar density at z0 against the local background density."""
    import matplotlib.pyplot as plt
    from qso_pcolor.plotting import SERIES, save_figure

    b = d["b"]
    z0 = 1.8
    ref_mag, l0, b0 = 20.8, float(np.median(b["l"])), float(np.median(b["b"]))
    ix, iy = 0, 1
    gx = np.linspace(-0.2, 2.4, 160)
    gy = np.linspace(-0.2, 3.4, 160)
    gxx, gyy = np.meshgrid(gx, gy, indexing="ij")
    npts = gxx.size
    pts = np.full((npts, qso.n_dim), np.nan)
    pts[:, ix], pts[:, iy] = gxx.ravel(), gyy.ravel()
    obs = np.zeros((npts, qso.n_dim), dtype=bool)
    obs[:, [ix, iy]] = True

    lq = qso.log_p_colour_given_z(pts, None, np.array([z0]), observed=obs)[:, 0]
    lb = bkg.log_prob(pts, None, np.full(npts, ref_mag), np.full(npts, l0),
                      np.full(npts, b0), observed=obs)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    ax = axes[0]
    for lp, key, lab in ((lq, "same_z", f"quasar at $z={z0}$"),
                         (lb, "background", "background")):
        p = np.exp(lp - lp.max()).reshape(gxx.shape)
        cs = ax.contour(gx, gy, p.T, levels=[0.05, 0.3, 0.8],
                        colors=SERIES[key], linewidths=1.2)
        ax.plot([], [], color=SERIES[key], label=lab)
    ax.legend(loc="upper right")
    ax.set_xlabel("$f_g / f_r$")
    ax.set_ylabel("$f_z / f_r$")
    ax.set_title("Two class-conditional densities", loc="left")

    ax = axes[1]
    bf = (lq - lb).reshape(gxx.shape)
    lim = 12
    im = ax.pcolormesh(gx, gy, np.clip(bf, -lim, lim).T, cmap="RdBu_r",
                       vmin=-lim, vmax=lim, shading="auto", rasterized=True)
    ax.contour(gx, gy, bf.T, levels=[0.0], colors="#2b2b28", linewidths=0.9)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\log \mathrm{BF}$  (quasar / background)")
    cb.outline.set_visible(False)
    sel = okb & (np.abs(b["feat"].ref_mag - ref_mag) < 0.4)
    idx = np.flatnonzero(sel)[:4000]
    ax.scatter(b["feat"].x[idx, ix], b["feat"].x[idx, iy], s=1.2, color="k",
               alpha=0.18, linewidths=0, rasterized=True)
    ax.set_xlabel("$f_g / f_r$")
    ax.set_title("Colour Bayes factor; black line is unity", loc="left")
    ax.set_xlim(-0.2, 2.4)
    ax.set_ylim(-0.2, 3.4)

    fig.tight_layout()
    return save_figure(fig, "method/fig3_two_densities")


def fig_deconvolution(d, qso, okq):
    """Fig 4: controlled noise injection on real quasar colours.

    The honest way to show what deconvolution buys.  Take bright quasars, whose
    measurement errors are negligible, so their fitted width is effectively the
    intrinsic one.  Add Gaussian noise of known size and refit.  An ordinary
    mixture must broaden as sqrt(sigma_0^2 + sigma_added^2); extreme
    deconvolution, given the same noise as input, must stay flat.
    """
    import matplotlib.pyplot as plt
    from qso_pcolor.plotting import SERIES, save_figure
    from qso_pcolor.xd import fit_xd

    q = d["q"]
    z0 = 1.8
    dim, label = 0, "$f_g / f_r$"
    sel = (okq & (np.abs(q["z"] - z0) < 0.15) & (q["feat"].ref_mag < 20.0))
    x0 = q["feat"].x[sel][:, dim]
    v0 = q["feat"].cov[sel][:, dim, dim]
    keep = np.isfinite(x0) & (v0 > 0)
    x0, v0 = x0[keep], v0[keep]

    def width(mix):
        m = float(mix.weights @ mix.means[:, 0])
        v = float(mix.weights @ (mix.covs[:, 0, 0] + mix.means[:, 0] ** 2) - m**2)
        return np.sqrt(max(v, 0.0))

    base = fit_xd(x0[:, None], v0[:, None, None], n_components=2, seed=0,
                  max_iter=400, regularization=1e-9)
    sigma0 = width(base.mixture)
    rng = np.random.default_rng(0)

    fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    w_xd, w_gmm, stored = [], [], {}
    for f in fractions:
        s_add = f * sigma0
        xn = x0 + rng.normal(0.0, s_add, size=x0.size) if s_add > 0 else x0.copy()
        vn = v0 + s_add**2
        xd = fit_xd(xn[:, None], vn[:, None, None], n_components=2, seed=0,
                    max_iter=400, regularization=1e-9)
        gmm = fit_xd(xn[:, None], None, n_components=2, seed=0, max_iter=400,
                     regularization=1e-9)
        w_xd.append(width(xd.mixture))
        w_gmm.append(width(gmm.mixture))
        if np.isclose(f, 1.0):
            stored = dict(xn=xn, xd=xd, gmm=gmm, s_add=s_add)
    w_xd, w_gmm = np.asarray(w_xd), np.asarray(w_gmm)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ax = axes[0]
    lo, hi = np.percentile(x0, [0.5, 99.5])
    grid = np.linspace(lo - 0.3, hi + 0.3, 500)[:, None]
    ax.hist(np.clip(stored["xn"], grid[0, 0], grid[-1, 0]), bins=60, density=True,
            range=(grid[0, 0], grid[-1, 0]), color="#e3e3de", edgecolor="none",
            label="after noise injection")
    ax.plot(grid[:, 0], np.exp(stored["gmm"].mixture.log_prob(grid)),
            color=SERIES["field_q"], label="ordinary mixture")
    ax.plot(grid[:, 0], np.exp(stored["xd"].mixture.log_prob(grid)),
            color=SERIES["same_z"], label="deconvolved")
    ax.plot(grid[:, 0], np.exp(base.mixture.log_prob(grid)), color="#2b2b28",
            linestyle="--", linewidth=1.0, label="truth (before injection)")
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    ax.set_title("Noise added at $\\sigma_{\\rm add} = \\sigma_0$", loc="left")
    ax.legend(loc="upper left", fontsize=7)

    ax = axes[1]
    ax.axhline(sigma0, color="#2b2b28", linestyle="--", linewidth=1.0)
    ax.annotate("true intrinsic width $\\sigma_0$", (fractions[-1], sigma0),
                xytext=(-4, 5), textcoords="offset points", ha="right",
                fontsize=7.5, color="#2b2b28")
    ax.plot(fractions, np.sqrt(sigma0**2 + (fractions * sigma0) ** 2),
            color=SERIES["neutral"], linestyle=":",
            label=r"$\sqrt{\sigma_0^2+\sigma_{\rm add}^2}$")
    ax.plot(fractions, w_gmm, color=SERIES["field_q"], marker="o", markersize=3.5,
            label="ordinary mixture")
    ax.plot(fractions, w_xd, color=SERIES["same_z"], marker="o", markersize=3.5,
            label="deconvolved")
    ax.set_xlabel(r"injected noise $\sigma_{\rm add} / \sigma_0$")
    ax.set_ylabel(f"recovered width of {label}")
    ax.set_title("Recovery under known added noise", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    path = save_figure(fig, "method/fig4_deconvolution")
    print(f"    sigma_0 = {sigma0:.3f} from N = {x0.size} bright quasars; "
          f"at sigma_add = 2 sigma_0 the ordinary mixture gives "
          f"{w_gmm[-1]:.3f} and XD gives {w_xd[-1]:.3f}")
    return path


def fig_model_across_redshift(d, qso, okq):
    """Fig 8: the fitted XD model itself, across all of redshift and every feature.

    Figure 2 shows the model as contours in one plane at three redshifts. This
    shows what was actually trained: a density that moves continuously with
    redshift, in each of the four features the model uses. Plotting the model's
    intrinsic interval against the data's observed interval also makes the
    deconvolution visible in the place it matters -- the model band is narrower
    than the scatter of the points it was fitted to.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from qso_pcolor.plotting import SERIES, save_figure

    q = d["q"]
    labels = qso.labels
    zs = np.linspace(Z_MIN + 0.02, Z_MAX - 0.02, 120)

    def model_quantiles(dim, qs=(0.16, 0.5, 0.84)):
        """Quantiles of the model's marginal in one feature, versus redshift.

        The grid must span the model, not a guessed range: the WISE relative
        fluxes reach several tens, and a grid stopping at 4 silently clamped
        their upper quantile, making the model look far tighter than it is.
        """
        out = np.zeros((zs.size, len(qs)))
        mu_all = np.concatenate([m.means[:, dim] for m in qso.mixtures])
        sd_all = np.concatenate([np.sqrt(m.covs[:, dim, dim]) for m in qso.mixtures])
        grid = np.linspace((mu_all - 6 * sd_all).min(),
                           (mu_all + 6 * sd_all).max(), 4000)
        for i, z in enumerate(zs):
            j = int(np.clip(np.searchsorted(qso.z_centres, z) - 1,
                            0, qso.z_centres.size - 2))
            lo, hi = qso.z_centres[j], qso.z_centres[j + 1]
            f = (z - lo) / (hi - lo)
            cdf = np.zeros_like(grid)
            for mix, wgt in ((qso.mixtures[j], 1 - f), (qso.mixtures[j + 1], f)):
                mu = mix.means[:, dim]
                sd = np.sqrt(mix.covs[:, dim, dim])
                cdf += wgt * (mix.weights[None, :] *
                              norm.cdf((grid[:, None] - mu) / sd)).sum(axis=1)
            out[i] = np.interp(qs, cdf, grid)
        return out

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for dim, ax in enumerate(axes.ravel()):
        sel = okq & np.isfinite(q["feat"].x[:, dim]) & q["feat"].observed[:, dim]
        y = q["feat"].x[sel, dim]
        zz = q["z"][sel]
        lo_y, hi_y = np.percentile(y, [0.5, 99.0])
        pad = 0.15 * (hi_y - lo_y)

        ax.hexbin(zz, y, gridsize=70, extent=(Z_MIN, Z_MAX, lo_y - pad, hi_y + pad),
                  bins="log", cmap="Greys", mincnt=1, linewidths=0)

        # what the data look like, errors included
        edges = np.linspace(Z_MIN, Z_MAX, 40)
        cen = 0.5 * (edges[:-1] + edges[1:])
        obs = np.array([
            np.percentile(y[(zz >= a) & (zz < b)], [16, 84])
            if ((zz >= a) & (zz < b)).sum() > 30 else [np.nan, np.nan]
            for a, b in zip(edges[:-1], edges[1:])
        ])
        ax.plot(cen, obs[:, 0], color=SERIES["field_q"], lw=1.0, ls=":")
        ax.plot(cen, obs[:, 1], color=SERIES["field_q"], lw=1.0, ls=":",
                label="data, 16–84%" if dim == 0 else None)

        # what the model says the intrinsic distribution is
        mq = model_quantiles(dim)
        ax.fill_between(zs, mq[:, 0], mq[:, 2], color=SERIES["same_z"], alpha=0.25,
                        lw=0, label="model, 16–84%" if dim == 0 else None)
        ax.plot(zs, mq[:, 1], color=SERIES["same_z"], lw=1.4,
                label="model median" if dim == 0 else None)

        ax.set_ylabel(f"$f_{{{labels[dim].split('/')[0]}}} / f_r$")
        ax.set_ylim(lo_y - pad, hi_y + pad)
        ax.set_xlim(Z_MIN, Z_MAX)
        if dim >= 2:
            ax.set_xlabel("redshift")
    axes[0, 0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("The fitted quasar model across redshift, in every feature it uses",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, "method/fig8_model_across_redshift")


def fig_window(d, qso, qp, bd, bkg, okq):
    """Fig 5: the redshift posterior against the velocity window. The key figure."""
    import matplotlib.pyplot as plt
    from qso_pcolor.plotting import SERIES, save_figure
    from qso_pcolor.qso_model import RedshiftMatch

    q = d["q"]
    z_grid = np.linspace(Z_MIN + 0.01, Z_MAX - 0.01, 900)
    z0 = 1.8
    sel = np.flatnonzero(okq & (np.abs(q["z"] - z0) < 0.03))[:3]

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0]})

    ax = axes[0]
    for j, i in enumerate(sel):
        post = qso.redshift_posterior(
            q["feat"].x[i:i + 1], q["feat"].cov[i:i + 1], z_grid,
            observed=q["feat"].observed[i:i + 1],
        )[0]
        ax.plot(z_grid, post, color=SERIES["same_z"], alpha=0.85 - 0.22 * j,
                label="quasar-only $p(z\\,|\\,\\mathbf{c},Q)$" if j == 0 else None)
    ax.axvline(z0, color=SERIES["neutral"], linewidth=0.9, linestyle="--")
    ax.set_ylabel("$p(z\\,|\\,\\mathbf{c}, Q)$")
    ax.legend(loc="upper right")
    ax.set_title("Three real quasars with $z_{\\rm spec} \\approx 1.8$", loc="left")
    ax.annotate("$z_0$", (z0, ax.get_ylim()[1] * 0.92), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=SERIES["neutral"])

    # Zoom: the window against the posterior it is being integrated over.
    ax = axes[1]
    i = sel[0]
    post = qso.redshift_posterior(
        q["feat"].x[i:i + 1], q["feat"].cov[i:i + 1], z_grid,
        observed=q["feat"].observed[i:i + 1],
    )[0]
    ax.plot(z_grid, post, color=SERIES["same_z"])
    for kms, key, ls in ((2000.0, "field_q", "-"),):
        m = RedshiftMatch(half_width_kms=kms)
        hw = m.half_width(z0)
        ax.axvspan(z0 - hw, z0 + hw, color=SERIES[key], alpha=0.85, linewidth=0)
        ax.annotate(
            f"$\\pm{kms:.0f}$ km s$^{{-1}}$\n$\\Delta z = {2*hw:.3f}$",
            (z0 + hw, ax.get_ylim()[1] * 0.6), xytext=(10, 0),
            textcoords="offset points", fontsize=8, color=SERIES["field_q"],
            va="center",
        )
    sigma = float(np.sqrt(np.trapezoid(post * (z_grid - np.trapezoid(
        post * z_grid, z_grid))**2, z_grid)))
    ax.annotate(
        f"posterior width $\\sigma_z \\approx {sigma:.2f}$",
        (0.03, 0.9), xycoords="axes fraction", fontsize=8,
        color=SERIES["same_z"],
    )
    ax.set_xlabel("redshift")
    ax.set_ylabel("$p(z\\,|\\,\\mathbf{c}, Q)$")
    ax.set_title("The match window against the measurement", loc="left")
    fig.tight_layout()
    return save_figure(fig, "method/fig5_window")


def fig_factorisation(d, qso, bkg, qp, bd, okq):
    """Fig 6: p_sameq is linear in the window; R is not."""
    import matplotlib.pyplot as plt
    from qso_pcolor.features import FeatureSet
    from qso_pcolor.plotting import SERIES, save_figure
    from qso_pcolor.qso_model import RedshiftMatch
    from qso_pcolor.score import score_candidates

    q = d["q"]
    z0 = 1.8
    i = int(np.flatnonzero(okq & (np.abs(q["z"] - z0) < 0.03))[0])
    fs = q["feat"]
    one = FeatureSet(fs.x[i:i + 1], fs.cov[i:i + 1], fs.observed[i:i + 1],
                     fs.ref_flux[i:i + 1], fs.ref_mag[i:i + 1],
                     fs.ref_snr[i:i + 1], fs.labels, {})

    kms = np.geomspace(200.0, 20000.0, 22)
    ps, rs, widths = [], [], []
    for v in kms:
        s = score_candidates(
            one, z_primary=np.array([z0]), l_deg=q["l"][i:i + 1],
            b_deg=q["b"][i:i + 1], qso_model=qso, background_model=bkg,
            match=RedshiftMatch(half_width_kms=float(v)), qso_prior=qp,
            background_density=bd,
            z_grid=np.linspace(Z_MIN + 0.01, Z_MAX - 0.01, 600),
        )[0]
        ps.append(s.p_sameq)
        rs.append(np.exp(s.log_r_per_unit_z))
        widths.append(s.dz_match_eff)
    ps, rs, widths = map(np.asarray, (ps, rs, widths))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ax = axes[0]
    ax.plot(widths, ps, color=SERIES["same_z"], marker="o", markersize=3.5)
    ax.plot(widths, rs[0] * widths, color=SERIES["neutral"], linestyle=":",
            linewidth=1.0)
    ax.annotate("$p_{\\rm same} = R\\,\\Delta Z$", (widths[-6], rs[0] * widths[-6]),
                xytext=(-6, 12), textcoords="offset points", fontsize=8,
                color=SERIES["neutral"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"window area $\Delta Z_{\rm eff}$")
    ax.set_ylabel(r"$p_{\rm same}$")
    ax.set_title("The posterior tracks the window", loc="left")

    ax = axes[1]
    ax.plot(widths, rs, color=SERIES["field_q"], marker="o", markersize=3.5)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.35 * rs.max())
    ax.set_xlabel(r"window area $\Delta Z_{\rm eff}$")
    ax.set_ylabel(r"$R$  [redshift$^{-1}$]")
    ax.set_title("The evidence does not", loc="left")
    ax.annotate(
        "flat until the window stops\nbeing narrow ($\\Delta Z \\gtrsim \\sigma_z$)",
        (0.04, 0.12), xycoords="axes fraction", fontsize=7.5,
        color=SERIES["neutral"],
    )
    fig.tight_layout()
    return save_figure(fig, "method/fig6_factorisation")


def fig_separation(d, qso, bkg, qp, bd, okq, okb):
    """Fig 7: what the statistic does to real quasars and real field sources."""
    import matplotlib.pyplot as plt
    from qso_pcolor.features import FeatureSet
    from qso_pcolor.plotting import SERIES, save_figure
    from qso_pcolor.qso_model import RedshiftMatch
    from qso_pcolor.score import score_candidates

    q, b = d["q"], d["b"]
    rng = np.random.default_rng(0)
    z0 = 1.8
    z_grid = np.linspace(Z_MIN + 0.01, Z_MAX - 0.01, 500)
    match = RedshiftMatch(half_width_kms=2000.0)

    def subset(src, idx):
        fs = src["feat"]
        return FeatureSet(fs.x[idx], fs.cov[idx], fs.observed[idx],
                          fs.ref_flux[idx], fs.ref_mag[idx], fs.ref_snr[idx],
                          fs.labels, {}), src["l"][idx], src["b"][idx]

    # Quasars whose own redshift matches z0 (true same_z), quasars far from it
    # (field_q), and random catalogue sources (background).
    i_same = rng.choice(np.flatnonzero(okq & (np.abs(q["z"] - z0) < 0.05)), 400)
    i_field = rng.choice(np.flatnonzero(okq & (np.abs(q["z"] - z0) > 0.6)), 400)
    i_bkg = rng.choice(
        np.flatnonzero(okb & (b["feat"].ref_mag > 19.5) & (b["feat"].ref_mag < 21.5)),
        400,
    )

    out = {}
    for key, src, idx in (("same_z", q, i_same), ("field_q", q, i_field),
                          ("background", b, i_bkg)):
        fs, l, bb = subset(src, idx)
        rows = score_candidates(
            fs, z_primary=np.full(idx.size, z0), l_deg=l, b_deg=bb,
            qso_model=qso, background_model=bkg, match=match, qso_prior=qp,
            background_density=bd, z_grid=z_grid, min_bands=3,
        )
        out[key] = np.array([r.log_r_per_unit_z for r in rows])

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bins = np.linspace(-45, 5, 60)
    labels = {"same_z": f"quasars at $z \\approx {z0}$",
              "field_q": "quasars at other $z$",
              "background": "random catalogue sources"}
    for key in ("same_z", "field_q", "background"):
        v = out[key]
        v = v[np.isfinite(v)]
        ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype="step",
                color=SERIES[key], linewidth=1.4, density=True, label=labels[key])
        ax.axvline(np.median(v), color=SERIES[key], linewidth=0.7, alpha=0.5)
    ax.set_xlabel(r"$\log R$   [redshift$^{-1}$]")
    ax.set_ylabel("density of objects")
    ax.set_title(f"Ranking statistic against a primary at $z_0 = {z0}$", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = save_figure(fig, "method/fig7_separation")
    for k, v in out.items():
        print(f"    {k:11s} median log R = {np.nanmedian(v):7.2f}")
    return path


# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ra", type=float, default=180.0)
    ap.add_argument("--dec", type=float, default=0.0)
    ap.add_argument("--qso-radius", type=float, default=12.0, help="degrees")
    ap.add_argument("--bkg-radius", type=float, default=1.0, help="degrees")
    ap.add_argument("--n-slices", type=int, default=24)
    ap.add_argument("--n-components", type=int, default=6)
    ap.add_argument("--cache", type=Path, default=Path("data"))
    ap.add_argument("--refit", action="store_true",
                    help="rebuild the models instead of loading models/method_*.json")
    ap.add_argument("--only", nargs="*", default=None,
                    help="regenerate only these figures, e.g. --only fig4 fig7")
    args = ap.parse_args()

    from qso_pcolor.plotting import use_paper_style

    use_paper_style()

    print("loading")
    d = load(args)
    print("fitting")
    qso, bkg, qp, bd, okq, okb = fit_models(d, args)

    print("figures")
    for fn, call in (
        ("fig1", lambda: fig_locus(d, okq, okb)),
        ("fig2", lambda: fig_slices(d, qso, okq)),
        ("fig3", lambda: fig_two_densities(d, qso, bkg, okb)),
        ("fig4", lambda: fig_deconvolution(d, qso, okq)),
        ("fig8", lambda: fig_model_across_redshift(d, qso, okq)),
        ("fig5", lambda: fig_window(d, qso, qp, bd, bkg, okq)),
        ("fig6", lambda: fig_factorisation(d, qso, bkg, qp, bd, okq)),
        ("fig7", lambda: fig_separation(d, qso, bkg, qp, bd, okq, okb)),
    ):
        if args.only and fn not in args.only:
            continue
        t0 = time.time()
        path = call()
        print(f"  {path.relative_to(path.parents[2])}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
