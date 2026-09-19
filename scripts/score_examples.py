#!/usr/bin/env python
"""Score a handful of real objects and show the two models they are scored against.

Takes five spectroscopic quasars and five point-like field sources from the same
patch of sky, scores each against a target redshift, and draws the quasar and
background densities in the plane the decision is being made in.

**Optical only.** Everything here uses the DECaLS bands alone -- g/r and z/r --
with W1 and W2 marginalised out exactly by the observed-dimension mask, not
dropped or imputed. That is deliberately the hard case: WISE carries most of the
quasar/star discrimination, so optical-only shows what the colours alone can do.

The quasars are drawn from the model's **reserved** spatial blocks, so none of
them was used in fitting.

    python scripts/score_examples.py --ra 180 --dec 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BANDS = ("g", "r", "z", "w1", "w2")
OPTICAL = (0, 1)          # g/r and z/r within the 4-feature vector
MAG_EDGES = np.array([17.0, 19.5, 20.5, 21.5, 22.5])

STAR_QUERY = """
SELECT ra, dec, type, maskbits,
       flux_g, flux_r, flux_z, flux_w1, flux_w2,
       flux_ivar_g, flux_ivar_r, flux_ivar_z, flux_ivar_w1, flux_ivar_w2,
       mw_transmission_g, mw_transmission_r, mw_transmission_z,
       mw_transmission_w1, mw_transmission_w2
FROM decals_dr9.main
WHERE q3c_radial_query(ra, dec, %(ra)s, %(dec)s, %(radius)s)
  AND maskbits = 0 AND type = 'PSF'
  AND flux_ivar_r > 0 AND flux_r > 0
  AND 22.5 - 2.5*log(flux_r) BETWEEN %(mlo)s AND %(mhi)s
"""


def stack(r, prefix, idx=None):
    a = np.stack([np.asarray(r[f"{prefix}{b}"], float) for b in BANDS], axis=1)
    return a if idx is None else a[idx]


def build_sigma_q(zmin, zmax, plateau_per_deg2):
    """Global, isotropic Sigma_Q(z, m), normalised to the coverage plateau.

    The observed spectroscopic density varies by ~50% across the sky purely
    through DESI's fibre coverage. That is a property of the survey, not of the
    sky: the true quasar density in a redshift bin is isotropic. So the shape is
    measured globally and the normalisation is set by the density in the
    best-covered cells, where completeness is highest -- a lower bound on the
    truth, but a far better one than the raw mean.
    """
    import healpy as hp

    from qso_pcolor.data import fetch_desi_qso_training, galactic_from_equatorial
    from qso_pcolor.features import RelativeFluxTransform, deredden
    from qso_pcolor.priors import EmpiricalQSOPrior

    r = fetch_desi_qso_training("data/desi_qso_full.npz", zmin=zmin, zmax=zmax)
    sel = np.asarray(r["release"], int) == 9010
    f, v = deredden(stack(r, "flux_", sel), stack(r, "flux_ivar_", sel),
                    stack(r, "mw_transmission_", sel))
    fs = RelativeFluxTransform(reference_band="r")(f, v, BANDS)
    ok = fs.usable(min_dims=3) & np.isfinite(fs.ref_mag)
    z = np.asarray(r["zspec"])[sel][ok]
    m = fs.ref_mag[ok]

    l, b = galactic_from_equatorial(np.asarray(r["ra"])[sel][ok],
                                    np.asarray(r["dec"])[sel][ok])
    nside = 16
    pix = hp.ang2pix(nside, l, b, nest=True, lonlat=True)
    cnt = np.bincount(pix)
    occupied = cnt > 200
    area = occupied.sum() * hp.nside2pixarea(nside, degrees=True)

    prior = EmpiricalQSOPrior.build(
        z, m, area_deg2=area, z_edges=np.linspace(zmin, zmax, 33),
        mag_edges=MAG_EDGES,
    )
    # rescale the whole grid so the integrated density matches the plateau
    zg = np.linspace(zmin + 0.05, zmax - 0.05, 200)
    tot = sum(np.trapezoid(prior(zg, mm), zg) * dm
              for mm, dm in zip(prior.mag_centres, np.diff(MAG_EDGES)))
    prior.sigma *= plateau_per_deg2 / tot
    prior.meta["normalised_to_plateau"] = plateau_per_deg2
    prior.meta["raw_area_deg2"] = float(area)
    print(f"  Sigma_Q: shape from {z.size:,} quasars over {area:,.0f} deg^2, "
          f"rescaled {plateau_per_deg2 / tot:.2f}x to {plateau_per_deg2:.0f}/deg^2")
    return prior


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ra", type=float, default=180.0)
    ap.add_argument("--dec", type=float, default=0.0)
    ap.add_argument("--radius", type=float, default=0.5, help="local background cone, deg")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--plateau", type=float, default=240.0,
                    help="true quasar density per deg^2 from the coverage plateau")
    ap.add_argument("--model", type=Path, default=Path("models/qso_south_full.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)

    from qso_pcolor.background import fit_local_background, galactic_healpix
    from qso_pcolor.data import (
        cached_query, drop_known_quasars, fetch_desi_qso_training,
        fetch_known_quasars, galactic_from_equatorial,
    )
    from qso_pcolor.features import FeatureSet, RelativeFluxTransform, deredden
    from qso_pcolor.qso_model import RedshiftMatch, SlicedColourRedshiftModel
    from qso_pcolor.score import score_candidates

    rng = np.random.default_rng(args.seed)
    qso = SlicedColourRedshiftModel.load(args.model)
    tr = RelativeFluxTransform(reference_band="r", min_ref_snr=5.0)
    print(f"quasar model: {qso.meta['n_train']:,} training quasars, "
          f"{len(qso.mixtures)} slices x {qso.mixtures[0].n_components} components")

    # ---- five quasars, from the model's RESERVED blocks ------------------
    r = fetch_desi_qso_training("data/desi_qso_full.npz", zmin=0.4, zmax=3.6)
    sel = np.asarray(r["release"], int) == 9010
    f, v = deredden(stack(r, "flux_", sel), stack(r, "flux_ivar_", sel),
                    stack(r, "mw_transmission_", sel))
    fq = tr(f, v, BANDS)
    zq = np.asarray(r["zspec"])[sel]
    lq, bq = galactic_from_equatorial(np.asarray(r["ra"])[sel],
                                      np.asarray(r["dec"])[sel])
    blocks = galactic_healpix(lq, bq, 4)
    rh = np.random.default_rng(qso.meta.get("holdout_seed", 0))
    ub = np.unique(blocks)
    nh = max(1, int(round(qso.meta.get("holdout_frac", 0.2) * ub.size)))
    held = set(rh.choice(ub, size=nh, replace=False).tolist())
    is_held = np.array([g in held for g in blocks])

    okq = (fq.usable(min_dims=3) & np.isfinite(fq.ref_mag) & is_held
           & (fq.ref_mag > 19.0) & (fq.ref_mag < 21.5))
    iq = rng.choice(np.flatnonzero(okq), args.n, replace=False)
    print(f"  {args.n} quasars drawn from the {nh} reserved blocks "
          f"(never used in fitting)")

    # ---- five point sources from the same cone ---------------------------
    st = cached_query(
        STAR_QUERY % {"ra": args.ra, "dec": args.dec, "radius": args.radius,
                      "mlo": 19.0, "mhi": 21.5},
        Path("data") / "example_stars.npz",
    )
    qq = fetch_known_quasars(args.ra, args.dec, args.radius,
                             cache=Path("data") / "example_qso.npz")
    notq = drop_known_quasars(st["ra"], st["dec"], qq["ra"], qq["dec"])
    fsr, vsr = deredden(stack(st, "flux_"), stack(st, "flux_ivar_"),
                        stack(st, "mw_transmission_"))
    fst = tr(fsr, vsr, BANDS)
    oks = fst.usable(min_dims=3) & np.isfinite(fst.ref_mag) & notq
    isx = rng.choice(np.flatnonzero(oks), args.n, replace=False)
    print(f"  {args.n} PSF sources from a {args.radius} deg cone, "
          f"{int((~notq).sum())} known quasars excluded")

    # ---- local background, quasars removed -------------------------------
    bkg, bdens, info = fit_local_background(
        args.ra, args.dec, args.radius, transform=tr, bands=BANDS,
        mag_edges=MAG_EDGES, system=qso.system, n_components=8,
        max_ref_mag=22.5, cache=Path("data") / "example_bkg.npz",
        seed=0, max_iter=300, regularization=1e-6,
    )
    print(f"  background: {info['n_fitted']:,} sources, "
          f"{info['n_known_quasars_removed']:,} quasars removed, "
          f"mask fraction {info['mask_fraction']:.3f}, "
          f"area {info['area_deg2']:.3f} deg^2")

    prior = build_sigma_q(0.4, 3.6, args.plateau)

    # ---- score, OPTICAL ONLY ---------------------------------------------
    def optical_only(fs, idx):
        obs = np.zeros((len(idx), 4), dtype=bool)
        obs[:, OPTICAL] = fs.observed[idx][:, OPTICAL]
        return FeatureSet(fs.x[idx], fs.cov[idx], obs, fs.ref_flux[idx],
                          fs.ref_mag[idx], fs.ref_snr[idx], fs.labels, {})

    match = RedshiftMatch(half_width_kms=2000.0)
    zgrid = np.linspace(0.46, 3.54, 400)
    rows = {}
    for tag, fs, idx, zt, ll, bb in (
        ("quasar", fq, iq, zq[iq], lq[iq], bq[iq]),
        ("PSF source", fst, isx, rng.choice(zq[iq], args.n), *galactic_from_equatorial(
            st["ra"][isx], st["dec"][isx])),
    ):
        rows[tag] = score_candidates(
            optical_only(fs, idx), z_primary=np.atleast_1d(zt),
            l_deg=np.atleast_1d(ll), b_deg=np.atleast_1d(bb),
            qso_model=qso, background_model=bkg, match=match,
            qso_prior=prior, background_density=bdens, z_grid=zgrid, min_bands=2,
        )

    print(f"\n{'':4s} {'target z':>9s} {'r':>6s} {'log BF':>8s} {'log R':>8s} "
          f"{'p_sameq':>9s} {'p_z|Q':>7s}")
    for tag in rows:
        print(f"-- {tag}s")
        for k, s in enumerate(rows[tag]):
            print(f"{k+1:4d} {s.z_primary:9.3f} {s.ref_mag:6.2f} "
                  f"{s.log_bayes_factor_qz_bkg:+8.2f} {s.log_r_per_unit_z:+8.2f} "
                  f"{s.p_sameq:9.2e} {s.p_zmatch_given_qso:7.3f}")

    make_figure(qso, bkg, fq, iq, zq[iq], fst, isx, rows, args)


def make_figure(qso, bkg, fq, iq, zq, fst, isx, rows, args):
    """Two rows of panels: the models, in the plane the decision is made in."""
    import matplotlib.pyplot as plt
    from qso_pcolor.plotting import SERIES, save_figure, use_paper_style

    use_paper_style()
    gx = np.linspace(-0.1, 2.0, 200)
    gy = np.linspace(-0.1, 3.0, 200)
    XX, YY = np.meshgrid(gx, gy, indexing="ij")
    pts = np.full((XX.size, 4), np.nan)
    pts[:, 0], pts[:, 1] = XX.ravel(), YY.ravel()
    obs = np.zeros((XX.size, 4), dtype=bool)
    obs[:, OPTICAL] = True

    fig, axes = plt.subplots(2, args.n, figsize=(2.6 * args.n, 5.6),
                             sharex=True, sharey=True)
    for row, (tag, fs, idx, zt) in enumerate(
        (("quasar", fq, iq, zq),
         ("PSF source", fst, isx, [s.z_primary for s in rows["PSF source"]]))
    ):
        for k in range(args.n):
            ax = axes[row, k]
            s = rows[tag][k]
            z0 = float(zt[k])

            lq_ = qso.log_p_colour_given_z(pts, None, np.array([z0]),
                                           observed=obs)[:, 0]
            lb_ = bkg.log_prob(pts, None, np.full(XX.size, s.ref_mag),
                               np.full(XX.size, 0.0), np.full(XX.size, 60.0),
                               observed=obs)
            for lp, key in ((lq_, "same_z"), (lb_, "background")):
                P = np.exp(lp - lp.max()).reshape(XX.shape)
                ax.contour(gx, gy, P.T, levels=[0.05, 0.3, 0.8],
                           colors=SERIES[key], linewidths=1.0)

            x0, y0 = fs.x[idx[k], 0], fs.x[idx[k], 1]
            sx = np.sqrt(fs.cov[idx[k], 0, 0]); sy = np.sqrt(fs.cov[idx[k], 1, 1])
            ax.errorbar(x0, y0, xerr=sx, yerr=sy, fmt="o", ms=5,
                        color="#2b2b28", mfc="white", mew=1.4, zorder=5, lw=1.2)
            ax.set_title(f"$z_0={z0:.2f}$,  $r={s.ref_mag:.1f}$", loc="left",
                         fontsize=8)
            ax.text(0.04, 0.96,
                    f"log BF {s.log_bayes_factor_qz_bkg:+.1f}\n"
                    f"$p_{{\\rm same}}$ {s.p_sameq:.1e}",
                    transform=ax.transAxes, va="top", fontsize=7,
                    color="#2b2b28",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                              boxstyle="round,pad=0.25"))
            ax.set_xlim(-0.1, 2.0); ax.set_ylim(-0.1, 3.0)
            if row == 1:
                ax.set_xlabel("$f_g/f_r$")
        axes[row, 0].set_ylabel(f"{tag}s\n$f_z/f_r$")
    # Legend in the bottom-right panel, the emptiest corner of the grid.
    axes[-1, -1].plot([], [], color=SERIES["same_z"], label="quasar model at $z_0$")
    axes[-1, -1].plot([], [], color=SERIES["background"], label="field model")
    axes[-1, -1].legend(loc="lower right", fontsize=7, framealpha=0.9,
                        frameon=True, edgecolor="none")
    fig.suptitle("DECaLS colours only: W1 and W2 marginalised out, not dropped",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    print(f"\nwrote {save_figure(fig, 'examples/optical_only_examples')}")


if __name__ == "__main__":
    main()
