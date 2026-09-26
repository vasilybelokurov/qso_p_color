#!/usr/bin/env python
"""Score a handful of real objects and show the two models they are scored against.

Takes five spectroscopic quasars and five point-like field sources from separate
fields, scores each against a target redshift, and draws the quasar and background
densities. Figure 10 now displays the original flux-ratio densities in luptitude
colours, with the coordinate Jacobian included and the scores unchanged.

**Optical only.** Everything here uses the DECaLS bands alone -- g/r and z/r --
with W1 and W2 marginalised out exactly by the observed-dimension mask, not
dropped or imputed. That is deliberately the hard case: WISE carries most of the
quasar/star discrimination, so optical-only shows what the colours alone can do.

The quasars are drawn from the model's **reserved** spatial blocks, so none of
them was used in fitting.

    python scripts/score_examples.py

For a plot-only redraw from the committed inputs, without queries or fitting:

    python scripts/redraw_optical_examples.py
"""

from __future__ import annotations

import argparse
import hashlib
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


def fit_global_background(transform, bands, mag_edges, system, n_cones=8,
                          radius=0.5, seed=0):
    """One stellar model averaged over widely separated fields.

    The local model describes the candidate's own neighbourhood; this one
    describes the southern footprint as a whole. Comparing them shows how much
    the answer depends on where you look.
    """
    import numpy as np
    from pathlib import Path

    from qso_pcolor.background import fit_background_model
    from qso_pcolor.data import (cached_query, drop_known_quasars,
                                 fetch_known_quasars, galactic_from_equatorial)
    from qso_pcolor.features import deredden
    from qso_pcolor.priors import BackgroundSurfaceDensity
    from qso_pcolor.xd import fit_xd

    rng = np.random.default_rng(seed)
    centres = [(float(a), float(d)) for a, d in
               zip(rng.uniform(140, 350, n_cones), rng.uniform(-15, 15, n_cones))]
    X, V, O, M, L, B, area = [], [], [], [], [], [], 0.0
    for i, (a, d) in enumerate(centres):
        r = cached_query(
            STAR_QUERY % {"ra": a, "dec": d, "radius": radius,
                          "mlo": 17.0, "mhi": 22.5},
            Path("data") / f"global_bkg_{i:02d}.npz")
        if r["ra"].size < 100:
            continue
        q = fetch_known_quasars(a, d, radius,
                                cache=Path("data") / f"global_qso_{i:02d}.npz")
        keep = drop_known_quasars(r["ra"], r["dec"], q["ra"], q["dec"])
        f, v = deredden(stack(r, "flux_")[keep], stack(r, "flux_ivar_")[keep],
                        stack(r, "mw_transmission_")[keep])
        fs = transform(f, v, bands)
        ok = fs.usable(min_dims=3) & np.isfinite(fs.ref_mag) & (fs.ref_mag < 22.5)
        l, b = galactic_from_equatorial(r["ra"][keep][ok], r["dec"][keep][ok])
        X.append(fs.x[ok]); V.append(fs.cov[ok]); O.append(fs.observed[ok])
        M.append(fs.ref_mag[ok])
        L.append(l); B.append(b); area += np.pi * radius**2
    X = np.concatenate(X); V = np.concatenate(V); O = np.concatenate(O)
    M = np.concatenate(M); L = np.concatenate(L); B = np.concatenate(B)
    print(f"  global stellar model: {X.shape[0]:,} sources from {len(centres)} "
          f"fields across the south, {area:.1f} deg^2")
    model = fit_background_model(
        X, V, M, L, B, mag_edges=mag_edges, nside=1, nside_parent=1,
        observed=O,                              # bands the survey could not measure
        n_components=8, min_per_cell=10**9,      # global level only
        n0=500.0, system=system, labels=transform(
            np.ones((1, len(bands))), np.ones((1, len(bands))), bands).labels,
        seed=0, max_iter=300, regularization=1e-6)
    dens = BackgroundSurfaceDensity.from_catalogue(
        M, L, B, mag_edges=mag_edges, nside=1, nside_parent=1,
        total_area_deg2=area)
    return model, dens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ra", type=float, default=180.0)
    ap.add_argument("--dec", type=float, default=0.0)
    ap.add_argument("--radius", type=float, default=0.5, help="local background cone, deg")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--plateau", type=float, default=240.0,
                    help="true quasar density per deg^2 from the coverage plateau")
    ap.add_argument("--model", type=Path, default=Path("models/archive/original_legacy_south/qso_south_full.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refit", action="store_true",
                    help="rebuild the background models instead of "
                         "loading models/examples/*.json")
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
    # READ the holdout, never re-derive it. Re-deriving is what went wrong
    # before: this script rebuilt the block list from DESI alone, without
    # de-duplication or the quality cut, getting 78 candidate blocks where
    # training had 81. np.random.choice over a different array is a different
    # draw however equal the seed, so 7 of its 16 "reserved" blocks were
    # training blocks and 52% of the objects it called held out had been
    # fitted. Holdout membership is a property of the *block*, so the block
    # list is the only thing needed -- this sample need not match training's.
    held = qso.meta.get("holdout_blocks") or qso.meta.get("holdout_blocks_recovered")
    if not held:
        raise SystemExit(
            f"{args.model} records no holdout_blocks. Re-deriving the split "
            "here is exactly the bug this replaces; run\n"
            "    python scripts/recover_holdout_blocks.py --model "
            f"{args.model}\nto recover and record it, or retrain."
        )
    held = set(int(x) for x in held)
    blocks = galactic_healpix(lq, bq, int(qso.meta.get("holdout_nside", 4)))
    is_held = np.isin(blocks, list(held))

    okq = (fq.usable(min_dims=3) & np.isfinite(fq.ref_mag) & is_held
           & (fq.ref_mag > 19.0) & (fq.ref_mag < 21.5))
    iq = rng.choice(np.flatnonzero(okq), args.n, replace=False)
    src = "recorded" if "holdout_blocks" in qso.meta else "recovered"
    print(f"  {args.n} quasars drawn from the {len(held)} reserved blocks "
          f"({src} in the model file; never used in fitting)")

    # ---- five point sources, from five RANDOM places on the sky ----------
    # Not five from one cone: the field population varies across the sky, so
    # five objects from a single 0.5 deg patch share one stellar environment
    # and are not a random sample of anything.
    star_ra, star_dec, star_rows = [], [], []
    tries = 0
    while len(star_ra) < args.n and tries < 40:
        tries += 1
        a = float(rng.uniform(150.0, 340.0))
        d = float(rng.uniform(-12.0, 12.0))
        rs = cached_query(
            STAR_QUERY % {"ra": a, "dec": d, "radius": 0.15,
                          "mlo": 19.0, "mhi": 21.5},
            Path("data") / f"example_star_field_{len(star_ra):02d}.npz")
        if rs["ra"].size < 50:
            continue
        qs = fetch_known_quasars(a, d, 0.15,
                                 cache=Path("data") / f"example_starq_{len(star_ra):02d}.npz")
        nq = drop_known_quasars(rs["ra"], rs["dec"], qs["ra"], qs["dec"])
        fsr, vsr = deredden(stack(rs, "flux_"), stack(rs, "flux_ivar_"),
                            stack(rs, "mw_transmission_"))
        fse = tr(fsr, vsr, BANDS)
        good = np.flatnonzero(fse.usable(min_dims=3) & np.isfinite(fse.ref_mag) & nq)
        if good.size == 0:
            continue
        pick = int(rng.choice(good))
        star_ra.append(float(rs["ra"][pick])); star_dec.append(float(rs["dec"][pick]))
        star_rows.append({k: np.asarray(v)[pick : pick + 1] for k, v in rs.items()})
    st = {k: np.concatenate([r[k] for r in star_rows]) for k in star_rows[0]}
    fsr, vsr = deredden(stack(st, "flux_"), stack(st, "flux_ivar_"),
                        stack(st, "mw_transmission_"))
    fst = tr(fsr, vsr, BANDS)
    isx = np.arange(len(star_ra))
    print(f"  {args.n} PSF sources, one from each of {args.n} random fields "
          f"across the south")

    # ---- one local background PER OBJECT ---------------------------------
    # The candidates are scattered across the footprint, so a single cone's
    # field population is the wrong one for all but the object it was fitted
    # around: different stellar density, depth and reddening. Each object gets
    # a background fitted in its own neighbourhood, which is what the local
    # mode is for.
    print(f"\n  fitting one {args.radius} deg background cone per object")
    obj_ra = np.concatenate([np.asarray(r["ra"])[sel][iq], st["ra"][isx]])
    obj_dec = np.concatenate([np.asarray(r["dec"])[sel][iq], st["dec"][isx]])
    backgrounds = []
    mdir = Path("models") / "examples"
    mdir.mkdir(parents=True, exist_ok=True)
    for j, (ora, odec) in enumerate(zip(obj_ra, obj_dec)):
        # Key the fitted model on WHAT was fitted, not on the loop index. Keyed
        # on j, changing --seed draws different objects and silently reloads the
        # previous run's cones -- scoring each candidate against someone else's
        # sky. This mirrors qso_pcolor.data.cached_query, which hashes the query
        # text for the same reason.
        tag = hashlib.sha1(repr((
            round(float(ora), 6), round(float(odec), 6), float(args.radius),
            qso.system, MAG_EDGES.tolist(), 8, 22.5,
        )).encode()).hexdigest()[:10]
        mp, dp = mdir / f"local_{tag}.json", mdir / f"localdens_{tag}.json"
        if mp.exists() and dp.exists() and not args.refit:
            from qso_pcolor.background import BackgroundColourModel
            from qso_pcolor.priors import BackgroundSurfaceDensity
            bkg_j = BackgroundColourModel.load(mp)
            bd_j = BackgroundSurfaceDensity.load(dp)
            backgrounds.append((bkg_j, bd_j, bkg_j.meta))
            print(f"    {j+1:2d}  ({ora:7.3f},{odec:+7.3f})  cached")
            continue
        bkg_j, bd_j, info_j = fit_local_background(
            float(ora), float(odec), args.radius, transform=tr, bands=BANDS,
            mag_edges=MAG_EDGES, system=qso.system, n_components=8,
            max_ref_mag=22.5,
            cache=Path("data") / f"example_bkg_{tag}.npz",
            seed=0, max_iter=300, regularization=1e-6,
        )
        bkg_j.save(mp); bd_j.save(dp)
        backgrounds.append((bkg_j, bd_j, info_j))
        print(f"    {j+1:2d}  ({ora:7.3f},{odec:+7.3f})  "
              f"{info_j['n_fitted']:6,d} sources, "
              f"{info_j['n_known_quasars_removed']:3d} quasars out, "
              f"mask {info_j['mask_fraction']:.3f}")

    gmp, gdp = Path("models/examples/global.json"), Path("models/examples/globaldens.json")
    if gmp.exists() and gdp.exists() and not args.refit:
        from qso_pcolor.background import BackgroundColourModel
        from qso_pcolor.priors import BackgroundSurfaceDensity
        gbkg, gdens = BackgroundColourModel.load(gmp), BackgroundSurfaceDensity.load(gdp)
        print("  global stellar model: cached")
    else:
        gbkg, gdens = fit_global_background(tr, BANDS, MAG_EDGES, qso.system)
        gmp.parent.mkdir(parents=True, exist_ok=True)
        gbkg.save(gmp); gdens.save(gdp)

    prior = build_sigma_q(0.4, 3.6, args.plateau)

    # ---- score, OPTICAL ONLY ---------------------------------------------
    def optical_only(fs, idx):
        obs = np.zeros((len(idx), 4), dtype=bool)
        obs[:, OPTICAL] = fs.observed[idx][:, OPTICAL]
        return FeatureSet(fs.x[idx], fs.cov[idx], obs, fs.ref_flux[idx],
                          fs.ref_mag[idx], fs.ref_snr[idx], fs.labels, {})

    match = RedshiftMatch(half_width_kms=2000.0)
    zgrid = np.linspace(0.46, 3.54, 400)
    lsx, bsx = galactic_from_equatorial(st["ra"][isx], st["dec"][isx])
    z_for_psf = rng.choice(zq[iq], args.n)
    rows = {}
    j = 0
    for tag, fs, idx, zt, ll, bb in (
        ("quasar", fq, iq, zq[iq], lq[iq], bq[iq]),
        ("PSF source", fst, isx, z_for_psf, lsx, bsx),
    ):
        out = []
        for k in range(len(idx)):
            bkg_j, bd_j, _ = backgrounds[j]; j += 1
            out += score_candidates(
                optical_only(fs, idx[k : k + 1]),
                z_primary=np.atleast_1d(zt[k]),
                l_deg=np.atleast_1d(ll[k]), b_deg=np.atleast_1d(bb[k]),
                qso_model=qso, background_model=bkg_j, match=match,
                qso_prior=prior, background_density=bd_j,
                z_grid=zgrid, min_bands=2,
            )
        rows[tag] = out

    print(f"\n{'':4s} {'target z':>9s} {'r':>6s} {'log BF':>8s} {'log R':>8s} "
          f"{'p_sameq':>9s} {'p_z|Q':>7s}")
    for tag in rows:
        print(f"-- {tag}s")
        for k, s in enumerate(rows[tag]):
            print(f"{k+1:4d} {s.z_primary:9.3f} {s.ref_mag:6.2f} "
                  f"{s.log_bayes_factor_qz_bkg:+8.2f} {s.log_r_per_unit_z:+8.2f} "
                  f"{s.p_sameq:9.2e} {s.p_zmatch_given_qso:7.3f}")

    gal = (np.concatenate([lq[iq], lsx]), np.concatenate([bq[iq], bsx]))
    make_figure(qso, backgrounds, gbkg, fq, iq, zq[iq], fst, isx, rows, gal, args)


def make_figure(qso, backgrounds, gbkg, fq, iq, zq, fst, isx, rows, gal, args):
    """Cache the original inputs and render them on luptitude-colour axes."""
    import json
    from redraw_optical_examples import export_sample, render
    # Use the same finite JSON convention as the seven-survey examples.
    from make_multisurvey_examples import write_json

    cfg = json.loads(Path("configs/optical_examples_plot.json").read_text())
    sample = export_sample(qso, backgrounds, gbkg, fq, iq, zq, fst, isx, rows, gal, args)
    write_json(Path(cfg["sample"]), sample)
    print(f"\nwrote {render(sample, cfg)}")


def make_flux_ratio_figure(qso, backgrounds, gbkg, fq, iq, zq, fst, isx, rows, gal, args):
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

    # Limits from the data: fixed limits silently dropped points off-scale.
    allx = np.concatenate([fq.x[iq, 0], fst.x[isx, 0]])
    ally = np.concatenate([fq.x[iq, 1], fst.x[isx, 1]])
    xlo, xhi = min(-0.1, allx.min() - 0.2), max(2.0, allx.max() + 0.2)
    ylo, yhi = min(-0.1, ally.min() - 0.2), max(3.0, ally.max() + 0.3)
    gx = np.linspace(xlo, xhi, 200)
    gy = np.linspace(ylo, yhi, 200)
    XX, YY = np.meshgrid(gx, gy, indexing="ij")
    pts = np.full((XX.size, 4), np.nan)
    pts[:, 0], pts[:, 1] = XX.ravel(), YY.ravel()
    obs = np.zeros((XX.size, 4), dtype=bool)
    obs[:, OPTICAL] = True

    fig, axes = plt.subplots(2, args.n, figsize=(2.8 * args.n, 5.8),
                             sharex=True, sharey=True)
    for row, (tag, fs, idx, zt) in enumerate(
        (("quasar", fq, iq, zq),
         ("PSF source", fst, isx, [s.z_primary for s in rows["PSF source"]]))
    ):
        for k in range(args.n):
            ax = axes[row, k]
            s = rows[tag][k]
            z0 = float(zt[k])
            lbkg = backgrounds[row * args.n + k][0]

            lq_ = qso.log_p_colour_given_z(pts, None, np.array([z0]),
                                           observed=obs)[:, 0]
            lg_ = gbkg.log_prob(pts, None, np.full(XX.size, s.ref_mag),
                                np.zeros(XX.size), np.full(XX.size, 60.0),
                                observed=obs)
            ll_ = lbkg.log_prob(pts, None, np.full(XX.size, s.ref_mag),
                                np.zeros(XX.size), np.full(XX.size, 60.0),
                                observed=obs)
            for lp, key, ls in ((lq_, "same_z", "-"),
                                ("global", "field_q", "--"),
                                (ll_, "background", "-")):
                arr = lg_ if isinstance(lp, str) else lp
                P = np.exp(arr - arr.max()).reshape(XX.shape)
                ax.contour(gx, gy, P.T, levels=[0.05, 0.3, 0.8],
                           colors=SERIES[key], linewidths=1.0, linestyles=ls)

            x0, y0 = fs.x[idx[k], 0], fs.x[idx[k], 1]
            sx = np.sqrt(fs.cov[idx[k], 0, 0]); sy = np.sqrt(fs.cov[idx[k], 1, 1])
            ax.errorbar(x0, y0, xerr=sx, yerr=sy, fmt="o", ms=5,
                        color="#2b2b28", mfc="white", mew=1.4, zorder=5, lw=1.2)
            gl, gb = gal[0][row * args.n + k], gal[1][row * args.n + k]
            ax.set_title(f"$z_0={z0:.2f}$,  $r={s.ref_mag:.1f}$\n"
                         f"$\\ell={gl:.1f}^\\circ$, $b={gb:+.1f}^\\circ$",
                         loc="left", fontsize=8)
            ax.text(0.96, 0.04,
                    f"log BF {s.log_bayes_factor_qz_bkg:+.1f}\n"
                    f"$p_{{\\rm same}}$ {s.p_sameq:.1e}",
                    transform=ax.transAxes, va="bottom", ha="right", fontsize=7.5,
                    color="#2b2b28",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                              boxstyle="round,pad=0.25"))
            ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            if row == 1:
                ax.set_xlabel("$f_g/f_r$")
        axes[row, 0].set_ylabel(f"{tag}s\n$f_z/f_r$")

    h = [plt.Line2D([], [], color=SERIES["same_z"]),
         plt.Line2D([], [], color=SERIES["field_q"], ls="--"),
         plt.Line2D([], [], color=SERIES["background"])]
    fig.legend(h, ["quasar model at $z_0$", "stellar model, global",
                   "stellar model, local to this object"],
               loc="upper right", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.995, 1.0))
    fig.suptitle("DECaLS colours only: W1 and W2 marginalised out, not dropped",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    print(f"\nwrote {save_figure(fig, 'examples/optical_only_examples_flux_ratio')}")


if __name__ == "__main__":
    main()
