#!/usr/bin/env python
"""Global versus local field model: where on the sky does the choice matter?

The scorer can use the shipped footprint-average background or a model fitted
in the candidate's own 0.5-degree cone (``fit_local_background``). The first
needs no database; the second does. This script turns the choice into a rule
by measuring, in ~40 cones stratified in Galactic latitude, what the local
model buys:

1. **Does it describe the field better?**  Each cone's sources are split 80/20;
   the local model is fitted on the 80% and both models are scored on the held
   20%.  The local-minus-global held-out log p(c | B) is the information the
   global model lacks about that patch.
2. **Does it change the quasar scores?**  DESI quasars in the cone (reserved
   blocks preferred) are scored at their own redshift under both backgrounds.
   The quasar model is identical, so the shift in log R is the background term
   alone.
3. **Does it change the false-positive rate?**  Fraction of held-out field
   sources with log BF > 0 against a primary at z0 = 1.8, under each model.

All three are reported against |b| and against the cone's source density, and
the output is the latitude below which the local model is worth the database
call.

    python scripts/compare_background_modes.py
    python scripts/compare_background_modes.py --n-cones 20 --seed 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

BANDS = ("g", "r", "z", "w1", "w2")
MAG_EDGES = np.array([17.0, 19.5, 20.5, 21.5, 22.5])
SYSTEM = "ls_dr9_south_grzw"
B_BINS = [(10, 20), (20, 30), (30, 45), (45, 60), (60, 90)]


def draw_cones(n_total, seed, per_bin):
    """Stratified in |b| and split between the centre- and anticentre-facing
    hemispheres, inside the DECaLS-south footprint where DESI/SDSS observed.

    Quotas that the footprint cannot fill (there is almost no sky below
    |b| = 20 deg) are left short rather than forced; the caller sees fewer
    cones, and the summary table shows how many landed in each bin.
    """
    from qso_pcolor.data import galactic_from_equatorial

    rng = np.random.default_rng(seed)
    n_cand = 20000
    ra = np.where(rng.random(n_cand) < 0.5, rng.uniform(130, 250, n_cand),
                  rng.uniform(335, 400, n_cand) % 360)
    dec = rng.uniform(-8, 25, n_cand)
    l, b = galactic_from_equatorial(ra, dec)          # one vectorised conversion
    quota = {(bi, hemi): per_bin for bi in range(len(B_BINS)) for hemi in (0, 1)}
    out = []
    for a, d, li, bi_ in zip(ra, dec, l, b):
        bi = next((i for i, (lo, hi) in enumerate(B_BINS) if lo <= abs(bi_) < hi), None)
        if bi is None:
            continue
        hemi = 0 if (li < 90 or li > 270) else 1      # 0: towards the centre, 1: anticentre
        if quota[(bi, hemi)] > 0:
            quota[(bi, hemi)] -= 1
            out.append({"ra": float(a), "dec": float(d), "l": float(li), "b": float(bi_),
                        "hemi": hemi, "b_bin": bi})
        if not any(quota.values()):
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-cones", type=int, default=40)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z0", type=float, default=1.8, help="primary redshift for the FPR test")
    ap.add_argument("--out", type=Path, default=Path("data/background_modes.json"))
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw the figure from --out without refitting anything")
    args = ap.parse_args()
    if args.figure_only:
        make_figure(json.loads(args.out.read_text())["cones"])
        return

    from qso_pcolor.background import (_RELEASES_FOR_SYSTEM, BackgroundColourModel,
                                       fit_background_model, galactic_healpix)
    from qso_pcolor.data import (drop_known_quasars, fetch_desi_qso_training,
                                 fetch_known_quasars, fetch_ls_background,
                                 galactic_from_equatorial)
    from qso_pcolor.features import RelativeFluxTransform, deredden
    from qso_pcolor.priors import BackgroundSurfaceDensity, GridQSOPrior
    from qso_pcolor.qso_model import RedshiftMatch, SlicedColourRedshiftModel
    from qso_pcolor.score import score_candidates

    qso = SlicedColourRedshiftModel.load("models/archive/original_legacy_south/qso_south_full.json")
    gbkg = BackgroundColourModel.load("models/archive/original_legacy_south/background_south_global.json")
    gdens = BackgroundSurfaceDensity.load("models/archive/original_legacy_south/background_density_south_global.json")
    prior = GridQSOPrior.load("models/archive/original_legacy_south/sigma_q_south.json")
    held = set(int(x) for x in qso.meta["holdout_blocks"])
    tr = RelativeFluxTransform(reference_band="r", min_ref_snr=5.0)
    match = RedshiftMatch(half_width_kms=2000.0)

    # DESI quasars for measurement 2, once
    r = fetch_desi_qso_training("data/desi_qso_full.npz", zmin=0.1, zmax=4.4)
    qsel = np.isin(np.asarray(r["release"], int), list(_RELEASES_FOR_SYSTEM[SYSTEM])) \
        & (np.asarray(r["maskbits"], int) == 0)
    qra, qdec, qz = (np.asarray(r[k], float)[qsel] for k in ("ra", "dec", "zspec"))
    st = lambda p: np.stack([np.asarray(r[f"{p}{b}"], float)[qsel] for b in BANDS], 1)  # noqa: E731
    qf, qv = deredden(st("flux_"), st("flux_ivar_"), st("mw_transmission_"))
    qfs = tr(qf, qv, BANDS)
    qok = qfs.usable(min_dims=3) & np.isfinite(qfs.ref_mag) & (qfs.ref_mag < MAG_EDGES[-1]) \
        & qso.in_support(qz)
    ql, qb = galactic_from_equatorial(qra, qdec)
    q_in_held = np.isin(galactic_healpix(ql, qb, 4), list(held))

    cones = draw_cones(args.n_cones, args.seed, per_bin=max(1, args.n_cones // 10))
    print(f"{len(cones)} candidate cones; fitting local models")
    rng = np.random.default_rng(args.seed)
    results = []
    t_all = time.time()
    for i, c in enumerate(cones):
        a, d = c["ra"], c["dec"]
        rr = fetch_ls_background(Path("data") / f"modes_cone_{i:02d}.npz",
                                 ra=a, dec=d, radius_deg=args.radius)
        n_raw = int(np.size(rr["ra"]))
        if n_raw < 20000 * (args.radius / 0.5) ** 2:
            print(f"  skip ({a:6.1f},{d:+5.1f}) |b|={abs(c['b']):4.1f}: {n_raw:,} sources, not fully covered")
            continue
        rel = set(int(x) for x in np.unique(np.asarray(rr["release"], int)))
        if not rel <= _RELEASES_FOR_SYSTEM[SYSTEM]:
            print(f"  skip ({a:6.1f},{d:+5.1f}): releases {sorted(rel)}")
            continue
        kq = fetch_known_quasars(a, d, args.radius, cache=Path("data") / f"modes_qso_{i:02d}.npz")
        if np.size(kq["ra"]) == 0:
            print(f"  skip ({a:6.1f},{d:+5.1f}): no known quasars to remove")
            continue
        unmasked = np.asarray(rr["maskbits"], int) == 0
        keep = unmasked & drop_known_quasars(rr["ra"], rr["dec"], kq["ra"], kq["dec"])
        f, v = deredden(
            np.stack([np.asarray(rr[f"flux_{b}"], float) for b in BANDS], 1)[keep],
            np.stack([np.asarray(rr[f"flux_ivar_{b}"], float) for b in BANDS], 1)[keep],
            np.stack([np.asarray(rr[f"mw_transmission_{b}"], float) for b in BANDS], 1)[keep])
        fs = tr(f, v, BANDS)
        ok = fs.usable(min_dims=3) & np.isfinite(fs.ref_mag) & (fs.ref_mag < MAG_EDGES[-1])
        idx = np.flatnonzero(ok)
        if idx.size < 3000:
            print(f"  skip ({a:6.1f},{d:+5.1f}): only {idx.size} usable")
            continue
        rng.shuffle(idx)
        n_fit = int(0.8 * idx.size)
        i_fit, i_val = idx[:n_fit], idx[n_fit:]
        l, b = galactic_from_equatorial(np.asarray(rr["ra"])[keep], np.asarray(rr["dec"])[keep])
        mask_frac = float(np.mean(~unmasked))
        area = np.pi * args.radius**2 * (1 - mask_frac) * 0.8   # the 80% fitted

        t0 = time.time()
        lbkg = fit_background_model(
            fs.x[i_fit], fs.cov[i_fit], fs.ref_mag[i_fit], l[i_fit], b[i_fit],
            mag_edges=MAG_EDGES, nside=1, nside_parent=1, observed=fs.observed[i_fit],
            n_components=8, min_per_cell=10**9, n0=500.0, system=SYSTEM,
            labels=fs.labels, seed=0, max_iter=300, regularization=1e-6)
        ldens = BackgroundSurfaceDensity.from_catalogue(
            fs.ref_mag[i_fit], l[i_fit], b[i_fit], mag_edges=MAG_EDGES,
            nside=1, nside_parent=1, total_area_deg2=area)

        # 1. held-out field density under each model
        lp_g = gbkg.log_prob(fs.x[i_val], fs.cov[i_val], fs.ref_mag[i_val], l[i_val], b[i_val],
                             observed=fs.observed[i_val])
        lp_l = lbkg.log_prob(fs.x[i_val], fs.cov[i_val], fs.ref_mag[i_val], l[i_val], b[i_val],
                             observed=fs.observed[i_val])
        lp_g = np.asarray(lp_g[0] if isinstance(lp_g, tuple) else lp_g)
        lp_l = np.asarray(lp_l[0] if isinstance(lp_l, tuple) else lp_l)
        good = np.isfinite(lp_g) & np.isfinite(lp_l)
        d_field = float(np.mean(lp_l[good] - lp_g[good]))

        # 3. false-positive rate at z0 under each model (held-out field sources)
        sub = i_val[good]
        fsub = fs.subset(sub)
        fpr = {}
        for tag, bk, dn in (("global", gbkg, gdens), ("local", lbkg, ldens)):
            rows = score_candidates(fsub, z_primary=np.full(sub.size, args.z0),
                                    l_deg=l[sub], b_deg=b[sub], qso_model=qso,
                                    background_model=bk, background_density=dn,
                                    qso_prior=prior, match=match, min_bands=3)
            bf = np.array([x.log_bayes_factor_qz_bkg for x in rows])
            fpr[tag] = float(np.mean(bf[np.isfinite(bf)] > 0))

        # 2. quasars in the cone under each model
        sep = np.degrees(np.arccos(np.clip(
            np.sin(np.radians(qdec)) * np.sin(np.radians(d))
            + np.cos(np.radians(qdec)) * np.cos(np.radians(d)) * np.cos(np.radians(qra - a)), -1, 1)))
        in_cone = (sep < args.radius) & qok
        pick = in_cone & q_in_held if (in_cone & q_in_held).sum() >= 30 else in_cone
        qi = np.flatnonzero(pick)
        dlogr = dlogbf = np.nan
        n_q = int(qi.size)
        if n_q >= 10:
            qsub = qfs.subset(qi)
            out = {}
            for tag, bk, dn in (("global", gbkg, gdens), ("local", lbkg, ldens)):
                rows = score_candidates(qsub, z_primary=qz[qi], l_deg=ql[qi], b_deg=qb[qi],
                                        qso_model=qso, background_model=bk,
                                        background_density=dn, qso_prior=prior,
                                        match=match, min_bands=3)
                out[tag] = (np.array([x.log_r_per_unit_z for x in rows]),
                            np.array([x.log_bayes_factor_qz_bkg for x in rows]))
            dlogr = float(np.nanmedian(out["local"][0] - out["global"][0]))
            dlogbf = float(np.nanmedian(out["local"][1] - out["global"][1]))

        res = {**c, "n_raw": n_raw, "n_usable": int(idx.size), "mask_fraction": mask_frac,
               "density_per_deg2": float(idx.size / (np.pi * args.radius**2 * (1 - mask_frac))),
               "d_heldout_logp_field": d_field, "fpr_global": fpr["global"], "fpr_local": fpr["local"],
               "n_quasars": n_q, "quasars_held_out": bool((in_cone & q_in_held).sum() >= 30),
               "d_logR_quasars": dlogr, "d_logBF_quasars": dlogbf}
        results.append(res)
        print(f"  {len(results):2d} ({a:6.1f},{d:+5.1f}) l={c['l']:5.1f} b={c['b']:+5.1f}  "
              f"n={idx.size:6,d}  dens={res['density_per_deg2']:7,.0f}/deg2  "
              f"field d(logp)={d_field:+.3f}  FPR {fpr['global']:.3f}->{fpr['local']:.3f}  "
              f"quasars n={n_q:3d} d(logR)={dlogr:+.2f}  [{time.time() - t0:.0f} s]", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"settings": {k: str(v) for k, v in vars(args).items()},
                                    "cones": results}, indent=1))
    print(f"\nwrote {args.out}  ({len(results)} cones, {time.time() - t_all:.0f} s)")

    # -- summary by |b| bin ---------------------------------------------------
    print(f"\n{'|b|':>8s} {'n':>3s} {'field d(logp)':>14s} {'FPR glob':>9s} {'FPR loc':>8s} {'d logR (q)':>11s}")
    for lo, hi in B_BINS:
        rs = [x for x in results if lo <= abs(x["b"]) < hi]
        if not rs:
            continue
        df = np.array([x["d_heldout_logp_field"] for x in rs])
        fg = np.array([x["fpr_global"] for x in rs]); fl = np.array([x["fpr_local"] for x in rs])
        dq = np.array([x["d_logR_quasars"] for x in rs]); dq = dq[np.isfinite(dq)]
        print(f"{lo:3d}-{hi:<3d} {len(rs):3d} {np.median(df):+9.3f} (max {df.max():+.2f}) "
              f"{np.median(fg):9.3f} {np.median(fl):8.3f} {np.median(dq) if dq.size else float('nan'):+11.2f}")
    make_figure(results)


def make_figure(results):
    import matplotlib.pyplot as plt
    from qso_pcolor.plotting import SERIES, save_figure, use_paper_style

    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    b = np.array([abs(x["b"]) for x in results]); hemi = np.array([x["hemi"] for x in results])
    col = np.where(hemi == 0, SERIES["field_q"], SERIES["same_z"])
    ax = axes[0]
    ax.scatter(b, [x["d_heldout_logp_field"] for x in results], c=col, s=28)
    ax.axhline(0, color="0.6", lw=1, ls="--")
    ax.set_xlabel("|b| (deg)"); ax.set_ylabel("held-out log p(c | B): local − global (nats)")
    ax.set_title("does the local model describe the field better?")
    ax = axes[1]
    ax.scatter(b, [x["fpr_global"] for x in results], c=col, s=28, marker="o", label="global")
    ax.scatter(b, [x["fpr_local"] for x in results], c=col, s=28, marker="x", label="local")
    ax.set_yscale("log"); ax.set_xlabel("|b| (deg)")
    ax.set_ylabel(f"fraction of field sources with log BF > 0")
    ax.set_title("false-positive rate against a primary at z0 = 1.8")
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[2]
    m = np.isfinite([x["d_logR_quasars"] for x in results])
    ax.scatter(b[m], np.array([x["d_logR_quasars"] for x in results])[m], c=col[m], s=28)
    ax.axhline(0, color="0.6", lw=1, ls="--")
    ax.set_xlabel("|b| (deg)"); ax.set_ylabel("median Δ log R for quasars: local − global")
    ax.set_title("does it move the quasar scores?")
    for ax in axes:
        ax.scatter([], [], c=SERIES["field_q"], label="towards the Galactic centre")
        ax.scatter([], [], c=SERIES["same_z"], label="anticentre")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = save_figure(fig, "validation/background_modes")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
