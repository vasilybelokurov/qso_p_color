#!/usr/bin/env python
"""Widen a trained model's redshift range by appending slices to its ends.

Why append rather than retrain. A ``SlicedColourRedshiftModel`` is a list of
independent per-slice mixtures, so slices can be added at either end without
touching the ones already there. Nothing that has been validated moves, the run
takes tens of minutes rather than hours, and -- now that the model records its
spatial holdout -- the new slices can use the *identical* split.

Why it is worth doing. The trained range 0.4 < z < 3.6 was a default, not a
data limit: 3.3% of DESI DR1 quasars sit below it and 1.1% above. Outside the
model's support ``log_p_colour_given_z`` clamps its interpolation index, so it
replays the edge slice as though it were measured. For the README candidate
that fiction carries 7.1% of the no-prior redshift normalisation. Extending
replaces extrapolation with fitted slices; where the data run out the fit falls
back to a single Gaussian, which is crude but honest.

**Homogeneity is the constraint.** The new slices must differ from the old ones
in redshift and nothing else -- same transform, same S/N floor, same release,
same quality cuts, same seed and EM settings; K is selected for sparse slices. Applying (say) the
``maskbits`` cut only to the extension of an uncut model would put a discontinuity in sample selection at z = 0.4
and z = 3.6, exactly where it would later be read as a feature of quasar
colour. The cut is inherited from the saved model metadata, including for the
current mask-clean model. Selection changes belong to a full retrain.

    python scripts/extend_qso_model_redshift.py --dry-run
    python scripts/extend_qso_model_redshift.py --zmin 0.1 --zmax 4.4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

BANDS = ("g", "r", "z", "w1", "w2")


def load_edge_sample(cache_dir: Path, zlo: float, zhi: float, release: int,
                     use_sdss: bool, *, maskbits_cut: bool) -> dict:
    """DESI (+ SDSS) quasars in a redshift interval, cut exactly like training.

    The caches are keyed on the query text, so asking for a new interval fetches
    a new, small file rather than re-pulling the 1.57e6 rows already on disk.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_qso_model import deduplicate, load_desi, load_sdss, stack

    tag = f"{zlo:g}_{zhi:g}"
    parts = [(load_desi(cache_dir / f"desi_qso_z{tag}.npz", zlo, zhi), "desi")]
    if use_sdss:
        parts.append((load_sdss(cache_dir / f"dr16q_ls_z{tag}.npz", zlo, zhi), "sdss"))

    ra, dec, z, flux, ivar, trans, channel = [], [], [], [], [], [], []
    for r, name in parts:
        sel = np.asarray(r["release"], int) == release
        # SDSS is mask-clean in its SQL query; DESI exposes maskbits.
        if maskbits_cut and name == "desi":
            sel &= np.asarray(r["maskbits"], int) == 0
        if not sel.any():
            continue
        ra.append(np.asarray(r["ra"])[sel])
        dec.append(np.asarray(r["dec"])[sel])
        z.append(np.asarray(r["zspec"])[sel])
        flux.append(stack(r, "flux_")[sel])
        ivar.append(stack(r, "flux_ivar_")[sel])
        trans.append(stack(r, "mw_transmission_")[sel])
        channel.append(np.full(int(sel.sum()), name))
    if not ra:
        raise SystemExit(f"no objects in release {release} for {zlo} < z < {zhi}")

    ra, dec, z = (np.concatenate(x) for x in (ra, dec, z))
    flux, ivar, trans = (np.concatenate(x) for x in (flux, ivar, trans))
    channel = np.concatenate(channel)

    keep = deduplicate(ra, dec, z)
    return {
        "ra": ra[keep], "dec": dec[keep], "z": z[keep], "channel": channel[keep],
        "flux": flux[keep], "ivar": ivar[keep], "trans": trans[keep],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("models/archive/original_legacy_south/qso_south_full.json"))
    ap.add_argument("--out", type=Path, default=None,
                    help="default: overwrite --model in place")
    ap.add_argument("--zmin", type=float, default=0.1, help="new low edge")
    ap.add_argument("--zmax", type=float, default=4.4, help="new high edge")
    ap.add_argument("--cache", type=Path, default=Path("data"))
    ap.add_argument("--min-per-slice", type=int, default=500)
    ap.add_argument("--k-grid", type=int, nargs="*", default=[1, 2, 4, 8, 12, 20],
                    help="K candidates for the sparse new slices")
    ap.add_argument("--fixed-k", action="store_true",
                    help="use the model's K everywhere, without selection "
                         "(only sensible if every new slice is as well "
                         "populated as the existing ones)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the slice plan and object counts, fit nothing")
    args = ap.parse_args()

    from qso_pcolor.background import galactic_healpix
    from qso_pcolor.data import galactic_from_equatorial
    from qso_pcolor.features import RelativeFluxTransform, deredden
    from qso_pcolor.qso_model import SlicedColourRedshiftModel
    from qso_pcolor.xd import fit_xd, select_n_components

    model = SlicedColourRedshiftModel.load(args.model)
    meta = dict(model.meta)
    zc = model.z_centres
    width = float(np.round(np.diff(zc).mean(), 6))
    lo_c, hi_c = float(zc[0]), float(zc[-1])
    print(f"{args.model}: {zc.size} slices, centres {lo_c:.2f}-{hi_c:.2f}, "
          f"width {width:g}, K={model.mixtures[0].n_components}")

    held = meta.get("holdout_blocks") or meta.get("holdout_blocks_recovered")
    if not held:
        raise SystemExit(
            "the model records no holdout blocks; run "
            "scripts/recover_holdout_blocks.py first. Fitting new slices on a "
            "different split would make the extended model's holdout undefined."
        )
    held = set(int(x) for x in held)
    nside = int(meta.get("holdout_nside", 4))
    release = int(meta["release"])
    n_comp = int(model.mixtures[0].n_components)
    overlap = float(meta.get("overlap", 0.5))

    # New slice centres on the SAME grid as the existing ones, so the
    # interpolation in log_p_colour_given_z stays uniform across the join.
    n_lo = int(np.floor((lo_c - args.zmin) / width + 1e-9))
    n_hi = int(np.floor((args.zmax - hi_c) / width + 1e-9))
    new_lo = np.round(lo_c - width * np.arange(n_lo, 0, -1), 6)
    new_hi = np.round(hi_c + width * np.arange(1, n_hi + 1), 6)
    print(f"appending {n_lo} slices below ({new_lo[0]:.2f}-{new_lo[-1]:.2f}) and "
          f"{n_hi} above ({new_hi[0]:.2f}-{new_hi[-1]:.2f})" if n_lo and n_hi else
          f"appending {n_lo} below, {n_hi} above")

    tr = RelativeFluxTransform(reference_band="r", min_ref_snr=5.0)
    added: list[tuple] = []

    for label, centres, (qlo, qhi) in (
        ("low", new_lo, (args.zmin - width, lo_c)),
        ("high", new_hi, (hi_c, args.zmax + width)),
    ):
        if centres.size == 0:
            continue
        print(f"\n--- {label} end: {centres.size} slices, querying "
              f"{qlo:g} < z < {qhi:g}")
        d = load_edge_sample(args.cache, qlo, qhi, release,
                             int(meta.get("n_sdss", 0)) > 0,
                             maskbits_cut=bool(meta.get("maskbits_cut_applied", False)))
        f, v = deredden(d["flux"], d["ivar"], d["trans"])
        fs = tr(f, v, BANDS)
        ok = fs.usable(min_dims=3) & np.isfinite(fs.ref_mag)
        l, b = galactic_from_equatorial(d["ra"][ok], d["dec"][ok])
        is_held = np.isin(galactic_healpix(l, b, nside), list(held))
        fit = np.flatnonzero(ok)[~is_held]
        print(f"  {int(ok.sum()):,} usable, {int(is_held.sum()):,} in reserved "
              f"blocks, {fit.size:,} available to fit")

        edges = np.round(np.concatenate([centres - width / 2,
                                         [centres[-1] + width / 2]]), 6)
        counts = [int(((d["z"][fit] >= e0 - overlap * width) &
                       (d["z"][fit] < e1 + overlap * width)).sum())
                  for e0, e1 in zip(edges[:-1], edges[1:])]
        for c, n in zip(centres, counts):
            print(f"    z = {c:.2f}   n = {n:7,d}   "
                  f"K = {n_comp if n >= args.min_per_slice else 1}")
        if args.dry_run:
            continue
        if min(counts) < 2:
            raise SystemExit(
                f"slice with {min(counts)} objects; narrow --zmin/--zmax")

        # The existing slices carry ~35,000 objects for K = 20, about 117
        # objects per free parameter. The sparsest new slice has ~900, which is
        # ~3 per parameter: K = 20 there would fit noise. ``min_per_slice`` is a
        # don't-crash fallback, not a quality criterion, so choose K per slice
        # from held-out density instead of asserting one. Folds are blocked on
        # nside=8 cells so spatially adjacent objects never straddle the split.
        t0 = time.time()
        mixtures, n_train, chosen = [], [], []
        lf, bf = galactic_from_equatorial(d["ra"][fit], d["dec"][fit])
        cv_groups = galactic_healpix(lf, bf, 8)
        for e0, e1, centre in zip(edges[:-1], edges[1:], centres):
            sel = ((d["z"][fit] >= e0 - overlap * width)
                   & (d["z"][fit] < e1 + overlap * width))
            idx = fit[sel]
            grid = [k for k in args.k_grid if k <= n_comp]
            if args.fixed_k or int(sel.sum()) >= 20000:
                k, scores = n_comp, {}
            else:
                k, scores = select_n_components(
                    fs.x[idx], fs.cov[idx], grid, observed=fs.observed[idx],
                    groups=cv_groups[sel], n_folds=4, seed=0,
                    max_iter=200, tol=1e-6, regularization=1e-6,
                )
            res = fit_xd(fs.x[idx], fs.cov[idx], n_components=k,
                         observed=fs.observed[idx], labels=fs.labels,
                         seed=0, max_iter=300, tol=1e-6, regularization=1e-6)
            mixtures.append(res.mixture)
            n_train.append(int(sel.sum()))
            chosen.append({"z": float(centre), "n": int(sel.sum()), "k": int(k),
                           "k_scores": {str(a): float(b) for a, b in scores.items()}})
            best = (f"  {max(scores, key=scores.get)}" if scores else " (fixed)")
            print(f"    z = {centre:.2f}  n = {int(sel.sum()):7,d}  K = {k:2d}"
                  f"{best}  {'converged' if res.converged else 'hit max_iter'}")
        print(f"  fitted in {time.time() - t0:.0f} s")
        added.append((centres, mixtures, n_train, label, chosen))

    if args.dry_run:
        print("\n--dry-run: nothing fitted, nothing written")
        return
    if not added:
        print("nothing to append")
        return

    lo_new = next((a for a in added if a[3] == "low"), None)
    hi_new = next((a for a in added if a[3] == "high"), None)
    centres = np.concatenate(
        [x for x in (lo_new[0] if lo_new else None, zc,
                     hi_new[0] if hi_new else None) if x is not None])
    mixtures = ((lo_new[1] if lo_new else []) + list(model.mixtures)
                + (hi_new[1] if hi_new else []))
    n_train = ((lo_new[2] if lo_new else []) + model.n_train.tolist()
               + (hi_new[2] if hi_new else []))

    meta["extended"] = {
        "date": time.strftime("%Y-%m-%d"),
        "by": "scripts/extend_qso_model_redshift.py",
        "previous_support": [lo_c, hi_c],
        "slices_added_low": int(lo_new[0].size) if lo_new else 0,
        "slices_added_high": int(hi_new[0].size) if hi_new else 0,
        "note": "appended slices only; the original mixtures are unchanged and "
                "the new ones use the same cuts, K, holdout and EM settings, so "
                "the model stays homogeneous in redshift",
    }
    meta["z_range"] = [float(centres[0] - width / 2), float(centres[-1] + width / 2)]
    meta["per_slice"] = [
        {"z": float(c), "n": int(n), "k": int(m.n_components)}
        for c, n, m in zip(centres, n_train, mixtures)
    ]
    meta["extended"]["k_selection_per_new_slice"] = (
        (lo_new[4] if lo_new else []) + (hi_new[4] if hi_new else []))
    out = SlicedColourRedshiftModel(centres, mixtures, np.asarray(n_train, float),
                                    model.system, model.labels, meta)
    dest = args.out or args.model
    out.save(dest)
    print(f"\nwrote {dest}: {centres.size} slices, support "
          f"{centres[0]:.2f}-{centres[-1]:.2f}")


if __name__ == "__main__":
    main()
