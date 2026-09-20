#!/usr/bin/env python
"""Train the quasar colour-redshift model on the full spectroscopic sample.

This is Milestone M2. It supersedes the small illustrative fit inside
``make_method_figures.py``, which used a single 12-degree cone (about 80,000
quasars) because it had to run in minutes -- a figure-script convenience that
should never have stood in for the model.

What this uses instead
----------------------
**DESI DR1**, joined to Legacy Surveys DR9 photometry by ``targetid``, so the
photometry belongs to the object that was observed and no crossmatch radius is
involved. In ``0.4 < z < 3.6`` that is 1,243,822 quasars in the south
(``release`` 9010) and 319,190 in the north (9011).

**SDSS DR16Q**, positionally matched to ``decals_dr9.main`` so that its quasars
carry photometry in the *same* system. Of the 624,977 DR16Q quasars in range,
277,506 were re-observed by DESI; the remaining **347,471 were selected by
BOSS/eBOSS targeting rather than DESI's random forest**. That is the point of
including them. DESI selects quasars with a classifier built on these very
colours, so a DESI-only training set teaches us
:math:`p(\\mathbf{c}\\mid Q, z, \\text{DESI would target it})`. A second channel
with different selection is the only handle we have on that bias, and the
difference in held-out likelihood between the two is a measurement of it.

What it does not mix
--------------------
North and south are **different photometric systems** (BASS/MzLS versus DECam)
and are fitted separately, never pooled. Objects are de-duplicated by sky
position, not identifier: ``zcat_primary`` leaves thousands of repeated objects
under distinct ``targetid`` values, and DR16Q overlaps DESI heavily.

Selection consistency (2026-09-20)
----------------------------------
``maskbits = 0`` is applied to **both** channels. The SDSS match query always
required it; the DESI branch returned the column and never used it, so 5.4% of
the DESI training quasars sat in regions the background model and the candidate
selection exclude -- mostly WISE bright-star halos, i.e. contaminated in the
bands that carry most of the discrimination. The three samples the Bayes
factor compares are now built the same way.

K is fixed at ``--n-components`` where a slice has at least ``--select-k-below``
objects (measured: no held-out gain available there) and selected per slice
from ``--k-grid`` below that, where it matters and is cheap.

The holdout is READ from an existing model (``--holdout-from``) so that old
and new can be compared on the same reserved sky; it is drawn afresh only when
no such file exists. Output goes to a dated file, never over the shipped one.

Usage
-----
    python scripts/train_qso_model.py --system south
    python scripts/train_qso_model.py --system north --no-sdss
    python scripts/train_qso_model.py --system south --max-objects 20000   # plumbing check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

BANDS = ("g", "r", "z", "w1", "w2")
RELEASE = {"south": 9010, "north": 9011}

# DR16Q carries SDSS photometry; we need Legacy Surveys photometry for the same
# objects, so the quasar list is uploaded and joined positionally. 1 arcsec is
# tight enough to be unambiguous at these densities and loose enough for the
# astrometric differences between the two surveys.
SDSS_MATCH_QUERY = """
SELECT m.idx, m.zspec, x.*
FROM mytmptable AS m
CROSS JOIN LATERAL (
    SELECT c.ra, c.dec, c.release,
           c.flux_g, c.flux_r, c.flux_z, c.flux_w1, c.flux_w2,
           c.flux_ivar_g, c.flux_ivar_r, c.flux_ivar_z,
           c.flux_ivar_w1, c.flux_ivar_w2,
           c.mw_transmission_g, c.mw_transmission_r, c.mw_transmission_z,
           c.mw_transmission_w1, c.mw_transmission_w2,
           q3c_dist(c.ra, c.dec, m.ra, m.dec) * 3600 AS sep_arcsec
    FROM decals_dr9.main AS c
    WHERE q3c_radial_query(c.ra, c.dec, m.ra, m.dec, 1.0/3600)
      AND c.maskbits = 0 AND c.flux_ivar_r > 0
    ORDER BY q3c_dist(c.ra, c.dec, m.ra, m.dec)
    LIMIT 1
) AS x
"""
# Why this shape, and not the obvious ``JOIN ... ON q3c_join(...)``:
#
# ``decals_dr9.main`` has 2.0e9 rows. Written as a plain join with the quality
# cuts in the WHERE clause, the planner produces a sequential scan of that table
# with the q3c condition demoted to a *join filter* -- measured, not guessed --
# and the query cannot finish. Even without the cuts it drives the nested loop
# from ``main``'s index rather than from our positions.
#
# The lateral form fixes it by making each of our objects the outer row and the
# survey the inner lookup, so ``q3c_radial_query`` becomes an index condition on
# ``main_q3c_ang2ipix_idx``. Note the argument order: the *indexed* table's
# columns come first, the search centre second. Reversing them silently
# disables the index.
#
# ``ORDER BY ... LIMIT 1`` inside the lateral takes the nearest match per input
# without a global sort; an outer ``ORDER BY idx, sep`` would force the whole
# result to be materialised and sorted before the first row is returned.
#
# Measured: 81 s for 20,000 positions, so about 40 minutes for DR16Q.


def stack(r, prefix):
    return np.stack([np.asarray(r[f"{prefix}{b}"], float) for b in BANDS], axis=1)


def load_desi(cache: Path, zmin: float, zmax: float) -> dict:
    from qso_pcolor.data import fetch_desi_qso_training

    t0 = time.time()
    r = fetch_desi_qso_training(cache, zmin=zmin, zmax=zmax)
    print(f"  DESI DR1: {r['zspec'].size:,} quasars ({time.time() - t0:.0f} s)")
    return r


def load_sdss(cache: Path, zmin: float, zmax: float, refresh: bool = False) -> dict:
    """DR16Q quasars with Legacy Surveys photometry attached positionally."""
    import sqlutilpy as sqlutil

    from qso_pcolor.data import cached_query

    from qso_pcolor.data import _load_npz, _save_npz

    # Key this cache on what produced it, like every other cache here. Checking
    # only that the file exists meant changing --zmin/--zmax silently reused the
    # previous sample: the positions query was hash-keyed, but the match result
    # it fed was not, so the hash protected the wrong half of the operation.
    positions_q = (
        f"SELECT ra, dec, z AS zspec FROM sdssdr16qso.main "
        f"WHERE z > {zmin} AND z < {zmax} AND zwarning = 0"
    )
    digest = hashlib.sha1((positions_q + SDSS_MATCH_QUERY).encode()).hexdigest()[:10]
    cache = Path(cache)
    cache = cache.with_name(f"{cache.stem}_{digest}.npz")
    if cache.exists() and not refresh:
        out = _load_npz(cache)
        print(f"  SDSS DR16Q x LS DR9: {out['zspec'].size:,} (cached)")
        return out

    q = cached_query(positions_q, cache.with_name("dr16q_positions.npz"))
    idx = np.arange(q["ra"].size, dtype=np.int64)
    t0 = time.time()
    m = sqlutil.local_join(
        SDSS_MATCH_QUERY, "mytmptable",
        (idx, q["ra"], q["dec"], q["zspec"]), ("idx", "ra", "dec", "zspec"),
        asDict=True, intNullVal=-1,
    )
    # The lateral LIMIT 1 already returns at most one row per input object.
    print(f"  SDSS DR16Q x LS DR9: {m['zspec'].size:,} matched "
          f"of {idx.size:,} ({time.time() - t0:.0f} s)")
    _save_npz(cache, **m)
    return m


def deduplicate(ra, dec, zspec, radius_arcsec=1.0) -> np.ndarray:
    """Keep one row per physical object; see scripts/build_pair_validation.py."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    c = SkyCoord(ra * u.deg, dec * u.deg)
    i1, i2, _, _ = c.search_around_sky(c, radius_arcsec * u.arcsec)
    m = i1 < i2
    i1, i2 = i1[m], i2[m]
    dz = np.abs(zspec[i1] - zspec[i2]) / (1.0 + np.minimum(zspec[i1], zspec[i2]))
    keep = np.ones(ra.size, dtype=bool)
    for a, b in zip(i1[dz < 1e-3], i2[dz < 1e-3]):
        if keep[a]:
            keep[b] = False
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", choices=("south", "north"), default="south")
    ap.add_argument("--zmin", type=float, default=0.1)
    ap.add_argument("--zmax", type=float, default=4.4)
    ap.add_argument("--n-slices", type=int, default=43, help="43 x 0.1 over 0.1-4.4")
    ap.add_argument("--n-components", type=int, default=20,
                    help="K for well-populated slices")
    ap.add_argument("--select-k-below", type=int, default=10000,
                    help="select K per slice by held-out density when the slice "
                         "has fewer objects than this; 0 disables")
    ap.add_argument("--k-grid", type=int, nargs="*", default=[2, 4, 8, 12, 20])
    ap.add_argument("--no-maskbits-cut", action="store_true",
                    help="do NOT require maskbits = 0 on the DESI channel "
                         "(reproduces the pre-2026-09-20 selection)")
    ap.add_argument("--holdout-from", type=Path, default=Path("models/qso_south_full.json"),
                    help="model file whose holdout_blocks to reuse; drawn afresh "
                         "if the file or the field is missing")
    ap.add_argument("--out-name", type=str, default=None,
                    help="output file name; default qso_<system>_<date>.json. The "
                         "shipped model is never overwritten by this script")
    ap.add_argument("--no-sdss", action="store_true")
    ap.add_argument("--max-objects", type=int, default=None,
                    help="subsample for a quick run; omit to use everything")
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="fraction of nside=4 sky blocks reserved before fitting")
    ap.add_argument("--holdout-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--cache", type=Path, default=Path("data"))
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="  %(message)s")

    from qso_pcolor.background import galactic_healpix
    from qso_pcolor.data import galactic_from_equatorial
    from qso_pcolor.features import RelativeFluxTransform, deredden
    from qso_pcolor.qso_model import fit_sliced_model

    rel = RELEASE[args.system]
    print(f"Training the {args.system} model (release {rel})\n")

    print("loading")
    desi = load_desi(args.cache / "desi_qso_full.npz", args.zmin, args.zmax)
    parts = [(desi, "desi")]
    if not args.no_sdss:
        parts.append((load_sdss(args.cache / "dr16q_ls.npz", args.zmin, args.zmax),
                      "sdss"))

    ra, dec, z, flux, ivar, trans, channel = [], [], [], [], [], [], []
    n_masked_removed = 0
    for r, tag in parts:
        sel = np.asarray(r["release"], int) == rel
        if not sel.any():
            print(f"  {tag}: no objects in release {rel}")
            continue
        n_rel = int(sel.sum())
        # Same quality cut as the background model and the candidates. The
        # SDSS query already imposes it in SQL; applying it here too costs
        # nothing and makes the two channels demonstrably identical.
        if not args.no_maskbits_cut and "maskbits" in r:
            mb = np.asarray(r["maskbits"], int)
            sel &= mb == 0
            n_masked_removed += n_rel - int(sel.sum())
            print(f"  {tag}: {n_rel:,} in release {rel}, "
                  f"{n_rel - int(sel.sum()):,} in masked regions removed "
                  f"({100 * (n_rel - int(sel.sum())) / n_rel:.2f}%)")
        else:
            print(f"  {tag}: {n_rel:,} in release {rel}")
        ra.append(np.asarray(r["ra"])[sel])
        dec.append(np.asarray(r["dec"])[sel])
        z.append(np.asarray(r["zspec"])[sel])
        flux.append(stack(r, "flux_")[sel])
        ivar.append(stack(r, "flux_ivar_")[sel])
        trans.append(stack(r, "mw_transmission_")[sel])
        channel.append(np.full(int(sel.sum()), tag))

    ra, dec, z = (np.concatenate(x) for x in (ra, dec, z))
    flux, ivar, trans = (np.concatenate(x) for x in (flux, ivar, trans))
    channel = np.concatenate(channel)

    print("\nde-duplicating by sky position")
    keep = deduplicate(ra, dec, z)
    print(f"  removed {int((~keep).sum()):,} repeats "
          f"({100 * (~keep).mean():.2f}%), {int(keep.sum()):,} remain")
    ra, dec, z, channel = ra[keep], dec[keep], z[keep], channel[keep]
    flux, ivar, trans = flux[keep], ivar[keep], trans[keep]
    for tag in np.unique(channel):
        print(f"    {tag}: {int((channel == tag).sum()):,}")

    print("\nbuilding features")
    f, v = deredden(flux, ivar, trans)
    fs = RelativeFluxTransform(reference_band="r", min_ref_snr=5.0)(f, v, BANDS)
    ok = fs.usable(min_dims=3) & np.isfinite(fs.ref_mag)
    if args.max_objects and ok.sum() > args.max_objects:
        rng = np.random.default_rng(0)
        chosen = rng.choice(np.flatnonzero(ok), args.max_objects, replace=False)
        ok = np.zeros_like(ok)
        ok[chosen] = True
    print(f"  usable: {int(ok.sum()):,}   flags: {fs.flag_summary()}")

    l, b = galactic_from_equatorial(ra[ok], dec[ok])
    groups = galactic_healpix(l, b, 4)          # spatial CV blocks

    # Reserve whole sky blocks BEFORE fitting. An earlier version fitted every
    # usable object and then sampled those same objects to report a "held-out"
    # likelihood -- which measured training density, not generalisation, and
    # would have made the selection-channel comparison meaningless.
    blocks = np.unique(groups)
    held_blocks: set[int] | None = None
    holdout_source = "drawn"
    if args.holdout_from and Path(args.holdout_from).exists():
        prev = json.loads(Path(args.holdout_from).read_text()).get("meta", {})
        prev_blocks = prev.get("holdout_blocks") or prev.get("holdout_blocks_recovered")
        if prev_blocks and int(prev.get("holdout_nside", 4)) == 4:
            held_blocks = set(int(b) for b in prev_blocks)
            holdout_source = f"read from {args.holdout_from}"
    if held_blocks is None:
        # No recorded split to reuse: draw one. This is the ONLY place a draw
        # happens, and the result is written to the model so it never has to
        # be re-derived -- re-deriving it is what put training objects into a
        # "held-out" figure (AGENTS.md M6b).
        rng_h = np.random.default_rng(args.holdout_seed)
        n_hold = max(1, int(round(args.holdout_frac * blocks.size)))
        held_blocks = set(rng_h.choice(blocks, size=n_hold, replace=False).tolist())
    is_held = np.isin(groups, list(held_blocks))
    print(f"  spatial holdout ({holdout_source}): {len(held_blocks)} blocks, "
          f"{int(is_held.sum()):,} objects ({100 * is_held.mean():.1f}%) reserved; "
          f"{blocks.size} blocks populated")

    n_comp = args.n_components
    print(f"\nfitting {args.n_slices} slices, K = {n_comp} where n >= "
          f"{args.select_k_below:,}, selected from {args.k_grid} below")
    t0 = time.time()
    fit_idx = np.flatnonzero(ok)[~is_held]
    # finer blocks than the holdout for the K-selection folds: nside=8 cells
    # keep neighbours together without leaving a sparse slice with one fold
    l_fit, b_fit = galactic_from_equatorial(ra[ok][~is_held], dec[ok][~is_held])
    select_k = None
    if args.select_k_below > 0:
        select_k = {"grid": args.k_grid, "below_n": args.select_k_below,
                    "groups": galactic_healpix(l_fit, b_fit, 8), "n_folds": 2}
    ref_mag_fit = fs.ref_mag[fit_idx]
    model = fit_sliced_model(
        fs.x[fit_idx], fs.cov[fit_idx], z[fit_idx],
        z_edges=np.linspace(args.zmin, args.zmax, args.n_slices + 1),
        observed=fs.observed[fit_idx], n_components=n_comp, min_per_slice=500,
        overlap=0.5, system=f"ls_dr9_{args.system}_grzw", labels=fs.labels,
        seed=0, max_iter=300, tol=1e-6, regularization=1e-6,
        select_k=select_k,
        meta={
            "maskbits_cut_applied": not args.no_maskbits_cut,
            "n_masked_removed": int(n_masked_removed),
            "holdout_source": holdout_source,
            "k_rule": {"fixed_k": n_comp, "select_below_n": args.select_k_below,
                       "grid": args.k_grid},
            "validity": {
                "ref_mag_1_50_99": [float(x) for x in np.percentile(ref_mag_fit, [1, 50, 99])],
                "min_ref_snr": 5.0,
                "min_dims": 3,
            },
            "trained": time.strftime("%Y-%m-%d"),
            "release": rel,
            "n_train": int(fit_idx.size),
            "n_holdout": int(is_held.sum()),
            "holdout_frac": args.holdout_frac,
            "holdout_nside": 4,
            # Record the split itself, not just the recipe for it. An earlier
            # version stored only frac and nside, so every consumer had to
            # re-derive the draw -- and score_examples.py re-derived it from a
            # different sample, calling 52% of the training set "reserved".
            # np.random.choice is a function of the input array, so the same
            # seed over a different block list is a different split.
            "holdout_seed": args.holdout_seed,
            "holdout_blocks": sorted(int(x) for x in held_blocks),
            "n_blocks_total": int(blocks.size),
            "n_desi": int((channel[ok][~is_held] == "desi").sum()),
            "n_sdss": int((channel[ok][~is_held] == "sdss").sum()),
            "z_range": [args.zmin, args.zmax],
            "dedup_arcsec": 1.0,
        },
    )
    print(f"  done in {time.time() - t0:.0f} s; objects per slice "
          f"{model.n_train.astype(int).min():,}-{model.n_train.astype(int).max():,}")

    args.out.mkdir(exist_ok=True)
    name = args.out_name or f"qso_{args.system}_{time.strftime('%Y%m%d')}.json"
    path = args.out / name
    if path.resolve() == Path("models/qso_south_full.json").resolve():
        raise SystemExit("refusing to overwrite the shipped model; validate first, "
                         "then copy it into place deliberately")
    model.save(path)
    ks = [p_["k"] for p_ in model.meta["per_slice"]]
    print(f"\nwrote {path}: {len(ks)} slices, K from {min(ks)} to {max(ks)}, "
          f"{sum(1 for p_ in model.meta['per_slice'] if 'k_scores' in p_)} selected")

    # Held-out likelihood per selection channel: the measurement of how much the
    # DESI targeting bias matters. A large gap is a warning, not a curiosity.
    if not args.no_sdss and (channel[ok] == "sdss").any():
        print("\nheld-out mean log density by selection channel")
        print("  (evaluated on the reserved spatial blocks only)")
        for tag in ("desi", "sdss"):
            m = np.flatnonzero((channel[ok] == tag) & is_held)
            if m.size < 100:
                continue
            rng = np.random.default_rng(0)
            m = rng.choice(m, size=min(20000, m.size), replace=False)
            # log_p_colour_given_z returns (n_obj, n_z); each object must be
            # evaluated at its OWN redshift, so take the diagonal in chunks --
            # asking for all 20000 redshifts at once would allocate a
            # 20000 x 20000 array.
            vals = []
            for a in range(0, m.size, 500):
                sl = m[a:a + 500]
                lp = model.log_p_colour_given_z(
                    fs.x[ok][sl], fs.cov[ok][sl], z[ok][sl],
                    observed=fs.observed[ok][sl],
                )
                vals.append(np.diagonal(lp))
            v = np.concatenate(vals)
            print(f"  {tag}: median {float(np.nanmedian(v)):+.3f}  "
                  f"(N = {m.size:,} of {int((channel[ok] == tag).sum()):,})")


if __name__ == "__main__":
    main()
