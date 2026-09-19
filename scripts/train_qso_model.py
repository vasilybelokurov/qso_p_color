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

Usage
-----
    python scripts/train_qso_model.py --system south
    python scripts/train_qso_model.py --system north --no-sdss
    python scripts/train_qso_model.py --system south --select-k    # slower
"""

from __future__ import annotations

import argparse
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
SELECT m.idx, m.zspec,
       c.ra, c.dec, c.release,
       c.flux_g, c.flux_r, c.flux_z, c.flux_w1, c.flux_w2,
       c.flux_ivar_g, c.flux_ivar_r, c.flux_ivar_z,
       c.flux_ivar_w1, c.flux_ivar_w2,
       c.mw_transmission_g, c.mw_transmission_r, c.mw_transmission_z,
       c.mw_transmission_w1, c.mw_transmission_w2,
       q3c_dist(m.ra, m.dec, c.ra, c.dec) * 3600 AS sep_arcsec
FROM mytmptable AS m
JOIN decals_dr9.main AS c
  ON q3c_join(m.ra, m.dec, c.ra, c.dec, 1.0/3600)
WHERE c.maskbits = 0 AND c.flux_ivar_r > 0
ORDER BY m.idx, sep_arcsec
"""


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

    cache = Path(cache)
    if cache.exists() and not refresh:
        with np.load(cache, allow_pickle=True) as z:
            out = {k: z[k] for k in z.files}
        print(f"  SDSS DR16Q x LS DR9: {out['zspec'].size:,} (cached)")
        return out

    q = cached_query(
        f"SELECT ra, dec, z AS zspec FROM sdssdr16qso.main "
        f"WHERE z > {zmin} AND z < {zmax} AND zwarning = 0",
        cache.with_name("dr16q_positions.npz"),
    )
    idx = np.arange(q["ra"].size, dtype=np.int64)
    t0 = time.time()
    m = sqlutil.local_join(
        SDSS_MATCH_QUERY, "mytmptable",
        (idx, q["ra"], q["dec"], q["zspec"]), ("idx", "ra", "dec", "zspec"),
        asDict=True, intNullVal=-1,
    )
    # local_join returns every match; keep the nearest per input object.
    first = np.concatenate([[True], np.diff(m["idx"]) != 0])
    m = {k: v[first] for k, v in m.items()}
    print(f"  SDSS DR16Q x LS DR9: {m['zspec'].size:,} matched "
          f"of {idx.size:,} ({time.time() - t0:.0f} s)")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **m)
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
    ap.add_argument("--zmin", type=float, default=0.4)
    ap.add_argument("--zmax", type=float, default=3.6)
    ap.add_argument("--n-slices", type=int, default=32)
    ap.add_argument("--n-components", type=int, default=12)
    ap.add_argument("--select-k", action="store_true",
                    help="choose K by spatially blocked held-out density (slow)")
    ap.add_argument("--k-grid", type=int, nargs="*", default=[6, 12, 20])
    ap.add_argument("--no-sdss", action="store_true")
    ap.add_argument("--max-objects", type=int, default=None,
                    help="subsample for a quick run; omit to use everything")
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--cache", type=Path, default=Path("data"))
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="  %(message)s")

    from qso_pcolor.background import galactic_healpix
    from qso_pcolor.data import galactic_from_equatorial
    from qso_pcolor.features import RelativeFluxTransform, deredden
    from qso_pcolor.qso_model import fit_sliced_model
    from qso_pcolor.xd import select_n_components

    rel = RELEASE[args.system]
    print(f"Training the {args.system} model (release {rel})\n")

    print("loading")
    desi = load_desi(args.cache / "desi_qso_full.npz", args.zmin, args.zmax)
    parts = [(desi, "desi")]
    if not args.no_sdss:
        parts.append((load_sdss(args.cache / "dr16q_ls.npz", args.zmin, args.zmax),
                      "sdss"))

    ra, dec, z, flux, ivar, trans, channel = [], [], [], [], [], [], []
    for r, tag in parts:
        sel = np.asarray(r["release"], int) == rel
        if not sel.any():
            print(f"  {tag}: no objects in release {rel}")
            continue
        print(f"  {tag}: {int(sel.sum()):,} in release {rel}")
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

    n_comp = args.n_components
    scores = {}
    if args.select_k:
        print("\nchoosing K by spatially blocked held-out density")
        sub = np.flatnonzero(np.abs(z[ok] - 1.8) < 0.1)[:40000]
        n_comp, scores = select_n_components(
            fs.x[ok][sub], fs.cov[ok][sub], args.k_grid,
            observed=fs.observed[ok][sub], groups=groups[sub],
            n_folds=4, seed=0, max_iter=200, regularization=1e-6,
        )
        for k, sc in sorted(scores.items()):
            print(f"  K = {k:3d}   held-out mean log density {sc:+.4f}")
        print(f"  -> K = {n_comp}")

    print(f"\nfitting {args.n_slices} slices, K = {n_comp}")
    t0 = time.time()
    model = fit_sliced_model(
        fs.x[ok], fs.cov[ok], z[ok],
        z_edges=np.linspace(args.zmin, args.zmax, args.n_slices + 1),
        observed=fs.observed[ok], n_components=n_comp, min_per_slice=500,
        overlap=0.5, system=f"ls_dr9_{args.system}_grzw", labels=fs.labels,
        seed=0, max_iter=300, tol=1e-6, regularization=1e-6,
        meta={
            "trained": time.strftime("%Y-%m-%d"),
            "release": rel,
            "n_train": int(ok.sum()),
            "n_desi": int((channel[ok] == "desi").sum()),
            "n_sdss": int((channel[ok] == "sdss").sum()),
            "z_range": [args.zmin, args.zmax],
            "k_selection": scores or "fixed",
            "dedup_arcsec": 1.0,
        },
    )
    print(f"  done in {time.time() - t0:.0f} s; objects per slice "
          f"{model.n_train.astype(int).min():,}-{model.n_train.astype(int).max():,}")

    args.out.mkdir(exist_ok=True)
    path = args.out / f"qso_{args.system}_full.json"
    model.save(path)
    print(f"\nwrote {path}")

    # Held-out likelihood per selection channel: the measurement of how much the
    # DESI targeting bias matters. A large gap is a warning, not a curiosity.
    if not args.no_sdss and (channel[ok] == "sdss").any():
        print("\nheld-out mean log density by selection channel")
        for tag in ("desi", "sdss"):
            m = np.flatnonzero(channel[ok] == tag)
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
