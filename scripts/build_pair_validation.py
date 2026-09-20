#!/usr/bin/env python
"""Build the labelled close-pair validation set from DESI DR1.

This is the sample that makes the whole exercise testable.  Around each
spectroscopically confirmed quasar we look for a second object that also has a
spectrum, and label it by what that spectrum says:

``same_z``
    A quasar whose redshift matches the primary — the positives.
``field_q``
    A quasar at a different redshift — the hardest negatives, and the class that
    a two-hypothesis (quasar versus star) score cannot distinguish from a
    positive at all.
``non_qso``
    A star or galaxy — the easy negatives.

Because both objects have spectra, the labels are not inferred from the colours
we are trying to score, so a model validated here is not marking its own
homework.

Measured yield (DESI DR1, run 2026-09-18)
------------------------------------------
======== ============ ============ ==============
sep      same_z pairs  field_q      duplicates
======== ============ ============ ==============
0-0.5"           9549           28  yes, all of it
0.5-3"            958          101  mostly
3-5"              212          333  no
5-10"             499         1553  no
10-20"            900         6523  no
======== ============ ============ ==============

**The duplicates matter.** 9,577 "pairs" closer than 0.5 arcsec with identical
redshifts are the *same physical quasar* entered twice under two ``targetid``
values.  ``zcat_primary`` de-duplicates per target, not per sky position, so a
naive build inflates the positive class by a factor of six and trains the model
on repeated objects.  ``--dedup-arcsec`` removes them, and the count of what it
removed is reported rather than hidden.

Usage
-----
    python scripts/build_pair_validation.py --out data/pairs_desi_dr1.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# Why three queries and not one.  The obvious build -- pull every DESI DR1
# spectrum with photometry and search for pairs locally -- is 20.4 million rows
# (measured 2026-09-20) and several gigabytes.  A single SQL statement that
# joins the pair search to the companions' spectra is planned as sequential
# scans of both large tables (measured: both ``photometry`` and ``zpix`` go to
# Seq Scan the moment the second join is added).  Split, each step is
# index-driven: the pair search runs on the q3c index (2.6 s per 20,000
# primaries), and the spectra/photometry attach through the ``targetid`` primary
# keys.

PRIMARIES_QUERY = """
SELECT z.targetid, z.z AS zspec, z.zerr, p.ra, p.dec, p.release
FROM desi_dr1.zpix z
JOIN desi_dr1.photometry p ON p.targetid = z.targetid
WHERE z.zwarn = 0 AND z.zcat_primary AND z.spectype = 'QSO'
"""

# Every DESI *target* within the search radius of every quasar primary.  Most
# neighbours will turn out to have no usable DR1 spectrum; they are dropped in
# the next step.  Plain ``JOIN ON q3c_join`` with the index on the second pair
# -- adding a third table here defeats the index (see above).
PAIRS_QUERY = """
WITH q AS MATERIALIZED (
  SELECT z.targetid, p.ra, p.dec
  FROM desi_dr1.zpix z JOIN desi_dr1.photometry p ON p.targetid = z.targetid
  WHERE z.zwarn = 0 AND z.zcat_primary AND z.spectype = 'QSO'
)
SELECT q.targetid AS prim_targetid, c.targetid AS comp_targetid,
       q3c_dist(q.ra, q.dec, c.ra, c.dec) * 3600 AS sep_arcsec
FROM q JOIN desi_dr1.photometry c
  ON q3c_join(q.ra, q.dec, c.ra, c.dec, %(radius_deg)s)
WHERE c.targetid <> q.targetid
"""

# Spectrum and photometry for a list of companion target ids, uploaded as a
# temporary table and joined on the primary keys.
COMPANIONS_QUERY = """
SELECT t.targetid, z.z AS zspec, z.zerr, z.spectype, z.survey, z.program,
       p.ra, p.dec, p.release,
       p.flux_g, p.flux_r, p.flux_z, p.flux_w1, p.flux_w2,
       p.flux_ivar_g, p.flux_ivar_r, p.flux_ivar_z,
       p.flux_ivar_w1, p.flux_ivar_w2,
       p.mw_transmission_g, p.mw_transmission_r, p.mw_transmission_z,
       p.mw_transmission_w1, p.mw_transmission_w2,
       p.fracflux_g, p.fracflux_r, p.fracflux_z,
       p.maskbits, p.morphtype
FROM mytmptable t
JOIN desi_dr1.zpix z ON z.targetid = t.targetid
JOIN desi_dr1.photometry p ON p.targetid = t.targetid
WHERE z.zwarn = 0 AND z.zcat_primary
  AND z.spectype IN ('QSO', 'STAR', 'GALAXY')
"""


def deduplicate(ra, dec, zspec, radius_arcsec: float) -> np.ndarray:
    """Keep one row per physical object.

    Two rows within ``radius_arcsec`` whose redshifts agree to better than
    ``dv/c/(1+z) = 1e-3`` are the same object observed twice.  We keep the first
    of each such group.  Returns a boolean mask of rows to keep.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    c = SkyCoord(ra * u.deg, dec * u.deg)
    i1, i2, _, _ = c.search_around_sky(c, radius_arcsec * u.arcsec)
    m = i1 < i2
    i1, i2 = i1[m], i2[m]
    dz = np.abs(zspec[i1] - zspec[i2]) / (1.0 + np.minimum(zspec[i1], zspec[i2]))
    same = dz < 1e-3

    keep = np.ones(ra.size, dtype=bool)
    for a, b in zip(i1[same], i2[same]):
        if keep[a]:
            keep[b] = False
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/pairs_cache.npz"),
                    help="caches are written beside this path")
    ap.add_argument("--max-sep-arcsec", type=float, default=30.0)
    ap.add_argument("--min-sep-arcsec", type=float, default=3.0,
                    help="below this, Legacy Surveys deblending is unreliable and "
                         "the colours are not independent measurements")
    ap.add_argument("--dedup-arcsec", type=float, default=1.0)
    ap.add_argument("--dv-match-kms", type=float, default=3000.0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    from astropy.coordinates import SkyCoord
    import astropy.units as u

    import hashlib

    import sqlutilpy as sqlutil

    from qso_pcolor.data import _load_npz, _save_npz, cached_query

    cache_dir = args.cache.parent
    t0 = time.time()
    prim = cached_query(PRIMARIES_QUERY, cache_dir / "pair_primaries.npz",
                        refresh=args.refresh)
    print(f"primaries: {prim['targetid'].size:,} DESI DR1 quasars "
          f"({time.time() - t0:.0f} s)")

    t0 = time.time()
    pairs = cached_query(
        PAIRS_QUERY % {"radius_deg": args.max_sep_arcsec / 3600.0},
        cache_dir / "pair_search.npz", refresh=args.refresh,
    )
    print(f"pair search: {pairs['prim_targetid'].size:,} neighbours within "
          f"{args.max_sep_arcsec:g} arcsec ({time.time() - t0:.0f} s)")

    # companions with a usable spectrum: attach by targetid
    comp_ids = np.unique(pairs["comp_targetid"]).astype(np.int64)
    digest = hashlib.sha1(
        (COMPANIONS_QUERY + str(comp_ids.size) + str(int(comp_ids[:1000].sum()))).encode()
    ).hexdigest()[:10]
    comp_cache = cache_dir / f"pair_companions_{digest}.npz"
    if comp_cache.exists() and not args.refresh:
        comp = _load_npz(comp_cache)
    else:
        t0 = time.time()
        comp = sqlutil.local_join(
            COMPANIONS_QUERY, "mytmptable", (comp_ids,), ("targetid",),
            asDict=True, intNullVal=-1,
        )
        _save_npz(comp_cache, **comp)
        print(f"companion spectra: {comp['targetid'].size:,} of {comp_ids.size:,} "
              f"neighbours have a usable DR1 spectrum ({time.time() - t0:.0f} s)")

    # -- de-duplicate BOTH ends by position + redshift ----------------------
    keep_p = deduplicate(prim["ra"], prim["dec"], prim["zspec"], args.dedup_arcsec)
    keep_c = deduplicate(comp["ra"], comp["dec"], comp["zspec"], args.dedup_arcsec)
    print(f"de-duplicated: {int((~keep_p).sum()):,} repeat primaries, "
          f"{int((~keep_c).sum()):,} repeat companions removed")
    prim = {k: v[keep_p] for k, v in prim.items()}
    comp = {k: v[keep_c] for k, v in comp.items()}

    # -- assemble pairs from the surviving rows ----------------------------
    p_pos = {int(t): i for i, t in enumerate(prim["targetid"])}
    c_pos = {int(t): i for i, t in enumerate(comp["targetid"])}
    primary = np.array([p_pos.get(int(t), -1) for t in pairs["prim_targetid"]])
    companion = np.array([c_pos.get(int(t), -1) for t in pairs["comp_targetid"]])
    sep = np.asarray(pairs["sep_arcsec"], float)
    ok = (primary >= 0) & (companion >= 0) & (sep >= args.min_sep_arcsec)
    primary, companion, sep = primary[ok], companion[ok], sep[ok]
    print(f"pairs with both spectra at {args.min_sep_arcsec:g}-"
          f"{args.max_sep_arcsec:g} arcsec: {primary.size:,}")

    # The two catalogues below share a column set for the output block.
    r = {"prim": prim, "comp": comp}

    zp = prim["zspec"][primary]
    zc = comp["zspec"][companion]
    dv = np.abs(zc - zp) / (1.0 + zp) * 299792.458
    comp_is_qso = comp["spectype"][companion] == "QSO"

    label = np.full(primary.size, "non_qso", dtype="<U8")
    label[comp_is_qso & (dv < args.dv_match_kms)] = "same_z"
    label[comp_is_qso & (dv >= args.dv_match_kms)] = "field_q"

    print(f"\nlabelled companions at {args.min_sep_arcsec}-{args.max_sep_arcsec} arcsec:")
    for lab in ("same_z", "field_q", "non_qso"):
        print(f"  {lab:8s} {int((label == lab).sum()):6,d}")

    out = {
        "primary_idx": primary,
        "companion_idx": companion,
        "sep_arcsec": sep,
        "z_primary": zp,
        "z_companion": zc,
        "dv_kms": dv,
        "label": label,
    }
    for k, v in comp.items():
        out[f"comp_{k}"] = v[companion]
    for k, v in prim.items():
        out[f"prim_{k}"] = v[primary]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
