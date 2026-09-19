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

QUERY = """
SELECT z.targetid, z.z AS zspec, z.zerr, z.spectype, z.survey, z.program,
       p.ra, p.dec, p.release,
       p.flux_g, p.flux_r, p.flux_z, p.flux_w1, p.flux_w2,
       p.flux_ivar_g, p.flux_ivar_r, p.flux_ivar_z,
       p.flux_ivar_w1, p.flux_ivar_w2,
       p.mw_transmission_g, p.mw_transmission_r, p.mw_transmission_z,
       p.mw_transmission_w1, p.mw_transmission_w2,
       p.fracflux_g, p.fracflux_r, p.fracflux_z,
       p.maskbits, p.morphtype
FROM desi_dr1.zpix z
JOIN desi_dr1.photometry p ON p.targetid = z.targetid
WHERE z.zwarn = 0 AND z.zcat_primary AND z.spectype IN ('QSO', 'STAR', 'GALAXY')
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
    ap.add_argument("--cache", type=Path, default=Path("data/desi_spec_cache.npz"))
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

    from qso_pcolor.data import cached_query

    t0 = time.time()
    r = cached_query(QUERY, args.cache, refresh=args.refresh)
    print(f"pulled {r['ra'].size:,} spectra in {time.time() - t0:.0f} s")

    keep = deduplicate(r["ra"], r["dec"], r["zspec"], args.dedup_arcsec)
    print(f"de-duplicated: removed {int((~keep).sum()):,} repeat rows "
          f"({100 * (~keep).mean():.2f}% of the sample)")
    r = {k: v[keep] for k, v in r.items()}

    is_qso = r["spectype"] == "QSO"
    print(f"  quasars {int(is_qso.sum()):,}, other spectra {int((~is_qso).sum()):,}")

    # Primaries are quasars; companions are anything with a spectrum.
    cq = SkyCoord(r["ra"][is_qso] * u.deg, r["dec"][is_qso] * u.deg)
    call = SkyCoord(r["ra"] * u.deg, r["dec"] * u.deg)
    ip, ic, sep, _ = cq.search_around_sky(call, args.max_sep_arcsec * u.arcsec)
    # search_around_sky(other) returns (idx_into_self=call, idx_into_other=cq)
    sep = sep.arcsec

    qso_idx = np.flatnonzero(is_qso)
    primary = qso_idx[ic]
    companion = ip
    ok = (primary != companion) & (sep >= args.min_sep_arcsec)
    primary, companion, sep = primary[ok], companion[ok], sep[ok]

    zp = r["zspec"][primary]
    zc = r["zspec"][companion]
    dv = np.abs(zc - zp) / (1.0 + zp) * 299792.458
    comp_is_qso = r["spectype"][companion] == "QSO"

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
    for k, v in r.items():
        out[f"comp_{k}"] = v[companion]
        out[f"prim_{k}"] = v[primary]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
