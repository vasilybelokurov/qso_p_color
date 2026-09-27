#!/usr/bin/env python
"""Fetch the reusable master catalogues into the local store (qso_pcolor.store).

Run once; every analysis then reads the files. Each master carries every Legacy
DR9 column the project uses, so adding a derived quantity never needs a query.

    desi_dr1_qso        DESI DR1 zpix QSO, zwarn = 0, zcat_primary, all redshifts,
                        x desi_dr1.photometry (DR9 Tractor) by targetid.
                        desi_dr1.photometry has no nobs_w1/w2: stored as 1 where
                        the W inverse variance is positive (recorded).
    dr16q_dr9           SDSS DR16Q, all rows with their selection flags, x nearest
                        DR9 row of the position's hemisphere within 1 arcsec.
    pairs_desi_dr1, pairs_desi_dr1_dr9, qso_draw_dr9
                        copied from the project's existing products, not re-queried.

    python scripts/build_store.py                  # everything missing
    python scripts/build_store.py --only dr16q_dr9 --refresh
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

DESI_QUERY = """
SELECT z.targetid, z.z AS zspec, z.zerr AS zspec_err, z.zwarn, z.spectype, z.subtype,
       z.survey, z.program, z.deltachi2, z.desi_target, z.sv1_desi_target, z.sv2_desi_target,
       z.sv3_desi_target, z.scnd_target,
       p.ls_id, p.release, p.brickid, p.brick_objid, p.morphtype AS type, p.ra, p.dec,
       p.ebv, p.maskbits, p.fracflux_g, p.fracflux_r, p.fracflux_z,
       p.flux_g, p.flux_r, p.flux_z, p.flux_w1, p.flux_w2,
       p.flux_ivar_g, p.flux_ivar_r, p.flux_ivar_z, p.flux_ivar_w1, p.flux_ivar_w2,
       p.mw_transmission_g, p.mw_transmission_r, p.mw_transmission_z,
       p.mw_transmission_w1, p.mw_transmission_w2,
       p.nobs_g, p.nobs_r, p.nobs_z, p.psfdepth_g, p.psfdepth_r, p.psfdepth_z,
       p.shape_r, p.parallax, p.parallax_ivar, p.pmra, p.pmra_ivar, p.pmdec, p.pmdec_ivar,
       p.gaia_phot_g_mean_mag
FROM desi_dr1.zpix z JOIN desi_dr1.photometry p ON p.targetid = z.targetid
WHERE z.spectype = 'QSO' AND z.zwarn = 0 AND z.zcat_primary
"""

DR16Q_QUERY = """
SELECT sdss_name, ra, dec, z AS zspec, zwarning, sdss_morpho,
       boss_target1, eboss_target0, eboss_target1, eboss_target2, ancillary_target1,
       ancillary_target2
FROM sdssdr16qso.main
"""


def _plain(d: dict) -> dict:
    """No object arrays in the store (loads without pickle)."""
    return {k: (np.asarray(v).astype(str) if np.asarray(v).dtype == object else np.asarray(v))
            for k, v in d.items() if not k.startswith("_")}


def build_desi():
    import sqlutilpy as sqlutil
    t0 = time.time()
    r = sqlutil.get(DESI_QUERY, asDict=True, preamb="SET jit=off; SET statement_timeout='14400s'")
    r = _plain(r)
    for b in ("w1", "w2"):
        r[f"nobs_{b}"] = (np.asarray(r[f"flux_ivar_{b}"]) > 0).astype(np.int16)
    return r, dict(query=DESI_QUERY, elapsed_s=time.time() - t0,
                   notes="nobs_w1/w2 not in desi_dr1.photometry: set to 1 where flux_ivar > 0")


def build_dr16q():
    import sqlutilpy as sqlutil
    from qso_pcolor.legacy import legacy_match
    from qso_pcolor.store import root
    t0 = time.time()
    q = _plain(sqlutil.get(DR16Q_QUERY, asDict=True))
    m = legacy_match(q["ra"], q["dec"], root() / "queries", radius_arcsec=1.0)
    out = dict(q)
    for k, v in _plain(m).items():
        if k in ("idx", "input_ra", "input_dec"):
            continue
        out[k if k not in q else f"ls_{k}"] = v            # ls_ra, ls_dec: the DR9 position
    return out, dict(query=DR16Q_QUERY, match="qso_pcolor.legacy.legacy_match, 1 arcsec",
                     elapsed_s=time.time() - t0)


def copy(src: str, notes: str):
    def f():
        d = np.load(src, allow_pickle=True)
        return _plain({k: d[k] for k in d.files}), dict(copied_from=src, notes=notes)
    return f


BUILDERS = {
    "desi_dr1_qso": build_desi,
    "dr16q_dr9": build_dr16q,
    "pairs_desi_dr1": copy("data/pairs_desi_dr1.npz", "scripts/build_pair_validation.py output"),
    "pairs_desi_dr1_dr9": copy("data/legacy_baseline_v2/companions.npz",
                               "companion positions x legacy_match (hemisphere rule), with mw_transmission"),
    "qso_draw_dr9": copy("data/legacy_baseline_v2/quasars.npz",
                         "multi-survey target draw x legacy_match, with mw_transmission"),
}


def main():
    from qso_pcolor.store import path, save
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(BUILDERS))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    for name in args.only or BUILDERS:
        if path(name).exists() and not args.refresh:
            print(f"{name}: present")
            continue
        t0 = time.time()
        arrays, info = BUILDERS[name]()
        n = len(next(iter(arrays.values())))
        info.update(built=time.strftime("%Y-%m-%d %H:%M"), rows=n, columns=sorted(arrays))
        p = save(name, arrays, info)
        print(f"{name}: {n:,} rows, {len(arrays)} columns -> {p} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
