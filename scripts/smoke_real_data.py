#!/usr/bin/env python
"""End-to-end smoke test on real WSDB photometry, restricted to one sky patch.

Proves that the queries in ``qso_pcolor.data`` run, that real Legacy Surveys
fluxes survive the feature transform, and that a model fitted on real quasars
scores real candidates.  It is a plumbing check, not science: the patch is small,
the model is under-trained, and no conclusion about performance should be drawn
from it.

    python scripts/smoke_real_data.py --ra 180 --dec 0 --radius 3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

BANDS = ("g", "r", "z", "w1", "w2")

QSO_CONE = """
SELECT z.targetid, z.z AS zspec, p.ra, p.dec, p.release, p.maskbits,
       p.flux_g, p.flux_r, p.flux_z, p.flux_w1, p.flux_w2,
       p.flux_ivar_g, p.flux_ivar_r, p.flux_ivar_z,
       p.flux_ivar_w1, p.flux_ivar_w2,
       p.mw_transmission_g, p.mw_transmission_r, p.mw_transmission_z,
       p.mw_transmission_w1, p.mw_transmission_w2
FROM desi_dr1.zpix z
JOIN desi_dr1.photometry p ON p.targetid = z.targetid
WHERE z.spectype = 'QSO' AND z.zwarn = 0 AND z.zcat_primary
  AND z.z > 0.5 AND z.z < 3.5
  AND q3c_radial_query(p.ra, p.dec, {ra}, {dec}, {radius})
"""

BKG_CONE = """
SELECT ra, dec, release, type, maskbits,
       flux_g, flux_r, flux_z, flux_w1, flux_w2,
       flux_ivar_g, flux_ivar_r, flux_ivar_z, flux_ivar_w1, flux_ivar_w2,
       mw_transmission_g, mw_transmission_r, mw_transmission_z,
       mw_transmission_w1, mw_transmission_w2
FROM decals_dr9.main
WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {radius})
  AND maskbits = 0 AND flux_ivar_r > 0
  AND flux_r > 0 AND flux_r < 100
"""


def stack(r: dict, prefix: str) -> np.ndarray:
    return np.stack([np.asarray(r[f"{prefix}{b}"], float) for b in BANDS], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ra", type=float, default=180.0)
    ap.add_argument("--dec", type=float, default=0.0)
    ap.add_argument("--radius", type=float, default=3.0, help="degrees")
    ap.add_argument("--cache", type=Path, default=Path("data/smoke"))
    args = ap.parse_args()

    from qso_pcolor.background import fit_background_model
    from qso_pcolor.data import cached_query, galactic_from_equatorial
    from qso_pcolor.features import RelativeFluxTransform, deredden
    from qso_pcolor.priors import BackgroundSurfaceDensity, EmpiricalQSOPrior
    from qso_pcolor.qso_model import RedshiftMatch, fit_sliced_model
    from qso_pcolor.score import score_candidates

    fmt = {"ra": args.ra, "dec": args.dec, "radius": args.radius}
    t0 = time.time()
    q = cached_query(QSO_CONE.format(**fmt), args.cache.with_name("smoke_qso.npz"))
    b = cached_query(BKG_CONE.format(**fmt), args.cache.with_name("smoke_bkg.npz"))
    print(f"quasars {q['ra'].size:,}, background {b['ra'].size:,} "
          f"({time.time() - t0:.0f} s)")

    tr = RelativeFluxTransform(reference_band="r", min_ref_snr=5.0)

    def featurise(r):
        f, v = deredden(stack(r, "flux_"), stack(r, "flux_ivar_"),
                        stack(r, "mw_transmission_"))
        return tr(f, v, BANDS)

    fq, fb = featurise(q), featurise(b)
    print(f"features: {fq.labels}")
    print(f"  quasars usable {int(fq.usable().sum()):,}; flags {fq.flag_summary()}")
    print(f"  background usable {int(fb.usable().sum()):,}; flags {fb.flag_summary()}")

    okq = fq.usable(min_dims=3) & np.isfinite(fq.ref_mag)
    okb = fb.usable(min_dims=3) & np.isfinite(fb.ref_mag) & (fb.ref_mag < 22.5)

    lq, bq = galactic_from_equatorial(q["ra"][okq], q["dec"][okq])
    lb, bb = galactic_from_equatorial(b["ra"][okb], b["dec"][okb])

    t0 = time.time()
    qso_model = fit_sliced_model(
        fq.x[okq], fq.cov[okq], q["zspec"][okq],
        z_edges=np.linspace(0.5, 3.5, 13),
        observed=fq.observed[okq], n_components=4, min_per_slice=150,
        system="ls_dr9_grzw", labels=fq.labels, seed=0, max_iter=150,
        regularization=1e-6,
    )
    print(f"quasar model fitted in {time.time() - t0:.0f} s; "
          f"objects per slice {qso_model.n_train.astype(int)}")

    t0 = time.time()
    bkg_model = fit_background_model(
        fb.x[okb], fb.cov[okb], fb.ref_mag[okb], lb, bb,
        mag_edges=np.array([17.0, 19.0, 20.5, 21.5, 22.5]),
        nside=8, nside_parent=2, observed=fb.observed[okb],
        n_components=6, min_per_cell=800, n0=500.0,
        system="ls_dr9_grzw", labels=fb.labels, seed=0, max_iter=150,
        regularization=1e-6,
    )
    print(f"background model fitted in {time.time() - t0:.0f} s")

    qso_prior = EmpiricalQSOPrior.build(
        q["zspec"][okq], fq.ref_mag[okq],
        area_deg2=np.pi * args.radius**2,
        z_edges=np.linspace(0.5, 3.5, 13),
        mag_edges=np.array([17.0, 19.0, 20.5, 21.5, 22.5]),
    )
    bkg_density = BackgroundSurfaceDensity.from_catalogue(
        fb.ref_mag[okb], lb, bb,
        mag_edges=np.array([17.0, 19.0, 20.5, 21.5, 22.5]),
        nside=8, nside_parent=2,
    )

    # Score held-out real quasars against real background sources, using each
    # object's own spectroscopic redshift as the "primary".  A working pipeline
    # should separate the two populations; this is a sanity check on the
    # plumbing, not a validation.
    rng = np.random.default_rng(0)
    iq = rng.choice(np.flatnonzero(okq), size=min(200, int(okq.sum())), replace=False)
    ib = rng.choice(np.flatnonzero(okb), size=200, replace=False)

    match = RedshiftMatch(dz_half_width=0.3)

    def sub(fs, idx):
        from qso_pcolor.features import FeatureSet

        return FeatureSet(
            fs.x[idx], fs.cov[idx], fs.observed[idx], fs.ref_flux[idx],
            fs.ref_mag[idx], fs.ref_snr[idx], fs.labels,
            {k: v[idx] for k, v in fs.flags.items()},
        )

    zq = q["zspec"][iq]
    lqs, bqs = galactic_from_equatorial(q["ra"][iq], q["dec"][iq])
    lbs, bbs = galactic_from_equatorial(b["ra"][ib], b["dec"][ib])

    common = dict(
        qso_model=qso_model, background_model=bkg_model, match=match,
        qso_prior=qso_prior, background_density=bkg_density,
        z_grid=np.linspace(0.52, 3.48, 200), min_bands=3,
    )
    s_q = score_candidates(sub(fq, iq), z_primary=zq, l_deg=lqs, b_deg=bqs, **common)
    s_b = score_candidates(
        sub(fb, ib), z_primary=rng.choice(zq, size=ib.size),
        l_deg=lbs, b_deg=bbs, **common,
    )

    def summarise(name, rows):
        bf = np.array([r.log_bayes_factor_qz_bkg for r in rows])
        p = np.array([r.p_sameq for r in rows])
        ok = np.isfinite(bf)
        print(f"  {name:22s} n={ok.sum():3d}  "
              f"median logBF {np.median(bf[ok]):+7.2f}  "
              f"median p_sameq {np.nanmedian(p[ok]):.3f}")

    print("\nscored (note: quasars were in the training set; plumbing check only)")
    summarise("real quasars at z_spec", s_q)
    summarise("random background", s_b)

    statuses = {}
    for r in s_q + s_b:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    print(f"  statuses: {statuses}")


if __name__ == "__main__":
    main()
