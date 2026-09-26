#!/usr/bin/env python
"""Fit the broad "unmodelled" field component on fields the background never saw.

The outlier density (``qso_pcolor.outlier``) is a normalised Gaussian at the
centre of the shipped background, with covariance kappa^2 times an envelope
that is at least the background's own second moment and at least every quasar
and background component (``envelope_covariance``). Two numbers need data:

* ``kappa``, its breadth. It must exceed ``kappa_min``, the smallest breadth
  that dominates every quasar and background component in every direction --
  otherwise the tail problem it exists to fix survives in some direction.
* ``eta(m)``, the share of the field surface density assigned to it in each
  reference-magnitude bin.

Both are set by held-out field likelihood. The fields are the latitude-
stratified cones of ``scripts/compare_background_modes.py`` (cached as
``data/modes_cone_*.npz``) that lie away from the eight cones the shipped
background was fitted to; cones 00-07 of that script coincide with them and
are excluded. The remaining cones are split alternately into a fitting set A
and a check set B, so the reported gain on B is out of sample in both kappa
and eta. Selection is identical to ``build_global_background.py``:
``maskbits = 0``, known quasars removed, >= 3 usable dimensions, 17 <= r < 22.5.

    python scripts/fit_outlier_model.py
    python scripts/fit_outlier_model.py --kappa-factors 1.05 1.5 2 3 4 6
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

BANDS = ("g", "r", "z", "w1", "w2")
SYSTEM = "ls_dr9_south_grzw"


def load_modes_script():
    spec = importlib.util.spec_from_file_location(
        "compare_background_modes", Path(__file__).with_name("compare_background_modes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def heldout_fields(radius, first_cone, cache):
    """Features of every usable source in the held-out cones, by cone."""
    if cache.exists():
        d = dict(np.load(cache, allow_pickle=False))
        return d, json.loads(str(d.pop("_cones")))
    from qso_pcolor.background import _RELEASES_FOR_SYSTEM
    from qso_pcolor.data import drop_known_quasars, fetch_known_quasars, fetch_ls_background
    from qso_pcolor.features import RelativeFluxTransform, deredden

    modes = load_modes_script()
    cones = modes.draw_cones(40, 0, per_bin=4)     # the script's own defaults
    tr = RelativeFluxTransform(reference_band="r", min_ref_snr=5.0)
    out = {k: [] for k in ("x", "cov", "observed", "ref_mag", "cone")}
    used = []
    for i, c in enumerate(cones):
        if i < first_cone:
            continue
        rr = fetch_ls_background(Path("data") / f"modes_cone_{i:02d}.npz",
                                 ra=c["ra"], dec=c["dec"], radius_deg=radius)
        n_raw = int(np.size(rr["ra"]))
        if n_raw < 20000 * (radius / 0.5) ** 2:
            continue
        rel = set(int(x) for x in np.unique(np.asarray(rr["release"], int)))
        if not rel <= _RELEASES_FOR_SYSTEM[SYSTEM]:
            continue
        kq = fetch_known_quasars(c["ra"], c["dec"], radius,
                                 cache=Path("data") / f"modes_qso_{i:02d}.npz")
        if np.size(kq["ra"]) == 0:
            continue
        keep = (np.asarray(rr["maskbits"], int) == 0) & drop_known_quasars(
            rr["ra"], rr["dec"], kq["ra"], kq["dec"])
        f, v = deredden(
            np.stack([np.asarray(rr[f"flux_{b}"], float) for b in BANDS], 1)[keep],
            np.stack([np.asarray(rr[f"flux_ivar_{b}"], float) for b in BANDS], 1)[keep],
            np.stack([np.asarray(rr[f"mw_transmission_{b}"], float) for b in BANDS], 1)[keep])
        fs = tr(f, v, BANDS)
        ok = (fs.usable(min_dims=3) & np.isfinite(fs.ref_mag)
              & (fs.ref_mag >= 17.0) & (fs.ref_mag < 22.5))
        out["x"].append(fs.x[ok]); out["cov"].append(fs.cov[ok])
        out["observed"].append(fs.observed[ok]); out["ref_mag"].append(fs.ref_mag[ok])
        out["cone"].append(np.full(int(ok.sum()), i))
        used.append({"index": i, "ra": c["ra"], "dec": c["dec"], "l": c["l"], "b": c["b"],
                     "n_used": int(ok.sum())})
        print(f"  cone {i:02d} (l,b)=({c['l']:6.1f},{c['b']:+5.1f})  {int(ok.sum()):7,d} sources")
    d = {k: np.concatenate(v) for k, v in out.items()}
    np.savez(cache, **d, _cones=json.dumps(used))
    return d, used


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--first-cone", type=int, default=8,
                    help="cones below this index coincide with the global background's fields")
    ap.add_argument("--kappa-factors", type=float, nargs="+",
                    default=[1.02, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
                    help="kappa values to try, as multiples of kappa_min")
    ap.add_argument("--cache", type=Path, default=Path("data/outlier_heldout_fields.npz"))
    ap.add_argument("--out", type=Path, default=Path("models/outlier_south.json"))
    ap.add_argument("--report", type=Path, default=Path("data/outlier_fit_report.json"))
    args = ap.parse_args()

    from qso_pcolor.background import BackgroundColourModel
    from qso_pcolor.outlier import (OutlierModel, envelope_covariance, fit_outlier_fraction,
                                    min_dominant_kappa, mixture_moments)
    from qso_pcolor.qso_model import SlicedColourRedshiftModel

    qso = SlicedColourRedshiftModel.load("models/qso_south_full.json")
    bkg = BackgroundColourModel.load("models/background_south_global.json")
    if bkg.local or bkg.parent:
        raise SystemExit("expected the footprint-average background (global mixtures only)")
    edges = bkg.mag_edges
    reference = list(bkg.global_)
    check = list(qso.mixtures) + list(bkg.global_)
    mean, base = mixture_moments(reference)
    env = envelope_covariance(base, check)
    kmin = min_dominant_kappa(env, check)
    print(f"kappa_min = {kmin:.3f}  (checked against {sum(m.n_components for m in check)} components)")

    t0 = time.time()
    d, cones = heldout_fields(args.radius, args.first_cone, args.cache)
    print(f"held-out field sources: {d['x'].shape[0]:,} in {len(cones)} cones ({time.time()-t0:.0f} s)")
    set_a = np.isin(d["cone"], [c["index"] for c in cones[0::2]])
    mb = bkg.mag_bin(d["ref_mag"])

    # log p_B, vectorised per magnitude bin; equal to bkg.log_prob for a global-only model
    log_pb = np.empty(d["x"].shape[0])
    for j in range(bkg.n_mag_bins):
        s = mb == j
        log_pb[s] = bkg.global_[j].log_prob(d["x"][s], d["cov"][s], observed=d["observed"][s])
    probe = np.arange(0, d["x"].shape[0], max(1, d["x"].shape[0] // 500))
    ref = bkg.log_prob(d["x"][probe], d["cov"][probe], d["ref_mag"][probe],
                       np.zeros(probe.size), np.full(probe.size, 60.0),
                       observed=d["observed"][probe])
    assert np.allclose(ref, log_pb[probe], rtol=0, atol=1e-9), "vectorised p_B disagrees"

    rows = []
    for fac in args.kappa_factors:
        kappa = fac * kmin
        um = OutlierModel(mean, kappa**2 * env, np.zeros(bkg.n_mag_bins), edges, kappa,
                          kmin, SYSTEM, qso.labels)
        log_pu = um.log_prob(d["x"], d["cov"], observed=d["observed"])
        eta = np.array([fit_outlier_fraction(log_pb[set_a & (mb == j)], log_pu[set_a & (mb == j)])
                        for j in range(bkg.n_mag_bins)])
        e = eta[mb]
        with np.errstate(divide="ignore"):
            gain = np.logaddexp(np.log1p(-e) + log_pb, np.log(e) + log_pu) - log_pb
        row = {"kappa_factor": fac, "kappa": kappa, "eta": eta.tolist(),
               "gain_A_nats_per_obj": float(gain[set_a].mean()),
               "gain_B_nats_per_obj": float(gain[~set_a].mean()),
               "gain_B_by_bin": [float(gain[~set_a & (mb == j)].mean())
                                 for j in range(bkg.n_mag_bins)]}
        rows.append(row)
        print(f"  kappa = {fac:4.2f} x kappa_min = {kappa:6.2f}   eta = "
              + " ".join(f"{x:.2e}" for x in eta)
              + f"   gain A {row['gain_A_nats_per_obj']:+.5f}  B {row['gain_B_nats_per_obj']:+.5f} nats/obj")

    best = max(rows, key=lambda r: r["gain_A_nats_per_obj"])
    print(f"chosen on A: kappa = {best['kappa_factor']} x kappa_min; held-out gain on B "
          f"{best['gain_B_nats_per_obj']:+.5f} nats/obj")
    meta = {
        "built": time.strftime("%Y-%m-%d"), "by": "scripts/fit_outlier_model.py",
        "reference": "models/background_south_global.json (global mixtures, equal weights)",
        "shape": "envelope_covariance(reference moments, all checked components)",
        "reference_sd": np.sqrt(np.diag(base)).tolist(),
        "envelope_sd": np.sqrt(np.diag(env)).tolist(),
        "dominance_checked_against": ["models/qso_south_full.json (all slices)",
                                      "models/background_south_global.json (all bins)"],
        "fields": cones, "set_A": [c["index"] for c in cones[0::2]],
        "set_B": [c["index"] for c in cones[1::2]],
        "n_A": int(set_a.sum()), "n_B": int((~set_a).sum()),
        "selection": "decals_dr9.main, maskbits = 0, known quasars removed, >= 3 dims, "
                     "17 <= r < 22.5; cones of compare_background_modes.py from index "
                     f"{args.first_cone}",
        "kappa_scan": rows, "chosen_by": "max held-out log-likelihood on set A",
    }
    model = OutlierModel(mean, best["kappa"]**2 * env, best["eta"], edges, best["kappa"],
                         kmin, SYSTEM, qso.labels, meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    args.report.write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
