#!/usr/bin/env python
"""Measure a local refit's predictions on independent, thinned cone rows.

This deliberately differs from the whole-cell spatial test: a local fit sees
part of its own neighbourhood. Random thinning gives independent training and
validation catalogues, with the measured cone area multiplied by each sampling
fraction in the count likelihood. No new sky-independent validation is claimed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from qso_pcolor.baseline import XDQSOBaseline
from qso_pcolor.data import galactic_from_equatorial
from qso_pcolor.legacy import dereddened_relative_fluxes
from qso_pcolor.spatial import fit_local_background


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adaptation", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/legacy_spatial.json"))
    ap.add_argument("--fit-fraction", type=float, required=True)
    ap.add_argument("--max-cones", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 0 < args.fit_fraction < 1 or args.max_cones < 1:
        raise ValueError("invalid validation thinning fraction or cone count")
    cfg = json.loads(args.config.read_text())
    sc = json.loads(Path(cfg["sample_config"]).read_text())
    root = Path(sc["data_dir"])
    design = json.loads((root / "design.json").read_text())
    area = json.loads((root / "areas.json").read_text())
    base = XDQSOBaseline.load(cfg["source_bundle"], background=args.adaptation)
    rng = np.random.default_rng(args.seed)
    cones = [c for c in design["cones"] if c["role"] == "test" and c["hemisphere"] == "south"]
    if len(cones) > args.max_cones:
        cones = [cones[j] for j in rng.choice(len(cones), args.max_cones, replace=False)]
    result = []
    for cone in cones:
        h = cone["hemisphere"]
        rows = dict(np.load(root / "cones" / f"cone_{cone['cone']:03d}.npz", allow_pickle=False))
        usable_area = area[str(cone["cone"])]["area_deg2"]
        train = rng.random(len(rows["ra"])) < args.fit_fraction
        dec = base.selection.decide(rows)
        fs, ok = dereddened_relative_fluxes(rows, h)
        lo, hi = base.manifest["domain"]["ref_mag"]
        eligible = dec["accepted"] & ok & (dec["hemisphere"] == h) & (fs.ref_mag >= lo) & (fs.ref_mag < hi)
        # Exclude two real sources as surrogate primary/companion positions.
        targets = np.flatnonzero(eligible)[:2]
        train[targets] = False
        selected = base.manifest["background_adaptation"]["selected"][h]
        adapted = fit_local_background(base, {k: v[train] for k, v in rows.items()}, hemisphere=h,
            centre_ra=cone["ra"], centre_dec=cone["dec"], radius_deg=sc["cone_radius_deg"],
            usable_area_deg2=usable_area * args.fit_fraction,
            exclude_ra=rows["ra"][targets], exclude_dec=rows["dec"][targets],
            exclusion_arcsec=cfg["local_exclusion_arcsec"], colour_n0=selected["colour_n0"],
            density_n0=selected["density_n0"], max_iter=cfg["weight_max_iter"], tol=cfg["weight_tol"],
            min_density_fraction=cfg["min_density_fraction"])
        local = adapted.apply(base)
        valid = eligible & ~train & ~rows["known_quasar"].astype(bool)
        valid[targets] = False
        # Remove the same small apertures from the validation rows. Their
        # geometric area is recorded as a bound relative to the usable area.
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        positions = SkyCoord(rows["ra"] * u.deg, rows["dec"] * u.deg)
        for t in targets:
            valid &= positions.separation(positions[t]).arcsec > cfg["local_exclusion_arcsec"]
        f = fs.subset(np.flatnonzero(valid))
        l, b = galactic_from_equatorial(rows["ra"][valid], rows["dec"][valid])
        prior = base.parts[h]["qso_prior"]
        wz = np.array([np.interp(f.ref_mag, prior.mag_centres, s) for s in prior.sigma]).T * np.diff(prior.z_edges)
        kappa = adapted.meta["recognised_fraction"]
        with np.errstate(divide="ignore"):
            logq = logsumexp(base.parts[h]["qso"]._log_p_slices(f.x, f.cov, f.observed) + np.log(wz), axis=1) + np.log(1 - kappa)
        sq = np.sum(prior.sigma * np.diff(prior.z_edges)[:, None], axis=0)
        scores, predictions = {}, {}
        lc, bc = galactic_from_equatorial(np.array([cone["ra"]]), np.array([cone["dec"]]))
        for name, model in (("spatial", base), ("local", local)):
            p = model.parts[h]; out = p["outlier"]
            lb = p["background"].log_prob(f.x, f.cov, f.ref_mag, l, b, observed=f.observed)
            eta = out.fraction_at(f.ref_mag)
            with np.errstate(divide="ignore"):
                field = np.log(p["background_density"](f.ref_mag, l, b)) + np.logaddexp(
                    np.log1p(-eta) + lb, np.log(eta) + out.log_prob(f.x, f.cov, observed=f.observed))
            rate = p["background_density"](prior.mag_centres, np.full(len(sq), lc[0]), np.full(len(sq), bc[0]))
            predicted = (1 - args.fit_fraction) * usable_area * np.diff(prior.mag_edges) * (rate + (1 - kappa) * sq)
            scores[name] = float((np.logaddexp(field, logq).sum() - predicted.sum()) / len(f.ref_mag))
            predictions[name] = float(predicted.sum())
        row = dict(cone=cone["cone"], n_train=adapted.meta["n_used"], n_validation=len(f.ref_mag),
            log_score=scores, delta_log_score=scores["local"] - scores["spatial"], predicted=predictions,
            excluded_area_upper_fraction=2 * np.pi * (cfg["local_exclusion_arcsec"] / 3600)**2 / usable_area,
            fit_converged=all(r["converged"] for r in adapted.meta["weight_fits"]))
        result.append(row)
        print(f"cone {cone['cone']}: {len(f.ref_mag)} held-out sources, delta {row['delta_log_score']:+.4f}", flush=True)
    report = dict(adaptation_id=base.bundle_id, seed=args.seed, fit_fraction=args.fit_fraction,
        scope="Independent random thinning within previously reserved southern cones; validation of candidate-local adaptation, not whole-cell prediction. Tiny exclusion-hole area is bounded below.",
        cones=result, mean_delta_log_score=float(np.average([r["delta_log_score"] for r in result],
            weights=[r["n_validation"] for r in result])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
