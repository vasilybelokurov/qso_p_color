#!/usr/bin/env python
"""Restore the PSF background's sky dependence using the cached field cones.

No catalogue queries. The immutable pooled bundle supplies the component
shapes and quasar model. All geometry/pooling choices precede test evaluation.
Outputs are a portable BackgroundAdaptation and a measured validation report.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).parent))


def prepare_cone(job):
    from qso_pcolor.baseline import XDQSOBaseline
    from qso_pcolor.data import galactic_from_equatorial
    from qso_pcolor.legacy import dereddened_relative_fluxes
    from qso_pcolor.spatial import component_log_prob
    from qso_pcolor.background import galactic_healpix
    bundle, root, cone, area, cfg, cache = job
    path = Path(root) / "cones" / f"cone_{cone['cone']:03d}.npz"
    source = path.read_bytes()
    key = hashlib.sha256(source + json.dumps((bundle, cone, area, cfg), sort_keys=True).encode()
        + Path(__file__).read_bytes() + Path("src/qso_pcolor/spatial.py").read_bytes()).hexdigest()
    target = Path(cache) / f"cone_{cone['cone']:03d}_{key}.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as f:
            info = json.loads(str(f["info"]))
            return {k: f[k] for k in f.files if k != "info"}, info
    rows = dict(np.load(path, allow_pickle=False))
    bl = XDQSOBaseline.load(bundle)
    h = cone["hemisphere"]; part = bl.parts[h]
    fs, ok = dereddened_relative_fluxes(rows, h)
    dec = bl.selection.decide(rows)
    lo, hi = bl.manifest["domain"]["ref_mag"]
    use = ok & dec["accepted"] & (dec["hemisphere"] == h) & (fs.ref_mag >= lo) & (fs.ref_mag < hi)
    known = rows["known_quasar"].astype(bool)
    n_known = int((use & known).sum())
    index = np.flatnonzero(use & ~known)
    f = fs.subset(index)
    l, b = galactic_from_equatorial(rows["ra"][index], rows["dec"][index])
    # Check positions independently of the whole-cone geometry filter.
    ns = cfg["partition_nside"]
    cell = int(galactic_healpix(np.array([cone["l"]]), np.array([cone["b"]]), ns)[0])
    if np.any(galactic_healpix(l, b, ns) != cell):
        raise ValueError(f"cone {cone['cone']} crosses its declared spatial partition")
    prior, bg = part["qso_prior"], part["background"]
    sq_bins = (prior.sigma * np.diff(prior.z_edges)[:, None]).sum(0)
    kappa = min(1., n_known / (area * np.sum(sq_bins * np.diff(prior.mag_edges))))
    wz = np.array([np.interp(f.ref_mag, prior.mag_centres, s) for s in prior.sigma]).T * np.diff(prior.z_edges)
    with np.errstate(divide="ignore"):
        logq = logsumexp(part["qso"]._log_p_slices(f.x, f.cov, f.observed) + np.log(wz), axis=1) + np.log(1 - kappa)
    lp = np.zeros((len(index), max(m.n_components for m in bg.global_)))
    logb = np.zeros(len(index)); imag = bg.mag_bin(f.ref_mag)
    for j, mix in enumerate(bg.global_):
        take = imag == j
        comp = component_log_prob(mix, f.x[take], f.cov[take], f.observed[take])
        lp[take, :mix.n_components] = comp
        logb[take] = logsumexp(comp + np.log(mix.weights), axis=1)
    sb = part["background_density"](f.ref_mag, l, b)
    nonq = np.exp(logb + np.log(sb) - np.logaddexp(logb + np.log(sb), logq))
    out = part["outlier"]
    eta = out.fraction_at(f.ref_mag)
    logu = out.log_prob(f.x, f.cov, observed=f.observed)
    counts = np.histogram(f.ref_mag, prior.mag_edges)[0]
    corrected = np.maximum(counts - (1 - kappa) * sq_bins * area * np.diff(prior.mag_edges),
                           cfg["min_density_fraction"] * counts)
    info = dict(cone=cone["cone"], cell=cone["cell"], role=cone["role"], hemisphere=h,
        ra=cone["ra"], dec=cone["dec"], l=cone["l"], b=cone["b"], area=float(area),
        n_known=n_known, recognised_fraction=kappa, nonq_counts=corrected.tolist(),
        counts=counts.tolist(), sampling_weight=cone["sampling_weight"],
        stratum_area=cone["stratum_area"],
        n=len(index), input_sha256=hashlib.sha256(source).hexdigest(),
        density_floor_active=(corrected > counts - (1 - kappa) * sq_bins * area * np.diff(prior.mag_edges)).tolist())
    data = dict(m=f.ref_mag, l=l, b=b, cone=np.full(len(index), cone["cone"], int),
        nonq=nonq, sampling_weight=np.full(len(index), cone["sampling_weight"]),
        log_components=lp, logq=logq, logu=logu, eta=eta)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.npz")
    np.savez(tmp, **data, info=np.array(json.dumps(info)))
    tmp.replace(target)
    return data, info


def subset(data, cones, ids):
    take = np.isin(data["cone"], ids)
    selected = [dict(c) for c in cones if c["cone"] in ids]
    subset_data = {k: v[take] for k, v in data.items()}
    # Withheld areas must not dilute the remaining cones' sampling weights.
    areas = {}
    for c in selected:
        areas[c["cell"]] = areas.get(c["cell"], 0.) + c["area"]
    for c in selected:
        c["sampling_weight"] = c["stratum_area"] / areas[c["cell"]]
        subset_data["sampling_weight"][subset_data["cone"] == c["cone"]] = c["sampling_weight"]
    return subset_data, selected


def prediction(background, density, part, data, cones):
    """Observable population score, including quasars left in the field sample."""
    from qso_pcolor.background import galactic_healpix
    from qso_pcolor.spatial import effective_weights
    pix = galactic_healpix(data["l"], data["b"], background.nside)
    mb = background.mag_bin(data["m"])
    lb = np.empty(len(mb))
    for cell, j in np.unique(np.column_stack((pix, mb)), axis=0):
        use = (pix == cell) & (mb == j)
        first = np.flatnonzero(use)[0]
        w = effective_weights(background, data["m"][first], data["l"][first], data["b"][first])
        with np.errstate(divide="ignore"):
            lb[use] = logsumexp(data["log_components"][use, :len(w)] + np.log(w), axis=1)
    rates = density(data["m"], data["l"], data["b"])
    with np.errstate(divide="ignore"):
        field = np.log(rates) + np.logaddexp(np.log1p(-data["eta"]) + lb, np.log(data["eta"]) + data["logu"])
    intensity = np.logaddexp(field, data["logq"])
    prior = part["qso_prior"]
    sq = np.sum(prior.sigma * np.diff(prior.z_edges)[:, None], axis=0)
    result = []
    for c in cones:
        take = data["cone"] == c["cone"]
        rate = density(prior.mag_centres, np.full(len(sq), c["l"]), np.full(len(sq), c["b"]))
        pred = c["area"] * np.diff(prior.mag_edges) * (rate + (1 - c["recognised_fraction"]) * sq)
        result.append(dict(cone=c["cone"], cell=c["cell"], n=int(take.sum()),
            log_score_sum=float(intensity[take].sum() - pred.sum()),
            observed=c["counts"], predicted=pred.tolist()))
    return result


def mean_score(rows):
    return sum(r["log_score_sum"] for r in rows) / sum(r["n"] for r in rows)


def main():
    from qso_pcolor.baseline import XDQSOBaseline, resolve_bundle
    from qso_pcolor.spatial import BackgroundAdaptation, fit_spatial_background, cone_within_cell
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/legacy_spatial.json"))
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    sample = json.loads(Path(cfg["sample_config"]).read_text())
    root = Path(sample["data_dir"])
    bl = XDQSOBaseline.load(cfg["source_bundle"])
    # Resolve the pointer before the cache key is made.
    bundle = str(resolve_bundle(cfg["source_bundle"]))
    design = json.loads((root / "design.json").read_text())
    areas = json.loads((root / "areas.json").read_text())
    cfg["partition_nside"] = max(sample["cell_nside"], max(cfg["nside_grid"]))
    retained, rejected = [], []
    for c in design["cones"]:
        reason = None
        if not cone_within_cell(c["l"], c["b"], sample["cone_radius_deg"],
                nside=cfg["partition_nside"], boundary_factor=cfg["boundary_factor"]):
            reason = "crosses_sky_cell_boundary"
        elif abs(c["b"]) - sample["cone_radius_deg"] < bl.manifest["domain"]["min_abs_b_deg"]:
            reason = "crosses_latitude_boundary"
        if reason:
            rejected.append(dict(cone=c["cone"], role=c["role"], reason=reason))
        else:
            retained.append(c)
    print(f"Geometry: retained {len(retained)} whole cones; excluded {len(rejected)} boundary cones", flush=True)
    sums = {}
    for c in retained:
        key = c["hemisphere"], c["cell"]
        sums[key] = sums.get(key, 0.) + areas[str(c["cone"])]["area_deg2"]
    jobs = []
    for c in retained:
        stratum_area = design["cells"][str(c["cell"])]["by_hemisphere"][c["hemisphere"]]
        w = stratum_area / sums[c["hemisphere"], c["cell"]]
        jobs.append((bundle, str(root), dict(c, sampling_weight=w, stratum_area=stratum_area),
                     areas[str(c["cone"])]["area_deg2"], cfg, str(args.cache)))
    print(f"Preparing {len(jobs)} cones", flush=True)
    prepared = []
    with ProcessPoolExecutor(cfg["workers"]) as pool:
        for i, item in enumerate(pool.map(prepare_cone, jobs)):
            prepared.append(item)
            if (i + 1) % 20 == 0:
                print(f"Prepared {i + 1}/{len(jobs)}", flush=True)
    geometry = dict(partition_nside=cfg["partition_nside"], retained_cones=[c["cone"] for c in retained],
        rejected_cones=rejected, source_cell_check="All retained source positions verified inside their cone centre's partition cell.",
        scope="Partitions isolate new spatial fits. Global component shapes and priors retain their original training provenance.")
    outputs, report = {}, dict(source_bundle=bl.bundle_id, config=cfg, hemispheres={}, geometry=geometry)
    for h, part in bl.parts.items():
        chosen = [(d, c) for d, c in prepared if c["hemisphere"] == h]
        data = {k: np.concatenate([d[k] for d, c in chosen]) for k in chosen[0][0]}
        cones = [c for d, c in chosen]
        rng = np.random.default_rng(cfg["seed"])
        hold = []
        for cell in sorted({c["cell"] for c in cones if c["role"] == "fit"}):
            ids = [c["cone"] for c in cones if c["cell"] == cell and c["role"] == "fit"]
            if len(ids) > 1:
                hold.append(int(rng.choice(ids)))
        fitids = [c["cone"] for c in cones if c["role"] == "fit" and c["cone"] not in hold]
        tr, ct = subset(data, cones, fitids)
        va, cv = subset(data, cones, hold)
        base, prior = part["background"], part["qso_prior"]
        glob = part["background_density"](prior.mag_centres, np.zeros(len(prior.mag_centres)), np.full(len(prior.mag_centres), 90.))
        kw = dict(density_edges=prior.mag_edges, global_density=glob,
                  max_iter=cfg["weight_max_iter"], tol=cfg["weight_tol"], meta=dict(source_bundle_id=bl.bundle_id))
        trials = []
        for ns in cfg["nside_grid"]:
            bg, den = fit_spatial_background(base, tr, ct, nside=ns, nside_parent=ns // cfg["parent_ratio"],
                colour_n0=cfg["colour_n0_grid"][0], density_n0=cfg["density_n0_grid"][0], **kw)
            for cn in cfg["colour_n0_grid"]:
                for dn in cfg["density_n0_grid"]:
                    bg.n0, den.n0 = cn, dn
                    pred = prediction(bg, den, part, va, cv)
                    row = dict(nside=ns, nside_parent=ns // cfg["parent_ratio"], colour_n0=cn,
                               density_n0=dn, score=mean_score(pred))
                    trials.append(row)
            print(f"[{h}] nside={ns} selected-cone trials complete", flush=True)
        best = max(trials, key=lambda x: x["score"])
        # Fit adaptation on fit cells only. Select/calibration and test cones
        # stay out of the adaptation and provide separate whole-cell checks.
        full, cf = subset(data, cones, [c["cone"] for c in cones if c["role"] == "fit"])
        bg, den = fit_spatial_background(base, full, cf, **{k: best[k] for k in
            ("nside", "nside_parent", "colour_n0", "density_n0")}, **kw)
        outputs[h] = dict(background=bg, density=den)
        validations = {}
        for role in ("select", "calib", "test"):
            vd, vc = subset(data, cones, [c["cone"] for c in cones if c["role"] == role])
            if not vc:
                validations[role] = dict(status="no cones")
                continue
            pooled = prediction(base, part["background_density"], part, vd, vc)
            spatial = prediction(bg, den, part, vd, vc)
            validations[role] = dict(n=sum(c["n"] for c in spatial), cones=len(vc),
                delta_log_score=mean_score(spatial) - mean_score(pooled), pooled=pooled, spatial=spatial)
        report["hemispheres"][h] = dict(selected=best, trials=trials, fit_cones=[c["cone"] for c in cf],
            choice_holdout_cones=hold, conditional_cv_scope="Global shapes and original prior fixed; inner cones withheld only from adaptation.",
            validations=validations, n_colour_cells=len(bg.local), n_parent_cells=len(bg.parent),
            weight_converged=sum(r["converged"] for r in bg.meta["weight_fits"]),
            weight_fits=len(bg.meta["weight_fits"]))
        print(f"[{h}] chose {best}; {len(bg.local)} cell/magnitude fits", flush=True)
    meta = dict(mode="spatial", built=time.strftime("%Y-%m-%d"), config=cfg,
        geometry=geometry,
        selected={h: r["selected"] for h, r in report["hemispheres"].items()},
        fit_cones={h: r["fit_cones"] for h, r in report["hemispheres"].items()},
        inputs={str(c["cone"]): c["input_sha256"] for d, c in prepared if c["role"] != "test"},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes() + Path("src/qso_pcolor/spatial.py").read_bytes()).hexdigest(),
        status="Spatial field fitted; inherited quasar-prior and tail limitations remain.")
    adaptation = BackgroundAdaptation(bl.bundle_id, bl.selection.identity, outputs, meta)
    adaptation.save(args.output)
    report["adaptation_id"] = adaptation.identity
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved {args.output} and {args.report}", flush=True)


if __name__ == "__main__":
    main()
