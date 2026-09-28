#!/usr/bin/env python
"""Check spatial background predictions on reserved field cones."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np



def field_summary(fit_report, draws, seed):
    result = {}
    rng = np.random.default_rng(seed)
    for h, record in fit_report["hemispheres"].items():
        result[h] = {}
        for role, r in record["validations"].items():
            if "spatial" not in r:
                result[h][role] = r
                continue
            a, b = r["pooled"], r["spatial"]
            cells = np.array([x["cell"] for x in a]); unique = np.unique(cells)
            delta = np.array([y["log_score_sum"] - x["log_score_sum"] for x, y in zip(a, b)])
            n = np.array([x["n"] for x in a])
            vals = []
            for _ in range(draws):
                take = np.concatenate([np.flatnonzero(cells == c) for c in rng.choice(unique, len(unique), replace=True)])
                vals.append(delta[take].sum() / n[take].sum())
            modes = {}
            for mode, rows in (("pooled", a), ("spatial", b)):
                observed = np.array([x["observed"] for x in rows])
                predicted = np.array([x["predicted"] for x in rows])
                modes[mode] = dict(total_observed=int(observed.sum()), total_predicted=float(predicted.sum()),
                    observed_over_predicted=float(observed.sum() / predicted.sum()),
                    mean_absolute_cone_fractional_error=float(np.mean(abs(observed.sum(1) / predicted.sum(1) - 1))))
            result[h][role] = dict(delta_log_score=float(delta.sum() / n.sum()),
                ci95=np.percentile(vals, [2.5, 97.5]).tolist(), cells=len(unique), cones=len(a), counts=modes)
    return result


def main():
    from qso_pcolor.spatial import BackgroundAdaptation
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adaptation", type=Path, required=True)
    ap.add_argument("--fit-report", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    fit = json.loads(args.fit_report.read_text())
    cfg = fit["config"]
    adaptation = BackgroundAdaptation.load(args.adaptation)
    if fit["adaptation_id"] != adaptation.identity:
        raise ValueError("validation inputs refer to different fits")
    fields = field_summary(fit, cfg["validation"]["bootstrap_draws"], cfg["seed"])
    checks = dict(southern_field_improvement=fields["south"]["test"]["ci95"][0] > 0,
        spatial_colour_models_present=all(len(p["background"].local) > 0 for p in adaptation.parts.values()),
        spatial_count_models_present=all(len(p["density"].area) > 1 for p in adaptation.parts.values()),
        all_weight_fits_converged=all(r["weight_converged"] == r["weight_fits"]
                                     for r in fit["hemispheres"].values()))
    geometry = fit["geometry"]
    retained = set(geometry["retained_cones"])
    excluded = {r["cone"] for r in geometry["rejected_cones"]}
    checks["boundary_cones_excluded"] = not bool(retained & excluded)
    for h, r in fit["hemispheres"].items():
        test = r["validations"]["test"].get("spatial", [])
        checks[f"{h}_spatial_fit_test_disjoint"] = not bool(set(r["fit_cones"]) & {x["cone"] for x in test})
    report = dict(source_bundle=adaptation.source_bundle_id, adaptation_id=adaptation.identity,
        field=fields, geometry=geometry, config=cfg, verdict=dict(spatial_restoration_pass=all(checks.values()), checks=checks,
        scope="Spatial colour and surface-density restoration, evaluated on field cones. No claim of full probability calibration or resolution of inherited prior/tail limitations."))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report["verdict"], indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
