#!/usr/bin/env python
"""Validate the PSF recovery on reserved objects and arbitrary band masks.

No close-pair catalogue is used. Mathematical subset checks are distinguished
from predictive checks on the original reserved field cones and quasar blocks.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from qso_pcolor.multisurvey import MultiSurveyModel, _ConditionalQSO
from qso_pcolor.multisurvey_baseline import PSFMultiSurveyBaseline
from qso_pcolor.multisurvey_data import Photometry, SURVEYS
from qso_pcolor.qso_model import RedshiftMatch
from qso_pcolor.score import BlendPolicy


def reserved_densities(model, data, kind, chunk):
    selected = np.flatnonzero(data["held"])
    if kind == "quasars":
        selected = selected[model.qso.in_support(data["zspec"][selected])]
    result = np.empty(len(selected))
    for lo in range(0, len(selected), chunk):
        rows = selected[lo:lo + chunk]
        phot = Photometry(data["flux"][rows], data["variance"][rows], tuple(data["bands"]))
        f = model.transform(phot)
        priority = np.array([f.labels.index(label) for label in model.reference_priority])
        anchors = priority[np.argmax(f.observed[:, priority], axis=1)]
        for a in np.unique(anchors):
            local = np.flatnonzero(anchors == a)
            features = f.subset(local)
            if kind == "quasars":
                qso = _ConditionalQSO(model.qso, int(a))
                result[lo + local] = qso.log_p_colour_given_z(features.x, features.cov,
                    data["zspec"][rows[local]], observed=features.observed).diagonal()
            else:
                result[lo + local] = model.background_log_prob(features.x, features.cov,
                    features.observed, int(a), l_deg=data["l"][rows[local]], b_deg=data["b"][rows[local]])
    return selected, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/multisurvey_psf_recovery.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/VALIDATION_multisurvey_psf_2026-09-28.json"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    candidate = MultiSurveyModel.load(cfg["output"])
    root = Path(candidate.meta["training_cache"])
    baseline = PSFMultiSurveyBaseline.load(root / "bundle")
    model = baseline.model
    source = MultiSurveyModel.load(cfg["source_model"])
    report = dict(bundle_id=baseline.manifest["bundle_id"], close_pairs_used=False,
                  population="PSF", bands=len(model.transform.bands), reserved={}, checks={})
    datasets = {}
    for kind in ("quasars", "background"):
        data = dict(np.load(root / f"{kind}.npz")); datasets[kind] = data
        rows, fitted = reserved_densities(model, data, kind, cfg["validation"]["chunk_size"])
        old_rows, previous = reserved_densities(source, data, kind, cfg["validation"]["chunk_size"])
        assert np.array_equal(rows, old_rows)
        delta = fitted - previous
        report["reserved"][kind] = dict(n=len(rows), all_finite=bool(np.isfinite(fitted).all()),
            mean_log_density=float(fitted.mean()), source_mean_log_density=float(previous.mean()),
            mean_gain_nats=float(delta.mean()), median_gain_nats=float(np.median(delta)))
        print(kind, report["reserved"][kind], flush=True)

    bands = model.transform.bands
    d = len(bands)
    mixtures = model.qso.mixtures[len(model.qso.mixtures) // 2]
    centre = np.average(mixtures.means, axis=0, weights=mixtures.weights)
    soften = model.transform.softening
    flux = 2 * soften * np.sinh((22.5 - centre) / (2.5 / np.log(10.)) - np.log(soften))
    # Mathematical interface checks, not a claim that every pair co-occurs in data.
    subsets = [(j,) for j in range(d)] + list(combinations(range(d), 2))
    rng = np.random.default_rng(cfg["seed"])
    subsets += [tuple(sorted(rng.choice(d, size, replace=False))) for size in range(3, d + 1)]
    f = np.full((len(subsets), d), np.nan); v = np.full_like(f, np.inf)
    for i, subset in enumerate(subsets):
        f[i, list(subset)] = flux[list(subset)]
        v[i, list(subset)] = soften[list(subset)] ** 2
    phot = Photometry(f, v, bands)
    raw = model.score(phot, z_primary=2., l_deg=180., b_deg=45.,
                      match=RedshiftMatch(half_width_kms=cfg["validation"]["window_kms"]))
    report["checks"]["band_masks"] = dict(singletons=d, pairs=d * (d - 1) // 2,
        larger_subsets=d - 2, all_finite_evidence=bool(all(np.isfinite(s.log_bayes_factor_qz_bkg) for s in raw)),
        singleton_log_bf_zero=bool(np.allclose([s.log_bayes_factor_qz_bkg for s in raw[:d]], 0.)))
    print("Arbitrary band masks checked", flush=True)

    tested = 0
    q = datasets["quasars"]
    full = Photometry(q["flux"], q["variance"], bands)
    for size in range(1, len(SURVEYS) + 1):
        for surveys in combinations(SURVEYS, size):
            p = full.keep_surveys(surveys)
            available = np.flatnonzero(q["held"] & p.observed.any(axis=1) & model.qso.in_support(q["zspec"]))
            if not len(available):
                continue
            i = int(available[len(available) // 2])
            scores, decision = baseline.score(p.subset([i]), morphology=["PSF"],
                z_primary=q["zspec"][i], l_deg=q["l"][i], b_deg=q["b"][i],
                match=RedshiftMatch(half_width_kms=cfg["validation"]["window_kms"]),
                blend_policy=BlendPolicy(**cfg["validation"]["blend_policy"]),
                ood_flag_sigma=cfg["validation"]["ood_flag_sigma"],
                separation_arcsec=cfg["validation"]["test_separation_arcsec"],
                fracflux=cfg["validation"]["test_fracflux"])
            score = scores[0]
            assert score is not None and np.isfinite(score.log_bayes_factor_qz_bkg)
            if decision["eligible"][0]:
                assert np.isclose(score.p_sameq, np.exp(score.log_r_per_unit_z) * score.dz_match_eff)
            tested += 1
    report["checks"]["survey_subsets_public_interface"] = tested
    report["checks"]["priors_available"] = len(baseline.priors)
    report["checks"]["spatial_nside"] = model.spatial_background.nside
    report["numerical_fits"] = dict(qso_converged=sum(r["converged"] for r in model.qso.meta["per_slice"]),
        qso_slices=len(model.qso.mixtures), field_converged=model.meta["background"]["converged"],
        note="Iteration caps are reported, not treated as an accuracy measurement.")
    report["functional_pass"] = bool(tested == 127 and report["checks"]["band_masks"]["all_finite_evidence"]
        and report["checks"]["band_masks"]["singleton_log_bf_zero"]
        and all(r["all_finite"] for r in report["reserved"].values()))
    report["limitations"] = ["This is functional and held-out density validation, not probability calibration.",
        "Spatial pooling settings inherited from the Legacy PSF restoration; only 24 multi-survey field cones are available.",
        "Sparse population priors are flagged; the original completeness and inferred survey-footprint assumptions remain."]
    args.report.write_text(json.dumps(report, indent=2))
    if not report["functional_pass"]:
        raise SystemExit("Functional validation failed")
    print(f"Validation saved: {args.report}", flush=True)


if __name__ == "__main__":
    main()
