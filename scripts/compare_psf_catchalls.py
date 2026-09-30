#!/usr/bin/env python
"""Fit both joint catch-alls with pooled weights; select on training-cone CV.

The PSF QSO/stars/spatial fits and population priors are unchanged. Original
reserved cones are evaluated only after each family's settings are frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from qso_pcolor import PSFMultiSurveyBaseline, Photometry
from qso_pcolor.multisurvey import MultiSurveyOutlier
from qso_pcolor.outlier import (envelope_covariance, min_dominant_kappa,
                              mixture_moments, fit_pooled_outlier_fraction)


def fit_fractions(lb, lu, anchors, magnitude, bands, *, n_bins, strength, config):
    """Pooled Beta-MAP weights, with quantile edges fitted only on input rows."""
    pooled = fit_pooled_outlier_fraction(lb, lu,
        prior_mean=config["global_prior_mean"], prior_strength=config["global_prior_strength"])
    fractions = {"*": (np.array([magnitude.min(), magnitude.max()]), np.array([pooled]))}
    for a in np.unique(anchors):
        use = anchors == a
        bins = min(n_bins, int(use.sum()) // config["minimum_bin_count"])
        if bins < 1:
            continue
        edges = np.unique(np.quantile(magnitude[use], np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            continue
        index = np.clip(np.digitize(magnitude[use], edges) - 1, 0, len(edges) - 2)
        eta = [fit_pooled_outlier_fraction(lb[use][index == j], lu[use][index == j],
                    prior_mean=pooled, prior_strength=strength) for j in range(len(edges)-1)]
        fractions[bands[a]] = edges, np.array(eta)
    return fractions


def fractions_at(fractions, anchors, magnitude, bands):
    eta = np.empty(len(anchors))
    for a in np.unique(anchors):
        use = anchors == a
        edges, values = fractions.get(bands[a], fractions["*"])
        eta[use] = values[np.clip(np.digitize(magnitude[use], edges)-1, 0, len(values)-1)]
    return eta


def field_gain(lb, lu, eta):
    with np.errstate(divide="ignore"):
        return np.logaddexp(np.log1p(-eta) + lb, np.log(eta) + lu) - lb


def evaluate_densities(model, data, rows, outliers, *, chunk, background=True):
    """Evaluate the deployed spatial conditional densities, in mag^-(N-1)."""
    bands = model.transform.bands
    result = {"anchor": np.empty(len(rows), int), "magnitude": np.empty(len(rows)),
              **{name: np.empty(len(rows)) for name in outliers}}
    if background:
        result["background"] = np.empty(len(rows))
    for lo in range(0, len(rows), chunk):
        r = rows[lo:lo+chunk]
        photometry = Photometry(data["flux"][r], data["variance"][r], bands)
        f = model.transform(photometry)
        anchors = model.reference_indices(photometry)
        result["anchor"][lo:lo+len(r)] = anchors
        result["magnitude"][lo:lo+len(r)] = f.x[np.arange(len(r)), anchors]
        for a in np.unique(anchors):
            local = np.flatnonzero(anchors == a)
            fs = f.subset(local)
            target = lo + local
            if background:
                result["background"][target] = model.background_log_prob(
                    fs.x, fs.cov, fs.observed, int(a), l_deg=data["l"][r[local]], b_deg=data["b"][r[local]])
            for name, outlier in outliers.items():
                result[name][target] = outlier.conditional(int(a), bands[a], model.qso.system).log_prob(
                    fs.x, fs.cov, observed=fs.observed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/psf_catchall_comparison.json"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    baseline = PSFMultiSurveyBaseline.load(cfg["baseline"])
    model = baseline.model
    root = Path(cfg["work"]); root.mkdir(parents=True, exist_ok=True)
    sample = Path(model.meta["training_cache"]) / "background.npz"
    data = dict(np.load(sample))
    signature = hashlib.sha256(json.dumps(dict(config=cfg,
        files=baseline.manifest["files"], sample=hashlib.sha256(sample.read_bytes()).hexdigest(),
        code={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
              (__file__, "src/qso_pcolor/outlier.py", "src/qso_pcolor/multisurvey.py")}),
        sort_keys=True).encode()).hexdigest()[:16]
    cache = root / signature; cache.mkdir(exist_ok=True)
    train = ~data["held"].copy()
    train[np.asarray(model.meta["background"]["fitted_rows"])] = False
    rows = np.flatnonzero(train)
    held = np.flatnonzero(data["held"])
    bands = model.transform.bands
    mean, base = mixture_moments([model.background])
    check = list(model.qso.mixtures) + [model.background]
    envelope = envelope_covariance(base, check)
    kmin = min_dominant_kappa(envelope, check)
    candidates = {}
    for factor in cfg["gaussian_factors"]:
        kappa = factor * kmin
        name = f"gaussian_{factor}"
        candidates[name] = MultiSurveyOutlier(mean, kappa**2 * envelope, kappa, kmin,
            bands, model.transform_id, {"*": ([0., 1.], [.5])}, family="gaussian")
    for nu in cfg["student_t_nu"]:
        for factor in cfg["student_t_factors"]:
            name = f"student_t_{nu}_{factor}"
            candidates[name] = MultiSurveyOutlier(mean, factor**2 * base, factor, 0.,
                bands, model.transform_id, {"*": ([0., 1.], [.5])}, family="student_t",
                nu=nu, noise=cfg["student_t_noise"])
    common_path = cache / "common.npz"
    if not common_path.exists():
        np.savez(common_path, **evaluate_densities(model, data, rows, {}, chunk=cfg["chunk_size"]))
    common = dict(np.load(common_path))
    lb, anchor, mag = (common[k] for k in ("background", "anchor", "magnitude"))
    rng = np.random.default_rng(cfg["seed"])
    fields = rng.permutation(np.unique(data["field"][rows]))
    fold = np.empty(len(rows), int)
    for j, field in enumerate(fields):
        fold[data["field"][rows] == field] = j % cfg["folds"]
    trials = []
    logs = {}
    print(f"Fit/CV: {len(rows)} objects, {len(fields)} cones; untouched test: {len(held)}", flush=True)
    for name, candidate in candidates.items():
        path = cache / f"{name}.npz"
        if not path.exists():
            np.savez(path, **evaluate_densities(model, data, rows, {name: candidate},
                chunk=cfg["chunk_size"], background=False))
        lu = np.load(path)[name]; logs[name] = lu
        for bins in cfg["magnitude_bins"]:
            for strength in cfg["pooling_strengths"]:
                gain = np.empty(len(rows))
                for k in range(cfg["folds"]):
                    fit, val = fold != k, fold == k
                    frac = fit_fractions(lb[fit], lu[fit], anchor[fit], mag[fit], bands,
                        n_bins=bins, strength=strength, config=cfg)
                    eta = fractions_at(frac, anchor[val], mag[val], bands)
                    gain[val] = field_gain(lb[val], lu[val], eta)
                trials.append(dict(name=name, family=candidate.family, bins=bins, strength=strength,
                    mean_cv_gain=float(gain.mean()),
                    fold_gain=[float(gain[fold==k].mean()) for k in range(cfg["folds"])]))
        best = max((t for t in trials if t["name"] == name), key=lambda t:t["mean_cv_gain"])
        print(name, best, flush=True)
    winners, selected = {}, {}
    for family in ("gaussian", "student_t"):
        best = max((t for t in trials if t["family"] == family), key=lambda t:t["mean_cv_gain"])
        selected[family] = best
        outlier = candidates[best["name"]]
        outlier.fractions = fit_fractions(lb, logs[best["name"]], anchor, mag, bands,
            n_bins=best["bins"], strength=best["strength"], config=cfg)
        outlier.meta = dict(model_run_id=model.meta["run_id"], population="psf",
            source_bundle=baseline.manifest["bundle_id"], fit_signature=signature,
            config=cfg, selected=best, n_fit=len(rows), cv_fields=fields.tolist(),
            heldout_fields=np.unique(data["field"][held]).tolist(),
            split_note="Catch-all CV excludes held-out cones and Gaussian-shape fit rows; spatial weights used training cones.")
        outlier.save(root / f"{family}.json")
        winners[family] = outlier
    # Freeze family settings before using the original test cones.
    result = evaluate_densities(model, data, held, dict(winners, previous=baseline.outlier),
                                chunk=cfg["chunk_size"])
    test = {}
    gains = {}
    for name, outlier in dict(winners, previous=baseline.outlier).items():
        eta = fractions_at(outlier.fractions, result["anchor"], result["magnitude"], bands)
        gain = field_gain(result["background"], result[name], eta); gains[name] = gain
        test[name] = dict(mean_gain=float(gain.mean()), median_gain=float(np.median(gain)),
            eta_min=float(eta.min()), eta_max=float(eta.max()),
            by_cone={str(f): float(gain[data["field"][held] == f].mean())
                     for f in np.unique(data["field"][held])})
    report = dict(source_bundle=baseline.manifest["bundle_id"], signature=signature,
        fit_rows=len(rows), heldout_rows=len(held), selected=selected, trials=trials,
        heldout=test, field_density_bin_counts={k: len(p[1].mag_edges)-1 for k,p in baseline.priors.items()},
        stellar_colour_magnitude_dependence="Continuous conditioning of one joint Gaussian mixture; no magnitude bins.",
        history=["ddb1892", "f09de39", "406cabf"],
        limitations=["Six original test cones; no probability calibration.",
                     "Catch-all CV rows were not used for component shapes; spatial weights were fitted on their training cones."])
    (root / "fit_report.json").write_text(json.dumps(report, indent=2))
    np.savez(cache / "heldout.npz", **result, **{f"gain_{k}": v for k,v in gains.items()})
    print("Reserved field results:", test, flush=True)


if __name__ == "__main__":
    main()
