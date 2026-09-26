#!/usr/bin/env python
"""Fit the unmodelled hypothesis for the multi-survey model.

Same construction as ``fit_outlier_model.py`` for the original model, in the
multi-survey coordinates (native luptitudes, conditioned on a reference band):

* shape: one joint Gaussian centred on the background mixture, covariance
  kappa^2 E with E an envelope over every quasar-slice component, every
  background component and the embedded components of each background
  marginal, so the conditional density dominates every conditional tail;
* eta: per reference band and magnitude bin, by EM, against the field density
  the scorer actually uses (joint background, or a declared marginal where it
  covers the object's bands).

Calibration objects (set A) are training-field sources the background fit did
not use -- it drew ``max_background_fit`` of them, reproduced here from the
same seed. The check set (B) is the reserved fields. kappa is chosen on A.

    python scripts/fit_multisurvey_outlier.py --config configs/multisurvey_lsw.json \\
        --model models/multisurvey_lsw.json --out models/multisurvey_lsw_outlier.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="sample configuration")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-background-fit", type=int, default=20000,
                    help="must equal the value used in training")
    ap.add_argument("--kappa-factors", type=float, nargs="+",
                    default=[1.02, 1.25, 1.5, 2.0, 3.0, 5.0])
    ap.add_argument("--n-mag-bins", type=int, default=3,
                    help="quantile bins of the reference luptitude per band")
    ap.add_argument("--min-per-bin", type=int, default=500)
    args = ap.parse_args()

    from train_multisurvey_model import sample_patterns

    from qso_pcolor.multisurvey import MultiSurveyModel, MultiSurveyOutlier, conditional_log_prob
    from qso_pcolor.multisurvey_data import Photometry
    from qso_pcolor.outlier import envelope_covariance, fit_outlier_fraction, \
        min_dominant_kappa, mixture_moments

    cfg = json.loads(args.config.read_text())
    model = MultiSurveyModel.load(args.model)
    bands = model.transform.bands
    b = dict(np.load(Path(cfg["data_dir"]) / "background.npz"))
    if tuple(str(x) for x in b["bands"]) != bands:
        raise SystemExit("background sample and model band layouts differ")
    bp = Photometry(b["flux"], b["variance"], bands)
    eligible = ~b["held"] & (bp.observed.sum(1) >= cfg["min_bands"])
    rng = np.random.default_rng(cfg["seed"] + 10)          # as in training
    chosen, _ = sample_patterns(bp.observed, eligible, args.max_background_fit,
                                cfg["min_band_training"], rng)
    set_a = eligible.copy()
    set_a[chosen] = False
    set_b = b["held"] & (bp.observed.sum(1) >= cfg["min_bands"])
    print(f"calibration A: {set_a.sum():,} unused training-field sources; "
          f"check B: {set_b.sum():,} in reserved fields")

    feats = model.transform(bp)
    order = np.array([bands.index(x) for x in model.reference_priority])
    anchor = order[np.argmax(feats.observed[:, order], axis=1)]
    use = set_a | set_b
    log_pb = np.full(len(anchor), np.nan)
    for a in np.unique(anchor[use]):
        s = use & (anchor == a)
        log_pb[s] = model.background_log_prob(feats.x[s], feats.cov[s], feats.observed[s], int(a))

    # envelope over every component the scorer compares against
    extra = []
    for m in model.background_marginals:
        idx = np.array([bands.index(x) for x in m.labels])
        for v in m.covs:
            e = np.zeros((len(bands), len(bands)))
            e[np.ix_(idx, idx)] = v
            extra.append(e)
    check = list(model.qso.mixtures) + [model.background]
    mean, base = mixture_moments([model.background])
    env = envelope_covariance(base, check, np.array(extra) if extra else None)
    kmin = min_dominant_kappa(env, check, np.array(extra) if extra else None)
    print(f"envelope: kappa_min = {kmin:.3f} over "
          f"{sum(m.n_components for m in check) + len(extra)} components")

    from qso_pcolor.gaussmix import GaussianMixture

    rows = []
    for fac in args.kappa_factors:
        kappa = fac * kmin
        mix = GaussianMixture(np.ones(1), mean[None], (kappa**2 * env)[None], labels=bands)
        log_pu = np.full(len(anchor), np.nan)
        for a in np.unique(anchor[use]):
            s = use & (anchor == a)
            log_pu[s] = conditional_log_prob(mix, feats.x[s], feats.cov[s], feats.observed[s],
                                             int(a))
        fractions, gain = {}, np.zeros(len(anchor))
        for a in np.unique(anchor[set_a]):
            label = bands[a]
            sa = set_a & (anchor == a)
            if sa.sum() < args.min_per_bin * args.n_mag_bins:
                continue
            u = feats.x[sa, a]
            inner = np.quantile(u, np.linspace(0, 1, args.n_mag_bins + 1)[1:-1])
            edges = np.concatenate([[u.min()], inner, [u.max()]])
            ib = np.clip(np.digitize(u, edges) - 1, 0, args.n_mag_bins - 1)
            eta = np.array([fit_outlier_fraction(log_pb[sa][ib == j], log_pu[sa][ib == j])
                            for j in range(args.n_mag_bins)])
            fractions[label] = (edges, eta)
            s = use & (anchor == a)
            e = eta[np.clip(np.digitize(feats.x[s, a], edges) - 1, 0, args.n_mag_bins - 1)]
            with np.errstate(divide="ignore"):
                gain[s] = np.logaddexp(np.log1p(-e) + log_pb[s], np.log(e) + log_pu[s]) - log_pb[s]
        # pooled fallback for reference bands with too few calibration objects
        pooled = fit_outlier_fraction(log_pb[set_a], log_pu[set_a])
        fractions["*"] = (np.array([-99.0, 99.0]), np.array([pooled]))   # all magnitudes
        rest = use & ~np.isin(anchor, [bands.index(k) for k in fractions if k != "*"])
        with np.errstate(divide="ignore"):
            gain[rest] = np.logaddexp(np.log1p(-pooled) + log_pb[rest],
                                      np.log(pooled) + log_pu[rest]) - log_pb[rest]
        covered = use
        row = {"kappa_factor": fac, "kappa": kappa,
               "gain_A": float(gain[set_a & covered].mean()),
               "gain_B": float(gain[set_b & covered].mean()),
               "n_B_covered": int((set_b & covered).sum()),
               "fractions": {k: [e.tolist(), f.tolist()] for k, (e, f) in fractions.items()}}
        rows.append(row)
        print(f"  kappa = {fac:4.2f} x kappa_min: gain A {row['gain_A']:+.5f}  "
              f"B {row['gain_B']:+.5f} nats/obj over {len(fractions)} reference bands")
    best = max(rows, key=lambda r: r["gain_A"])
    print(f"chosen on A: {best['kappa_factor']} x kappa_min; gain on B {best['gain_B']:+.5f}")
    for k, (e, f) in best["fractions"].items():
        print(f"    {k:22s} eta = " + " ".join(f"{x:.1e}" for x in f))
    out = MultiSurveyOutlier(
        mean, best["kappa"] ** 2 * env, best["kappa"], kmin, bands, model.transform_id,
        {k: tuple(v) for k, v in best["fractions"].items()},
        meta={"built": time.strftime("%Y-%m-%d"), "by": "scripts/fit_multisurvey_outlier.py",
              "model": str(args.model), "model_run_id": model.meta.get("run_id"),
              "set_A": "training-field sources unused by the background fit",
              "set_B": "reserved background fields", "n_A": int(set_a.sum()),
              "n_B": int(set_b.sum()), "kappa_scan": [{k: v for k, v in r.items()
                                                       if k != "fractions"} for r in rows],
              "chosen_by": "max held-out log-likelihood on set A"})
    out.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
