#!/usr/bin/env python
"""Colour outliers, with and without the unmodelled hypothesis: numbers and figure.

Two measurements, both on the shipped models:

1. **Synthetic sweep.** The README candidate (z0 = 1.8, r ~ 21) has one band
   flux, or a pair, multiplied by s = 1 ... 32 at fixed inverse variance, which
   walks it away from both colour loci. At each step the scorer is run with and
   without ``models/outlier_south.json``.
2. **Validation tails.** Reads the two pair-validation reports written by
   ``validate_pairs.py`` (three and four hypotheses) and tabulates, in bins of
   the distance to the nearer model, how many non-quasars and quasars are
   called quasars.

    python scripts/validate_pairs.py --max-fracflux 0.2 --ood-flag-sigma 4 --no-figure \\
        --out data/pair_validation_results_3hyp_20260926.npz \\
        --report data/pair_validation_report_3hyp_20260926.json
    python scripts/validate_pairs.py ... --outlier models/outlier_south.json \\
        --out data/pair_validation_results_outlier_20260926.npz \\
        --report data/pair_validation_report_outlier_20260926.json
    python scripts/outlier_analysis.py

Writes ``data/outlier_analysis.json`` and ``plots/method/fig11_outlier_tails.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

BANDS = ("g", "r", "z", "w1", "w2")
DIRECTIONS = {            # which of g, r, z, W1, W2 is multiplied by s
    "g": (1, 0, 0, 0, 0),
    "z": (0, 0, 1, 0, 0),
    "W2": (0, 0, 0, 0, 1),
    "W1+W2": (0, 0, 0, 1, 1),
}
FACTORS = (1, 2, 4, 8, 16, 32)


def p_quasar(r) -> float:
    lam = [r.log_lambda_sameq, r.log_lambda_fieldq, r.log_lambda_bkg]
    every = lam + ([r.log_lambda_out] if np.isfinite(r.log_lambda_out) else [])
    return float(np.exp(np.logaddexp(lam[0], lam[1]) - logsumexp(every)))


def sweep():
    from qso_pcolor.background import BackgroundColourModel
    from qso_pcolor.features import RelativeFluxTransform, deredden
    from qso_pcolor.outlier import OutlierModel
    from qso_pcolor.priors import BackgroundSurfaceDensity, GridQSOPrior
    from qso_pcolor.qso_model import RedshiftMatch, SlicedColourRedshiftModel
    from qso_pcolor.score import BlendPolicy, score_candidates

    qso = SlicedColourRedshiftModel.load("models/qso_south_full.json")
    kw = dict(
        z_primary=np.array([1.8]), l_deg=np.array([276.337]), b_deg=np.array([60.189]),
        qso_model=qso,
        background_model=BackgroundColourModel.load("models/background_south_global.json"),
        background_density=BackgroundSurfaceDensity.load(
            "models/background_density_south_global.json"),
        qso_prior=GridQSOPrior.load("models/sigma_q_south.json"),
        match=RedshiftMatch(half_width_kms=2000.0),
        blend_policy=BlendPolicy(min_separation_arcsec=3.0, max_fracflux=0.2),
        separation_arcsec=np.array([6.0]), fracflux=np.array([0.05]))
    um = OutlierModel.load("models/outlier_south.json")
    tr = RelativeFluxTransform(reference_band="r")
    f0, v0 = deredden(np.array([[1.9, 2.6, 3.1, 11.0, 14.0]]),
                      np.array([[120.0, 150.0, 60.0, 8.0, 3.0]]),
                      np.array([[0.97, 0.98, 0.99, 1.0, 1.0]]))
    out = {}
    for name, dvec in DIRECTIONS.items():
        rows = []
        for s in FACTORS:
            fs = tr(f0 * np.where(np.array(dvec) > 0, s, 1.0), v0, BANDS)
            a = score_candidates(fs, **kw)[0]
            b = score_candidates(fs, outlier_model=um, **kw)[0]
            rows.append({
                "s": s, "d_qso_any_z": a.qso_ood_sigma_any_z, "d_bkg": a.bkg_ood_sigma,
                "three": {"ln_bf": a.log_bayes_factor_qz_bkg, "p_quasar": p_quasar(a),
                          "ln_r": a.log_r_per_unit_z},
                "four": {"ln_bf": b.log_bayes_factor_qz_bkg, "p_quasar": p_quasar(b),
                         "ln_r": b.log_r_per_unit_z, "p_outlier": b.p_outlier},
            })
            print(f"{name:6s} s={s:3d}  dQ {a.qso_ood_sigma_any_z:5.1f} dB {a.bkg_ood_sigma:5.1f}"
                  f" | 3-hyp lnBF {a.log_bayes_factor_qz_bkg:+8.1f} P(Q) {p_quasar(a):.3f}"
                  f" lnR {a.log_r_per_unit_z:+8.2f}"
                  f" | 4-hyp lnBF {b.log_bayes_factor_qz_bkg:+8.1f} P(Q) {p_quasar(b):.3f}"
                  f" lnR {b.log_r_per_unit_z:+8.2f} P(U) {b.p_outlier:.3f}")
        out[name] = rows
    return out


def figure(sw, tails3, tails4):
    import matplotlib.pyplot as plt

    from qso_pcolor.plotting import SERIES, save_figure, use_paper_style

    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, name in zip(axes[:2], ("g", "z")):
        rows = sw[name]
        s = [r["s"] for r in rows]
        ax.plot(s, [r["three"]["p_quasar"] for r in rows], "o-", color=SERIES["field_q"],
                ms=3, lw=1, label="three hypotheses")
        ax.plot(s, [r["four"]["p_quasar"] for r in rows], "s-", color=SERIES["same_z"],
                ms=3, lw=1, label="with unmodelled term")
        ax.plot(s, [r["four"]["p_outlier"] for r in rows], "s--", color=SERIES["neutral"],
                ms=3, lw=1, label="P(unmodelled)")
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(f"{name}-band flux multiplier $s$")
        for r in rows:
            ax.annotate(f"{min(r['d_qso_any_z'], r['d_bkg']):.0f}", (r["s"], 1.03),
                        fontsize=6, ha="center", va="bottom", color=SERIES["neutral"],
                        annotation_clip=False)
    axes[0].set_ylabel("P(quasar at any $z$)")
    axes[0].legend(fontsize=6, loc="center left", frameon=False)

    ax = axes[2]
    lab = [f"{r['sigma'][0]:g}–{r['sigma'][1]:g}" if np.isfinite(r["sigma"][1])
           else f">{r['sigma'][0]:g}" for r in tails3["rows"]]
    x = np.arange(len(lab))

    def frac(rows):
        return np.array([r["non_qso_called_quasar"] / r["n_non_qso"] if r["n_non_qso"] else np.nan
                         for r in rows])
    ax.bar(x - 0.2, frac(tails3["rows"]), 0.4, color=SERIES["field_q"], label="three")
    ax.bar(x + 0.2, frac(tails4["rows"]), 0.4, color=SERIES["same_z"], label="four")
    for xi, r in zip(x, tails3["rows"]):
        ax.annotate(str(r["n_non_qso"]), (xi, 1.0), fontsize=6, ha="center", va="bottom",
                    color=SERIES["neutral"], annotation_clip=False)
    ax.set_xticks(x, lab, fontsize=6)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"distance to nearer model ($\sigma$)")
    ax.set_ylabel("non-quasars called quasar")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    return save_figure(fig, "method/fig11_outlier_tails")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-3", type=Path,
                    default=Path("data/pair_validation_report_3hyp_20260926.json"))
    ap.add_argument("--report-4", type=Path,
                    default=Path("data/pair_validation_report_outlier_20260926.json"))
    ap.add_argument("--out", type=Path, default=Path("data/outlier_analysis.json"))
    args = ap.parse_args()
    sw = sweep()
    r3 = json.loads(args.report_3.read_text())
    r4 = json.loads(args.report_4.read_text())
    summary = {"sweep": sw, "tails_three": r3["tails"], "tails_four": r4["tails"],
               "auc_three": r3["auc"], "auc_four": r4["auc"],
               "hard_negatives_three": r3["hard_negatives"],
               "hard_negatives_four": r4["hard_negatives"],
               "non_qso_by_spectype_three": r3["non_qso_by_spectype"],
               "non_qso_by_spectype_four": r4["non_qso_by_spectype"]}
    args.out.write_text(json.dumps(summary, indent=1, default=float))
    print(f"figure: {figure(sw, r3['tails'], r4['tails'])}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
