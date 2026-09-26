#!/usr/bin/env python
"""Figures for the method note's validation and outlier sections, one model.

Reads the saved outputs of ``validate_pairs_multisurvey.py`` (with and without
the unmodelled term) and scores a synthetic sweep with the shipped model; no
fitting.

    python scripts/validate_pairs_multisurvey.py --max-fracflux 0.2 \\
        --ms-outlier models/multisurvey_outlier.json \\
        --out data/pair_validation_multisurvey_lsw.json --scores data/pair_validation_multisurvey_lsw.npz
    python scripts/validate_pairs_multisurvey.py --max-fracflux 0.2 \\
        --out data/pair_validation_multisurvey_lsw_noU.json --scores data/pair_validation_multisurvey_lsw_noU.npz
    python scripts/make_validation_figures.py

Writes plots/validation/pair_validation_multisurvey.png,
plots/method/fig_outliers_multisurvey.png and data/outlier_sweep_multisurvey.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).parent))
from validate_pairs import roc_curve  # noqa: E402

LEGACY = tuple(f"decals_dr9_south:{b}" for b in ("g", "r", "z", "w1", "w2"))


def p_quasar(r) -> float:
    lam = [r.log_lambda_sameq, r.log_lambda_fieldq, r.log_lambda_bkg]
    every = lam + ([r.log_lambda_out] if np.isfinite(r.log_lambda_out) else [])
    return float(np.exp(np.logaddexp(lam[0], lam[1]) - logsumexp(every)))


def sweep():
    from qso_pcolor.multisurvey import MultiSurveyModel, MultiSurveyOutlier, load_priors
    from qso_pcolor.multisurvey_data import Photometry
    from qso_pcolor.qso_model import RedshiftMatch

    model = MultiSurveyModel.load("models/multisurvey.json")
    priors = load_priors("models/multisurvey_priors.json", model)
    out = MultiSurveyOutlier.load("models/multisurvey_outlier.json")
    flux0 = np.array([1.9, 2.6, 3.1, 11.0, 14.0])
    var = 1.0 / np.array([[120.0, 150.0, 60.0, 8.0, 3.0]])
    kw = dict(z_primary=np.array([1.8]), match=RedshiftMatch(half_width_kms=2000.0), min_bands=2,
              l_deg=np.array([276.337]), b_deg=np.array([60.189]), priors=priors)
    res = {}
    for name, band in (("g", 0), ("z", 2), ("W2", 4)):
        rows = []
        for s in (1, 2, 4, 8, 16, 32):
            f = flux0.copy()
            f[band] *= s
            ph = Photometry(f[None], var, LEGACY)
            a = model.score(ph, **kw)[0]
            b = model.score(ph, outlier=out, **kw)[0]
            rows.append(dict(s=s, d=float(min(a.qso_ood_sigma_any_z, a.bkg_ood_sigma)),
                             d_q=a.qso_ood_sigma_any_z, d_b=a.bkg_ood_sigma,
                             three=dict(ln_bf=a.log_bayes_factor_qz_bkg, p_q=p_quasar(a),
                                        ln_r=a.log_r_per_unit_z),
                             four=dict(ln_bf=b.log_bayes_factor_qz_bkg, p_q=p_quasar(b),
                                       ln_r=b.log_r_per_unit_z, p_u=b.p_outlier)))
            print(f"{name:3s} s={s:3d} dQ {a.qso_ood_sigma_any_z:5.1f} dB {a.bkg_ood_sigma:5.1f} | "
                  f"3: lnBF {a.log_bayes_factor_qz_bkg:+8.1f} P(Q) {p_quasar(a):.3f} lnR "
                  f"{a.log_r_per_unit_z:+8.2f} | 4: lnBF {b.log_bayes_factor_qz_bkg:+8.1f} "
                  f"P(Q) {p_quasar(b):.3f} lnR {b.log_r_per_unit_z:+8.2f} P(U) {b.p_outlier:.3f}")
        res[name] = rows
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-u", type=Path, default=Path("data/pair_validation_multisurvey_lsw"))
    ap.add_argument("--without-u", type=Path,
                    default=Path("data/pair_validation_multisurvey_lsw_noU"))
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    from qso_pcolor.plotting import SERIES, save_figure, use_paper_style

    use_paper_style()
    s4 = dict(np.load(args.with_u.with_suffix(".npz"), allow_pickle=False))
    r4 = json.loads(args.with_u.with_suffix(".json").read_text())
    r3 = json.loads(args.without_u.with_suffix(".json").read_text())
    lab, held, spt = s4["label"], s4["held"].astype(bool), s4["spectype"]

    # -- validation figure -----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    ax = axes[0]
    for key, name, colour in (("A_original_grzW", "predecessor", SERIES["neutral"]),
                              ("D_multisurvey_grzW", "model", SERIES["same_z"])):
        ok = s4[key + "_ok"].astype(bool) & held
        fpr, tpr = roc_curve(s4[key + "_log_r"][ok & (lab == "same_z")],
                             s4[key + "_log_r"][ok & (lab == "field_q")])
        auc = r4[key]["held_out"]["same_vs_field_by_log_r"]
        ax.plot(fpr, tpr, color=colour, lw=1.2, label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], color=SERIES["grid"], lw=0.8, ls="--")
    ax.set_xlabel("wrong-z quasars above threshold")
    ax.set_ylabel("same-z quasars above threshold")
    ax.legend(fontsize=6, frameon=False, loc="lower right")

    ax = axes[1]
    key = "D_multisurvey_grzW"
    ok = s4[key + "_ok"].astype(bool)
    bf = s4[key + "_log_bf"]
    bins = np.linspace(-40, 30, 71)
    for cls, name, colour in (("same_z", "same-z quasar", SERIES["same_z"]),
                              ("field_q", "wrong-z quasar", SERIES["field_q"]),
                              ("non_qso", "star or galaxy", SERIES["background"])):
        v = np.clip(bf[ok & (lab == cls)], bins[0], bins[-1])
        ax.hist(v, bins, density=True, histtype="step", color=colour, lw=1.1, label=name)
    ax.set_xlabel(r"$\ln\mathrm{BF}$ (quasar at $z_0$ / field)")
    ax.set_ylabel("density")
    ax.legend(fontsize=6, frameon=False, loc="upper left")

    ax = axes[2]
    for key, name, colour, mk in (("A_original_grzW", "predecessor", SERIES["neutral"], "s"),
                                  ("D_multisurvey_grzW", "model", SERIES["same_z"], "o")):
        c = r4[key]["calibration_by_separation"]
        x = [0.5 * (t["sep"][0] + t["sep"][1]) for t in c]
        ax.plot(x, [t["ratio"] for t in c], mk + "-", color=colour, ms=3.5, lw=1, label=name)
    ax.axhline(1, color=SERIES["grid"], lw=0.8, ls="--")
    ax.set_xlabel("separation (arcsec)")
    ax.set_ylabel("observed / predicted")
    ax.set_ylim(0, 11)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    print(save_figure(fig, "validation/pair_validation_multisurvey"))

    # -- outlier figure ---------------------------------------------------------
    sw = sweep()
    Path("data/outlier_sweep_multisurvey.json").write_text(json.dumps(sw, indent=1, default=float))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, name in zip(axes[:2], ("g", "z")):
        rows = sw[name]
        s = [r["s"] for r in rows]
        ax.plot(s, [r["three"]["p_q"] for r in rows], "o-", color=SERIES["field_q"], ms=3, lw=1,
                label="three hypotheses")
        ax.plot(s, [r["four"]["p_q"] for r in rows], "s-", color=SERIES["same_z"], ms=3, lw=1,
                label="with unmodelled term")
        ax.plot(s, [r["four"]["p_u"] for r in rows], "s--", color=SERIES["neutral"], ms=3, lw=1,
                label="P(unmodelled)")
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(f"Legacy {name}-band flux multiplier $s$")
        for r in rows:
            ax.annotate(f"{r['d']:.0f}", (r["s"], 1.03), fontsize=6, ha="center", va="bottom",
                        color=SERIES["neutral"], annotation_clip=False)
    axes[0].set_ylabel("P(quasar at any $z$)")
    h, lbl = axes[0].get_legend_handles_labels()
    ax = axes[2]
    t3, t4 = r3["D_multisurvey_grzW"]["tails"], r4["D_multisurvey_grzW"]["tails"]
    lab_ = [f"{t['sigma'][0]:g}-{t['sigma'][1]:g}" if np.isfinite(t["sigma"][1])
            else f">{t['sigma'][0]:g}" for t in t3]
    x = np.arange(len(lab_))
    frac = lambda ts: np.array([t["non_qso_called_quasar"] / t["n_non_qso"] if t["n_non_qso"]  # noqa: E731
                                else np.nan for t in ts])
    ax.bar(x - 0.2, frac(t3), 0.4, color=SERIES["field_q"], label="three")
    ax.bar(x + 0.2, frac(t4), 0.4, color=SERIES["same_z"], label="four")
    for xi, t in zip(x, t3):
        ax.annotate(str(t["n_non_qso"]), (xi, 0.6), fontsize=6, ha="center", va="bottom",
                    color=SERIES["neutral"], annotation_clip=False)
    ax.set_xticks(x, lab_, fontsize=6)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel(r"distance to nearer model ($\sigma$)")
    ax.set_ylabel("non-quasars called quasar")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.legend(h, lbl, loc="lower center", ncol=3, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    print(save_figure(fig, "method/fig_outliers_multisurvey"))


if __name__ == "__main__":
    main()
