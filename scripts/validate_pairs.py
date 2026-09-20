#!/usr/bin/env python
"""Validate the scorer on spectroscopically labelled close pairs.

This is the first measurement of whether the method works, as opposed to
whether its machinery is self-consistent.  Every companion in
``data/pairs_desi_dr1.npz`` (from ``build_pair_validation.py``) has a spectrum,
so its label -- ``same_z``, ``field_q`` or ``non_qso`` -- is independent of the
colours being scored.

Three questions, each with a number:

1. **Can the colours tell a same-redshift quasar from a wrong-redshift one?**
   ROC and AUC for ``same_z`` against ``field_q``, ranked by
   ``log_r_per_unit_z``.  An AUC near 0.5 would mean the colours carry no
   redshift information at the +/-3000 km/s level, and the whole exercise is a
   quasar finder only.
2. **Does the Bayes factor fail on field quasars, as the README claims?**
   The same ROC ranked by ``log_bayes_factor_qz_bkg``, which has no ``field_q``
   term.  If its AUC is not clearly lower, the README's central claim is wrong.
3. **Is ``p_zmatch_given_qso`` calibrated?**  Among quasar companions, the
   empirical fraction with a matching redshift in bins of predicted
   probability.  A reliability curve far from the diagonal means the number is
   not a probability.

Plus one sanity line: what fraction of stars and galaxies score above the
median same-redshift quasar.

**Background model.**  One global model (the eight-field fit in
``models/examples/global.json``) is used for every pair.  Fitting a local cone
per candidate would take days for ~10^4 pairs, and it would not change the
answers to questions 1--3: those compare quasars against quasars, where the
background cancels, and for question 4 the quasar/background separation is of
order 10^6, so the background's fine structure is irrelevant.  This is a
deliberate approximation and is recorded in the output.

**Held-out subset.**  Companions that are themselves DESI quasars may have
been in the model's training set.  The in-sample effect was measured at
+0.018 nats, so it should not matter; every number is reported both for all
pairs and for pairs whose companion lies in the model's reserved sky blocks,
and a disagreement between the two columns would itself be a finding.

    python scripts/validate_pairs.py
    python scripts/validate_pairs.py --pairs data/pairs_desi_dr1.npz --min-sep 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

BANDS = ("g", "r", "z", "w1", "w2")
MAG_EDGES = np.array([17.0, 19.5, 20.5, 21.5, 22.5])


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    """Area under the ROC curve by the Mann-Whitney statistic.

    Ties count one half, so a constant score gives exactly 0.5.  Written out
    rather than imported so the validation has no dependency the package does
    not already carry.
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, float)
    # average ranks for ties
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def roc_curve(pos: np.ndarray, neg: np.ndarray, n: int = 200):
    """(false positive rate, true positive rate) along score thresholds."""
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    thr = np.quantile(np.concatenate([pos, neg]), np.linspace(0, 1, n))
    tpr = np.array([(pos >= t).mean() for t in thr])
    fpr = np.array([(neg >= t).mean() for t in thr])
    return fpr, tpr


def reliability(p: np.ndarray, y: np.ndarray, edges: np.ndarray):
    """Mean predicted probability and empirical rate per bin, with counts."""
    idx = np.clip(np.digitize(p, edges) - 1, 0, edges.size - 2)
    rows = []
    for b in range(edges.size - 1):
        m = idx == b
        if m.sum() >= 20:
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=Path("data/pairs_desi_dr1.npz"))
    ap.add_argument("--model", type=Path, default=Path("models/qso_south_full.json"))
    ap.add_argument("--background", type=Path,
                    default=Path("models/background_south_global.json"),
                    help="the shipped footprint-average background (all source "
                         "types). The PSF-only eight-field model used for the "
                         "first validation is models/examples/global.json")
    ap.add_argument("--background-density", type=Path,
                    default=Path("models/background_density_south_global.json"))
    ap.add_argument("--plateau", type=float, default=240.0,
                    help="quasar density per deg^2 for Sigma_Q, as in score_examples")
    ap.add_argument("--min-sep", type=float, default=3.0)
    ap.add_argument("--max-sep", type=float, default=30.0)
    ap.add_argument("--half-width-kms", type=float, default=3000.0,
                    help="match window; the pair labels use the same value")
    ap.add_argument("--max-non-qso", type=int, default=30000,
                    help="random subsample of star/galaxy companions; there are "
                         "~5e5 and the sanity check needs nothing like that many")
    ap.add_argument("--out", type=Path, default=Path("data/pair_validation_results.npz"))
    ap.add_argument("--report", type=Path, default=Path("data/pair_validation_report.json"))
    ap.add_argument("--figure", type=str, default="validation/pair_validation",
                    help="figure name under plots/; change it when validating a "
                         "candidate model so the shipped model's figure survives")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score_examples import build_sigma_q

    from qso_pcolor.background import BackgroundColourModel, galactic_healpix
    from qso_pcolor.data import galactic_from_equatorial
    from qso_pcolor.features import RelativeFluxTransform, deredden
    from qso_pcolor.priors import BackgroundSurfaceDensity
    from qso_pcolor.qso_model import RedshiftMatch, SlicedColourRedshiftModel
    from qso_pcolor.score import BlendPolicy, score_candidates

    d = dict(np.load(args.pairs, allow_pickle=False))
    label = d["label"].astype(str)
    sep = d["sep_arcsec"]
    qso = SlicedColourRedshiftModel.load(args.model)
    bkg = BackgroundColourModel.load(args.background)
    bdens = BackgroundSurfaceDensity.load(args.background_density)
    print(f"model {args.model}: support {qso.support[0]:.2f}-{qso.support[1]:.2f}")
    print(f"pairs: {label.size:,}  "
          + "  ".join(f"{k} {int((label == k).sum()):,}" for k in ("same_z", "field_q", "non_qso")))

    # -- selection: southern photometry, in-support primary, clean separation --
    rel = np.asarray(d["comp_release"], int)
    zp = np.asarray(d["z_primary"], float)
    keep = ((rel == int(qso.meta["release"])) & (sep >= args.min_sep) & (sep <= args.max_sep)
            & qso.in_support(zp))
    nq = np.flatnonzero(keep & (label == "non_qso"))
    if nq.size > args.max_non_qso:
        drop = np.random.default_rng(0).choice(nq, nq.size - args.max_non_qso, replace=False)
        keep[drop] = False
        print(f"non_qso subsampled {nq.size:,} -> {args.max_non_qso:,}")
    print(f"kept {int(keep.sum()):,} with release {qso.meta['release']}, "
          f"{args.min_sep:g}-{args.max_sep:g} arcsec, primary in model support")

    st = lambda p: np.stack([np.asarray(d[f"comp_{p}{b}"], float)[keep] for b in BANDS], axis=1)  # noqa: E731
    f, v = deredden(st("flux_"), st("flux_ivar_"), st("mw_transmission_"))
    tr = RelativeFluxTransform(reference_band="r", min_ref_snr=5.0)
    fs = tr(f, v, BANDS)
    usable = fs.usable(min_dims=3) & np.isfinite(fs.ref_mag) & (fs.ref_mag < MAG_EDGES[-1])
    print(f"  usable photometry: {int(usable.sum()):,}")

    idx = np.flatnonzero(keep)[usable]
    label, sep, zp = label[idx], sep[idx], zp[idx]
    spectype = np.asarray(d["comp_spectype"]).astype(str)[idx]
    feat = _subset(fs, usable)

    l, b = galactic_from_equatorial(np.asarray(d["comp_ra"], float)[idx],
                                    np.asarray(d["comp_dec"], float)[idx])
    held = set(int(x) for x in (qso.meta.get("holdout_blocks")
                                or qso.meta["holdout_blocks_recovered"]))
    in_held = np.isin(galactic_healpix(l, b, int(qso.meta.get("holdout_nside", 4))), list(held))
    print(f"  companions in reserved blocks: {int(in_held.sum()):,}")

    # -- score ---------------------------------------------------------------
    prior = build_sigma_q(qso.support[0], qso.support[1], args.plateau)
    match = RedshiftMatch(half_width_kms=args.half_width_kms)
    frac = np.stack([np.asarray(d[f"comp_fracflux_{bnd}"], float)[idx] for bnd in ("g", "r", "z")], 1)
    frac = np.nanmax(np.where(np.isfinite(frac), frac, 0.0), axis=1)
    t0 = time.time()
    rows = score_candidates(
        feat, z_primary=zp, l_deg=l, b_deg=b,
        qso_model=qso, background_model=bkg, background_density=bdens,
        qso_prior=prior, match=match,
        blend_policy=BlendPolicy(min_separation_arcsec=args.min_sep, max_fracflux=None),
        separation_arcsec=sep, fracflux=frac, min_bands=3,
    )
    print(f"scored {len(rows):,} companions in {time.time() - t0:.0f} s")

    logr = np.array([r.log_r_per_unit_z for r in rows])
    logbf = np.array([r.log_bayes_factor_qz_bkg for r in rows])
    pz = np.array([r.p_zmatch_given_qso for r in rows])
    psame = np.array([r.p_sameq for r in rows])
    status = np.array([r.status for r in rows])
    scored = status == "ok"
    print(f"  status ok: {int(scored.sum()):,}   other: "
          + ", ".join(f"{s} {int((status == s).sum())}" for s in np.unique(status[~scored])))

    # -- the three measurements, on all pairs and on the reserved blocks ------
    report = {"n": {}, "auc": {}, "reliability": {}, "non_qso_above_same_z_median": {},
              "settings": vars(args) | {"background_note":
                  "single global 8-field background for every pair; see docstring"}}
    for name, m in (("all", scored), ("held_out", scored & in_held)):
        S, F, N = m & (label == "same_z"), m & (label == "field_q"), m & (label == "non_qso")
        report["n"][name] = {"same_z": int(S.sum()), "field_q": int(F.sum()), "non_qso": int(N.sum())}
        report["auc"][name] = {
            "same_z_vs_field_q_by_log_r": auc_rank(logr[S], logr[F]),
            "same_z_vs_field_q_by_log_bf": auc_rank(logbf[S], logbf[F]),
            "same_z_vs_field_q_by_p_zmatch": auc_rank(pz[S], pz[F]),
            "quasar_vs_non_qso_by_log_bf": auc_rank(logbf[S | F], logbf[N]),
        }
        Q = S | F
        y = (label == "same_z")[Q].astype(float)
        report["reliability"][name] = reliability(
            pz[Q], y, np.array([0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 1.0]))
        med = np.median(logbf[S]) if S.any() else np.nan
        report["non_qso_above_same_z_median"][name] = (
            float((logbf[N] > med).mean()) if N.any() else float("nan"))

    # -- diagnostics that decide how to read the numbers above ----------------
    # 1. who was refused, and why.  A status other than ok removes the object
    #    from every metric, so a class-dependent refusal biases the comparison.
    report["refused"] = {}
    for st_ in np.unique(status[~scored]):
        m = status == st_
        report["refused"][str(st_)] = {
            "n": int(m.sum()),
            "by_label": {k: int((m & (label == k)).sum()) for k in ("same_z", "field_q", "non_qso")},
            "ref_mag_16_50_84": [float(x) for x in np.percentile(feat.ref_mag[m], [16, 50, 84])],
            "z_primary_16_50_84": [float(x) for x in np.percentile(zp[m], [16, 50, 84])],
        }
    # 2. calibration ratio (empirical / predicted) by separation.  p(z in W|Q)
    #    assumes the companion's redshift is drawn from the FIELD prior.  A
    #    physical pair is not: quasars cluster, so at small separation the true
    #    prior has an excess at z0 that the scorer does not model (deliberately,
    #    see AGENTS.md M7).  If the ratio falls toward 1 with separation the
    #    excess is clustering; if it is flat it is a bug in the scorer.
    report["calibration_by_separation"] = []
    Q = scored & ((label == "same_z") | (label == "field_q"))
    for lo, hi in ((3, 5), (5, 10), (10, 20), (20, 30)):
        m = Q & (sep >= lo) & (sep < hi)
        if m.sum() >= 50:
            pred = float(pz[m].mean()); emp = float((label[m] == "same_z").mean())
            report["calibration_by_separation"].append(
                {"sep": [lo, hi], "n": int(m.sum()), "mean_predicted": pred,
                 "empirical": emp, "ratio": emp / pred if pred > 0 else float("nan")})
    # 3. the non_qso class is DESI TARGETS with spectra -- objects the survey
    #    chose to observe, many as quasar candidates -- not a random sample of
    #    the imaging.  Split it by what the spectrum said.
    report["non_qso_by_spectype"] = {}
    for stype in np.unique(spectype[scored & (label == "non_qso")]):
        m = scored & (label == "non_qso") & (spectype == stype)
        Qs = scored & ((label == "same_z") | (label == "field_q"))
        report["non_qso_by_spectype"][str(stype)] = {
            "n": int(m.sum()),
            "auc_quasar_vs_this_by_log_bf": auc_rank(logbf[Qs], logbf[m]),
            "frac_above_same_z_median_log_bf": float((logbf[m] > np.median(logbf[scored & (label == "same_z")])).mean()),
        }

    for k, v in report["settings"].items():
        if isinstance(v, Path):
            report["settings"][k] = str(v)

    print("\n=== results ===")
    for name in ("all", "held_out"):
        n = report["n"][name]; a = report["auc"][name]
        print(f"\n[{name}]  same_z {n['same_z']:,}  field_q {n['field_q']:,}  non_qso {n['non_qso']:,}")
        print(f"  AUC same_z vs field_q   by log R      {a['same_z_vs_field_q_by_log_r']:.3f}")
        print(f"                           by log BF     {a['same_z_vs_field_q_by_log_bf']:.3f}")
        print(f"                           by p(z in W|Q) {a['same_z_vs_field_q_by_p_zmatch']:.3f}")
        print(f"  AUC quasar vs non_qso    by log BF     {a['quasar_vs_non_qso_by_log_bf']:.3f}")
        print(f"  non_qso above same_z median log BF:    "
              f"{100 * report['non_qso_above_same_z_median'][name]:.2f}%")
        print("  reliability of p(z in W | Q):  mean p  ->  empirical   (n)")
        for pm, em, nn in report["reliability"][name]:
            print(f"      {pm:7.4f}  ->  {em:7.4f}   ({nn:,})")

    print("\n=== diagnostics ===")
    for st_, r in report["refused"].items():
        print(f"  refused '{st_}': {r['n']:,}  by label {r['by_label']}  "
              f"r-mag 16/50/84 {['%.1f' % x for x in r['ref_mag_16_50_84']]}  "
              f"z0 16/50/84 {['%.2f' % x for x in r['z_primary_16_50_84']]}")
    print("  calibration ratio empirical/predicted by separation:")
    for r in report["calibration_by_separation"]:
        print(f"    {r['sep'][0]:2d}-{r['sep'][1]:<2d} arcsec  n={r['n']:6,d}  "
              f"pred {r['mean_predicted']:.4f}  emp {r['empirical']:.4f}  ratio {r['ratio']:.2f}")
    print("  non_qso by spectype:")
    for k, r in report["non_qso_by_spectype"].items():
        print(f"    {k:8s} n={r['n']:6,d}  AUC vs quasars {r['auc_quasar_vs_this_by_log_bf']:.3f}  "
              f"above same_z median {100 * r['frac_above_same_z_median_log_bf']:.1f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, pair_idx=idx, label=label, spectype=spectype, sep=sep,
                        z_primary=zp, in_held=in_held,
                        log_r=logr, log_bf=logbf, p_zmatch=pz, p_sameq=psame, status=status,
                        ref_mag=feat.ref_mag)
    args.report.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out} and {args.report}")
    make_figure(label, scored, in_held, logr, logbf, pz, report, args.figure)


def _subset(fs, mask):
    """A FeatureSet restricted to ``mask`` rows, whatever its field list is."""
    import dataclasses

    kw = {}
    for fld in dataclasses.fields(fs):
        val = getattr(fs, fld.name)
        if isinstance(val, np.ndarray) and val.shape[:1] == (mask.size,):
            kw[fld.name] = val[mask]
        else:
            kw[fld.name] = val
    return type(fs)(**kw)


def make_figure(label, scored, in_held, logr, logbf, pz, report, fig_name="validation/pair_validation"):
    import matplotlib.pyplot as plt

    from qso_pcolor.plotting import SERIES, save_figure, use_paper_style

    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    S = scored & (label == "same_z"); F = scored & (label == "field_q"); N = scored & (label == "non_qso")

    ax = axes[0]
    for score, name, c in ((logr, "log R (three hypotheses)", SERIES["same_z"]),
                           (logbf, "log BF (quasar vs background)", SERIES["field_q"]),
                           (pz, "p(z in W | Q)", SERIES["background"])):
        fpr, tpr = roc_curve(score[S], score[F])
        a = report["auc"]["all"][{"log R (three hypotheses)": "same_z_vs_field_q_by_log_r",
                                  "log BF (quasar vs background)": "same_z_vs_field_q_by_log_bf",
                                  "p(z in W | Q)": "same_z_vs_field_q_by_p_zmatch"}[name]]
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name}  AUC {a:.2f}")
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
    ax.set_xlabel("field_q accepted (false positive rate)")
    ax.set_ylabel("same_z recovered (true positive rate)")
    ax.set_title("same-redshift vs wrong-redshift quasar")
    # below the axes: every quadrant of a ROC panel has a curve in it
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), fontsize=8, frameon=False)

    ax = axes[1]
    bins = np.linspace(np.nanpercentile(logbf[scored], 0.5), np.nanpercentile(logbf[scored], 99.5), 60)
    for m, name, c in ((S, "same_z", SERIES["same_z"]), (F, "field_q", SERIES["field_q"]), (N, "non_qso", SERIES["neutral"])):
        ax.hist(logbf[m], bins=bins, histtype="step", lw=1.8, color=c, density=True, label=f"{name} ({int(m.sum()):,})")
    ax.set_xlabel("log Bayes factor, quasar at z0 vs background")
    ax.set_ylabel("density")
    ax.set_title("the quasar finder")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    for name, c, mk in (("all", SERIES["same_z"], "o"), ("held_out", SERIES["background"], "s")):
        r = report["reliability"][name]
        if r:
            pm, em, nn = zip(*r)
            err = np.sqrt(np.array(em) * (1 - np.array(em)) / np.array(nn))
            ax.errorbar(pm, em, yerr=err, fmt=mk, color=c, ms=5, capsize=2, label=name)
    # axis limits from BOTH predicted and empirical values, or a miscalibrated
    # model pushes its own evidence off the plot
    # log-log: probabilities span 1e-3 to ~1, and a constant multiplicative
    # miscalibration -- the physical-pair clustering excess -- is then a line
    # parallel to the diagonal rather than a curve squashed into one corner
    vals = [v for name in ("all", "held_out") for row in report["reliability"][name] for v in row[:2]]
    lo = max(3e-4, 0.5 * min(vals)) if vals else 1e-3
    hi = min(1.0, 2.0 * max(vals)) if vals else 1.0
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=1, ls="--", label="calibrated")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("predicted p(z in W | Q)")
    ax.set_ylabel("fraction of quasar companions with |dv| < W")
    ax.set_title("is the redshift probability calibrated?")
    ax.legend(loc="upper left", fontsize=8)

    # NOT `name`: the reliability loop above reuses that identifier, and the
    # figure once landed at plots/held_out.png because of it
    path = save_figure(fig, fig_name)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
