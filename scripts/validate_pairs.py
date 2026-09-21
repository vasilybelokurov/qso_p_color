#!/usr/bin/env python
"""Validate the saved scorer on spectroscopically labelled close pairs.

Apply maskbits=0 and an explicitly chosen reference-band fracflux limit, then
report full-sample and spatially held-out metrics. Labels are derived from the
requested velocity window. The shipped all-source global background and saved
quasar prior are used; no model is fitted by this script. Log R depends on the
background, whereas p_zmatch_given_qso does not.

    python scripts/validate_pairs.py --max-fracflux 0.2 --hard-negative-max-kms 6000 10000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

BANDS = ("g", "r", "z", "w1", "w2")


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


def validation_labels(spectype: np.ndarray, dv_kms: np.ndarray,
                      half_width_kms: float) -> np.ndarray:
    """Spectroscopic classes for the declared absolute velocity window (km/s)."""
    if not np.isfinite(half_width_kms) or half_width_kms <= 0:
        raise ValueError("half_width_kms must be positive and finite")
    qso = np.char.strip(np.asarray(spectype).astype(str)) == "QSO"
    dv = np.abs(np.asarray(dv_kms, float))
    labels = np.full(qso.shape, "non_qso", dtype="U16")
    labels[qso] = "field_q"
    labels[qso & (dv < half_width_kms)] = "same_z"
    labels[qso & ~np.isfinite(dv)] = "invalid_redshift"
    return labels


def clean_photometry(maskbits: np.ndarray, fracflux_r: np.ndarray,
                     max_fracflux: float) -> np.ndarray:
    """Survey mask and reference-band contamination cut; missing is not clean."""
    return ((np.asarray(maskbits) == 0) & np.isfinite(fracflux_r)
            & (np.asarray(fracflux_r) <= max_fracflux))


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
    ap.add_argument("--prior", type=Path, default=Path("models/sigma_q_south.json"))
    ap.add_argument("--max-fracflux", type=float, required=True,
                    help="maximum reference-band fracflux_r; missing values are rejected")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard-negative-max-kms", type=float, nargs="*", default=[],
                    help="report field quasars between the match boundary and each limit")
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

    from qso_pcolor.background import BackgroundColourModel, galactic_healpix
    from qso_pcolor.data import galactic_from_equatorial, _save_npz
    from qso_pcolor.features import RelativeFluxTransform, deredden
    from qso_pcolor.priors import BackgroundSurfaceDensity, GridQSOPrior
    from qso_pcolor.qso_model import RedshiftMatch, SlicedColourRedshiftModel
    from qso_pcolor.score import BlendPolicy, score_candidates

    d = dict(np.load(args.pairs, allow_pickle=False))
    label = validation_labels(d["comp_spectype"], d["dv_kms"], args.half_width_kms)
    if (not np.isfinite(args.max_fracflux) or args.max_fracflux < 0
            or any(x <= args.half_width_kms for x in args.hard_negative_max_kms)):
        ap.error("fracflux must be finite and non-negative; hard-negative limits must exceed the window")
    sep = d["sep_arcsec"]
    selection = {}
    def record_selection(stage, mask):
        selection[stage] = {k: int(np.sum(mask & (label == k)))
                            for k in ("same_z", "field_q", "non_qso", "invalid_redshift")}
    record_selection("input", np.ones(label.size, bool))
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
            & qso.in_support(zp) & (label != "invalid_redshift"))
    record_selection("system_separation_redshift", keep)
    keep &= clean_photometry(d["comp_maskbits"], d["comp_fracflux_r"], args.max_fracflux)
    record_selection("mask_and_fracflux", keep)
    nq = np.flatnonzero(keep & (label == "non_qso"))
    if nq.size > args.max_non_qso:
        drop = np.random.default_rng(args.seed).choice(nq, nq.size - args.max_non_qso, replace=False)
        keep[drop] = False
        print(f"non_qso subsampled {nq.size:,} -> {args.max_non_qso:,}")
    record_selection("after_non_qso_subsample", keep)
    print(f"kept {int(keep.sum()):,} with release {qso.meta['release']}, "
          f"{args.min_sep:g}-{args.max_sep:g} arcsec, primary in model support")

    st = lambda p: np.stack([np.asarray(d[f"comp_{p}{b}"], float)[keep] for b in BANDS], axis=1)  # noqa: E731
    f, v = deredden(st("flux_"), st("flux_ivar_"), st("mw_transmission_"))
    tr = RelativeFluxTransform(reference_band="r", min_ref_snr=5.0)
    fs = tr(f, v, BANDS)
    usable = (fs.usable(min_dims=3) & np.isfinite(fs.ref_mag)
              & (fs.ref_mag >= bkg.mag_edges[0]) & (fs.ref_mag < bkg.mag_edges[-1]))
    print(f"  usable photometry: {int(usable.sum()):,}")

    idx = np.flatnonzero(keep)[usable]
    final_selection = np.zeros(label.size, bool)
    final_selection[idx] = True
    record_selection("usable_in_background_magnitude_range", final_selection)
    label, sep, zp = label[idx], sep[idx], zp[idx]
    dv = np.abs(np.asarray(d["dv_kms"], float)[idx])
    spectype = np.asarray(d["comp_spectype"]).astype(str)[idx]
    feat = fs.subset(usable)

    l, b = galactic_from_equatorial(np.asarray(d["comp_ra"], float)[idx],
                                    np.asarray(d["comp_dec"], float)[idx])
    held = set(int(x) for x in (qso.meta.get("holdout_blocks")
                                or qso.meta["holdout_blocks_recovered"]))
    in_held = np.isin(galactic_healpix(l, b, int(qso.meta.get("holdout_nside", 4))), list(held))
    print(f"  companions in reserved blocks: {int(in_held.sum()):,}")

    # -- score ---------------------------------------------------------------
    prior = GridQSOPrior.load(args.prior)
    match = RedshiftMatch(half_width_kms=args.half_width_kms)
    frac = np.asarray(d["comp_fracflux_r"], float)[idx]
    t0 = time.time()
    rows = score_candidates(
        feat, z_primary=zp, l_deg=l, b_deg=b,
        qso_model=qso, background_model=bkg, background_density=bdens,
        qso_prior=prior, match=match,
        blend_policy=BlendPolicy(min_separation_arcsec=args.min_sep, max_fracflux=args.max_fracflux),
        separation_arcsec=sep, fracflux=frac, min_bands=3,
        candidate_id=d["comp_targetid"][idx], primary_id=d["prim_targetid"][idx],
    )
    print(f"scored {len(rows):,} companions in {time.time() - t0:.0f} s")

    logr = np.array([r.log_r_per_unit_z for r in rows])
    logbf = np.array([r.log_bayes_factor_qz_bkg for r in rows])
    pz = np.array([r.p_zmatch_given_qso for r in rows])
    psame = np.array([r.p_sameq for r in rows])
    status = np.array([r.status for r in rows])
    scored = status == "ok"
    evidence_ok = np.isfinite(logbf) & np.array([
        "background_out_of_mag_range" not in r.quality_flags for r in rows])
    print(f"  status ok: {int(scored.sum()):,}   other: "
          + ", ".join(f"{s} {int((status == s).sum())}" for s in np.unique(status[~scored])))

    # -- the three measurements, on all pairs and on the reserved blocks ------
    report = {"n": {}, "auc": {}, "reliability": {}, "non_qso_above_same_z_median": {},
              "selection_counts": selection, "hard_negatives": {}, "evidence_n": {},
              "model_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (args.model, args.background, args.background_density, args.prior)},
              "settings": vars(args) | {"maskbits": 0, "fracflux_band": "r",
                  "magnitude_range": bkg.mag_edges[[0, -1]].tolist(),
                  "background_note": "saved all-source global background; no refit"}}
    for name, m in (("all", scored), ("held_out", scored & in_held)):
        S, F, N = m & (label == "same_z"), m & (label == "field_q"), m & (label == "non_qso")
        report["n"][name] = {"same_z": int(S.sum()), "field_q": int(F.sum()), "non_qso": int(N.sum())}
        E = evidence_ok & (in_held if name == "held_out" else True)
        ES, EF, EN = E & (label == "same_z"), E & (label == "field_q"), E & (label == "non_qso")
        report["evidence_n"][name] = {"same_z": int(ES.sum()), "field_q": int(EF.sum()), "non_qso": int(EN.sum())}
        report["auc"][name] = {
            "same_z_vs_field_q_by_log_r": auc_rank(logr[S], logr[F]),
            "same_z_vs_field_q_by_log_bf": auc_rank(logbf[ES], logbf[EF]),
            "same_z_vs_field_q_by_p_zmatch": auc_rank(pz[S], pz[F]),
            "quasar_vs_non_qso_by_log_bf": auc_rank(logbf[ES | EF], logbf[EN]),
        }
        report["hard_negatives"][name] = []
        for upper in args.hard_negative_max_kms:
            H = F & (dv < upper)
            report["hard_negatives"][name].append({
                "dv_kms": [args.half_width_kms, upper],
                "n_same_z": int(S.sum()), "n_field_q": int(H.sum()),
                "auc_log_r": auc_rank(logr[S], logr[H]),
                "auc_p_zmatch": auc_rank(pz[S], pz[H]),
            })
        Q = S | F
        y = (label == "same_z")[Q].astype(float)
        report["reliability"][name] = reliability(
            pz[Q], y, np.array([0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 1.0]))
        med = np.median(logbf[S]) if S.any() else np.nan
        report["non_qso_above_same_z_median"][name] = (
            float((logbf[N] > med).mean()) if N.any() else float("nan"))

    # -- diagnostics that decide how to read the numbers above ----------------
    # 1. Posterior refusals are counted separately from usable colour evidence.
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
    # Retain the full PairScore contract as well as historical plotting aliases.
    records = [r.as_row() for r in rows]
    columns = {key: np.asarray([r[key] for r in records]) for key in records[0]}
    columns.update(pair_idx=idx, label=label, spectype=spectype, sep=sep,
                   in_held=in_held, dv_kms=dv, log_r=logr, log_bf=logbf, p_zmatch=pz,
                   maskbits=d["comp_maskbits"][idx], fracflux_r=frac)
    _save_npz(args.out, **columns)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out} and {args.report}")
    make_figure(label, scored, in_held, logr, logbf, pz, report, args.figure)


def make_figure(label, scored, in_held, logr, logbf, pz, report, fig_name="validation/pair_validation"):
    import matplotlib

    matplotlib.use("Agg")
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
