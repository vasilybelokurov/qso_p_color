#!/usr/bin/env python
"""Can the multi-survey model, with its new priors, replace the original?

Scores the same spectroscopically labelled companions as ``validate_pairs.py``
three ways, on exactly the same rows:

A. original model, dereddened Legacy g, r, z + Legacy forced W1, W2 (shipped);
B. original model, g, r, z only (W1, W2 marked unmeasured);
C. multi-survey model, native Legacy DR9-south g, r, z, with its priors;
D. (with ``--ms-model`` trained on Legacy forced W1/W2) the multi-survey
   model on native Legacy g, r, z, W1, W2, with its priors and, if given,
   its outlier term.

The pair catalogue's W1/W2 are Legacy forced unWISE fluxes, a different
system from AllWISE; only a model with ``decals_dr9_south:w1/w2`` bands can
use them. B is the like-for-like comparison for C, A for D.
Rows are those ``validate_pairs.py`` scores (same selection, same blend
policy); held-out means the original model's reserved sky blocks, which the
multi-survey quasar fit also reserved.

    python scripts/validate_pairs_multisurvey.py --max-fracflux 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from validate_pairs import auc_rank, clean_photometry, validation_labels  # noqa: E402

BANDS = ("g", "r", "z", "w1", "w2")


def metrics(label, spectype, held, rows, dv, hard_kms):
    col = lambda k: np.array([getattr(r, k) for r in rows], float)  # noqa: E731
    logr, logbf, pz = col("log_r_per_unit_z"), col("log_bayes_factor_qz_bkg"), col(
        "p_zmatch_given_qso")
    ok = np.array([r.status == "ok" for r in rows])
    out = {"n_ok": int(ok.sum())}
    for name, m in (("all", ok), ("held_out", ok & held)):
        S, F, N = m & (label == "same_z"), m & (label == "field_q"), m & (label == "non_qso")
        H = F & (dv < hard_kms)
        out[name] = {
            "n": [int(S.sum()), int(F.sum()), int(N.sum())],
            "same_vs_field_by_log_r": auc_rank(logr[S], logr[F]),
            "same_vs_field_by_p_zmatch": auc_rank(pz[S], pz[F]),
            "same_vs_field_by_log_bf": auc_rank(logbf[S], logbf[F]),
            "same_vs_hard_by_log_r": auc_rank(logr[S], logr[H]),
            "quasar_vs_non_qso_by_log_bf": auc_rank(logbf[S | F], logbf[N]),
            "quasar_vs_star_by_log_bf": auc_rank(logbf[S | F], logbf[N & (spectype == "STAR")]),
            "quasar_vs_galaxy_by_log_bf": auc_rank(logbf[S | F],
                                                   logbf[N & (spectype == "GALAXY")]),
            "same_vs_non_qso_by_log_r": auc_rank(logr[S], logr[N]),
        }
    return out, logr, logbf, pz, ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=Path("data/pairs_desi_dr1.npz"))
    ap.add_argument("--max-fracflux", type=float, required=True)
    ap.add_argument("--half-width-kms", type=float, default=3000.0)
    ap.add_argument("--hard-negative-max-kms", type=float, default=6000.0)
    ap.add_argument("--min-sep", type=float, default=3.0)
    ap.add_argument("--max-sep", type=float, default=30.0)
    ap.add_argument("--max-non-qso", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--priors", type=Path, default=Path("models/multisurvey_priors.json"))
    ap.add_argument("--ms-model", type=Path, default=Path("models/multisurvey.json"))
    ap.add_argument("--ms-outlier", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/pair_validation_multisurvey.json"))
    ap.add_argument("--scores", type=Path, default=Path("data/pair_validation_multisurvey.npz"))
    args = ap.parse_args()

    from qso_pcolor.background import BackgroundColourModel, galactic_healpix
    from qso_pcolor.data import _save_npz, galactic_from_equatorial
    from qso_pcolor.features import RelativeFluxTransform, deredden
    from qso_pcolor.multisurvey import MultiSurveyModel, MultiSurveyOutlier, load_priors
    from qso_pcolor.multisurvey_data import Photometry
    from qso_pcolor.priors import BackgroundSurfaceDensity, GridQSOPrior
    from qso_pcolor.qso_model import RedshiftMatch, SlicedColourRedshiftModel
    from qso_pcolor.score import BlendPolicy, score_candidates

    d = dict(np.load(args.pairs, allow_pickle=False))
    label = validation_labels(d["comp_spectype"], d["dv_kms"], args.half_width_kms)
    qso = SlicedColourRedshiftModel.load("models/archive/original_legacy_south/qso_south_full.json")
    bkg = BackgroundColourModel.load("models/archive/original_legacy_south/background_south_global.json")
    dens = BackgroundSurfaceDensity.load("models/archive/original_legacy_south/background_density_south_global.json")
    prior = GridQSOPrior.load("models/archive/original_legacy_south/sigma_q_south.json")
    ms = MultiSurveyModel.load(args.ms_model)
    ms_priors = load_priors(args.priors, ms)

    # -- the selection of validate_pairs.py, verbatim in effect -----------------
    sep, zp = d["sep_arcsec"], np.asarray(d["z_primary"], float)
    keep = ((np.asarray(d["comp_release"], int) == int(qso.meta["release"]))
            & (sep >= args.min_sep) & (sep <= args.max_sep) & qso.in_support(zp)
            & (label != "invalid_redshift"))
    keep &= clean_photometry(d["comp_maskbits"], d["comp_fracflux_r"], args.max_fracflux)
    nq = np.flatnonzero(keep & (label == "non_qso"))
    if nq.size > args.max_non_qso:
        drop = np.random.default_rng(args.seed).choice(nq, nq.size - args.max_non_qso,
                                                       replace=False)
        keep[drop] = False
    st = lambda p: np.stack([np.asarray(d[f"comp_{p}{b}"], float)[keep] for b in BANDS], 1)  # noqa: E731
    flux, ivar, trans = st("flux_"), st("flux_ivar_"), st("mw_transmission_")
    f, v = deredden(flux, ivar, trans)
    tr = RelativeFluxTransform(reference_band="r", min_ref_snr=5.0)
    fs = tr(f, v, BANDS)
    usable = (fs.usable(min_dims=3) & np.isfinite(fs.ref_mag)
              & (fs.ref_mag >= bkg.mag_edges[0]) & (fs.ref_mag < bkg.mag_edges[-1]))
    idx = np.flatnonzero(keep)[usable]
    fs = fs.subset(usable)
    flux, ivar = flux[usable], ivar[usable]
    lab, spt = label[idx], np.asarray(d["comp_spectype"]).astype(str)[idx]
    sep, zp, dv = sep[idx], zp[idx], np.abs(np.asarray(d["dv_kms"], float)[idx])
    l, b = galactic_from_equatorial(np.asarray(d["comp_ra"], float)[idx],
                                    np.asarray(d["comp_dec"], float)[idx])
    held = np.isin(galactic_healpix(l, b, int(qso.meta.get("holdout_nside", 4))),
                   [int(x) for x in qso.meta["holdout_blocks"]])
    frac = np.asarray(d["comp_fracflux_r"], float)[idx]
    print(f"rows: {idx.size:,}  held out: {held.sum():,}")

    common = dict(z_primary=zp, l_deg=l, b_deg=b, match=RedshiftMatch(half_width_kms=args.half_width_kms),
                  blend_policy=BlendPolicy(min_separation_arcsec=args.min_sep,
                                           max_fracflux=args.max_fracflux),
                  separation_arcsec=sep, fracflux=frac)
    report, scores = {"settings": {k: str(v) for k, v in vars(args).items()},
                      "n_rows": int(idx.size)}, {}

    t0 = time.time()
    rows_a = score_candidates(fs, qso_model=qso, background_model=bkg, background_density=dens,
                              qso_prior=prior, min_bands=3, **common)
    print(f"A scored ({time.time() - t0:.0f} s)")
    fs_b = fs.subset(np.arange(fs.n_obs))
    fs_b.observed = fs_b.observed.copy()
    fs_b.observed[:, [fs.labels.index("w1/r"), fs.labels.index("w2/r")]] = False
    fs_b.cov = np.where(fs_b.observed[:, :, None] & fs_b.observed[:, None, :], fs_b.cov, 0.0)
    t0 = time.time()
    rows_b = score_candidates(fs_b, qso_model=qso, background_model=bkg, background_density=dens,
                              qso_prior=prior, min_bands=2, **common)
    print(f"B scored ({time.time() - t0:.0f} s)")
    # C: native (not dereddened) Legacy south g, r, z in nanomaggies
    ph = Photometry(flux[:, :3], np.where(ivar[:, :3] > 0, 1.0 / np.where(ivar[:, :3] > 0,
                    ivar[:, :3], 1.0), np.inf),
                    ("decals_dr9_south:g", "decals_dr9_south:r", "decals_dr9_south:z"))
    t0 = time.time()
    rows_c = ms.score(ph, min_bands=2, priors=ms_priors, **common)
    print(f"C scored ({time.time() - t0:.0f} s); reference bands: "
          + ", ".join(f"{k} {v}" for k, v in zip(*np.unique([r.reference_band for r in rows_c],
                                                            return_counts=True))))
    configs = [("A_original_grzW", rows_a), ("B_original_grz", rows_b),
               ("C_multisurvey_grz", rows_c)]
    if "decals_dr9_south:w1" in ms.transform.bands:
        var = np.where(ivar > 0, 1.0 / np.where(ivar > 0, ivar, 1.0), np.inf)
        ph5 = Photometry(flux, var, tuple(f"decals_dr9_south:{b}" for b in BANDS))
        out_model = MultiSurveyOutlier.load(args.ms_outlier) if args.ms_outlier else None
        t0 = time.time()
        rows_d = ms.score(ph5, min_bands=2, priors=ms_priors, outlier=out_model, **common)
        print(f"D scored ({time.time() - t0:.0f} s)")
        configs.append(("D_multisurvey_grzW", rows_d))

    for name, rows in configs:
        m, logr, logbf, pz, ok = metrics(lab, spt, held, rows, dv, args.hard_negative_max_kms)
        col = lambda k: np.array([getattr(r, k) for r in rows], float)  # noqa: E731
        # calibration of p(z in W | Q) by separation: the clustering excess
        qq = ok & np.isin(lab, ["same_z", "field_q"])
        m["calibration_by_separation"] = []
        for lo_, hi_ in ((3, 5), (5, 10), (10, 20), (20, 30)):
            s_ = qq & (sep >= lo_) & (sep < hi_)
            if s_.sum() >= 50:
                pred, emp = float(pz[s_].mean()), float((lab[s_] == "same_z").mean())
                m["calibration_by_separation"].append(
                    {"sep": [lo_, hi_], "n": int(s_.sum()), "predicted": pred,
                     "empirical": emp, "ratio": emp / pred})
        # the tails: far from both models, called a quasar at any z?
        far = np.fmin(col("qso_ood_sigma_any_z"), col("bkg_ood_sigma"))
        with np.errstate(invalid="ignore"):
            lq = np.logaddexp(col("log_lambda_sameq"), col("log_lambda_fieldq"))
            lo = col("log_lambda_out")
            ln = np.where(np.isfinite(lo), np.logaddexp(col("log_lambda_bkg"), lo),
                          col("log_lambda_bkg"))
        called = (lq - ln) > 0
        isq = spt == "QSO"
        m["tails"] = []
        for a_, b_ in ((0, 2), (2, 3), (3, 4), (4, 6), (6, np.inf)):
            t_ = ok & (far >= a_) & (far < b_)
            m["tails"].append({"sigma": [a_, b_], "n_non_qso": int((t_ & ~isq).sum()),
                               "non_qso_called_quasar": int((t_ & ~isq & called).sum()),
                               "n_qso": int((t_ & isq).sum()),
                               "qso_called_quasar": int((t_ & isq & called).sum())})
        m["non_qso_called_quasar"] = int((ok & ~isq & called).sum())
        scores[name + "_far"], scores[name + "_called"] = far, called
        scores[name + "_p_outlier"] = col("p_outlier")
        report[name] = m
        scores[name + "_log_r"], scores[name + "_log_bf"] = logr, logbf
        scores[name + "_p_zmatch"], scores[name + "_ok"] = pz, ok
        h = m["held_out"]
        print(f"\n[{name}] ok {m['n_ok']:,}  held-out n(same, field, nonQ) = {h['n']}")
        for k in ("same_vs_field_by_log_r", "same_vs_field_by_p_zmatch", "same_vs_field_by_log_bf",
                  "same_vs_hard_by_log_r", "quasar_vs_non_qso_by_log_bf",
                  "quasar_vs_star_by_log_bf", "quasar_vs_galaxy_by_log_bf",
                  "same_vs_non_qso_by_log_r"):
            print(f"  {k:30s} {h[k]:.3f}   (all {m['all'][k]:.3f})")
    if "D_multisurvey_grzW_ok" in scores:
        from scipy.stats import spearmanr as _sp
        both = scores["A_original_grzW_ok"].astype(bool) & scores["D_multisurvey_grzW_ok"].astype(bool)
        rho_ad = _sp(scores["A_original_grzW_log_r"][both], scores["D_multisurvey_grzW_log_r"][both])
        report["spearman_log_r_A_D"] = float(rho_ad.statistic)
        print(f"Spearman(log R: A vs D) on {both.sum():,} rows: {rho_ad.statistic:.3f}")
    both = scores["B_original_grz_ok"].astype(bool) & scores["C_multisurvey_grz_ok"].astype(bool)
    from scipy.stats import spearmanr
    rho = spearmanr(scores["B_original_grz_log_r"][both], scores["C_multisurvey_grz_log_r"][both])
    report["spearman_log_r_B_C"] = float(rho.statistic)
    print(f"\nSpearman(log R: B vs C) on {both.sum():,} rows: {rho.statistic:.3f}")
    _save_npz(args.scores, label=lab, spectype=spt, held=held, dv_kms=dv, sep=sep, **scores)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out} and {args.scores}")


if __name__ == "__main__":
    main()
