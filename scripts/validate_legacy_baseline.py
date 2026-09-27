#!/usr/bin/env python
"""Acceptance gates for a Legacy-PSF baseline bundle (docs/PLAN_FIELD_MODEL.md, section 7).

Everything is measured on the ``test`` cells (the archived held-out blocks),
which no fitting step read. Gates:

ranking      DESI companions accepted by the bundle's own selection, scored by
             the baseline and by the current multi-survey model on identical
             rows; paired spatial bootstrap (nside-8 blocks inside the test
             cells) of Delta AUC for same-z vs wrong-z (ln R), quasar vs star
             and quasar vs PSF galaxy (ln BF); contamination at a fixed quasar
             retention; strata by hemisphere and magnitude.
counts       Sigma_B of the bundle against the counted PSF field density of
             the test cones (usable area from the brick images), per 0.5 mag
             bin and pooled, with a cell-bootstrap interval.
tails        observed vs expected numbers of test field sources in bins of
             their distance from the nearest field component, the expectation
             by drawing from the fitted conditional density (mixture + t) with
             each object's own noise and observed bands.
normalise    the PSF prior integral equals C x the weighted PSF count / area.
retention    PSF fraction of the all-morphology quasars by z, magnitude, and
             of the labelled companions by separation.
numerics     a band made uninformative changes ln R by < 1e-3.

    python scripts/validate_legacy_baseline.py --bundle models/legacy_psf/<id>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from validate_pairs import auc_rank, validation_labels  # noqa: E402

GATES = dict(delta_auc_lower=-0.01, counts_pooled=0.05, counts_bin=0.10,
             tails_factor=2.0, tails_sigma=3.0, tails_min_expected=20.0,
             numerics_max=1e-3, normalisation_rel=1e-3)


def col(rows, key, n):
    return np.array([getattr(r, key) if r is not None else np.nan for r in rows], float) \
        if rows is not None else np.full(n, np.nan)


def block_bootstrap(blocks, fn, n_boot=1000, seed=0):
    """Percentile interval of fn(rows) resampling blocks with replacement."""
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    members = [np.flatnonzero(blocks == b) for b in ub]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, ub.size, ub.size)
        idx = np.concatenate([members[i] for i in pick])
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals.size else [np.nan, np.nan]


# -- ranking -----------------------------------------------------------------------

def companions(cfg, bl, args):
    from qso_pcolor.background import galactic_healpix
    from qso_pcolor.data import galactic_from_equatorial
    root = Path(cfg["data_dir"])
    p = dict(np.load(cfg["pairs"], allow_pickle=False))
    m = dict(np.load(root / "companions.npz", allow_pickle=False))
    j = m.pop("pair_row_to_match")
    rows = {k: np.asarray(v)[j] for k, v in m.items() if np.ndim(v) == 1 and k not in ("input_ra", "input_dec")}
    rows["ra"], rows["dec"] = np.asarray(p["comp_ra"], float), np.asarray(p["comp_dec"], float)
    label = validation_labels(p["comp_spectype"], p["dv_kms"], args.half_width_kms)
    sep, zp = np.asarray(p["sep_arcsec"], float), np.asarray(p["z_primary"], float)
    zsupp = bl.models["south"].qso.in_support(zp)
    keep = (sep >= args.min_sep) & (sep <= args.max_sep) & zsupp & (label != "invalid_redshift")
    nq = np.flatnonzero(keep & (label == "non_qso"))
    if nq.size > args.max_non_qso:
        drop = np.random.default_rng(args.seed).choice(nq, nq.size - args.max_non_qso, replace=False)
        keep[drop] = False
    idx = np.flatnonzero(keep)
    rows = {k: v[idx] for k, v in rows.items()}
    l, b = galactic_from_equatorial(rows["ra"], rows["dec"])
    cell = galactic_healpix(l, b, cfg["cell_nside"])
    roles = {int(k): v for k, v in bl.manifest["partition"].items()}
    return dict(rows=rows, label=label[idx], spectype=np.char.strip(np.asarray(p["comp_spectype"])[idx].astype(str)),
                sep=sep[idx], zp=zp[idx], dv=np.abs(np.asarray(p["dv_kms"], float)[idx]),
                l=l, b=b, cell=cell, role=np.array([roles.get(int(c), "none") for c in cell]),
                block8=galactic_healpix(l, b, 8), pair_index=idx)


def score_current(c, eligible, hemi, args):
    """The current multi-survey model on the same rows, same native bands."""
    from qso_pcolor.legacy import legacy_photometry
    from qso_pcolor.multisurvey import MultiSurveyModel, MultiSurveyOutlier, load_priors
    from qso_pcolor.qso_model import RedshiftMatch
    from qso_pcolor.score import BlendPolicy
    ms = MultiSurveyModel.load(args.current_model)
    pri = load_priors(args.current_priors, ms)
    out = MultiSurveyOutlier.load(args.current_outlier)
    n = len(c["label"])
    res = [None] * n
    for h in ("south", "north"):
        use = np.flatnonzero(eligible & (hemi == h))
        if not use.size:
            continue
        sub = {k: v[use] for k, v in c["rows"].items()}
        ph = legacy_photometry(sub, h)
        sc = ms.score(ph, z_primary=c["zp"][use], match=RedshiftMatch(half_width_kms=args.half_width_kms),
                      min_bands=2, l_deg=c["l"][use], b_deg=c["b"][use], priors=pri, outlier=out,
                      blend_policy=BlendPolicy(min_separation_arcsec=args.min_sep,
                                               max_fracflux=args.max_fracflux),
                      separation_arcsec=c["sep"][use], fracflux=sub["fracflux_r"])
        for i, s in zip(use, sc):
            res[i] = s
    return res


def ranking_metrics(c, scores, mask):
    lab, spt = c["label"], c["spectype"]
    logr = col(scores, "log_r_per_unit_z", len(lab))
    logbf = col(scores, "log_bayes_factor_qz_bkg", len(lab))
    pz = col(scores, "p_zmatch_given_qso", len(lab))
    ok = mask & np.array([s is not None and s.status == "ok" for s in scores])

    def aucs(i):
        # i may repeat (bootstrap): index, never mask, so multiplicities are kept
        i = np.asarray(i, int)
        i = i[ok[i]]
        L, T = lab[i], spt[i]
        S, F, N = L == "same_z", L == "field_q", L == "non_qso"
        lr, lbf, pzi, dv = logr[i], logbf[i], pz[i], c["dv"][i]
        return dict(same_vs_field_log_r=auc_rank(lr[S], lr[F]),
                    same_vs_field_p_zmatch=auc_rank(pzi[S], pzi[F]),
                    same_vs_hard_log_r=auc_rank(lr[S], lr[F & (dv < 6000)]),
                    quasar_vs_star_log_bf=auc_rank(lbf[S | F], lbf[N & (T == "STAR")]),
                    quasar_vs_galaxy_log_bf=auc_rank(lbf[S | F], lbf[N & (T == "GALAXY")]),
                    n=[int(S.sum()), int(F.sum()), int((N & (T == "STAR")).sum()),
                       int((N & (T == "GALAXY")).sum())])
    return aucs, ok, logr, logbf


def contamination_at_retention(logbf, lab, spt, m, retention=0.9):
    q = m & np.isin(lab, ("same_z", "field_q")) & np.isfinite(logbf)
    thr = np.quantile(logbf[q], 1 - retention)
    out = {}
    for name, s in (("star", "STAR"), ("galaxy", "GALAXY")):
        n = m & (lab == "non_qso") & (spt == s) & np.isfinite(logbf)
        out[name] = float(np.mean(logbf[n] >= thr)) if n.any() else np.nan
    return dict(threshold=float(thr), **out)


# -- counts, tails -----------------------------------------------------------------

def test_field(cfg, bl, h):
    """Accepted PSF field sources of the test cones of one hemisphere, with areas."""
    from qso_pcolor.legacy import legacy_photometry
    root = Path(cfg["data_dir"])
    design = json.loads((root / "design.json").read_text())
    areas = json.loads((root / "areas.json").read_text())
    parts = []
    for c in design["cones"]:
        if c["role"] != "test" or c["hemisphere"] != h:
            continue
        r = dict(np.load(root / "cones" / f"cone_{c['cone']:03d}.npz", allow_pickle=False))
        d = bl.selection.decide(r)
        keep = d["accepted"] & ~r["known_quasar"].astype(bool) & (d["hemisphere"] == h)
        kq = d["accepted"] & r["known_quasar"].astype(bool) & (d["hemisphere"] == h)
        with np.errstate(divide="ignore", invalid="ignore"):
            mag_r = 22.5 - 2.5 * np.log10(np.asarray(r["flux_r"], float))
        lo, hi = cfg["fit"]["candidate_ref_range"]
        parts.append(dict(rows={k: np.asarray(v)[keep] for k, v in r.items() if np.ndim(v) == 1},
                          cone=c["cone"], cell=c["cell"], area=areas[str(c["cone"])]["area_deg2"],
                          cell_area=design["cells"][str(c["cell"])]["by_hemisphere"][h],
                          n_known_candidate=int((kq & (mag_r >= lo) & (mag_r < hi)).sum())))
    model = bl.models[h]
    for p in parts:
        ph = legacy_photometry(p["rows"], h)
        p["features"] = model.transform(ph)
    return parts


def counts_gate(bl, h, parts, edges_lo, edges_hi):
    """Counted (known quasars removed) vs predicted Sigma_B + (1 - kappa_cone) Sigma_Q."""
    label = bl.models[h].transform.bands[1]
    gq, bd = bl.priors[h][label]
    edges = bd.mag_edges
    inside = (edges[:-1] >= edges_lo - 1e-9) & (edges[1:] <= edges_hi + 1e-9)
    mc = 0.5 * (edges[:-1] + edges[1:])
    sig_b = bd(mc, np.zeros(mc.size), np.full(mc.size, 90.0))
    sig_q = (gq.sigma * np.diff(gq.z_edges)[:, None]).sum(0)
    q_total = float(np.sum((sig_q * np.diff(edges))[inside]))
    cells = sorted({p["cell"] for p in parts})
    per_cell = {}
    for cc in cells:
        ps = [p for p in parts if p["cell"] == cc]
        n = sum(np.histogram(p["features"].x[:, 1], edges)[0] for p in ps)
        a = sum(p["area"] for p in ps)
        # recognised fraction of each test cone, as in the fit (candidate range, capped at 1)
        unk = sum(p["area"] * (1 - min(1.0, p["n_known_candidate"] / (p["area"] * q_total))) for p in ps) / a
        per_cell[cc] = (n, a, ps[0]["cell_area"], sig_b + unk * sig_q)

    def density(cs, k):
        w = np.array([per_cell[c][2] for c in cs])
        d = np.array([per_cell[c][0] / per_cell[c][1] / np.diff(edges) if k == 0 else per_cell[c][3]
                      for c in cs])
        return (w[:, None] * d).sum(0) / w.sum()

    obs, pred = density(cells, 0), density(cells, 1)
    rng = np.random.default_rng(1)
    boots = np.array([density([cells[i] for i in rng.integers(0, len(cells), len(cells))], 0)
                      for _ in range(1000)])
    pooled_pred = float(np.sum((pred * np.diff(edges))[inside]))
    pooled_obs = float(np.sum((obs * np.diff(edges))[inside]))
    pooled_boot = (boots * np.diff(edges))[:, inside].sum(1)
    n_obs = sum(per_cell[c][0] for c in cells)
    rows = [dict(mag=[float(edges[i]), float(edges[i + 1])], predicted=float(pred[i]),
                 observed=float(obs[i]), ratio=float(obs[i] / pred[i]) if pred[i] > 0 else np.nan,
                 ci=np.percentile(boots[:, i] / pred[i], [2.5, 97.5]).tolist() if pred[i] > 0 else None,
                 n=int(n_obs[i]))
            for i in np.flatnonzero(inside)]
    return dict(n_cells=len(cells), n_cones=len(parts), bins=rows,
                pooled=dict(predicted=pooled_pred, observed=pooled_obs, ratio=pooled_obs / pooled_pred,
                            ci=np.percentile(pooled_boot / pooled_pred, [2.5, 97.5]).tolist()))


def _draw_mixture(mix, x, S, a, o, rng):
    """Observed bands o given observed reference a, from a Gaussian mixture with noise S."""
    V = mix.covs + S[None]
    lw = np.log(mix.weights) - 0.5 * ((x[a] - mix.means[:, a]) ** 2 / V[:, a, a] + np.log(V[:, a, a]))
    pw = np.exp(lw - lw.max()); pw /= pw.sum()
    k = rng.choice(mix.n_components, p=pw)
    mu = mix.means[k, o] + V[k, o, a] / V[k, a, a] * (x[a] - mix.means[k, a])
    C = V[k][np.ix_(o, o)] - np.outer(V[k, o, a], V[k, o, a]) / V[k, a, a]
    return rng.multivariate_normal(mu, C)


_LG = np.linspace(-25.0, 4.5, 400)


def _draw_t_exact(mean, scale, nu, x, S, a, o, rng):
    """Same, from the t as a Gaussian scale mixture convolved with the noise (noise='exact')."""
    from scipy.special import gammaln
    g = np.exp(_LG)
    va = scale[a, a] / g + S[a, a]
    lp = (0.5 * nu * _LG - 0.5 * nu * g + _LG - 0.5 * np.log(va) - 0.5 * (x[a] - mean[a]) ** 2 / va)
    p = np.exp(lp - lp.max()); p /= p.sum()
    gi = g[rng.choice(g.size, p=p)]
    V = scale / gi + S
    mu = mean[o] + V[o, a] / V[a, a] * (x[a] - mean[a])
    C = V[np.ix_(o, o)] - np.outer(V[o, a], V[o, a]) / V[a, a]
    return rng.multivariate_normal(mu, C)


def draw_observable(model, outlier, gq, sigma_b_at, x, cov, obs, anchor, kappa, eta, rng):
    """One draw of the observed colours of every source from the full observable model.

    Class by its prior share at the source's reference magnitude: unrecognised
    quasar (1 - kappa) Sigma_Q against non-quasar Sigma_B; within non-quasars the
    unmodelled term with probability eta. Quasar redshift by Sigma_Q(z, u_r).
    """
    n, d = x.shape
    out = x.copy()
    u = x[:, anchor]
    sq_z = np.array([np.interp(u, gq.mag_centres, gq.sigma[j]) for j in range(gq.z_centres.size)]).T
    sq_z = sq_z * np.diff(gq.z_edges)[None]
    s_unk = (1 - kappa) * sq_z.sum(1)
    p_q = s_unk / (s_unk + sigma_b_at)
    student = outlier.family == "student_t" and outlier.noise == "exact"
    for i in range(n):
        o = np.flatnonzero(obs[i] & (np.arange(d) != anchor))
        if not o.size:
            continue
        r = rng.random()
        if r < p_q[i]:
            j = rng.choice(gq.z_centres.size, p=sq_z[i] / sq_z[i].sum())
            out[i, o] = _draw_mixture(model.qso.mixtures[j], x[i], cov[i], anchor, o, rng)
        elif rng.random() < eta[i]:
            if not student:
                raise ValueError("tails simulator implements the exact Student-t only")
            out[i, o] = _draw_t_exact(outlier.mean, outlier.cov, outlier.nu, x[i], cov[i], anchor, o, rng)
        else:
            out[i, o] = _draw_mixture(model.background, x[i], cov[i], anchor, o, rng)
    return out, p_q


def tails_gate(bl, h, parts, n_draws, rng, lo, hi):
    """Observed test field sources (known quasars removed) against draws of the full
    observable model, in bins of distance from the nearest non-quasar component."""
    from qso_pcolor.multisurvey import _conditional_min_mahalanobis
    model, outlier = bl.models[h], bl.outliers[h]
    f = [p["features"] for p in parts]
    x = np.concatenate([q.x for q in f]); cov = np.concatenate([q.cov for q in f])
    obs = np.concatenate([q.observed for q in f])
    ir = 1
    label = model.transform.bands[ir]
    gq, bd = bl.priors[h][label]
    q_total = float(np.sum(((gq.sigma * np.diff(gq.z_edges)[:, None]).sum(0) * np.diff(gq.mag_edges))[
        (gq.mag_edges[:-1] >= lo - 1e-9) & (gq.mag_edges[1:] <= hi + 1e-9)]))
    kappa = np.concatenate([np.full(p["features"].n_obs, min(1.0, p["n_known_candidate"] / (p["area"] * q_total)))
                            for p in parts])
    dom = obs[:, ir] & (x[:, ir] >= lo) & (x[:, ir] < hi) & (obs.sum(1) >= 2)
    x, cov, obs, kappa = x[dom], cov[dom], obs[dom], kappa[dom]
    sig_b = bd(x[:, ir], np.zeros(len(x)), np.full(len(x), 90.0))
    eta = outlier.conditional(ir, label, model.qso.system).fraction_at(x[:, ir])
    sig = _conditional_min_mahalanobis([model.background], x, cov, obs, ir)
    edges = np.array([0, 1, 2, 3, 4, 6, 10, np.inf])
    n_obs = np.histogram(sig, edges)[0]
    sims = []
    for _ in range(n_draws):
        xd, p_q = draw_observable(model, outlier, gq, sig_b, x, cov, obs, ir, kappa, eta, rng)
        sims.append(np.histogram(_conditional_min_mahalanobis([model.background], xd, cov, obs, ir), edges)[0])
    sims = np.array(sims, float)
    exp = sims.mean(0)
    rows = []
    for i in range(edges.size - 1):
        e = exp[i]
        s_ = np.sqrt(e + sims[:, i].var() / n_draws) if e > 0 else np.nan
        z = (n_obs[i] - e) / s_ if e > 0 else np.nan
        fail = bool(e >= GATES["tails_min_expected"] and
                    max(n_obs[i] / e, e / max(n_obs[i], 1e-9)) > GATES["tails_factor"]
                    and abs(z) > GATES["tails_sigma"])
        rows.append(dict(sigma=[float(edges[i]), float(edges[i + 1])], observed=int(n_obs[i]),
                         expected=float(e), z=float(z) if np.isfinite(z) else None, fail=fail))
    return dict(n=int(dom.sum()), expected_unrecognised_quasars=float(p_q.sum()), n_draws=n_draws,
                bins=rows, pass_=not any(r["fail"] for r in rows),
                note="observed (unweighted) vs draws of the full observable model: non-quasars, "
                     "unmodelled term and unrecognised quasars by their prior shares")


# -- main --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/legacy_baseline.json"))
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--current-model", type=Path, default=Path("models/multisurvey.json"))
    ap.add_argument("--current-priors", type=Path, default=Path("models/multisurvey_priors.json"))
    ap.add_argument("--current-outlier", type=Path, default=Path("models/multisurvey_outlier.json"))
    ap.add_argument("--half-width-kms", type=float, default=3000.0)
    ap.add_argument("--min-sep", type=float, default=3.0)
    ap.add_argument("--max-sep", type=float, default=30.0)
    ap.add_argument("--max-fracflux", type=float, default=0.2)
    ap.add_argument("--max-non-qso", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tail-draws", type=int, default=5)
    ap.add_argument("--convergence", type=int, default=200,
                    help="continue each selected field fit this many EM iterations (0: skip)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())

    from qso_pcolor.baseline import LegacyBaseline
    from qso_pcolor.qso_model import RedshiftMatch
    from qso_pcolor.score import BlendPolicy

    bl = LegacyBaseline.load(args.bundle)
    lo, hi = bl.candidate_range
    report = dict(bundle=bl.bundle_id, gates=GATES, built=time.strftime("%Y-%m-%d"))
    t0 = time.time()

    # -- ranking
    c = companions(cfg, bl, args)
    policy = BlendPolicy(min_separation_arcsec=args.min_sep, max_fracflux=args.max_fracflux)
    base, dec = bl.score_rows(c["rows"], z_primary=c["zp"],
                              match=RedshiftMatch(half_width_kms=args.half_width_kms),
                              separation_arcsec=c["sep"], fracflux=c["rows"]["fracflux_r"],
                              blend_policy=policy)
    eligible = dec["eligible"]
    cur = score_current(c, eligible, dec["hemisphere"], args)
    print(f"companions: {len(eligible):,} rows, {eligible.sum():,} eligible "
          f"({time.time() - t0:.0f} s)", flush=True)
    report["companion_eligibility"] = dict(zip(*[x.tolist() for x in np.unique(dec["reason"], return_counts=True)]))
    held = c["role"] == "test"
    fb, okb, logr_b, logbf_b = ranking_metrics(c, base, eligible)
    fc, okc, logr_c, logbf_c = ranking_metrics(c, cur, eligible)
    both = okb & okc
    rank = {}
    for name, m in (("test", held & both), ("test_south", held & both & (dec["hemisphere"] == "south")),
                    ("test_north", held & both & (dec["hemisphere"] == "north")),
                    ("all_cells", both)):
        idx = np.flatnonzero(m)
        mb, mc = fb(idx), fc(idx)
        entry = dict(baseline=mb, current=mc)
        if name == "test":
            blocks = c["block8"][idx]
            for k in ("same_vs_field_log_r", "quasar_vs_star_log_bf", "quasar_vs_galaxy_log_bf",
                      "same_vs_field_p_zmatch"):
                d = mb[k] - mc[k]
                ci = block_bootstrap(blocks, lambda i, k=k: fb(idx[i])[k] - fc(idx[i])[k], n_boot=400)
                entry[f"delta_{k}"] = dict(value=d, ci95=ci,
                                           pass_=bool(ci[0] > GATES["delta_auc_lower"])
                                           if k != "same_vs_field_p_zmatch" else None)
            entry["contamination_at_90pc_retention"] = dict(
                baseline=contamination_at_retention(logbf_b, c["label"], c["spectype"], m),
                current=contamination_at_retention(logbf_c, c["label"], c["spectype"], m))
        rank[name] = entry
        print(f"[{name}] n(same, field, star, gal) {mb['n']}: lnR AUC {mb['same_vs_field_log_r']:.3f} "
              f"vs {mc['same_vs_field_log_r']:.3f}; star {mb['quasar_vs_star_log_bf']:.3f} vs "
              f"{mc['quasar_vs_star_log_bf']:.3f}; PSF gal {mb['quasar_vs_galaxy_log_bf']:.3f} vs "
              f"{mc['quasar_vs_galaxy_log_bf']:.3f}", flush=True)
    # magnitude strata on the test cells
    u = np.array([s.ref_mag if s is not None else np.nan for s in base])
    rank["test_by_mag"] = {}
    for a, b in ((17, 20), (20, 21.5), (21.5, 22.5)):
        idx = np.flatnonzero(held & both & (u >= a) & (u < b))
        rank["test_by_mag"][f"{a}-{b}"] = dict(baseline=fb(idx), current=fc(idx))
    report["ranking"] = rank
    print("delta AUC (baseline - current), test cells: " + "; ".join(
        f"{k[6:]} {v['value']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]"
        for k, v in rank["test"].items() if k.startswith("delta_")), flush=True)

    # -- retention of the morphology cut
    from qso_pcolor.legacy import LegacySelection
    sel_all = LegacySelection("all", bl.selection.maskbits_zero, bl.selection.reference,
                              bl.selection.version)
    d_all = sel_all.decide(c["rows"])
    qq = np.isin(c["label"], ("same_z", "field_q")) & d_all["accepted"]
    ret = {}
    for a, b in ((3, 5), (5, 10), (10, 20), (20, 30)):
        s = qq & (c["sep"] >= a) & (c["sep"] < b)
        ret[f"sep {a}-{b}"] = dict(n=int(s.sum()), psf_fraction=float(dec["morphology"][s].tolist().count("point") / max(s.sum(), 1)))
    report["retention_companions"] = ret
    gal = (c["label"] == "non_qso") & (c["spectype"] == "GALAXY") & d_all["accepted"]
    report["galaxies_psf_fraction"] = float(np.mean(dec["morphology"][gal] == "point"))

    # -- counts and tails per hemisphere
    rng = np.random.default_rng(args.seed)
    report["counts"], report["tails"] = {}, {}
    for h in bl.models:
        parts = test_field(cfg, bl, h)
        if not parts:
            continue
        report["counts"][h] = counts_gate(bl, h, parts, lo, hi)
        p = report["counts"][h]["pooled"]
        print(f"[{h}] counts: pooled observed/predicted {p['ratio']:.3f} "
              f"[{p['ci'][0]:.3f}, {p['ci'][1]:.3f}] over {report['counts'][h]['n_cells']} test cells",
              flush=True)
        report["tails"][h] = tails_gate(bl, h, parts, args.tail_draws, rng, lo, hi)
        print(f"[{h}] tails: " + "; ".join(f"{r['sigma'][0]:g}-{r['sigma'][1]:g}: {r['observed']:.0f}/"
                                          f"{r['expected']:.0f}{' FAIL' if r['fail'] else ''}"
                                          for r in report["tails"][h]["bins"]), flush=True)

    # -- numerics: an uninformative band must not move ln R
    from qso_pcolor.legacy import legacy_photometry
    idx = np.flatnonzero(eligible & held)[:500]
    h = "south"
    idx = idx[dec["hemisphere"][idx] == h]
    sub = {k: v[idx] for k, v in c["rows"].items()}
    ph = legacy_photometry(sub, h)
    m = bl.models[h]
    kw = dict(z_primary=c["zp"][idx], match=RedshiftMatch(half_width_kms=args.half_width_kms),
              min_bands=2, l_deg=c["l"][idx], b_deg=c["b"][idx], priors=bl.priors[h],
              outlier=bl.outliers[h])
    ph_drop = legacy_photometry(sub, h); ph_drop.flux[:, 4] = np.nan; ph_drop.variance[:, 4] = np.inf
    ph_huge = legacy_photometry(sub, h); ph_huge.variance[:, 4] = np.where(
        ph_huge.observed[:, 4], ph_huge.variance[:, 4] * 1e12, np.inf)
    a = np.array([s.log_r_per_unit_z for s in m.score(ph_drop, **kw)])
    bb = np.array([s.log_r_per_unit_z for s in m.score(ph_huge, **kw)])
    dd = np.abs(a - bb)[np.isfinite(a - bb)]
    report["numerics"] = dict(n=int(dd.size), max_abs_delta_log_r=float(dd.max()),
                              pass_=bool(dd.max() < GATES["numerics_max"]))
    print(f"numerics: uninformative W2 vs dropped W2, max |d lnR| {dd.max():.2e}", flush=True)

    # -- normalisation identity, recomputed from the data
    from build_legacy_baseline_sample import cell_areas
    from fit_legacy_baseline import (integrate, parent_cells_and_weights, quasar_rows,
                                     raw_qso_density, roles_of)
    from qso_pcolor.legacy import load_bricks
    root = Path(cfg["data_dir"])
    sample_cfg = json.loads(Path(cfg["sample_config"]).read_text())
    q, qrows = quasar_rows(cfg, root)
    roles = roles_of(json.loads((root / "design.json").read_text()))
    _, observed = cell_areas(cfg)
    z_edges, _, w_q, pcells, qpix, pix_area = parent_cells_and_weights(
        cfg, sample_cfg, q, roles, observed, load_bricks(cfg["bricks"]))
    C = bl.manifest["completeness"]["C"]
    norm = {}
    for h, m in bl.models.items():
        label = m.transform.bands[1]
        gq, _ = bl.priors[h][label]
        for name, sel in (("point", bl.selection), ("all", sel_all)):
            d = sel.decide(qrows)
            use = d["accepted"] & (d["hemisphere"] == h)
            u = m.transform(legacy_photometry({k: np.asarray(v)[use] for k, v in qrows.items()
                                               if np.ndim(v) == 1}, h)).x[:, 1]
            raw = raw_qso_density(pcells[h][qpix[use]] & np.isfinite(u), q["target_zspec"][use], u,
                                  w_q[use], z_edges, gq.mag_edges, pcells[h].sum() * pix_area)
            norm.setdefault(h, {})[name] = integrate(raw, z_edges, gq.mag_edges, lo, hi)
        prior_total = integrate(gq.sigma,
                                z_edges, gq.mag_edges, lo, hi)
        rel = abs(prior_total / (C * norm[h]["point"]) - 1)
        norm[h].update(prior_total=prior_total, C=C, relative_error=rel,
                       retention_point_over_all=norm[h]["point"] / norm[h]["all"],
                       pass_=bool(rel < GATES["normalisation_rel"]))
        print(f"[{h}] normalisation: prior {prior_total:.2f} vs C x PSF {C * norm[h]['point']:.2f} "
              f"deg^-2 (rel {rel:.1e}); PSF/all {norm[h]['retention_point_over_all']:.3f}", flush=True)
    report["normalisation"] = norm

    # -- retention of the PSF cut among quasars, by redshift and magnitude
    ret_z = {}
    zq = q["target_zspec"]
    d_pt, d_al = bl.selection.decide(qrows), sel_all.decide(qrows)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = 22.5 - 2.5 * np.log10(np.asarray(qrows["flux_r"], float))
    for a, b in ((0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.4)):
        s_ = d_al["accepted"] & (zq >= a) & (zq < b) & (mag >= lo) & (mag < hi)
        ret_z[f"z {a}-{b}"] = dict(n=int(s_.sum()), psf_fraction=float(d_pt["accepted"][s_].mean()))
    report["retention_quasars_by_z"] = ret_z
    print("PSF retention by z: " + "; ".join(f"{k} {v['psf_fraction']:.2f}" for k, v in ret_z.items()))

    # -- does the quasar colour model depend on brightness? conditional mean colours
    #    (band minus r) at r = 18 and r = 22 in each slice, noiseless
    def cond_mean(mix, a, ua):
        lw = np.log(mix.weights) - 0.5 * ((ua - mix.means[:, a]) ** 2 / mix.covs[:, a, a]
                                          + np.log(mix.covs[:, a, a]))
        p_ = np.exp(lw - lw.max()); p_ /= p_.sum()
        mu = mix.means + mix.covs[:, :, a] / mix.covs[:, a, a][:, None] * (ua - mix.means[:, a])[:, None]
        return (p_[:, None] * mu).sum(0) - ua
    report["quasar_magnitude_dependence"] = {}
    for h, m in bl.models.items():
        shifts = np.array([cond_mean(mx, 1, 22.0) - cond_mean(mx, 1, 18.0) for mx in m.qso.mixtures])
        shifts = np.delete(shifts, 1, axis=1)                       # colours: g, z, W1, W2 minus r
        report["quasar_magnitude_dependence"][h] = dict(
            colours=[b.split(":")[1] + "-r" for i, b in enumerate(m.transform.bands) if i != 1],
            z=m.qso.z_centres.tolist(), shift_22_minus_18=shifts.tolist(),
            median_abs_shift=np.median(np.abs(shifts), 0).tolist(), max_abs_shift=np.abs(shifts).max(0).tolist())
        print(f"[{h}] quasar colours, r=22 minus r=18: median |shift| "
              + ", ".join(f"{c} {v:.3f}" for c, v in zip(report["quasar_magnitude_dependence"][h]["colours"],
                                                         report["quasar_magnitude_dependence"][h]["median_abs_shift"]))
              + " mag", flush=True)

    # -- fit records: convergence, start agreement, K
    report["fits"] = {}
    for h, m in bl.models.items():
        ff = m.meta["field_fit"]
        report["fits"][h] = dict(selected_k=ff["selected_k"], largest_k_won=ff["largest_k_won"],
                                 start_spread_at_selected_k=ff["start_spread_at_selected_k"],
                                 all_converged=all(t["converged"] for t in ff["trials"]),
                                 quasar_slices_all_converged=all(
                                     t["converged"] for sl in m.qso.meta["per_slice"] for t in sl["trials"]),
                                 unrecognised=ff.get("unrecognised_quasars", {}).get("history"))
    if args.convergence:
        import fit_legacy_baseline as F
        from qso_pcolor.multisurvey import conditional_log_prob
        from qso_pcolor.xd import fit_xd
        rows_f, cones_f, cone_area, w_cone, cone_report, design_f = F.load_field(cfg, root, bl.selection)
        for h, m in bl.models.items():
            pf = F.prepare_field(h, rows_f, cones_f, w_cone, bl.selection, cfg["fit"], cfg, roles)
            f, fr, sr = pf["features"], pf["fit_rows"], pf["sel_rows"]
            mix = m.background
            # the weights of the final fit: 1 - P(unrecognised quasar) under the shipped model
            uq = m.meta["field_fit"]["unrecognised_quasars"]
            kap = {int(k): v for k, v in uq["kappa_by_cone"].items()}
            gq_h = bl.priors[h][m.transform.bands[1]][0]
            need = fr | sr
            lq_, sq_ = F.log_quasar_colour_density(m.qso, gq_h, f, need, 1)
            r_ = np.zeros(f.n_obs)
            r_[need] = F.unrecognised_probability(pf, mix, lq_, sq_, kap, np.array(uq["sigma_b"]),
                                                  gq_h.mag_edges, need)
            w_nq = pf["weights"] * (1 - r_)
            res = fit_xd(f.x[fr], f.cov[fr], observed=f.observed[fr], weights=w_nq[fr], init=mix,
                         max_iter=args.convergence, tol=0.0,
                         regularization=m.meta["field_fit"].get("selected_regularization",
                                                                cfg["fit"]["regularization"]),
                         labels=f.labels, n_threads=cfg["fit"]["n_threads"])
            sc = lambda mm: float(np.average(conditional_log_prob(mm, f.x[sr], f.cov[sr], f.observed[sr], 1),  # noqa: E731
                                             weights=w_nq[sr]))
            ds = sc(res.mixture) - sc(mix)
            # ln R of test companions with the continued field mixture
            idx_c = np.flatnonzero(eligible & held & (dec["hemisphere"] == h))
            sub = {k: v[idx_c] for k, v in c["rows"].items()}
            ph = legacy_photometry(sub, h)
            kw = dict(z_primary=c["zp"][idx_c], match=RedshiftMatch(half_width_kms=args.half_width_kms),
                      min_bands=2, l_deg=c["l"][idx_c], b_deg=c["b"][idx_c], priors=bl.priors[h],
                      outlier=bl.outliers[h])
            from dataclasses import replace as _replace
            a0 = np.array([s_.log_r_per_unit_z for s_ in m.score(ph, **kw)])
            a1 = np.array([s_.log_r_per_unit_z for s_ in _replace(m, background=res.mixture).score(ph, **kw)])
            dl = np.abs(a1 - a0)[np.isfinite(a1 - a0)]
            report["fits"][h]["continuation"] = dict(iterations=args.convergence, delta_select_score=ds,
                                                     median_abs_delta_log_r=float(np.median(dl)),
                                                     pass_=bool(abs(ds) < 0.01 and np.median(dl) < 0.02))
            print(f"[{h}] continuation {args.convergence} it: select score {ds:+.4f} nats/obj, "
                  f"median |d lnR| {np.median(dl):.4f}", flush=True)

    # -- verdict: the hard requirements (docs/PLAN_FIELD_MODEL.md section 9)
    t = rank["test"]
    checks = {
        "delta_auc_same_vs_field": t["delta_same_vs_field_log_r"]["pass_"],
        "delta_auc_star": t["delta_quasar_vs_star_log_bf"]["pass_"],
        "delta_auc_psf_galaxy": t["delta_quasar_vs_galaxy_log_bf"]["pass_"],
        "numerics": report["numerics"]["pass_"],
        "normalisation": all(v["pass_"] for v in norm.values()),
        "fits_converged": all(v["all_converged"] and v["quasar_slices_all_converged"]
                              for v in report["fits"].values()),
        "tails_south": report["tails"].get("south", {}).get("pass_", False),
    }
    if args.convergence:
        checks["continuation"] = all(v["continuation"]["pass_"] for v in report["fits"].values()
                                     if "continuation" in v)
    report["verdict"] = dict(checks=checks, pass_=all(checks.values()),
                             reported_not_gated=["counts (spatial scatter of a global model)",
                                                 "start agreement", "north ranking (39 same-z)"],
                             status=bl.manifest.get("status"))
    out = args.out or Path(cfg["data_dir"]) / f"validation_{bl.bundle_id}.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    print("verdict: " + ("PASS" if report["verdict"]["pass_"] else "FAIL") + "  " +
          ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in checks.items()))
    print(f"wrote {out} ({time.time() - t0:.0f} s)")
    sys.exit(0 if report["verdict"]["pass_"] else 1)


if __name__ == "__main__":
    main()
