#!/usr/bin/env python
"""Validate a Legacy-PSF XDQSO-style bundle (docs/BASELINE_PLAN.md, step 5).

Everything is measured on the ``test`` cells (the archived held-out blocks),
which no fitting step read. Hard gates (non-zero exit if any fails):

ranking      DESI companions accepted by the bundle's selection, scored by the
             baseline, the ORIGINAL archived Legacy model (dereddened, all
             morphologies, south) and the CURRENT multi-survey model, on identical
             rows. Paired nside-8 block bootstrap: the lower 95 % bound of
             Delta AUC (baseline - reference) > -0.01 for same-z vs wrong-z (ln R),
             quasar vs star and quasar vs PSF galaxy (ln BF), against both.
numerics     an uninformative W2 changes ln R by < 1e-3.
normalise    the PSF prior integral equals C x the weighted PSF count / area.
continuation each field bin's selected mixture, continued 200 EM iterations under
             its own non-quasar weights: held-out mean log density changes < 0.01.

Not yet implemented here: the coarse tails check (listed as such in the report).
Reported, not gated: counts, retention by z, per-hemisphere / magnitude strata.

    python scripts/validate_xdqso_baseline.py --bundle models/legacy_psf_xdqso/<id>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from validate_pairs import auc_rank, validation_labels  # noqa: E402
from validate_legacy_baseline import (block_bootstrap, contamination_at_retention,  # noqa: E402
                                      ranking_metrics, score_current)

GATES = dict(delta_auc_lower=-0.01, numerics_max=1e-3, normalisation_rel=1e-3,
             continuation=0.01, tails_factor=2.0, tails_sigma=3.0, tails_min_expected=20.0)


def companions(cfg, bl, args):
    from qso_pcolor.background import galactic_healpix
    from qso_pcolor.data import galactic_from_equatorial
    from qso_pcolor.store import load
    p = load("pairs_desi_dr1")
    m = load("pairs_desi_dr1_dr9")
    j = m.pop("pair_row_to_match")
    rows = {k: np.asarray(v)[j] for k, v in m.items() if np.ndim(v) == 1 and k not in ("input_ra", "input_dec")}
    rows["ra"], rows["dec"] = np.asarray(p["comp_ra"], float), np.asarray(p["comp_dec"], float)
    label = validation_labels(p["comp_spectype"], p["dv_kms"], args.half_width_kms)
    sep, zp = np.asarray(p["sep_arcsec"], float), np.asarray(p["z_primary"], float)
    zsupp = bl.parts["south"]["qso"].in_support(zp)
    keep = (sep >= args.min_sep) & (sep <= args.max_sep) & zsupp & (label != "invalid_redshift")
    nq = np.flatnonzero(keep & (label == "non_qso"))
    if nq.size > args.max_non_qso:
        keep[np.random.default_rng(args.seed).choice(nq, nq.size - args.max_non_qso, replace=False)] = False
    idx = np.flatnonzero(keep)
    rows = {k: v[idx] for k, v in rows.items()}
    l, b = galactic_from_equatorial(rows["ra"], rows["dec"])
    cell = galactic_healpix(l, b, cfg["cell_nside"])
    roles = {int(k): v for k, v in bl.manifest["partition"].items()}
    return dict(rows=rows, label=label[idx],
                spectype=np.char.strip(np.asarray(p["comp_spectype"])[idx].astype(str)),
                sep=sep[idx], zp=zp[idx], dv=np.abs(np.asarray(p["dv_kms"], float)[idx]), l=l, b=b,
                role=np.array([roles.get(int(c), "none") for c in cell]),
                block8=galactic_healpix(l, b, 8))


def score_original(c, eligible, hemi, args):
    """The archived original Legacy model (south only) on the same rows, same features."""
    from qso_pcolor.background import BackgroundColourModel
    from qso_pcolor.legacy import dereddened_relative_fluxes
    from qso_pcolor.outlier import OutlierModel
    from qso_pcolor.priors import BackgroundSurfaceDensity, GridQSOPrior
    from qso_pcolor.qso_model import RedshiftMatch, SlicedColourRedshiftModel
    from qso_pcolor.score import BlendPolicy, score_candidates
    a = "models/archive/original_legacy_south/"
    qso = SlicedColourRedshiftModel.load(a + "qso_south_full.json")
    bkg = BackgroundColourModel.load(a + "background_south_global.json")
    dens = BackgroundSurfaceDensity.load(a + "background_density_south_global.json")
    prior = GridQSOPrior.load(a + "sigma_q_south.json")
    out = OutlierModel.load(a + "outlier_south.json")
    n = len(c["label"])
    res = [None] * n
    use = np.flatnonzero(eligible & (hemi == "south"))
    sub = {k: v[use] for k, v in c["rows"].items()}
    fs, ok = dereddened_relative_fluxes(sub, "south")
    sc = score_candidates(fs, z_primary=c["zp"][use], l_deg=c["l"][use], b_deg=c["b"][use],
                          qso_model=qso, background_model=bkg, match=RedshiftMatch(half_width_kms=args.half_width_kms),
                          qso_prior=prior, background_density=dens, outlier_model=out, min_bands=3,
                          blend_policy=BlendPolicy(min_separation_arcsec=args.min_sep, max_fracflux=args.max_fracflux),
                          separation_arcsec=c["sep"][use], fracflux=sub["fracflux_r"])
    for i, s in zip(use, sc):
        res[i] = s
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/legacy_baseline.json"))
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--half-width-kms", type=float, default=3000.0)
    ap.add_argument("--min-sep", type=float, default=3.0)
    ap.add_argument("--max-sep", type=float, default=30.0)
    ap.add_argument("--max-fracflux", type=float, default=0.2)
    ap.add_argument("--max-non-qso", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--continuation", type=int, default=200)
    ap.add_argument("--tail-draws", type=int, default=3)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())

    from qso_pcolor.baseline import XDQSOBaseline
    from qso_pcolor.legacy import dereddened_relative_fluxes
    from qso_pcolor.qso_model import RedshiftMatch
    from qso_pcolor.score import BlendPolicy, score_candidates

    bl = XDQSOBaseline.load(args.bundle)
    report = dict(bundle=bl.bundle_id, gates=GATES, built=time.strftime("%Y-%m-%d %H:%M"),
                  status=bl.manifest.get("status"))
    t0 = time.time()
    c = companions(cfg, bl, args)
    policy = BlendPolicy(min_separation_arcsec=args.min_sep, max_fracflux=args.max_fracflux)
    match = RedshiftMatch(half_width_kms=args.half_width_kms)
    base, dec = bl.score_rows(c["rows"], z_primary=c["zp"], match=match, separation_arcsec=c["sep"],
                              fracflux=c["rows"]["fracflux_r"], blend_policy=policy,
                              ood_flag_sigma=cfg["science_scoring"]["ood_flag_sigma"])
    el = dec["eligible"]
    report["companion_eligibility"] = dict(zip(*[x.tolist() for x in np.unique(dec["reason"], return_counts=True)]))
    ref = dict(original=score_original(c, el, dec["hemisphere"], args),
               current=score_current(c, el, dec["hemisphere"], SimpleNamespace(
                   current_model=Path("models/multisurvey.json"), current_priors=Path("models/multisurvey_priors.json"),
                   current_outlier=Path("models/multisurvey_outlier.json"), **vars(args))))
    print(f"companions: {len(el):,} rows, {el.sum():,} eligible ({time.time() - t0:.0f} s)", flush=True)

    held = c["role"] == "test"
    fb, okb, logr_b, logbf_b = ranking_metrics(c, base, el)
    checks = {}
    report["ranking"] = {}
    for name, rows_ref in ref.items():
        fr, okr, logr_r, logbf_r = ranking_metrics(c, rows_ref, el)
        both = okb & okr & held
        idx = np.flatnonzero(both)
        mb, mr = fb(idx), fr(idx)
        entry = dict(baseline=mb, reference=mr)
        for k in ("same_vs_field_log_r", "quasar_vs_star_log_bf", "quasar_vs_galaxy_log_bf",
                  "same_vs_field_p_zmatch", "same_vs_hard_log_r"):
            ci = block_bootstrap(c["block8"][idx], lambda i, k=k: fb(idx[i])[k] - fr(idx[i])[k], n_boot=400)
            entry[f"delta_{k}"] = dict(value=mb[k] - mr[k], ci95=ci)
        entry["contamination_at_90pc_retention"] = dict(
            baseline=contamination_at_retention(logbf_b, c["label"], c["spectype"], both),
            reference=contamination_at_retention(logbf_r, c["label"], c["spectype"], both))
        u = np.array([s.ref_mag if s is not None else np.nan for s in base])
        entry["by_mag"] = {f"{a}-{b}": dict(baseline=fb(np.flatnonzero(both & (u >= a) & (u < b))),
                                            reference=fr(np.flatnonzero(both & (u >= a) & (u < b))))
                           for a, b in ((17, 20), (20, 21.5), (21.5, 22.5))}
        report["ranking"][name] = entry
        for k in ("same_vs_field_log_r", "quasar_vs_star_log_bf", "quasar_vs_galaxy_log_bf"):
            checks[f"{name}:{k}"] = bool(entry[f"delta_{k}"]["ci95"][0] > GATES["delta_auc_lower"])
        print(f"[vs {name}] n(same, field, star, gal) {mb['n']}: " + "; ".join(
            f"{k.split('_log')[0]} {mb[k]:.3f} vs {mr[k]:.3f} (d {entry['delta_' + k]['value']:+.3f} "
            f"[{entry['delta_' + k]['ci95'][0]:+.3f},{entry['delta_' + k]['ci95'][1]:+.3f}])"
            for k in ("same_vs_field_log_r", "quasar_vs_star_log_bf", "quasar_vs_galaxy_log_bf")), flush=True)

    # -- numerics: an uninformative W2 must not move ln R
    idx = np.flatnonzero(el & held & (dec["hemisphere"] == "south"))[:500]
    sub = {k: v[idx] for k, v in c["rows"].items()}
    p = bl.parts["south"]
    kw = dict(z_primary=c["zp"][idx], l_deg=c["l"][idx], b_deg=c["b"][idx], qso_model=p["qso"],
              background_model=p["background"], match=match, qso_prior=p["qso_prior"],
              background_density=p["background_density"], outlier_model=p["outlier"], min_bands=3)
    fs, _ = dereddened_relative_fluxes(sub, "south")
    j = fs.labels.index("w2/r")
    a_ = fs.subset(np.arange(fs.n_obs)); a_.observed = a_.observed.copy(); a_.observed[:, j] = False
    a_.cov = np.where(a_.observed[:, :, None] & a_.observed[:, None, :], a_.cov, 0.0)
    b_ = fs.subset(np.arange(fs.n_obs)); b_.cov = b_.cov.copy(); b_.cov[:, j, j] += 1e12
    la = np.array([s.log_r_per_unit_z for s in score_candidates(a_, **kw)])
    lb = np.array([s.log_r_per_unit_z for s in score_candidates(b_, **kw)])
    d = np.abs(la - lb)[np.isfinite(la - lb)]
    report["numerics"] = dict(n=int(d.size), max_abs=float(d.max()))
    checks["numerics"] = bool(d.max() < GATES["numerics_max"])
    print(f"numerics: dropped vs uninformative W2, max |d lnR| {d.max():.2e}", flush=True)

    # -- normalisation, recomputed from the data
    from build_legacy_baseline_sample import cell_areas
    from fit_legacy_baseline import integrate, parent_cells_and_weights, quasar_rows, raw_qso_density, roles_of
    from qso_pcolor.legacy import LegacySelection, load_bricks
    root = Path(cfg["data_dir"])
    sample_cfg = json.loads(Path(cfg["sample_config"]).read_text())
    q, qrows = quasar_rows(cfg, root)
    roles = roles_of(json.loads((root / "design.json").read_text()))
    _, observed = cell_areas(cfg)
    z_edges, _, w_q, pcells, qpix, pix_area = parent_cells_and_weights(cfg, sample_cfg, q, roles, observed,
                                                                        load_bricks(cfg["bricks"]))
    C = bl.manifest["completeness"]["C"]
    lo, hi = bl.manifest["domain"]["ref_mag"]
    report["normalisation"] = {}
    for h, p in bl.parts.items():
        gq = p["qso_prior"]
        d_ = bl.selection.decide(qrows)
        m = d_["accepted"] & (d_["hemisphere"] == h)
        fsq, okq = dereddened_relative_fluxes({k: np.asarray(v)[m] for k, v in qrows.items() if np.ndim(v) == 1}, h)
        raw = raw_qso_density(okq & pcells[h][qpix[m]], q["target_zspec"][m], fsq.ref_mag, w_q[m], z_edges,
                              gq.mag_edges, pcells[h].sum() * pix_area)
        rel = abs(integrate(gq.sigma, z_edges, gq.mag_edges, lo, hi) / (C * integrate(raw, z_edges, gq.mag_edges, lo, hi)) - 1)
        report["normalisation"][h] = dict(relative_error=rel, retention=gq.meta.get("retention_point_over_all"))
        checks[f"normalisation:{h}"] = bool(rel < GATES["normalisation_rel"])
        print(f"[{h}] normalisation rel error {rel:.1e}; PSF/all quasar retention {gq.meta.get('retention_point_over_all'):.3f}", flush=True)

    # -- continuation of the selected field fits under their own non-quasar weights
    import fit_legacy_psf_xdqso as X
    from fit_legacy_baseline import load_field
    from qso_pcolor.xd import fit_xd
    rows_f, cones, cone_area, w_cone, _, design = load_field(cfg, root, bl.selection)
    report["continuation"] = {}
    for h, p in bl.parts.items():
        hemi = np.array([cones[int(cc)]["hemisphere"] for cc in rows_f["cone"]])
        sub = {k: v[hemi == h] for k, v in rows_f.items()}
        fsf, okf = X.features(sub, h)
        role = np.array([roles[cones[int(cc)]["cell"]] for cc in sub["cone"]])
        dom = okf & (fsf.ref_mag >= lo) & (fsf.ref_mag < hi)
        w = np.array([w_cone[int(cc)] for cc in sub["cone"]])
        kap = {int(k): v for k, v in p["background"].meta["contamination"]["kappa_by_cone"].items()}
        krow = np.array([kap[int(cc)] for cc in sub["cone"]])
        sigma_b = p["background_density"](0.5 * (p["background_density"].mag_edges[1:] + p["background_density"].mag_edges[:-1]),
                                         np.zeros(p["background_density"].mag_edges.size - 1),
                                         np.full(p["background_density"].mag_edges.size - 1, 90.0))
        rng = np.random.default_rng(1)
        fit = dom & (role == "fit"); sel_r = dom & (role == "select")
        for mask, cap in ((fit, 60000), (sel_r, 30000)):
            ii = np.flatnonzero(mask)
            if ii.size > cap:
                mask[:] = False; mask[rng.choice(ii, cap, replace=False)] = True
        need = fit | sel_r
        lq = np.full(fsf.n_obs, -np.inf); sq = np.zeros(fsf.n_obs)
        lq[need], sq[need] = X.log_quasar_density(p["qso"], p["qso_prior"], fsf, need)
        mix = p["background"].global_
        r = np.zeros(fsf.n_obs)
        r[need] = X.unrecognised(X.background_log_prob(mix, fsf, need), lq[need], sq[need], krow[need], sigma_b,
                                 fsf.ref_mag[need])
        wn = w * (1 - r)
        imag = np.clip(np.digitize(fsf.ref_mag, X.MAG_EDGES) - 1, 0, len(mix) - 1)
        worst = 0.0
        for mb, mx in enumerate(mix):
            a, b = fit & (imag == mb), sel_r & (imag == mb)
            reg = p["background"].meta["per_bin"][mb]["selected_floor"]
            res = fit_xd(fsf.x[a], fsf.cov[a], observed=fsf.observed[a], weights=wn[a], init=mx,
                         max_iter=args.continuation, tol=0.0, regularization=reg, labels=fsf.labels)
            s0 = np.average(mx.log_prob(fsf.x[b], fsf.cov[b], observed=fsf.observed[b]), weights=wn[b])
            s1 = np.average(res.mixture.log_prob(fsf.x[b], fsf.cov[b], observed=fsf.observed[b]), weights=wn[b])
            worst = max(worst, abs(s1 - s0))
        report["continuation"][h] = dict(iterations=args.continuation, max_abs_delta_select=worst)
        checks[f"continuation:{h}"] = bool(worst < GATES["continuation"])
        print(f"[{h}] continuation {args.continuation} it: max |d held-out log density| {worst:.4f} nats/obj", flush=True)

    report["verdict"] = dict(checks=checks, pass_=all(checks.values()),
                             reported_not_gated=["counts", "north ranking (few held-out companions)", "tails (to add)"])
    out = root / f"validation_xdqso_{bl.bundle_id}.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    print("verdict: " + ("PASS" if report["verdict"]["pass_"] else "FAIL") + "  " +
          ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in checks.items()))
    print(f"wrote {out} ({time.time() - t0:.0f} s)")
    sys.exit(0 if report["verdict"]["pass_"] else 1)


if __name__ == "__main__":
    main()
