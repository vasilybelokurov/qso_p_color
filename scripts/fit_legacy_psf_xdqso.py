#!/usr/bin/env python
"""Legacy-PSF baseline in the original XDQSO-style design: field, priors, outlier, bundle.

The quasar colour model comes from ``train_qso_model.py --morphology point``
(relative fluxes g/r, z/r, W1/r, W2/r per redshift slice, independent of
magnitude, dereddened). This script builds, per hemisphere, everything else
from the 304-cone PSF field sample and the quasar draw of
``build_legacy_baseline_sample.py``, exactly as the original model did:

* photometry dereddened: flux / mw_transmission, variance / mw_transmission^2;
  relative fluxes to r; >= 3 usable dimensions; 17 <= r < 22.5 (dereddened);
* field: one mixture per r bin (17, 19.5, 20.5, 21.5, 22.5), K and covariance
  floor per bin chosen on the ``select`` cells, fits in parallel;
  sources weighted by cell area x (1 - P(unrecognised quasar)), with
  P from the quasar model, Sigma_Q and the cone's recognised fraction kappa;
* Sigma_Q(z, r): weighted quasar draw over the parent coverage in the
  footprint, one completeness constant C from the all-morphology south
  quasars applied unchanged to the PSF prior;
* Sigma_B(r): counted PSF field (known quasars removed) minus the expected
  unrecognised quasars;
* unmodelled term: the original broad Gaussian (``OutlierModel``), kappa and
  eta per r bin on the ``calib`` cells.

The ``test`` cells are read by nothing here.

    python scripts/fit_legacy_psf_xdqso.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

BANDS = ("g", "r", "z", "w1", "w2")
MAG_EDGES = np.array([17.0, 19.5, 20.5, 21.5, 22.5])       # field mixtures (original)
DENS_EDGES = np.arange(17.0, 22.5 + 1e-9, 0.5)              # surface densities


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(rows: dict, h: str):
    from qso_pcolor.legacy import dereddened_relative_fluxes
    return dereddened_relative_fluxes(rows, h)


def _fit_one(task):
    """One (bin, K, floor, start) fit and its held-out score; runs in a worker."""
    from qso_pcolor.xd import _init_mixture, fit_xd
    key, x, cov, obs, w, xs, cs, os_, ws, k, reg, seed, max_iter, tol, labels = task
    init = _init_mixture(x, obs, k, np.random.default_rng(seed))
    r = fit_xd(x, cov, observed=obs, weights=w, init=init, max_iter=max_iter, tol=tol,
               regularization=reg, labels=labels)
    lp = r.mixture.log_prob(xs, cs, observed=os_)
    return key, dict(converged=bool(r.converged), n_iter=r.n_iter,
                     select_score=float(np.average(lp, weights=ws)), mixture=r.mixture)


def fit_bins(fs, fit, sel, w, fc, seed, pool, ks=None, regs=None, starts=None):
    """Per r bin, every (K, floor, start) in parallel; the best converged by selection."""
    ks, regs, starts = ks or fc["field_k_grid"], regs or fc["floor_grid"], starts or fc["starts"]
    imag = np.digitize(fs.ref_mag, MAG_EDGES) - 1
    tasks = []
    for mb in range(MAG_EDGES.size - 1):
        a, b = fit & (imag == mb), sel & (imag == mb)
        for k in ks:
            for reg in regs:
                for s in range(starts):
                    tasks.append(((mb, k, reg, s), fs.x[a], fs.cov[a], fs.observed[a], w[a],
                                  fs.x[b], fs.cov[b], fs.observed[b], w[b], k, reg,
                                  seed + 1000 * k + 10 * mb + s, fc["max_iter"], fc["tol"], fs.labels))
    tasks.sort(key=lambda t: -t[9] * len(t[1]))
    res = dict(pool.map(_fit_one, tasks, chunksize=1))
    mixtures, records = [], []
    for mb in range(MAG_EDGES.size - 1):
        trials = [dict(k=k, floor=reg, start=s, **{kk: v for kk, v in r.items() if kk != "mixture"})
                  for (m_, k, reg, s), r in res.items() if m_ == mb]
        in_bin = [(key, r) for key, r in res.items() if key[0] == mb]
        conv = [kv for kv in in_bin if kv[1]["converged"]]
        # prefer converged fits; otherwise the best at the iteration cap, recorded as such
        # (the ship criterion is stability under continuation, checked by the validator)
        key, best = max(conv or in_bin, key=lambda kv: kv[1]["select_score"])
        mixtures.append(best["mixture"])
        records.append(dict(mag=[float(MAG_EDGES[mb]), float(MAG_EDGES[mb + 1])], selected_k=key[1],
                            selected_converged=bool(best["converged"]), n_converged=len(conv),
                            selected_floor=key[2], largest_k_won=key[1] == max(ks),
                            n_fit=int((fit & (imag == mb)).sum()), n_select=int((sel & (imag == mb)).sum()),
                            trials=sorted(trials, key=lambda t: (t["k"], t["floor"], t["start"]))))
    return mixtures, records


def background_log_prob(mixtures, fs, rows):
    imag = np.clip(np.digitize(fs.ref_mag[rows], MAG_EDGES) - 1, 0, len(mixtures) - 1)
    out = np.empty(rows.sum())
    x, c, o = fs.x[rows], fs.cov[rows], fs.observed[rows]
    for mb, mix in enumerate(mixtures):
        s = imag == mb
        if s.any():
            out[s] = mix.log_prob(x[s], c[s], observed=o[s])
    return out


def log_quasar_density(qso, gq, fs, rows):
    """log p_Q(c | r) = log sum_z Sigma_Q(z, r) p(c | z) dz / Sigma_Q(r), and Sigma_Q(r)."""
    from scipy.special import logsumexp
    x, c, o, u = fs.x[rows], fs.cov[rows], fs.observed[rows], fs.ref_mag[rows]
    lp = qso.log_p_colour_given_z(x, c, qso.z_centres, observed=o)                  # (n, J)
    if not np.allclose(qso.z_centres, gq.z_centres):
        raise ValueError("quasar slices and prior redshift grid differ")
    sig = np.array([np.interp(u, gq.mag_centres, gq.sigma[j])
                    for j in range(gq.z_centres.size)]).T                          # (n, J)
    w = sig * np.diff(gq.z_edges)[None]
    tot = w.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lq = logsumexp(lp + np.log(w), axis=1) - np.log(tot)
    return np.where(tot > 0, lq, -np.inf), tot


def unrecognised(lb, lq, sq, kappa_row, sigma_b, u):
    s_b = sigma_b[np.clip(np.digitize(u, DENS_EDGES) - 1, 0, sigma_b.size - 1)]
    with np.errstate(divide="ignore"):
        lo = np.log((1 - kappa_row) * sq) + lq - np.log(s_b) - lb
    return 1 / (1 + np.exp(-np.clip(lo, -700, 700)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/legacy_baseline.json"))
    ap.add_argument("--qso", nargs=2, action="append", metavar=("HEMISPHERE", "PATH"),
                    default=None, help="quasar model per hemisphere")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    fc = cfg["xdqso"]
    root = Path(cfg["data_dir"])
    sample_cfg = json.loads(Path(cfg["sample_config"]).read_text())
    qso_paths = dict(args.qso or [("south", "models/legacy_psf_xdqso/qso_south_psf.json"),
                                  ("north", "models/legacy_psf_xdqso/qso_north_psf.json")])

    import healpy as hp  # noqa: F401
    from concurrent.futures import ProcessPoolExecutor
    from build_legacy_baseline_sample import cell_areas
    from fit_legacy_baseline import (completeness_constant, integrate, load_field,
                                     parent_cells_and_weights, quasar_rows, raw_qso_density,
                                     roles_of)
    from qso_pcolor.background import BackgroundColourModel, galactic_healpix
    from qso_pcolor.data import galactic_from_equatorial
    from qso_pcolor.legacy import LegacySelection, load_bricks
    from qso_pcolor.outlier import OutlierModel, fit_outlier_fraction
    from qso_pcolor.priors import BackgroundSurfaceDensity, GridQSOPrior
    from qso_pcolor.qso_model import SlicedColourRedshiftModel

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    pool = ProcessPoolExecutor(fc["n_workers"])

    sel = LegacySelection(**cfg["selection"])
    sel_all = LegacySelection(**dict(cfg["selection"], morphology="all"))
    inputs = {n: sha(root / n) for n in ("design.json", "areas.json", "quasars.npz")}
    inputs.update({f"qso_{h}": sha(p) for h, p in qso_paths.items()})
    identity = dict(selection_id=sel.identity, xdqso=fc, inputs=inputs,
                    code={p: sha(p) for p in ("scripts/fit_legacy_psf_xdqso.py",
                                              "src/qso_pcolor/legacy.py", "src/qso_pcolor/xd.py")})
    bundle_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    out_dir = Path(fc["bundle_dir"]) / bundle_id
    tmp = out_dir.with_name(out_dir.name + ".partial"); tmp.mkdir(parents=True, exist_ok=True)
    print(f"bundle {bundle_id} -> {out_dir}", flush=True)

    # -- samples --------------------------------------------------------------------
    rows, cones, cone_area, w_cone, _, design = load_field(cfg, root, sel)
    roles = roles_of(design)
    q, qrows = quasar_rows(cfg, root)
    _, observed = cell_areas(cfg)
    z_edges, _, w_q, pcells, qpix, pix_area = parent_cells_and_weights(
        cfg, sample_cfg, q, roles, observed, load_bricks(cfg["bricks"]))
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    lo, hi = MAG_EDGES[0], MAG_EDGES[-1]

    # -- Sigma_Q per hemisphere; C from the all-morphology south quasars ------------
    dens = {}
    for h in qso_paths:
        for name, s_ in (("point", sel), ("all", sel_all)):
            d = s_.decide(qrows)
            m = d["accepted"] & (d["hemisphere"] == h)
            fs, ok = features({k: np.asarray(v)[m] for k, v in qrows.items() if np.ndim(v) == 1}, h)
            use = ok & pcells[h][qpix[m]]
            dens[(h, name)] = raw_qso_density(use, q["target_zspec"][m], fs.ref_mag, w_q[m],
                                              z_edges, DENS_EDGES, pcells[h].sum() * pix_area)
    C, ref_total, raw_total = completeness_constant(cfg, dens[("south", "all")], z_edges, DENS_EDGES)
    print(f"completeness C = {C:.3f} (reference {ref_total:.1f} / all-morphology south "
          f"{raw_total:.1f} deg^-2)", flush=True)

    manifest = dict(kind="legacy_xdqso_bundle", version=1, bundle_id=bundle_id,
                    built=time.strftime("%Y-%m-%d"), selection=sel.to_dict(), selection_id=sel.identity,
                    domain=dict(ref_mag=[float(lo), float(hi)], min_abs_b_deg=cfg["min_abs_b_deg"],
                                min_ref_snr=5.0, min_dims=3, photometry="dereddened relative fluxes to r"),
                    identity=identity, completeness=dict(C=C, reference_total=ref_total,
                                                         raw_total_all_south=raw_total),
                    partition={str(k): v for k, v in sorted(roles.items())}, hemispheres={}, files={})
    art = dict(bundle_id=bundle_id, selection_id=sel.identity)

    for h, qpath in qso_paths.items():
        t0 = time.time()
        qso = SlicedColourRedshiftModel.load(qpath)
        gq = GridQSOPrior(zc, 0.5 * (DENS_EDGES[1:] + DENS_EDGES[:-1]), dens[(h, "point")] * C,
                          dict(art, completeness_constant=C, reference="dereddened r",
                               retention_point_over_all=integrate(dens[(h, "point")], z_edges, DENS_EDGES, lo, hi)
                               / integrate(dens[(h, "all")], z_edges, DENS_EDGES, lo, hi)),
                          z_edges=z_edges, mag_edges=DENS_EDGES)
        # field rows of this hemisphere
        hemi = np.array([cones[int(c)]["hemisphere"] for c in rows["cone"]])
        sub = {k: v[hemi == h] for k, v in rows.items()}
        fs, ok = features(sub, h)
        cone = sub["cone"]
        role = np.array([roles[cones[int(c)]["cell"]] for c in cone])
        dom = ok & (fs.ref_mag >= lo) & (fs.ref_mag < hi)
        w = np.array([w_cone[int(c)] for c in cone])
        rng = np.random.default_rng(cfg["seed"])
        fit = dom & (role == "fit"); sel_r = dom & (role == "select"); cal = dom & (role == "calib")
        for mask, cap in ((fit, fc["max_fit"]), (sel_r, fc["max_select"])):
            idx = np.flatnonzero(mask)
            if idx.size > cap:
                mask[:] = False; mask[rng.choice(idx, cap, replace=False)] = True
        # counted density, cones weighted by cell area (non-test)
        cones_h = [c for c in cones.values() if c["hemisphere"] == h and roles[c["cell"]] != "test"]
        cells_h = sorted({c["cell"] for c in cones_h})
        a_cell = {c: design["cells"][str(c)]["by_hemisphere"][h] for c in cells_h}
        total_area = sum(a_cell.values())
        nt = dom & (role != "test")
        sigma_cnt = np.histogram(fs.ref_mag[nt], DENS_EDGES, weights=w[nt])[0] / total_area / np.diff(DENS_EDGES)
        # recognised fraction per cone: known quasars in range / expected
        sq_m = (gq.sigma * np.diff(z_edges)[:, None]).sum(0)
        expected = float(np.sum(sq_m * np.diff(DENS_EDGES)))
        kappa = {}
        for c in cones.values():
            if c["hemisphere"] != h:
                continue
            r_ = dict(np.load(root / "cones" / f"cone_{c['cone']:03d}.npz", allow_pickle=False))
            d = sel.decide(r_)
            kq = d["accepted"] & r_["known_quasar"].astype(bool) & (d["hemisphere"] == h)
            fk, okk = features({k: np.asarray(v)[kq] for k, v in r_.items() if np.ndim(v) == 1}, h)
            n_known = int((okk & (fk.ref_mag >= lo) & (fk.ref_mag < hi)).sum())
            kappa[c["cone"]] = min(1.0, n_known / (cone_area[c["cone"]] * expected))
        cone_cell = {c["cone"]: c["cell"] for c in cones_h}
        cca = {cc: sum(cone_area[k] for k in cone_cell if cone_cell[k] == cc) for cc in cells_h}
        wc = {k: a_cell[cone_cell[k]] * cone_area[k] / cca[cone_cell[k]] for k in cone_cell}
        unk = sum(wc[k] * (1 - kappa[k]) for k in wc) / sum(wc.values())
        sigma_b = np.maximum(sigma_cnt - unk * sq_m, 0.05 * sigma_cnt)
        floor_active = (sigma_cnt - unk * sq_m < 0.05 * sigma_cnt).tolist()
        # quasar colour density for the rows that need r_i
        need = fit | sel_r | cal
        lq = np.full(fs.n_obs, -np.inf); sq = np.zeros(fs.n_obs)
        lq[need], sq[need] = log_quasar_density(qso, gq, fs, need)
        krow = np.array([kappa[int(c)] for c in cone])
        print(f"[{h}] field: fit {fit.sum():,}, select {sel_r.sum():,}, calib {cal.sum():,}; "
              f"mean unrecognised fraction {unk:.2f}", flush=True)
        # initial fit without the correction, then alternate r_i and the fit
        mix, _ = fit_bins(fs, fit, sel_r, w, fc, cfg["seed"], pool, ks=[8], regs=[1e-6], starts=1)
        r = np.zeros(fs.n_obs)
        r[need] = unrecognised(background_log_prob(mix, fs, need), lq[need], sq[need], krow[need],
                               sigma_b, fs.ref_mag[need])
        history = []
        for it in range(fc["contamination_iterations"]):
            mix, recs = fit_bins(fs, fit, sel_r, w * (1 - r), fc, cfg["seed"], pool)
            r_new = np.zeros(fs.n_obs)
            r_new[need] = unrecognised(background_log_prob(mix, fs, need), lq[need], sq[need],
                                       krow[need], sigma_b, fs.ref_mag[need])
            change = float(np.abs(r_new[fit] - r[fit]).mean())
            history.append(dict(iteration=it + 1, mean_r_fit=float(r_new[fit].mean()), mean_abs_change=change))
            print(f"[{h}] iteration {it + 1}: K per bin {[x['selected_k'] for x in recs]}, floors "
                  f"{[x['selected_floor'] for x in recs]}; mean r {r_new[fit].mean():.4f}, "
                  f"mean |change| {change:.1e}", flush=True)
            r = r_new
            if change < fc["contamination_tol"]:
                break
        wnq = w * (1 - r)
        system = qso.system
        bkg = BackgroundColourModel(1, 1, MAG_EDGES, {}, {}, list(mix), {}, {}, 500.0, "background",
                                    system, fs.labels, meta=dict(art, per_bin=recs, contamination=dict(
                                        history=history, mean_unrecognised_fraction=unk,
                                        kappa_by_cone={str(k): v for k, v in kappa.items()},
                                        floor_active_by_bin=floor_active)))
        # unmodelled term: original broad Gaussian, kappa and eta(r) on the calib cells
        lb_cal = background_log_prob(mix, fs, cal)
        ib = np.clip(np.digitize(fs.ref_mag[cal], MAG_EDGES) - 1, 0, MAG_EDGES.size - 2)
        best = None
        for fac in fc["kappa_factors"]:
            from qso_pcolor.outlier import envelope_covariance, min_dominant_kappa, mixture_moments
            mean, base = mixture_moments(list(mix))
            check = list(qso.mixtures) + list(mix)
            env = envelope_covariance(base, check)
            kmin = min_dominant_kappa(env, check)
            kap = fac * kmin
            om = OutlierModel(mean, kap ** 2 * env, np.zeros(MAG_EDGES.size - 1), MAG_EDGES, kap, kmin,
                              system, fs.labels)
            lu = om.log_prob(fs.x[cal], fs.cov[cal], observed=fs.observed[cal])
            eta = np.array([fit_outlier_fraction(lb_cal[ib == j], lu[ib == j], weights=wnq[cal][ib == j])
                            for j in range(MAG_EDGES.size - 1)])
            e = eta[ib]
            with np.errstate(divide="ignore"):
                gain = float(np.average(np.logaddexp(np.log1p(-e) + lb_cal, np.log(e) + lu) - lb_cal,
                                        weights=wnq[cal]))
            print(f"[{h}] outlier kappa = {fac:g} x kappa_min ({kmin:.2f}): eta {np.round(eta, 4).tolist()}, "
                  f"calib gain {gain:+.5f}", flush=True)
            if best is None or gain > best[0]:
                best = (gain, OutlierModel(mean, kap ** 2 * env, eta, MAG_EDGES, kap, kmin, system,
                                           fs.labels, meta=dict(art, kappa_factor=fac, calib_gain=gain)))
        outlier = best[1]
        pix = int(galactic_healpix(np.zeros(1), np.full(1, 90.0), 1)[0])
        bd = BackgroundSurfaceDensity(1, 1, DENS_EDGES,
                                      {(pix, i): float(v * total_area * dm) for i, (v, dm)
                                       in enumerate(zip(sigma_b, np.diff(DENS_EDGES)))},
                                      {pix: float(total_area)},
                                      meta=dict(art, counted=sigma_cnt.tolist(), mean_unrecognised_fraction=unk,
                                                kind="PSF non-quasars: counted (known quasars removed) - (1 - kappa) Sigma_Q"))
        files = dict(qso=f"{h}_qso.json", background=f"{h}_background.json",
                     qso_prior=f"{h}_qso_prior.json", background_density=f"{h}_background_density.json",
                     outlier=f"{h}_outlier.json")
        qso.meta.update(art); qso.save(tmp / files["qso"])
        bkg.save(tmp / files["background"]); gq.save(tmp / files["qso_prior"])
        bd.save(tmp / files["background_density"]); outlier.save(tmp / files["outlier"])
        manifest["hemispheres"][h] = files
        manifest.setdefault("status", {})[h] = ("validated_cells" if any(
            roles[c["cell"]] == "test" for c in cones.values() if c["hemisphere"] == h)
            else "provisional: no held-out field cells")
        print(f"[{h}] done ({time.time() - t0:.0f} s)", flush=True)
    for files in manifest["hemispheres"].values():
        for name in files.values():
            manifest["files"][name] = sha(tmp / name)
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; bundles are immutable")
    tmp.rename(out_dir)
    pool.shutdown()
    print(f"wrote bundle {out_dir}")


if __name__ == "__main__":
    main()
