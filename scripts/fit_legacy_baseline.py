#!/usr/bin/env python
"""Fit the Legacy-PSF baseline bundle (docs/PLAN_FIELD_MODEL.md, section 7).

Per hemisphere (south, north), from the samples of
``build_legacy_baseline_sample.py`` and the one selection of
``configs/legacy_baseline.json``:

1. field model: the accepted field sources (known quasars removed) of the
   ``fit`` cells with the reference luptitude in ``fit_ref_range``, weighted
   so each cell counts by its footprint area; K and start chosen by the mean
   conditional colour log density p(g, z, W1, W2 | r) on the ``select`` cells;
   only converged fits are eligible;
2. quasar model: redshift slices as in the multi-survey model, fitted to the
   accepted quasars of the fit and calib cells, K and start chosen on the
   select cells;
3. priors: Sigma_Q(z, u_r) from the weighted quasar draw over the parent's
   coverage inside the hemisphere footprint, scaled by one completeness
   constant C computed from the ALL-morphology south quasars (so the PSF
   loss is kept); Sigma_B(u_r) from the field counts over the cones' usable
   areas, each cell weighted by its footprint area. Test cells are excluded
   from both, and from C;
4. unmodelled term: Student-t, nu and scale and eta(u_r) chosen on the
   ``calib`` cells.

Nothing here reads the ``test`` cells. The bundle is written to a new
directory ``<bundle_dir>/<bundle_id>``; ``current`` is switched only by the
validation step.

    python scripts/fit_legacy_baseline.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- data ----------------------------------------------------------------------

def load_field(cfg, root, sel):
    design = json.loads((root / "design.json").read_text())
    areas = json.loads((root / "areas.json").read_text())
    cones = {c["cone"]: c for c in design["cones"]}
    parts, report = [], {}
    for k, c in cones.items():
        r = dict(np.load(root / "cones" / f"cone_{k:03d}.npz", allow_pickle=False))
        d = sel.decide(r)
        keep = d["accepted"] & ~r["known_quasar"].astype(bool) & (d["hemisphere"] == c["hemisphere"])
        kq = d["accepted"] & r["known_quasar"].astype(bool) & (d["hemisphere"] == c["hemisphere"])
        # candidate range in r magnitude (luptitude = magnitude to < 1e-3 mag here: flux >> softening)
        with np.errstate(divide="ignore", invalid="ignore"):
            mag_r = 22.5 - 2.5 * np.log10(np.asarray(r["flux_r"], float))
        lo, hi = cfg["fit"]["candidate_ref_range"]
        report[k] = dict(n_rows=int(len(r["ra"])), n_accepted=int(d["accepted"].sum()),
                         n_known_quasar=int(kq.sum()),
                         n_known_quasar_candidate_range=int((kq & (mag_r >= lo) & (mag_r < hi)).sum()),
                         n_kept=int(keep.sum()))
        parts.append({kk: np.asarray(v)[keep] for kk, v in r.items() if np.ndim(v) == 1})
    rows = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    # cone weights: a cell counts by its footprint area in that hemisphere
    area = {int(k): v["area_deg2"] for k, v in areas.items()}
    cell_cone_area = {}
    for c in cones.values():
        key = (c["cell"], c["hemisphere"])
        cell_cone_area[key] = cell_cone_area.get(key, 0.0) + area[c["cone"]]
    w = {}
    for c in cones.values():
        a_cell = design["cells"][str(c["cell"])]["by_hemisphere"][c["hemisphere"]]
        w[c["cone"]] = a_cell / cell_cone_area[(c["cell"], c["hemisphere"])]
    return rows, cones, area, w, report, design


def roles_of(design):
    return {int(k): v["role"] for k, v in design["cells"].items()}


# -- mixtures --------------------------------------------------------------------

def _fit_one(task):
    """One (floor, K, start) fit and its selection score; runs in a worker process."""
    from qso_pcolor.multisurvey import conditional_log_prob
    from qso_pcolor.xd import _init_mixture, fit_xd
    (key, x, cov, obs, w, xs, cs, os_, ws, k, reg, seed, max_iter, tol, anchor, labels) = task
    init = _init_mixture(x, obs, k, np.random.default_rng(seed))
    r = fit_xd(x, cov, observed=obs, weights=w, init=init, max_iter=max_iter, tol=tol,
               regularization=reg, labels=labels, n_threads=1)
    lp = conditional_log_prob(r.mixture, xs, cs, os_, anchor)
    return key, dict(converged=bool(r.converged), n_iter=r.n_iter, train_ll=r.mean_loglike,
                     select_score=float(np.average(lp, weights=ws)), _mixture=r.mixture)


_POOL = None


def pool(n_workers):
    """One process pool for the run; workers are single-threaded (BLAS pinned to 1)."""
    global _POOL
    if _POOL is None:
        import os
        from concurrent.futures import ProcessPoolExecutor
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
            os.environ[v] = "1"
        _POOL = ProcessPoolExecutor(n_workers)
    return _POOL


def grid_tasks(name, features, fit_rows, sel_rows, weights, grid, fc, seed, anchor, regs, starts):
    """The independent fits of one grid, as picklable tasks keyed by (name, floor, K, start)."""
    x, cov, obs = features.x, features.cov, features.observed
    wf = None if weights is None else weights[fit_rows]
    ws = None if weights is None else weights[sel_rows]
    base = (x[fit_rows], cov[fit_rows], obs[fit_rows], wf,
            x[sel_rows], cov[sel_rows], obs[sel_rows], ws)
    return [((name, reg, k, s),) + base + (k, reg, seed + 1000 * k + s, fc["max_iter"], fc["tol"],
                                           anchor, features.labels)
            for reg in regs for k in grid for s in range(starts)]


def run_grids(grids, fc):
    """Fit every task of several grids in parallel; return per-grid (mixture, record).

    ``grids`` maps name -> (tasks, grid, n_fit, n_select). Selection within a grid:
    the best converged fit by selection score, as before.
    """
    t0 = time.time()
    tasks = [t for g in grids.values() for t in g[0]]
    # largest K first so the long fits start early
    tasks.sort(key=lambda t: -t[9])
    done = {}
    ex = pool(fc["n_workers"])
    for i, (key, res) in enumerate(ex.map(_fit_one, tasks, chunksize=1)):
        done[key] = res
        name, reg, k, s = key
        if k >= 16 or (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(tasks)}] {name}: floor {reg:.0e} K={k:3d} start {s}: "
                  f"{'conv' if res['converged'] else 'NOT conv'} after {res['n_iter']:5d} it, "
                  f"select {res['select_score']:.4f} ({time.time() - t0:.0f} s)", flush=True)
    out = {}
    for name, (tl, grid, n_fit, n_sel) in grids.items():
        trials = [dict(k=t[0][2], start=t[0][3], regularization=t[0][1], **done[t[0]]) for t in tl]
        ok = [t for t in trials if t["converged"]]
        if not ok:
            raise RuntimeError(f"{name}: no fit converged; refusing to publish")
        best = max(ok, key=lambda t: t["select_score"])
        same = [t["select_score"] for t in ok
                if t["k"] == best["k"] and t["regularization"] == best["regularization"]]
        out[name] = (best["_mixture"], dict(
            selected_k=best["k"], selected_start=best["start"],
            selected_regularization=best["regularization"], largest_k_won=best["k"] == max(grid),
            start_spread_at_selected_k=float(max(same) - min(same)), n_fit=n_fit, n_select=n_sel,
            trials=[{k: v for k, v in t.items() if not k.startswith("_")} for t in trials]))
    return out


def fit_grid(name, features, fit_rows, sel_rows, weights, grid, fc, seed, anchor, regs=None,
             starts=None):
    """Every (floor, K, start) fitted to convergence, in parallel; best converged by selection.

    The covariance floor (``regularization``) is a hyperparameter like K and is
    chosen the same way: a floor of 1e-3 mag^2 made every colour direction at
    least 0.045 mag wide, broader than the stellar locus.
    """
    tasks = grid_tasks(name, features, fit_rows, sel_rows, weights, grid, fc, seed, anchor,
                       regs or [fc["regularization"]], starts or fc["starts"])
    return run_grids({name: (tasks, grid, int(fit_rows.sum()), int(sel_rows.sum()))}, fc)[name]


def prepare_field(h, rows, cones, w_cone, sel, fc, cfg, roles):
    """Features, roles, cell-area weights and the fit/select subsamples of one hemisphere."""
    from qso_pcolor.legacy import legacy_photometry
    from qso_pcolor.multisurvey import BandLuptitudeTransform
    cone_role = np.array([roles[cones[int(c)]["cell"]] for c in rows["cone"]])
    hemi = np.array([cones[int(c)]["hemisphere"] for c in rows["cone"]])
    m = hemi == h
    sub = {k: v[m] for k, v in rows.items()}
    role = cone_role[m]
    phot = legacy_photometry(sub, h, maskbits_zero=sel.maskbits_zero)
    train = role == "fit"
    softening = [float(np.median(np.sqrt(phot.variance[train & phot.observed[:, j], j])))
                 for j in range(len(phot.bands))]
    transform = BandLuptitudeTransform(phot.bands, np.array(softening))
    f = transform(phot)
    ir = phot.bands.index(f"decals_dr9_{h}:{sel.reference}")
    lo, hi = fc["fit_ref_range"]
    in_fit = f.observed[:, ir] & (f.x[:, ir] >= lo) & (f.x[:, ir] < hi)
    weights = np.array([w_cone[int(c)] for c in sub["cone"]])
    rng = np.random.default_rng(cfg["seed"] + (0 if h == "south" else 1))
    fit_idx = np.flatnonzero(train & in_fit)
    if fit_idx.size > fc["max_field_fit"]:
        fit_idx = np.sort(rng.choice(fit_idx, fc["max_field_fit"], replace=False))
    sel_idx = np.flatnonzero((role == "select") & in_fit)
    if sel_idx.size > fc["max_field_select"]:
        sel_idx = np.sort(rng.choice(sel_idx, fc["max_field_select"], replace=False))
    fit_rows = np.zeros(len(role), bool); fit_rows[fit_idx] = True
    sel_rows = np.zeros(len(role), bool); sel_rows[sel_idx] = True
    print(f"[{h}] field: {m.sum():,} accepted sources; fit {fit_rows.sum():,}, select "
          f"{sel_rows.sum():,}", flush=True)
    return dict(transform=transform, features=f, role=role, weights=weights, in_fit=in_fit,
                cone=sub["cone"], ir=ir, softening=softening, fit_rows=fit_rows, sel_rows=sel_rows)


# -- unrecognised quasars in the field sample --------------------------------------

def log_quasar_colour_density(qso, gq, f, rows, ir):
    """log p_Q(colours | u_r) = log sum_z Sigma_Q(z, u_r) p(colours | u_r, z) / Sigma_Q(u_r)."""
    from qso_pcolor.multisurvey import _ConditionalQSO
    x, cov, obs = f.x[rows], f.cov[rows], f.observed[rows]
    lp = _ConditionalQSO(qso, ir)._log_p_slices(x, cov, obs)            # (n, J)
    u = x[:, ir]
    sig = np.array([np.interp(u, gq.mag_centres, gq.sigma[j]) for j in range(gq.z_centres.size)]).T
    # slices sit at the prior's z centres (both 0.15 ... 4.35 in 0.1)
    if not np.allclose(qso.z_centres, gq.z_centres):
        raise ValueError("quasar slices and prior redshift grid differ")
    w = sig * np.diff(gq.z_edges)[None]
    tot = w.sum(1)
    from scipy.special import logsumexp
    with np.errstate(divide="ignore", invalid="ignore"):
        lq = logsumexp(lp + np.log(w), axis=1) - np.log(tot)
    return np.where(tot > 0, lq, -np.inf), tot


def cone_recognised_fraction(pf, gq, known_by_cone, area, lo, hi):
    """kappa = known quasars / expected quasars in each cone, candidate range, capped at 1."""
    mc = 0.5 * (gq.mag_edges[1:] + gq.mag_edges[:-1])
    ins = (mc > lo) & (mc < hi)
    expected = float((gq.sigma[:, ins] * np.diff(gq.z_edges)[:, None]
                      * np.diff(gq.mag_edges)[None, ins]).sum())
    raw = {c: known_by_cone[c] / (area[c] * expected) for c in known_by_cone}
    return {c: min(1.0, v) for c, v in raw.items()}, raw


def unrecognised_probability(pf, mix, lq, sq, kappa, sigma_b, m_edges, rows):
    """r_i: probability a field-sample source is an unrecognised quasar.

    odds = (1 - kappa_cone) Sigma_Q(u) p_Q(c|u) / [Sigma_B(u) p_B(c|u)], with
    Sigma_B the one non-quasar density the bundle ships.
    """
    from qso_pcolor.multisurvey import conditional_log_prob
    f, ir = pf["features"], pf["ir"]
    lb = conditional_log_prob(mix, f.x[rows], f.cov[rows], f.observed[rows], ir)
    u = f.x[rows, ir]
    k = np.array([kappa[int(c)] for c in pf["cone"][rows]])
    s_b = sigma_b[np.clip(np.digitize(u, m_edges) - 1, 0, sigma_b.size - 1)]
    s_unk = (1 - k) * sq
    with np.errstate(divide="ignore"):
        lo = np.log(s_unk) + lq - np.log(s_b) - lb
    return 1.0 / (1.0 + np.exp(-np.clip(lo, -700, 700)))


BANDS_ = ("g", "r", "z", "w1", "w2")


def quasar_rows(cfg, root):
    """Matched target rows with the target position as the object's position."""
    q = dict(np.load(root / "quasars.npz", allow_pickle=False))
    rows = {k: v for k, v in q.items() if not k.startswith("target_")}
    rows["ra"], rows["dec"] = q["target_ra"], q["target_dec"]
    return q, rows


def quasar_model(h, q, rows, dq_point, transform, fc, cfg, sample_cfg, roles, ir):
    from qso_pcolor.background import galactic_healpix
    from qso_pcolor.legacy import legacy_photometry
    from qso_pcolor.qso_model import SlicedColourRedshiftModel
    from train_multisurvey_model import redshift_edges
    cell = galactic_healpix(q["target_l"], q["target_b"], cfg["cell_nside"])
    role = np.array([roles.get(int(c), "none") for c in cell])
    m = dq_point["accepted"] & (dq_point["hemisphere"] == h)
    phot = legacy_photometry({k: np.asarray(v)[m] for k, v in rows.items() if np.ndim(v) == 1},
                             h, maskbits_zero=cfg["selection"]["maskbits_zero"])
    f = transform(phot)
    z = q["target_zspec"][m]
    role = role[m]
    lo, hi = fc["fit_ref_range"]
    in_fit = f.observed[:, ir] & (f.x[:, ir] >= lo) & (f.x[:, ir] < hi)
    train = np.isin(role, ("fit", "calib")) & in_fit
    select = (role == "select") & in_fit
    edges = redshift_edges(sample_cfg["z_min"], sample_cfg["z_max"], sample_cfg["z_step"])
    centres = 0.5 * (edges[1:] + edges[:-1])
    mixtures, records, counts = [], [], []
    print(f"[{h}] quasars: {m.sum():,} accepted; fit {train.sum():,}, select {select.sum():,}",
          flush=True)
    grids, names = {}, []
    for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        s = (z >= a - 0.5 * (b - a)) & (z < b + 0.5 * (b - a))
        tr, va = train & s, select & s
        if va.sum() < fc["min_select_per_slice"]:
            # too few selection objects in this slice: choose on the pooled
            # neighbouring slices' selection objects instead (recorded)
            s2 = (z >= a - 2 * (b - a)) & (z < b + 2 * (b - a))
            va = select & s2
        name = f"{h} z={centres[j]:.2f}"
        grids[name] = (grid_tasks(name, f, tr, va, None, fc["qso_k_grid"], fc, cfg["seed"] + 7 * j,
                                  ir, fc["regularization_grid"], fc["starts"]),
                       fc["qso_k_grid"], int(tr.sum()), int(va.sum()))
        names.append(name); counts.append(int(tr.sum()))
    fitted = run_grids(grids, fc)
    for name in names:
        mix, rec = fitted[name]
        rec["n_select_used"] = rec["n_select"]
        mixtures.append(mix); records.append(rec)
    return SlicedColourRedshiftModel(centres, mixtures, np.array(counts), f"legacy_psf_{h}",
                                     transform.bands, meta=dict(per_slice=records)), dict(
        n_accepted=int(m.sum()), n_fit=int(train.sum()), n_select=int(select.sum()))


# -- priors ----------------------------------------------------------------------

def parent_cells_and_weights(cfg, sample_cfg, q, roles, observed_bricks, bricks):
    """Sampling weights of the target draw and the parent-coverage cells, test excluded."""
    import healpy as hp
    from build_multisurvey_priors import parent_bin_counts
    from qso_pcolor.background import galactic_healpix
    from qso_pcolor.data import galactic_from_equatorial
    from qso_pcolor.legacy import assign_bricks, is_north
    z_edges, n_parent, parent_lb = parent_bin_counts(sample_cfg, exclude_heldout=True)
    cell4 = galactic_healpix(q["target_l"], q["target_b"], cfg["cell_nside"])
    train = np.array([roles.get(int(c), "none") != "test" for c in cell4])
    # the draw's own held-out flag must be the test cells
    assert np.array_equal(~train, q["target_held"].astype(bool)), \
        "test cells differ from the target draw's held-out blocks"
    ibin = np.clip(np.digitize(q["target_zspec"], z_edges) - 1, 0, n_parent.size - 1)
    n_drawn = np.bincount(ibin[train], minlength=n_parent.size)
    w = np.where(train, n_parent[ibin] / np.maximum(n_drawn[ibin], 1), 0.0)
    nside = cfg["fit"]["prior_footprint_nside"]
    ppix = galactic_healpix(parent_lb[:, 0], parent_lb[:, 1], nside)
    parent_ok = np.bincount(ppix, minlength=hp.nside2npix(nside)) >= cfg["fit"]["parent_min_per_cell"]
    # footprint of each hemisphere: cell centre on an observed brick of it
    lc, bc = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=True, lonlat=True)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    eq = SkyCoord(lc * u.deg, bc * u.deg, frame="galactic").icrs
    k = assign_bricks(bricks, eq.ra.deg, eq.dec.deg)
    names = np.where(k >= 0, bricks["brickname"][np.clip(k, 0, None)], "")
    north = is_north(eq.ra.deg, eq.dec.deg, bc)
    shift = 2 * int(round(np.log2(nside / cfg["cell_nside"])))
    block_role = np.array([roles.get(int(p >> shift), "none") for p in range(hp.nside2npix(nside))])
    cells = {}
    for h in ("south", "north"):
        on = np.array([n in observed_bricks[h] for n in names]) & (north == (h == "north"))
        cells[h] = parent_ok & on & (block_role != "test") & (np.abs(bc) >= sample_cfg["min_abs_b_deg"])
    qpix = galactic_healpix(q["target_l"], q["target_b"], nside)
    return z_edges, n_parent, w, cells, qpix, hp.nside2pixarea(nside, degrees=True)


def raw_qso_density(mask, z, u, w, z_edges, m_edges, area):
    h2, _, _ = np.histogram2d(z[mask], u[mask], [z_edges, m_edges], weights=w[mask])
    return h2 / (area * np.diff(z_edges)[:, None] * np.diff(m_edges)[None])


def integrate(dens, z_edges, m_edges, lo, hi):
    mc = 0.5 * (m_edges[:-1] + m_edges[1:])
    inside = (mc > lo) & (mc < hi)
    return float(np.sum(dens[:, inside] * np.diff(z_edges)[:, None] * np.diff(m_edges)[None, inside]))


def completeness_constant(cfg, dens_all_south, z_edges, m_edges):
    from qso_pcolor.priors import GridQSOPrior
    norm = cfg["fit"]["normalisation"]
    ref = GridQSOPrior.load(norm["reference_prior"])
    lo, hi = norm["mag_range"]
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    dm = 0.01
    ref_total = sum(float(np.sum(ref(zc, mm) * np.diff(z_edges))) * dm
                    for mm in np.arange(lo + dm / 2, hi, dm))
    raw_total = integrate(dens_all_south, z_edges, m_edges, lo, hi)
    return ref_total / raw_total, ref_total, raw_total


# -- unmodelled term -------------------------------------------------------------

def fit_student_t(h, model, fm, oc, bundle_meta):
    from qso_pcolor.multisurvey import MultiSurveyOutlier
    from qso_pcolor.outlier import (fit_outlier_fraction, mixture_moments, student_t_logpdf,
                                    student_t_noisy_logpdf)
    tpdf = student_t_noisy_logpdf if oc.get("noise", "scale") == "exact" else student_t_logpdf
    f, ir = fm["features"], fm["ir"]
    lo, hi = bundle_meta["domain"]["candidate_ref_range"]
    dom = f.observed[:, ir] & (f.x[:, ir] >= lo) & (f.x[:, ir] < hi)
    cal = (fm["role"] == "calib") & dom
    chk = (fm["role"] == "select") & dom
    use = cal | chk
    log_pb = np.full(f.n_obs, np.nan)
    log_pb[use] = model.background_log_prob(f.x[use], f.cov[use], f.observed[use], ir)
    mean, base = mixture_moments([model.background])
    ref = np.zeros_like(f.observed); ref[:, ir] = f.observed[:, ir]
    wts = fm["weights"]
    rows = []
    for nu in oc["nu"]:
        for c in oc["scale"]:
            cov = c ** 2 * base
            log_pu = np.full(f.n_obs, np.nan)
            log_pu[use] = (tpdf(f.x[use], mean, cov, nu, f.cov[use], observed=f.observed[use])
                           - tpdf(f.x[use], mean, cov, nu, f.cov[use], observed=ref[use]))
            u = f.x[cal, ir]
            inner = np.quantile(u, np.linspace(0, 1, oc["n_mag_bins"] + 1)[1:-1])
            edges = np.concatenate([[lo], inner, [hi]])
            ib = lambda s: np.clip(np.digitize(f.x[s, ir], edges) - 1, 0, oc["n_mag_bins"] - 1)  # noqa: E731
            ic = ib(cal)
            eta = np.array([fit_outlier_fraction(log_pb[cal][ic == j], log_pu[cal][ic == j],
                                                 weights=wts[cal][ic == j])
                            for j in range(oc["n_mag_bins"])])
            gain = {}
            for name, s in (("calib", cal), ("select", chk)):
                e = eta[ib(s)]
                with np.errstate(divide="ignore"):
                    g = np.logaddexp(np.log1p(-e) + log_pb[s], np.log(e) + log_pu[s]) - log_pb[s]
                gain[name] = float(np.average(g, weights=wts[s]))
            rows.append(dict(nu=nu, scale=c, eta=eta.tolist(), edges=edges.tolist(),
                             gain_calib=gain["calib"], gain_select=gain["select"], _cov=cov))
            print(f"  [{h}] t nu={nu:g} scale={c:g}: eta {np.round(eta, 5).tolist()} "
                  f"gain calib {gain['calib']:+.5f} select {gain['select']:+.5f}", flush=True)
    best = max(rows, key=lambda r: r["gain_calib"])
    label = model.transform.bands[ir]
    fr = {label: (np.array(best["edges"]), np.array(best["eta"])),
          "*": (np.array([-99.0, 99.0]), np.array([float(np.median(best["eta"]))]))}
    meta = dict(bundle_meta["artifact_meta"], by="scripts/fit_legacy_baseline.py",
                chosen_by="max calib-cell log-likelihood gain",
                n_calib=int(cal.sum()), n_check_select=int(chk.sum()),
                scan=[{k: v for k, v in r.items() if k != "_cov"} for r in rows])
    return MultiSurveyOutlier(mean, best["_cov"], best["scale"], 0.0, model.transform.bands,
                              model.transform_id, fr, meta=meta, family="student_t", nu=best["nu"],
                              noise=oc.get("noise", "scale"))


# -- main ------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/legacy_baseline.json"))
    ap.add_argument("--hemispheres", nargs="+", default=["south", "north"])
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    fc = cfg["fit"]
    root = Path(cfg["data_dir"])
    if json.loads((root / "config.json").read_text()) != {k: v for k, v in cfg.items() if k != "fit"}:
        raise SystemExit("sample was built with a different configuration")
    sample_cfg = json.loads(Path(cfg["sample_config"]).read_text())

    from qso_pcolor.legacy import LegacySelection, load_bricks
    from qso_pcolor.multisurvey import MultiSurveyModel
    from qso_pcolor.priors import BackgroundSurfaceDensity, GridQSOPrior
    from build_legacy_baseline_sample import cell_areas

    sel = LegacySelection(**cfg["selection"])
    sel_all = LegacySelection(**dict(cfg["selection"], morphology="all"))
    inputs = {n: sha(root / n) for n in ("design.json", "areas.json", "quasars.npz")}
    identity = dict(selection_id=sel.identity, config=cfg, inputs=inputs,
                    code={p: sha(p) for p in ("scripts/fit_legacy_baseline.py",
                                              "src/qso_pcolor/legacy.py", "src/qso_pcolor/xd.py")})
    bundle_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    out_dir = Path(fc["bundle_dir"]) / bundle_id
    tmp = out_dir.with_name(out_dir.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"bundle {bundle_id} -> {out_dir}", flush=True)

    rows, cones, cone_area, w_cone, cone_report, design = load_field(cfg, root, sel)
    roles = roles_of(design)
    q, qrows = quasar_rows(cfg, root)
    dq_point, dq_all = sel.decide(qrows), sel_all.decide(qrows)
    _, observed = cell_areas(cfg)
    bricks = load_bricks(cfg["bricks"])
    z_edges, n_parent, w_q, pcells, qpix, pix_area = parent_cells_and_weights(
        cfg, sample_cfg, q, roles, observed, bricks)
    m_edges = np.arange(fc["fit_ref_range"][0], fc["fit_ref_range"][1] + 1e-9, fc["mag_bin_width"])
    domain = dict(candidate_ref_range=fc["candidate_ref_range"], fit_ref_range=fc["fit_ref_range"],
                  reference=sel.reference, min_abs_b_deg=cfg["min_abs_b_deg"])
    artifact_meta = dict(bundle_id=bundle_id, selection_id=sel.identity, domain=domain)
    bundle_meta = dict(domain=domain, artifact_meta=artifact_meta)
    manifest = dict(kind="legacy_baseline_bundle", version=1, bundle_id=bundle_id,
                    built=time.strftime("%Y-%m-%d"), selection=sel.to_dict(),
                    selection_id=sel.identity, domain=domain, identity=identity,
                    partition={str(k): v for k, v in sorted(roles.items())},
                    cones=cone_report, hemispheres={}, files={})

    # 1. per hemisphere: field features, then the quasar colour model
    from qso_pcolor.legacy import legacy_photometry
    prep, qsos = {}, {}
    for h in args.hemispheres:
        t0 = time.time()
        prep[h] = pf = prepare_field(h, rows, cones, w_cone, sel, fc, cfg, roles)
        qsos[h] = quasar_model(h, q, qrows, dq_point, pf["transform"], fc, cfg, sample_cfg,
                               roles, pf["ir"])
        print(f"[{h}] quasar model done ({time.time() - t0:.0f} s)", flush=True)

    # 2. Sigma_Q: C from the all-morphology south quasars, applied unchanged to PSF
    dens = {}
    for h, pf in prep.items():
        tr, ir = pf["transform"], pf["ir"]
        for name, d in (("point", dq_point), ("all", dq_all)):
            m = d["accepted"] & (d["hemisphere"] == h)
            phot = legacy_photometry({k: np.asarray(v)[m] for k, v in qrows.items() if np.ndim(v) == 1},
                                     h, maskbits_zero=sel.maskbits_zero)
            u = tr(phot).x[:, ir]
            inarea = pcells[h][qpix[m]]
            dens[(h, name)] = raw_qso_density(inarea & np.isfinite(u), q["target_zspec"][m], u,
                                              w_q[m], z_edges, m_edges, pcells[h].sum() * pix_area)
    if ("south", "all") not in dens:
        raise SystemExit("C is defined on the south all-morphology quasars: fit south too")
    C, ref_total, raw_total = completeness_constant(cfg, dens[("south", "all")], z_edges, m_edges)
    print(f"completeness C = {C:.3f} (reference {ref_total:.1f} / raw all-morphology south "
          f"{raw_total:.1f} deg^-2)", flush=True)
    manifest["completeness"] = dict(C=C, reference_total=ref_total, raw_total_all_south=raw_total,
                                    defined_on="south, all morphologies, " + str(fc["normalisation"]))
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    lo_c, hi_c = fc["candidate_ref_range"]
    from qso_pcolor.background import galactic_healpix
    pix = int(galactic_healpix(np.zeros(1), np.full(1, 90.0), 1)[0])

    for h, pf in prep.items():
        t0 = time.time()
        tr, ir, f = pf["transform"], pf["ir"], pf["features"]
        qso, qrec = qsos[h]
        label = tr.bands[ir]
        meta_q = dict(artifact_meta, reference_band=label, transform_id=None,
                      completeness_constant=C,
                      retention_point_over_all=integrate(dens[(h, "point")], z_edges, m_edges, lo_c, hi_c)
                      / integrate(dens[(h, "all")], z_edges, m_edges, lo_c, hi_c),
                      kind="Sigma_Q(z,u_r): weighted draw, PSF, parent coverage x footprint, C from all-morphology south")
        gq = GridQSOPrior(zc, 0.5 * (m_edges[:-1] + m_edges[1:]), dens[(h, "point")] * C, meta_q,
                          z_edges=z_edges, mag_edges=m_edges)

        # 3. counted field density (known quasars removed), non-test cones, cell-area weights
        cones_h = [c for c in cones.values() if c["hemisphere"] == h and roles[c["cell"]] != "test"]
        cells_h = {c["cell"] for c in cones_h}
        total_area = sum(design["cells"][str(c)]["by_hemisphere"][h] for c in cells_h)
        usable = (pf["role"] != "test") & f.observed[:, ir]
        counts, _ = np.histogram(f.x[usable, ir], m_edges, weights=pf["weights"][usable])
        sigma_cnt = counts / total_area / np.diff(m_edges)

        # 4. unrecognised quasars: recognised fraction per cone, then one Sigma_B of
        #    non-quasars (counted minus the expected unrecognised quasars, cones weighted
        #    exactly as in the counts), then r_i against that same Sigma_B
        known = {c["cone"]: cone_report[c["cone"]]["n_known_quasar_candidate_range"]
                 for c in cones.values() if c["hemisphere"] == h}
        kappa, kappa_raw = cone_recognised_fraction(pf, gq, known, cone_area, lo_c, hi_c)
        sq_m = (gq.sigma * np.diff(z_edges)[:, None]).sum(0)                 # Sigma_Q(u) per mag
        a_cell = {c: design["cells"][str(c)]["by_hemisphere"][h] for c in cells_h}
        cone_cell = {c["cone"]: c["cell"] for c in cones_h}
        cell_cone_area = {c: sum(cone_area[k] for k in cone_cell if cone_cell[k] == c) for c in cells_h}
        wc = {k: a_cell[cone_cell[k]] * cone_area[k] / cell_cone_area[cone_cell[k]] for k in cone_cell}
        unk = sum(wc[k] * (1 - kappa[k]) for k in wc) / sum(wc.values())
        sigma_b = np.maximum(sigma_cnt - unk * sq_m, 0.05 * sigma_cnt)
        floor_active = (sigma_cnt - unk * sq_m < 0.05 * sigma_cnt).tolist()
        need = pf["in_fit"] & (pf["role"] != "test")
        lq = np.full(f.n_obs, np.nan); sq = np.full(f.n_obs, np.nan)
        lq[need], sq[need] = log_quasar_colour_density(qso, gq, f, need, ir)
        from qso_pcolor.xd import _init_mixture, fit_xd
        fr = pf["fit_rows"]
        init = _init_mixture(f.x[fr], f.observed[fr], 32, np.random.default_rng(cfg["seed"] + 5))
        mix0 = fit_xd(f.x[fr], f.cov[fr], observed=f.observed[fr], weights=pf["weights"][fr],
                      init=init, max_iter=fc["max_iter"], tol=fc["tol"],
                      regularization=fc["regularization"], n_threads=fc["n_threads"]).mixture
        r = np.zeros(f.n_obs)
        r[need] = unrecognised_probability(pf, mix0, lq[need], sq[need], kappa, sigma_b, m_edges, need)
        history = [dict(step="initial K=32 fit, all sources", mean_r_fit=float(r[fr].mean()))]
        print(f"[{h}] unrecognised quasars: mean r {r[fr].mean():.4f} on fit rows "
              f"(weighted expected {np.sum(pf['weights'][fr] * r[fr]):.0f} of {fr.sum()})", flush=True)
        w_nonq = pf["weights"] * (1 - r)
        _, rec_reg = fit_grid(f"{h} field floor", f, fr, pf["sel_rows"], w_nonq,
                              [fc["regularization_choice_k"]], fc, cfg["seed"] + 11, ir,
                              regs=fc["regularization_grid"], starts=1)
        reg = rec_reg["selected_regularization"]
        print(f"[{h}] field covariance floor chosen: {reg:.0e}", flush=True)
        mix, rec = fit_grid(f"{h} field", f, fr, pf["sel_rows"], w_nonq, fc["field_k_grid"], fc,
                            cfg["seed"], ir, regs=[reg])
        rec["regularization_choice"] = rec_reg
        for it in range(fc.get("contamination_iterations", 2)):
            r_new = np.zeros(f.n_obs)
            r_new[need] = unrecognised_probability(pf, mix, lq[need], sq[need], kappa, sigma_b,
                                                   m_edges, need)
            change = float(np.abs(r_new[fr] - r[fr]).mean())
            history.append(dict(step=f"update {it + 1}", mean_r_fit=float(r_new[fr].mean()),
                                mean_abs_change=change, max_abs_change=float(np.abs(r_new - r).max())))
            print(f"[{h}] r update {it + 1}: mean {r_new[fr].mean():.4f}, mean |change| {change:.2e}",
                  flush=True)
            r = r_new
            w_nonq = pf["weights"] * (1 - r)
            if change < fc.get("contamination_tol", 1e-3):
                break
            res = fit_xd(f.x[fr], f.cov[fr], observed=f.observed[fr], weights=w_nonq[fr], init=mix,
                         max_iter=fc["max_iter"], tol=fc["tol"], regularization=reg,
                         labels=f.labels, n_threads=fc["n_threads"])
            if not res.converged:
                raise RuntimeError(f"{h}: refit after the r update did not converge")
            mix = res.mixture
        rec["unrecognised_quasars"] = dict(history=history, kappa_by_cone={str(k): v for k, v in kappa.items()},
                                           kappa_uncapped_by_cone={str(k): v for k, v in kappa_raw.items()},
                                           sigma_b=sigma_b.tolist(), mean_unrecognised_fraction=unk)
        pf["weights_nonq"] = w_nonq
        pf["r"] = r
        bounds = np.array([[np.min(f.x[fr & f.observed[:, j], j]), np.max(f.x[fr & f.observed[:, j], j])]
                           for j in range(len(tr.bands))])
        model = MultiSurveyModel(qso, mix, tr, (label,) + tuple(b for b in tr.bands if b != label),
                                 bounds, meta=dict(artifact_meta, hemisphere=h, field_fit=rec,
                                                   quasar_fit=qrec, softening=pf["softening"],
                                                   by="scripts/fit_legacy_baseline.py"))
        gq.meta["transform_id"] = model.transform_id

        # 5. Sigma_B (computed in step 4); report the model-based expectation of
        #    unrecognised quasars too (not an independent check: both use Sigma_Q, kappa)
        model_unk = np.histogram(f.x[usable, ir], m_edges, weights=(pf["weights"] * r)[usable])[0] \
            / total_area / np.diff(m_edges)
        mc = 0.5 * (m_edges[1:] + m_edges[:-1]); ins = (mc > lo_c) & (mc < hi_c)
        check = dict(prior_based_per_deg2=float(np.sum((unk * sq_m * np.diff(m_edges))[ins])),
                     model_based_per_deg2=float(np.sum((model_unk * np.diff(m_edges))[ins])),
                     note="not independent: both use Sigma_Q and kappa; r_i only where computed",
                     floor_active_by_bin=floor_active)
        bd = BackgroundSurfaceDensity(1, 1, m_edges,
                                      {(pix, i): float(v * total_area * dm) for i, (v, dm)
                                       in enumerate(zip(sigma_b, np.diff(m_edges)))},
                                      {pix: float(total_area)},
                                      meta=dict(artifact_meta, reference_band=label,
                                                transform_id=model.transform_id, n_cones=len(cones_h),
                                                n_cells=len(cells_h), mean_unrecognised_fraction=unk,
                                                counted_per_deg2_per_mag=sigma_cnt.tolist(),
                                                unrecognised_check=check,
                                                kind="Sigma_B(u_r): PSF non-quasars = counted (known quasars removed) - (1 - kappa) Sigma_Q"))
        priors = dict(kind="multisurvey_priors", transform_id=model.transform_id,
                      completeness_constant=C, **artifact_meta,
                      anchors={label: {"qso_prior": gq.to_dict(), "background_density": bd.to_dict()}})
        pf["mixture_for_outlier"] = mix
        out = fit_student_t(h, model, dict(pf, weights=pf["weights_nonq"]), fc["outlier"], bundle_meta)
        files = {"model": f"{h}_model.json", "priors": f"{h}_priors.json", "outlier": f"{h}_outlier.json"}
        model.save(tmp / files["model"])
        (tmp / files["priors"]).write_text(json.dumps(priors))
        out.save(tmp / files["outlier"])
        manifest["hemispheres"][h] = files
        print(f"[{h}] PSF/all quasar retention {meta_q['retention_point_over_all']:.3f}; Sigma_B over "
              f"{total_area:.0f} deg^2 in {len(cells_h)} cells; unrecognised quasars "
              f"{check['prior_based_per_deg2']:.0f} (prior) vs {check['model_based_per_deg2']:.0f} "
              f"(model) deg^-2 in {lo_c}-{hi_c} ({time.time() - t0:.0f} s)", flush=True)
    manifest["status"] = {h: ("validated_cells" if any(roles[c["cell"]] == "test" for c in cones.values()
                                                       if c["hemisphere"] == h) else "provisional: no held-out field cells")
                          for h in manifest["hemispheres"]}
    for h, files in manifest["hemispheres"].items():
        for name in files.values():
            manifest["files"][name] = sha(tmp / name)
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; bundles are immutable")
    tmp.rename(out_dir)
    print(f"wrote bundle {out_dir}")


if __name__ == "__main__":
    main()
