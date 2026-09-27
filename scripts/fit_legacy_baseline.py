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
        report[k] = dict(n_rows=int(len(r["ra"])), n_accepted=int(d["accepted"].sum()),
                         n_known_quasar=int((d["accepted"] & r["known_quasar"].astype(bool)).sum()),
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

def fit_grid(name, features, fit_rows, sel_rows, weights, grid, fc, seed, anchor):
    """Every (K, start) fitted to convergence; the best converged by selection score."""
    from qso_pcolor.multisurvey import conditional_log_prob
    from qso_pcolor.xd import _init_mixture, fit_xd
    x, cov, obs = features.x, features.cov, features.observed
    trials = []
    t0 = time.time()
    for k in grid:
        for s in range(fc["starts"]):
            rng = np.random.default_rng(seed + 1000 * k + s)
            init = _init_mixture(x[fit_rows], obs[fit_rows], k, rng)
            r = fit_xd(x[fit_rows], cov[fit_rows], observed=obs[fit_rows],
                       weights=None if weights is None else weights[fit_rows], init=init,
                       max_iter=fc["max_iter"], tol=fc["tol"], regularization=fc["regularization"],
                       labels=features.labels, n_threads=fc["n_threads"])
            lp = conditional_log_prob(r.mixture, x[sel_rows], cov[sel_rows], obs[sel_rows], anchor)
            ws = None if weights is None else weights[sel_rows]
            score = float(np.average(lp, weights=ws))
            trials.append(dict(k=k, start=s, converged=bool(r.converged), n_iter=r.n_iter,
                               train_ll=r.mean_loglike, select_score=score,
                               _mixture=r.mixture, _history=r.history[-3:]))
            print(f"  {name}: K={k:3d} start {s}: {'conv' if r.converged else 'NOT conv'} "
                  f"after {r.n_iter:5d} it, select {score:.4f} ({time.time() - t0:.0f} s)", flush=True)
    ok = [t for t in trials if t["converged"]]
    if not ok:
        raise RuntimeError(f"{name}: no fit converged; refusing to publish")
    best = max(ok, key=lambda t: t["select_score"])
    same_k = [t["select_score"] for t in ok if t["k"] == best["k"]]
    record = dict(selected_k=best["k"], selected_start=best["start"],
                  largest_k_won=best["k"] == max(grid),
                  start_spread_at_selected_k=float(max(same_k) - min(same_k)),
                  n_fit=int(fit_rows.sum()), n_select=int(sel_rows.sum()),
                  trials=[{k: v for k, v in t.items() if not k.startswith("_")} for t in trials])
    return best["_mixture"], record


def field_model(h, rows, cones, w_cone, sel, fc, cfg, roles):
    from qso_pcolor.legacy import legacy_photometry
    from qso_pcolor.multisurvey import BandLuptitudeTransform
    cone_role = np.array([roles[cones[int(c)]["cell"]] for c in rows["cone"]])
    hemi = np.array([cones[int(c)]["hemisphere"] for c in rows["cone"]])
    m = hemi == h
    sub = {k: v[m] for k, v in rows.items()}
    role = cone_role[m]
    phot = legacy_photometry(sub, h, maskbits_zero=sel.maskbits_zero)
    train = role == "fit"
    softening = []
    for j, label in enumerate(phot.bands):
        e = np.sqrt(phot.variance[train & phot.observed[:, j], j])
        softening.append(float(np.median(e)))
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
    mix, rec = fit_grid(f"{h} field", f, fit_rows, sel_rows, weights, fc["field_k_grid"], fc,
                        cfg["seed"], ir)
    bounds = np.array([[np.min(f.x[fit_rows & f.observed[:, j], j]),
                        np.max(f.x[fit_rows & f.observed[:, j], j])] for j in range(len(BANDS_))])
    return dict(transform=transform, mixture=mix, record=rec, bounds=bounds, features=f,
                role=role, weights=weights, in_fit=in_fit, cone=sub["cone"], ir=ir,
                softening=softening)


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
    for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        s = (z >= a - 0.5 * (b - a)) & (z < b + 0.5 * (b - a))
        tr, va = train & s, select & s
        if va.sum() < fc["min_select_per_slice"]:
            # too few selection objects in this slice: choose on the pooled
            # neighbouring slices' selection objects instead (recorded)
            s2 = (z >= a - 2 * (b - a)) & (z < b + 2 * (b - a))
            va = select & s2
        mix, rec = fit_grid(f"{h} z={centres[j]:.2f}", f, tr, va, None, fc["qso_k_grid"], fc,
                            cfg["seed"] + 7 * j, ir)
        rec["n_select_used"] = int(va.sum())
        mixtures.append(mix); records.append(rec); counts.append(int(tr.sum()))
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
    from qso_pcolor.outlier import fit_outlier_fraction, mixture_moments, student_t_logpdf
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
            log_pu[use] = (student_t_logpdf(f.x[use], mean, cov, nu, f.cov[use], observed=f.observed[use])
                           - student_t_logpdf(f.x[use], mean, cov, nu, f.cov[use], observed=ref[use]))
            u = f.x[cal, ir]
            inner = np.quantile(u, np.linspace(0, 1, oc["n_mag_bins"] + 1)[1:-1])
            edges = np.concatenate([[lo], inner, [hi]])
            ib = lambda s: np.clip(np.digitize(f.x[s, ir], edges) - 1, 0, oc["n_mag_bins"] - 1)  # noqa: E731
            ic = ib(cal)
            eta = np.array([fit_outlier_fraction(log_pb[cal][ic == j], log_pu[cal][ic == j])
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
                              model.transform_id, fr, meta=meta, family="student_t", nu=best["nu"])


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

    # C from the all-morphology south quasars, then applied to every PSF prior
    from qso_pcolor.legacy import legacy_photometry
    results = {}
    for h in args.hemispheres:
        t0 = time.time()
        fm = field_model(h, rows, cones, w_cone, sel, fc, cfg, roles)
        qso, qrec = quasar_model(h, q, qrows, dq_point, fm["transform"], fc, cfg, sample_cfg,
                                 roles, fm["ir"])
        model = MultiSurveyModel(
            qso, fm["mixture"], fm["transform"],
            (fm["transform"].bands[fm["ir"]],) + tuple(b for i, b in enumerate(fm["transform"].bands)
                                                     if i != fm["ir"]),
            fm["bounds"], meta=dict(artifact_meta, hemisphere=h, field_fit=fm["record"],
                                    quasar_fit=qrec, softening=fm["softening"],
                                    by="scripts/fit_legacy_baseline.py"))
        results[h] = dict(model=model, fm=fm)
        print(f"[{h}] mixtures done ({time.time() - t0:.0f} s)", flush=True)

    # priors (needs the south transform for C: C is defined on south r)
    dens = {}
    for h, r in results.items():
        tr = r["model"].transform
        ir = r["fm"]["ir"]
        for name, d in (("point", dq_point), ("all", dq_all)):
            m = d["accepted"] & (d["hemisphere"] == h)
            phot = legacy_photometry({k: np.asarray(v)[m] for k, v in qrows.items() if np.ndim(v) == 1},
                                     h, maskbits_zero=True)
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
    for h, r in results.items():
        model, fm = r["model"], r["fm"]
        tr, ir, f = model.transform, fm["ir"], fm["features"]
        label = tr.bands[ir]
        zc = 0.5 * (z_edges[:-1] + z_edges[1:])
        meta_q = dict(artifact_meta, reference_band=label, transform_id=model.transform_id,
                      completeness_constant=C,
                      retention_point_over_all=integrate(dens[(h, "point")], z_edges, m_edges,
                                                         *fc["candidate_ref_range"])
                      / integrate(dens[(h, "all")], z_edges, m_edges, *fc["candidate_ref_range"]),
                      kind="Sigma_Q(z,u_r): weighted draw, PSF, parent coverage x footprint, C from all-morphology south")
        gq = GridQSOPrior(zc, 0.5 * (m_edges[:-1] + m_edges[1:]), dens[(h, "point")] * C, meta_q,
                          z_edges=z_edges, mag_edges=m_edges)
        # field density: non-test cones, each cell by its footprint area
        usable = (fm["role"] != "test") & f.observed[:, ir]
        cones_h = [c for c in cones.values() if c["hemisphere"] == h and roles[c["cell"]] != "test"]
        cells_h = {c["cell"] for c in cones_h}
        total_area = sum(design["cells"][str(c)]["by_hemisphere"][h] for c in cells_h)
        counts, _ = np.histogram(f.x[usable, ir], m_edges, weights=fm["weights"][usable])
        pix = 0
        from qso_pcolor.background import galactic_healpix
        pix = int(galactic_healpix(np.zeros(1), np.full(1, 90.0), 1)[0])
        bd = BackgroundSurfaceDensity(1, 1, m_edges, {(pix, i): float(c) for i, c in enumerate(counts)},
                                      {pix: float(total_area)},
                                      meta=dict(artifact_meta, reference_band=label,
                                                transform_id=model.transform_id,
                                                n_cones=len(cones_h), n_cells=len(cells_h),
                                                kind="Sigma_B(u_r): PSF field, known quasars removed, cone counts x cell footprint area"))
        priors = dict(kind="multisurvey_priors", transform_id=model.transform_id,
                      completeness_constant=C, **artifact_meta,
                      anchors={label: {"qso_prior": gq.to_dict(), "background_density": bd.to_dict()}})
        out = fit_student_t(h, model, fm, fc["outlier"], bundle_meta)
        files = {"model": f"{h}_model.json", "priors": f"{h}_priors.json", "outlier": f"{h}_outlier.json"}
        model.save(tmp / files["model"])
        (tmp / files["priors"]).write_text(json.dumps(priors))
        out.save(tmp / files["outlier"])
        manifest["hemispheres"][h] = files
        print(f"[{h}] priors: PSF/all quasar retention {meta_q['retention_point_over_all']:.3f}; "
              f"Sigma_B over {total_area:.0f} deg^2 in {len(cells_h)} cells", flush=True)
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
