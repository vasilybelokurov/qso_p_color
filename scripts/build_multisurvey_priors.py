#!/usr/bin/env python
"""Surface densities for the multi-survey model, one pair per reference band.

The multi-survey scorer conditions on a measured reference band ``a`` and
evaluates p(other bands | u_a, class). Turning that into intensities needs
Sigma_Q(z, u_a) and Sigma_B(u_a), per unit native luptitude in that band,
for objects **with band a measured** -- the same condition for both classes.

Quasars. ``data/multisurvey/quasars.npz`` is a flat draw of up to 2,000 per
dz = 0.1 from the de-duplicated DESI + SDSS parent at |b| >= 25 deg. Each
object is weighted back by N_parent(bin) / n_drawn(bin), which restores the
parent's redshift distribution; within a bin the draw is uniform, so the
magnitude distribution is unbiased. Only quasars inside the footprint of a's
survey count, and the area is that footprint's share of the parent coverage.
A band can be missing because the object is outside the survey or because it
is too faint; only the second is information about the population, so
counting objects outside the footprint as "not detected" would bias the prior.

Fields. ``data/multisurvey/background.npz`` holds every clean source (known
quasars removed) in 24 cones of fixed radius. A cone is in a survey's
footprint if that survey detected anything in it; outside, the fraction is
exactly zero (measured). Sigma_B,a = sources with a measured / area of the
cones in the footprint.

Normalisation. The raw quasar densities carry the spectroscopic
incompleteness of the parent. One constant C is applied to every anchor,
fixed so that the Legacy-south r prior integrates to the validated original
prior (``models/archive/original_legacy_south/sigma_q_south.json``) over its own magnitude range. That
assumes the completeness does not depend on which band is the reference.

    python scripts/build_multisurvey_priors.py
    python scripts/build_multisurvey_priors.py --config configs/multisurvey_priors.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def parent_bin_counts(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eligible de-duplicated parent: counts per sampling bin, and positions."""
    from build_multisurvey_sample import deduplicate
    from qso_pcolor.data import galactic_from_equatorial

    parts = [dict(np.load(cfg[k])) for k in ("desi_cache", "sdss_cache")]
    ra, dec, z = [np.concatenate([p[k] for p in parts]) for k in ("ra", "dec", "zspec")]
    keep = deduplicate(ra, dec, cfg["dedup_radius_arcsec"])
    ra, dec, z = ra[keep], dec[keep], z[keep]
    l, b = galactic_from_equatorial(ra, dec)
    eligible = np.abs(b) >= cfg["min_abs_b_deg"]
    edges = np.arange(cfg["z_min"], cfg["z_max"] + cfg["z_step"] / 2, cfg["z_step"])
    counts, _ = np.histogram(z[eligible], edges)
    return edges, counts, np.column_stack([l[eligible], b[eligible]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/multisurvey_priors.json"))
    args = ap.parse_args()
    pc = json.loads(args.config.read_text())
    cfg = json.loads(Path(pc["sample_config"]).read_text())

    import healpy as hp

    from qso_pcolor.background import galactic_healpix
    from qso_pcolor.multisurvey import MultiSurveyModel
    from qso_pcolor.multisurvey_data import Photometry, survey_of
    from qso_pcolor.priors import BackgroundSurfaceDensity, GridQSOPrior

    model = MultiSurveyModel.load(pc["model_path"])
    bands = model.transform.bands
    tid = model.transform_id
    nside = pc["footprint_nside"]
    cell_area = hp.nside2pixarea(nside, degrees=True)

    # -- quasars ---------------------------------------------------------------
    t0 = time.time()
    z_edges, n_parent, parent_lb = parent_bin_counts(cfg)
    q = dict(np.load(Path(cfg["data_dir"]) / "quasars.npz"))
    qp = Photometry(q["flux"], q["variance"], tuple(str(x) for x in q["bands"])).align(bands)
    qf = model.transform(qp)
    qobs = qf.observed & (qf.observed.sum(1) >= pc["min_bands"])[:, None]
    ibin = np.clip(np.digitize(q["zspec"], z_edges) - 1, 0, n_parent.size - 1)
    n_drawn = np.bincount(ibin, minlength=n_parent.size)
    w = n_parent[ibin] / n_drawn[ibin]
    assert np.isclose(w.sum(), n_parent.sum()), "weights must restore the parent count"
    ppix = galactic_healpix(parent_lb[:, 0], parent_lb[:, 1], nside)
    pc_counts = np.bincount(ppix, minlength=hp.nside2npix(nside))
    parent_cells = pc_counts >= pc["parent_min_per_cell"]
    qpix = galactic_healpix(q["l"], q["b"], nside)
    print(f"parent: {n_parent.sum():,} eligible quasars, {parent_cells.sum()} cells "
          f"= {parent_cells.sum() * cell_area:,.0f} deg^2 ({time.time() - t0:.0f} s)")

    # -- fields ----------------------------------------------------------------
    f = dict(np.load(Path(cfg["data_dir"]) / "background.npz"))
    fp = Photometry(f["flux"], f["variance"], tuple(str(x) for x in f["bands"])).align(bands)
    ff = model.transform(fp)
    fobs = ff.observed & (ff.observed.sum(1) >= pc["min_bands"])[:, None]
    field_ids = np.unique(f["field"])
    cone_area = np.pi * cfg["field_radius_deg"] ** 2

    raw_q, raw_b, report = {}, {}, {}
    for a, label in enumerate(bands):
        survey = survey_of(label)
        system = label.split(":")[0]
        cols = [i for i, b in enumerate(bands) if b.split(":")[0] == system]
        # quasar footprint: fine cells in which this survey detected >= N quasars
        det = np.bincount(qpix[qobs[:, cols].any(1)], minlength=parent_cells.size)
        fp_cells = parent_cells & (det >= pc["footprint_min_detections"])
        in_fp = fp_cells[qpix]
        area_q = fp_cells.sum() * cell_area
        use_q = in_fp & qobs[:, a]
        # field footprint: cones in which this survey detected anything
        cones = [k for k in field_ids if fobs[f["field"] == k][:, cols].any()]
        use_b = np.isin(f["field"], cones) & fobs[:, a]
        area_b = len(cones) * cone_area
        entry = {"survey": survey, "n_quasars": int(use_q.sum()),
                 "n_quasars_weighted": float(w[use_q].sum()), "area_q_deg2": float(area_q),
                 "n_field": int(use_b.sum()), "n_cones": len(cones), "area_b_deg2": area_b}
        report[label] = entry
        if (use_q.sum() < pc["min_quasars_per_anchor"]
                or use_b.sum() < pc["min_field_sources_per_anchor"]):
            entry["skipped"] = "too few objects"
            continue
        um = np.concatenate([qf.x[use_q, a], ff.x[use_b, a]])
        lo, hi = np.quantile(um, pc["mag_quantiles"])
        width = pc["mag_bin_width"]
        m_edges = np.arange(np.floor(lo / width) * width, np.ceil(hi / width) * width + width / 2,
                            width)
        hq, _, _ = np.histogram2d(q["zspec"][use_q], qf.x[use_q, a], [z_edges, m_edges],
                                  weights=w[use_q])
        raw_q[label] = (hq / (area_q * np.diff(z_edges)[:, None] * np.diff(m_edges)[None]),
                        m_edges)
        raw_b[label] = (ff.x[use_b, a], area_b, m_edges, cones)
        entry["mag_edges"] = [float(m_edges[0]), float(m_edges[-1])]

    # -- one completeness constant ----------------------------------------------
    norm = pc["normalisation"]
    ref = GridQSOPrior.load(norm["reference_prior"])
    lo, hi = norm["mag_range"]
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    dm_fine = 0.01                      # quadrature step for the reference integral
    ref_total = sum(float(np.sum(ref(zc, mm) * np.diff(z_edges))) * dm_fine
                    for mm in np.arange(lo + dm_fine / 2, hi, dm_fine))
    s, m_edges = raw_q[norm["anchor"]]
    mc = 0.5 * (m_edges[:-1] + m_edges[1:])
    inside = (mc > lo) & (mc < hi)
    raw_total = float(np.sum(s[:, inside] * np.diff(z_edges)[:, None]
                             * np.diff(m_edges)[None, inside]))
    completeness_c = ref_total / raw_total
    print(f"normalisation: reference {ref_total:.1f} deg^-2, raw {raw_total:.1f} deg^-2 "
          f"over {norm['anchor']} {lo}-{hi}: C = {completeness_c:.3f}")

    out = {"kind": "multisurvey_priors", "transform_id": tid,
           "model_run_id": model.meta.get("run_id"),
           "built": time.strftime("%Y-%m-%d"), "by": "scripts/build_multisurvey_priors.py",
           "config": pc, "completeness_constant": completeness_c,
           "parent_eligible": int(n_parent.sum()),
           "parent_area_deg2": float(parent_cells.sum() * cell_area),
           "anchors": {}, "report": report}
    for label, (s, m_edges) in raw_q.items():
        meta = {"reference_band": label, "transform_id": tid,
                "kind": "Sigma_Q(z, u_a): weighted DESI+SDSS draw, footprint-restricted, "
                        "one completeness constant"}
        gq = GridQSOPrior(zc, 0.5 * (m_edges[:-1] + m_edges[1:]), s * completeness_c, meta,
                          z_edges=z_edges, mag_edges=m_edges)
        mags, area_b, _, cones = raw_b[label]
        # every field source into one cell: a single global density, area exact
        bd = BackgroundSurfaceDensity.from_catalogue(
            mags, np.zeros(mags.size), np.full(mags.size, 90.0), mag_edges=m_edges,
            nside=1, nside_parent=1,
            area_per_pixel={int(galactic_healpix(np.zeros(1), np.full(1, 90.0), 1)[0]): area_b},
            meta={"reference_band": label, "transform_id": tid, "cones": [int(c) for c in cones],
                  "kind": "Sigma_B(u_a): clean field sources with a measured / cone area"})
        out["anchors"][label] = {"qso_prior": gq.to_dict(), "background_density": bd.to_dict()}
    Path(pc["out"]).write_text(json.dumps(out))
    print(f"{len(out['anchors'])} anchors with priors; skipped: "
          + ", ".join(k for k, v in report.items() if "skipped" in v))
    for k, v in report.items():
        print(f"  {k:24s} Q {v['n_quasars']:6d} over {v['area_q_deg2']:7.0f} deg^2   "
              f"B {v['n_field']:6d} in {v['n_cones']:2d} cones  {v.get('skipped', '')}")
    print(f"wrote {pc['out']}")


if __name__ == "__main__":
    main()
