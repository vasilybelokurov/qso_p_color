#!/usr/bin/env python
"""Fit the spatial weights, PSF count priors and outlier share of a candidate.

Every fit excludes the original held-out field cones/quasar blocks. The result
is an inspectable candidate bundle; promotion is a separate validation step.
"""
import argparse
import hashlib
import json
from pathlib import Path

import healpy as hp
import numpy as np

from build_multisurvey_priors import parent_bin_counts
from qso_pcolor.background import galactic_healpix
from qso_pcolor.data import _save_npz
from qso_pcolor.joint_spatial import fit_joint_spatial_weights
from qso_pcolor.multisurvey import MultiSurveyModel, MultiSurveyOutlier
from qso_pcolor.multisurvey_data import Photometry, survey_of
from qso_pcolor.outlier import fit_outlier_fraction, mixture_moments
from qso_pcolor.priors import GridQSOPrior
from qso_pcolor.spatial import SpatialSurfaceDensity


def merge_empty_bins(values, edges):
    """Merge empty field bins rather than inventing a positive count floor."""
    edges = np.asarray(edges, float)
    while len(edges) > 2:
        counts = np.histogram(values, edges)[0]
        empty = np.flatnonzero(counts == 0)
        if not len(empty):
            break
        i = int(empty[0])
        edges = np.delete(edges, 1 if i == 0 else i)
    return edges


def native_values(phot, transform):
    """Luptitude values alone, avoiding allocation of unused dense covariances."""
    aligned = phot.align(transform.bands)
    values = 22.5 - 2.5 / np.log(10.) * (np.arcsinh(
        aligned.flux / (2 * transform.softening)) + np.log(transform.softening))
    return values, aligned.observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/multisurvey_psf_recovery.json"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    model = MultiSurveyModel.load(cfg["output"])
    root = Path(model.meta["training_cache"])
    bundle = root / "bundle"
    bundle.mkdir(exist_ok=True)
    signature = hashlib.sha256(json.dumps(dict(
        configuration={key: cfg[key] for key in ("area", "spatial", "prior", "outlier", "sample_config")},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(Path(cfg["output"]).read_bytes()).hexdigest(),
        area_sha256=hashlib.sha256((Path(cfg["cache_dir"]) / "areas.json").read_bytes()).hexdigest(),
        source_priors_sha256=hashlib.sha256(Path(cfg["prior"]["source_priors"]).read_bytes()).hexdigest(),
        source_outlier_sha256=hashlib.sha256(Path(cfg["outlier"]["source"]).read_bytes()).hexdigest()),
        sort_keys=True).encode()).hexdigest()
    signature_path = root / "population_fit_signature.txt"
    reuse = signature_path.exists() and signature_path.read_text().strip() == signature
    sample_cfg = json.loads(Path(cfg["sample_config"]).read_text())
    b = dict(np.load(root / "background.npz"))
    q = dict(np.load(root / "quasars.npz"))
    bands = model.transform.bands
    areas = json.loads((Path(cfg["cache_dir"]) / "areas.json").read_text())
    area_by_id = {entry["field"]: entry for entry in areas}
    train = ~b["held"] & np.isin(b["field"], [a["field"] for a in areas if a["contained"]])
    bp = Photometry(b["flux"], b["variance"], bands)
    features = model.transform(bp.subset(np.flatnonzero(train)))
    sky = cfg["spatial"]
    spatial_path = root / "spatial_weights.json"
    if reuse and spatial_path.exists():
        from qso_pcolor.joint_spatial import JointSpatialWeights
        model.spatial_background = JointSpatialWeights.from_dict(json.loads(spatial_path.read_text()))
    else:
        model.spatial_background = fit_joint_spatial_weights(model.background,
            features.x, features.cov, features.observed, b["l"][train], b["b"][train],
            np.ones(train.sum()), nside=sky["nside"], nside_parent=sky["nside_parent"],
            n0=sky["colour_n0"], max_iter=sky["max_iter"], tol=sky["tol"],
            meta=dict(population="psf", policy_origin=sky["policy_origin"],
                      heldout_fields=np.unique(b["field"][b["held"]]).tolist()))
        spatial_path.write_text(json.dumps(model.spatial_background.to_dict()))
    del features
    print("Spatial joint weights fitted", flush=True)

    priors_path = bundle / "priors.json"
    if not reuse or not priors_path.exists():
        original = dict(np.load(Path(cfg["sample_dir"]) / "quasars.npz"))
        original_phot = Photometry(original["flux"], original["variance"], bands)
        qvalues, qobserved = native_values(original_phot, model.transform)
        bvalues, bobserved = native_values(bp, model.transform)
        qkeep = np.zeros(len(original["ra"]), bool); qkeep[q["original_row"]] = True
        parent_cache = root / "parent_counts.npz"
        if parent_cache.exists():
            parent = dict(np.load(parent_cache)); zedges, nparent, positions = (
                parent["z_edges"], parent["counts"], parent["positions"])
        else:
            zedges, nparent, positions = parent_bin_counts(sample_cfg, True)
            _save_npz(parent_cache, z_edges=zedges, counts=nparent, positions=positions)
        old = json.loads(Path(cfg["prior"]["source_priors"]).read_text())
        pc = old["config"]
        nside = pc["footprint_nside"]
        npix = hp.nside2npix(nside)
        parent_cells = np.bincount(galactic_healpix(positions[:, 0], positions[:, 1], nside),
                                  minlength=npix) >= pc["parent_min_per_cell"]
        hpmeta = json.loads(Path(sample_cfg["holdout_model"]).read_text())["meta"]
        shift = 2 * int(round(np.log2(nside / hpmeta["holdout_nside"])))
        parent_cells &= ~np.isin(np.arange(npix) >> shift, hpmeta["holdout_blocks"])
        qpix = galactic_healpix(original["l"], original["b"], nside)
        iz = np.clip(np.digitize(original["zspec"], zedges) - 1, 0, len(nparent) - 1)
        qtrain = ~original["held"]
        # Restore the ORIGINAL flat draw before selecting the PSF population.
        drawn = np.bincount(iz[qtrain], minlength=len(nparent))
        weights = np.where(qtrain, nparent[iz] / np.maximum(drawn[iz], 1), 0.)
        result = dict(kind="multisurvey_priors", transform_id=model.transform_id,
            model_run_id=model.meta["run_id"], population="psf", anchors={}, report={},
            completeness_constant=old["completeness_constant"],
            completeness_origin=cfg["prior"]["source_priors"],
            note="Original completeness constant retained without renormalising away PSF selection; survey footprints inferred as in the original prior.")
        for a, label in enumerate(bands):
            system = label.split(":")[0]
            cols = [j for j, name in enumerate(bands) if name.split(":")[0] == system]
            detections = np.bincount(qpix[qobserved[:, cols].any(axis=1)], minlength=npix)
            footprint = parent_cells & (detections >= pc["footprint_min_detections"])
            use_q = qtrain & qkeep & qobserved[:, a] & footprint[qpix]
            # Coverage is measured before selecting the anchor's detections.
            cones = [f for f in np.unique(b["field"][train])
                     if bobserved[b["field"] == f][:, cols].any()]
            use_b = train & bobserved[:, a] & np.isin(b["field"], cones)
            nq, nb = int(use_q.sum()), int(use_b.sum())
            result["report"][label] = dict(n_quasars=nq, n_field=nb)
            if min(nq, nb) < cfg["prior"]["minimum_count"]:
                result["report"][label]["status"] = "no_prior: insufficient population counts"
                continue
            values = np.r_[qvalues[use_q, a], bvalues[use_b, a]]
            lo, hi = np.quantile(values, pc["mag_quantiles"])
            width = pc["mag_bin_width"]
            edges = np.arange(np.floor(lo / width) * width, np.ceil(hi / width) * width + width / 2, width)
            edges = merge_empty_bins(bvalues[use_b, a], edges)
            aq = footprint.sum() * hp.nside2pixarea(nside, degrees=True)
            counts = np.histogram2d(original["zspec"][use_q], qvalues[use_q, a],
                                    [zedges, edges], weights=weights[use_q])[0]
            sigma = counts * old["completeness_constant"] / (aq * np.diff(zedges)[:, None] * np.diff(edges))
            meta = dict(reference_band=label, transform_id=model.transform_id,
                model_run_id=model.meta["run_id"], population="psf", n_quasars=nq, n_field=nb,
                sparse_population_prior=(nq < pc["min_quasars_per_anchor"] or nb < pc["min_field_sources_per_anchor"]))
            qp = GridQSOPrior((zedges[1:] + zedges[:-1]) / 2, (edges[1:] + edges[:-1]) / 2,
                               sigma, meta=dict(meta, area_deg2=float(aq)), z_edges=zedges, mag_edges=edges)
            cell_area, cell_counts = {}, {}
            for f in cones:
                entry = area_by_id[int(f)]
                cell = int(galactic_healpix([entry["l"]], [entry["b"]], sky["nside"])[0])
                cell_area[cell] = cell_area.get(cell, 0.) + entry["area"]["area_deg2"]
                hist = np.histogram(bvalues[use_b & (b["field"] == f), a], edges)[0]
                for j, count in enumerate(hist):
                    cell_counts[cell, j] = cell_counts.get((cell, j), 0.) + int(count)
            pooled = np.histogram(bvalues[use_b, a], edges)[0] / (sum(cell_area.values()) * np.diff(edges))
            density = SpatialSurfaceDensity(sky["nside"], sky["nside_parent"], edges,
                cell_counts, cell_area, pooled, sky["density_n0"],
                dict(meta, area_origin="Legacy maskbits/nexp images; survey coverage from cone detections"))
            result["anchors"][label] = dict(qso_prior=qp.to_dict(), background_density=density.to_dict())
        priors_path.write_text(json.dumps(result))
        print(f"PSF spatial population priors: {len(result['anchors'])} bands", flush=True)

    outlier_path = bundle / "outlier.json"
    if not reuse or not outlier_path.exists():
        source = MultiSurveyOutlier.load(cfg["outlier"]["source"])
        fit_rows = np.asarray(model.meta["background"]["fitted_rows"])
        calibrate = train.copy(); calibrate[fit_rows] = False
        rows = np.flatnonzero(calibrate)
        features = model.transform(bp.subset(rows))
        order = [bands.index(band) for band in model.reference_priority]
        anchors = np.array(order)[np.argmax(features.observed[:, order], axis=1)]
        mean, covariance = mixture_moments([model.background])
        outlier = MultiSurveyOutlier(mean, source.kappa**2 * covariance, source.kappa, 0.,
            bands, model.transform_id, {"*": ([0., 1.], [0.])},
            meta=dict(model_run_id=model.meta["run_id"], population="psf",
                shape_hyperparameters_from=cfg["outlier"]["source"], calibration_rows=len(rows)),
            family="student_t", nu=source.nu, noise=source.noise)
        lp, lu = np.empty(len(rows)), np.empty(len(rows))
        fractions = {}
        for a in np.unique(anchors):
            select = anchors == a
            label = bands[a]
            f = features.subset(np.flatnonzero(select))
            lp[select] = model.background_log_prob(f.x, f.cov, f.observed, int(a),
                l_deg=b["l"][rows[select]], b_deg=b["b"][rows[select]])
            lu[select] = outlier.conditional(int(a), label, model.qso.system).log_prob(
                f.x, f.cov, observed=f.observed)
            if select.sum() >= cfg["outlier"]["min_per_bin"] * cfg["outlier"]["n_mag_bins"]:
                values = f.x[:, a]
                edges = np.unique(np.quantile(values, np.linspace(0, 1, cfg["outlier"]["n_mag_bins"] + 1)))
                indices = np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)
                eta = np.array([fit_outlier_fraction(lp[select][indices == j], lu[select][indices == j])
                                for j in range(len(edges) - 1)])
                fractions[label] = (edges, eta)
        pooled = fit_outlier_fraction(lp, lu)
        values = features.x[np.arange(len(rows)), anchors]
        fractions["*"] = (np.array([values.min(), values.max()]), np.array([pooled]))
        outlier.fractions = fractions
        outlier.save(outlier_path)
        print(f"Outlier share fitted on {len(rows)} unused shape-training rows", flush=True)
    model.meta["completion_config"] = cfg
    model.meta["validation_status"] = "PSF spatial fit; validation scope is recorded in the bundle manifest"
    model.save(bundle / "model.json")
    files = {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
             for name in ("model.json", "priors.json", "outlier.json")}
    manifest = dict(kind="multisurvey_psf_bundle", files=files,
        bundle_id=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16],
        min_abs_b_deg=sample_cfg["min_abs_b_deg"], population="psf",
        status="candidate: requires reserved-field validation", method="one joint 41-band model")
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))
    signature_path.write_text(signature)
    print(f"Complete candidate bundle: {bundle}", flush=True)


if __name__ == "__main__":
    main()
