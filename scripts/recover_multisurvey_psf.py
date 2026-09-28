#!/usr/bin/env python
"""Refit the 41-band PSF population without changing the band transform.

This writes a candidate, never a release pointer. Original reserved sky blocks
and field cones remain reserved. Quasar slices retain their previous component
counts and start from the previous fit; field capacity is selected on separate
training cones. All 41 coordinates share the same missing-band XD likelihood.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from qso_pcolor.data import _save_npz
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.legacy import legacy_match, morphology_status
from qso_pcolor.multisurvey import MultiSurveyModel
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.qso_model import SlicedColourRedshiftModel
from qso_pcolor.xd import fit_xd
from train_multisurvey_model import fit_and_select, sample_patterns


def psf_rows(rows: dict) -> np.ndarray:
    """Morphology/quality selection independent of supplied photometric bands."""
    return ((morphology_status(rows["type"]) == "point") &
            (np.asarray(rows["maskbits"]) == 0) &
            np.isfinite(rows["ra"]) & np.isfinite(rows["dec"]))


def subset_catalogue(data, keep):
    return {k: (v[keep] if k != "bands" and v.ndim and len(v) == len(keep) else v)
            for k, v in data.items()}


def fit_part(config, root, part):
    root = Path(root)
    path = root / ("background.json" if part == "background" else f"qso_{part:02d}.json")
    if path.exists():
        cached = json.loads(path.read_text())
        if part != "background" or "bounds" in cached:
            return part, cached
    started = time.monotonic()
    model = MultiSurveyModel.load(config["source_model"])
    if part == "background":
        data = dict(np.load(root / "background.npz"))
        phot = Photometry(data["flux"], data["variance"], tuple(data["bands"]))
        eligible = ~data["held"] & phot.observed.any(axis=1)
        rng = np.random.default_rng(config["seed"] + 10)
        choose, weights = sample_patterns(phot.observed, eligible,
            config["max_background_fit"], config["min_band_training"], rng)
        fitted = np.zeros(len(eligible), bool); fitted[choose] = True
        fields = np.unique(data["field"][fitted])
        if not 0 < config["background_selection_fraction"] < 1:
            raise ValueError("background selection fraction must lie between zero and one")
        for _ in range(10000):
            reserved = rng.choice(fields, max(1, round(config["background_selection_fraction"] * len(fields))), replace=False)
            train = fitted & ~np.isin(data["field"], reserved)
            if (phot.observed[train].sum(axis=0) >= config["min_band_training"]).all():
                break
        else:
            raise ValueError("cannot reserve selection cones and retain all 41 bands")
        features = model.transform(phot.subset(choose))
        cfg = dict(config, _global_train=train[choose], _weights=weights[choose])
        _, record = fit_and_select("background", features, train[choose], ~train[choose],
            cfg, path, model.reference_priority, config["seed"])
        record.update(selection_fields=reserved.tolist(), fitted_rows=choose.tolist(),
                      heldout_fields=np.unique(data["field"][data["held"]]).tolist(),
                      bounds=[[float(np.min(features.x[features.observed[:, j], j])),
                               float(np.max(features.x[features.observed[:, j], j]))]
                              for j in range(len(features.labels))])
    else:
        data = dict(np.load(root / "quasars.npz"))
        centres = model.qso.z_centres
        width = float(model.meta["settings"]["z_step"])
        selected = (~data["held"] & (data["zspec"] >= centres[part] - width) &
                    (data["zspec"] < centres[part] + width))
        phot = Photometry(data["flux"][selected], data["variance"][selected], tuple(data["bands"]))
        if len(phot.flux) < model.qso.mixtures[part].n_components:
            raise ValueError(f"slice {part} has insufficient PSF training rows")
        features = model.transform(phot)
        result = fit_xd(features.x, features.cov, observed=features.observed,
            init=model.qso.mixtures[part], seed=config["seed"] + part,
            labels=model.transform.bands, max_iter=config["max_iter"], tol=config["tol"],
            regularization=config["regularization"])
        record = dict(mixture=result.mixture.to_dict(), n_fit=len(phot.flux),
            n_iter=result.n_iter, converged=result.converged,
            mean_training_log_density=result.mean_loglike, convergence_tail=result.history[-5:],
            band_counts=phot.observed.sum(axis=0).tolist(),
            elapsed_s=time.monotonic() - started)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record)); tmp.replace(path)
    return part, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/multisurvey_psf_recovery.json"))
    parser.add_argument("--resume", type=Path,
                        help="reuse completed fits with identical inputs/configuration after a metadata-only repair")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source = MultiSurveyModel.load(cfg["source_model"])
    sample = Path(cfg["sample_dir"])
    q = dict(np.load(sample / "quasars.npz"))
    b = dict(np.load(sample / "background.npz"))
    qm = dict(np.load(cfg["quasar_morphology"]))
    if not (np.array_equal(q["ra"], qm["target_ra"]) and
            np.array_equal(q["dec"], qm["target_dec"])):
        raise ValueError("quasar morphology rows do not match the photometric sample")
    bm = legacy_match(b["ra"], b["dec"], Path(cfg["cache_dir"]) / "queries",
                      radius_arcsec=cfg["match_radius_arcsec"])
    qkeep, bkeep = psf_rows(qm), psf_rows(bm)
    provenance = dict(config=cfg, source_sha256=hashlib.sha256(Path(cfg["source_model"]).read_bytes()).hexdigest(),
        quasars_sha256=hashlib.sha256((sample / "quasars.npz").read_bytes()).hexdigest(),
        background_sha256=hashlib.sha256((sample / "background.npz").read_bytes()).hexdigest(),
        quasar_selection_sha256=hashlib.sha256(qkeep.tobytes()).hexdigest(),
        field_selection_sha256=hashlib.sha256(bkeep.tobytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    run_id = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(cfg["cache_dir"]) / run_id
    if args.resume is not None:
        previous = json.loads((args.resume / "provenance.json").read_text())
        excluded = {"script_sha256", "config"}
        fit_keys = ("seed", "source_model", "sample_dir", "quasar_morphology",
                    "match_radius_arcsec", "max_background_fit", "min_band_training",
                    "background_k_candidates", "selection_max_iter", "max_iter", "tol",
                    "regularization", "population")
        if ({k: v for k, v in previous.items() if k not in excluded} !=
                {k: v for k, v in provenance.items() if k not in excluded} or
                any(previous["config"][k] != cfg[k] for k in fit_keys) or
                previous["config"].get("background_selection_fraction", .2) != cfg["background_selection_fraction"]):
            raise ValueError("resume inputs, population or configuration changed")
        root = args.resume
        run_id = root.name
        provenance = dict(previous, assembly_script_sha256=provenance["script_sha256"])
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "provenance.json").exists():
        (root / "provenance.json").write_text(json.dumps(provenance, indent=2))
    _save_npz(root / "population_masks.npz", qso=qkeep, background=bkeep)
    for name, data, keep in (("quasars", q, qkeep), ("background", b, bkeep)):
        path = root / f"{name}.npz"
        if not path.exists():
            _save_npz(path, **subset_catalogue(data, keep), original_row=np.flatnonzero(keep))
        print(f"{name}: {keep.sum()} PSF rows, {data['held'][keep].sum()} reserved", flush=True)
    records = {}
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = [pool.submit(fit_part, cfg, str(root), part)
                   for part in ["background", *range(len(source.qso.z_centres))]]
        for future in as_completed(futures):
            part, record = future.result()
            records[part] = record
            print(f"{part}: n={record['n_fit']}, iterations={record['n_iter']}, "
                  f"converged={record['converged']}", flush=True)
    qrecords = [records[i] for i in range(len(source.qso.z_centres))]
    qso = SlicedColourRedshiftModel(source.qso.z_centres,
        [GaussianMixture.from_dict(r["mixture"]) for r in qrecords],
        np.array([r["n_fit"] for r in qrecords]), f"multisurvey_psf_{run_id}", source.qso.labels,
        meta=dict(source.qso.meta, per_slice=[{k: v for k, v in r.items() if k != "mixture"}
                                            for r in qrecords], population="psf",
                  n_fit=int((qkeep & ~q["held"]).sum()),
                  n_holdout=int((qkeep & q["held"]).sum()),
                  selection_blocks=[], source_selection_blocks=source.qso.meta.get("selection_blocks", []),
                  component_count_policy="inherited slice counts; PSF refit initialised from source model"))
    field = GaussianMixture.from_dict(records["background"]["mixture"])
    model = MultiSurveyModel(qso, field, source.transform, source.reference_priority,
        np.array(records["background"]["bounds"]), meta=dict(run_id=run_id, population="psf", provenance=provenance,
            training_cache=str(root), validation_status="candidate; not yet validated",
            background={k: v for k, v in records["background"].items() if k != "mixture"}))
    model.save(cfg["output"])
    print(f"Candidate saved: {cfg['output']}", flush=True)


if __name__ == "__main__":
    main()
