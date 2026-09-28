#!/usr/bin/env python
"""Compare frozen catch-alls on colour planes and reserved individual objects.

No close pairs. The unchanged QSO and stellar intensities are evaluated once;
only the fourth term and the stellar (1-eta) factor are replaced afterwards.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from compare_psf_catchalls import evaluate_densities, fractions_at
from qso_pcolor import PSFMultiSurveyBaseline, Photometry, RedshiftMatch, BlendPolicy
from qso_pcolor.multisurvey import MultiSurveyOutlier


def reweight(raw, log_u, eta):
    """Exact score update at fixed QSO/stars/priors/window; all logs are natural."""
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.logaddexp(raw["log_lambda_sameq"], raw["log_lambda_fieldq"])
        log_sigma_b = raw["log_lambda_bkg"] - raw["loglike_bkg"]
        b = raw["log_lambda_bkg"] + np.log1p(-eta)
        u = log_sigma_b + np.log(eta) + log_u
        total = logsumexp(np.stack([q, b, u]), axis=0)
        return dict(p_quasar=np.exp(q-total), p_outlier=np.exp(u-total),
            log_r=raw["log_lambda_sameq"]-np.log(raw["dz_match_eff"])-total,
            log_bf=raw["loglike_qso_zprimary"]-np.logaddexp(b,u)+log_sigma_b)


def raw_scores(baseline, data, cfg):
    args = cfg["validation"]
    model = baseline.model
    rows = []
    for lo in range(0, len(data["flux"]), cfg["chunk_size"]):
        r = slice(lo,lo+cfg["chunk_size"])
        scores = model.score(Photometry(data["flux"][r], data["variance"][r], tuple(data["bands"])),
            z_primary=data["zprimary"][r], l_deg=data["l"][r], b_deg=data["b"][r],
            match=RedshiftMatch(half_width_kms=args["window_kms"]), priors=baseline.priors,
            ood_flag_sigma=args["ood_flag_sigma"])
        rows.extend(scores)
    keys = ("log_lambda_sameq", "log_lambda_fieldq", "log_lambda_bkg", "loglike_bkg",
            "loglike_qso_zprimary", "dz_match_eff", "qso_ood_sigma_any_z", "bkg_ood_sigma")
    result = {k: np.array([getattr(r,k) for r in rows]) for k in keys}
    result["eligible"] = np.array([r.status == "ok" and
        not {"outside_both_models", "background_out_of_mag_range"}.intersection(r.quality_flags) for r in rows])
    return result


def grid_data(model, hemisphere, magnitude, cfg):
    v = cfg["validation"]
    bandnames = tuple(f"decals_dr9_{hemisphere}:{b}" for b in ("g","r","z"))
    axes = np.linspace(*v["grid_colour_range"], v["grid_size"])
    a,b = np.meshgrid(axes,axes)
    x = np.column_stack([magnitude+a.ravel(), np.full(a.size,magnitude), magnitude-b.ravel()])
    softening = model.transform.softening[[model.transform.bands.index(b) for b in bandnames]]
    scale = 2.5/np.log(10.)
    flux = 2*softening*np.sinh((22.5-x)/scale-np.log(softening))
    variance = v["luptitude_error"]**2*(flux**2+(2*softening)**2)/scale**2
    return dict(flux=flux, variance=variance, bands=bandnames,
        l=np.full(a.size,180.), b=np.full(a.size,45.), zprimary=np.full(a.size,v["primary_z"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/psf_catchall_comparison.json"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text()); v = cfg["validation"]
    base = PSFMultiSurveyBaseline.load(cfg["baseline"])
    root = Path(cfg["work"])
    alternatives = {name: MultiSurveyOutlier.load(root/f"{name}.json") for name in ("gaussian","student_t")}
    outliers = dict(previous=base.outlier, **alternatives)
    signature = hashlib.sha256(json.dumps(dict(config=cfg, files=base.manifest["files"],
        script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outliers={k:hashlib.sha256(json.dumps(u.to_dict(),sort_keys=True).encode()).hexdigest()
                  for k,u in outliers.items()}),sort_keys=True).encode()).hexdigest()[:16]
    cache = root/f"validation_{signature}"; cache.mkdir(exist_ok=True)
    report = dict(source_bundle=base.manifest["bundle_id"], signature=signature, grids={}, real={},
                  no_close_pairs=True, config=cfg)
    datasets = {}
    for hemisphere in ("south","north"):
        for mag in v["reference_magnitudes"]:
            datasets[f"{hemisphere}_{mag}"] = grid_data(base.model,hemisphere,mag,cfg)
    rng = np.random.default_rng(cfg["seed"])
    for kind in ("quasars","background"):
        data = dict(np.load(Path(base.model.meta["training_cache"])/f"{kind}.npz"))
        rows = np.flatnonzero(data["held"])
        if kind == "quasars":
            rows = rows[base.model.qso.in_support(data["zspec"][rows])]
        rows = np.sort(rng.choice(rows,min(v["real_rows_per_class"],len(rows)),replace=False))
        datasets[kind] = {k:data[k][rows] for k in ("flux","variance","l","b")}
        datasets[kind].update(bands=tuple(data["bands"]),
            zprimary=data["zspec"][rows] if kind=="quasars" else np.full(len(rows),v["primary_z"]))
        np.savez(cache/f"{kind}_rows.npz",rows=rows)
    for name,data in datasets.items():
        path = cache/f"{name}.npz"
        if path.exists():
            saved = dict(np.load(path)); raw = {k[4:]:x for k,x in saved.items() if k.startswith("raw_")}
            density = {k[5:]:x for k,x in saved.items() if k.startswith("dens_")}
        else:
            raw = raw_scores(base,data,cfg)
            # evaluate_densities expects the full native schema, without inventing measurements.
            aligned = Photometry(data["flux"],data["variance"],tuple(data["bands"])).align(base.model.transform.bands)
            full = dict(data,flux=aligned.flux,variance=aligned.variance)
            density = evaluate_densities(base.model,full,np.arange(len(aligned.flux)),outliers,
                                         chunk=cfg["chunk_size"],background=False)
            np.savez(path,**{f"raw_{k}":x for k,x in raw.items()},**{f"dens_{k}":x for k,x in density.items()})
        results = {"none":reweight(raw,np.zeros(len(data["flux"])),np.zeros(len(data["flux"]))) }
        for key,outlier in outliers.items():
            eta = fractions_at(outlier.fractions,density["anchor"],density["magnitude"],base.model.transform.bands)
            results[key] = reweight(raw,density[key],eta)
        if name in ("quasars","background"):
            valid = raw["eligible"] & np.isfinite(results["previous"]["log_r"])
            entry = dict(n=len(valid),eligible=int(valid.sum()),variants={})
            for key,r in results.items():
                shift = r["log_r"][valid]-results["previous"]["log_r"][valid]
                entry["variants"][key] = dict(qso_above_half=int((r["p_quasar"][valid]>.5).sum()),
                    median_log_r_shift=float(np.median(shift)),
                    log_r_shift_quantiles=np.quantile(shift,[.05,.95]).tolist(),
                    median_p_outlier=float(np.median(r["p_outlier"][valid])))
            report["real"][name] = entry
        else:
            q = np.logaddexp(raw["log_lambda_sameq"],raw["log_lambda_fieldq"])
            b = raw["log_lambda_bkg"]
            entry = dict(n=len(q),eligible=int(raw["eligible"].sum()),low_density={})
            for fraction in v["low_density_peak_fractions"]:
                use = (q<np.nanmax(q)+np.log(fraction)) & (b<np.nanmax(b)+np.log(fraction))
                metrics = {"n":int(use.sum()),"variants":{}}
                for key,r in results.items():
                    high = use & (r["p_quasar"]>.5)
                    metrics["variants"][key] = dict(high_qso=int(high.sum()),
                        high_qso_after_guard=int((high&raw["eligible"]).sum()),
                        high_bf=int((use&(r["log_bf"]>v["log_bf_threshold"])).sum()),
                        max_log_r=float(np.nanmax(r["log_r"][use])) if use.any() else None)
                entry["low_density"][str(fraction)] = metrics
            report["grids"][name] = entry
        print(name,entry,flush=True)
    # Check the fast comparison against the complete guarded public scorer.
    phot = Photometry([[1.9,2.6,3.1],[1.9*16,2.6,3.1]],[[1/120,1/150,1/60]]*2,
        tuple(f"decals_dr9_south:{b}" for b in ("g","r","z")))
    check = dict(flux=phot.flux,variance=phot.variance,bands=phot.bands,
                 l=np.full(2,180.),b=np.full(2,45.),zprimary=np.full(2,v["primary_z"]))
    raw = raw_scores(base,check,cfg)
    for key,outlier in alternatives.items():
        variant = replace(base,outlier=outlier)
        scores,decision = variant.score(phot,morphology=["PSF"]*2,z_primary=v["primary_z"],
            l_deg=180.,b_deg=45.,match=RedshiftMatch(half_width_kms=v["window_kms"]),
            blend_policy=BlendPolicy(v["min_separation_arcsec"],v["max_fracflux"]),
            separation_arcsec=v["test_separation_arcsec"],fracflux=v["test_fracflux"],ood_flag_sigma=v["ood_flag_sigma"])
        f=base.model.transform(phot); a=f.labels.index("decals_dr9_south:r")
        c=outlier.conditional(a,f.labels[a],base.model.qso.system)
        expected=reweight(raw,c.log_prob(f.x,f.cov,observed=f.observed),c.fraction_at(f.x[:,a]))
        assert np.isclose(scores[0].log_r_per_unit_z,expected["log_r"][0],atol=1e-10)
        assert not decision["eligible"][1] and np.isnan(scores[1].log_r_per_unit_z)
    report["public_score_check"] = True
    def finite_json(value):
        if isinstance(value, dict):
            return {k:finite_json(v) for k,v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    (root/"validation_report.json").write_text(json.dumps(finite_json(report),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
