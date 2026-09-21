#!/usr/bin/env python
"""Evaluate every survey subset on reserved quasars and reserved field sources."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import rankdata

from qso_pcolor.multisurvey import MultiSurveyModel, _ConditionalQSO, conditional_log_prob
from qso_pcolor.multisurvey_data import Photometry, survey_of
from qso_pcolor.plotting import save_figure
from qso_pcolor.qso_model import RedshiftMatch


def auc(positive, negative):
    """Mann--Whitney AUC, including half weight for tied scores."""
    if not len(positive) or not len(negative):
        return None
    n=len(positive);m=len(negative)
    ranks=rankdata(np.concatenate([positive,negative]))
    return float((ranks[:n].sum()-n*(n+1)/2)/(n*m))


def evaluate(model, phot, z, grid, levels, alternative_z=None):
    """Conditional colour densities and flat-z credible intervals, vectorised."""
    features=model.transform(phot)
    n=features.n_obs
    order=np.array([features.labels.index(b) for b in model.reference_priority])
    anchors=order[np.argmax(features.observed[:,order],axis=1)]
    logbf=np.empty(n);mode=np.empty(n);true_cdf=np.empty(n);mean_lp=np.empty(n)
    log_pz=np.empty(n);alternative_log_pz=np.empty(n)
    for a in np.unique(anchors):
        sel=np.flatnonzero(anchors==a);f=features.subset(sel)
        q=_ConditionalQSO(model.qso,int(a))
        slices=q._log_p_slices(f.x,f.cov,f.observed)
        lp=q.log_p_colour_given_z(f.x,f.cov,grid,_log_slices=slices)
        post=np.exp(lp-lp.max(1,keepdims=True))
        norm=np.trapezoid(post,grid,axis=1)
        cdf=cumulative_trapezoid(post,grid,axis=1,initial=0.)/norm[:,None]
        # Each row's likelihood at its own spectroscopic/assigned redshift.
        at=q.log_p_colour_given_z(f.x,f.cov,z[sel],_log_slices=slices).diagonal()
        pb=conditional_log_prob(model.background,f.x,f.cov,f.observed,int(a))
        logbf[sel]=at-pb;mode[sel]=grid[np.argmax(post,axis=1)]
        mean_lp[sel]=at
        lognorm=lp.max(1)+np.log(norm)
        log_pz[sel]=at-lognorm
        if alternative_z is not None:
            alternative_log_pz[sel]=q.log_p_colour_given_z(
                f.x,f.cov,alternative_z[sel],_log_slices=slices).diagonal()-lognorm
        true_cdf[sel]=[np.interp(zz,grid,c) for zz,c in zip(z[sel],cdf)]
    return dict(log_bf=logbf,z_mode=mode,log_p_at_z=mean_lp,
                log_redshift_density=log_pz,
                alternative_log_redshift_density=alternative_log_pz if alternative_z is not None else None,
                coverage={str(level):float(np.mean((true_cdf>=(1-level)/2)&
                                                   (true_cdf<=(1+level)/2))) for level in levels})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",type=Path,default=Path("configs/multisurvey.json"))
    ap.add_argument("--validation-config",type=Path,default=Path("configs/multisurvey_validation.json"))
    args=ap.parse_args();cfg=json.loads(args.config.read_text());vc=json.loads(args.validation_config.read_text())
    root=Path(cfg["data_dir"]);model=MultiSurveyModel.load(cfg["model_path"])
    if json.loads((root/"config.json").read_text()) != cfg or model.meta["settings"]["config"] != cfg:
        raise ValueError("validation sample, fit, and requested configuration differ")
    q,b=[dict(np.load(root/f"{name}.npz")) for name in ("quasars","background")]
    n_background_original = len(b["ra"])
    n_background_original_reserved = int(b["held"].sum())
    extra_fields = vc.get("extra_background_fields", [])
    n_background_extra = 0
    if extra_fields:
        extra_path = root / "validation_background.npz"
        selection = json.loads(extra_path.with_suffix(".json").read_text())
        if (selection["fields"] != extra_fields or selection["training_config"] != cfg):
            raise ValueError("additional validation field selection differs from its cache")
        extra = dict(np.load(extra_path))
        if not np.array_equal(extra["bands"], b["bands"]) or not extra["held"].all():
            raise ValueError("extra validation data must use the same bands and all be held out")
        n_background_extra = len(extra["ra"])
        b = {k: b[k] if k == "bands" else np.concatenate([b[k], extra[k]]) for k in b}
    qp,bp=[Photometry(r["flux"],r["variance"],tuple(r["bands"])) for r in (q,b)]
    rng=np.random.default_rng(vc["seed"])
    grid=np.unique(np.concatenate([np.linspace(*model.qso.support,vc["redshift_grid_points"]),model.qso.z_centres]))
    report=[];checks=0
    for size in range(1,len(cfg["surveys"])+1):
        for surveys in combinations(cfg["surveys"],size):
            photos=[p.keep_surveys(surveys) for p in (qp,bp)]
            rows=[];counts=[]
            for r,p,cap in zip((q,b),photos,(vc["max_quasars_per_combination"],vc["max_background_per_combination"])):
                ok=r["held"].copy() & (p.observed.sum(1)>=cfg["min_bands"])
                for s in surveys:
                    idx=[i for i,label in enumerate(p.bands) if survey_of(label)==s]
                    ok &= p.observed[:,idx].sum(1)>=vc["minimum_bands_per_requested_survey"]
                if "zspec" in r:
                    ok &= model.qso.in_support(r["zspec"])
                available=np.flatnonzero(ok);counts.append(len(available))
                rows.append(np.sort(rng.choice(available,min(cap,len(available)),replace=False)))
            qi,bi=rows
            record=dict(surveys=list(surveys),n_available_qso=counts[0],n_available_background=counts[1],
                        n_evaluated_qso=len(qi),n_evaluated_background=len(bi))
            if len(qi):
                true_z=q["zspec"][qi]
                # Real withheld quasars also test the competing field-Q case.
                # Draw primary redshifts from the same held-out population,
                # rejecting only the declared same-redshift window.
                pool=q["zspec"][q["held"] & model.qso.in_support(q["zspec"])]
                alternative_z=rng.choice(pool,len(qi))
                match=RedshiftMatch(half_width_kms=vc["half_width_kms"])
                for _ in range(1000):
                    inside=abs(alternative_z-true_z)<=match.half_width_kms/299792.458*(1+alternative_z)
                    if not inside.any():
                        break
                    alternative_z[inside]=rng.choice(pool,int(inside.sum()))
                else:
                    raise ValueError("cannot draw field-quasar redshifts outside the declared window")
                qres=evaluate(model,photos[0].subset(qi),true_z,grid,vc["credible_levels"],alternative_z)
                record.update(coverage=qres["coverage"],
                    median_abs_dz_over_1pz=float(np.median(abs(qres["z_mode"]-q["zspec"][qi])/(1+q["zspec"][qi]))),
                    mean_conditional_log_density=float(np.mean(qres["log_p_at_z"])),
                    auc_true_vs_other_redshift_by_conditional_density=auc(
                        qres["log_redshift_density"],qres["alternative_log_redshift_density"]))
                # Check the public scorer, including its supported-window path.
                i=qi[0]
                row=model.score(photos[0].subset([i]),z_primary=q["zspec"][[i]],
                    l_deg=q["l"][[i]],b_deg=q["b"][[i]],min_bands=cfg["min_bands"],
                    match=RedshiftMatch(half_width_kms=vc["half_width_kms"]))[0]
                if not np.isclose(row.log_bayes_factor_qz_bkg,qres["log_bf"][0],atol=1e-9):
                    raise AssertionError("public scorer and independent batched evaluation disagree")
                if not (0<=row.p_zmatch_given_qso<=1):
                    raise AssertionError("invalid conditional redshift probability")
                checks+=1
                if len(bi):
                    assigned_z=rng.choice(q["zspec"][qi],len(bi),replace=True)
                    bres=evaluate(model,photos[1].subset(bi),assigned_z,grid,vc["credible_levels"])
                    record["auc_qso_vs_field_by_colour_bf"]=auc(qres["log_bf"],bres["log_bf"])
                else:
                    record["auc_qso_vs_field_by_colour_bf"]=None
            report.append(record)
            print('+'.join(surveys),json.dumps(record),flush=True)
    slices=model.qso.meta["per_slice"]
    fit_summary=dict(n_quasars_selected=len(q["ra"]),n_quasars_fit=model.qso.meta["n_fit"],
                     n_quasars_reserved=model.qso.meta["n_holdout"],
                     n_background_total=n_background_original,
                     n_background_reserved=n_background_original_reserved,
                     n_background_extra_validation=n_background_extra,
                     background=model.meta["background"],
                     quasar_component_counts={str(k):sum(s["selected_k"]==k for s in slices)
                                              for k in sorted({s["selected_k"] for s in slices})},
                     quasar_slices_converged=sum(s["converged"] for s in slices),
                     quasar_slices_total=len(slices),
                     median_final_iteration_change=float(np.median([
                         abs(s["convergence_tail"][-1]-s["convergence_tail"][-2]) for s in slices])),
                     quasar_per_slice=slices)
    result=dict(model=cfg["model_path"],run_id=model.meta["run_id"],validation_config=vc,
                training=fit_summary,
                public_scorer_checks=checks,combinations=report,
                qso_heldout_by_band={label:int((qp.observed[:,j]&q["held"]).sum()) for j,label in enumerate(qp.bands)},
                background_heldout_by_band={label:int((bp.observed[:,j]&b["held"]).sum()) for j,label in enumerate(bp.bands)},
                interpretation="High-latitude availability-selected subsets; at most 200 objects per class and combination. Different survey rows use different available objects. Redshift PDFs use a flat z prior; coverage is a diagnostic, not a calibration claim. Population posteriors require separate matched surface-density priors.")
    (root/"validation.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    Path('docs/MULTISURVEY_VALIDATION.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    selected=[r for r in report if len(r["surveys"])==1 or set(r["surveys"]) in
              ({"sdss","allwise"},{"decals","allwise"},{"ps1","allwise"},
               {"sdss","ps1","allwise"},{"allwise","vhs"},set(cfg["surveys"]))]
    lines=["# Multi-survey model validation", "",f"Run `{model.meta['run_id']}`; model `{cfg['model_path']}`.","",
           result['interpretation'],"",
           f"The quasar fit uses {fit_summary['n_quasars_fit']:,} objects, with {fit_summary['n_quasars_reserved']:,} reserved in the existing spatial holdout. It spans {len(slices)} slices over model support {model.qso.support[0]:.2f}–{model.qso.support[1]:.2f}; the selected component counts (K: number of slices) are `{fit_summary['quasar_component_counts']}`.","",
           f"The original field sample has {n_background_original:,} sources from {len(model.meta['fields'])} fields, with {n_background_original_reserved:,} sources in reserved fields. The background fit uses {model.meta['background']['n_fit']:,} sampled rows with weights restoring the training population and K={model.background.n_components}.","",
           f"An additional {len(extra_fields)} overlap field contributes {n_background_extra:,} exclusively held-out sources. It was selected for survey coverage before inspecting validation scores and is absent from all fitting and component selection. Its coordinates and selection rule are recorded in the validation configuration.","",
           f"The requested EM tolerance was reached by {fit_summary['quasar_slices_converged']}/{len(slices)} quasar slices; background convergence: {model.meta['background']['converged']}. Fits that reached the iteration cap are recorded as such, not declared converged. The held-out results below assess the saved fits at that stopping point.","",
           f"The median absolute change in mean training log density over the final quasar-fit iteration was {fit_summary['median_final_iteration_change']:.5f} nats per object.","",
           f"The public scorer was checked against a separate batched calculation for {checks} real-data combinations. The unit tests exercise all 127 non-empty survey subsets, including two-band infrared-only inputs.","",
           "| Surveys | Held-out QSOs available | QSOs / field objects evaluated | QSO vs field AUC | True vs other z AUC | Median absolute dz/(1+z) | 68% interval coverage |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in selected:
        def fmt(v):return '—' if v is None else f'{v:.3f}'
        lines.append(f"| {' + '.join(r['surveys'])} | {r['n_available_qso']} | {r['n_evaluated_qso']} / {r['n_evaluated_background']} | {fmt(r.get('auc_qso_vs_field_by_colour_bf'))} | {fmt(r.get('auc_true_vs_other_redshift_by_conditional_density'))} | {fmt(r.get('median_abs_dz_over_1pz'))} | {fmt(r.get('coverage',{}).get('0.68'))} |")
    singles_by_name = {r['surveys'][0]:r for r in report if len(r['surveys'])==1}
    wise_auc = singles_by_name['allwise']['auc_qso_vs_field_by_colour_bf']
    vhs_auc = singles_by_name['vhs']['auc_qso_vs_field_by_colour_bf']
    lines += ["", f"Infrared-only inputs receive scores, but their information content differs. On the available held-out samples, ALLWISE-only gives QSO/field AUC {wise_auc:.3f}; VHS-only gives {vhs_auc:.3f}, so VHS-only classification is weak. These rows use different objects and do not measure the improvement from adding a survey to the same sources.","",
              "The QSO-versus-field AUC uses colour evidence at the true redshift for quasars and assigned primary redshifts for field objects. The true-versus-other-redshift AUC scores each withheld quasar at its true redshift and at a primary redshift drawn from the withheld population outside the declared matching window, using the normalised quasar-only redshift density. These are discrimination checks, not physical-pair probabilities or substitutes for the field-quasar term in scoring.","",
              "The data use native observed photometry, with quality cuts and high-latitude selection recorded in `configs/multisurvey.json`. The background contains all source types and excludes known quasars. Whole fields were reserved before fitting; internal component selection used additional fields from the training partition.","",
              "The [full machine-readable report](MULTISURVEY_VALIDATION.json) includes all 127 combinations, per-band holdout counts, and fit diagnostics. A local copy is also kept in `data/multisurvey/validation.json`. The existing southern models were retained."]
    Path('docs/MULTISURVEY_VALIDATION.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    singles=[r for r in report if len(r['surveys'])==1]
    fig,ax=plt.subplots(figsize=(9,4))
    ax.bar([r['surveys'][0] for r in singles],
           [r.get('auc_qso_vs_field_by_colour_bf') or 0 for r in singles])
    ax.set_ylim(0,1);ax.set_ylabel('QSO versus field AUC (colour BF)')
    ax.set_title('Reserved objects; each survey uses its available subset')
    fig.tight_layout();save_figure(fig,'validation/multisurvey_auc');plt.close(fig)


if __name__=='__main__':
    main()
