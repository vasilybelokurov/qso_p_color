#!/usr/bin/env python
"""Matched small-sky control: free native coordinates versus shared latent ones.

Same rows, initial native predictions, K, noise coordinates and iteration
budget. No QSO-support tuning and no production fitting. Rejection-free
scores below are density/ranking diagnostics, never promoted classifications.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import mannwhitneyu

from qso_pcolor import PSFMultiSurveyBaseline, Photometry
from qso_pcolor.full_sample import write_json, file_hash
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.joint_spatial import fit_joint_spatial_log_prob
from qso_pcolor.multisurvey import MultiSurveyModel, conditional_log_prob
from qso_pcolor.projected_xd import fit_projected
from qso_pcolor.qso_model import SlicedColourRedshiftModel
from qso_pcolor.sky_acquisition import acquisition_lock
from qso_pcolor.spatial import component_log_prob
from qso_pcolor.unified import native_view
from complete_unified_pilot import get_root
from run_unified_pilot import arrays, task_rows
from validate_unified_pilot import choose
from validate_full_sample_release import run_scores


def fit_control(task,cfg,root):
    root=Path(root);out=root/'native_control'/'fits'/(task['name']+'.json')
    if out.exists():return task['name']
    data=arrays(root,task['kind']);rows=task_rows(data,task,cfg)
    layout=json.loads((root/'layout.json').read_text())
    init=native_view(GaussianMixture.from_dict(task['init']),layout,task['kind'])
    d=init.n_dim
    def source():
        for lo in range(0,len(rows),cfg['batch_size']):
            rr=rows[lo:lo+cfg['batch_size']];cov=np.zeros((len(rr),d,d));cov[:,np.arange(d),np.arange(d)]=data['noise'][rr]
            yield data['y'][rr],cov,data['observed'][rr],np.zeros(len(rr),int)
    start=time.monotonic()
    def progress(iteration,mix,ll,n):
        write_json(root/'native_control'/'progress'/(task['name']+'.json'),dict(iteration=iteration,rows=n,mean_loglike=ll,complete=False))
    result=fit_projected(source,init=init,operators={0:(np.eye(d),np.zeros(d),np.zeros((d,d)))},
        expected_rows=len(rows),max_iter=cfg['max_iter'],tol=cfg['tol'],regularization=cfg['regularization'],progress=progress)
    write_json(out,dict(mixture=result.mixture.to_dict(),n=len(rows),n_iter=result.n_iter,history=result.history,
        mean_loglike=result.mean_loglike,converged=result.converged,elapsed_seconds=time.monotonic()-start))
    write_json(root/'native_control'/'progress'/(task['name']+'.json'),dict(iteration=result.n_iter,rows=len(rows),complete=True))
    return task['name']


def raw_rank(prediction):
    q=np.logaddexp(prediction['log_lambda_sameq'],prediction['log_lambda_fieldq'])
    return prediction['log_lambda_sameq']-logsumexp(np.stack([q,prediction['log_lambda_bkg'],prediction['log_lambda_out']]),axis=0)-np.log(prediction['dz_match_eff'])


def make_control(root,cfg,unified):
    target=root/'native_control'/'model.json'
    if target.exists():model=MultiSurveyModel.load(target)
    else:
        fitted=[json.loads((root/'native_control'/'fits'/f'qso_{j:02d}.json').read_text()) for j in range(43)]
        q=SlicedColourRedshiftModel(unified.model.qso.z_centres,[GaussianMixture.from_dict(f['mixture']) for f in fitted],
            [f['n'] for f in fitted],unified.model.qso.system,unified.model.transform.bands,dict(matched_control=True))
        b=GaussianMixture.from_dict(json.loads((root/'native_control'/'fits'/'stars_00.json').read_text())['mixture'])
        data=arrays(root,'stars');rows=np.flatnonzero(np.isin(data['role'],cfg['fit_roles']) & np.isin(data['cell'],cfg['training_cells']))
        lp=np.empty((len(rows),b.n_components))
        for lo in range(0,len(rows),512):
            rr=rows[lo:lo+512];f=unified.model.transform(Photometry(data['flux'][rr],data['variance'][rr],unified.model.transform.bands))
            lp[lo:lo+len(rr)]=component_log_prob(b,f.x,f.cov,f.observed)
        spatial=fit_joint_spatial_log_prob(b,lp,data['l'][rows],data['b'][rows],np.ones(len(rows)),**cfg['spatial'],meta=dict(matched_control=True,roles=['fit','select']))
        model=MultiSurveyModel(q,b,unified.model.transform,unified.model.reference_priority,unified.model.background_bounds,
            meta=dict(unified.model.meta,matched_control=True),spatial_background=spatial)
        model.save(target)
    # Identical nuisance terms deliberately isolate shape changes.
    return PSFMultiSurveyBaseline(model,unified.priors,unified.outlier,unified.manifest)


def main():
    root=get_root();cfg=json.loads((root/'config.json').read_text());start=time.monotonic()
    out=root/'native_control'
    for path in (out,out/'fits',out/'progress'):path.mkdir(exist_ok=True)
    tasks=json.loads((root/'tasks.json').read_text())
    settings=dict(score_rows_per_class_per_hemisphere=512,predictive_rows_per_class_per_hemisphere=2000,
        seed=cfg['seed']+218,maximum_auc_loss=.03,maximum_conditional_density_loss=.1,
        gate_policy='No rejection rules in density/ranking comparison; no support thresholds changed',
        control='Independent 41 native coordinates; same projected initial distributions, row sets, K and eight-iteration budget; no N/S tying')
    write_json(out/'comparison_settings.json',settings)
    with acquisition_lock(out/'fit.lock'):
        budget=sum(t['n']*t['k']*cfg['max_iter'] for t in tasks)
        with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
            pending={pool.submit(fit_control,t,cfg,str(root)):t for t in tasks if not (out/'fits'/(t['name']+'.json')).exists()}
            for done in as_completed(pending):
                print('CONTROL COMPLETE',done.result(),flush=True)
                work=complete=0
                for t in tasks:
                    p=out/'progress'/(t['name']+'.json');s=json.loads(p.read_text()) if p.exists() else dict(iteration=0,complete=False)
                    work+=t['n']*t['k']*(cfg['max_iter'] if s['complete'] else s['iteration']);complete+=s['complete']
                write_json(out/'progress.json',dict(complete=complete,total=len(tasks),fraction=work/budget,elapsed_seconds=time.monotonic()-start))
    unified=PSFMultiSurveyBaseline.load(root/'bundle');control=make_control(root,cfg,unified)
    bases={'unified':unified,'native_control':control};bands=unified.model.transform.bands
    data={k:arrays(root,k) for k in ('qso','stars')};report=dict(settings=settings,root=str(root),predictive={},ranking={},checks={},
        active_model_changed=False,rejection_rules_tuned=False,probability_calibration=False)
    masks={'all':np.ones(len(bands),bool),'legacy_optical':np.array([b.startswith('decals_') and b.split(':')[1] in ('g','r','z') for b in bands])}
    rcfg=json.loads(Path('configs/full_sample_release.json').read_text());rcfg['batch_size']=32
    for mode,mask in masks.items():
        for kind,d in data.items():
            chosen=choose(d,3,settings['predictive_rows_per_class_per_hemisphere'],bands,settings['seed'],unified.model.qso.support if kind=='qso' else None)
            for hemi,rows in chosen.items():
                v=np.array(d['variance'][rows]);v[:,~mask]=np.inf;phot=Photometry(d['flux'][rows],v,bands)
                keep=phot.observed.sum(axis=1)>=2;rows=rows[keep];phot=phot.subset(keep)
                anchor=unified.model.reference_indices(phot);aligned=phot.align(bands)
                snr=aligned.flux[np.arange(len(rows)),anchor]/np.sqrt(aligned.variance[np.arange(len(rows)),anchor])
                in_domain=snr>=cfg['reference_min_snr'];predictions={}
                for name,base in bases.items():
                    path=out/f'predictive_{mode}_{kind}_{hemi}_{name}.npz'
                    if path.exists():values=np.load(path)['logp']
                    else:
                        values=np.empty(len(rows))
                        for lo in range(0,len(rows),128):
                            ii=np.arange(lo,min(lo+128,len(rows)));f=base.model.transform(phot.subset(ii));aa=anchor[ii]
                            zz=np.argmin(abs(d['zspec'][rows[ii],None]-base.model.qso.z_centres),axis=1) if kind=='qso' else np.zeros(len(ii),int)
                            for j in np.unique(zz):
                                for a in np.unique(aa[zz==j]):
                                    take=(zz==j)&(aa==a)
                                    if kind=='qso':lp=conditional_log_prob(base.model.qso.mixtures[j],f.x[take],f.cov[take],f.observed[take],int(a))
                                    else:lp=base.model.background_log_prob(f.x[take],f.cov[take],f.observed[take],int(a),l_deg=d['l'][rows[ii[take]]],b_deg=d['b'][rows[ii[take]]])
                                    values[ii[take]]=lp
                        np.savez_compressed(path,rows=rows,logp=values,reference_snr=snr)
                    predictions[name]=values
                delta=predictions['unified']-predictions['native_control'];summary=dict(n=len(rows),n_reference_detected=int(in_domain.sum()),
                    mean_delta_all=float(delta.mean()),mean_delta_reference_detected=float(delta[in_domain].mean()),
                    median_delta_reference_detected=float(np.median(delta[in_domain])))
                key=f'{mode}_{kind}_{hemi}';report['predictive'][key]=summary
                report['checks']['density_'+key]=summary['mean_delta_reference_detected']>=-settings['maximum_conditional_density_loss']
                print('DENSITY',key,json.dumps(summary),flush=True)
        for hemi in ('south','north'):
            ranks={};zs=None
            for kind,d in data.items():
                rr=choose(d,3,settings['score_rows_per_class_per_hemisphere'],bands,settings['seed']+1,
                    unified.model.qso.support if kind=='qso' else None)[hemi]
                v=np.array(d['variance'][rr]);v[:,~mask]=np.inf;phot=Photometry(d['flux'][rr],v,bands);keep=phot.observed.sum(axis=1)>=2
                rr=rr[keep];phot=phot.subset(keep)
                if kind=='qso':zs=d['zspec'][rr]
                z=zs if kind=='qso' else np.resize(zs,len(rr))
                values=dict(flux=phot.flux,variance=phot.variance,bands=bands,l=d['l'][rr],b=d['b'][rr],zprimary=z)
                ranks[kind]={}
                for name,base in bases.items():
                    p=run_scores(base,values,rcfg,out/f'scores_{mode}_{kind}_{hemi}_{name}.npz');ranks[kind][name]=raw_rank(p)
            commonq=np.isfinite(ranks['qso']['unified'])&np.isfinite(ranks['qso']['native_control'])
            commonb=np.isfinite(ranks['stars']['unified'])&np.isfinite(ranks['stars']['native_control'])
            result={name:float(mannwhitneyu(ranks['qso'][name][commonq],ranks['stars'][name][commonb]).statistic/(commonq.sum()*commonb.sum())) for name in bases}
            result.update(n_qso=int(commonq.sum()),n_background=int(commonb.sum()));report['ranking'][mode+'_'+hemi]=result
            report['checks']['ranking_'+mode+'_'+hemi]=result['unified']>=result['native_control']-settings['maximum_auc_loss']
            print('RANKING',mode,hemi,json.dumps(result),flush=True)
        write_json(out/'report.json',report)
    report['checks_pass']=all(report['checks'].values());report['elapsed_seconds']=time.monotonic()-start
    report['implementation_sha256']=file_hash(Path(__file__))
    write_json(out/'report.json',report);write_json(Path('docs/UNIFIED_DENSITY_CONTROL_2026-09-30.json'),report)
    print('DONE',json.dumps(report['checks']),flush=True)


if __name__=='__main__':main()
