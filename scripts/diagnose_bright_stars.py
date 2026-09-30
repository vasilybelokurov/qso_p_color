#!/usr/bin/env python
"""Bounded calibration-role diagnosis of bright stellar optical evidence.

This is an evidence diagnostic, not a posterior calibration or release audit.
The stellar-background sample may contain unrecognized quasars. A high-evidence
row is therefore not automatically a labelled false positive.
"""
import json
from pathlib import Path
import numpy as np
from qso_pcolor import PSFMultiSurveyBaseline, Photometry
from qso_pcolor.multisurvey import _ConditionalQSO, _ConditionalBackground
from qso_pcolor.full_population import population_rows
from qso_pcolor.full_sample import write_json, file_hash
from compare_full_sample_health import paired_qso, summarize


def main():
    cfg=json.loads(Path('configs/full_sample_completion.json').read_text());diag=cfg['diagnostic']
    outroot=Path(cfg['output_root']);candidate=outroot/json.loads((outroot/'candidate.json').read_text())['bundle']
    old=PSFMultiSurveyBaseline.load(cfg['source_bundle']);new=PSFMultiSurveyBaseline.load(candidate)
    data={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (Path(cfg['input_root'])/'stars').glob('*.npy')}
    eligible=population_rows(data,(diag['role'],))
    rows=np.sort(np.random.default_rng(cfg['seed']).choice(eligible,min(len(eligible),diag['maximum_rows']),replace=False))
    bands=new.model.transform.bands;order=np.array([bands.index(b) for b in new.model.reference_priority])
    legacy=np.array([b.startswith('decals_') and b.split(':')[1] in ('g','r','z') for b in bands])
    sdss=np.array([b.startswith('sdss:') for b in bands]);ps1=np.array([b.startswith('ps1:') for b in bands])
    optical=legacy|sdss|ps1
    # Brightness is defined once with the combined-optical anchor and then fixed
    # across survey ablations; switching references never changes the sample.
    bright=np.zeros(len(rows),bool);ref=np.empty(len(rows),int)
    for lo in range(0,len(rows),cfg['density_batch_size']):
        rr=rows[lo:lo+cfg['density_batch_size']];v=np.array(data['variance'][rr]);v[:,~optical]=np.inf
        f=new.model.transform(Photometry(data['flux'][rr],v,bands));anchors=order[np.argmax(f.observed[:,order],axis=1)]
        bright[lo:lo+len(rr)]=(f.x[np.arange(len(rr)),anchors]<diag['bright_reference_luptitude']) & (f.observed.sum(axis=1)>=2)
        ref[lo:lo+len(rr)]=anchors
    rows=rows[bright];ref=ref[bright]
    modes=dict(optical=optical,legacy_optical=legacy,sdss_only=sdss,ps1_only=ps1,legacy_sdss=legacy|sdss,legacy_ps1=legacy|ps1,all_available=np.ones(len(bands),bool))
    report=dict(configuration=diag,seed=cfg['seed'],sampled_calibration_rows=min(len(eligible),diag['maximum_rows']),bright_rows=len(rows),
                candidate=str(candidate),source=cfg['source_bundle'],candidate_manifest_sha256=file_hash(candidate/'manifest.json'),
                caveats=['Calibration-role diagnosis; these rows also enter final catch-all share fitting.',
                         'High QSO evidence is not a confirmed false-positive label.',
                         'Maximum over five fixed redshifts; not an integrated posterior or complete-score validation.'],modes={})
    save=dict(rows=rows,field=data['field'][rows],reference=ref)
    for name,mask in modes.items():
        # model, object, metric: stellar log density, raw max log BF,
        # catch-all-adjusted max log BF, hard-guard exclusion of high-BF rows.
        vals=np.full((2,len(rows),4),np.nan);use=np.zeros(len(rows),bool)
        for lo in range(0,len(rows),cfg['density_batch_size']):
            rr=rows[lo:lo+cfg['density_batch_size']];v=np.array(data['variance'][rr]);v[:,~mask]=np.inf
            f=new.model.transform(Photometry(data['flux'][rr],v,bands));anchors=order[np.argmax(f.observed[:,order],axis=1)]
            use[lo:lo+len(rr)]=f.observed.sum(axis=1)>=2
            for a in np.unique(anchors):
                local=np.flatnonzero((anchors==a)&(f.observed.sum(axis=1)>=2))
                if not len(local):continue
                dest=lo+local;x,c,o=f.x[local],f.cov[local],f.observed[local]
                for m,baseline in enumerate((old,new)):
                    model=baseline.model
                    bg=_ConditionalBackground(model.background,model.qso.system,int(a),model.background_bounds,model._field_marginals,model.spatial_background)
                    lb=model.background_log_prob(x,c,o,int(a),l_deg=data['l'][rr[local]],b_deg=data['b'][rr[local]])
                    out=baseline.outlier.conditional(int(a),bands[a],model.qso.system)
                    lu=out.log_prob(x,c,observed=o);eta=out.fraction_at(x[:,a])
                    lbout=np.logaddexp(np.log1p(-eta)+lb,np.log(eta)+lu)
                    lq=np.column_stack([paired_qso(model.qso,x,c,o,int(a),np.full(len(x),z)) for z in diag['primary_redshifts']]).max(axis=1)
                    vals[m,dest,0]=lb;vals[m,dest,1]=lq-lb;vals[m,dest,2]=lq-lbout
                    vals[m,dest,3]=0
                    high=(lq-lbout)>diag['log_bf_threshold']
                    # Only potential high-evidence rows need the expensive guard
                    # calculation for the reported post-guard tail count.
                    q=_ConditionalQSO(model.qso,int(a))
                    for start in range(0,int(high.sum()),32):
                        ii=np.flatnonzero(high)[start:start+32]
                        dq=q.ood_score_any_z(x[ii],c[ii],observed=o[ii])
                        db=bg.ood_score(x[ii],c[ii],x[ii,a],data['l'][rr[local[ii]]],data['b'][rr[local[ii]]],observed=o[ii])
                        vals[m,dest[ii],3]=(np.minimum(dq,db)>diag['ood_flag_sigma'])
        groups=data['field'][rows]
        result=summarize(vals[0,use,0],vals[1,use,0],groups[use]);result['strata']={}
        for label,sel in [('north',data['dec'][rows]>=32),('south',data['dec'][rows]<32),*[(bands[a],ref==a) for a in np.unique(ref)]]:
            s=use&sel
            if s.any():result['strata'][label]=summarize(vals[0,s,0],vals[1,s,0],groups[s])
        result['evidence']={}
        for m,label in enumerate(('small','full')):
            x=vals[m,use]
            if not len(x):continue
            result['evidence'][label]=dict(raw_above_threshold=int((x[:,1]>diag['log_bf_threshold']).sum()),
                    catchall_above_threshold=int((x[:,2]>diag['log_bf_threshold']).sum()),
                    after_support_guard=int(((x[:,2]>diag['log_bf_threshold'])&(x[:,3]==0)).sum()),
                    raw_p99=float(np.quantile(x[:,1],diag['tail_quantile'])),catchall_p99=float(np.quantile(x[:,2],diag['tail_quantile'])))
        newly=use&(vals[1,:,2]>diag['log_bf_threshold'])&(vals[1,:,3]==0)&~((vals[0,:,2]>diag['log_bf_threshold'])&(vals[0,:,3]==0))
        result['new_high_evidence_rows']=rows[newly].tolist()
        report['modes'][name]=result;save[name]=vals;save[name+'_used']=use
        print(name,'n',result['n'],'gain',result.get('mean_gain'),'evidence',result['evidence'],flush=True)
        write_json(candidate/'bright_star_diagnostic.json',report)
    np.savez_compressed(candidate/'bright_star_predictions.npz',**save)
    write_json(Path('docs/BRIGHT_STAR_DIAGNOSTIC_2026-09-30.json'),report)


if __name__=='__main__':main()
