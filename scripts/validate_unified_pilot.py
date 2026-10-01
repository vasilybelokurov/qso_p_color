#!/usr/bin/env python
"""Calibrate QSO support and run a bounded complete-score unified pilot audit."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from qso_pcolor import PSFMultiSurveyBaseline, Photometry, RedshiftMatch, BlendPolicy
from qso_pcolor.full_sample import file_hash, write_json
from qso_pcolor.legacy import is_north
from qso_pcolor.unified import UnifiedPSFModel, qso_support
from run_unified_pilot import arrays
from complete_unified_pilot import get_root
from validate_full_sample_release import FIELDS, run_scores, metrics, auc
from validate_psf_catchalls import grid_data


def mask_phot(data,rows,bands,mask):
    variance=np.array(data['variance'][rows]);variance[:,~mask]=np.inf
    return Photometry(data['flux'][rows],variance,bands)


def choose(data,role,n,bands,seed,z_range=None):
    rng=np.random.default_rng(seed);p=Photometry(data['flux'],data['variance'],bands)
    with np.errstate(invalid='ignore',divide='ignore'):
        detected=(p.observed&(p.flux/np.sqrt(p.variance)>=5)).any(axis=1)
    keep=(data['role']==role)&detected&(p.observed.sum(axis=1)>=2)
    if z_range is not None:keep&=(data['zspec']>=z_range[0])&(data['zspec']<=z_range[1])
    north=is_north(data['ra'],data['dec'],data['b']);rows={}
    for h,value in [('south',False),('north',True)]:
        ii=np.flatnonzero(keep&(north==value))
        rows[h]=rng.choice(ii,min(n,len(ii)),replace=False)
    return rows


def support_file(base,phot,z,cfg,path,l=None,b=None):
    if path.exists():return dict(np.load(path))
    s=qso_support(base.model,phot,z,draws=cfg['validation']['support_draws'],seed=cfg['seed'],l_deg=l,b_deg=b)
    np.savez_compressed(path,**s);return s


def calibration(root,base,data,cfg,masks):
    out=root/'bundle';path=out/'support.json'
    if path.exists():return json.loads(path.read_text())
    chosen=choose(data,2,cfg['validation']['calibration_qsos_per_hemisphere'],base.model.transform.bands,cfg['seed'],base.model.qso.support)
    rows=np.concatenate(list(chosen.values()));np.save(root/'support_calibration_rows.npy',rows)
    values=[];counts={}
    for name in ('all','legacy_optical','sdss','ps1'):
        phot=mask_phot(data,rows,base.model.transform.bands,masks[name]);use=phot.observed.sum(axis=1)>=2
        s=support_file(base,phot.subset(use),data['zspec'][rows[use]],cfg,root/f'support_calibration_{name}.npz',
                       l=data['l'][rows[use]],b=data['b'][rows[use]])
        finite=np.isfinite(s['percentile']);values.extend(s['percentile'][finite]);counts[name]=int(finite.sum())
        print('SUPPORT CALIBRATION',name,counts[name],flush=True)
    values=np.array(values);target=cfg['validation']['support_retention']
    threshold=float(np.sort(values)[int(np.floor((1-target)*len(values)))])
    record=dict(threshold=threshold,draws=cfg['validation']['support_draws'],seed=cfg['seed'],
        roles=['calib'],target_retention=target,calibration_retention=float(np.mean(values>=threshold)),
        calibration_rows=len(rows),calibration_evaluations=len(values),by_mask=counts,
        row_indices_sha256=hashlib.sha256(data['source_row'][rows].tobytes()).hexdigest(),
        informative_threshold=bool(threshold>1/(cfg['validation']['support_draws']+1)),
        interpretation='conditional density percentile at target redshift; not a class probability',
        scope='Pooled pilot cutoff calibrated on all/Legacy-optical/SDSS/PS1 masks; other masks exploratory')
    write_json(path,record);return record


def run_unified(model,phot,ra,dec,z,cfg,path):
    if path.exists():return dict(np.load(path))
    records=[];eligible=[];supports=[];rejected=[];raw=[]
    for lo in range(0,len(phot.flux),cfg['batch_size']):
        ii=np.arange(lo,min(lo+cfg['batch_size'],len(phot.flux)))
        scores,decision=model.score(phot.subset(ii),ra_deg=ra[ii],dec_deg=dec[ii],z_primary=z[ii],
            morphology=['PSF']*len(ii),match=RedshiftMatch(half_width_kms=cfg['window_kms']),
            blend_policy=BlendPolicy(**cfg['blend_policy']),ood_flag_sigma=cfg['ood_flag_sigma'],
            separation_arcsec=cfg['fixture_separation_arcsec'],fracflux=cfg['fixture_fracflux'])
        records.extend(scores);eligible.extend(decision['eligible']);supports.extend(decision['qso_support']['percentile']);rejected.extend(decision['support_rejected'])
    r={k:np.array([getattr(s,k) for s in records]) for k in FIELDS}
    r.update(eligible=np.array(eligible,bool),status=np.array([s.status for s in records]),
        flags=np.array([','.join(s.quality_flags) for s in records]),reference=np.array([s.reference_band for s in records]),
        support=np.array(supports),support_rejected=np.array(rejected,bool))
    q=np.logaddexp(r['log_lambda_sameq'],r['log_lambda_fieldq'])
    r['p_quasar']=np.exp(q-logsumexp(np.stack([q,r['log_lambda_bkg'],r['log_lambda_out']]),axis=0))
    np.savez_compressed(path,**r);return r


def main():
    root=get_root();cfg=json.loads((root/'config.json').read_text());vcfg=cfg['validation'];started=time.monotonic()
    base=PSFMultiSurveyBaseline.load(root/'bundle');parent=PSFMultiSurveyBaseline.load(cfg['parent'])
    # Both compare using the same reference-choice rule; the active saved bundle is untouched.
    parent.model.meta['reference_min_snr']=cfg['reference_min_snr']
    rcfg=json.loads(Path('configs/full_sample_release.json').read_text());rcfg['batch_size']=vcfg['score_batch_size']
    bands=base.model.transform.bands
    masks=dict(all=np.ones(len(bands),bool),legacy_optical=np.array([b.startswith('decals_') and b.split(':')[1] in ('g','r','z') for b in bands]),
        sdss=np.array([b.startswith('sdss:') for b in bands]),ps1=np.array([b.startswith('ps1:') for b in bands]))
    data={kind:arrays(root,kind) for kind in ('qso','stars')}
    policy=calibration(root,base,data['qso'],cfg,masks)
    manifest=json.loads((root/'bundle'/'manifest.json').read_text())
    manifest['unified_files']={name:file_hash(root/'bundle'/name) for name in ('latent.json','support.json')}
    manifest['status']='small-sky engineering pilot; not active, not full probability calibration'
    write_json(root/'bundle'/'manifest.json',manifest)
    unified=UnifiedPSFModel.load(root/'bundle')
    chosen={kind:choose(d,3,vcfg['rows_per_class_per_hemisphere'],bands,cfg['seed']+j+1,
        base.model.qso.support if kind=='qso' else None) for j,(kind,d) in enumerate(data.items())}
    np.savez(root/'score_rows.npz',**{kind+'_'+h:r for kind,halves in chosen.items() for h,r in halves.items()})
    report=dict(root=str(root),config=cfg,support_policy=policy,real={},subsets={},grids={},paired={},checks={},
        probability_calibration=False,active_model_changed=False,reference_policy_control='same >=5-sigma reference preference in both alternatives',
        limits=['Eight-iteration warm-start small-sky refit; not fully converged production training',
                'QSO abundance inherited; external survey areas approximate',
                'NSC u/VR and VHS H have no pilot measurements; retained inherited coordinates are not empirically certified',
                'Synthetic clean-blend fixtures isolate photometric effects',
                'Test-role data previously inspected during development; not new untouched confirmation'])
    for mode in ('all','legacy_optical','sdss','ps1'):
        for hemi in ('south','north'):
            predictions={};summary={}
            for kind in ('qso','stars'):
                d=data[kind];rr=chosen[kind][hemi];phot=mask_phot(d,rr,bands,masks[mode]);keep=phot.observed.sum(axis=1)>=2
                rr=rr[keep];phot=phot.subset(keep)
                z=d['zspec'][rr] if kind=='qso' else np.resize(data['qso']['zspec'][chosen['qso'][hemi]],len(rr))
                values=dict(flux=phot.flux,variance=phot.variance,bands=bands,l=d['l'][rr],b=d['b'][rr],zprimary=z)
                old=run_scores(parent,values,rcfg,root/f'{mode}_{hemi}_{kind}_parent.npz')
                pre=run_scores(base,values,rcfg,root/f'{mode}_{hemi}_{kind}_raw.npz')
                new=run_unified(unified,phot,d['ra'][rr],d['dec'][rr],z,rcfg,root/f'{mode}_{hemi}_{kind}_unified.npz')
                predictions[kind]=(old,pre,new)
                summary[kind]=dict(parent=metrics(old,rcfg),raw=metrics(pre,rcfg),unified=metrics(new,rcfg),
                    support_retention=float(np.mean(~new['support_rejected'])) if len(rr) else None,
                    support_rejections=int(new['support_rejected'].sum()),rows=len(rr))
                if kind=='qso' and mode=='all':
                    lo,hi=base.model.qso.support;wrong=lo+np.mod(z-lo+rcfg['wrong_redshift_shift'],hi-lo)
                    other=run_scores(base,dict(values,zprimary=wrong),rcfg,root/f'{mode}_{hemi}_wrongz.npz')
                    ok=pre['eligible']&other['eligible']
                    summary[kind]['true_z_rank_higher_fraction']=float(np.mean(pre['log_r_per_unit_z'][ok]>other['log_r_per_unit_z'][ok])) if ok.any() else None
            summary['auc']={name:auc(predictions['qso'][i],predictions['stars'][i]) for i,name in enumerate(('parent','raw','unified'))}
            report['real'][mode+'_'+hemi]=summary
            report['checks']['ranking_'+mode+'_'+hemi]=summary['auc']['unified']>=summary['auc']['parent']-vcfg['maximum_auc_loss']
            report['checks']['support_retention_'+mode+'_'+hemi]=summary['qso']['support_retention']>=vcfg['minimum_test_support_retention']
            print('REAL',mode,hemi,json.dumps(dict(auc=summary['auc'],qso_support_retention=summary['qso']['support_retention'])),flush=True)
            write_json(root/'validation_progress.json',report)
    # Every single band plus deterministic sparse masks exercises the interface.
    rng=np.random.default_rng(cfg['seed']);d=data['qso'];rr=np.r_[chosen['qso']['south'][:vcfg['subset_rows']],chosen['qso']['north'][:vcfg['subset_rows']]]
    tests=[('single_'+b,np.arange(len(bands))==j) for j,b in enumerate(bands)]
    tests += [(f'random_{i}',rng.uniform(size=len(bands))<.25) for i in range(6)]
    for name,mask in tests:
        phot=mask_phot(d,rr,bands,mask);keep=phot.observed.any(axis=1)
        if not keep.any():report['subsets'][name]=dict(n=0,status='no empirical pilot measurements');continue
        r=run_unified(unified,phot.subset(keep),d['ra'][rr[keep]],d['dec'][rr[keep]],d['zspec'][rr[keep]],rcfg,root/f'subset_{name.replace(":","_")}.npz')
        report['subsets'][name]=metrics(r,rcfg)
    report['checks']['subset_numerics']=all(all(v.get(k,True) for k in ('finite_eligible','probabilities_bounded','window_identity','hard_guard')) for v in report['subsets'].values())
    # Original northern tail and the four remaining southern intermediate probes.
    for hemi in ('south','north'):
        for mag in rcfg['grid']['reference_magnitudes'][:2]:
            g=grid_data(base.model,hemi,mag,{'validation':rcfg['grid']})
            phot=Photometry(g['flux'],g['variance'],g['bands'])
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            sky=SkyCoord(l=g['l']*u.deg,b=g['b']*u.deg,frame='galactic').icrs
            old=run_scores(parent,g,rcfg,root/f'grid_{hemi}_{mag}_parent.npz')
            pre=run_scores(base,g,rcfg,root/f'grid_{hemi}_{mag}_raw.npz')
            new=run_unified(unified,phot,sky.ra.deg,sky.dec.deg,g['zprimary'],rcfg,root/f'grid_{hemi}_{mag}_unified.npz')
            old_cache=Path(json.loads(Path('docs/FULL_SAMPLE_RELEASE_2026-09-30.json').read_text())['cache'])
            low=np.ones(len(phot.flux),bool)
            for label in ('baseline','candidate'):
                saved=np.load(old_cache/f'grid_{hemi}_{mag}_{label}.npz');q=np.logaddexp(saved['log_lambda_sameq'],saved['log_lambda_fieldq']);b=saved['log_lambda_bkg']
                low&=(q<np.nanmax(q)+np.log(.01))&(b<np.nanmax(b)+np.log(.01))
            report['grids'][f'{hemi}_{mag}']={name:dict(**metrics(r,rcfg),high_in_original_low_density_mask=int((low&r['eligible']&(r['p_quasar']>.5)).sum())) for name,r in [('parent',old),('raw',pre),('unified',new)]}
            print('GRID',hemi,mag,report['grids'][f'{hemi}_{mag}']['unified']['high_in_original_low_density_mask'],flush=True)
    # Same-object paired native views, preserving the actual catalogue system.
    overlap=dict(np.load(cfg['overlap']));reserved=np.isin(overlap['role'],['calib','test'])&overlap['observed'][:,:,:3].all(axis=(1,2))
    pairs=np.flatnonzero(reserved);pairs=rng.choice(pairs,min(vcfg['paired_rows'],len(pairs)),replace=False);np.save(root/'paired_rows.npy',pairs)
    views={}
    for system,hemi in enumerate(('north','south')):
        labels=tuple('decals_dr9_'+hemi+':'+b for b in ('g','r','z'))
        phot=Photometry(overlap['flux'][pairs,system,:3],overlap['variance'][pairs,system,:3],labels)
        z=np.where(np.isfinite(overlap['zspec'][pairs]),overlap['zspec'][pairs],rcfg['grid']['primary_z'])
        z=np.clip(z,*base.model.qso.support)
        views[hemi]=run_unified(unified,phot,overlap['ra'][pairs],overlap['dec'][pairs],z,rcfg,root/f'paired_{hemi}.npz')
    common=views['north']['eligible']&views['south']['eligible']
    report['paired']=dict(n=len(pairs),both_eligible=int(common.sum()),mean_absolute_qso_score_difference=float(np.mean(abs(views['north']['p_quasar'][common]-views['south']['p_quasar'][common]))) if common.any() else None,
        support_decision_agreement=float(np.mean(views['north']['support_rejected']==views['south']['support_rejected'])))
    report['checks']['support_has_rejection_power']=policy['informative_threshold']
    report['checks']['northern_original_tail']=all(report['grids'][f'north_{m}']['unified']['high_in_original_low_density_mask']==0 for m in rcfg['grid']['reference_magnitudes'][:2])
    report['checks']['southern_original_tail']=all(report['grids'][f'south_{m}']['unified']['high_in_original_low_density_mask']==0 for m in rcfg['grid']['reference_magnitudes'][:2])
    fits=[json.loads(p.read_text()) for p in (root/'fits').glob('*.json')]
    report['training']=dict(fits=len(fits),converged=sum(f['converged'] for f in fits),iteration_limit=sum(not f['converged'] for f in fits),
        maximum_budget_fraction=1.,elapsed_seconds=json.loads((root/'progress.json').read_text())['elapsed_seconds'],
        minimum_likelihood_increment=float(min(min(np.diff(f['history'])) for f in fits)))
    report['checks']['active_pointer_unchanged']=file_hash(Path(cfg['active_pointer']))==json.loads((root/'prepared.json').read_text())['active_pointer_hash']
    report['checks_pass']=all(report['checks'].values());report['validation_elapsed_seconds']=time.monotonic()-started
    write_json(root/'report.json',report);write_json(Path('docs/UNIFIED_PILOT_2026-09-30.json'),report)
    print('DONE',json.dumps(report['checks']),flush=True)


if __name__=='__main__':main()
