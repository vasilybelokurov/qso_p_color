#!/usr/bin/env python
"""Bounded complete-score release audit; no fitting, downloads or promotion."""
import json
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.special import logsumexp
from scipy.stats import mannwhitneyu
from qso_pcolor import PSFMultiSurveyBaseline, Photometry, RedshiftMatch, BlendPolicy
from qso_pcolor.background import galactic_healpix
from qso_pcolor.full_sample import write_json,file_hash
from validate_psf_catchalls import grid_data

FIELDS=('log_r_per_unit_z','p_sameq','p_outlier','dz_match_eff','log_lambda_sameq','log_lambda_fieldq',
        'log_lambda_bkg','log_lambda_out','log_bayes_factor_qz_bkg','loglike_qso_zprimary','loglike_bkg',
        'qso_ood_sigma_any_z','bkg_ood_sigma','outlier_fraction','background_local_weight','background_density_level','ref_mag')


def run_scores(base,data,cfg,path):
    """Use the complete guarded public scorer and retain every rejection."""
    if path.exists(): return dict(np.load(path,allow_pickle=False))
    scores=[];eligible=[]
    for lo in range(0,len(data['flux']),cfg['batch_size']):
        hi=min(lo+cfg['batch_size'],len(data['flux']))
        rr=slice(lo,hi)
        out,decision=base.score(Photometry(data['flux'][rr],data['variance'][rr],data['bands']),
             morphology=['PSF']*(hi-lo),z_primary=data['zprimary'][rr],l_deg=data['l'][rr],b_deg=data['b'][rr],
             match=RedshiftMatch(half_width_kms=cfg['window_kms']),blend_policy=BlendPolicy(**cfg['blend_policy']),
             ood_flag_sigma=cfg['ood_flag_sigma'],separation_arcsec=cfg['fixture_separation_arcsec'],fracflux=cfg['fixture_fracflux'])
        scores.extend(out);eligible.extend(decision['eligible'])
    result={k:np.array([getattr(s,k) for s in scores]) for k in FIELDS}
    result.update(eligible=np.array(eligible,bool),status=np.array([s.status for s in scores]),
                  flags=np.array([','.join(s.quality_flags) for s in scores]),
                  reference=np.array([s.reference_band for s in scores]))
    q=np.logaddexp(result['log_lambda_sameq'],result['log_lambda_fieldq'])
    total=logsumexp(np.stack([q,result['log_lambda_bkg'],result['log_lambda_out']]),axis=0)
    result['p_quasar']=np.exp(q-total)
    with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**result)
    path.with_suffix('.tmp').replace(path)
    return result


def metrics(r,cfg):
    ok=r['eligible'];outside=np.array(['outside_both_models' in f for f in r['flags']])
    finite=all(np.isfinite(r[k][ok]).all() for k in ('log_r_per_unit_z','p_sameq','p_outlier','p_quasar','dz_match_eff'))
    bounded=all(((r[k][ok]>=0)&(r[k][ok]<=1+1e-12)).all() for k in ('p_sameq','p_outlier','p_quasar'))
    window=bool(np.allclose(r['p_sameq'][ok],np.exp(r['log_r_per_unit_z'][ok])*r['dz_match_eff'][ok],rtol=1e-10,atol=1e-12))
    guard=bool(not ok[outside].any() and np.isnan(r['log_r_per_unit_z'][outside]).all() and np.isnan(r['p_sameq'][outside]).all())
    return dict(n=len(ok),eligible=int(ok.sum()),status=dict(Counter(r['status'].tolist())),finite_eligible=bool(finite),
       probabilities_bounded=bool(bounded),window_identity=window,hard_guard=guard,outside_both=int(outside.sum()),
       qso_above_half=int((ok&(r['p_quasar']>cfg['high_qso_probability'])).sum()),
       median_log_r=float(np.median(r['log_r_per_unit_z'][ok])) if ok.any() else None,
       spatial_weight_mean=float(np.mean(r['background_local_weight'][ok])) if ok.any() else None,
       density_levels=dict(Counter(r['background_density_level'][ok].astype(int).tolist())))


def auc(q,b):
    qs=np.where(q['eligible'],q['log_r_per_unit_z'],-np.inf)
    bs=np.where(b['eligible'],b['log_r_per_unit_z'],-np.inf)
    return float(mannwhitneyu(qs,bs).statistic/(len(qs)*len(bs)))


def main():
    cfg=json.loads(Path('configs/full_sample_release.json').read_text());models={k:PSFMultiSurveyBaseline.load(cfg[k]) for k in ('baseline','candidate')}
    bands=models['candidate'].model.transform.bands
    signature=hashlib.sha256(json.dumps(dict(config=cfg,models={k:b.manifest['files'] for k,b in models.items()},
                           code=file_hash(Path(__file__))),sort_keys=True).encode()).hexdigest()[:16]
    root=Path(cfg['output'])/signature;root.mkdir(parents=True,exist_ok=True)
    report=dict(signature=signature,config=cfg,models={k:b.manifest['bundle_id'] for k,b in models.items()},
                cache=str(root),real={},grids={},targeted={},numerical_pass=True,scientific_review=[],probability_calibration=False)
    arrays={};chosen={};rng=np.random.default_rng(cfg['seed'])
    for kind in ('qso','stars'):
        d={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (Path(cfg['inputs'])/kind).glob('*.npy')};arrays[kind]=d
        use=(d['role']==3)&d['eligible']
        if kind=='qso':
            use &= np.isin(galactic_healpix(d['l'],d['b'],4),models['baseline'].model.qso.meta['heldout_blocks'])
            use &= models['candidate'].model.qso.in_support(d['zspec'])
        halves=[np.flatnonzero(use&(d['dec']<cfg['hemisphere_split_dec'])),np.flatnonzero(use&(d['dec']>=cfg['hemisphere_split_dec']))]
        chosen[kind]=np.concatenate([rng.choice(x,cfg['rows_per_class_per_hemisphere'],replace=False) for x in halves])
    np.savez(root/'rows.npz',**chosen)
    legacy=np.array([b.startswith('decals_') and b.split(':')[1] in ('g','r','z') for b in bands])
    sdss=np.array([b.startswith('sdss:') for b in bands]);ps1=np.array([b.startswith('ps1:') for b in bands])
    masks=dict(all_available=np.ones(len(bands),bool),optical=legacy|sdss|ps1,legacy_optical=legacy,
               legacy_wise=np.array([b.startswith('decals_') for b in bands]),sdss_only=sdss,ps1_only=ps1)
    for mode,mask in masks.items():
        entry={};predictions={}
        for kind in ('qso','stars'):
            d=arrays[kind];rr=chosen[kind];v=np.array(d['variance'][rr]);v[:,~mask]=np.inf
            phot=Photometry(d['flux'][rr],v,bands);keep=phot.observed.sum(axis=1)>=2
            z=arrays['qso']['zspec'][chosen['qso']]
            data=dict(flux=phot.flux[keep],variance=phot.variance[keep],bands=bands,l=d['l'][rr[keep]],b=d['b'][rr[keep]],zprimary=z[keep])
            entry[kind]={};predictions[kind]={}
            for name,base in models.items():
                r=run_scores(base,data,cfg,root/f'{mode}_{kind}_{name}.npz');predictions[kind][name]=r
                m=metrics(r,cfg);entry[kind][name]=m
                report['numerical_pass'] &= all(m[k] for k in ('finite_eligible','probabilities_bounded','window_identity','hard_guard'))
                print(mode,kind,name,'eligible',m['eligible'],'/',m['n'],'highQ',m['qso_above_half'],flush=True)
            # Same QSO photometry, primary at a different redshift: field_q must compete.
            if kind=='qso' and mode in ('all_available','optical','legacy_optical'):
                lo,hi=models['candidate'].model.qso.support
                wrong=dict(data,zprimary=lo+np.mod(data['zprimary']-lo+cfg['wrong_redshift_shift'],hi-lo))
                for name,base in models.items():
                    r=run_scores(base,wrong,cfg,root/f'{mode}_wrongz_{name}.npz')
                    true=predictions[kind][name];joint=r['eligible']&true['eligible']
                    entry[kind][name]['wrong_z_check']=dict(n=int(joint.sum()),
                        fraction_true_rank_higher=float(np.mean(true['log_r_per_unit_z'][joint]>r['log_r_per_unit_z'][joint])))
        entry['auc']={name:auc(predictions['qso'][name],predictions['stars'][name]) for name in models}
        entry['auc_gain']=entry['auc']['candidate']-entry['auc']['baseline']
        if min(entry[k]['candidate']['n'] for k in ('qso','stars'))>=cfg['minimum_rows_for_auc'] and entry['auc_gain'] < -cfg['maximum_auc_decrease']:
            report['scientific_review'].append(mode+': AUC decrease exceeds declared tolerance')
        report['real'][mode]=entry
        write_json(root/'report.json',report)
    # Exact cases identified previously, including SDSS-only exceptions.
    diag=json.loads((Path(cfg['candidate'])/'bright_star_diagnostic.json').read_text());d=arrays['stars']
    for mode,details in diag['modes'].items():
        if mode not in masks or not details['new_high_evidence_rows']:continue
        rr=np.array(details['new_high_evidence_rows']);v=np.array(d['variance'][rr]);v[:,~masks[mode]]=np.inf
        target={}
        for name,base in models.items():
            perz=[]
            for z in diag['configuration']['primary_redshifts']:
                data=dict(flux=d['flux'][rr],variance=v,bands=bands,l=d['l'][rr],b=d['b'][rr],zprimary=np.full(len(rr),z))
                r=run_scores(base,data,cfg,root/f'targeted_{mode}_{z}_{name}.npz');perz.append(r)
                m=metrics(r,cfg);report['numerical_pass'] &= all(m[k] for k in ('finite_eligible','probabilities_bounded','window_identity','hard_guard'))
            target[name]=dict(max_p_quasar=[float(np.max([p['p_quasar'][i] if p['eligible'][i] else 0 for p in perz])) for i in range(len(rr))],
                             max_p_same=[float(np.max([p['p_sameq'][i] if p['eligible'][i] else 0 for p in perz])) for i in range(len(rr))])
        report['targeted'][mode]=dict(rows=rr.tolist(),scores=target)
        print('targeted',mode,target,flush=True)
    gridcfg={'validation':cfg['grid']}
    for hemisphere in ('south','north'):
        for mag in cfg['grid']['reference_magnitudes']:
            name=f'{hemisphere}_{mag}';data=grid_data(models['candidate'].model,hemisphere,mag,gridcfg)
            scored={k:run_scores(b,data,cfg,root/f'grid_{name}_{k}.npz') for k,b in models.items()}
            entry={}
            for key,r in scored.items():
                entry[key]=metrics(r,cfg);report['numerical_pass'] &= all(entry[key][k] for k in ('finite_eligible','probabilities_bounded','window_identity','hard_guard'))
            # Define identical low-density grid points using BOTH models, avoiding
            # a changing selection when comparing tails.
            for frac in cfg['low_density_peak_fractions']:
                low=np.ones(len(data['flux']),bool)
                for r in scored.values():
                    q=np.logaddexp(r['log_lambda_sameq'],r['log_lambda_fieldq']);b=r['log_lambda_bkg']
                    low &= (q<np.nanmax(q)+np.log(frac)) & (b<np.nanmax(b)+np.log(frac))
                stats={k:dict(high_qso_after_guard=int((low&r['eligible']&(r['p_quasar']>cfg['high_qso_probability'])).sum()),
                               rejected=int((low&~r['eligible']).sum())) for k,r in scored.items()}
                entry[str(frac)]=dict(n=int(low.sum()),**stats)
                increase=(stats['candidate']['high_qso_after_guard']-stats['baseline']['high_qso_after_guard'])/max(1,int(low.sum()))
                if increase>cfg['maximum_grid_high_rate_increase']:report['scientific_review'].append(f'{name} fraction{frac}: grid-tail increase {increase:.3f}')
            report['grids'][name]=entry;print('grid',name,entry,flush=True);write_json(root/'report.json',report)
    report['automatic_checks_pass']=report['numerical_pass'] and not report['scientific_review']
    write_json(root/'report.json',report);write_json(Path('docs/FULL_SAMPLE_RELEASE_2026-09-30.json'),report)
    print('COMPLETE',report['automatic_checks_pass'],report['scientific_review'],flush=True)


if __name__=='__main__':main()
