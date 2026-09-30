#!/usr/bin/env python
"""Tied-optical diagnostic using existing southern shapes; not pooled training.

Freeze parent priors, catch-all and spatial weights to isolate the shape change.
Rebind provenance explicitly and save a separate diagnostic bundle. WISE remains
untied here because native north/south luptitude softenings differ.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from qso_pcolor import PSFMultiSurveyBaseline, Photometry
from qso_pcolor.full_sample import write_json,file_hash
from qso_pcolor.legacy_homogenization import tied_native_mixture
from validate_full_sample_release import run_scores, metrics, auc
from validate_psf_catchalls import grid_data


def main():
    cfg=json.loads(Path('configs/legacy_unification_test.json').read_text())
    phot=json.loads((Path(cfg['output'])/'photometry_report.json').read_text())
    release=json.loads(Path('docs/FULL_SAMPLE_RELEASE_2026-09-30.json').read_text())
    rcfg=release['config'];parent=PSFMultiSurveyBaseline.load(cfg['parent_candidate'])
    mapping=phot['diagnostic_forward_map']
    identity=dict(parent=parent.manifest['files'],map=mapping,code=file_hash(Path(__file__)),
                  implementation=file_hash(Path('src/qso_pcolor/legacy_homogenization.py')))
    tag=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    out=Path(cfg['output'])/('shared_optical_'+tag);out.mkdir(parents=True,exist_ok=True)
    model=deepcopy(parent.model);labels=model.transform.bands
    ni=np.array([labels.index(f'decals_dr9_north:{b}') for b in 'grz'])
    si=np.array([labels.index(f'decals_dr9_south:{b}') for b in 'grz'])
    def tie(mix):return tied_native_mixture(mix,ni,si,np.array(mapping['north_from_south']),np.array(mapping['offset']),np.array(mapping['residual_covariance']))
    model.qso.mixtures=[tie(m) for m in model.qso.mixtures]
    model.background=tie(model.background)
    runid='diagnostic_shared_optical_'+tag
    model.meta=dict(model.meta,run_id=runid,diagnostic_only=True,parent_run_id=parent.model.meta['run_id'],
       unification_scope='Southern marginal shapes reused; north grz is an affine view with total paired residual scatter. No pooled density fitting. Priors, spatial weights and catch-all numerically inherited for intervention test only.')
    if not (out/'manifest.json').exists():
        model.save(out/'model.json')
        for fn in ('priors.json','outlier.json'):
            d=json.loads((Path(cfg['parent_candidate'])/fn).read_text())
            def rebind(value):
                if isinstance(value,dict):
                    if 'model_run_id' in value:
                        value['inherited_model_run_id']=value['model_run_id'];value['model_run_id']=runid
                        value['diagnostic_inherited_calibration']=True
                    for v in list(value.values()):rebind(v)
                elif isinstance(value,list):
                    for v in value:rebind(v)
            rebind(d);write_json(out/fn,d)
        manifest=dict(parent.manifest,bundle_id=runid,diagnostic_only=True,
            files={f:file_hash(out/f) for f in ('model.json','priors.json','outlier.json')})
        write_json(out/'manifest.json',manifest)
    base=PSFMultiSurveyBaseline.load(out)
    report=dict(bundle=str(out),identity=identity,scope=model.meta['unification_scope'],real={},grids={},active_model_changed=False)
    cache=Path(release['cache']);chosen=dict(np.load(cache/'rows.npz'))
    arrays={kind:{k:np.load(Path(rcfg['inputs'])/kind/(k+'.npy'),mmap_mode='r')
             for k in ('flux','variance','l','b','zspec')} for kind in ('qso','stars')}
    legacy=np.array([s.startswith('decals_') and s.split(':')[1] in ('g','r','z') for s in labels])
    for mode,mask in [('legacy_optical',legacy),('all_available',np.ones(len(labels),bool))]:
        preds={}; entry={}
        for kind in ('qso','stars'):
            d=arrays[kind];rows=chosen[kind];v=np.array(d['variance'][rows]);v[:,~mask]=np.inf
            p=Photometry(d['flux'][rows],v,labels);keep=p.observed.sum(axis=1)>=2
            data=dict(flux=p.flux[keep],variance=p.variance[keep],bands=labels,l=d['l'][rows[keep]],b=d['b'][rows[keep]],zprimary=arrays['qso']['zspec'][chosen['qso']][keep])
            r=run_scores(base,data,rcfg,out/f'{mode}_{kind}.npz');original=dict(np.load(cache/f'{mode}_{kind}_candidate.npz'))
            assert len(r['eligible'])==len(original['eligible'])
            north=p.observed[keep][:,ni].any(axis=1);preds[kind]=r
            entry[kind]={}
            for region,use in [('all',np.ones(len(north),bool)),('north',north),('south',~north)]:
                entry[kind][region]={name:metrics({k:v[use] for k,v in vals.items()},rcfg) for name,vals in [('parent',original),('shared',r)]}
            print(mode,kind,'shared eligible/highQ',r['eligible'].sum(),np.sum(r['eligible']&(r['p_quasar']>.5)),flush=True)
        entry['auc']=dict(parent=release['real'][mode]['auc']['candidate'],shared=auc(preds['qso'],preds['stars']))
        report['real'][mode]=entry;write_json(out/'report.json',report)
    for hemisphere in ('north','south'):
        for mag in rcfg['grid']['reference_magnitudes']:
            data=grid_data(base.model,hemisphere,mag,{'validation':rcfg['grid']})
            r=run_scores(base,data,rcfg,out/f'grid_{hemisphere}_{mag}.npz')
            # Use the ORIGINAL old/new common-low-density selection, never move goalposts.
            low=np.ones(len(r['eligible']),bool)
            for name in ('baseline','candidate'):
                orig=dict(np.load(cache/f'grid_{hemisphere}_{mag}_{name}.npz'))
                q=np.logaddexp(orig['log_lambda_sameq'],orig['log_lambda_fieldq']);b=orig['log_lambda_bkg']
                low&=(q<np.nanmax(q)+np.log(cfg['low_density_fraction']))&(b<np.nanmax(b)+np.log(cfg['low_density_fraction']))
            high=int((low&r['eligible']&(r['p_quasar']>rcfg['high_qso_probability'])).sum())
            entry=dict(metrics=metrics(r,rcfg),common_low_density=int(low.sum()),high_qso_low_density=high,
                parent_high_qso_low_density=release['grids'][f'{hemisphere}_{mag}'][str(cfg['low_density_fraction'])]['candidate']['high_qso_after_guard'])
            if mag==cfg['probe_reference_magnitude']:
                grid=np.linspace(*rcfg['grid']['grid_colour_range'],rcfg['grid']['grid_size']);gx,gy=np.meshgrid(grid,grid)
                pi=np.flatnonzero(np.isclose(gx.ravel(),cfg['probe_colours'][0])&np.isclose(gy.ravel(),cfg['probe_colours'][1]))[0]
                entry['probe']=dict(p_quasar=float(r['p_quasar'][pi]),eligible=bool(r['eligible'][pi]))
            report['grids'][f'{hemisphere}_{mag}']=entry
            print('grid',hemisphere,mag,'high QSO',entry['parent_high_qso_low_density'],'->',high,flush=True)
            write_json(out/'report.json',report)
    write_json(Path('docs/LEGACY_UNIFICATION_SHARED_DENSITY_2026-09-30.json'),report)
    print('DONE',out,flush=True)


if __name__=='__main__':main()
