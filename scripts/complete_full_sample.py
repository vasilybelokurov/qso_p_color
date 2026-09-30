#!/usr/bin/env python
"""Complete a frozen full-data candidate with counts and a calib-only catch-all.

No density fitting, database access or active-pointer changes. Configuration and
input hashes identify a resumable output; every eligible count/calibration row
is used. External-survey area approximations remain explicit in the priors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from qso_pcolor import PSFMultiSurveyBaseline, Photometry
from qso_pcolor.multisurvey import MultiSurveyModel, MultiSurveyOutlier
from qso_pcolor.full_sample import file_hash, write_json
from qso_pcolor.full_population import population_rows, count_prior, rebind_qso_prior
from qso_pcolor.outlier import mixture_moments
from compare_psf_catchalls import fit_fractions, evaluate_densities


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_priors(cfg, model, data, source, root):
    roles = json.loads((Path(cfg['stellar_root'])/'spatial_roles.json').read_text())
    regions = {r['cone']:r for r in roles['regions']}
    rows = population_rows(data, tuple(cfg['count_roles']))
    fields = np.unique(data['field'][rows]); bands = model.transform.bands
    old = json.loads((Path(cfg['source_bundle'])/'priors.json').read_text())
    counts = {b:np.zeros((len(fields),len(old['anchors'][b]['background_density']['mag_edges'])-1),int) for b in bands}
    areas = np.zeros((len(fields),len(bands))); observed_count=np.zeros_like(areas,dtype=int)
    cells = np.array([regions[int(f)]['cell'] for f in fields])
    origins = {}
    selected = np.zeros(len(data['role']),bool);selected[rows]=True
    for i,field in enumerate(fields):
        region=regions[int(field)]
        if region['role'] not in cfg['count_roles']: raise ValueError('count role/region mismatch')
        rr=np.flatnonzero(selected & (data['field']==field))
        area=json.loads((Path(cfg['stellar_root'])/'areas'/f'cone_{field:03d}.json').read_text())
        phot=Photometry(data['flux'][rr],data['variance'][rr],bands)
        obs=phot.observed
        values=22.5-2.5/np.log(10)*(np.arcsinh(phot.flux/(2*model.transform.softening))+np.log(model.transform.softening))
        for a,label in enumerate(bands):
            system,band=label.split(':')
            observed_count[i,a]=obs[:,a].sum()
            if system.startswith('decals_') and band in ('g','r','z'):
                origins[label]='measured per-band Legacy mask/exposure area'
                if system.endswith(region['hemisphere']): areas[i,a]=area['band_area_deg2'][band]
            else:
                origins[label]='approximate: base Legacy usable area of cones with any usable survey measurement'
                cols=[j for j,b in enumerate(bands) if b.split(':')[0]==system]
                if obs[:,cols].any(): areas[i,a]=area['area_deg2']
            edges=old['anchors'][label]['background_density']['mag_edges']
            counts[label][i]=np.histogram(values[obs[:,a],a],edges)[0]
    result=dict(kind='multisurvey_priors',transform_id=model.transform_id,model_run_id=model.meta['run_id'],population='psf',
                completeness_constant=old['completeness_constant'],completeness_origin=str(Path(cfg['source_bundle'])/'priors.json'),
                note=cfg['prior_policy'],anchors={},report={})
    prior_path=Path(cfg['source_bundle'])/'priors.json';prior_sha=file_hash(prior_path)
    for a,label in enumerate(bands):
        old_pair=old['anchors'][label]
        meta=dict(reference_band=label,transform_id=model.transform_id,model_run_id=model.meta['run_id'],population='psf',
                  roles=cfg['count_roles'],area_origin=origins[label],external_area_approximation=not origins[label].startswith('measured'),
                  input_manifest_sha256=file_hash(Path(cfg['input_root'])/'manifest.json'),n_field=int(observed_count[:,a].sum()),
                  inherited_magnitude_edges=True,area_calibration_complete=origins[label].startswith('measured'))
        sky=cfg['spatial']
        prior=count_prior(counts[label],areas[:,a],cells,old_pair['background_density']['mag_edges'],
                          nside=sky['nside'],nside_parent=sky['nside_parent'],n0=sky['density_n0'],meta=meta)
        result['anchors'][label]=dict(qso_prior=rebind_qso_prior(old_pair['qso_prior'],run_id=model.meta['run_id'],
                                      origin=str(prior_path),origin_sha256=prior_sha),background_density=prior.to_dict())
        result['report'][label]=dict(n_observed=int(observed_count[:,a].sum()),n_counted=int(counts[label].sum()),
                                     outside_inherited_magnitude_range=int(observed_count[:,a].sum()-counts[label].sum()),
                                     area_deg2=float(areas[:,a].sum()),area_origin=origins[label],regions=int((areas[:,a]>0).sum()),
                                     cells=len(prior.area),magnitude_bins=len(prior.mag_edges)-1)
    write_json(root/'priors.json',result)
    return dict(count_rows=len(rows),count_regions=len(fields),anchors=len(result['anchors']),qso_abundance_unchanged=True,
                measured_area_anchors=sum(s.startswith('measured') for s in origins.values()),
                approximate_area_anchors=sum(not s.startswith('measured') for s in origins.values()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/full_sample_completion.json'))
    args=parser.parse_args();cfg=json.loads(args.config.read_text());started=time.monotonic()
    if cfg['count_roles']!=['fit','select'] or cfg['catchall_roles']!=['calib']:
        raise ValueError('frozen role contract changed')
    active_hash=file_hash(Path(cfg['active_pointer']))
    model=MultiSurveyModel.load(cfg['density_model']);source=PSFMultiSurveyBaseline.load(cfg['source_bundle'])
    if model.transform_id!=source.model.transform_id or model.transform.bands!=source.model.transform.bands:
        raise ValueError('cannot reuse priors across transforms')
    if model.spatial_background.meta.get('roles')!=cfg['count_roles']:
        raise ValueError('spatial weights have unexpected roles')
    manifest_path=Path(cfg['input_root'])/'manifest.json';manifest=json.loads(manifest_path.read_text())
    if not manifest['complete']:raise ValueError('input manifest incomplete')
    print('Checking prepared input hashes',flush=True)
    for name,digest in manifest['files'].items():
        if file_hash(Path(cfg['input_root'])/name)!=digest:raise ValueError('input hash mismatch: '+name)
    area_files=sorted((Path(cfg['stellar_root'])/'areas').glob('cone_*.json'))
    identity=dict(config=cfg,density_sha256=file_hash(Path(cfg['density_model'])),source_files=source.manifest['files'],
                  input_manifest_sha256=file_hash(manifest_path),areas={p.name:file_hash(p) for p in area_files},
                  regions_sha256=file_hash(Path(cfg['stellar_root'])/'spatial_roles.json'),
                  code={p:file_hash(Path(p)) for p in [__file__,'src/qso_pcolor/full_population.py','scripts/compare_psf_catchalls.py']})
    run=signature(identity)[:16];root=Path(cfg['output_root'])/run;root.mkdir(parents=True,exist_ok=True)
    write_json(root/'identity.json',identity)
    data={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (Path(cfg['input_root'])/'stars').glob('*.npy')}
    priors=build_priors(cfg,model,data,source,root)
    write_json(root/'prior_report.json',priors);print('Priors',priors,flush=True)
    catch=cfg['catchall'];mean,cov=mixture_moments([model.background])
    outlier=MultiSurveyOutlier(mean,catch['kappa']**2*cov,catch['kappa'],0.,model.transform.bands,model.transform_id,
            {'*':([0.,1.],[catch['global_prior_mean']])},meta=dict(model_run_id=model.meta['run_id'],population='psf'),
            family=catch['family'],nu=catch['nu'],noise=catch['noise'])
    rows=population_rows(data,tuple(cfg['catchall_roles']));parts=[]
    density_root=root/'catchall_densities';density_root.mkdir(exist_ok=True)
    for lo in range(0,len(rows),cfg['checkpoint_rows']):
        rr=rows[lo:lo+cfg['checkpoint_rows']];path=density_root/f'{lo:09d}.npz'
        if path.exists():
            saved=dict(np.load(path,allow_pickle=False))
            if not np.array_equal(saved.pop('rows'),rr):raise ValueError('density checkpoint rows differ')
        else:
            saved=evaluate_densities(model,data,rr,{'outlier':outlier},chunk=cfg['density_batch_size'])
            with path.with_suffix('.tmp').open('wb') as f:np.savez(f,rows=rr,**saved)
            path.with_suffix('.tmp').replace(path)
        if not all(np.isfinite(v).all() for v in saved.values()):raise ValueError('nonfinite calibration density')
        parts.append(saved)
        progress=dict(stage='catchall calibration densities',done=lo+len(rr),total=len(rows),fraction=(lo+len(rr))/len(rows),
                      elapsed_seconds=time.monotonic()-started,active_pointer_changed=False,density_fitting=False)
        write_json(root/'progress.json',progress)
        print(f"Catch-all {lo+len(rr):,}/{len(rows):,} ({progress['fraction']:.1%})",flush=True)
    logs={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    fractions=fit_fractions(logs['background'],logs['outlier'],logs['anchor'],logs['magnitude'],model.transform.bands,
              n_bins=catch['magnitude_bins'],strength=catch['pooling_strength'],config=catch)
    outlier.fractions=fractions
    outlier.meta.update(roles=cfg['catchall_roles'],n_fit=len(rows),fit_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
                        density_model_sha256=identity['density_sha256'],input_manifest_sha256=identity['input_manifest_sha256'],
                        fixed_hyperparameters_from=cfg['source_bundle'],config=catch,completion_run=run)
    if not all(np.all((v>0)&(v<1)) for _,v in fractions.values()):raise ValueError('catch-all weights must be positive')
    outlier.save(root/'outlier.json')
    shutil.copyfile(cfg['density_model'],root/'model.json')
    files={name:file_hash(root/name) for name in ('model.json','priors.json','outlier.json')}
    bundle=dict(kind='multisurvey_psf_bundle',files=files,bundle_id=signature(files)[:16],min_abs_b_deg=cfg['min_abs_b_deg'],population='psf',
                status='completed candidate; requires complete-score release checks',method='one joint 41-band model',
                full_probability_calibration=False,completion_run=run)
    write_json(root/'manifest.json',bundle)
    loaded=PSFMultiSurveyBaseline.load(root)
    assert len(loaded.priors)==len(model.transform.bands)==41
    assert files['model.json']==identity['density_sha256']
    if file_hash(Path(cfg['active_pointer']))!=active_hash:raise ValueError('active pointer changed')
    report=dict(candidate=str(root),bundle_id=bundle['bundle_id'],density_model_byte_identical=True,active_pointer_changed=False,
                priors=priors,catchall_rows=len(rows),catchall_roles=cfg['catchall_roles'],
                pooled_eta=float(fractions['*'][1][0]),eta_min=float(min(v.min() for _,v in fractions.values())),
                eta_max=float(max(v.max() for _,v in fractions.values())),elapsed_seconds=time.monotonic()-started,
                full_probability_calibration=False,release_ready=False,remaining=['bright-star evidence diagnosis','complete-score release checks'])
    write_json(root/'completion.json',report);write_json(Path(cfg['output_root'])/'candidate.json',dict(bundle=run))
    write_json(root/'progress.json',dict(stage='complete',fraction=1.,**report))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
