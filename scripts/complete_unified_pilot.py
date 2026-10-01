#!/usr/bin/env python
"""Complete the small-sky unified candidate: spatial weights, counts and catch-all."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from qso_pcolor.full_sample import file_hash, write_json
from qso_pcolor.full_population import count_prior
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.joint_spatial import fit_joint_spatial_log_prob
from qso_pcolor.multisurvey import MultiSurveyModel, MultiSurveyOutlier, BandLuptitudeTransform
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.outlier import mixture_moments
from qso_pcolor.qso_model import SlicedColourRedshiftModel
from qso_pcolor.spatial import component_log_prob
from qso_pcolor.unified import native_view
from compare_psf_catchalls import evaluate_densities, fit_fractions
from run_unified_pilot import arrays, selected_cells


def get_root(config='configs/unified_pilot.json'):
    cfg=json.loads(Path(config).read_text())
    return Path(json.loads((Path(cfg['output'])/'current.json').read_text())['directory'])


def fit_arrays(root,kind):
    """Prepared arrays with fluxes the density fits used: extinction-corrected when declared."""
    data=arrays(root,kind)
    if 'flux_dered' in data:data=dict(data,flux=data['flux_dered'],variance=data['variance_dered'])
    return data


def extinction_shift(root,cfg,kind):
    """Per-band magnitude shift R_j <E(B-V)> of a population's fit/select rows (0 without extinction)."""
    layout=json.loads((root/'layout.json').read_text())
    if 'extinction' not in layout:return np.zeros(len(layout['native_labels'])),0.
    d=arrays(root,kind);use=np.isin(d['role'],cfg['fit_roles'])&selected_cells(d['cell'],cfg['training_cells'])
    mean=float(np.mean(d['ebv'][use]));return np.asarray(layout['extinction']['coefficients'])*mean,mean


def remap_mag(value,oldsoft,newsoft):
    a=2.5/np.log(10);f=2*oldsoft*np.sinh((22.5-np.asarray(value))/a-np.log(oldsoft))
    new=22.5-a*(np.arcsinh(f/(2*newsoft))+np.log(newsoft))
    jac=np.hypot(f,2*newsoft)/np.hypot(f,2*oldsoft) # d old-mag / d new-mag
    return new,jac


def build_priors(root,cfg,model,data,parent):
    original=json.loads((Path(cfg['parent'])/'priors.json').read_text());saved=deepcopy(original)
    regions={r['cone']:r for r in json.loads((Path(cfg['stellar_root'])/'spatial_roles.json').read_text())['regions']}
    train=np.isin(data['role'],cfg['fit_roles']) & selected_cells(data['cell'],cfg['training_cells'])
    rows=np.flatnonzero(train);fields=np.unique(data['field'][rows]);bands=model.transform.bands
    p=Photometry(data['flux'][rows],data['variance'][rows],bands);obs=p.observed
    a=2.5/np.log(10);values=22.5-a*(np.arcsinh(p.flux/(2*model.transform.softening))+np.log(model.transform.softening))
    report={};qshift,qebv=extinction_shift(root,cfg,'qso');bshift,bebv=extinction_shift(root,cfg,'stars')
    for j,label in enumerate(bands):
        pair=saved['anchors'][label];q=pair['qso_prior'];oldsoft=parent.transform.softening[j];newsoft=model.transform.softening[j]
        if qshift[j]:
            # Inherited abundance is per observed magnitude; corrected magnitudes are brighter by R<E>.
            # A mean shift, not a per-object convolution: E(B-V) scatter is a second-order effect.
            q['mag_centres']=(np.array(q['mag_centres'])-qshift[j]).tolist();q['mag_edges']=(np.array(q['mag_edges'])-qshift[j]).tolist()
            q['meta']['extinction_shift_mag']=float(qshift[j]);q['meta']['extinction_mean_ebv']=qebv
        if oldsoft!=newsoft:
            centres,jac=remap_mag(q['mag_centres'],oldsoft,newsoft)
            q['mag_centres']=centres.tolist();q['mag_edges']=remap_mag(q['mag_edges'],oldsoft,newsoft)[0].tolist()
            q['sigma']=(np.array(q['sigma'])*jac[None,:]).tolist()
            q['meta']['coordinate_change']='exact magnitude-grid map with density Jacobian; interpolation remains a grid approximation'
        edges=remap_mag(pair['background_density']['mag_edges'],oldsoft,newsoft)[0]
        counts=np.zeros((len(fields),len(edges)-1),int);area=np.zeros(len(fields));cells=[]
        system,band=label.split(':')
        for k,field in enumerate(fields):
            field=int(field);reg=regions[field];cells.append(reg['cell']);use=data['field'][rows]==field
            info=json.loads((Path(cfg['stellar_root'])/'areas'/f'cone_{field:03d}.json').read_text())
            if system.startswith('decals_') and band in ('g','r','z'):
                if system.endswith(reg['hemisphere']):area[k]=info['band_area_deg2'][band]
            else:
                cols=[i for i,b in enumerate(bands) if b.split(':')[0]==system]
                if obs[use][:,cols].any():area[k]=info['area_deg2']
            counts[k]=np.histogram(values[use & obs[:,j],j],edges)[0]
        meta=dict(reference_band=label,transform_id=model.transform_id,model_run_id=model.meta['run_id'],population='psf',
            roles=['fit','select'],pilot=cfg.get('pilot',True),area_calibration_complete=system.startswith('decals_') and band in ('g','r','z'))
        if counts.sum() and area.sum():
            density=count_prior(counts,area,np.array(cells),edges,nside=cfg['nside'],nside_parent=cfg['spatial']['nside_parent'],n0=cfg['density_n0'],meta=meta)
            pair['background_density']=density.to_dict();origin='pilot fit/select counts'
        else:
            # A limited sky pilot cannot certify unobserved external-survey dimensions.
            old=pair['background_density'];oldedges=np.array(old['mag_edges']);edges=edges-bshift[j];old['mag_edges']=edges.tolist()
            old['global_density']=(np.array(old['global_density'])*np.diff(oldedges)/np.diff(edges)).tolist()
            old['meta'].update(meta,sparse_population_prior=True,inherited_no_pilot_counts=True)
            origin='inherited full-candidate counts; no pilot support'
        q['meta'].update(meta,abundance_source=str(Path(cfg['parent'])/'priors.json'),
                         abundance_policy='inherited abundance; never spectroscopic training counts')
        pair['background_density']['meta'].update(meta)
        report[label]=dict(n_counted=int(counts.sum()),area_deg2=float(area.sum()),origin=origin,
            area_measured=meta['area_calibration_complete'])
    saved.update(transform_id=model.transform_id,model_run_id=model.meta['run_id'],report=report,
        note='Pilot spatial contaminant counts; inherited QSO abundance with coordinate Jacobian where needed; external areas approximate.')
    write_json(root/'bundle'/'priors.json',saved);write_json(root/'prior_report.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/unified_pilot.json')
    parser.add_argument('--root',type=Path)
    args=parser.parse_args()
    root=args.root if args.root is not None else get_root(args.config)
    cfg=json.loads((root/'config.json').read_text());layout=json.loads((root/'layout.json').read_text())
    if not (root/'training_complete.json').exists():raise RuntimeError('pilot fits have not finished')
    out=root/'bundle';out.mkdir(exist_ok=True)
    parent=MultiSurveyModel.load(Path(cfg['parent'])/'model.json')
    fits=[json.loads((root/'fits'/f'qso_{j:02d}.json').read_text()) for j in range(len(parent.qso.z_centres))]
    stars=json.loads((root/'fits'/'stars_00.json').read_text())
    latent_q=[GaussianMixture.from_dict(f['mixture']) for f in fits];latent_b=GaussianMixture.from_dict(stars['mixture'])
    qmix=[native_view(m,layout,'qso') for m in latent_q];bmix=native_view(latent_b,layout,'stars')
    data=fit_arrays(root,'stars');rows=np.flatnonzero(np.isin(data['role'],cfg['fit_roles']) & selected_cells(data['cell'],cfg['training_cells']))
    transform=BandLuptitudeTransform(tuple(layout['native_labels']),np.array(layout['softening']))
    start=time.monotonic();path=root/'spatial.json'
    if path.exists():
        from qso_pcolor.joint_spatial import JointSpatialWeights
        spatial=JointSpatialWeights.from_dict(json.loads(path.read_text()))
    else:
        lp=np.empty((len(rows),bmix.n_components))
        for lo in range(0,len(rows),512):
            rr=rows[lo:lo+512];f=transform(Photometry(data['flux'][rr],data['variance'][rr],transform.bands))
            lp[lo:lo+len(rr)]=component_log_prob(bmix,f.x,f.cov,f.observed)
        spatial=fit_joint_spatial_log_prob(bmix,lp,data['l'][rows],data['b'][rows],np.ones(len(rows)),
            **cfg['spatial'],meta=dict(roles=['fit','select'],pilot=cfg.get('pilot',True)))
        write_json(path,spatial.to_dict())
    print('SPATIAL COMPLETE',len(rows),flush=True)
    qso=SlicedColourRedshiftModel(parent.qso.z_centres,qmix,np.array([f['n'] for f in fits]),'unified_'+root.name,transform.bands,
        dict(per_slice=[dict(z=float(z),band_counts=t['band_counts']) for z,t in zip(parent.qso.z_centres,[t for t in json.loads((root/'tasks.json').read_text()) if t['kind']=='qso'])]))
    bounds=parent.background_bounds.copy();bshift,_=extinction_shift(root,cfg,'stars')
    for j in range(len(bounds)):
        bounds[j]=remap_mag(bounds[j],parent.transform.softening[j],transform.softening[j])[0]-bshift[j]
    model=MultiSurveyModel(qso,bmix,transform,parent.reference_priority,bounds,
        meta=dict(population='psf',run_id='unified_'+root.name,reference_min_snr=cfg['reference_min_snr'],
                  pilot=cfg.get('pilot',True),latent_dimensions=len(layout['latent_labels']),background_population='empirical PSF non-QSO contaminants',
                  settings={'config':{'min_band_training':cfg.get('min_band_training',20)}},
                  **({'extinction':layout['extinction']} if 'extinction' in layout else {}),
                  **({'qso_colours':'magnitude-independent (fixed magnitude coordinate '+layout['latent_labels'][layout['qso_coordinates']['index']]+')'}
                     if 'qso_coordinates' in layout else {})),spatial_background=spatial)
    model.save(out/'model.json');build_priors(root,cfg,model,data,parent)
    print('PRIORS COMPLETE',flush=True)
    catch=cfg['catchall'];mean,cov=mixture_moments([model.background])
    outlier=MultiSurveyOutlier(mean,catch['kappa']**2*cov,catch['kappa'],0.,transform.bands,model.transform_id,
        {'*':([0.,1.],[catch['global_prior_mean']])},meta=dict(population='psf',model_run_id=model.meta['run_id']),
        family=catch['family'],nu=catch['nu'],noise=catch['noise'])
    rr=np.flatnonzero(data['role']==2);parts=[]
    for lo in range(0,len(rr),2048):
        ii=rr[lo:lo+2048];path=root/f'catchall_{lo:07d}.npz'
        if path.exists():part=dict(np.load(path))
        else:
            part=evaluate_densities(model,data,ii,{'outlier':outlier},chunk=128)
            np.savez_compressed(path,**part)
        parts.append(part)
        print('CATCHALL',lo+len(ii),'/',len(rr),flush=True)
    logs={k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
    outlier.fractions=fit_fractions(logs['background'],logs['outlier'],logs['anchor'],logs['magnitude'],transform.bands,
        n_bins=catch['magnitude_bins'],strength=catch['pooling_strength'],config=catch)
    outlier.meta.update(roles=['calib'],rows=len(rr),pilot=cfg.get('pilot',True));outlier.save(out/'outlier.json')
    write_json(out/'latent.json',dict(layout=layout,qso=[m.to_dict() for m in latent_q],background=latent_b.to_dict(),
        warm_start=cfg['parent'],scope=cfg['policy']))
    files={name:file_hash(out/name) for name in ('model.json','priors.json','outlier.json')}
    write_json(out/'manifest.json',dict(kind='multisurvey_psf_bundle',files=files,bundle_id=root.name,
        min_abs_b_deg=cfg.get('min_abs_b_deg',25.),population='psf',status='candidate; support calibration pending',
        full_probability_calibration=False))
    write_json(root/'completion.json',dict(count_rows=len(rows),catchall_rows=len(rr),elapsed_seconds=time.monotonic()-start,
        active_model_changed=False))
    print('BUNDLE COMPLETE',out,flush=True)


if __name__=='__main__':main()
