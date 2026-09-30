#!/usr/bin/env python
"""Prepare or fit the bounded, full-survey shared-latent pilot from cached data."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from qso_pcolor.background import galactic_healpix
from qso_pcolor.full_sample import file_hash, write_json
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.multisurvey import MultiSurveyModel, BandLuptitudeTransform
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.projected_xd import fit_projected
from qso_pcolor.sky_acquisition import acquisition_lock
from qso_pcolor.unified import observation_layout, operator


def arrays(root, kind):
    return {p.stem: np.load(p, mmap_mode='r') for p in (Path(root)/kind).glob('*.npy')}


def prepare(cfg):
    parent = MultiSurveyModel.load(Path(cfg['parent'])/'model.json')
    calibration = json.loads(Path(cfg['calibration']).read_text())
    layout = observation_layout(parent.transform, calibration, native_variance_floor=cfg['native_variance_floor'])
    identity = dict(config=cfg, layout=layout, parent=file_hash(Path(cfg['parent'])/'model.json'),
        inputs=file_hash(Path(cfg['inputs'])/'manifest.json'), calibration=file_hash(Path(cfg['calibration'])),
        projected_xd=file_hash(Path('src/qso_pcolor/projected_xd.py')),
        runner=file_hash(Path(__file__)))
    tag = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    root = Path(cfg['output'])/tag; root.mkdir(parents=True,exist_ok=True)
    for folder in ('qso','stars','fits','progress'): (root/folder).mkdir(exist_ok=True)
    write_json(root/'identity.json',identity); write_json(root/'layout.json',layout)
    write_json(root/'config.json',cfg); write_json(Path(cfg['output'])/'current.json',dict(directory=str(root)))
    if (root/'prepared.json').exists(): return root
    transform = BandLuptitudeTransform(tuple(layout['native_labels']),np.array(layout['softening']))
    summary = {}
    for kind in ('qso','stars'):
        source = arrays(cfg['inputs'],kind)
        cells = galactic_healpix(source['l'],source['b'],cfg['nside'])
        keep = source['eligible'] & np.isin(cells,cfg['training_cells']+cfg['test_cells'])
        rows = np.flatnonzero(keep)
        for key in ('flux','variance','ra','dec','l','b','role','field','zspec','eligible'):
            np.save(root/kind/(key+'.npy'),source[key][rows])
        np.save(root/kind/'source_row.npy',rows);np.save(root/kind/'cell.npy',cells[rows])
        n,d = len(rows),len(layout['native_labels'])
        y = np.lib.format.open_memmap(root/kind/'y.npy',mode='w+',dtype='f8',shape=(n,d))
        noise = np.lib.format.open_memmap(root/kind/'noise.npy',mode='w+',dtype='f8',shape=(n,d))
        obs = np.lib.format.open_memmap(root/kind/'observed.npy',mode='w+',dtype='?',shape=(n,d))
        for lo in range(0,n,2048):
            rr=rows[lo:lo+2048];f=transform(Photometry(source['flux'][rr],source['variance'][rr],transform.bands))
            y[lo:lo+len(rr)]=f.x;noise[lo:lo+len(rr)]=np.diagonal(f.cov,axis1=1,axis2=2);obs[lo:lo+len(rr)]=f.observed
        y.flush();noise.flush();obs.flush()
        fit=np.isin(source['role'][rows],cfg['fit_roles']) & np.isin(cells[rows],cfg['training_cells'])
        summary[kind]=dict(total=n,roles={str(i):int((source['role'][rows]==i).sum()) for i in range(4)},
            fit_rows=int(fit.sum()),fit_band_counts=obs[fit].sum(axis=0).tolist(),
            negative_flux_fit_rows=int((obs[fit]&(source['flux'][rows[fit]]<0)).any(axis=1).sum()),
            row_indices_sha256=hashlib.sha256(rows.tobytes()).hexdigest())
    tasks=[];ix=np.array(layout['canonical_indices'])
    for kind,mixtures in [('stars',[parent.background]),('qso',parent.qso.mixtures)]:
        data=arrays(root,kind)
        for j,mix in enumerate(mixtures):
            init=mix.marginal(ix)
            init=GaussianMixture(init.weights,init.means,init.covs,tuple(layout['latent_labels']))
            task=dict(name=f'{kind}_{j:02d}',kind=kind,index=j,init=init.to_dict(),k=init.n_components)
            if kind=='qso':task['z']=float(parent.qso.z_centres[j])
            rr=task_rows(data,task,cfg);task['n']=len(rr)
            if not len(rr):raise ValueError('pilot footprint has empty QSO slice: '+task['name'])
            task['band_counts']=data['observed'][rr].sum(axis=0).tolist()
            tasks.append(task)
    write_json(root/'tasks.json',tasks)
    write_json(root/'prepared.json',dict(counts=summary,tasks=len(tasks),identity=identity,
        active_pointer_hash=file_hash(Path(cfg['active_pointer']))))
    print('PREPARED',root,json.dumps(summary),flush=True)
    return root


def task_rows(data,task,cfg):
    keep=np.isin(data['role'],cfg['fit_roles'])&np.isin(data['cell'],cfg['training_cells'])
    if task['kind']=='qso':keep&=(data['zspec']>=task['z']-cfg['z_half_width'])&(data['zspec']<task['z']+cfg['z_half_width'])
    return np.flatnonzero(keep)


def fit_task(task,cfg,root):
    root=Path(root);out=root/'fits'/(task['name']+'.json')
    if out.exists():return task['name']
    data=arrays(root,task['kind']);rr=task_rows(data,task,cfg)
    layout=json.loads((root/'layout.json').read_text());op=operator(layout,task['kind'])
    def source():
        for lo in range(0,len(rr),cfg['batch_size']):
            ii=rr[lo:lo+cfg['batch_size']];v=data['noise'][ii];cov=np.zeros((len(ii),v.shape[1],v.shape[1]))
            cov[:,np.arange(v.shape[1]),np.arange(v.shape[1])]=v
            yield data['y'][ii],cov,data['observed'][ii],np.zeros(len(ii),int)
    start=time.monotonic()
    def progress(iteration,mix,ll,n):
        write_json(root/'progress'/(task['name']+'.json'),dict(iteration=iteration,complete=False,
            rows=n,mean_loglike=ll,elapsed_seconds=time.monotonic()-start))
        write_json(root/'progress'/(task['name']+'_checkpoint.json'),dict(mixture=mix.to_dict(),iteration=iteration))
    fit=fit_projected(source,init=GaussianMixture.from_dict(task['init']),operators={0:op},
        expected_rows=len(rr),max_iter=cfg['max_iter'],tol=cfg['tol'],regularization=cfg['regularization'],progress=progress)
    write_json(out,dict(task=task['name'],mixture=fit.mixture.to_dict(),n=len(rr),k=task['k'],
        n_iter=fit.n_iter,history=fit.history,converged=fit.converged,mean_loglike=fit.mean_loglike,
        elapsed_seconds=time.monotonic()-start,fit_rows_sha256=hashlib.sha256(data['source_row'][rr].tobytes()).hexdigest()))
    write_json(root/'progress'/(task['name']+'.json'),dict(iteration=fit.n_iter,complete=True,rows=len(rr),elapsed_seconds=time.monotonic()-start))
    return task['name']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fit',action='store_true');parser.add_argument('--config',default='configs/unified_pilot.json');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());root=prepare(cfg)
    if not args.fit:return
    tasks=json.loads((root/'tasks.json').read_text());budget=sum(t['n']*t['k']*cfg['max_iter'] for t in tasks)
    with acquisition_lock(root/'fit.lock'):
        start=time.monotonic()
        with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
            pending={pool.submit(fit_task,t,cfg,str(root)):t for t in tasks if not (root/'fits'/(t['name']+'.json')).exists()}
            while pending:
                done,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
                for f in done:print('FIT COMPLETE',f.result(),flush=True);del pending[f]
                work=actual=complete=0
                for t in tasks:
                    p=root/'progress'/(t['name']+'.json');s=json.loads(p.read_text()) if p.exists() else dict(iteration=0,complete=False)
                    complete+=s['complete'];actual+=t['n']*t['k']*s['iteration'];work+=t['n']*t['k']*(cfg['max_iter'] if s['complete'] else s['iteration'])
                state=dict(completed_fits=complete,total_fits=len(tasks),budget_progress_fraction=work/budget,
                    executed_fraction_of_maximum_budget=actual/budget,elapsed_seconds=time.monotonic()-start,
                    scope='Density-fit row x component x iteration budget; excludes completion/validation. Early convergence retires remaining fit budget.')
                write_json(root/'progress.json',state);print('PROGRESS',json.dumps(state),flush=True)
        write_json(root/'training_complete.json',dict(tasks=len(tasks),active_model_changed=False))
    print('TRAINING COMPLETE',root,flush=True)


if __name__=='__main__':main()
