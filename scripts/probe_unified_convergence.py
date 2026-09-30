#!/usr/bin/env python
"""Small matched convergence experiment; no production fitter or pointer changes."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from qso_pcolor.covariance_diagnostic import floor_covariance
from qso_pcolor.full_sample import file_hash, write_json
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.legacy import is_north
from qso_pcolor.multisurvey import MultiSurveyModel, conditional_log_prob
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.projected_parallel import ProjectedBatchFactory
from qso_pcolor.projected_xd import accumulate_projected
from qso_pcolor.sky_acquisition import acquisition_lock
from qso_pcolor.unified import native_view, operator
from run_unified_pilot import arrays, task_rows


def update_from_statistics(mix,stats,regularization,mode):
    count,first,second,ll,n=stats
    means,covs=mix.means.copy(),mix.covs.copy()
    for j in np.flatnonzero(count>1e-10):
        shift=first[j]/count[j];means[j]+=shift
        scatter=second[j]/count[j]-np.outer(shift,shift)
        covs[j]=(floor_covariance(scatter,regularization) if mode=='eigenvalue_floor'
                 else .5*(scatter+scatter.T)+regularization*np.eye(mix.n_dim))
    return GaussianMixture(count/count.sum(),means,covs,mix.labels)


def run_case(name,mode,options,out):
    out=Path(out);dest=out/f'{name}_{mode}.json'
    if dest.exists():return json.loads(dest.read_text())
    start=time.monotonic();root=Path(options['prepared_run']);cfg=json.loads((root/'config.json').read_text())
    task=next(t for t in json.loads((root/'tasks.json').read_text()) if t['name']==name)
    layout=json.loads((root/'layout.json').read_text());kind=task['kind'];d=arrays(root,kind)
    seed=options['seed']+int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little');rng=np.random.default_rng(seed)
    eligible=task_rows(d,task,cfg);train=np.sort(rng.choice(eligible,min(options['training_rows'],len(eligible)),replace=False))
    candidates=np.flatnonzero(d['role']==3)
    if kind=='qso':candidates=candidates[(d['zspec'][candidates]>=task['z']-cfg['z_half_width'])&(d['zspec'][candidates]<task['z']+cfg['z_half_width'])]
    f=d['flux'][candidates];v=d['variance'][candidates];obs=d['observed'][candidates]
    with np.errstate(invalid='ignore',divide='ignore'):
        good=(obs&(f/np.sqrt(v)>=cfg['reference_min_snr'])).any(axis=1)&(obs.sum(axis=1)>=2)
    candidates=candidates[good];north=is_north(d['ra'][candidates],d['dec'][candidates],d['b'][candidates])
    test=np.concatenate([rng.choice(candidates[north==value],min(options['evaluation_rows_per_hemisphere'],int((north==value).sum())),replace=False) for value in [False,True]])
    htest=is_north(d['ra'][test],d['dec'][test],d['b'][test])
    np.savez_compressed(out/f'{name}_{mode}_rows.npz',train=train,test=test)
    parent=MultiSurveyModel.load(Path(cfg['parent'])/'model.json');parent.meta['reference_min_snr']=cfg['reference_min_snr']
    anchor=parent.reference_indices(Photometry(d['flux'][test],d['variance'][test],parent.transform.bands))
    y=d['y'][test];noise=d['noise'][test];observed=d['observed'][test]
    mix=GaussianMixture.from_dict(task['init']);op=operator(layout,kind)
    if np.linalg.eigvalsh(mix.covs).min()<cfg['regularization']*(1-1e-9):raise ValueError('warm start violates floor')
    factory=ProjectedBatchFactory(root/kind,train,cfg['batch_size']);source=lambda:factory(0,len(train))
    history=[];evals={};hits={};snapshots={};previous=None
    def evaluate(it,model):
        native=native_view(model,layout,kind);values=np.empty(len(test))
        for a in np.unique(anchor):
            rr=np.flatnonzero(anchor==a)
            for lo in range(0,len(rr),128):
                ix=rr[lo:lo+128];cov=np.zeros((len(ix),len(noise[0]),len(noise[0])))
                cov[:,np.arange(len(noise[0])),np.arange(len(noise[0]))]=noise[ix]
                values[ix]=conditional_log_prob(native,y[ix],cov,observed[ix],int(a))
        evals[str(it)]=dict(mean=float(values.mean()),south=float(values[~htest].mean()) if (~htest).any() else None,
            north=float(values[htest].mean()) if htest.any() else None)
        snapshots[str(it)]=values
    evaluate(0,mix)
    # Histories describe the CURRENT model; snapshots and stopping indices agree.
    for it in range(options['max_iter']+1):
        stats=accumulate_projected(source,mix,{0:op});ll=stats[3]/stats[4];history.append(ll)
        if stats[4]!=len(train):raise ValueError('lost rows')
        check=it%options['evaluate_every']==0 or it==options['max_iter']
        if previous is not None:
            gain=ll-previous
            if mode=='eigenvalue_floor' and gain < -options['roundoff_tolerance']:
                raise ValueError(f'{name}: constrained likelihood decline {gain}')
            for tol in options['tolerances']:
                key=str(tol)
                if key not in hits and 0<=gain<tol*max(1,abs(previous)):
                    hits[key]=it;check=True
        if check and str(it) not in evals:evaluate(it,mix)
        write_json(out/f'{name}_{mode}_progress.json',dict(iteration=it,total=options['max_iter'],mean_loglike=ll,elapsed_seconds=time.monotonic()-start))
        if it==options['max_iter']:break
        mix=update_from_statistics(mix,stats,cfg['regularization'],mode);previous=ll
    final=snapshots[str(options['max_iter'])];comparison={}
    for tol,it in hits.items():
        delta=final-snapshots[str(it)]
        comparison[tol]=dict(stop_iteration=it,final_minus_stop_mean=float(delta.mean()),
            median_abs_change=float(np.median(abs(delta))),p95_abs_change=float(np.quantile(abs(delta),.95)))
    delta=final-snapshots[str(options['max_iter']-options['evaluate_every'])]
    report=dict(case=name,update=mode,z=task.get('z'),k=task['k'],train_rows=len(train),evaluation_rows=len(test),
        history=history,evaluations=evals,tolerance_comparison=comparison,
        last_block_prediction_change=dict(mean=float(delta.mean()),median_abs=float(np.median(abs(delta))),p95_abs=float(np.quantile(abs(delta),.95))),
        negative_steps=int((np.diff(history)<-options['roundoff_tolerance']).sum()),
        min_eigenvalue=float(np.linalg.eigvalsh(mix.covs).min()),elapsed_seconds=time.monotonic()-start,
        evaluation_note='Development density diagnostic; one QSO slice per case and global background; not full spatial/catch-all classification')
    np.savez_compressed(out/f'{name}_{mode}_predictions.npz',**snapshots)
    write_json(out/f'{name}_{mode}_mixture.json',mix.to_dict());write_json(dest,report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/unified_convergence_probe.json');args=parser.parse_args()
    options=json.loads(Path(args.config).read_text());identity=dict(options=options,
        prepared=file_hash(Path(options['prepared_run'])/'identity.json'),
        engine=file_hash(Path('src/qso_pcolor/projected_xd.py')),floor=file_hash(Path('src/qso_pcolor/covariance_diagnostic.py')),script=file_hash(Path(__file__)))
    tag=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16];out=Path(options['output'])/tag;out.mkdir(parents=True,exist_ok=True)
    write_json(out/'identity.json',identity);write_json(Path(options['output'])/'current.json',dict(directory=str(out)))
    start=time.monotonic();results=[]
    with acquisition_lock(out/'run.lock'),ProcessPoolExecutor(max_workers=options['workers']) as pool:
        pending=[pool.submit(run_case,name,mode,options,str(out)) for name in options['cases'] for mode in options['updates']]
        for done in as_completed(pending):
            r=done.result();results.append(r);print('DONE',r['case'],r['update'],r['tolerance_comparison'],'negative',r['negative_steps'],flush=True)
    report=dict(directory=str(out),config=options,results=results,elapsed_seconds=time.monotonic()-start,production_changed=False)
    write_json(out/'report.json',report);write_json(Path('docs/UNIFIED_CONVERGENCE_PROBE_2026-09-30.json'),report)
    print('COMPLETE',out,report['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
