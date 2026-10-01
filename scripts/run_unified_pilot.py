#!/usr/bin/env python
"""Prepare or fit the bounded, full-survey shared-latent pilot from cached data."""
import argparse
from contextlib import nullcontext
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
from qso_pcolor.multisurvey import conditional_log_prob
from qso_pcolor.projected_xd import accumulate_projected, fit_projected, native_mixture
from qso_pcolor.sky_acquisition import acquisition_lock
from qso_pcolor.xd import XDFitResult
from qso_pcolor.unified import observation_layout, operator


class FitPaused(Exception):
    """Raised at a stopping check when the run root contains a PAUSE file."""


def arrays(root, kind):
    return {p.stem: np.load(p, mmap_mode='r') for p in (Path(root)/kind).glob('*.npy')}


def selected_cells(cells, selection):
    """None explicitly selects the full footprint; an empty list selects none."""
    return np.ones(len(cells), bool) if selection is None else np.isin(cells, selection)


def magnitude_colour_matrix(labels, magnitude):
    """T with u = T x: u[magnitude] = x[magnitude], u[i] = x[i] - x[magnitude] otherwise."""
    k = list(labels).index(magnitude); t = np.eye(len(labels)); t[:, k] -= 1.; t[k, k] = 1.
    return t


def to_magnitude_colour(mix, t, index, fixed):
    """Express a latent mixture in (magnitude, colours) and impose the fixed magnitude block."""
    means = mix.means @ t.T; covs = t @ mix.covs @ t.T
    means[:, index] = fixed['mean']; covs[:, index, :] = 0.; covs[:, :, index] = 0.; covs[:, index, index] = fixed['variance']
    return means, covs


def finalize_layout(layout, cfg):
    """Add the declared extinction correction and the QSO magnitude-colour coordinates."""
    if cfg.get('extinction'):
        from qso_pcolor.extinction import extinction_record
        layout['extinction'] = extinction_record(layout, cfg['extinction'])
    rule = cfg.get('qso_magnitude_independent')
    if rule:
        t = magnitude_colour_matrix(layout['latent_labels'], rule['magnitude'])
        h = np.array(layout['operators']['qso']['matrix'])
        layout['operators']['qso']['matrix'] = (h @ np.linalg.inv(t)).tolist()
        k = layout['latent_labels'].index(rule['magnitude'])
        layout['qso_coordinates'] = dict(matrix=t.tolist(), index=k, mean=float(rule['mean']), variance=float(rule['variance']),
            labels=[('magnitude:' if i == k else 'colour:')+(l if i == k else l+'-'+rule['magnitude'])
                    for i, l in enumerate(layout['latent_labels'])],
            note='QSO colours relative to the magnitude coordinate are independent of it; magnitude dependence lives in the abundance prior.')
    return layout


def prepare(cfg):
    parent = MultiSurveyModel.load(Path(cfg['parent'])/'model.json')
    calibration = json.loads(Path(cfg['calibration']).read_text())
    layout = observation_layout(parent.transform, calibration, native_variance_floor=cfg['native_variance_floor'])
    layout = finalize_layout(layout, cfg)
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
        keep = source['eligible'] & (selected_cells(cells,cfg['training_cells']) |
                                     selected_cells(cells,cfg['test_cells']))
        rows = np.flatnonzero(keep)
        for key in ('flux','variance','ra','dec','l','b','role','field','zspec','eligible'):
            np.save(root/kind/(key+'.npy'),source[key][rows])
        np.save(root/kind/'source_row.npy',rows);np.save(root/kind/'cell.npy',cells[rows])
        n,d = len(rows),len(layout['native_labels'])
        # flux/variance stay as catalogued; *_dered are what every density fit uses.
        flux,variance = source['flux'],source['variance']
        if 'extinction' in layout:
            from qso_pcolor.extinction import sfd_ebv, deredden_arrays
            ebv=sfd_ebv(source['l'][rows],source['b'][rows]);np.save(root/kind/'ebv.npy',ebv)
            fd=np.lib.format.open_memmap(root/kind/'flux_dered.npy',mode='w+',dtype='f8',shape=(n,d))
            vd=np.lib.format.open_memmap(root/kind/'variance_dered.npy',mode='w+',dtype='f8',shape=(n,d))
            for lo in range(0,n,65536):
                fd[lo:lo+65536],vd[lo:lo+65536]=deredden_arrays(source['flux'][rows[lo:lo+65536]],source['variance'][rows[lo:lo+65536]],
                    ebv[lo:lo+65536],layout['extinction']['coefficients'])
            fd.flush();vd.flush();flux,variance=fd,vd;local=True
        else:local=False
        y = np.lib.format.open_memmap(root/kind/'y.npy',mode='w+',dtype='f8',shape=(n,d))
        noise = np.lib.format.open_memmap(root/kind/'noise.npy',mode='w+',dtype='f8',shape=(n,d))
        obs = np.lib.format.open_memmap(root/kind/'observed.npy',mode='w+',dtype='?',shape=(n,d))
        for lo in range(0,n,2048):
            rr=rows[lo:lo+2048];ii=np.arange(lo,lo+len(rr)) if local else rr
            f=transform(Photometry(flux[ii],variance[ii],transform.bands))
            y[lo:lo+len(rr)]=f.x;noise[lo:lo+len(rr)]=np.diagonal(f.cov,axis1=1,axis2=2);obs[lo:lo+len(rr)]=f.observed
        y.flush();noise.flush();obs.flush()
        fit=np.isin(source['role'][rows],cfg['fit_roles']) & selected_cells(cells[rows],cfg['training_cells'])
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
            if kind=='qso' and 'qso_coordinates' in layout:
                qc=layout['qso_coordinates'];means,covs=to_magnitude_colour(init,np.array(qc['matrix']),qc['index'],qc)
                init=GaussianMixture(init.weights,means,covs,tuple(qc['labels']))
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
    keep=np.isin(data['role'],cfg['fit_roles'])&selected_cells(data['cell'],cfg['training_cells'])
    if task['kind']=='qso':keep&=(data['zspec']>=task['z']-cfg['z_half_width'])&(data['zspec']<task['z']+cfg['z_half_width'])
    return np.flatnonzero(keep)


def fit_task(task,cfg,root):
    root=Path(root);out=root/'fits'/(task['name']+'.json')
    if out.exists():return task['name']
    data=arrays(root,task['kind']);rr=task_rows(data,task,cfg)
    row_hash=hashlib.sha256(data['source_row'][rr].tobytes()).hexdigest()
    identity_hash=file_hash(root/'identity.json')
    if len(rr)!=task['n']:raise ValueError('prepared fit row count changed')
    layout=json.loads((root/'layout.json').read_text());op=operator(layout,task['kind'])
    def source():
        for lo in range(0,len(rr),cfg['batch_size']):
            ii=rr[lo:lo+cfg['batch_size']];v=data['noise'][ii];cov=np.zeros((len(ii),v.shape[1],v.shape[1]))
            cov[:,np.arange(v.shape[1]),np.arange(v.shape[1])]=v
            yield data['y'][ii],cov,data['observed'][ii],np.zeros(len(ii),int)
    start=time.monotonic();history=[];init=GaussianMixture.from_dict(task['init'])
    checkpoint_path=root/'progress'/(task['name']+'_checkpoint.json')
    if checkpoint_path.exists():
        saved=json.loads(checkpoint_path.read_text())
        if saved.get('identity_sha256')!=identity_hash or saved.get('fit_rows_sha256')!=row_hash:
            raise ValueError('checkpoint identity or selected rows mismatch')
        history=saved['history']
        if saved['iteration']!=len(history):raise ValueError('checkpoint history incomplete')
        init=GaussianMixture.from_dict(saved['mixture'])
    initial_history=tuple(history)
    def progress(iteration,mix,ll,n):
        history.append(ll)
        if iteration!=len(history):raise ValueError('checkpoint iteration mismatch')
        write_json(checkpoint_path,dict(mixture=mix.to_dict(),iteration=iteration,history=history,
            identity_sha256=identity_hash,fit_rows_sha256=row_hash,rows=n))
        write_json(root/'progress'/(task['name']+'.json'),dict(iteration=iteration,complete=False,
            rows=n,mean_loglike=ll,elapsed_seconds=time.monotonic()-start))
    context=nullcontext(None)
    if task['kind']=='stars' and cfg.get('stellar_workers',1)>1:
        from qso_pcolor.projected_parallel import ProjectedBatchFactory, ProjectedParallelAccumulator
        context=ProjectedParallelAccumulator(ProjectedBatchFactory(root/'stars',rr,cfg['batch_size']),
            len(rr),workers=cfg['stellar_workers'],task_rows=cfg['stellar_task_rows'],status_directory=root/'progress')
    qc=layout.get('qso_coordinates') if task['kind']=='qso' else None
    options=dict(operators={0:op},expected_rows=len(rr),tol=cfg['tol'],regularization=cfg['regularization'],
        progress=progress,covariance_update=cfg.get('covariance_update','additive'),
        prior_strength=cfg.get('prior_strength',1.0),
        fixed_coordinate=None if qc is None else dict(index=qc['index'],mean=qc['mean'],variance=qc['variance']))
    with context as accumulator:
        if not cfg.get('predictive_stopping'):
            fit=fit_projected(source,init=init,max_iter=cfg['max_iter'],initial_history=initial_history,
                accumulator=accumulator,**options)
            chosen,mean_loglike,extra=fit.mixture,fit.mean_loglike,{}
        else:
            chosen,mean_loglike,extra,fit=predictive_fit(task,cfg,root,data,layout,source,init,history,
                accumulator,options,identity_hash,row_hash)
    history=list(fit.history)
    write_json(out,dict(task=task['name'],mixture=chosen.to_dict(),n=len(rr),k=task['k'],
        n_iter=fit.n_iter,history=history,converged=fit.converged,mean_loglike=mean_loglike,
        elapsed_seconds=time.monotonic()-start,fit_rows_sha256=row_hash,
        resumed_iteration=len(initial_history),covariance_update=cfg.get('covariance_update','additive'),
        history_quantity='mean log posterior' if cfg.get('covariance_update')=='map' else 'mean log likelihood',
        # A MAP history is likelihood plus log prior; never append the plain likelihood to it.
        likelihood_decreased=bool(np.any(np.diff(history if cfg.get('covariance_update')=='map'
                                                 else history+[fit.mean_loglike])<0)),**extra))
    write_json(root/'progress'/(task['name']+'.json'),dict(iteration=fit.n_iter,complete=True,rows=len(rr),elapsed_seconds=time.monotonic()-start))
    return task['name']


def stopping_density(mix,layout,kind,data,panel,batch):
    """Mean held-out conditional log density per object (per native mag^(N-1)).

    ``panel`` holds prepared-row indices and the reference band fixed at
    preparation, so every checkpoint is scored on identical conditioning.
    """
    native=native_mixture(mix,*operator(layout,kind));rows,anchors=panel['rows'],panel['anchors']
    out=np.empty(len(rows))
    for lo in range(0,len(rows),batch):
        ix=np.arange(lo,min(lo+batch,len(rows)));rr=rows[ix];v=data['noise'][rr]
        cov=np.zeros((len(rr),v.shape[1],v.shape[1]));cov[:,np.arange(v.shape[1]),np.arange(v.shape[1])]=v
        for a in np.unique(anchors[ix]):
            t=anchors[ix]==a
            out[ix[t]]=conditional_log_prob(native,data['y'][rr[t]],cov[t],data['observed'][rr[t]],int(a))
    if not np.isfinite(out).all():raise ValueError('nonfinite stopping density')
    return float(out.mean())


def predictive_fit(task,cfg,root,data,layout,source,init,history,accumulator,options,identity_hash,row_hash):
    """Fit in blocks; stop on a held-out plateau and keep the best checkpoint.

    Every ``block_iterations`` updates (and at the start, the end and on
    objective convergence) the model is scored on a fixed non-training stopping
    panel. Fitting stops when a block gains less than ``min_gain`` nats/object,
    the objective tolerance is met, or ``max_iter`` is reached. The best scored
    checkpoint, including the warm start, is returned. This is predictive early
    stopping, not convergence; the panel must be excluded from final assessment.
    """
    rule=cfg['predictive_stopping'];block=rule['block_iterations'];name=task['name']
    panel=dict(np.load(root/'stopping'/(name+'.npz')))
    trace_path=root/'progress'/(name+'_stopping.json');best_path=root/'progress'/(name+'_best.json')
    trace=(json.loads(trace_path.read_text()) if trace_path.exists() else
           dict(evaluations=[],stop_reason=None,identity_sha256=identity_hash,fit_rows_sha256=row_hash))
    if trace['identity_sha256']!=identity_hash or trace['fit_rows_sha256']!=row_hash:
        raise ValueError('stopping trace identity or rows mismatch')
    mix,it,converged=init,len(history),False
    if not trace['evaluations'] and it:raise ValueError('stopping trace missing for resumed fit')
    fit=None
    while trace['stop_reason'] is None:
        last=trace['evaluations'][-1] if trace['evaluations'] else None
        # Score only at block boundaries, the end, or convergence, so a resume
        # from a mid-block checkpoint follows the uninterrupted schedule.
        due=it%block==0 or it>=cfg['max_iter'] or converged
        if last is None or (last['iteration']!=it and due):
            value=stopping_density(mix,layout,task['kind'],data,panel,cfg['batch_size'])
            trace['evaluations'].append(dict(iteration=it,mean=value))
            if last is None or value>max(e['mean'] for e in trace['evaluations'][:-1]):
                write_json(best_path,dict(iteration=it,mean=value,mixture=mix.to_dict()))
            if converged:trace['stop_reason']='objective_tolerance'
            elif last is not None and value-last['mean']<rule['min_gain']:trace['stop_reason']='predictive_plateau'
            elif it>=cfg['max_iter']:trace['stop_reason']='iteration_limit'
            write_json(trace_path,trace)
            if trace['stop_reason']:break
        # Pause only here: model, history and evaluation are all saved, so a
        # resume continues exactly as an uninterrupted run would.
        if (root/'PAUSE').exists():raise FitPaused(name)
        elif converged and last['iteration']==it:
            trace['stop_reason']='objective_tolerance';write_json(trace_path,trace);break
        fit=fit_projected(source,init=mix,max_iter=min((it//block+1)*block,cfg['max_iter']),
            initial_history=tuple(history),accumulator=accumulator,final_evaluation=False,**options)
        mix,it,converged=fit.mixture,fit.n_iter,fit.converged
    if fit is None:
        # Resumed after the decision was recorded: rebuild the returned record.
        fit=XDFitResult(mix,it,float('nan'),trace['stop_reason']=='objective_tolerance',list(history))
    best=json.loads(best_path.read_text());chosen=GaussianMixture.from_dict(best['mixture'])
    stats=(accumulator or accumulate_projected)(source,chosen,options['operators'])
    if stats[-1]!=options['expected_rows']:raise ValueError('row accounting in final evaluation')
    extra=dict(selected_iteration=best['iteration'],selected_stopping_density=best['mean'],
        stop_reason=trace['stop_reason'],stopping_evaluations=trace['evaluations'],
        stopping_rule=rule,stopping_rows=int(len(panel['rows'])),
        interpretation='Predictive early stopping on a fixed non-training panel; not optimizer convergence.')
    return chosen,float(stats[3]/stats[4]),extra,fit


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
