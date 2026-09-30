#!/usr/bin/env python
"""Queued, bounded constrained-update trial on held slices; never promote."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from qso_pcolor.covariance_diagnostic import constrained_step
from qso_pcolor.convergence_review import checked_rows, verify_parent_identity, density_probe, compare_probes
from qso_pcolor.full_sample import TrainingRows, write_json, file_hash
from qso_pcolor.full_training import load_inputs, make_source, slice_ranges
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.sky_acquisition import acquisition_lock


def trial_slice(name, zr, root, source, cfg, options):
    """Resume only this diagnostic's history; every update uses original full rows."""
    parent=Path(options['parent']);out=Path(options['output'])
    record=json.loads((parent/(name+'.json')).read_text())
    data=TrainingRows(root/'qso',source.transform.bands)
    rows=checked_rows(data,zr,cfg,record)
    original=GaussianMixture.from_dict(record['mixture']);mix=original
    batches=make_source(data,rows,source,cfg)
    checkpoint=out/(name+'.checkpoint.json');history=[]
    if checkpoint.exists():
        previous=json.loads(checkpoint.read_text())
        if previous['parent_sha256']!=file_hash(parent/(name+'.json')):
            raise ValueError('trial parent changed')
        mix=GaussianMixture.from_dict(previous['mixture']);history=previous['history']
    status=out/(name+'.status.json')
    write_json(status,dict(state='running',pid=os.getpid(),iteration=len(history),rows=len(rows),k=mix.n_components))
    for iteration in range(len(history)+1,options['iterations']+1):
        candidate,metrics=constrained_step(batches,mix,floor=cfg['regularization'],expected_rows=len(rows))
        if metrics['change'] < -options['roundoff_tolerance']:
            write_json(status,dict(state='failed_monotonicity',iteration=iteration,metrics=metrics))
            raise ValueError(f'{name}: constrained trial decreased likelihood')
        mix=candidate;history.append(metrics)
        write_json(checkpoint,dict(parent_sha256=file_hash(parent/(name+'.json')),
            parent_iteration=record['n_iter'],iteration=iteration,history=history,mixture=mix.to_dict(),
            covariance_update='eigenvalue_floor',floor=cfg['regularization'],diagnostic_only=True))
        write_json(status,dict(state='running',pid=os.getpid(),iteration=iteration,rows=len(rows),k=mix.n_components))
        print(name,'trial iteration',iteration,'gain',metrics['change'],flush=True)
    seed=options['diagnostic_seed']+int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
    sample=np.sort(np.random.default_rng(seed).choice(rows,min(len(rows),options['diagnostic_rows']),replace=False))
    before=density_probe(data,sample,source,cfg,original)
    after=density_probe(data,sample,source,cfg,mix)
    np.savez(out/(name+'.probe.npz'),rows=sample,**{'before_'+k:v for k,v in before.items()},
             **{'after_'+k:v for k,v in after.items()})
    comparison=compare_probes(before,after,options)
    result=dict(name=name,rows=len(rows),k=mix.n_components,iterations=len(history),history=history,
        total_gain=history[-1]['after']-history[0]['before'],prediction_change=comparison,
        meets_original_gain_tolerance=bool(abs(history[-1]['change']) < cfg['tol']*max(1.,abs(history[-1]['before']))),
        release_ready=False,diagnostic_only=True)
    write_json(out/(name+'.json'),result)
    write_json(status,dict(state='completed',pid=os.getpid(),iteration=len(history),rows=len(rows),k=mix.n_components))
    return result


def run_trial(options):
    if options['iterations'] < 1 or options['workers'] < 1 or options['roundoff_tolerance'] < 0:
        raise ValueError('invalid trial budget, workers or roundoff tolerance')
    out=Path(options['output']);out.mkdir(parents=True,exist_ok=True)
    execution=dict(pid=os.getpid(),state='waiting_for_stellar',workers=options['workers'],diagnostic_only=True)
    execution_path=out/'execution.json'
    with acquisition_lock(out/'worker.lock'), ExitStack() as locks:
        pinned_paths=('scripts/trial_covariance_floor.py','src/qso_pcolor/covariance_diagnostic.py',
            'src/qso_pcolor/convergence_review.py', options['training_config'], options['audit'])
        pinned={p:file_hash(Path(p)) for p in pinned_paths}
        queued=dict(options=options,pinned_files=pinned)
        queued_path=out/'queued_identity.json'
        if queued_path.exists() and json.loads(queued_path.read_text())!=queued:
            raise ValueError('queued diagnostic implementation or settings changed')
        write_json(queued_path,queued)
        write_json(execution_path,execution)
        try:
            dependency=Path(options['wait_for_execution'])
            while True:
                status=json.loads(dependency.read_text())
                if status['state']=='completed':break
                if status['state']!='running':raise RuntimeError('prior continuation did not finish normally')
                os.kill(status['pid'],0)
                time.sleep(30)
            if any(file_hash(Path(p))!=h for p,h in pinned.items()):
                raise ValueError('diagnostic files changed while queued')
            cfg=json.loads(Path(options['training_config']).read_text())
            parent=Path(options['parent'])
            locks.enter_context(acquisition_lock(parent/'worker.lock'))
            root,_,source,identity=load_inputs(cfg)
            verify_parent_identity(json.loads((parent/'identity.json').read_text()),identity)
            provenance=dict(options=options,training_identity=identity,
                parent_hashes={p.name:file_hash(p) for p in parent.glob('*.json')},
                engine_hashes={p:file_hash(Path(p)) for p in
                    ('scripts/trial_covariance_floor.py','src/qso_pcolor/covariance_diagnostic.py','src/qso_pcolor/convergence_review.py')})
            lineage=out/'lineage.json'
            if lineage.exists() and json.loads(lineage.read_text())!=provenance:
                raise ValueError('trial lineage changed')
            write_json(lineage,provenance)
            audit=json.loads(Path(options['audit']).read_text())
            names=[n for n,r in audit['fits'].items() if n.startswith('qso_') and r['action']=='hold_for_diagnosis']
            ranges=slice_ranges(source,cfg['z_step'])
            execution.update(state='running',slices=names);write_json(execution_path,execution)
            results=[]
            with ProcessPoolExecutor(max_workers=options['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
                jobs=[pool.submit(trial_slice,n,ranges[int(n.split('_')[1])],root,source,cfg,options) for n in names]
                for future in as_completed(jobs):results.append(future.result())
            write_json(out/'results.json',dict(results=results,diagnostic_only=True,release_ready=False))
            lines=['# Covariance-floor trial results','','All rows are original fit+selection rows. This is not held-out validation.',
                '', '| Slice | Iterations | Likelihood gain/object | Last gain | Meets original tolerance |',
                '|---|---:|---:|---:|---|']
            for r in sorted(results,key=lambda x:x['name']):
                lines.append(f"| {r['name']} | {r['iterations']} | {r['total_gain']:.6g} | {r['history'][-1]['change']:.6g} | {r['meets_original_gain_tolerance']} |")
            lines+=['','Next: review the fixed-row prediction comparisons; validate model selection on fit-only',
                'checkpoints and frozen selection rows before adopting this covariance update throughout training.',
                'Do not promote these diagnostic fits or treat reaching 20 iterations as convergence.']
            (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
            if any(file_hash(parent/n)!=h for n,h in provenance['parent_hashes'].items()):
                raise ValueError('parent changed during trial')
            if file_hash(Path(cfg['source_pointer']))!=identity['source_pointer_sha256']:
                raise ValueError('active pointer changed during trial')
            execution['state']='completed'
        except BaseException as error:
            execution.update(state='failed',error=repr(error));raise
        finally:write_json(execution_path,execution)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/covariance_floor_trial.json'))
    parser.add_argument('--fit',action='store_true')
    args=parser.parse_args();options=json.loads(args.config.read_text())
    if not args.fit:print(json.dumps(options,indent=2));return
    run_trial(options)


if __name__=='__main__':main()
