#!/usr/bin/env python
"""Prepare, or explicitly fit, the uncapped unified model with separate worker pools."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing
import os
from pathlib import Path
import time

# One numerical thread per process: four QSO plus four stellar workers.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from qso_pcolor.full_sample import file_hash, write_json
from qso_pcolor.sky_acquisition import acquisition_lock
from run_unified_pilot import prepare, fit_task, arrays


def validate_config(cfg):
    if cfg['training_cells'] is not None or cfg['test_cells'] != [] or cfg['fit_roles'] != [0, 1]:
        raise ValueError('full refit requires all sky cells and exactly frozen fit/select roles')
    if cfg.get('pilot', True):
        raise ValueError('full refit must explicitly disable pilot scope')
    for key in ('qso_workers', 'stellar_workers', 'stellar_task_rows', 'batch_size', 'max_iter'):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')


def prepare_full(cfg):
    """Verify frozen local data and prepare all rows; never start a numerical fit."""
    validate_config(cfg)
    cfg = dict(cfg)
    code = [Path('src/qso_pcolor') / name for name in (
        'projected_xd.py', 'projected_parallel.py', 'unified.py', 'gaussmix.py',
        'xd.py', 'multisurvey.py', 'multisurvey_data.py', 'full_sample.py')]
    code += [Path(__file__), Path(__file__).with_name('run_unified_pilot.py')]
    cfg['execution_hashes'] = {str(p): file_hash(p) for p in code}
    inputs = Path(cfg['inputs'])
    manifest = json.loads((inputs/'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if file_hash(inputs/name) != digest:
            raise ValueError('frozen input changed: '+name)
    output = Path(cfg['output']); output.mkdir(parents=True, exist_ok=True)
    with acquisition_lock(output/'prepare.lock'):
        root = prepare(cfg)
        counts = json.loads((root/'prepared.json').read_text())['counts']
        tasks = json.loads((root/'tasks.json').read_text())
        for kind in ('qso', 'stars'):
            source = arrays(inputs, kind)
            expected = int((source['eligible'] & np.isin(source['role'], cfg['fit_roles'])).sum())
            if counts[kind]['fit_rows'] != expected:
                raise ValueError('full-data accounting mismatch: '+kind)
        qso = arrays(root, 'qso')
        fitting = np.isin(qso['role'], cfg['fit_roles'])
        covered = np.zeros(len(fitting), bool)
        for task in tasks:
            if task['kind'] == 'qso':
                covered |= (qso['zspec'] >= task['z']-cfg['z_half_width']) & (qso['zspec'] < task['z']+cfg['z_half_width'])
        if (fitting & ~covered).any():
            raise ValueError('eligible QSO training rows fall outside configured slices')
        write_json(root/'preflight.json', dict(ready=True, verified_input_files=len(manifest['files']),
            counts=counts, qso_workers=cfg['qso_workers'], stellar_workers=cfg['stellar_workers'],
            numerical_workers=cfg['qso_workers']+cfg['stellar_workers'], threads_per_worker=1,
            full_sky=True, no_snr_training_cut=True, no_row_caps=True, max_iter=cfg['max_iter'],
            tolerance=cfg['tol'], active_model_changed=False, fits_launched=False))
    return root, cfg


def progress_state(root, tasks, cfg, start):
    total = sum(t['n']*t['k']*cfg['max_iter'] for t in tasks)
    work = actual = complete = 0
    by_kind = {}
    for task in tasks:
        p = root/'progress'/(task['name']+'.json')
        state = json.loads(p.read_text()) if p.exists() else dict(iteration=0, complete=False)
        done = (root/'fits'/(task['name']+'.json')).exists()
        complete += done
        actual += task['n']*task['k']*state['iteration']
        work += task['n']*task['k']*(cfg['max_iter'] if done else state['iteration'])
        by_kind.setdefault(task['kind'], dict(completed=0, total=0))
        by_kind[task['kind']]['completed'] += done; by_kind[task['kind']]['total'] += 1
    return dict(completed_fits=complete, total_fits=len(tasks), populations=by_kind,
        budget_progress_fraction=work/total, executed_fraction_of_maximum_budget=actual/total,
        elapsed_seconds=time.monotonic()-start,
        scope='Row x component x iteration estimate, not measured remaining wall time; excludes completion/validation.')


def run_fits(root, cfg):
    """Four independent QSO fits; four cooperating workers for one stellar fit."""
    tasks = json.loads((root/'tasks.json').read_text())
    with acquisition_lock(root/'fit.lock'):
        start = time.monotonic()
        record = dict(state='running', coordinator_pid=os.getpid(), qso_workers=cfg['qso_workers'],
            stellar_workers=cfg['stellar_workers'], threads_per_worker=1,
            note='Stellar coordinator is a thread; only its E-step pool performs parallel stellar work.')
        write_json(root/'execution.json', record)
        qpool = ProcessPoolExecutor(max_workers=cfg['qso_workers'], mp_context=multiprocessing.get_context('spawn'))
        coordinator = ThreadPoolExecutor(max_workers=1)
        pending = {}
        try:
            for task in sorted(tasks, key=lambda t: t['n']*t['k'], reverse=True):
                if (root/'fits'/(task['name']+'.json')).exists():
                    continue
                pool = qpool if task['kind'] == 'qso' else coordinator
                pending[pool.submit(fit_task, task, cfg, str(root))] = task
            while pending:
                done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    print('FIT COMPLETE', future.result(), flush=True)
                    del pending[future]
                state = progress_state(root, tasks, cfg, start)
                write_json(root/'progress.json', state); print('PROGRESS', json.dumps(state), flush=True)
        except BaseException as error:
            for future in pending:
                future.cancel()
            record.update(state='failed', error=repr(error)); write_json(root/'execution.json', record)
            raise
        finally:
            qpool.shutdown(wait=True, cancel_futures=True)
            coordinator.shutdown(wait=True, cancel_futures=True)
        if file_hash(Path(cfg['active_pointer'])) != json.loads((root/'prepared.json').read_text())['active_pointer_hash']:
            raise ValueError('active pointer changed during candidate training')
        health = {}
        for task in tasks:
            fit = json.loads((root/'fits'/(task['name']+'.json')).read_text())
            health[task['name']] = {k:fit[k] for k in ('n_iter', 'converged', 'likelihood_decreased', 'mean_loglike')}
        write_json(root/'training_complete.json', dict(tasks=len(tasks),health=health,active_model_changed=False))
        record['state'] = 'completed'; write_json(root/'execution.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/unified_full_training.json')
    parser.add_argument('--fit', action='store_true', help='Explicitly launch production fits; default only prepares/verifies')
    args = parser.parse_args()
    root, cfg = prepare_full(json.loads(Path(args.config).read_text()))
    print('PREFLIGHT READY', root, flush=True)
    if args.fit:
        run_fits(root, cfg)


if __name__ == '__main__':
    main()
