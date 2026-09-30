#!/usr/bin/env python
"""Prepare and fit an isolated, uncapped pooled-versus-separate grz experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from qso_pcolor.full_sample import file_hash, write_json
from qso_pcolor.gaussmix import GaussianMixture, log_gauss_batch
from qso_pcolor.multisurvey import MultiSurveyModel
from qso_pcolor.projected_xd import fit_projected, native_mixture


def load_arrays(path):
    return {p.stem: np.load(p, mmap_mode='r') for p in Path(path).glob('*.npy')}


def diagonal(variance):
    out = np.zeros((len(variance), variance.shape[1], variance.shape[1]))
    out[:, np.arange(variance.shape[1]), np.arange(variance.shape[1])] = variance
    return out


def marginal(mix, indices):
    return GaussianMixture(mix.weights, mix.means[:, indices], mix.covs[:, indices][:, :, indices], ('g', 'r', 'z'))


def prepare_arrays(cfg, model, out):
    """Retain every quality-eligible source with any usable native optical band."""
    report = {}
    for kind in ('qso', 'stars'):
        dest = out / kind; dest.mkdir(exist_ok=True)
        raw = load_arrays(Path(cfg['inputs']) / kind)
        indices = np.array([[model.transform.bands.index(f'decals_dr9_{s}:{b}') for b in 'grz'] for s in ('south', 'north')])
        flux = np.asarray(raw['flux'][:, indices]); var = np.asarray(raw['variance'][:, indices])
        obs = np.isfinite(flux) & np.isfinite(var) & (var > 0)
        has = obs.any(axis=2)
        if has.all(axis=1).any():
            raise ValueError('input has both optical systems; do not silently double-count it')
        rows = np.flatnonzero(raw['eligible'] & has.any(axis=1))
        system = has[rows, 1].astype('u1'); pick = np.arange(len(rows))
        f, v, o = flux[rows][pick, system], var[rows][pick, system], obs[rows][pick, system]
        soft = model.transform.softening[indices][system]
        factor = 2.5/np.log(10)
        y = 22.5-factor*(np.arcsinh(np.where(o, f, 0)/(2*soft))+np.log(soft))
        variance = np.where(o, v, 0)*factor**2/(np.where(o, f, 0)**2+4*soft**2)
        arrays = dict(y=np.where(o, y, np.nan), variance=variance, observed=o,
                      system=system, source_row=rows, negative=(o & (f < 0)).any(axis=1))
        arrays.update({k: raw[k][rows] for k in ('role', 'zspec', 'l', 'b', 'field')})
        for k, a in arrays.items(): np.save(dest/(k+'.npy'), a)
        report[kind] = dict(total=len(rows), optical_unavailable=int(len(raw['role'])-len(rows)),
            rows={str(r): {s: int(((arrays['role'] == r) & (system == j)).sum()) for j, s in enumerate(('south', 'north'))} for r in range(4)},
            negative_flux_rows=int(arrays['negative'].sum()), source_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest())
    return report


def calibrate(cfg, model, out):
    """Fit offsets and diagonal excess variance from fit/select pairs only."""
    data = np.load(cfg['overlap'])
    ix = np.array([[model.transform.bands.index(f'decals_dr9_{s}:{b}') for b in 'grz'] for s in ('north', 'south')])
    soft = model.transform.softening[ix]
    f, v, obs = data['flux'][:, :, :3], data['variance'][:, :, :3], data['observed'][:, :, :3]
    factor = 2.5/np.log(10)
    u = 22.5-factor*(np.arcsinh(f/(2*soft))+np.log(soft))
    variance = v*factor**2/(f*f+4*soft*soft)
    bright = obs.all(axis=(1, 2)) & (f/np.sqrt(v) > cfg['correction_min_snr']).all(axis=(1, 2))
    fit = bright & np.isin(data['role'], cfg['correction_fit_roles'])
    h = np.linalg.solve(np.array(cfg['desi_matrix']), np.eye(3))
    base_offset = -h @ np.array(cfg['desi_offset'])
    residual = u[:, 0] - u[:, 1] @ h.T - base_offset
    result = dict(matrix=h.tolist(), base_offset=base_offset.tolist(), populations={},
        scope='Fixed DESI slopes in native asinh coordinates; fit/select offsets and excess variance with both measurement covariances. Extra QSO variance may include variability; no redshift-dependent parameters.')
    for kind, klass in [('stars', ~np.isfinite(data['zspec'])), ('qso', np.isfinite(data['zspec']))]:
        take = fit & klass; r = residual[take]
        noise = diagonal(variance[take, 0]) + h @ diagonal(variance[take, 1]) @ h.T
        floor = cfg['correction_variance_floor']
        initial = np.r_[np.median(r, axis=0), np.log(np.maximum(np.var(r, axis=0)-np.mean(np.diagonal(noise, axis1=1, axis2=2), axis=0), floor))]
        def objective(p):
            return -log_gauss_batch(r, p[:3], noise+np.diag(np.exp(p[3:]))).sum()+.5*np.sum((p[:3]/cfg['correction_offset_prior_sigma'])**2)
        opt = minimize(objective, initial, method='L-BFGS-B', bounds=[(None, None)]*3+[(np.log(floor), None)]*3,
                       options={'maxiter': cfg['correction_optimizer_max_iter']})
        if not opt.success: raise RuntimeError(f'calibration failed: {kind}: {opt.message}')
        offset, extra = base_offset+opt.x[:3], np.diag(np.exp(opt.x[3:]))
        record = dict(n=int(take.sum()), fields=np.unique(data['cone'][take]).tolist(),
            fit_rows_sha256=hashlib.sha256(np.flatnonzero(take).tobytes()).hexdigest(),
            offset=offset.tolist(), residual_mean=opt.x[:3].tolist(), extra_covariance=extra.tolist(), optimizer=str(opt.message), heldout={})
        for label, subset in [('bright', bright), ('all_snr', obs.all(axis=(1, 2)))]:
            use = subset & klass & np.isin(data['role'], ['calib', 'test'])
            rr = residual[use]-opt.x[:3]
            record['heldout'][label] = dict(n=int(use.sum()), median=np.median(rr, axis=0).tolist(),
                nmad=(1.4826*np.median(abs(rr-np.median(rr, axis=0)), axis=0)).tolist())
        result['populations'][kind] = record
    write_json(out/'calibration.json', result)
    return result


def operators_for(calibration, kind, mode):
    identity = (np.eye(3), np.zeros(3), np.zeros((3, 3)))
    if mode != 'pooled': return {0: identity, 1: identity}
    pop = calibration['populations'][kind]
    return {0: identity, 1: (np.array(calibration['matrix']), np.array(pop['offset']), np.array(pop['extra_covariance']))}


def task_rows(data, task, cfg):
    keep = np.isin(data['role'], cfg['fit_roles'])
    if task['mode'] != 'pooled': keep &= data['system'] == int(task['mode'] == 'north')
    if task['kind'] == 'qso': keep &= (data['zspec'] >= task['z']-cfg['z_half_width']) & (data['zspec'] < task['z']+cfg['z_half_width'])
    return np.flatnonzero(keep)


def fit_task(task, cfg, root):
    out = Path(root); saved = out/'fits'/(task['name']+'.json')
    if saved.exists(): return task['name']
    data = load_arrays(out/task['kind']); rows = task_rows(data, task, cfg)
    calibration = json.loads((out/'calibration.json').read_text())
    operators = operators_for(calibration, task['kind'], task['mode'])
    init = GaussianMixture.from_dict(task['init'])
    if task['mode'] == 'north':
        init = native_mixture(init, *operators_for(calibration, task['kind'], 'pooled')[1], labels=init.labels)
    if not len(rows): raise ValueError(f'no training rows: {task["name"]}')
    def source():
        for start in range(0, len(rows), cfg['batch_size']):
            rr = rows[start:start+cfg['batch_size']]
            yield data['y'][rr], diagonal(data['variance'][rr]), data['observed'][rr], data['system'][rr]
    history = []; started = time.time()
    def progress(iteration, mix, ll, seen):
        history.append(ll)
        write_json(out/'progress'/(task['name']+'.json'), dict(task=task['name'], iteration=iteration,
            rows=seen, k=mix.n_components, mean_loglike=ll, elapsed_seconds=time.time()-started, complete=False))
    fit = fit_projected(source, init=init, operators=operators, expected_rows=len(rows),
        max_iter=cfg['max_iter'], tol=cfg['tol'], regularization=cfg['regularization'], progress=progress)
    record = dict(task=task['name'], kind=task['kind'], mode=task['mode'], z=task.get('z'),
        mixture=fit.mixture.to_dict(), history=fit.history, n_iter=fit.n_iter, converged=fit.converged,
        mean_loglike=fit.mean_loglike, n=len(rows), k=init.n_components,
        row_indices_sha256=hashlib.sha256(data['source_row'][rows].tobytes()).hexdigest(), elapsed_seconds=time.time()-started)
    write_json(saved, record)
    write_json(out/'progress'/(task['name']+'.json'), dict(task=task['name'], iteration=fit.n_iter,
        rows=len(rows), k=init.n_components, complete=True, elapsed_seconds=record['elapsed_seconds']))
    return task['name']


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--fit', action='store_true'); args = parser.parse_args()
    cfg = json.loads(Path('configs/pooled_optical_prototype.json').read_text())
    model = MultiSurveyModel.load(Path(cfg['parent'])/'model.json')
    identity = dict(config=cfg, parent=file_hash(Path(cfg['parent'])/'model.json'),
        inputs=file_hash(Path(cfg['inputs'])/'manifest.json'), overlap=file_hash(Path(cfg['overlap'])),
        code={p: file_hash(Path(p)) for p in ('scripts/run_pooled_optical_prototype.py', 'src/qso_pcolor/projected_xd.py')})
    tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    out = Path(cfg['output'])/tag; out.mkdir(parents=True, exist_ok=True)
    for folder in ('fits', 'progress'): (out/folder).mkdir(exist_ok=True)
    write_json(Path(cfg['output'])/'current.json', dict(directory=str(out)))
    if not (out/'prepared.json').exists():
        counts = prepare_arrays(cfg, model, out)
        calibrate(cfg, model, out)
        write_json(out/'prepared.json', dict(identity=identity, counts=counts))
    si = np.array([model.transform.bands.index('decals_dr9_south:'+b) for b in 'grz'])
    tasks = []
    for kind, mixtures in [('stars', [model.background]), ('qso', model.qso.mixtures)]:
        data = load_arrays(out/kind)
        for j, mix in enumerate(mixtures):
            for mode in ('pooled', 'north', 'south'):
                task = dict(name=f'{kind}_{j:02d}_{mode}', kind=kind, mode=mode, init=marginal(mix, si).to_dict())
                if kind == 'qso': task['z'] = float(model.qso.z_centres[j])
                task['n'] = len(task_rows(data, task, cfg)); task['k'] = mix.n_components
                tasks.append(task)
    write_json(out/'tasks.json', tasks)
    print('PREPARED', out, 'tasks', len(tasks), flush=True)
    print(json.dumps(json.loads((out/'prepared.json').read_text())['counts']), flush=True)
    if not args.fit: return
    # One coordinator per content-addressed experiment. Never overwrite a live run.
    from qso_pcolor.sky_acquisition import acquisition_lock
    with acquisition_lock(out/'fit.lock'):
        started = time.time(); budget = sum(t['n']*t['k']*cfg['max_iter'] for t in tasks)
        pending = [t for t in tasks if not (out/'fits'/(t['name']+'.json')).exists()]
        with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
            futures = {pool.submit(fit_task, t, cfg, str(out)): t['name'] for t in pending}
            while futures:
                done, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
                for f in done:
                    print('FIT COMPLETE', f.result(), flush=True); del futures[f]
                work = actual = completed = 0
                for t in tasks:
                    path = out/'progress'/(t['name']+'.json')
                    p = json.loads(path.read_text()) if path.exists() else dict(iteration=0, complete=False)
                    completed += p['complete']; actual += t['n']*t['k']*p['iteration']
                    work += t['n']*t['k']*(cfg['max_iter'] if p['complete'] else p['iteration'])
                state = dict(completed_fits=completed, total_fits=len(tasks), budget_progress_fraction=work/budget,
                    executed_fraction_of_maximum_budget=actual/budget, elapsed_seconds=time.time()-started,
                    scope='Density-fit row x component x iteration budget only; excludes validation. Early completion retires unused iteration budget.')
                write_json(out/'progress.json', state)
                print('PROGRESS', json.dumps(state), flush=True)
        write_json(out/'training_complete.json', dict(identity=identity, tasks=len(tasks), active_model_changed=False))
    print('TRAINING COMPLETE', out, flush=True)


if __name__ == '__main__': main()
