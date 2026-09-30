#!/usr/bin/env python
"""Bounded full-row test of the MAP covariance update against the additive update.

Both updates start from identical canonical warm starts and use every fit/select
row of the selected QSO slices. Fitting goes through the production
``fit_projected`` in restartable blocks; the wall budget is checked between
blocks. Nothing here changes a production configuration or the active model.

Usage::

    python scripts/test_map_update.py            # prepare and verify only
    python scripts/test_map_update.py --fit      # run the bounded test
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.stats import mannwhitneyu

from qso_pcolor import PSFMultiSurveyBaseline
from qso_pcolor.full_sample import file_hash, write_json
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.legacy import is_north
from qso_pcolor.projected_parallel import ProjectedBatchFactory
from qso_pcolor.projected_xd import fit_projected
from qso_pcolor.sky_acquisition import acquisition_lock
from qso_pcolor.unified import native_view, operator
from run_unified_pilot import arrays, task_rows
from trial_unified_covariance import (conditional_values, digest_rows, raw_rank,
                                      read, select_development, shift_summary)
from validate_full_sample_release import run_scores


def prepare(options):
    """Fix rows, development panels and identity; never fits."""
    source = Path(options['prepared_run']); cfg = read(source/'config.json')
    if cfg['fit_roles'] != [0, 1] or cfg['training_cells'] is not None:
        raise ValueError('test requires the uncapped full-footprint preparation')
    code = [Path(__file__), Path('scripts/trial_unified_covariance.py'),
            Path('src/qso_pcolor/projected_xd.py'), Path('src/qso_pcolor/unified.py')]
    identity = dict(options=options, prepared=file_hash(source/'identity.json'),
                    tasks=file_hash(source/'tasks.json'), layout=file_hash(source/'layout.json'),
                    nuisance_manifest=file_hash(Path(options['nuisance_bundle'])/'manifest.json'),
                    code={str(p): file_hash(p) for p in code})
    tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(options['output'])/tag; root.mkdir(parents=True, exist_ok=True)
    if (root/'prepared.json').exists():
        if read(root/'identity.json') != identity:
            raise ValueError('test identity changed')
        return root
    tasks = [t for t in read(source/'tasks.json') if t['name'] in options['cases']]
    if {t['name'] for t in tasks} != set(options['cases']):
        raise ValueError('missing selected task')
    write_json(root/'identity.json', identity); write_json(root/'options.json', options)
    write_json(root/'tasks.json', tasks)
    previous = Path(options['previous_trial']); data = arrays(source, 'qso')
    new_exclusions, reports = [], {}
    for task in tasks:
        train = task_rows(data, task, cfg)
        if len(train) != task['n']:
            raise ValueError('training row accounting changed')
        old = previous/(task['name']+'_rows.npz')
        if old.exists():
            # Reuse the stopped trial's panel: already recorded as excluded.
            with np.load(old) as saved:
                if digest_rows(saved['train']) != digest_rows(train):
                    raise ValueError('previous trial rows differ: '+task['name'])
                selected = dict(density=saved['density'], score=saved['score'])
            origin = str(old)
        else:
            selected = select_development(data, task, cfg, options)
            selected.pop('random')
            new_exclusions.append(data['source_row'][np.unique(np.concatenate(list(selected.values())))])
            origin = 'new role-3 selection'
        used = np.unique(np.concatenate(list(selected.values())))
        if np.intersect1d(train, used).size or not (data['role'][used] == 3).all():
            raise ValueError('development leakage')
        np.savez_compressed(root/(task['name']+'_rows.npz'), train=train, **selected)
        reports[task['name']] = dict(role=options['case_roles'][task['name']], z=task['z'], k=task['k'],
            train_rows=len(train), density_rows=len(selected['density']), score_rows=len(selected['score']),
            panel_origin=origin)
    with np.load(previous/'stars_00_rows.npz') as saved:
        stars = saved['score']
    np.save(root/'star_score_rows.npy', stars)
    qso_new = np.unique(np.concatenate(new_exclusions)) if new_exclusions else np.array([], np.int64)
    np.savez_compressed(root/'final_assessment_exclusions.npz', qso=qso_new, stars=np.array([], np.int64))
    write_json(root/'development_manifest.json', dict(inputs=cfg['inputs'],
        exclusions='final_assessment_exclusions.npz', sha256=file_hash(root/'final_assessment_exclusions.npz'),
        counts=dict(qso=len(qso_new), stars=0), also_excluded=str(previous/'final_assessment_exclusions.npz'),
        rule='New role-3 development source rows. Exclude from every future independent final assessment, '
             'together with the earlier trial exclusions. Star score rows are reused from that trial.'))
    write_json(root/'prepared.json', dict(tasks=reports, star_score_rows=len(stars),
        active_pointer_hash=file_hash(Path(cfg['active_pointer'])), source=str(source),
        production_changed=False))
    write_json(Path(options['output'])/'current.json', dict(directory=str(root)))
    return root


def fit_case(task, mode, root_string, deadline):
    """Fit in restartable blocks through the production code path."""
    root = Path(root_string); options = read(root/'options.json'); source = Path(options['prepared_run'])
    cfg = read(source/'config.json'); layout = read(source/'layout.json')
    dest = root/f"{task['name']}_{mode}"; dest.mkdir(exist_ok=True); state_path = dest/'state.json'
    data = arrays(source, 'qso')
    with np.load(root/(task['name']+'_rows.npz')) as saved:
        train, dev = saved['train'], saved['density']
    identity = file_hash(root/'identity.json'); row_hash = digest_rows(data['source_row'][train])
    state = read(state_path) if state_path.exists() else dict(iteration=0, history=[], mixture=task['init'],
        converged=False, identity=identity, rows=row_hash, blocks=[])
    if state['identity'] != identity or state['rows'] != row_hash:
        raise ValueError('state identity mismatch')
    factory = ProjectedBatchFactory(source/'qso', train, cfg['batch_size'])
    ops = {0: operator(layout, 'qso')}
    model = PSFMultiSurveyBaseline.load(options['nuisance_bundle']).model

    def record(mix, iteration):
        values = conditional_values(mix, layout, 'qso', data, dev, model)
        np.savez_compressed(dest/f'prediction_{iteration:03d}.npz', values=values, rows=dev)
        write_json(dest/f'model_{iteration:03d}.json', mix.to_dict())

    if state['iteration'] == 0 and not (dest/'prediction_000.npz').exists():
        record(GaussianMixture.from_dict(task['init']), 0)
    while not state['converged'] and state['iteration'] < options['max_iter']:
        if time.time() >= deadline:
            state['stopped_by_wall_budget'] = True; break
        start = time.monotonic()
        target = min(state['iteration']+options['block_iterations'], options['max_iter'])
        fit = fit_projected(lambda: factory(0, len(train)), init=GaussianMixture.from_dict(state['mixture']),
            operators=ops, expected_rows=len(train), max_iter=target, tol=options['tolerance'],
            regularization=options['regularization'], initial_history=tuple(state['history']),
            covariance_update=mode, prior_strength=options['prior_strength'])
        state.update(iteration=fit.n_iter, history=list(fit.history), mixture=fit.mixture.to_dict(),
                     converged=bool(fit.converged), final_mean_loglike=fit.mean_loglike)
        state['blocks'].append(dict(iteration=fit.n_iter, seconds=time.monotonic()-start,
                                    mean_loglike=fit.mean_loglike))
        record(fit.mixture, fit.n_iter)
        write_json(state_path, state)
    write_json(state_path, state)
    return dict(task=task['name'], mode=mode, iteration=state['iteration'], converged=state['converged'])


def final_model(root, name, mode):
    state = read(root/f'{name}_{mode}'/'state.json')
    return GaussianMixture.from_dict(state['mixture']), state


def evaluate(root):
    """Development density and complete-score ranking for the final models."""
    options = read(root/'options.json'); source = Path(options['prepared_run']); tasks = read(root/'tasks.json')
    layout = read(source/'layout.json'); data = arrays(source, 'qso'); stars = arrays(source, 'stars')
    star_rows = np.load(root/'star_score_rows.npy'); acceptance = options['acceptance']
    cfg = read('configs/full_sample_release.json'); cfg['batch_size'] = 32
    report = dict(cases={}, ranking={}, acceptance=acceptance)
    scores_dir = root/'scores'; scores_dir.mkdir(exist_ok=True)
    bases = {}
    for mode in options['updates']:
        base = deepcopy(PSFMultiSurveyBaseline.load(options['nuisance_bundle']))
        for task in tasks:
            base.model.qso.mixtures[task['index']] = native_view(final_model(root, task['name'], mode)[0], layout, 'qso')
        base.model.meta['diagnostic_only'] = True; bases[mode] = base
    for task in tasks:
        name = task['name']; entry = dict(role=options['case_roles'][name], n=task['n'])
        with np.load(root/(name+'_rows.npz')) as saved:
            dev, score_rows = saved['density'], saved['score']
        north = is_north(data['ra'][dev], data['dec'][dev], data['b'][dev])
        values = {}
        for mode in options['updates']:
            _, state = final_model(root, name, mode)
            h = np.array(state['history']); steps = np.diff(h)
            values[mode] = np.load(root/f'{name}_{mode}'/f"prediction_{state['iteration']:03d}.npz")['values']
            start = np.load(root/f'{name}_{mode}'/'prediction_000.npz')['values']
            decline = -options['roundoff_tolerance']*np.maximum(1., abs(h[:-1])) if len(h) > 1 else np.array([])
            entry[mode] = dict(iterations=state['iteration'], converged=state['converged'],
                stopped_by_wall_budget=state.get('stopped_by_wall_budget', False),
                history_quantity='mean log posterior' if mode == 'map' else 'mean log likelihood',
                history_declines=int((steps < decline).sum()),
                largest_decline=float(-steps.min()) if len(steps) and steps.min() < 0 else 0.,
                final_mean_training_loglike=state.get('final_mean_loglike'),
                development_mean=float(values[mode].mean()), north=float(values[mode][north].mean()),
                south=float(values[mode][~north].mean()),
                change_from_warm_start=shift_summary(start, values[mode]),
                seconds=float(sum(b['seconds'] for b in state['blocks'])))
        entry['map_minus_additive'] = shift_summary(values['additive'], values['map'])
        entry['map_minus_additive_by_hemisphere'] = dict(
            north=float((values['map']-values['additive'])[north].mean()),
            south=float((values['map']-values['additive'])[~north].mean()))
        report['cases'][name] = entry
        qpanel = dict(flux=data['flux'][score_rows], variance=data['variance'][score_rows],
            bands=bases['map'].model.transform.bands, l=data['l'][score_rows], b=data['b'][score_rows],
            zprimary=data['zspec'][score_rows])
        spanel = dict(flux=stars['flux'][star_rows], variance=stars['variance'][star_rows],
            bands=bases['map'].model.transform.bands, l=stars['l'][star_rows], b=stars['b'][star_rows],
            zprimary=np.full(len(star_rows), task['z']))
        for mode in options['updates']:
            q = raw_rank(run_scores(bases[mode], qpanel, cfg, scores_dir/f'{mode}_{name}_qso.npz'))
            b = raw_rank(run_scores(bases[mode], spanel, cfg, scores_dir/f'{mode}_{name}_stars.npz'))
            if np.isnan(q).any() or np.isnan(b).any():
                raise ValueError('undefined raw scores in ranking panel')
            report['ranking'].setdefault(name, {})[mode] = float(mannwhitneyu(q, b).statistic/(len(q)*len(b)))
    cases = report['cases'].values()
    report['checks'] = dict(
        map_no_objective_declines=all(e['map']['history_declines'] == 0 for e in cases),
        map_reached_tolerance={n: e['map']['converged'] for n, e in report['cases'].items()},
        map_density_within_tolerance=all(e['map_minus_additive']['mean'] >= -acceptance['maximum_mean_density_loss'] for e in cases),
        map_auc_within_tolerance=all(v['map'] >= v['additive']-acceptance['maximum_auc_loss'] for v in report['ranking'].values()))
    report['passed'] = bool(report['checks']['map_no_objective_declines'] and all(report['checks']['map_reached_tolerance'].values())
                            and report['checks']['map_density_within_tolerance'] and report['checks']['map_auc_within_tolerance'])
    report['interpretation'] = ('Development comparison on role-3 rows with fixed pilot nuisance terms; other slices and the '
        'background stay at canonical warm starts. Tests the update, not a completed bundle or calibration.')
    write_json(root/'report.json', report)
    return report


def run(root):
    options = read(root/'options.json'); tasks = read(root/'tasks.json'); start = time.time()
    deadline = start + options['wall_budget_seconds']; cfg = read(Path(options['prepared_run'])/'config.json')
    with acquisition_lock(root/'run.lock'):
        write_json(root/'execution.json', dict(pid=os.getpid(), state='running', start_unix=start, deadline_unix=deadline))
        try:
            jobs = sorted(((t, m) for t in tasks for m in options['updates']), key=lambda x: -x[0]['n']*x[0]['k'])
            with ProcessPoolExecutor(max_workers=options['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
                pending = {pool.submit(fit_case, t, m, str(root), deadline) for t, m in jobs}
                while pending:
                    done, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
                    for future in done:
                        print('FIT DONE', json.dumps(future.result()), flush=True)
                    status = {}
                    for t, m in jobs:
                        p = root/f"{t['name']}_{m}"/'state.json'
                        status[f"{t['name']}_{m}"] = read(p)['iteration'] if p.exists() else 0
                    print('PROGRESS', round((time.time()-start)/60, 1), 'min', json.dumps(status), flush=True)
            report = evaluate(root)
            if file_hash(Path(cfg['active_pointer'])) != read(root/'prepared.json')['active_pointer_hash']:
                raise ValueError('active pointer changed')
            write_json(root/'execution.json', dict(pid=os.getpid(), state='finished', passed=report['passed'],
                                                   elapsed_seconds=time.time()-start))
            print('FINISHED', root, 'passed', report['passed'], json.dumps(report['checks']), flush=True)
        except BaseException as error:
            write_json(root/'execution.json', dict(pid=os.getpid(), state='failed', error=repr(error)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/map_update_test.json')
    parser.add_argument('--fit', action='store_true', help='run the bounded test; default only prepares')
    args = parser.parse_args(); root = prepare(read(args.config)); print('PREPARED', root, flush=True)
    if args.fit:
        run(root)


if __name__ == '__main__':
    main()
