#!/usr/bin/env python
"""Bounded full-row covariance comparison; never modifies a production bundle."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import mannwhitneyu, spearmanr

from qso_pcolor import PSFMultiSurveyBaseline, Photometry
from qso_pcolor.full_sample import file_hash, write_json
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.legacy import is_north
from qso_pcolor.multisurvey import conditional_log_prob
from qso_pcolor.projected_parallel import ProjectedBatchFactory, ProjectedParallelAccumulator
from qso_pcolor.projected_xd import accumulate_projected
from qso_pcolor.sky_acquisition import acquisition_lock
from qso_pcolor.unified import native_view, operator
from probe_unified_convergence import update_from_statistics
from run_unified_pilot import arrays, task_rows
from validate_full_sample_release import run_scores
from validate_psf_catchalls import grid_data


def read(path):
    return json.loads(Path(path).read_text())


def digest_rows(rows):
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()


def select_development(data, task, cfg, options):
    """Select fixed non-training, non-calibration rows; return prepared-row IDs."""
    candidates = np.flatnonzero(data['role'] == 3)
    if task['kind'] == 'qso':
        z = data['zspec'][candidates]
        candidates = candidates[(z >= task['z']-cfg['z_half_width']) &
                                (z < task['z']+cfg['z_half_width'])]
    obs = data['observed'][candidates]
    with np.errstate(invalid='ignore', divide='ignore'):
        detected = (obs & (data['flux'][candidates] /
                    np.sqrt(data['variance'][candidates]) >= cfg['reference_min_snr'])).any(axis=1)
    candidates = candidates[detected & (obs.sum(axis=1) >= 2) &
                            (abs(data['b'][candidates]) >= cfg['min_abs_b_deg'])]
    seed = options['seed'] + int.from_bytes(hashlib.sha256(task['name'].encode()).digest()[:4], 'little')
    rng = np.random.default_rng(seed)
    north = is_north(data['ra'][candidates], data['dec'][candidates], data['b'][candidates])
    halves = [rng.permutation(candidates[north == h])[:options['development_rows_per_hemisphere']]
              for h in (False, True)]
    if any(len(r) == 0 for r in halves):
        raise ValueError('development panel lacks a hemisphere: '+task['name'])
    scores = np.concatenate([r[:options['score_rows_per_hemisphere']] for r in halves])
    # This unstratified draw estimates incidence in the sampled detectable PSF
    # background population. It is not an all-sky, area-weighted incidence rate.
    random_rows = (rng.permutation(candidates)[:options['random_background_score_rows']]
                   if task['kind'] == 'stars' else np.array([], dtype=int))
    return dict(density=np.concatenate(halves), score=scores, random=random_rows)


def prepare(options):
    source = Path(options['prepared_run']); cfg = read(source/'config.json')
    if cfg['fit_roles'] != [0, 1] or cfg['training_cells'] is not None:
        raise ValueError('trial requires the uncapped full-footprint preparation')
    code = [Path(__file__), Path('scripts/probe_unified_convergence.py'),
            Path('src/qso_pcolor/projected_parallel.py'), Path('src/qso_pcolor/projected_xd.py'),
            Path('src/qso_pcolor/covariance_diagnostic.py'), Path('src/qso_pcolor/unified.py')]
    identity = dict(options=options, prepared=file_hash(source/'identity.json'),
                    tasks=file_hash(source/'tasks.json'), layout=file_hash(source/'layout.json'),
                    nuisance_manifest=file_hash(Path(options['nuisance_bundle'])/'manifest.json'),
                    code={str(p): file_hash(p) for p in code})
    tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(options['output'])/tag; root.mkdir(parents=True, exist_ok=True)
    if (root/'prepared.json').exists():
        if read(root/'identity.json') != identity:
            raise ValueError('trial identity changed')
        return root
    tasks = [t for t in read(source/'tasks.json') if t['name'] in options['cases']]
    if {t['name'] for t in tasks} != set(options['cases']):
        raise ValueError('missing selected task')
    write_json(root/'identity.json', identity)
    write_json(root/'options.json', options); write_json(root/'tasks.json', tasks)
    exclusions = {}; reports = {}
    for task in tasks:
        data = arrays(source, task['kind']); train = task_rows(data, task, cfg)
        if len(train) != task['n']:
            raise ValueError('training row accounting changed')
        selected = select_development(data, task, cfg, options)
        used = np.unique(np.concatenate(list(selected.values())))
        if np.intersect1d(train, used).size or not (data['role'][used] == 3).all():
            raise ValueError('development leakage')
        np.savez_compressed(root/(task['name']+'_rows.npz'), train=train, **selected)
        exclusions.setdefault(task['kind'], []).append(data['source_row'][used])
        reports[task['name']] = dict(train_rows=len(train), train_source_hash=digest_rows(data['source_row'][train]),
            density_rows=len(selected['density']), score_rows=len(selected['score']), random_rows=len(selected['random']),
            development_role=3, training_roles=[0, 1], training_cut='quality eligibility only; no S/N cut')
        dev = selected['density']; obs = data['observed'][dev]
        reports[task['name']]['development_coverage'] = dict(
            band_counts=obs.sum(axis=0).tolist(), distinct_masks=len(np.unique(obs, axis=0)),
            observed_band_count_quantiles=np.quantile(obs.sum(axis=1), [0,.25,.5,.75,1]).tolist())
        for mode in options['updates']:
            (root/(task['name']+'_'+mode)).mkdir(exist_ok=True)
    exclusions = {kind: np.unique(np.concatenate(parts)) for kind, parts in exclusions.items()}
    np.savez_compressed(root/'final_assessment_exclusions.npz', **exclusions)
    # Source indices refer to the immutable full_training_inputs, not the
    # filtered/prepared array offsets. Calibration (role 2) remains unchanged.
    write_json(root/'development_manifest.json', dict(inputs=cfg['inputs'],
        exclusions='final_assessment_exclusions.npz', sha256=file_hash(root/'final_assessment_exclusions.npz'),
        counts={k:len(v) for k,v in exclusions.items()},
        rule='These source-row IDs are development data. Exclude from every future independent final assessment. Earlier inspected test subsets also remain development evidence. Frozen roles are not rewritten.'))
    write_json(root/'prepared.json', dict(tasks=reports, active_pointer_hash=file_hash(Path(cfg['active_pointer'])),
        source=str(source), full_campaign_launched=False, covariance_policy_changed=False))
    write_json(Path(options['output'])/'current.json', dict(directory=str(root)))
    return root


def conditional_values(mix, layout, kind, data, rows, model):
    native = native_view(mix, layout, kind)
    phot = Photometry(data['flux'][rows], data['variance'][rows], model.transform.bands)
    anchors = model.reference_indices(phot); result = np.empty(len(rows))
    for lo in range(0, len(rows), 128):
        ix = np.arange(lo, min(lo+128, len(rows))); rr = rows[ix]
        noise = data['noise'][rr]; cov = np.zeros((len(rr), noise.shape[1], noise.shape[1]))
        cov[:, np.arange(noise.shape[1]), np.arange(noise.shape[1])] = noise
        for a in np.unique(anchors[ix]):
            take = anchors[ix] == a
            result[ix[take]] = conditional_log_prob(native, data['y'][rr[take]], cov[take],
                                                    data['observed'][rr[take]], int(a))
    if not np.isfinite(result).all():
        raise ValueError('nonfinite development density')
    return result


def spectra(mix, stats, floor):
    count, first, second = stats[:3]; active = count > 1e-10
    unconstrained = np.full((mix.n_components, mix.n_dim), np.nan)
    for j in np.flatnonzero(active):
        shift = first[j]/count[j]; scatter = second[j]/count[j]-np.outer(shift, shift)
        unconstrained[j] = np.linalg.eigvalsh(.5*(scatter+scatter.T))
    return dict(effective_membership=count.tolist(), eigenvalues=np.linalg.eigvalsh(mix.covs).tolist(),
        unconstrained_eigenvalues=[row.tolist() if use else None for row,use in zip(unconstrained,active)],
        fraction_below_floor=[float((row < floor).mean()) if use else None for row,use in zip(unconstrained,active)])


def fit_block(task, mode, root_string, through, deadline):
    """Checkpoint every full pass; restart never accepts a partial E step."""
    root = Path(root_string); options = read(root/'options.json'); source = Path(options['prepared_run'])
    cfg = read(source/'config.json'); layout = read(source/'layout.json'); kind = task['kind']
    dest = root/(task['name']+'_'+mode); latest = dest/'checkpoint.json'
    data = arrays(source, kind)
    with np.load(root/(task['name']+'_rows.npz')) as saved:
        train, dev = saved['train'], saved['density']
    row_hash = digest_rows(data['source_row'][train]); identity = file_hash(root/'identity.json')
    state = read(latest) if latest.exists() else dict(iteration=0, history=[], mixture=task['init'],
        fit_source_hash=row_hash, identity=identity, tolerance_hits=[], elapsed_seconds=0.)
    if state['fit_source_hash'] != row_hash or state['identity'] != identity:
        raise ValueError('checkpoint/row identity mismatch')
    if len(train) != task['n'] or not np.isin(data['role'][train], [0, 1]).all():
        raise ValueError('training membership mismatch')
    mix = GaussianMixture.from_dict(state['mixture'])
    if mode == 'eigenvalue_floor' and np.linalg.eigvalsh(mix.covs).min() < cfg['regularization']*(1-1e-8):
        raise ValueError('infeasible floor initialization')
    factory = ProjectedBatchFactory(source/kind, train, cfg['batch_size'])
    batches = lambda: factory(0, len(train)); op = {0:operator(layout, kind)}
    context = nullcontext(accumulate_projected)
    if kind == 'stars':
        context = ProjectedParallelAccumulator(factory, len(train), workers=options['stellar_workers'],
            task_rows=options['task_rows'], status_directory=dest)
    model = PSFMultiSurveyBaseline.load(options['nuisance_bundle']).model
    elapsed_before = state['elapsed_seconds']; start = time.monotonic()
    with context as accumulate:
        while state['iteration'] <= through:
            it = state['iteration']
            if (dest/f'evaluation_{it:03d}.json').exists() and it == through:
                break
            if time.time() >= deadline:
                return dict(task=task['name'], mode=mode, iteration=it, budget_exhausted=True)
            boundary = dest/'boundary_stats.npz'
            stats = None
            if boundary.exists():
                with np.load(boundary) as saved:
                    if int(saved['iteration']) == it:
                        stats = (saved['count'], saved['first'], saved['second'], float(saved['ll']), int(saved['n']))
            if stats is None:
                stats = accumulate(batches, mix, op)
            if stats[-1] != len(train):
                raise ValueError('partial training pass')
            ll = stats[3]/stats[4]
            if state['history']:
                previous = state['history'][-1]; gain = ll-previous
                if mode == 'eigenvalue_floor' and gain < -options['roundoff_tolerance']:
                    raise ValueError(f'constrained likelihood decreased: {task["name"]} {gain}')
                if 0 <= gain < options['tolerance']*max(1., abs(previous)):
                    if it not in state['tolerance_hits']:
                        state['tolerance_hits'].append(it)
            if it % options['block_iterations'] == 0:
                values = conditional_values(mix, layout, kind, data, dev, model)
                north = is_north(data['ra'][dev], data['dec'][dev], data['b'][dev])
                np.savez_compressed(dest/f'prediction_{it:03d}.npz', values=values, rows=dev)
                write_json(dest/f'model_{it:03d}.json', mix.to_dict())
                write_json(dest/f'evaluation_{it:03d}.json', dict(iteration=it, mean_loglike=ll,
                    development_mean=float(values.mean()), north=float(values[north].mean()),
                    south=float(values[~north].mean()), covariance=spectra(mix, stats, cfg['regularization'])))
            # At block boundaries retain the current model and its already
            # computed sufficient statistics, so the next block wastes no pass.
            state['elapsed_seconds'] = elapsed_before+time.monotonic()-start
            write_json(dest/'progress.json', dict(iteration=it, target=options['max_iter'],
                mean_loglike=ll, elapsed_seconds=state['elapsed_seconds'], tolerance_hits=state['tolerance_hits'],
                train_rows=len(train), complete=it == options['max_iter']))
            if it == through:
                with (dest/'boundary_stats.tmp').open('wb') as handle:
                    np.savez(handle, iteration=it, count=stats[0], first=stats[1], second=stats[2], ll=stats[3], n=stats[4])
                (dest/'boundary_stats.tmp').replace(boundary)
                # history excludes the current model until its M step is applied.
                write_json(latest, state)
                break
            state['history'].append(ll)
            mix = update_from_statistics(mix, stats, cfg['regularization'], mode)
            state.update(iteration=it+1, mixture=mix.to_dict())
            write_json(latest, state)
    return dict(task=task['name'], mode=mode, iteration=state['iteration'], budget_exhausted=False)


def star_blocks(task, root, through, deadline):
    return [fit_block(task, mode, root, through, deadline) for mode in read(Path(root)/'options.json')['updates']]


def raw_rank(prediction):
    total = logsumexp(np.stack([np.logaddexp(prediction['log_lambda_sameq'], prediction['log_lambda_fieldq']),
        prediction['log_lambda_bkg'], prediction['log_lambda_out']]), axis=0)
    return prediction['log_lambda_sameq']-total-np.log(prediction['dz_match_eff'])


def shift_summary(before, after):
    valid = np.isfinite(before) & np.isfinite(after); delta = after[valid]-before[valid]
    same_zero = np.isneginf(before) & np.isneginf(after)
    return dict(n=int(valid.sum()), stable_zero_weight=int(same_zero.sum()),
        nonfinite=int((~valid & ~same_zero).sum()),
        entered_zero_weight=int((np.isfinite(before)&np.isneginf(after)).sum()),
        left_zero_weight=int((np.isneginf(before)&np.isfinite(after)).sum()),
        mean=float(delta.mean()) if len(delta) else None,
        median_abs=float(np.median(abs(delta))) if len(delta) else None,
        p95_abs=float(np.quantile(abs(delta), .95)) if len(delta) else None,
        rank_correlation=float(spearmanr(before[valid], after[valid]).statistic) if len(delta)>2 and np.ptp(before[valid]) and np.ptp(after[valid]) else None)


def score_base(root, mode, iteration):
    options = read(root/'options.json'); source = Path(options['prepared_run']); layout = read(source/'layout.json')
    base = deepcopy(PSFMultiSurveyBaseline.load(options['nuisance_bundle']))
    # All 43 slices start from the same full-data canonical parent. Only the
    # selected two and background change; priors/catch-all/spatial weights are
    # fixed to the already completed unified pilot, never refitted on dev rows.
    for task in read(source/'tasks.json'):
        path = root/(task['name']+'_'+mode)/f'model_{iteration:03d}.json'
        latent = GaussianMixture.from_dict(read(path) if path.exists() else task['init'])
        mix = native_view(latent, layout, task['kind'])
        if task['kind'] == 'qso': base.model.qso.mixtures[task['index']] = mix
        else: base.model.background = mix
    base.model.meta['diagnostic_only'] = True
    return base


def evaluate_scores(root, iteration):
    """Full scorer, frozen nuisance terms; retain guarded and ungated diagnostics."""
    options = read(root/'options.json'); source = Path(options['prepared_run']); dest = root/'scores'
    dest.mkdir(exist_ok=True); cfg = read('configs/full_sample_release.json'); cfg['batch_size'] = 32
    tasks = read(root/'tasks.json'); panels = {}; baseline = score_base(root, 'additive', 0)
    for task in tasks:
        data = arrays(source, task['kind'])
        with np.load(root/(task['name']+'_rows.npz')) as saved:
            choices = {'score':saved['score']}
            if task['kind'] == 'stars': choices['random'] = saved['random']
        for label, rows in choices.items():
            panels[task['name']+'_'+label] = dict(flux=data['flux'][rows], variance=data['variance'][rows],
                bands=baseline.model.transform.bands, l=data['l'][rows], b=data['b'][rows],
                zprimary=data['zspec'][rows] if task['kind']=='qso' else
                    np.resize([t['z'] for t in tasks if t['kind']=='qso'], len(rows)))
    for hemi in ('north', 'south'):
        for mag in (18.5, 21.):
            panels[f'grid_{hemi}_{mag}'] = grid_data(baseline.model, hemi, mag, {'validation':cfg['grid']})
    summary = dict(iteration=iteration, panels={}, comparisons={}, ranking={}, diagnostic_only=True,
        nuisance_policy='Frozen unified-pilot priors/catch-all/spatial component weights. Other 41 QSO slices fixed at full-data canonical warm starts. Not a completed refit or probability calibration.')
    results = {}
    for mode in options['updates']:
        base = score_base(root, mode, iteration); results[mode] = {}
        for name, panel in panels.items():
            result = run_scores(base, panel, cfg, dest/f'{mode}_{iteration:03d}_{name}.npz')
            results[mode][name] = result; rank = raw_rank(result)
            summary['panels'].setdefault(name, {})[mode] = dict(n=len(rank),
                finite_raw=int(np.isfinite(rank).sum()), eligible=int(result['eligible'].sum()),
                high_qso=int((result['p_quasar']>cfg['high_qso_probability']).sum()),
                high_qso_after_guard=int(((result['p_quasar']>cfg['high_qso_probability']) & result['eligible']).sum()))
            # Fixed baseline density quantiles flag a low-density proxy. This
            # is not a calibrated support test or a measured false-positive rate.
            if name == 'stars_00_random':
                baseline_path = dest/f'additive_000_{name}.npz'
                reference = dict(np.load(baseline_path))
                cuts = [float(np.quantile(reference[key], options['low_density_proxy_quantile'])) for key in ('loglike_qso_zprimary','loglike_bkg')]
                low = (result['loglike_qso_zprimary']<cuts[0]) & (result['loglike_bkg']<cuts[1])
                summary['panels'][name][mode].update(low_density_proxy_cuts=cuts,
                    low_density_proxy_rows=int(low.sum()),
                    high_qso_low_density_proxy=int((low & (result['p_quasar']>cfg['high_qso_probability'])).sum()),
                    note='Unweighted random detectable PSF-background sample; low-density proxy is not certified OOD or known contamination.')
            if iteration:
                old = dict(np.load(dest/f'{mode}_{iteration-options["block_iterations"]:03d}_{name}.npz'))
                summary['panels'][name][mode]['since_previous'] = shift_summary(raw_rank(old), rank)
                summary['panels'][name][mode]['qso_half_crossings'] = int(((old['p_quasar']>cfg['high_qso_probability'])!=(result['p_quasar']>cfg['high_qso_probability'])).sum())
        for task in tasks:
            if task['kind'] != 'qso': continue
            q = raw_rank(results[mode][task['name']+'_score']); b = raw_rank(results[mode]['stars_00_score'])
            z = panels['stars_00_score']['zprimary']; b = b[np.isclose(z, task['z'])]
            if np.isnan(q).any() or np.isnan(b).any() or np.isposinf(q).any() or np.isposinf(b).any():
                raise ValueError('undefined raw scores in ranking panel')
            summary['ranking'].setdefault(task['name'], {})[mode] = float(mannwhitneyu(q,b).statistic/(len(q)*len(b)))
    for name in panels:
        summary['comparisons'][name] = shift_summary(raw_rank(results['additive'][name]), raw_rank(results['eigenvalue_floor'][name]))
    write_json(root/f'scores_{iteration:03d}.json', summary)
    return summary


def summarize_block(root, through, scores):
    options = read(root/'options.json'); report = dict(iteration=through, cases={}, ranking=scores['ranking'])
    for task in read(root/'tasks.json'):
        name = task['name']; entry = {}
        predictions = {}
        for mode in options['updates']:
            folder = root/(name+'_'+mode); values = np.load(folder/f'prediction_{through:03d}.npz')['values']
            old = np.load(folder/f'prediction_{through-options["block_iterations"]:03d}.npz')['values']
            entry[mode] = read(folder/f'evaluation_{through:03d}.json')
            entry[mode]['since_previous'] = shift_summary(old, values)
            entry[mode]['tolerance_hits'] = read(folder/'checkpoint.json')['tolerance_hits']
            checkpoints = [read(folder/f'evaluation_{i:03d}.json') for i in range(0,through+1,options['block_iterations'])]
            entry[mode]['best_development_iteration'] = max(checkpoints,key=lambda x:x['development_mean'])['iteration']
            predictions[mode] = values
        entry['floor_minus_additive'] = shift_summary(predictions['additive'], predictions['eigenvalue_floor'])
        report['cases'][name] = entry
    # This gate establishes a predictive plateau, not formal EM convergence.
    # At least two blocks are required; no attempt to clear every 1e-5 flag.
    plateau = through >= 2*options['block_iterations']
    for entry in report['cases'].values():
        for mode in options['updates']:
            change = entry[mode]['since_previous']
            plateau &= abs(change['mean']) <= options['density_plateau_mean']
    for name, entry in scores['panels'].items():
        if name.startswith('grid_'): continue
        for mode in options['updates']:
            change = entry[mode]['since_previous']
            plateau &= change['nonfinite'] == 0 and change['p95_abs'] is not None and change['p95_abs'] <= options['score_plateau_p95']
    report['predictive_plateau'] = bool(plateau)
    report['floor_within_density_tolerance'] = all(
        e['eigenvalue_floor'][hemi]-e['additive'][hemi] >= -options['maximum_density_loss']
        for e in report['cases'].values() for hemi in ('development_mean','north','south'))
    report['floor_within_auc_tolerance'] = all(v['eigenvalue_floor'] >= v['additive']-options['maximum_auc_loss'] for v in scores['ranking'].values())
    report['interpretation'] = 'Development comparison only; budget limit without a plateau is inconclusive on stopping. No automatic promotion.'
    write_json(root/f'comparison_{through:03d}.json', report)
    return report


def progress(root, start):
    options = read(root/'options.json'); tasks = read(root/'tasks.json')
    actual = total = 0; states = {}
    for t in tasks:
        for mode in options['updates']:
            path = root/(t['name']+'_'+mode)/'progress.json'
            state = read(path) if path.exists() else dict(iteration=0)
            actual += state['iteration']*t['n']*t['k']; total += options['max_iter']*t['n']*t['k']
            states[t['name']+'_'+mode] = state
    result = dict(fraction_of_maximum_density_work=actual/total, fits=states,
        elapsed_seconds=time.time()-start, scope='row x K x iteration; scoring overhead and extra boundary passes excluded')
    write_json(root/'progress.json', result)
    lines = ['# Full-row covariance trial', '',
        f"Density-work budget executed: {100*actual/total:.2f}% (row x K x iteration).",
        f"Elapsed: {(time.time()-start)/3600:.2f} hours. Ceiling: 60 updates per fit / four hours.",
        'This is the six-fit diagnostic, not the full 44-fit campaign.', '',
        '| Fit | Completed updates | Mean training log density |', '|---|---:|---:|']
    for name, state in states.items():
        lines.append(f"| {name} | {state['iteration']} | {state.get('mean_loglike','pending')} |")
    lines += ['', 'Matched score comparisons: '+', '.join(p.name for p in sorted(root.glob('comparison_*.json'))),
        'Active model unchanged. See execution.json for running/completed/failed state.']
    (root/'PROGRESS.md').write_text('\n'.join(lines)+'\n')
    return result


def run(root):
    options = read(root/'options.json'); tasks = read(root/'tasks.json'); start = time.time()
    deadline = start+options['wall_budget_seconds']; source = Path(options['prepared_run']); cfg = read(source/'config.json')
    with acquisition_lock(root/'run.lock'):
        if (root/'finished.json').exists():
            print('ALREADY FINISHED', root, flush=True); return
        write_json(root/'execution.json', dict(pid=os.getpid(), state='running', start_unix=start, deadline_unix=deadline))
        state = 'budget_exhausted'; comparisons = []
        try:
            # Establish complete-score baseline before changing any shapes.
            evaluate_scores(root, 0)
            with ProcessPoolExecutor(max_workers=options['qso_workers'], mp_context=multiprocessing.get_context('spawn')) as qpool, ThreadPoolExecutor(max_workers=1) as coordinator:
                for through in range(options['block_iterations'], options['max_iter']+1, options['block_iterations']):
                    pending = []
                    for task in tasks:
                        if task['kind'] == 'stars':
                            pending.append(coordinator.submit(star_blocks, task, str(root), through, deadline))
                        else:
                            pending.extend(qpool.submit(fit_block, task, mode, str(root), through, deadline) for mode in options['updates'])
                    exhausted = False
                    while pending:
                        done, remaining = wait(pending, timeout=15, return_when=FIRST_COMPLETED); pending = list(remaining)
                        for future in done:
                            value = future.result(); entries = value if isinstance(value,list) else [value]
                            exhausted |= any(v['budget_exhausted'] for v in entries)
                            print('BLOCK FIT', json.dumps(value), flush=True)
                        p = progress(root,start); print('PROGRESS', round(100*p['fraction_of_maximum_density_work'],2), flush=True)
                    if exhausted: break
                    scores = evaluate_scores(root, through); report = summarize_block(root, through, scores)
                    comparisons.append(through); print('COMPARISON', through, 'plateau', report['predictive_plateau'], flush=True)
                    if report['predictive_plateau']:
                        state = 'predictive_plateau'; break
                    if through == options['max_iter']: state = 'iteration_budget_reached'
                    if time.time() >= deadline: break
            if file_hash(Path(cfg['active_pointer'])) != read(root/'prepared.json')['active_pointer_hash']:
                raise ValueError('active pointer changed')
            write_json(root/'finished.json', dict(state=state, comparisons=comparisons, elapsed_seconds=time.time()-start,
                active_model_changed=False, full_campaign_launched=False))
            write_json(root/'execution.json', dict(pid=os.getpid(), state=state, elapsed_seconds=time.time()-start))
            print('FINISHED', root, state, flush=True)
        except BaseException as error:
            write_json(root/'execution.json', dict(pid=os.getpid(), state='failed', error=repr(error)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/unified_covariance_trial.json')
    parser.add_argument('--fit', action='store_true')
    args = parser.parse_args(); root = prepare(read(args.config)); print('PREPARED', root, flush=True)
    if args.fit: run(root)


if __name__ == '__main__':
    main()
