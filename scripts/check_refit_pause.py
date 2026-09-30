#!/usr/bin/env python
"""Checks at the first pause of the unified refit; never fits or edits the run.

1. Background: held-out density at its first stopping check versus the warm
   start, and complete-score ranking for every QSO slice with only the
   background swapped (QSO slices held at their warm starts, pilot nuisance
   terms fixed), plus high-score counts on a random background panel.
2. Stopping rule: every recorded decision is on schedule, finite and matches
   the saved best checkpoint.
3. QSO slices: first-block held-out gain; declining slices must keep their
   warm start.

Usage::

    python scripts/check_refit_pause.py --run models/multisurvey_psf/work/unified_full/20260930/<tag>
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from qso_pcolor import PSFMultiSurveyBaseline
from qso_pcolor.full_sample import write_json
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.unified import native_view
from run_unified_pilot import arrays
from trial_unified_covariance import raw_rank, select_development
from validate_full_sample_release import run_scores

CRITERIA = dict(maximum_median_auc_change=0.002, maximum_slice_auc_loss=0.005,
                certain_qso_probability=0.99, minimum_background_density_gain=0.0)


def read(path):
    return json.loads(Path(path).read_text())


def background_models(root, tasks):
    """Warm start and the checkpoint at the background's first stopping check."""
    star = next(t for t in tasks if t['kind'] == 'stars')
    trace = read(root/'progress'/(star['name']+'_stopping.json'))
    iterations = [e['iteration'] for e in trace['evaluations']]
    if len(iterations) < 2:
        raise ValueError('background has not reached its first stopping check')
    # Use the latest evaluated checkpoint; a paused fit saves exactly that model.
    checkpoint = read(root/'progress'/(star['name']+'_checkpoint.json'))
    if checkpoint['iteration'] != iterations[-1]:
        raise ValueError('background checkpoint is not the latest evaluated model')
    return star, GaussianMixture.from_dict(star['init']), GaussianMixture.from_dict(checkpoint['mixture']), trace


def score_slice(args):
    root, task, which, out_dir = args
    root = Path(root); cfg = read(root/'config.json'); layout = read(root/'layout.json'); rule = cfg['predictive_stopping']
    tasks = read(root/'tasks.json'); star, start, first, _ = background_models(root, tasks)
    base = deepcopy(PSFMultiSurveyBaseline.load(rule['reference_bundle']))
    for t in tasks:                                       # every QSO slice at its warm start
        if t['kind'] == 'qso':
            base.model.qso.mixtures[t['index']] = native_view(GaussianMixture.from_dict(t['init']), layout, 'qso')
    base.model.background = native_view(start if which == 'start' else first, layout, 'stars')
    base.model.meta['diagnostic_only'] = True
    score_cfg = read('configs/full_sample_release.json'); score_cfg['batch_size'] = 32
    qso, stars = arrays(root, 'qso'), arrays(root, 'stars')
    options = dict(seed=rule['seed'], development_rows_per_hemisphere=rule['rows_per_hemisphere'],
                   score_rows_per_hemisphere=64, random_background_score_rows=1024)
    out = {}
    if task['kind'] == 'qso':
        rows = select_development(qso, task, cfg, options)['score']
        srows = select_development(stars, star, cfg, options)['score']
        qpanel = dict(flux=qso['flux'][rows], variance=qso['variance'][rows], bands=base.model.transform.bands,
                      l=qso['l'][rows], b=qso['b'][rows], zprimary=qso['zspec'][rows])
        spanel = dict(flux=stars['flux'][srows], variance=stars['variance'][srows], bands=base.model.transform.bands,
                      l=stars['l'][srows], b=stars['b'][srows], zprimary=np.full(len(srows), task['z']))
        out['qso'] = run_scores(base, qpanel, score_cfg, Path(out_dir)/f"{which}_{task['name']}_qso.npz")
        out['stars'] = run_scores(base, spanel, score_cfg, Path(out_dir)/f"{which}_{task['name']}_stars.npz")
    else:
        rows = select_development(stars, star, cfg, options)['random']
        zs = [t['z'] for t in tasks if t['kind'] == 'qso']
        panel = dict(flux=stars['flux'][rows], variance=stars['variance'][rows], bands=base.model.transform.bands,
                     l=stars['l'][rows], b=stars['b'][rows], zprimary=np.resize(zs, len(rows)))
        out['random'] = run_scores(base, panel, score_cfg, Path(out_dir)/f'{which}_random.npz')
    return task['name'], which, {k: {f: v[f] for f in ('p_quasar', 'log_lambda_sameq', 'log_lambda_fieldq',
                                                        'log_lambda_bkg', 'log_lambda_out', 'dz_match_eff')} for k, v in out.items()}


def auc(q, b):
    """Ranking AUC; undefined ranks (zero target-window weight) sit at the bottom."""
    q, b = (np.where(np.isnan(x), -np.inf, x) for x in (q, b))
    return float(mannwhitneyu(q, b).statistic/(len(q)*len(b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--tasks', nargs='*', help='QSO slices to score (default: all)')
    parser.add_argument('--stopping-only', action='store_true', help='check stopping records only; safe while fitting')
    args = parser.parse_args(); root = Path(args.run); tasks = read(root/'tasks.json'); cfg = read(root/'config.json')
    rule = cfg['predictive_stopping']; out_dir = root/'pause_check'; out_dir.mkdir(exist_ok=True)
    report = dict(criteria=CRITERIA, stopping={}, slices={}, background={})

    # 2 and 3: stopping traces.
    schedule_ok = True
    for t in tasks:
        path = root/'progress'/(t['name']+'_stopping.json')
        if not path.exists():
            report['stopping'][t['name']] = dict(evaluated=False); continue
        trace = read(path); ev = trace['evaluations']; it = [e['iteration'] for e in ev]; val = [e['mean'] for e in ev]
        on_schedule = all(i % rule['block_iterations'] == 0 or i == it[-1] for i in it) and it == sorted(set(it))
        best = read(root/'progress'/(t['name']+'_best.json'))
        consistent = best['iteration'] == it[int(np.argmax(val))] and np.isclose(best['mean'], max(val))
        entry = dict(evaluated=True, iterations=it, values=val, stop_reason=trace['stop_reason'],
                     finite=bool(np.isfinite(val).all()), on_schedule=bool(on_schedule), best_consistent=bool(consistent),
                     best_iteration=best['iteration'], first_block_gain=val[1]-val[0] if len(val) > 1 else None)
        if entry['first_block_gain'] is not None and entry['first_block_gain'] < 0:
            entry['declining_first_block'] = True
            entry['keeps_warm_start'] = best['iteration'] == 0
        schedule_ok &= entry['finite'] and entry['on_schedule'] and entry['best_consistent']
        report['stopping'][t['name']] = entry

    if args.stopping_only:
        declining = {n: dict(gain=e['first_block_gain'], best_iteration=e['best_iteration'], stop_reason=e['stop_reason'])
                     for n, e in report['stopping'].items() if e.get('declining_first_block')}
        summary = dict(stopping_records_valid=bool(schedule_ok),
                       evaluated_first_block=sum(e.get('first_block_gain') is not None for e in report['stopping'].values()),
                       stopped={n: e['stop_reason'] for n, e in report['stopping'].items() if e.get('stop_reason')},
                       declining_first_block=declining)
        write_json(root/'stopping_check.json', dict(report['stopping'], summary=summary))
        print(json.dumps(summary, indent=1)); return

    # 1: background swap.
    star, _, _, trace = background_models(root, tasks)
    report['background']['density'] = dict(start=trace['evaluations'][0]['mean'], latest=trace['evaluations'][-1]['mean'],
        iteration=trace['evaluations'][-1]['iteration'], trajectory=trace['evaluations'],
        gain=trace['evaluations'][-1]['mean']-trace['evaluations'][0]['mean'])
    chosen = [t for t in tasks if t['kind'] == 'qso' and (not args.tasks or t['name'] in args.tasks)] + [star]
    jobs = [(str(root), t, which, str(out_dir)) for t in chosen for which in ('start', 'first')]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = {}
        for name, which, out in pool.map(score_slice, jobs):
            results.setdefault(name, {})[which] = out
    high = read('configs/full_sample_release.json')['high_qso_probability']
    changes = []
    for t in chosen:
        r = results[t['name']]
        if t['kind'] != 'qso':
            report['background']['random_high_qso'] = {w: int((r[w]['random']['p_quasar'] > high).sum()) for w in r}
            continue
        entry = {}
        for w in ('start', 'first'):
            entry[w] = auc(raw_rank(r[w]['qso']), raw_rank(r[w]['stars']))
        entry['change'] = entry['first']-entry['start']
        # Losses explained by background objects both models already call certain QSOs.
        certain = (r['start']['stars']['p_quasar'] > CRITERIA['certain_qso_probability']) & \
                  (r['first']['stars']['p_quasar'] > CRITERIA['certain_qso_probability'])
        keep = ~certain
        entry['certain_qso_background_objects'] = int(certain.sum())
        entry['change_excluding_certain'] = (auc(raw_rank(r['first']['qso']), raw_rank(r['first']['stars'])[keep]) -
                                             auc(raw_rank(r['start']['qso']), raw_rank(r['start']['stars'])[keep])) if keep.any() else None
        entry['high_qso_background'] = {w: int((r[w]['stars']['p_quasar'] > high).sum()) for w in ('start', 'first')}
        entry['undefined_rank_rows'] = {w: int(np.isnan(raw_rank(r[w]['qso'])).sum() + np.isnan(raw_rank(r[w]['stars'])).sum())
                                        for w in ('start', 'first')}
        report['slices'][t['name']] = entry; changes.append(entry['change'])
    median = float(np.median(changes)) if changes else None
    worst = [n for n, e in report['slices'].items() if e['change'] < -CRITERIA['maximum_slice_auc_loss'] and
             (e['change_excluding_certain'] is None or e['change_excluding_certain'] < -CRITERIA['maximum_slice_auc_loss'])]
    declining = [n for n, e in report['stopping'].items() if e.get('declining_first_block')]
    report['checks'] = dict(
        background_density_not_worse=report['background']['density']['gain'] >= CRITERIA['minimum_background_density_gain'],
        median_auc_change=median, median_auc_within=median is not None and abs(median) <= CRITERIA['maximum_median_auc_change'],
        unexplained_slice_losses=worst, stopping_records_valid=bool(schedule_ok),
        declining_first_block=declining,
        declining_keep_warm_start=all(report['stopping'][n]['keeps_warm_start'] for n in declining
                                      if report['stopping'][n]['stop_reason'] is not None))
    report['passed'] = bool(report['checks']['background_density_not_worse'] and report['checks']['median_auc_within']
                            and not worst and schedule_ok)
    report['interpretation'] = ('Development evidence on role-3 stopping/score rows with fixed pilot nuisance terms and '
                                'QSO slices at warm starts; isolates the background update. Not a release validation.')
    write_json(root/'pause_check.json', report)
    print(json.dumps(dict(passed=report['passed'], checks=report['checks'], background=report['background']), indent=1))


if __name__ == '__main__':
    main()
