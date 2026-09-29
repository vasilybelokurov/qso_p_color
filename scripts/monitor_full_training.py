#!/usr/bin/env python
"""Read saved training checkpoints; write a live summary without altering fits."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(path.read_text())


def atomic(path, text):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def compute_progress(out, cfg, preflight):
    """Estimate EM work in row-component passes, including overlapping slices.

    Finished fits use their actual iteration counts. Remaining fits are budgeted
    to their configured limits; unknown final component counts use the largest
    candidate. This measures useful saved work, not elapsed wall time.
    """
    populations = [('background', preflight['roles']['stars'])] + [
        (f'qso_{s["index"]:02d}', {r: {'rows': n} for r, n in s['rows'].items()})
        for s in preflight['slices']]
    totals = {name: dict(done=0, remaining=0) for name in ('stars', 'qso')}
    for name, counts in populations:
        family = 'stars' if name == 'background' else 'qso'
        grid = cfg['background_k_candidates' if family == 'stars' else 'qso_k_candidates']
        fit_rows = sum(counts[r]['rows'] for r in cfg['fit_roles'])
        final_rows = sum(counts[r]['rows'] for r in cfg['final_shape_roles'])
        selection = {}
        for k in grid:
            saved = out / f'{name}.select_k{k}.json'
            checkpoint = out / f'{name}.select_k{k}.checkpoint.json'
            finished = saved.exists()
            record = read(saved) if finished else (read(checkpoint) if checkpoint.exists() else {})
            iterations = record.get('n_iter', record.get('iteration', 0))
            totals[family]['done'] += fit_rows * k * iterations
            totals[family]['remaining'] += fit_rows * k * (0 if finished else cfg['selection_max_iter']-iterations)
            if finished:
                selection[k] = record['score']
        saved = out / f'{name}.json'
        checkpoint = out / f'{name}.final.checkpoint.json'
        finished = saved.exists()
        record = read(saved) if finished else (read(checkpoint) if checkpoint.exists() else {})
        if record:
            k = record.get('selected_k', len(record['mixture']['weights']))
        else:
            k = max(selection, key=selection.get) if len(selection) == len(grid) else max(grid)
        iterations = record.get('n_iter', record.get('iteration', 0))
        totals[family]['done'] += final_rows * k * iterations
        totals[family]['remaining'] += final_rows * k * (0 if finished else cfg['max_iter']-iterations)
    totals['overall'] = {key: sum(totals[p][key] for p in ('stars', 'qso')) for key in ('done', 'remaining')}
    for value in totals.values():
        denominator = value['done'] + value['remaining']
        value['percent'] = 100 * value['done'] / denominator if denominator else 100.
    return dict(populations=totals, unit='row-component EM passes',
                assumptions='Remaining fits use iteration limits and largest K until K is selected; finished fits use actual iterations.',
                scope='Density EM only; excludes likelihood/scoring overhead, initialization, spatial weights, priors, catch-all and validation.',
                caveat='Work estimate, not a measured CPU-time fraction or wall-clock ETA; missing-band patterns affect pass costs.')


def snapshot(run, previous=None):
    launch = read(run / 'launch.json')
    out = Path(launch['output_directory'])
    identity = read(out / 'identity.json')
    cfg = identity['config']
    preflight = read(run / 'preflight.json')
    populations = ['background'] + [f'qso_{s["index"]:02d}' for s in preflight['slices']]
    stages = []
    for name in populations:
        grid = cfg['background_k_candidates' if name == 'background' else 'qso_k_candidates']
        stages.extend((f'{name}.select_k{k}', f'{name}.select_k{k}.json', cfg['selection_max_iter']) for k in grid)
        stages.append((name + '.final', name + '.json', cfg['max_iter']))
    completed = []
    current = None
    for stamp, filename, limit in stages:
        if (out / filename).exists():
            record = read(out / filename)
            completed.append(dict(stage=stamp, converged=record['converged'], iterations=record['n_iter'],
                                  selection_score=record.get('score'), selected_k=record.get('selected_k')))
        elif current is None:
            current = dict(stage=stamp, iteration_limit=limit, iteration=0)
            checkpoint = out / (stamp + '.checkpoint.json')
            if checkpoint.exists():
                record = read(checkpoint)
                current.update(iteration=record['iteration'], rows=record['rows'],
                               first_mean_loglike=record['history'][0],
                               mean_loglike=record['pre_update_mean_loglike'],
                               checkpoint_time=checkpoint.stat().st_mtime,
                               checkpoint_age_seconds=time.time() - checkpoint.stat().st_mtime)
    ps = subprocess.run(['ps', '-p', str(launch['pid']), '-o', 'etime=,time=,%cpu=,rss=,stat=,command='],
                        capture_output=True, text=True)
    fields = ps.stdout.strip().split(None, 5)
    alive = len(fields) == 6 and any(script in fields[5] for script in
            ('scripts/train_full_sample.py', 'scripts/train_full_sample_parallel.py',
             'scripts/train_stellar_parallel.py')) and 'Z' not in fields[4]
    process = dict(pid=launch['pid'], alive=alive)
    if alive:
        process.update(elapsed=fields[0], cpu_time=fields[1], cpu_percent=float(fields[2]),
                       rss_gib=int(fields[3]) / 1024**2)
    complete = (out / 'completion.json').exists()
    parallel_workers = []
    if (out / 'parallel_execution.json').exists():
        for path in sorted(out.glob('parallel_worker_*.json')):
            worker = read(path)
            if worker['state'] not in ('running', 'failed'):
                continue
            check = subprocess.run(['ps', '-p', str(worker['pid']), '-o', '%cpu=,rss=,stat='],
                                   capture_output=True, text=True).stdout.split()
            worker['alive'] = len(check) == 3 and 'Z' not in check[2]
            if worker['alive']:
                worker.update(cpu_percent=float(check[0]), rss_gib=int(check[1])/1024**2)
            pop = worker['population']
            planned = [(f'{pop}.select_k{k}', cfg['selection_max_iter'], f'{pop}.select_k{k}.json')
                       for k in cfg['qso_k_candidates']] + [(pop+'.final', cfg['max_iter'], pop+'.json')]
            for stamp, limit, completed_file in planned:
                if (out / completed_file).exists():
                    continue
                worker.update(stage=stamp, iteration=0, iteration_limit=limit)
                checkpoint = out / (stamp+'.checkpoint.json')
                if checkpoint.exists():
                    saved = read(checkpoint)
                    worker.update(iteration=saved['iteration'], rows=saved['rows'],
                                  mean_loglike=saved['pre_update_mean_loglike'],
                                  checkpoint_age_seconds=time.time()-checkpoint.stat().st_mtime)
                break
            parallel_workers.append(worker)
    stellar_workers = []
    for path in sorted(out.glob('stellar_worker_*.json')):
        worker = read(path)
        fields = subprocess.run(['ps', '-p', str(worker['pid']), '-o', '%cpu=,rss=,stat='],
                                capture_output=True, text=True).stdout.split()
        if len(fields) == 3 and 'Z' not in fields[2]:
            stellar_workers.append(dict(pid=worker['pid'], cpu_percent=float(fields[0]),
                                        rss_gib=int(fields[1])/1024**2))
    stellar_pass = read(out/'stellar_parallel_progress.json') if (out/'stellar_parallel_progress.json').exists() else None
    status = 'density fits complete' if complete else ('running' if alive else 'WARNING: worker exited before completion')
    if current and previous and previous.get('current', {}).get('stage') == current['stage']:
        old = previous['current']
        delta = current['iteration'] - old['iteration']
        if delta > 0 and 'checkpoint_time' in old:
            current['seconds_per_iteration'] = (current['checkpoint_time'] - old['checkpoint_time']) / delta
        elif 'seconds_per_iteration' in old:
            current['seconds_per_iteration'] = old['seconds_per_iteration']
    compute = compute_progress(out, cfg, preflight)
    result = dict(checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(), status=status,
                  process=process, current=current, completed_fits=completed, total_planned_fits=len(stages),
                  output_directory=str(out), density_complete=complete, release_ready=False,
                  parallel_workers=parallel_workers, compute_progress=compute,
                  stellar_workers=stellar_workers, stellar_pass=stellar_pass)
    lines = ['# Full-sample training progress', '', f"Updated: {result['checked_utc']}", '', f'**{status}**', '',
             f"**Estimated density-fitting compute completed: {compute['populations']['overall']['percent']:.2f}%.**",
             f"QSO: {compute['populations']['qso']['percent']:.2f}%; stellar: {compute['populations']['stars']['percent']:.2f}%.",
             'Weighted by rows × components × saved EM iterations. Remaining fits are budgeted at iteration limits; the estimate updates as complexity and convergence are measured.',
             'This covers density EM, not subsequent spatial/population fitting or validation, and is not a wall-clock percentage.', '',
             f"Completed fits: {len(completed)} / {len(stages)} (fits have different costs; this is not a time percentage)."]
    if alive:
        lines += [f"Worker {launch['pid']}: elapsed {process['elapsed']}, CPU {process['cpu_percent']:.1f}%, memory {process['rss_gib']:.2f} GiB."]
    if current and not complete:
        lines += ['', f"Current: **{current['stage']}**, iteration **{current['iteration']} / {current['iteration_limit']}** maximum."]
        if 'rows' in current:
            lines += [f"Rows per full pass: {current['rows']:,}.",
                      f"Mean training log-likelihood: {current['first_mean_loglike']:.6f} initially → {current['mean_loglike']:.6f} at latest checkpoint.",
                      f"Checkpoint age: {current['checkpoint_age_seconds']:.0f} seconds."]
        if 'seconds_per_iteration' in current:
            seconds = current['seconds_per_iteration']
            lines += [f"Recent full-pass time: {seconds:.1f} seconds; up to {(current['iteration_limit']-current['iteration'])*seconds/60:.1f} minutes of EM left in this fit at that rate.",
                      'Convergence can end a fit earlier; held-out scoring and other stages take additional time.']
    elif not complete:
        lines += ['', 'All shape fits saved; spatial weights and candidate assembly remain.']
    if stellar_workers:
        lines += ['', '## Parallel stellar E step', '',
                  '| PID | CPU | Memory GiB |', '|---:|---:|---:|']
        for worker in stellar_workers:
            lines += [f"| {worker['pid']} | {worker['cpu_percent']:.1f}% | {worker['rss_gib']:.2f} |"]
        if stellar_pass:
            lines += ['', f"Current E-step pass: {stellar_pass['rows_accumulated']:,} / {stellar_pass['expected_rows']:,} stars accumulated.",
                      'One global model update follows the complete pass.']
    if parallel_workers:
        lines += ['', '## Parallel QSO slices', '',
                  '| PID | Slice | Stage | Iteration / maximum | CPU | Memory GiB |',
                  '|---:|---|---|---:|---:|---:|']
        for w in parallel_workers:
            lines += [f"| {w['pid']} | {w['population']} | {w.get('stage', w['state'])} | "
                      f"{w.get('iteration', 0)} / {w.get('iteration_limit', '—')} | "
                      f"{w.get('cpu_percent', 'EXITED')} | {w.get('rss_gib', 0):.2f} |"]
    if completed:
        lines += ['', '| Completed fit | Iterations | Converged | Selection score |', '|---|---:|---|---:|']
        for item in completed:
            score = item['selection_score']
            lines += [f"| {item['stage']} | {item['iterations']} | {item['converged']} | {score if score is not None else '—'} |"]
    lines += ['', 'The active model is unchanged. Priors, catch-all fitting and reserved validation follow density training.',
              '', f"Training log: `{launch['log']}`", f'Fit/checkpoint directory: `{out}`', '']
    return result, '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=60)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    run = args.run or Path(read(Path('models/multisurvey_psf/work/full_training_runs/current.json'))['run'])
    with (run / 'monitor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid())); lock.flush()
        previous = read(run / 'progress.json') if (run / 'progress.json').exists() else None
        while True:
            result, summary = snapshot(run, previous)
            atomic(run / 'progress.json', json.dumps(result, indent=2) + '\n')
            atomic(run / 'PROGRESS.md', summary)
            with (run / 'progress_history.jsonl').open('a') as history:
                history.write(json.dumps(result) + '\n')
            print(summary, flush=True)
            if not args.watch or result['density_complete'] or not result['process']['alive']:
                break
            previous = result
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
