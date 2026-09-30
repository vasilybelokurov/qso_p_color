"""Audit and selectively continue fixed-K full-data fits without promotion.

Diagnostic densities are in natural logs per observed luptitude volume, or
per non-reference luptitude volume after conditioning. Diagnostic row samples
never limit training; every EM pass uses all original fit+selection rows.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np

from . import full_training as training
from .full_sample import TrainingRows, write_json
from .gaussmix import GaussianMixture
from .multisurvey import conditional_log_prob
from .parallel_em import ParallelAccumulator, TrainingBatchFactory
from .sky_acquisition import acquisition_lock
from .streaming_xd import accumulate_batches, fit_xd_batches


class DecliningLikelihood(RuntimeError):
    """Stop a continuation after a material likelihood reversal is measured."""


def audit_fit(record: dict, tol: float, window: int) -> dict:
    """Report likelihood gains (nats/object), weights and covariance spectra."""
    h = np.asarray(record['history'], float)
    mix = GaussianMixture.from_dict(record['mixture'])
    eig = np.linalg.eigvalsh(mix.covs)
    gains = np.diff(h[-window:])
    thresholds = tol * np.maximum(1., np.abs(h[-window:-1]))
    finite = all(np.isfinite(a).all() for a in (h, mix.weights, mix.means, mix.covs))
    resolution = np.finfo(float).eps * mix.means.shape[1] * eig[:, -1]
    numeric_ok = bool(finite and np.all(eig[:, 0] > resolution))
    decreases = int(np.count_nonzero(gains < -thresholds))
    action = ('retain_converged' if record['converged'] else
              'continue' if numeric_ok and not decreases else 'hold_for_diagnosis')
    return dict(k=record['selected_k'], rows=record['n_fit'], iteration=record['n_iter'],
        converged=record['converged'], numeric_ok=numeric_ok,
        last_gain=float(gains[-1]), threshold=float(thresholds[-1]),
        gain_threshold_ratio=float(gains[-1]/thresholds[-1]) if thresholds[-1] else None,
        window_gain=float(h[-1]-h[-window:][0]),
        first_ten_mean_gain=float(gains[:10].mean()), last_ten_mean_gain=float(gains[-10:].mean()),
        negative_steps=int((gains < 0).sum()), material_negative_steps=decreases,
        min_eigenvalue=float(eig.min()), max_condition=float((eig[:, -1]/eig[:, 0]).max()),
        min_weight=float(mix.weights.min()), min_weight_times_rows=float(mix.weights.min()*record['n_fit']),
        action=action)


def audit_parent(parent: Path, cfg: dict, options: dict) -> dict:
    """Audit final fits without changing saved checkpoints or active artifacts."""
    fits = {}
    for path in [parent/'background.json', *sorted(parent.glob('qso_[0-9][0-9].json'))]:
        fits[path.stem] = audit_fit(json.loads(path.read_text()), cfg['tol'], options['history_window'])
    return dict(parent=str(parent), fits=fits, not_held_out_validation=True)


def audit_markdown(report: dict) -> str:
    lines = ['# Saved-fit convergence audit', '',
        'Likelihood changes are nats per training object. Gain/threshold is signed.',
        'Negative steps count the last 49 differences (50 saved iterations).', '',
        '| Fit | K | Rows | Last gain | Gain/threshold | Negative steps | Minimum eigenvalue | Max condition | Action |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for name, r in report['fits'].items():
        if r['converged']:
            continue
        lines.append(f"| {name} | {r['k']} | {r['rows']} | {r['last_gain']:.6g} | "
            f"{r['gain_threshold_ratio']:.2f} | {r['negative_steps']} | {r['min_eigenvalue']:.4g} | "
            f"{r['max_condition']:.4g} | {r['action']} |")
    return '\n'.join(lines)+'\n'


def verify_parent_identity(old: dict, current: dict) -> None:
    """Reject changes to data, transforms, settings or the original EM engine."""
    if old != current:
        raise ValueError('parent identity differs from current inputs, configuration or implementation')


def checked_rows(data: TrainingRows, zr, cfg: dict, record: dict) -> np.ndarray:
    rows = data.select(tuple(cfg['final_shape_roles']), zr)
    if len(rows) != record['n_fit'] or hashlib.sha256(rows.tobytes()).hexdigest() != record['fit_rows_sha256']:
        raise ValueError('continuation row identities changed')
    return rows


def density_probe(data, rows, source, cfg, mix) -> dict:
    values = {key: [] for key in ('joint', 'conditional', 'anchor', 'magnitude', 'mask')}
    order = np.array([mix.labels.index(b) for b in source.reference_priority])
    for x, cov, obs, _ in training.make_source(data, rows, source, cfg)():
        anchor = order[np.argmax(obs[:, order], axis=1)]
        conditional = np.empty(len(x))
        for a in np.unique(anchor):
            use = anchor == a
            conditional[use] = conditional_log_prob(mix, x[use], cov[use], obs[use], int(a))
        values['joint'].append(mix.log_prob(x, cov, observed=obs))
        values['conditional'].append(conditional)
        values['anchor'].append(anchor)
        values['magnitude'].append(x[np.arange(len(x)), anchor])
        values['mask'].append(np.sum(obs.astype('uint64') << np.arange(obs.shape[1], dtype='uint64'), axis=1))
    result = {k: np.concatenate(v) for k, v in values.items()}
    if not all(np.isfinite(result[k]).all() for k in ('joint', 'conditional', 'magnitude')):
        raise ValueError('non-finite diagnostic density')
    return result


def compare_probes(before: dict, after: dict, options: dict) -> dict:
    """Compare exactly the same measurements; strata never compare raw densities across masks."""
    for key in ('anchor', 'magnitude', 'mask'):
        if not np.array_equal(before[key], after[key]):
            raise ValueError('diagnostic sample or measurements changed')
    delta = after['conditional']-before['conditional']
    def summary(use):
        v = delta[use]
        return dict(rows=int(len(v)), mean_change=float(v.mean()),
            median_abs_change=float(np.median(np.abs(v))),
            p95_abs_change=float(np.quantile(np.abs(v), .95)),
            max_abs_change=float(np.abs(v).max()))
    strata = {}
    for mask in np.unique(before['mask']):
        use = before['mask'] == mask
        # Mask fixes observed dimensions and priority anchor, making density
        # quantiles within this stratum meaningful; tails are sample-relative.
        anchor = int(before['anchor'][use][0])
        edges = np.unique(np.quantile(before['magnitude'][use], np.linspace(0, 1, options['magnitude_quantiles']+1)))
        bins = np.searchsorted(edges[1:-1], before['magnitude'], side='right')
        grouped = {}
        tail = np.zeros(len(delta), bool)
        for b in np.unique(bins[use]):
            members = use & (bins == b)
            grouped[str(int(b))] = summary(members)
            threshold = np.quantile(before['conditional'][members], options['tail_fraction'])
            tail |= members & (before['conditional'] <= threshold)
        strata[str(int(mask))] = dict(anchor=anchor, all=summary(use),
            magnitude_edges=edges.tolist(), magnitude_bins=grouped, low_density_tail=summary(tail))
    return dict(all=summary(np.ones(len(delta), bool)), by_band_mask=strata,
        joint_max_abs_change=float(np.max(np.abs(after['joint']-before['joint']))),
        interpretation='Fixed training-row diagnostic, not held-out validation or support certification. '
        'Natural-log conditional-density changes; sparse strata retain explicit row counts.')


def diagnose_step(source, mix, cfg: dict, expected_rows: int) -> dict:
    """Compare one unchanged update with a zero-ridge counterfactual; discard both."""
    stats = accumulate_batches(source, mix)
    if stats[-1] != expected_rows:
        raise ValueError('diagnostic E-step row loss')
    before = stats[3]/stats[4]
    results = {}
    for label, ridge in [('configured', cfg['regularization']), ('zero_ridge_counterfactual', 0.)]:
        try:
            fit = fit_xd_batches(source, init=mix, expected_rows=expected_rows,
                max_iter=1, tol=0., regularization=ridge, accumulator=lambda *args: stats)
            results[label] = dict(mean_log_density=fit.mean_loglike, change=fit.mean_loglike-before)
        except np.linalg.LinAlgError as error:
            results[label] = dict(error=str(error))
    return dict(before_mean_log_density=before, rows=expected_rows, updates=results,
        production_parameters_changed=False,
        interpretation='Same full-data E-step statistics; only the added covariance diagonal differs. '
        'The counterfactual is diagnostic and is never saved as a fit.')


def _run_fit(name, zr, root, source, cfg, options, out, action):
    parent = Path(options['parent'])
    saved = json.loads((parent/(name+'.json')).read_text())
    data = TrainingRows(root/('stars' if name == 'background' else 'qso'), source.transform.bands)
    rows = checked_rows(data, zr, cfg, saved)
    batches = training.make_source(data, rows, source, cfg)
    status = out/(name+'.status.json')
    started = time.monotonic()
    base = dict(name=name, pid=os.getpid(), parent_iteration=saved['n_iter'],
        target_iteration=saved['n_iter']+options['additional_iterations'], rows=len(rows), k=saved['selected_k'])
    write_json(status, dict(base, state='running', iteration=saved['n_iter']))
    try:
        mix = GaussianMixture.from_dict(saved['mixture'])
        if action == 'hold_for_diagnosis':
            diagnostic = out/(name+'.decline_diagnosis.json')
            if not diagnostic.exists():
                write_json(diagnostic, diagnose_step(batches, mix, cfg, len(rows)))
            write_json(status, dict(base, state='held_after_diagnosis', iteration=saved['n_iter']))
            return
        result_path = out/(name+'.json')
        checkpoint_path = out/(name+'.final.checkpoint.json')
        if not result_path.exists():
            history = list(saved['history'])
            if checkpoint_path.exists():
                previous = json.loads(checkpoint_path.read_text())
                if previous['rows'] != len(rows) or previous['history'][:len(saved['history'])] != saved['history']:
                    raise ValueError('checkpoint row accounting or history lineage changed')
                history = list(previous['history']); mix = GaussianMixture.from_dict(previous['mixture'])
            def checkpoint(it, fitted, ll, seen):
                history.append(ll)
                write_json(checkpoint_path, dict(iteration=it, rows=seen, history=history,
                    mixture=fitted.to_dict(), pre_update_mean_loglike=ll))
                elapsed = time.monotonic()-started
                write_json(status, dict(base, state='running', iteration=it,
                    elapsed_seconds=elapsed, last_mean_log_density=ll))
                print(name, 'iteration', it, 'mean log density', ll, flush=True)
                if len(history) > 1 and ll-history[-2] < -cfg['tol']*max(1., abs(history[-2])):
                    raise DecliningLikelihood('likelihood declined beyond the existing convergence tolerance')
            factory = TrainingBatchFactory(data.root, rows, source.transform, cfg['fit_batch_size'])
            context = (ParallelAccumulator(factory, len(rows), workers=options['workers'],
                task_rows=options['task_rows'], status_directory=out) if name == 'background' else nullcontext(None))
            stop_reason = 'convergence_or_iteration_budget'
            try:
                with context as accumulator:
                    fitted = fit_xd_batches(batches, init=mix, expected_rows=len(rows),
                        max_iter=base['target_iteration'], tol=cfg['tol'], regularization=cfg['regularization'],
                        initial_history=tuple(history), progress=checkpoint, accumulator=accumulator)
            except DecliningLikelihood:
                # The checkpoint includes the completed M step. Evaluate that
                # returned mixture without doing another update or resetting history.
                previous = json.loads(checkpoint_path.read_text())
                fitted = fit_xd_batches(batches, init=GaussianMixture.from_dict(previous['mixture']),
                    expected_rows=len(rows), max_iter=previous['iteration'], tol=cfg['tol'],
                    regularization=cfg['regularization'], initial_history=tuple(previous['history']))
                stop_reason = 'held_after_likelihood_decline'
            result = dict(saved, mixture=fitted.mixture.to_dict(), history=fitted.history,
                n_iter=fitted.n_iter, converged=fitted.converged, mean_training_log_density=fitted.mean_loglike,
                continuation_parent=str(parent), parent_sha256=training.file_hash(parent/(name+'.json')),
                continuation_stop_reason=stop_reason)
            write_json(result_path, result)
        else:
            result = json.loads(result_path.read_text())
        write_json(status, dict(base, state='comparing_predictions', iteration=result['n_iter']))
        probe_path = out/(name+'.probe.npz')
        if probe_path.exists():
            with np.load(probe_path) as f:
                chosen = f['rows']; before = {k:f[k] for k in f.files if k != 'rows'}
            if not np.isin(chosen, rows).all():
                raise ValueError('diagnostic rows are not shape-training rows')
        else:
            seed = options['diagnostic_seed']+int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
            rng = np.random.default_rng(seed)
            chosen = np.sort(rng.choice(rows, min(len(rows), options['diagnostic_rows']), replace=False))
            before = density_probe(data, chosen, source, cfg, GaussianMixture.from_dict(saved['mixture']))
            np.savez(probe_path, rows=chosen, **before)
        after = density_probe(data, chosen, source, cfg, GaussianMixture.from_dict(result['mixture']))
        np.savez(out/(name+'.probe_after.npz'), rows=chosen, **after)
        write_json(out/(name+'.prediction_change.json'), compare_probes(before, after, options))
        write_json(status, dict(base, state='completed', iteration=result['n_iter'], converged=result['converged'],
            elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        write_json(status, dict(base, state='failed', error=repr(error)))
        raise


def progress_report(out: Path, report: dict, options: dict) -> dict:
    populations = {}
    for population in ('qso', 'stellar'):
        done = total = 0.
        for name, r in report['fits'].items():
            if r['action'] != 'continue' or (name == 'background') != (population == 'stellar'):
                continue
            path = out/(name+'.status.json')
            s = json.loads(path.read_text()) if path.exists() else {}
            iterations = max(0, s.get('iteration', r['iteration'])-r['iteration'])
            budget = iterations if s.get('state') == 'completed' else options['additional_iterations']
            done += r['rows']*r['k']*iterations; total += r['rows']*r['k']*budget
        populations[population] = dict(done=done, budget=total, percent=100*done/total if total else 100.)
    total = sum(p['budget'] for p in populations.values())
    return dict(updated=datetime.now(timezone.utc).isoformat(), populations=populations,
        overall_percent=100*sum(p['done'] for p in populations.values())/total if total else 100.,
        scope='Additional EM for improving fits only; excludes decline diagnostics, prediction comparisons and later pipeline stages.')


def run_review(options: dict, *, fit: bool = False) -> Path:
    cfg = json.loads(Path(options['training_config']).read_text())
    parent, out = Path(options['parent']), Path(options['output'])
    if out.resolve() == parent.resolve():
        raise ValueError('must preserve parent directory')
    for key in ('additional_iterations', 'history_window', 'workers', 'task_rows', 'diagnostic_rows', 'magnitude_quantiles'):
        if not isinstance(options[key], int) or isinstance(options[key], bool) or options[key] < 1:
            raise ValueError('positive integer execution settings required')
    if options['history_window'] < 2 or not 0 < options['tail_fraction'] < 1:
        raise ValueError('invalid diagnostic window or tail fraction')
    if json.loads((parent/'identity.json').read_text())['config'] != cfg:
        raise ValueError('audit must use the original training configuration')
    out.mkdir(parents=True, exist_ok=True)
    report = audit_parent(parent, cfg, options)
    if not fit:
        with acquisition_lock(out/'worker.lock'):
            write_json(out/'audit.json', report); (out/'AUDIT.md').write_text(audit_markdown(report))
        return out
    with acquisition_lock(parent/'worker.lock'), acquisition_lock(out/'worker.lock'):
        root, _, source, identity = training.load_inputs(cfg)
        old = json.loads((parent/'identity.json').read_text())
        verify_parent_identity(old, identity)
        parent_files = {p.name:training.file_hash(p) for p in sorted(parent.glob('*.json'))}
        engine = {name:training.file_hash(Path('src/qso_pcolor')/name) for name in
            ('convergence_review.py', 'parallel_em.py', 'streaming_xd.py')}
        provenance = dict(parent=str(parent.resolve()), parent_identity=old,
            parent_files=parent_files, options=options, engine=engine)
        path = out/'lineage.json'
        if path.exists() and json.loads(path.read_text()) != provenance:
            raise ValueError('continuation provenance changed')
        write_json(path, provenance)
        write_json(out/'audit.json', report); (out/'AUDIT.md').write_text(audit_markdown(report))
        for name, r in report['fits'].items():
            if r['action'] != 'continue':
                dest = out/(name+'.json')
                if dest.exists() and training.file_hash(dest) != parent_files[name+'.json']:
                    raise ValueError('retained parent result changed')
                shutil.copy2(parent/(name+'.json'), dest)
        execution = dict(pid=os.getpid(), state='running', started=datetime.now(timezone.utc).isoformat(),
            workers=options['workers'], phase='qso', active_pointer_changed=False)
        write_json(out/'execution.json', execution)
        ranges = training.slice_ranges(source, cfg['z_step'])
        if set(report['fits']) != {'background', *(f'qso_{i:02d}' for i in range(len(ranges)))}:
            raise ValueError('parent does not contain every expected fit')
        try:
            jobs = [(n, ranges[int(n.split('_')[1])], r) for n,r in report['fits'].items()
                    if n.startswith('qso_') and r['action'] != 'retain_converged']
            # Longest full-data jobs first; diagnosis and continuation share one pool.
            jobs.sort(key=lambda v: v[2]['rows']*v[2]['k'], reverse=True)
            with ProcessPoolExecutor(max_workers=options['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(_run_fit, n, zr, root, source, cfg, options, out, r['action']) for n,zr,r in jobs]
                for future in as_completed(futures):
                    future.result()
                    write_json(out/'progress.json', progress_report(out, report, options))
            execution['phase'] = 'stellar'; write_json(out/'execution.json', execution)
            action = report['fits']['background']['action']
            if action != 'retain_converged':
                _run_fit('background', None, root, source, cfg, options, out, action)
            after = audit_parent(out, cfg, options)
            write_json(out/'after_audit.json', after)
            if any(training.file_hash(parent/name) != digest for name,digest in parent_files.items()):
                raise ValueError('parent results changed')
            if training.file_hash(Path(cfg['source_pointer'])) != identity['source_pointer_sha256']:
                raise ValueError('active model pointer changed')
            execution.update(state='completed', phase='diagnostic continuation complete',
                release_ready=False, held_fits=[n for n,r in report['fits'].items() if r['action']=='hold_for_diagnosis'])
        except BaseException as error:
            execution.update(state='failed', error=repr(error)); raise
        finally:
            write_json(out/'execution.json', execution)
            write_json(out/'progress.json', progress_report(out, report, options))
    return out
