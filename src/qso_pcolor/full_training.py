"""Uncapped shape fitting and spatial weights from immutable prepared inputs.

This builds a candidate density model. Priors and catch-all calibration are
separate population fits; an old bundle must never be relabelled as retrained.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .full_sample import TrainingRows, ROLES, file_hash, write_json
from .gaussmix import GaussianMixture
from .joint_spatial import fit_joint_spatial_log_prob
from .multisurvey import MultiSurveyModel, conditional_log_prob
from .qso_model import SlicedColourRedshiftModel
from .spatial import component_log_prob
from .streaming_xd import fit_xd_batches, initialise_batches
from .xd import _estep_chunk, _mask_groups
from .sky_acquisition import acquisition_lock


def validate_roles(cfg: dict) -> None:
    expected = dict(fit_roles=['fit'], selection_roles=['select'],
                    final_shape_roles=['fit', 'select'], catchall_roles=['calib'], test_roles=['test'])
    if any(cfg[k] != v for k, v in expected.items()):
        raise ValueError('training must preserve the frozen fit/select/calib/test roles')
    for name in ('background_k_candidates', 'qso_k_candidates'):
        if not cfg[name] or any(not isinstance(k, int) or k < 1 for k in cfg[name]):
            raise ValueError('component grids must contain positive integers')
    if cfg['fit_batch_size'] < 1 or cfg['memory_budget_gib'] <= 0:
        raise ValueError('invalid memory configuration')


def load_inputs(cfg: dict):
    validate_roles(cfg)
    pointer = json.loads((Path(cfg['input_root'])/'current.json').read_text())
    root = Path(pointer['directory']); manifest = root/'manifest.json'
    if file_hash(manifest) != pointer['manifest_sha256']:
        raise ValueError('input manifest changed')
    saved = json.loads(manifest.read_text())
    if not saved['complete']:
        raise ValueError('input assembly is incomplete')
    for key in ('qso_root', 'stellar_root', 'acquisition_complete_surveys', 'reuse_only_surveys'):
        if saved['identity']['config'][key] != cfg[key]:
            raise ValueError(f'trainer differs from input assembly: {key}')
    for path, digest in saved['files'].items():
        if file_hash(root/path) != digest:
            raise ValueError(f'training input changed: {path}')
    source_pointer = Path(cfg['source_pointer'])
    selected = json.loads(source_pointer.read_text())
    source_path = source_pointer.parent/selected['bundle']/'model.json'
    source = MultiSurveyModel.load(source_path)
    report = json.loads((root/'report.json').read_text())
    bands = tuple(report['qso']['bands'])
    if bands != tuple(report['stars']['bands']) or bands != source.transform.bands:
        raise ValueError('training/source-model band systems differ')
    identity = dict(config=cfg, inputs_sha256=pointer['manifest_sha256'],
        source_sha256=file_hash(source_path), source_pointer_sha256=file_hash(source_pointer),
        implementation={name: file_hash(Path('src/qso_pcolor')/name) for name in
                        ('full_training.py', 'full_sample.py', 'streaming_xd.py', 'xd.py', 'joint_spatial.py')})
    return root, report, source, identity


def slice_ranges(source: MultiSurveyModel, step: float) -> list[tuple[float, float]]:
    if not np.isfinite(step) or step <= 0 or not np.allclose(np.diff(source.qso.z_centres), step):
        raise ValueError('declared redshift step differs from the source model grid')
    return [(float(z-step), float(z+step)) for z in source.qso.z_centres]


def make_source(data: TrainingRows, rows: np.ndarray, source: MultiSurveyModel, cfg: dict):
    return lambda: data.batches(rows, source.transform, cfg['fit_batch_size'])


def score_batches(mix: GaussianMixture, batches, priority: tuple[str, ...]) -> tuple[float, int]:
    """Mean conditional log density per observed non-anchor magnitude space."""
    order = np.array([mix.labels.index(b) for b in priority])
    total = weight_sum = 0.; n = 0
    for x, cov, obs, w in batches():
        anchor = order[np.argmax(obs[:, order], axis=1)]
        for a in np.unique(anchor):
            use = anchor == a
            total += float(w[use] @ conditional_log_prob(mix, x[use], cov[use], obs[use], int(a)))
        n += len(x); weight_sum += w.sum()
    if weight_sum <= 0:
        raise ValueError('empty selection sample')
    return total/weight_sum, n


def scan_selection(data: TrainingRows, rows: np.ndarray, source: MultiSurveyModel, cfg: dict) -> dict:
    counts = np.zeros(len(source.transform.bands), dtype='i8'); seen = 0
    for _, _, obs, _ in make_source(data, rows, source, cfg)():
        counts += obs.sum(axis=0); seen += len(obs)
    if seen != len(rows):
        raise ValueError('preflight omitted rows')
    return dict(rows=seen, band_counts=counts.tolist(), row_indices_sha256=hashlib.sha256(rows.tobytes()).hexdigest())


def preflight(cfg: dict, *, benchmark: bool = False) -> dict:
    """Verify every selected row without fitting; optionally time E-step batches."""
    root, report, source, identity = load_inputs(cfg)
    bands = source.transform.bands
    q, b = TrainingRows(root/'qso', bands), TrainingRows(root/'stars', bands)
    result = dict(identity=identity, model_fitted=False, downloads_started=False,
                  caps=None, roles={}, slices=[], memory={}, benchmark=[],
                  density_fit_ready=True, release_ready=False,
                  population_completion='Rebuild count priors and fit catch-all on calib only after new shapes; no old population artifact is silently reused.')
    for name, data in [('qso', q), ('stars', b)]:
        result['roles'][name] = {}
        for role in ROLES:
            rows = data.select((role,))
            record = scan_selection(data, rows, source, cfg)
            if record['rows'] != report[name]['roles'][role]['eligible']:
                raise ValueError('preflight differs from assembly accounting')
            if role == 'fit' and min(record['band_counts']) == 0:
                raise ValueError(f'{name}: an entire band lacks fitting observations')
            result['roles'][name][role] = record
        print('preflight row accounting:', name, {r:v['rows'] for r,v in result['roles'][name].items()}, flush=True)
    ranges = slice_ranges(source, cfg['z_step'])
    covered = np.zeros(len(q.arrays['role']), bool)
    for j, zr in enumerate(ranges):
        counts = {r:len(q.select((r,), zr)) for r in ROLES}
        covered[q.select(tuple(cfg['final_shape_roles']), zr)] = True
        if min(counts['fit'], counts['select']) < max(cfg['qso_k_candidates']):
            raise ValueError(f'slice {j} lacks independent selection rows for declared K grid')
        result['slices'].append(dict(index=j, z_range=zr, rows=counts))
    final = q.select(tuple(cfg['final_shape_roles']))
    if not covered[final].all():
        raise ValueError('eligible quasars fall outside every declared slice')
    result['unique_qso_final_rows'] = len(final)
    result['slice_row_sum'] = sum(x['rows']['fit']+x['rows']['select'] for x in result['slices'])
    result['slice_overlap_note'] = 'Overlapping redshift slices intentionally reuse rows; unique population accounting is separate.'
    d = len(bands); k = max(cfg['background_k_candidates']+cfg['qso_k_candidates'])
    workspace = 8 * cfg['fit_batch_size'] * k * d*d * 8
    # Conservative simultaneous workspace plus memory-mapped input sizes. File
    # pages are reclaimable; report them separately from private batch arrays.
    input_bytes = sum((root/path).stat().st_size for path in json.loads((root/'manifest.json').read_text())['files'] if path.endswith('.npy'))
    result['memory'] = dict(batch_rows=cfg['fit_batch_size'], dimensions=d, max_components=k,
        estimated_batch_workspace_bytes=workspace, memory_mapped_input_bytes=input_bytes,
        budget_gib=cfg['memory_budget_gib'], full_covariance_allocated=False,
        omitted_full_covariance_bytes=8*d*d*(report['qso']['rows']+report['stars']['rows']),
        note='Workspace bound estimates eight simultaneous batch-sized K*D*D arrays; benchmark RSS is measured separately. Fits run sequentially.')
    if workspace > cfg['memory_budget_gib'] * 1024**3:
        raise ValueError('batch workspace exceeds declared memory budget')
    if benchmark:
        for name, data, mix in [('qso', q, source.qso.mixtures[len(source.qso.mixtures)//2]), ('stars', b, source.background)]:
            rows = data.select(('fit',))
            # Three separated batches; these timing probes do not update any
            # mixture and are never advertised as full training or validation.
            for components in sorted({mix.n_components, k}):
                pick = np.arange(components) % mix.n_components
                timing_mix = GaussianMixture(np.full(components, 1/components), mix.means[pick], mix.covs[pick], mix.labels)
                for start in np.linspace(0, max(0, len(rows)-cfg['fit_batch_size']), 3, dtype=int):
                    chosen = rows[start:start+cfg['fit_batch_size']]
                    t = time.monotonic()
                    x, cov, obs, w = next(make_source(data, chosen, source, cfg)())
                    for rr, dims in _mask_groups(obs):
                        _estep_chunk(rr, dims, timing_mix, x, cov, w)
                    result['benchmark'].append(dict(population=name, rows=len(chosen), k=components,
                        elapsed_seconds=time.monotonic()-t, max_observed=int(obs.sum(axis=1).max())))
        estimates = {}
        for name in ['qso', 'stars']:
            samples = [r['elapsed_seconds']/r['rows'] for r in result['benchmark'] if r['population']==name and r['k']==k]
            grid = cfg['qso_k_candidates'] if name=='qso' else cfg['background_k_candidates']
            fitting = sum(s['rows']['fit'] for s in result['slices']) if name=='qso' else result['roles'][name]['fit']['rows']
            final = result['slice_row_sum'] if name=='qso' else sum(result['roles'][name][r]['rows'] for r in cfg['final_shape_roles'])
            work = fitting * cfg['selection_max_iter'] * sum(grid)/k + final * cfg['max_iter']
            estimates[name] = dict(em_seconds_range_at_iteration_limits=[min(samples)*work,max(samples)*work],
                assumed_final_k=k, note='Three timing batches; approximate K scaling, all fits at iteration limits. Excludes extra likelihood passes, spatial/population completion, restart costs and server-independent local load variability.')
        result['runtime_estimates'] = estimates
    return result


def fit_population(name: str, data: TrainingRows, zr, source: MultiSurveyModel,
                   cfg: dict, root: Path, initial: dict[int, GaussianMixture]) -> dict:
    path = root/(name+'.json')
    if path.exists():
        return json.loads(path.read_text())
    tr = data.select(tuple(cfg['fit_roles']), zr); va = data.select(tuple(cfg['selection_roles']), zr)
    final = data.select(tuple(cfg['final_shape_roles']), zr)
    if np.intersect1d(tr, va).size or len(final) != len(tr)+len(va):
        raise ValueError('role overlap or loss')
    scores = {}; records = {}; candidates = {}
    grid = cfg['background_k_candidates'] if name == 'background' else cfg['qso_k_candidates']
    def fit(rows, init, iterations, stamp):
        checkpoint_path = root/(stamp+'.checkpoint.json')
        history = []
        if checkpoint_path.exists():
            previous = json.loads(checkpoint_path.read_text())
            if previous['rows'] != len(rows):
                raise ValueError('restart row accounting changed')
            history = previous['history']
            init = GaussianMixture.from_dict(previous['mixture'])
        def checkpoint(it, mix, ll, seen):
            history.append(ll)
            write_json(checkpoint_path, dict(iteration=it, rows=seen, history=history,
                mixture=mix.to_dict(), pre_update_mean_loglike=ll, status='unfinished fit; not a validated model'))
        return fit_xd_batches(make_source(data, rows, source, cfg), init=init,
            expected_rows=len(rows), max_iter=iterations, tol=cfg['tol'],
            regularization=cfg['regularization'], progress=checkpoint, initial_history=tuple(history))
    for k in grid:
        saved = root/(name+f'.select_k{k}.json')
        if saved.exists():
            record = json.loads(saved.read_text())
            candidates[k] = GaussianMixture.from_dict(record['mixture'])
        else:
            r = fit(tr, initial[k], cfg['selection_max_iter'], name+f'.select_k{k}')
            score, scored = score_batches(r.mixture, make_source(data, va, source, cfg), source.reference_priority)
            if scored != len(va): raise ValueError('selection rows lost')
            record = dict(k=k, score=score, fit_rows=len(tr), selection_rows=scored,
                          converged=r.converged, n_iter=r.n_iter, mixture=r.mixture.to_dict())
            write_json(saved, record); candidates[k] = r.mixture
        scores[k] = record['score']; records[k] = record
        print(name, 'K', k, 'selection score', scores[k], flush=True)
    k = max(scores, key=scores.get)
    r = fit(final, candidates[k], cfg['max_iter'], name+'.final')
    record = dict(selected_k=k, selection_records=records, n_fit=len(final),
        role_counts=dict(fit=len(tr), select=len(va), calib=0, test=0),
        fit_rows_sha256=hashlib.sha256(final.tobytes()).hexdigest(),
        converged=r.converged, n_iter=r.n_iter, history=r.history,
        mean_training_log_density=r.mean_loglike, mixture=r.mixture.to_dict(),
        capacity_grid_boundary=k == max(grid), caps=None)
    write_json(path, record)
    return record


def train(cfg: dict) -> Path:
    """Explicitly requested fitting only; never promote or write active pointers."""
    root, _, source, identity = load_inputs(cfg)
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    out = Path(cfg['output_root'])/version; out.mkdir(parents=True, exist_ok=True)
    with acquisition_lock(out/'worker.lock'):
        write_json(out/'identity.json', identity)
        return _train_candidate(cfg, root, source, identity, version, out)


def _train_candidate(cfg, root, source, identity, version, out):
    q, b = TrainingRows(root/'qso', source.transform.bands), TrainingRows(root/'stars', source.transform.bands)
    init = {}
    for name, data, grid in [('qso', q, cfg['qso_k_candidates']), ('background', b, cfg['background_k_candidates'])]:
        init[name] = {}
        base_path = out/(name+'.initial_base.json')
        if base_path.exists():
            base = GaussianMixture.from_dict(json.loads(base_path.read_text()))
        else:
            base = initialise_batches(make_source(data, data.select(('fit',)), source, cfg), max(grid),
                labels=source.transform.bands, seed=cfg['seed'])
            write_json(base_path, base.to_dict())
        for k in grid:
            path = out/(name+f'.initial_k{k}.json')
            if path.exists():
                mix = GaussianMixture.from_dict(json.loads(path.read_text()))
            else:
                mix = GaussianMixture(np.full(k, 1/k), base.means[:k], base.covs[:k], base.labels)
                write_json(path, mix.to_dict())
            init[name][k] = mix
    bg = fit_population('background', b, None, source, cfg, out, init['background'])
    qs = [fit_population(f'qso_{i:02d}', q, zr, source, cfg, out, init['qso']) for i,zr in enumerate(slice_ranges(source, cfg['z_step']))]
    mix = GaussianMixture.from_dict(bg['mixture']); rows = b.select(tuple(cfg['final_shape_roles']))
    lp = np.lib.format.open_memmap(out/'spatial_component_log_prob.npy', mode='w+', dtype='f8', shape=(len(rows), mix.n_components))
    offset = 0; bounds = np.column_stack((np.full(len(source.transform.bands), np.inf), np.full(len(source.transform.bands), -np.inf)))
    for x, cov, obs, _ in make_source(b, rows, source, cfg)():
        lp[offset:offset+len(x)] = component_log_prob(mix, x, cov, obs); offset += len(x)
        for j in range(x.shape[1]):
            if obs[:, j].any():
                bounds[j, 0] = min(bounds[j, 0], x[obs[:, j], j].min())
                bounds[j, 1] = max(bounds[j, 1], x[obs[:, j], j].max())
    if offset != len(rows): raise ValueError('spatial rows lost')
    lp.flush(); sky = cfg['spatial']
    spatial = fit_joint_spatial_log_prob(mix, lp, b.arrays['l'][rows], b.arrays['b'][rows], np.ones(len(rows)),
        nside=sky['nside'], nside_parent=sky['nside_parent'], n0=sky['colour_n0'],
        max_iter=sky['max_iter'], tol=sky['tol'], meta=dict(roles=cfg['final_shape_roles'], input_manifest=identity['inputs_sha256']))
    qso = SlicedColourRedshiftModel(source.qso.z_centres, [GaussianMixture.from_dict(r['mixture']) for r in qs],
        np.array([r['n_fit'] for r in qs]), 'multisurvey_psf_full_'+version, source.qso.labels,
        meta=dict(roles=cfg['final_shape_roles'], frozen_roles_sha256=identity['inputs_sha256'], per_slice=[{k:v for k,v in r.items() if k != 'mixture'} for r in qs]))
    model = MultiSurveyModel(qso, mix, source.transform, source.reference_priority, bounds,
        meta=dict(run_id=version, settings=dict(z_step=cfg['z_step'], config=cfg), population='psf',
                  training_identity=identity, caps=None, candidate_only=True,
                  validation_status='Unvalidated density candidate; priors and catch-all not yet refitted.', background=bg),
        spatial_background=spatial)
    model.save(out/'model.json')
    write_json(out/'completion.json', dict(density_candidate=str(out/'model.json'), active_pointer_changed=False,
        all_shapes_converged=all(r['converged'] for r in [bg,*qs]),
        capacity_grid_boundary=any(r['capacity_grid_boundary'] for r in [bg,*qs]),
        release_ready=False, remaining=['population priors', 'calib-only catch-all', 'reserved validation']))
    if file_hash(Path(cfg['source_pointer'])) != identity['source_pointer_sha256']:
        raise ValueError('active model pointer changed during training')
    return out
