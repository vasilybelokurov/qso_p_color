"""Schedule independent QSO slices without changing the fitted model or identity.

The existing density trainer owns initialization and final assembly. In the
coordinator process only, its population dispatcher waits for queued QSO work;
the spawned workers use the original, unchanged fitting function.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from unittest.mock import patch

from . import full_training as training
from .full_sample import TrainingRows, write_json
from .gaussmix import GaussianMixture
from .sky_acquisition import acquisition_lock

_worker_context = None


def _initialize_worker(root, source, cfg, out):
    global _worker_context
    data = TrainingRows(root / 'qso', source.transform.bands)
    initial = {k: GaussianMixture.from_dict(json.loads((out / f'qso.initial_k{k}.json').read_text()))
               for k in cfg['qso_k_candidates']}
    _worker_context = data, source, cfg, out, initial


def _fit_slice(index, redshift_range):
    data, source, cfg, out, initial = _worker_context
    name = f'qso_{index:02d}'
    status = out / f'parallel_worker_{os.getpid()}.json'
    write_json(status, dict(pid=os.getpid(), population=name, redshift_range=redshift_range, state='running'))
    try:
        with acquisition_lock(out / (name + '.worker.lock')):
            result = training.fit_population(name, data, redshift_range, source, cfg, out, initial)
    except BaseException as error:
        write_json(status, dict(pid=os.getpid(), population=name, state='failed', error=repr(error)))
        raise
    write_json(status, dict(pid=os.getpid(), population=name, state='completed'))
    return result


def train_parallel(cfg: dict, *, workers: int = 4) -> Path:
    """Resume the exact saved fit with four QSO processes and one stellar fit.

    No worker changes data, random seeds, component grids or EM batch order.
    The run lock excludes the earlier serial launcher and other coordinators.
    Initial mixtures must already exist; this is an explicit continuation path.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    root, _, source, identity = training.load_inputs(cfg)
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    out = Path(cfg['output_root']) / version
    if not (out / 'identity.json').exists() or json.loads((out / 'identity.json').read_text()) != identity:
        raise ValueError('parallel continuation requires an identical prepared training run')
    for population, grid in [('qso', cfg['qso_k_candidates']), ('background', cfg['background_k_candidates'])]:
        for k in grid:
            if not (out / f'{population}.initial_k{k}.json').exists():
                raise ValueError('initialization must finish before parallel continuation')
    with acquisition_lock(out / 'worker.lock'):
        ranges = training.slice_ranges(source, cfg['z_step'])
        data = TrainingRows(root / 'qso', source.transform.bands)
        # Longest slices first reduce the final tail; every slice is scheduled once.
        order = sorted(range(len(ranges)),
                       key=lambda i: len(data.select(tuple(cfg['final_shape_roles']), ranges[i])), reverse=True)
        original = training.fit_population
        record = dict(coordinator_pid=os.getpid(), qso_workers=workers, slice_order=order,
                      worker_threads=1, scientific_identity_unchanged=True, state='running',
                      scheduler_sha256=training.file_hash(Path(__file__)))
        write_json(out / 'parallel_execution.json', record)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_initialize_worker, initargs=(root, source, cfg, out)) as pool:
            futures = {f'qso_{i:02d}': pool.submit(_fit_slice, i, ranges[i]) for i in order}

            def dispatch(name, *args, **kwargs):
                if name in futures:
                    return futures[name].result()
                return original(name, *args, **kwargs)

            # Scoped to this single-threaded coordinator; spawned processes import
            # the original function. Final assembly remains the existing code path.
            try:
                with patch.object(training, 'fit_population', dispatch):
                    result = training._train_candidate(cfg, root, source, identity, version, out)
            except BaseException as error:
                record.update(state='failed', error=repr(error))
                write_json(out / 'parallel_execution.json', record)
                for future in futures.values():
                    future.cancel()
                raise
        record['state'] = 'completed'
        write_json(out / 'parallel_execution.json', record)
        return result
