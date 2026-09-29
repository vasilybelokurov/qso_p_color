"""Bounded parallel E steps, followed by the shared serial EM parameter update."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from .full_sample import TrainingRows, write_json
from .streaming_xd import accumulate_batches

_factory = None


@dataclass
class TrainingBatchFactory:
    """Picklable read-only row selection; fluxes remain in their native units."""
    directory: Path
    rows: np.ndarray
    transform: object
    batch_size: int

    def __getstate__(self):
        # Never pickle the cached mmap arrays after a serial benchmark/pass.
        return {key: value for key, value in self.__dict__.items() if key != '_data'}

    def __call__(self, lo, hi):
        if not hasattr(self, '_data'):
            self._data = TrainingRows(self.directory, self.transform.bands)
        return self._data.batches(self.rows[lo:hi], self.transform, self.batch_size)


def _initialize(factory, status_directory):
    global _factory
    _factory = factory
    if status_directory is not None:
        write_json(status_directory / f'stellar_worker_{os.getpid()}.json',
                   dict(pid=os.getpid(), state='ready', role='stellar E-step batches'))


def _task(lo, hi, mix):
    result = accumulate_batches(lambda: _factory(lo, hi), mix)
    if result[-1] != hi-lo:
        raise ValueError('parallel chunk lost or duplicated rows')
    return result


class ParallelAccumulator:
    """Sum disjoint row chunks in a fixed order, independent of completion order.

    At most twice the worker count is in flight. The original batch size and
    mask-group E-step kernel are retained. Reduction regrouping can change
    floating-point rounding, but not the full-data objective or M step.
    """

    def __init__(self, factory, expected_rows: int, *, workers: int, task_rows: int,
                 status_directory: Path | None = None):
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1
               for n in (expected_rows, workers, task_rows)):
            raise ValueError('row and worker counts must be positive integers')
        self.factory, self.expected_rows = factory, expected_rows
        self.workers, self.task_rows = workers, task_rows
        self.status_directory = status_directory
        self.executor = None
        self.pass_number = 0

    def __enter__(self):
        self.executor = ProcessPoolExecutor(max_workers=self.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=_initialize,
            initargs=(self.factory, self.status_directory))
        return self

    def __exit__(self, *args):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.executor = None

    def __call__(self, source, mix):
        if self.executor is None:
            raise RuntimeError('parallel accumulator must be used as a context manager')
        self.pass_number += 1
        start = time.monotonic()
        last_report = start
        k, d = mix.means.shape
        aq, adm, av = np.zeros(k), np.zeros((k, d)), np.zeros((k, d, d))
        ll = weight = 0.
        seen = 0
        chunks = iter((lo, min(lo+self.task_rows, self.expected_rows))
                      for lo in range(0, self.expected_rows, self.task_rows))
        pending = deque()

        def submit():
            chunk = next(chunks, None)
            if chunk is not None:
                pending.append(self.executor.submit(_task, *chunk, mix))

        for _ in range(2*self.workers):
            submit()
        try:
            while pending:
                q, dm, v, part_ll, part_weight, count = pending.popleft().result()
                aq += q; adm += dm; av += v
                ll += part_ll; weight += part_weight; seen += count
                submit()
                now = time.monotonic()
                if self.status_directory is not None and (now-last_report >= 5 or seen == self.expected_rows):
                    write_json(self.status_directory / 'stellar_parallel_progress.json',
                               dict(pass_number=self.pass_number, rows_accumulated=seen,
                                    expected_rows=self.expected_rows, elapsed_seconds=now-start,
                                    workers=self.workers, task_rows=self.task_rows))
                    last_report = now
        except BaseException:
            for future in pending:
                future.cancel()
            raise
        if seen != self.expected_rows:
            raise ValueError('parallel pass lost or duplicated rows')
        return aq, adm, av, ll, weight, seen
