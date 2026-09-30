"""Bounded parallel projected-XD E steps with deterministic full-row reduction."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from .full_sample import write_json
from .projected_xd import accumulate_projected

_factory = None


@dataclass
class ProjectedBatchFactory:
    """Replay selected memory-mapped observations (mag; variance in mag squared)."""
    directory: Path
    rows: np.ndarray
    batch_size: int

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key != '_data'}

    def __call__(self, lo: int, hi: int):
        if not hasattr(self, '_data'):
            self._data = {name: np.load(self.directory / (name+'.npy'), mmap_mode='r')
                          for name in ('y', 'noise', 'observed')}
        data = self._data
        for start in range(lo, hi, self.batch_size):
            rows = self.rows[start:min(start+self.batch_size, hi)]
            v = data['noise'][rows]
            cov = np.zeros((len(rows), v.shape[1], v.shape[1]))
            cov[:, np.arange(v.shape[1]), np.arange(v.shape[1])] = v
            yield data['y'][rows], cov, data['observed'][rows], np.zeros(len(rows), int)


def _initialize(factory, status_directory):
    global _factory
    _factory = factory
    if status_directory is not None:
        write_json(status_directory / f'stellar_worker_{os.getpid()}.json',
                   dict(pid=os.getpid(), role='projected stellar E-step', state='ready'))


def _task(lo, hi, mix, operators):
    result = accumulate_projected(lambda: _factory(lo, hi), mix, operators)
    if result[-1] != hi-lo:
        raise ValueError('parallel chunk lost or duplicated rows')
    return result


class ProjectedParallelAccumulator:
    """Reduce disjoint row chunks in fixed order, then permit one global M step.

    At most twice the worker count is queued. Every worker sees the same latent
    mixture and native observation operator; workers never fit separate models.
    """

    def __init__(self, factory, expected_rows: int, *, workers: int, task_rows: int,
                 status_directory: Path | None = None):
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1
               for n in (expected_rows, workers, task_rows)):
            raise ValueError('row and worker counts must be positive integers')
        self.factory, self.expected_rows = factory, expected_rows
        self.workers, self.task_rows = workers, task_rows
        self.status_directory, self.executor = status_directory, None
        self.pass_number = 0

    def __enter__(self):
        self.executor = ProcessPoolExecutor(max_workers=self.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=_initialize,
            initargs=(self.factory, self.status_directory))
        return self

    def __exit__(self, *args):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.executor = None

    def __call__(self, source, mix, operators):
        if self.executor is None:
            raise RuntimeError('parallel accumulator must be used as a context manager')
        self.pass_number += 1
        start = last_report = time.monotonic()
        k, d = mix.means.shape
        count, first, second = np.zeros(k), np.zeros((k, d)), np.zeros((k, d, d))
        ll, seen = 0., 0
        chunks = iter((lo, min(lo+self.task_rows, self.expected_rows))
                      for lo in range(0, self.expected_rows, self.task_rows))
        pending = deque()

        def submit():
            chunk = next(chunks, None)
            if chunk is not None:
                pending.append(self.executor.submit(_task, *chunk, mix, operators))

        for _ in range(2*self.workers):
            submit()
        try:
            while pending:
                c, f, s, part_ll, n = pending.popleft().result()
                count += c; first += f; second += s; ll += part_ll; seen += n
                submit()
                now = time.monotonic()
                if self.status_directory is not None and (now-last_report >= 5 or seen == self.expected_rows):
                    write_json(self.status_directory/'stellar_parallel_progress.json', dict(
                        pass_number=self.pass_number, rows_accumulated=seen,
                        expected_rows=self.expected_rows, elapsed_seconds=now-start,
                        workers=self.workers, task_rows=self.task_rows))
                    last_report = now
        except BaseException:
            for future in pending:
                future.cancel()
            raise
        if seen != self.expected_rows:
            raise ValueError('parallel pass lost or duplicated rows')
        return count, first, second, ll, seen
