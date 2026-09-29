from dataclasses import dataclass

import numpy as np
import pytest

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.parallel_em import ParallelAccumulator
from qso_pcolor.streaming_xd import fit_xd_batches


@dataclass
class ArrayFactory:
    x: np.ndarray
    cov: np.ndarray
    obs: np.ndarray
    weight: np.ndarray
    batch_size: int = 7

    def __call__(self, lo, hi):
        for start in range(lo, hi, self.batch_size):
            stop = min(start+self.batch_size, hi)
            yield self.x[start:stop], self.cov[start:stop], self.obs[start:stop], self.weight[start:stop]


def sample():
    rng = np.random.default_rng(524)
    x = rng.normal(size=(83, 3)) + [1e5, 0., 2.]
    obs = rng.uniform(size=x.shape) > .3
    obs[:, 0] = True
    x[~obs] = np.nan
    a = rng.normal(size=(len(x), 3, 3))
    factory = ArrayFactory(x, a @ a.swapaxes(1, 2)*.02, obs, rng.uniform(.2, 2, len(x)))
    mix = GaussianMixture(np.array([.4, .6]), np.array([[1e5, -.5, 1.], [1e5+1, .5, 3.]]),
                          np.tile(np.eye(3), (2, 1, 1)), ('a', 'b', 'c'))
    return factory, mix


def test_four_workers_match_serial_and_continue_a_serial_checkpoint():
    factory, init = sample()
    source = lambda: factory(0, len(factory.x))
    options = dict(expected_rows=len(factory.x), max_iter=6, tol=0., regularization=.001)
    serial = fit_xd_batches(source, init=init, **options)
    saved, history = {}, []

    def interrupt(it, mix, ll, rows):
        saved['mix'] = mix
        history.append(ll)
        assert rows == len(factory.x)
        if it == 2:
            raise InterruptedError

    with pytest.raises(InterruptedError):
        fit_xd_batches(source, init=init, progress=interrupt, **options)
    with ParallelAccumulator(factory, len(factory.x), workers=4, task_rows=21) as accumulator:
        resumed = fit_xd_batches(source, init=saved['mix'], initial_history=tuple(history),
                                 accumulator=accumulator, **options)
    for key in ('weights', 'means', 'covs'):
        np.testing.assert_allclose(getattr(resumed.mixture, key), getattr(serial.mixture, key), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(resumed.history, serial.history, rtol=1e-9, atol=1e-9)
    assert resumed.mean_loglike == pytest.approx(serial.mean_loglike, abs=1e-9)
    assert resumed.n_iter == serial.n_iter


def test_parallel_worker_failure_cannot_save_a_partial_iteration():
    factory, init = sample()
    # Advertise one more row than the source actually contains.
    checkpoints = []
    with ParallelAccumulator(factory, len(factory.x)+1, workers=2, task_rows=21) as accumulator:
        with pytest.raises(ValueError, match='lost or duplicated rows'):
            fit_xd_batches(lambda: factory(0, len(factory.x)), init=init,
                expected_rows=len(factory.x)+1, max_iter=2, tol=0., regularization=.001,
                accumulator=accumulator, progress=lambda *args: checkpoints.append(args))
    assert checkpoints == []
