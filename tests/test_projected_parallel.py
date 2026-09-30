from dataclasses import dataclass

import numpy as np
import pytest

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.projected_parallel import ProjectedParallelAccumulator, ProjectedBatchFactory
from qso_pcolor.projected_xd import fit_projected


@dataclass
class ArrayFactory:
    y: np.ndarray
    cov: np.ndarray
    obs: np.ndarray
    system: np.ndarray

    def __call__(self, lo, hi):
        for start in range(lo, hi, 7):
            stop = min(start+7, hi)
            yield self.y[start:stop], self.cov[start:stop], self.obs[start:stop], self.system[start:stop]


def sample():
    rng = np.random.default_rng(419)
    y = rng.normal(size=(87, 4))
    obs = rng.random(y.shape) > .25; obs[:, 0] = True; y[~obs] = np.nan
    a = rng.normal(size=(len(y), 4, 4))
    factory = ArrayFactory(y, a@a.swapaxes(1, 2)*.01, obs, np.arange(len(y)) % 2)
    h = np.array([[1., 0, 0], [0, 1, 0], [.1, 0, 1], [0, .2, 1.]])
    ops = {0:(h, np.zeros(4), np.eye(4)*.02),
           1:(h*1.02, np.ones(4)*.1, np.eye(4)*.03)}
    init = GaussianMixture(np.array([.4, .6]), np.array([[0., .4, -.2], [1., -.2, .2]]),
        np.tile(np.eye(3), (2, 1, 1)))
    return factory, init, ops


def test_four_projected_workers_match_serial_and_resume_interrupted_fit():
    factory, init, ops = sample()
    source = lambda: factory(0, len(factory.y))
    opts = dict(expected_rows=len(factory.y), max_iter=6, tol=0., regularization=.001, operators=ops)
    serial = fit_projected(source, init=init, **opts)
    saved, history = {}, []

    def interrupt(it, mix, ll, rows):
        saved['mix'] = mix; history.append(ll)
        if it == 2:
            raise InterruptedError

    with pytest.raises(InterruptedError):
        fit_projected(source, init=init, progress=interrupt, **opts)
    with ProjectedParallelAccumulator(factory, len(factory.y), workers=4, task_rows=23) as accumulate:
        resumed = fit_projected(source, init=saved['mix'], initial_history=tuple(history),
            accumulator=accumulate, **opts)
    for key in ('weights', 'means', 'covs'):
        np.testing.assert_allclose(getattr(resumed.mixture, key), getattr(serial.mixture, key), atol=1e-10)
    np.testing.assert_allclose(resumed.history, serial.history, atol=1e-10)
    assert resumed.n_iter == serial.n_iter
    assert resumed.mean_loglike == pytest.approx(serial.mean_loglike, abs=1e-10)


def test_projected_worker_failure_cannot_checkpoint_partial_iteration():
    factory, init, ops = sample(); saved = []
    with ProjectedParallelAccumulator(factory, len(factory.y)+1, workers=2, task_rows=23) as accumulate:
        with pytest.raises(ValueError, match='lost or duplicated rows'):
            fit_projected(lambda:factory(0,len(factory.y)), init=init, operators=ops,
                expected_rows=len(factory.y)+1, max_iter=2, tol=0., regularization=.001,
                accumulator=accumulate, progress=lambda *args:saved.append(args))
    assert saved == []


def test_mmap_factory_preserves_row_order_masks_and_variances(tmp_path):
    factory, _, _ = sample()
    for name, value in [('y',factory.y), ('noise',np.diagonal(factory.cov,axis1=1,axis2=2)), ('observed',factory.obs)]:
        np.save(tmp_path/(name+'.npy'),value)
    rows=np.array([13, 7, 84, 2, 1]);replay=ProjectedBatchFactory(tmp_path,rows,2)
    batches=list(replay(0,len(rows)))
    np.testing.assert_allclose(np.concatenate([b[0] for b in batches]),factory.y[rows])
    np.testing.assert_array_equal(np.concatenate([b[2] for b in batches]),factory.obs[rows])
    np.testing.assert_allclose(np.diagonal(np.concatenate([b[1] for b in batches]),axis1=1,axis2=2),np.diagonal(factory.cov[rows],axis1=1,axis2=2))


def test_small_likelihood_decline_is_not_reported_as_convergence():
    factory, init, ops = sample(); ll=iter([0.,-1e-8])
    from qso_pcolor.projected_xd import accumulate_projected
    def declining(source,mix,operators):
        count,first,second,_,n=accumulate_projected(source,mix,operators)
        return count,first,second,next(ll)*n,n
    result=fit_projected(lambda:factory(0,len(factory.y)),init=init,operators=ops,
        expected_rows=len(factory.y),max_iter=2,tol=1e-4,regularization=.001,accumulator=declining)
    assert not result.converged
